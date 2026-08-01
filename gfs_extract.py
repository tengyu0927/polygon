#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gfs_extract.py — 在存放 GFS 归档的服务器上，把 8 个机场站点的序列抽成一份小 CSV

单文件、不依赖本项目任何东西，拷过去直接跑。

    # 第一步: 盘点归档（只看文件名和大小，不读 GRIB，任何环境都能跑）
    python3 gfs_extract.py scan /path/to/gfs

    # 第二步: 看单个文件里有哪些变量（需要 GRIB 读取库，见下）
    python3 gfs_extract.py inspect /path/to/gfs.t12z.pgrb2.0p25.f018

    # 第三步: 抽取（几十分钟到几小时，可随时 Ctrl-C，再跑会自动续上）
    python3 gfs_extract.py extract /path/to/gfs --out gfs_stations.csv

    # 只抽一部分先试（强烈建议先这样试通）
    python3 gfs_extract.py extract /path/to/gfs --out test.csv --limit 20

抽出来的 CSV 每行一个 (站, 起报时刻, 时效, 变量) 的值，
8 个站 × 2 轮/天 × 时效档数 × 变量数 —— 两年的量级在几十 MB，
scp 回来很轻松，原始 GRIB 不用动。

--------------------------------------------------------------------------
GRIB 读取库（scan 不需要，inspect/extract 需要，装任意一个即可）

    pip install eccodes                # 最推荐: 最轻、自带二进制、最快
    pip install pygrib                 # 备选
    conda install -c conda-forge wgrib2   # 或系统装 wgrib2，脚本会自动调用

脚本自动探测装了哪个。都没有时 inspect/extract 会明确报错并给出安装命令，
不会静默失败。
--------------------------------------------------------------------------
速度与规模（在 500MB 的 gfs.t00z.pgrb2.0p25.fHHH 上实测）

    单文件约 1.7 秒（瓶颈是扫整个文件，~300MB/s 的 I/O，不是解码）
    两年 × 2 轮/天 × 3 小时步长到 48h ≈ 25000 个文件 ≈ 12 小时（单进程）

所以务必:
  1. 先 --limit 20 试通，确认变量和数值对
  2. 再用 --jobs（按服务器核数，比如 --jobs 8）跑全量，nohup 挂后台
  3. 中断了直接重跑，会读已有 CSV 自动跳过抽过的，不会重复劳动

    nohup python3 gfs_extract.py extract /path/to/gfs \
        --out gfs_stations.csv --jobs 8 > extract.log 2>&1 &
    tail -f extract.log
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

# 8 个机场。经度用 0-360，与 GFS 网格一致（脚本内部会同时兼容 -180~180）
STATIONS = {
    "ZBAA": (40.0801, 116.5846),   # 北京首都
    "ZSPD": (31.1443, 121.8083),   # 上海浦东
    "ZGGG": (23.3924, 113.2988),   # 广州白云
    "ZGSZ": (22.6393, 113.8107),   # 深圳宝安
    "ZUUU": (30.5785, 103.9471),   # 成都双流
    "ZUCK": (29.7192, 106.6417),   # 重庆江北
    "ZHHH": (30.7838, 114.2081),   # 武汉天河
    "ZSQD": (36.3617, 120.3864),   # 青岛胶东
}

# 想要的变量。key 是 cfgrib/pygrib 里的短名，值是给人看的说明。
# t2m 是必须的，其余有就抽、没有就跳过。
WANT = {
    "t2m":   "2m 气温（必须）",
    "r2":    "2m 相对湿度",
    "d2m":   "2m 露点",
    "tcc":   "总云量",
    "dswrf": "向下短波辐射",
    "hpbl":  "边界层高度（见顶时刻的直接度量）",
    "tsfc":  "地表皮温",
    "prmsl": "海平面气压",
    "sp":    "地面气压",
    "u10":   "10m 风 u",
    "v10":   "10m 风 v",
    "tp":    "累积降水",
    "prate": "降水率",
    "gust":  "阵风",
    "cape":  "对流有效位能",
}
# 各后端 / 各版本 ecCodes 的短名不统一，全部映射到上面的 key。
# 2026-08-01: 用户归档的 _surf 子集里是 2t / 2r / 10u / 10v / dswrf /
# hpbl / tp / prmsl / t(surface)，其中 2r(相对湿度) 和 hpbl(边界层高度)
# 原来没在表里，会被静默丢掉。
ALIAS = {
    "2t": "t2m", "2d": "d2m", "2r": "r2", "10u": "u10", "10v": "v10",
    "TMP": "t2m", "DPT": "d2m", "RH": "r2", "TCDC": "tcc", "DSWRF": "dswrf",
    "HPBL": "hpbl", "blh": "hpbl", "PRMSL": "prmsl", "PRES": "sp", "msl": "prmsl",
    "sdswrf": "dswrf", "avg_dswrf": "dswrf", "DSWRF_surface": "dswrf",
    "UGRD": "u10", "VGRD": "v10", "PRATE": "prate", "APCP": "tp",
    "GUST": "gust", "CAPE": "cape",
}

# dswrf/prate/cape/gust 在有些时效档里没有（f000 无累积量），
# 所以提前退出只能等这几个「一定有」的
CORE = {"t2m", "d2m", "tcc", "sp", "u10", "v10"}

# 末尾允许 _surf 之类的后缀，但**必须锚定行尾** —— 归档里每个数据文件旁边
# 都有一个同名的 .OK 零字节标记文件（gfs.t12z.pgrb2.0p25.f000_surf.OK），
# 不锚定的话它们会被当成 GRIB 去读，文件数翻倍、还全是读取失败。
FN = re.compile(r"gfs\.t(\d{2})z\.pgrb2\.?[^.]*\.f(\d{3})(?:_[A-Za-z0-9]+)?$")
DATE = re.compile(r"(20\d{2})(\d{2})(\d{2})")
UTC = timezone.utc


def parse_name(path):
    """从路径解出 (起报 datetime, 时效小时)。解不出返回 None。

    日期优先从路径里找 gfs.YYYYMMDD/ 这种目录名；找不到就在整个路径里搜。
    """
    m = FN.search(os.path.basename(path))
    if not m:
        return None
    hh, fh = int(m.group(1)), int(m.group(2))
    d = None
    for part in reversed(path.replace("\\", "/").split("/")):
        g = DATE.search(part)
        if g:
            d = g.groups()
            break
    if not d:
        return None
    try:
        init = datetime(int(d[0]), int(d[1]), int(d[2]), hh, tzinfo=UTC)
    except ValueError:
        return None
    return init, fh


def walk(root):
    """收数据文件。跳过 .OK 标记和空文件。"""
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith((".OK", ".ok", ".idx", ".tmp")):
                continue
            if not FN.search(fn):
                continue
            p = os.path.join(dp, fn)
            try:
                if os.path.getsize(p) < 1024:      # 零字节/残缺
                    continue
            except OSError:
                continue
            out.append(p)
    return sorted(out)


# ============================================================ scan

def cmd_scan(a):
    files = walk(a.root)
    if not files:
        print(f"[!] {a.root} 下没找到 gfs.tHHz.pgrb2....fHHH 形式的文件。")
        print("    确认路径对不对，或文件名是不是被改过。")
        return 1
    by, size, bad = defaultdict(list), 0, 0
    for p in files:
        g = parse_name(p)
        if not g:
            bad += 1
            continue
        by[g[0]].append(g[1])
        try:
            size += os.path.getsize(p)
        except OSError:
            pass
    print(f"文件总数 {len(files)}   可解析 {len(files)-bad}   解析失败 {bad}")
    print(f"总大小   {size/1073741824:.1f} GB")
    if not by:
        print("[!] 一个都没解析出日期。贴一个完整路径给我看看。")
        print(f"    示例: {files[0]}")
        return 1
    inits = sorted(by)
    print(f"起报时刻 {inits[0]:%Y-%m-%d %H}Z  ~  {inits[-1]:%Y-%m-%d %H}Z")
    days = sorted({d.date() for d in inits})
    print(f"覆盖 {len(days)} 天，{len(inits)} 轮起报")
    runs = defaultdict(int)
    for d in inits:
        runs[d.hour] += 1
    print("每轮次的次数: " + "   ".join(f"{h:02d}Z × {n}" for h, n in sorted(runs.items())))
    fhs = sorted({f for v in by.values() for f in v})
    print(f"时效档 f{min(fhs):03d} ~ f{max(fhs):03d}，共 {len(fhs)} 档")
    if len(fhs) > 1:
        steps = sorted({fhs[i+1]-fhs[i] for i in range(len(fhs)-1)})
        print(f"  步长 {steps}   完整档位: {fhs[:14]}{' …' if len(fhs) > 14 else ''}")
    n = [len(v) for v in by.values()]
    print(f"每轮平均 {sum(n)/len(n):.1f} 档，最少 {min(n)}，最多 {max(n)}")
    # 缺天
    miss = []
    d0, d1 = days[0], days[-1]
    have = set(days)
    i = d0
    while i <= d1:
        if i not in have:
            miss.append(i.isoformat())
        i += timedelta(days=1)
    print(f"缺失日期 {len(miss)} 天" + (f"，前 8 个: {miss[:8]}" if miss else "  （无缺失）"))
    # 只有 00Z 或只有 12Z 的天
    half = sorted(d.isoformat() for d in days
                  if len({x.hour for x in inits if x.date() == d}) < 2)
    print(f"只有单轮的日期 {len(half)} 天" + (f"，前 8 个: {half[:8]}" if half else ""))
    print(f"\n示例路径: {files[0]}")
    print(f"\n下一步: python3 {os.path.basename(__file__)} inspect \"{files[len(files)//2]}\"")
    return 0


# ============================================================ survey

def cmd_survey(a):
    """抽样扫描整个归档，回答「哪个变量从哪天开始有、00Z 和 12Z 是否一致」。

    归档跨度两年多，业务下载的变量集可能中途改过（实测 2024-04 是 9 个变量、
    7.6MB，2026-06 变成 14 个、14MB，多了 sp 和 lcc/mcc/hcc/tcc）。
    不先摸清楚就抽，训练集会出现说不清来源的缺测。
    """
    kind, obj = need_backend()
    files = walk(a.root)
    if not files:
        print(f"[!] {a.root} 下没找到文件。")
        return 1
    by = defaultdict(list)
    for p in files:
        g = parse_name(p)
        if g and g[1] <= a.fh_max:
            by[(g[0].date(), g[0].hour)].append(p)
    keys = sorted(by)
    if not keys:
        print("[!] 没有符合 --fh-max 的文件。")
        return 1
    # 按时间均匀抽 n 个 (日期,轮次)，两端一定取到\n")
    n = min(a.samples, len(keys))
    picks = [keys[round(i * (len(keys) - 1) / max(1, n - 1))] for i in range(n)]
    seen, order = {}, []
    print(f"共 {len(keys)} 个 (日期,轮次)，均匀抽 {len(set(picks))} 个探测")
    print(f"  {'日期':<12}{'轮':<5}{'MB':>6}  变量")
    for k in sorted(set(picks)):
        p = sorted(by[k])[len(by[k]) // 2]
        names = set()
        try:
            if kind == "eccodes":
                with open(p, "rb") as fh_:
                    while True:
                        gid = obj.codes_grib_new_from_file(fh_)
                        if gid is None:
                            break
                        try:
                            sn = obj.codes_get(gid, "shortName")
                            if sn == "t" and obj.codes_get(gid, "typeOfLevel") == "surface":
                                sn = "tsfc"
                            names.add(ALIAS.get(sn, sn))
                        finally:
                            obj.codes_release(gid)
            elif kind == "pygrib":
                gr = obj.open(p)
                try:
                    for m in gr:
                        sn = getattr(m, "shortName", "")
                        if sn == "t" and getattr(m, "typeOfLevel", "") == "surface":
                            sn = "tsfc"
                        names.add(ALIAS.get(sn, sn))
                finally:
                    gr.close()
            else:
                r = subprocess.run([obj, "-s", p], capture_output=True,
                                   text=True, timeout=300)
                for ln in r.stdout.splitlines():
                    f_ = ln.split(":")
                    if len(f_) > 3:
                        names.add(ALIAS.get(f_[3], f_[3]))
        except Exception as e:
            print(f"  {k[0]}  {k[1]:02d}Z  读取失败 {type(e).__name__}")
            continue
        use = sorted(names & set(WANT))
        seen[k] = use
        order.append(k)
        mb = os.path.getsize(p) / 1048576
        print(f"  {str(k[0]):<12}{k[1]:02d}Z  {mb:>5.1f}  {' '.join(use)}")
    if not seen:
        print("\n[!] 一个都没读成功。")
        return 1
    print(f"\n各变量的可用区间（按抽样点）")
    allv = sorted({v for u in seen.values() for v in u})
    print(f"  {'变量':<8}{'出现':>6}/{len(seen):<5}{'最早':<12}{'最晚':<12}  说明")
    for v in allv:
        hit = [k for k in order if v in seen[k]]
        print(f"  {v:<8}{len(hit):>6}/{len(seen):<5}{str(hit[0][0]):<12}"
              f"{str(hit[-1][0]):<12}  {WANT.get(v,'')}")
    miss = sorted(set(WANT) - set(allv))
    if miss:
        print(f"\n归档里从来没有的: {' '.join(miss)}")
    print(f"\n00Z vs 12Z 的变量集是否一致")
    for hh in sorted({k[1] for k in seen}):
        sets = {tuple(v) for k, v in seen.items() if k[1] == hh}
        print(f"  {hh:02d}Z: {len(sets)} 种不同的变量组合"
              + ("  （一致）" if len(sets) == 1 else "  （中途变过，见上表）"))
    a0 = {tuple(v) for k, v in seen.items() if k[1] == 0}
    a12 = {tuple(v) for k, v in seen.items() if k[1] == 12}
    if a0 and a12:
        print(f"  两轮次的组合集合{'相同' if a0 == a12 else '不同'}")
    return 0


# ============================================================ GRIB 后端

def backend():
    """返回 ('eccodes'|'cfgrib'|'pygrib'|'wgrib2', 模块或路径)。都没有返回 (None, None)。

    首选 eccodes 的底层接口: 它自带 codes_grib_find_nearest（取最近格点），
    不需要 xarray/pandas 那一整套，在服务器上最好装、最快。
    """
    try:
        import eccodes
        eccodes.codes_get_api_version()
        return "eccodes", eccodes
    except Exception:
        pass
    try:
        import cfgrib
        if hasattr(cfgrib, "open_datasets"):        # 没装 xarray 时这个属性不存在
            return "cfgrib", cfgrib
    except Exception:
        pass
    try:
        import pygrib
        return "pygrib", pygrib
    except Exception:
        pass
    for exe in ("wgrib2", "/usr/local/bin/wgrib2", "/opt/homebrew/bin/wgrib2"):
        try:
            subprocess.run([exe, "-version"], capture_output=True, timeout=20)
            return "wgrib2", exe
        except Exception:
            continue
    return None, None


def need_backend():
    kind, obj = backend()
    if kind is None:
        print("[!] 没有可用的 GRIB 读取库。装其中任意一个即可：\n")
        print("    pip install eccodes               # 最推荐，最轻，自带二进制")
        print("    pip install cfgrib eccodes xarray # cfgrib 需要 xarray 才有 open_datasets")
        print("    pip install pygrib")
        print("    conda install -c conda-forge wgrib2\n")
        print("  （scan 子命令不需要这些，可以先跑 scan 把归档情况发我）")
        sys.exit(2)
    return kind, obj


# ============================================================ inspect

def cmd_inspect(a):
    kind, obj = need_backend()
    p = a.path
    print(f"后端: {kind}")
    print(f"文件: {p}")
    try:
        print(f"大小: {os.path.getsize(p)/1048576:.1f} MB")
    except OSError as e:
        print(f"[!] 打不开: {e}")
        return 1
    g = parse_name(p)
    print(f"文件名解析: " + (f"起报 {g[0]:%Y-%m-%d %H}Z  时效 f{g[1]:03d}"
                            if g else "失败 —— 把完整路径发我"))
    names = set()
    print("\n变量清单:")
    if kind == "eccodes":
        ec = obj
        try:
            fh_ = open(p, "rb")
        except Exception as e:
            print(f"[!] 打不开: {e}")
            return 1
        seen_msg = 0
        with fh_:
            while True:
                gid = ec.codes_grib_new_from_file(fh_)
                if gid is None:
                    break
                seen_msg += 1
                try:
                    sn = ec.codes_get(gid, "shortName")
                    lv = ec.codes_get(gid, "typeOfLevel")
                    l0 = ec.codes_get(gid, "level")
                    nm = ec.codes_get(gid, "name")
                    key = ALIAS.get(sn, sn)
                    if key not in names:
                        names.add(key)
                        print(f"  {sn:<10}{str(nm)[:44]:<46}{lv}={l0}")
                finally:
                    ec.codes_release(gid)
        print(f"\n  （共 {seen_msg} 条 GRIB 消息）")
    elif kind == "cfgrib":
        try:
            dss = obj.open_datasets(p, backend_kwargs={"indexpath": ""})
        except Exception as e:
            print(f"[!] 读取失败: {type(e).__name__}: {e}")
            return 1
        for ds in dss:
            lv = [k for k in ds.coords
                  if k in ("surface", "heightAboveGround", "isobaricInhPa",
                           "atmosphere", "meanSea", "depthBelowLandLayer")]
            tag = f"{lv[0]}={ds[lv[0]].values}" if lv else ""
            for v in ds.data_vars:
                names.add(v)
                print(f"  {v:<10}{str(ds[v].attrs.get('long_name',''))[:46]:<48}{tag}")
    elif kind == "pygrib":
        try:
            gr = obj.open(p)
        except Exception as e:
            print(f"[!] 读取失败: {type(e).__name__}: {e}")
            return 1
        for m in gr:
            sn = getattr(m, "shortName", "?")
            names.add(ALIAS.get(sn, sn))
            print(f"  {sn:<10}{str(getattr(m,'name',''))[:46]:<48}"
                  f"{getattr(m,'typeOfLevel','')}={getattr(m,'level','')}")
        gr.close()
    else:
        r = subprocess.run([obj, "-s", p], capture_output=True, text=True, timeout=300)
        for ln in r.stdout.splitlines():
            f = ln.split(":")
            if len(f) > 4:
                names.add(ALIAS.get(f[3], f[3]))
                print(f"  {f[3]:<10}{f[4][:56]}")
    print(f"\n识别到 {len(names)} 个变量名。")
    print("\n我们要用的:")
    for k, why in WANT.items():
        hit = k in names or any(ALIAS.get(n) == k for n in names)
        print(f"  {'✓' if hit else '·'} {k:<8}{why}")
    ok = "t2m" in names or any(ALIAS.get(n) == "t2m" for n in names)
    print("\n" + ("  ✓ 有 t2m，可以往下走。"
                  if ok else
                  "  [!] 没有 t2m。若这是 pgrb2 的子集文件，2m 气温可能在另一个分片里，\n"
                  "      换个文件再 inspect 一次；或者告诉我下载时用的变量列表。"))
    print(f"\n下一步（先小批量试通）:")
    print(f"  python3 {os.path.basename(__file__)} extract \"{a.root or '/你的/gfs目录'}\""
          f" --out test.csv --limit 20")
    return 0


# ============================================================ extract

def nearest_idx(lats, lons, la, lo):
    """最近格点。lons 可能是 0-360 也可能是 -180~180，两种都试。"""
    import numpy as np
    lo2 = lo % 360.0
    cand = [lo, lo2, lo2 - 360.0]
    j = min(range(len(lons)),
            key=lambda k: min(abs(lons[k] - c) for c in cand))
    i = min(range(len(lats)), key=lambda k: abs(lats[k] - la))
    return i, j


def read_one(kind, obj, path):
    """读一个 GRIB，返回 {var: {station: value}}。读不出返回 {}。"""
    import numpy as np
    out = defaultdict(dict)
    if kind == "eccodes":
        ec = obj
        try:
            fh_ = open(path, "rb")
        except Exception:
            return {}
        with fh_:
            while True:
                try:
                    gid = ec.codes_grib_new_from_file(fh_)
                except Exception:
                    break
                if gid is None:
                    break
                try:
                    sn = ec.codes_get(gid, "shortName")
                    key = ALIAS.get(sn, sn)
                    if key == "t" and ec.codes_get(gid, "typeOfLevel") == "surface":
                        key = "tsfc"
                    if key not in WANT or key in out:
                        continue                    # 每个变量只取第一条（地面层）
                    for s_, (la, lo) in STATIONS.items():
                        try:
                            nr = ec.codes_grib_find_nearest(gid, la, lo % 360.0)
                        except Exception:
                            continue
                        if nr:
                            v = nr[0].value
                            if v is not None and v == v:
                                out[key][s_] = float(v)
                finally:
                    ec.codes_release(gid)
                # **不做提前退出**。实测想要的变量散布到文件 84% 处，
                # 提前停只省 16%（瓶颈是扫 500MB 文件本身，~300MB/s 的 I/O），
                # 却会丢掉排在后面的 prate / cape —— 这份数据抽一次要十几小时，
                # 为省 16% 留个缺口不划算。提速靠 --jobs 并行和 --fh-min/max 筛时效。
        return out
    if kind == "cfgrib":
        try:
            dss = obj.open_datasets(path, backend_kwargs={"indexpath": ""})
        except Exception:
            return {}
        for ds in dss:
            if "latitude" not in ds.coords or "longitude" not in ds.coords:
                continue
            lats = np.asarray(ds["latitude"].values).ravel()
            lons = np.asarray(ds["longitude"].values).ravel()
            for v in ds.data_vars:
                key = ALIAS.get(v, v)
                if key not in WANT:
                    continue
                arr = np.asarray(ds[v].values)
                if arr.ndim != 2:
                    continue
                for s, (la, lo) in STATIONS.items():
                    i, j = nearest_idx(lats, lons, la, lo)
                    try:
                        val = float(arr[i, j])
                    except (IndexError, ValueError):
                        continue
                    if val == val:                      # 排除 NaN
                        out[key][s] = val
        return out
    if kind == "pygrib":
        try:
            gr = obj.open(path)
        except Exception:
            return {}
        idx = None
        try:
            for m in gr:
                sn = getattr(m, "shortName", "")
                key = ALIAS.get(sn, sn)
                # surface 层的 t 是地表皮温，与 2m 气温不是一回事，单独存
                if key == "t" and getattr(m, "typeOfLevel", "") == "surface":
                    key = "tsfc"
                if key not in WANT or key in out:
                    continue
                try:
                    vals = m.values
                except Exception:
                    continue
                if idx is None:                     # 网格对所有消息都一样，只算一次
                    la2d, lo2d = m.latlons()
                    idx = {}
                    for s_, (la, lo) in STATIONS.items():
                        d = (la2d - la) ** 2 + (((lo2d - lo % 360 + 180) % 360) - 180) ** 2
                        idx[s_] = np.unravel_index(int(np.argmin(d)), d.shape)
                for s_, (i, j) in idx.items():
                    try:
                        v = float(vals[i, j])
                    except Exception:
                        continue
                    if v == v:
                        out[key][s_] = v
        finally:
            gr.close()
        return out
    # wgrib2: 一次调用把 8 个站全取出来
    pts = ":".join(f"{lo % 360}:{la}" for la, lo in STATIONS.values())
    try:
        r = subprocess.run([obj, path, "-lon", *sum(
            ([str(lo % 360), str(la)] for la, lo in STATIONS.values()), []), "-s"],
            capture_output=True, text=True, timeout=600)
    except Exception:
        return {}
    order = list(STATIONS)
    for ln in r.stdout.splitlines():
        f = ln.split(":")
        if len(f) < 5:
            continue
        key = ALIAS.get(f[3], f[3])
        if key not in WANT:
            continue
        vals = re.findall(r"val=([-\d.eE+]+)", ln)
        for s, v in zip(order, vals):
            try:
                out[key][s] = float(v)
            except ValueError:
                pass
    return out


_W = {}


def _init_worker(kind):
    """子进程各自建一次后端句柄。GRIB 库的句柄不能跨进程传。"""
    _W["kind"] = kind
    _W["obj"] = backend()[1]


def _work(path):
    try:
        return read_one(_W["kind"], _W["obj"], path)
    except Exception:
        return {}


def cmd_extract(a):
    kind, obj = need_backend()
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("[!] 需要 numpy: pip install numpy")
        return 2
    files = walk(a.root)
    if not files:
        print(f"[!] {a.root} 下没找到文件。")
        return 1
    done = set()
    new = not os.path.exists(a.out)
    if not new:                                  # 断点续跑
        try:
            with open(a.out, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    done.add((r["init_utc"], int(r["fhour"])))
            print(f"已有 {a.out}，其中 {len(done)} 轮×档已抽过，跳过它们。")
        except Exception as e:
            print(f"[warn] 旧文件读不了（{e}），将覆盖重来。")
            new = True
    todo = []
    for p in files:
        g = parse_name(p)
        if not g:
            continue
        if a.fh_max and g[1] > a.fh_max:
            continue
        if g[1] < a.fh_min:
            continue
        if (g[0].isoformat(), g[1]) in done:
            continue
        todo.append((p, g[0], g[1]))
    if a.limit:
        todo = todo[:a.limit]
    print(f"后端 {kind}   待处理 {len(todo)} 个文件")
    if not todo:
        print("没有要处理的。")
        return 0
    fh = open(a.out, "a" if not new else "w", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new:
        w.writerow(["station", "init_utc", "fhour", "valid_utc", "var", "val"])
    n_ok = n_bad = n_row = 0
    t0 = time.time()

    def emit(i, item, got):
        nonlocal n_ok, n_bad, n_row
        p_, init_, fhr_ = item
        if not got:
            n_bad += 1
        else:
            n_ok += 1
            valid = (init_ + timedelta(hours=fhr_)).isoformat()
            for var, per in got.items():
                for s_, v in per.items():
                    w.writerow([s_, init_.isoformat(), fhr_, valid, var, round(v, 3)])
                    n_row += 1
        if i % 50 == 0 or i == len(todo):
            fh.flush()
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            print(f"  {i}/{len(todo)}  成功 {n_ok} 失败 {n_bad}  已写 {n_row} 行  "
                  f"用时 {el/60:.0f} 分  预计还要 {eta/3600:.1f} 小时", flush=True)

    if a.jobs > 1:
        # 每个文件互不相干，并行就是线性加速。服务器上按核数给 --jobs
        import multiprocessing as mp
        with mp.Pool(a.jobs, initializer=_init_worker, initargs=(kind,)) as pool:
            for i, (item, got) in enumerate(
                    zip(todo, pool.imap(_work, [t[0] for t in todo], chunksize=4)), 1):
                emit(i, item, got)
    else:
        for i, item in enumerate(todo, 1):
            emit(i, item, read_one(kind, obj, item[0]))
    fh.close()
    print(f"\n完成: {a.out}  {n_row} 行  "
          f"{os.path.getsize(a.out)/1048576:.1f} MB")
    if n_bad:
        print(f"[warn] {n_bad} 个文件读取失败（可能是损坏或空文件）")
    print("\n把这个 CSV 发我就行，原始 GRIB 不用动。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="抽取 GFS 归档里 8 个机场站点的序列")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="盘点归档（不需要 GRIB 库）")
    s.add_argument("root")
    s.set_defaults(fn=cmd_scan)
    s = sub.add_parser("inspect", help="看单个文件有哪些变量")
    s.add_argument("path")
    s.add_argument("--root", default="")
    s.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("survey", help="抽样扫描: 哪个变量从哪天开始有")
    s.add_argument("root")
    s.add_argument("--samples", type=int, default=40, help="抽多少个时间点")
    s.add_argument("--fh-max", type=int, default=48)
    s.set_defaults(fn=cmd_survey)
    s = sub.add_parser("extract", help="抽成 CSV")
    s.add_argument("root")
    s.add_argument("--out", default="gfs_stations.csv")
    s.add_argument("--limit", type=int, default=0, help="只处理前 N 个文件（试跑用）")
    s.add_argument("--fh-max", type=int, default=48,
                   help="只要时效 <= 这个小时的（默认 48，D+1/D+2 够用）")
    s.add_argument("--fh-min", type=int, default=0, help="只要时效 >= 这个小时的")
    s.add_argument("--jobs", type=int, default=1,
                   help="并行进程数。服务器上按核数给，比如 --jobs 8")
    s.set_defaults(fn=cmd_extract)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

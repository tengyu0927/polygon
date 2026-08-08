#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gfs_local.py — 读本地 NCEP GFS GRIB2（0.25°），抽 8 个站点的逐时序列

    ./.venv-grib/bin/python gfs_local.py --inspect /path/to/gfs.t12z.pgrb2.0p25.f018
    ./.venv-grib/bin/python gfs_local.py --scan /path/to/gfs_root
    ./.venv-grib/bin/python gfs_local.py --extract /path/to/gfs_root --db gfs_local.sqlite

为什么要用本地文件而不是继续用 Open-Meteo:

Open-Meteo 的 `previous_day1` 是「约 24 小时时效那一轮」，粒度只到天，
**判定不了它到底是哪一次起报**。而本地文件名里 `t00z`/`t12z` + `f018`
把起报时刻和时效写死了 —— 时效可精确控制，不存在泄漏歧义。

时效差距（目标 = 当天 14 时北京 = 06Z）:

    现在用的 previous_day1                  ~24-30 h
    前一天 12Z（20:00 起报，次日 00:30 可用）   18 h   -> 9:15 起报能用
    当天 00Z（08:00 起报，当天 12:00 可用）      6 h   -> 13:15 起报能用

实测时效的价值（Open-Meteo，287 站日）: 48h 时效 MAE 2.146、
24h 1.994、最新一轮 1.770 —— 时效每近一档就明显更准。

取舍: 本地是 0.25°(~28km)，比 Open-Meteo 的 GFS 0.11°(~13km) 粗一倍。
**分辨率变粗、时效变新**，哪个赢必须实测，不能拍脑袋。
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from collections import defaultdict

import stations as _S  # 站点清单唯一真相源
STATIONS = _S.GFS_COORD

# 文件名形如 gfs.t12z.pgrb2.0p25.f018；有的归档还会带日期目录 gfs.20260731/
FN = re.compile(r"gfs\.t(\d{2})z\.pgrb2\.?(\w*)\.f(\d{3})")
DATE = re.compile(r"(20\d{6})")

DDL = """CREATE TABLE IF NOT EXISTS gfs (
  station TEXT, init_utc TEXT, fhour INT, valid_utc TEXT,
  var TEXT, val REAL,
  PRIMARY KEY (station, init_utc, fhour, var));
CREATE INDEX IF NOT EXISTS gfs_v ON gfs(station, valid_utc, var);"""


def parse_name(path):
    """从路径里解出 (起报日期, 起报小时, 预报时效)。解不出返回 None。"""
    m = FN.search(os.path.basename(path))
    if not m:
        return None
    d = DATE.search(path)
    return (d.group(1) if d else None, int(m.group(1)), int(m.group(3)))


def inspect(path):
    """列出这个文件里有哪些变量 —— 决定能不能用之前必须先看这个。"""
    import cfgrib
    print(f"文件: {path}")
    print(f"大小: {os.path.getsize(path)/1048576:.1f} MB")
    got = parse_name(path)
    print(f"文件名解析: 日期={got[0]} 起报={got[1]:02d}Z 时效=f{got[2]:03d}"
          if got else "文件名解析: 失败（名字不是 gfs.tHHz.pgrb2...fHHH 的形式）")
    print("\n变量清单:")
    seen = []
    try:
        dss = cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""})
    except Exception as e:
        print(f"  读取失败: {type(e).__name__}: {e}")
        return 1
    for ds in dss:
        lvl = [k for k in ds.coords if k in
               ("surface", "heightAboveGround", "isobaricInhPa",
                "depthBelowLandLayer", "atmosphere", "meanSea")]
        tag = f"{lvl[0]}={ds[lvl[0]].values}" if lvl else "?"
        for v in ds.data_vars:
            a = ds[v].attrs
            seen.append(v)
            print(f"  {v:<14}{a.get('long_name','')[:44]:<46}{tag}")
    print(f"\n共 {len(seen)} 个变量。")
    need = {"t2m": "2m 气温（必须有）", "tcc": "总云量", "dswrf": "向下短波辐射",
            "d2m": "2m 露点", "sp": "地面气压", "u10": "10m 风 u", "v10": "10m 风 v"}
    print("\n我们需要的:")
    for k, why in need.items():
        print(f"  {'✓' if k in seen else '✗'} {k:<8}{why}")
    if "t2m" not in seen:
        print("\n  [!] 没有 t2m。若是 pgrb2 的子集文件，2m 气温可能在别的分片里。")
    return 0


def scan(root):
    """盘点归档: 有多少天、每天几轮、时效到多少、总大小。"""
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if FN.search(fn):
                files.append(os.path.join(dp, fn))
    if not files:
        print(f"{root} 下没找到 gfs.tHHz.pgrb2...fHHH 形式的文件。", file=sys.stderr)
        return 1
    by = defaultdict(list)
    size = 0
    for p in files:
        g = parse_name(p)
        if not g:
            continue
        by[(g[0], g[1])].append(g[2])
        size += os.path.getsize(p)
    days = sorted({k[0] for k in by if k[0]})
    print(f"共 {len(files)} 个文件，{size/1073741824:.1f} GB")
    if days:
        print(f"日期范围 {days[0]} ~ {days[-1]}（{len(days)} 天）")
    runs = defaultdict(int)
    for (d, h), fh in by.items():
        runs[h] += 1
    print(f"起报轮次: " + "  ".join(f"{h:02d}Z×{n}" for h, n in sorted(runs.items())))
    fhs = sorted({f for v in by.values() for f in v})
    print(f"时效: f{min(fhs):03d} ~ f{max(fhs):03d}，共 {len(fhs)} 档"
          f"（步长 {fhs[1]-fhs[0] if len(fhs) > 1 else '?'}）")
    n_per = [len(v) for v in by.values()]
    print(f"每轮平均 {sum(n_per)/len(n_per):.1f} 个时效档")
    # 缺哪些天
    if len(days) > 1:
        from datetime import date, timedelta
        d0 = date(int(days[0][:4]), int(days[0][4:6]), int(days[0][6:]))
        d1 = date(int(days[-1][:4]), int(days[-1][4:6]), int(days[-1][6:]))
        want = {(d0 + timedelta(days=i)).strftime("%Y%m%d")
                for i in range((d1 - d0).days + 1)}
        miss = sorted(want - set(days))
        print(f"缺失 {len(miss)} 天" + (f"，前几个: {miss[:5]}" if miss else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", help="看单个文件里有哪些变量")
    ap.add_argument("--scan", help="盘点整个归档目录")
    ap.add_argument("--db", default="gfs_local.sqlite")
    a = ap.parse_args()
    if a.inspect:
        return inspect(a.inspect)
    if a.scan:
        return scan(a.scan)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

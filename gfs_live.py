#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gfs_live.py — 实时从 NCEP 取「当前能拿到的最新一轮」GFS，抽 8 站，出 mos.csv 格式

    python3 gfs_live.py --cutoff 13                     # 按 13 时起报选轮次
    python3 gfs_live.py --cutoff 9  --out gfs_now.csv
    python3 gfs_live.py --cutoff 13 --date 2026-08-02 --dry-run   # 只看会选哪一轮

为什么要这个: 实测更新一轮的模式带来**十六次尝试里最大的一次改善**
（2026-08-02，用户本地归档 855 天的数据验证）:

    9 时起报   30h -> 18h  MAE 1.1424 -> 1.0843  (-0.058)  ±1℃ 70.6% -> 73.6%
              30h ->  6h        -> 1.0309  (-0.111)  ±1℃ -> 75.1%   P=100%
    11 时      30h ->  6h  0.8736 -> 0.8197  (-0.054)  P=100%
    13 时      30h ->  6h  0.5206 -> 0.5076  (-0.013)  P=98%

**越早的起报时刻收益越大** —— 早时次实测少、更依赖模式。
而现在训练/预测用的 Open-Meteo `previous_day1` 是约 30 小时时效那一轮。

轮次选择（实测落地滞后约 4 小时，848 次 00Z / 854 次 12Z）:

    起报(北京)   能拿到的最新轮        到 14 时的时效
    09:15-11:15  前一天 12Z(20:00起报, 00:00落地)   18 h
    12:15-15:15  当天  00Z(08:00起报, 12:00落地)    6 h

    18Z(北京 02:00 起报、06:00 落地) 能让 9-11 时再降到 12 h，
    线性内插约再 -0.026 —— 但那要另外回补历史训练数据，见 README。

取数方式: 用 GRIB2 的 .idx 索引 + HTTP Range，**只下需要的那几个变量**，
单个时效档约 9 MB 而不是整文件 500 MB。并行下，一轮 17 个档约半分钟。
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CST = datetime.timezone(datetime.timedelta(hours=8))
UTC = datetime.timezone.utc
S3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
UA = "ploygon-nowcast/1.0 (station Tmax research)"

import stations as _S  # 站点清单唯一真相源
STATIONS = _S.GFS_COORD

# .idx 里的行长这样: 581:426136393:d=2026070218:TMP:2 m above ground:12 hour fcst:
# 匹配「变量:层次」两段，映射到我们要的名字
WANT = {
    ("TMP", "2 m above ground"): "t2m",
    ("RH", "2 m above ground"): "r2",
    ("UGRD", "10 m above ground"): "u10",
    ("VGRD", "10 m above ground"): "v10",
    ("DSWRF", "surface"): "dswrf",
    ("HPBL", "surface"): "hpbl",
    ("TCDC", "entire atmosphere"): "tcc",
    ("PRES", "surface"): "sp",
    ("PRMSL", "mean sea level"): "prmsl",
}
PEAK0, PEAK1 = 10, 19


def http(url, rng=None, retries=3, timeout=60):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:                       # noqa: BLE001
            last = e
            if i < retries - 1:
                import time
                time.sleep(2 * (i + 1))
    raise RuntimeError(f"{url} 取数失败: {last}")


# 有历史训练数据的时效。**预测用的时效必须与训练用的一致** ——
# 拿 18h 时效训练出来的订正量去喂 12h 时效的输入，模型的校准就错了，
# 这是本项目最忌讳的「训练/预测口径不一致」。
#
# 2026-08-03: 18Z 历史已回补（853 天、零失败，见 gfs_backfill18.py），
# 12 已加入。9/10/11 时因此从「前一天 12Z 的 18h」切到「前一天 18Z 的 12h」。
# 回测（15 个月滚动、按日配对自助 N=2万）: 全部 0.8369 -> 0.8307、
# Δ=-0.0062、P=95.2%，压着 95% 的闸门过。单档都没到线（9/10/11 时
# 分别 85%/89%/85%），是三档同向累出来的。
#
# 00Z / 12Z 对应时效 6h / 18h / 30h / 42h，18Z 对应 12h。
TRAINED_LEADS = {6, 12, 18, 30, 42}


# 落地滞后的判定阈值。实测（用户归档 848 次 00Z / 854 次 12Z 的文件 mtime）
# 00Z 的 f000~f018 落地中位是起报后 3.9~4.0 小时（北京时 11:53~12:00）。
#
# **这个值必须与训练时用的时效逐档对上**，否则就是训练/预测口径不一致:
#   4.25 -> 9/10/11 时选 12h、12/13 时选 6h  <- 与训练一致（当前）
#   4.5  -> 12 时被踢到 12h，而 12 时的模型是拿 6h 训的 -> 错配
# 2026-08-02 一开始写的 4.5 就是错的，被这条注释挡住的正是那个坑。
LAG_H = 4.25


def pick_run(cutoff_h, target: datetime.date, lag_h=LAG_H, match_lead=True):
    """按起报时刻选「那时能拿到的最新一轮」。

    lag_h 见上方 LAG_H 的说明 —— 改它等于改「哪个时次用哪个时效」，
    必须同步重训，不能单独调。
    match_lead=True 时只返回有训练数据的时效，避免口径不一致。
    返回 [(起报 datetime, 说明), ...]，按新到旧排，供逐级降级。
    """
    cut = datetime.datetime.combine(target, datetime.time(cutoff_h, 15), CST)
    out = []
    for back in range(0, 4):                       # 往回找 4 轮（00/06/12/18Z）
        init = (datetime.datetime.combine(target, datetime.time(0), UTC)
                - datetime.timedelta(hours=6 * back))
        # 目标日 14 时（北京）= 06Z
        peak = datetime.datetime.combine(target, datetime.time(6), UTC)
        lead = (peak - init).total_seconds() / 3600
        if lead <= 0:
            continue                                # 起报晚于见顶时段，无意义
        if init + datetime.timedelta(hours=lag_h) > cut:
            continue                                # 那时还没落地
        if match_lead and round(lead) not in TRAINED_LEADS:
            continue                                # 没有这个时效的训练数据
        out.append((init, f"{init:%m-%d %HZ} 时效 {lead:.0f}h"))
    return out


def fetch_run(init: datetime.datetime, target: datetime.date, jobs=8, verbose=True):
    """下一轮里覆盖目标日（北京时）的全部时效档，返回 {var: {北京小时: 值}}。"""
    try:
        import eccodes as ec
    except ImportError:
        print("[error] 需要 eccodes: pip install eccodes", file=sys.stderr)
        return None
    day0 = datetime.datetime.combine(target, datetime.time(0), CST)
    fhs = []
    for h in range(0, 73):
        v = init + datetime.timedelta(hours=h)
        if day0 <= v.astimezone(CST) < day0 + datetime.timedelta(days=1):
            fhs.append(h)
    if not fhs:
        return None
    base = (f"{S3}/gfs.{init:%Y%m%d}/{init:%H}/atmos/"
            f"gfs.t{init:%H}z.pgrb2.0p25.f")

    def one(fh):
        url = f"{base}{fh:03d}"
        try:
            lines = http(url + ".idx", timeout=30).decode().splitlines()
        except Exception:
            return fh, None
        offs = [int(l.split(":")[1]) for l in lines]
        jobs_ = []
        for i, l in enumerate(lines):
            f = l.split(":")
            if len(f) < 5:
                continue
            key = WANT.get((f[3], f[4]))
            if not key:
                continue
            a = int(f[1])
            b = (offs[i + 1] - 1) if i + 1 < len(offs) else a + 4_000_000
            jobs_.append((key, a, b))
        if not jobs_:
            return fh, None
        # **按变量分开存**，不要拼成一个 buf 再靠 shortName 认。
        # 实测 eccodes 把 DSWRF 解成 sdswrf、把 HPBL 解成 unknown ——
        # 靠 shortName 匹配会静默丢掉这两个变量（2026-08-02 踩过）。
        # 我们从 .idx 就知道每一段是什么，直接带着标签走，最可靠。
        out_ = {}
        for key, a, b in sorted(jobs_, key=lambda x: x[1]):
            try:
                out_[key] = http(url, (a, b), timeout=60)
            except Exception:
                continue
        return fh, (out_ or None)

    got = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for fh, buf in ex.map(one, fhs):
            if buf:
                got[fh] = buf
            if verbose and len(got) % 5 == 0 and got:
                print(f"    已取 {len(got)}/{len(fhs)} 档", file=sys.stderr, flush=True)
    if not got:
        return None

    out = {}
    tmp = f"/tmp/_gfs_live_{os.getpid()}.grb2"
    for fh, per_var in sorted(got.items()):
        v_local = (init + datetime.timedelta(hours=fh)).astimezone(CST)
        for key, buf in per_var.items():
            with open(tmp, "wb") as f:
                f.write(buf)
            with open(tmp, "rb") as f:
                gid = ec.codes_grib_new_from_file(f)
                if gid is None:
                    continue
                try:
                    for s, (la, lo) in STATIONS.items():
                        nr = ec.codes_grib_find_nearest(gid, la, lo % 360.0)
                        if nr and nr[0].value == nr[0].value:
                            out.setdefault(s, {}).setdefault(key, {})[v_local.hour] = \
                                float(nr[0].value)
                finally:
                    ec.codes_release(gid)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return out


def dewpoint(t_c, rh):
    if rh is None or rh <= 0:
        return None
    a, b = 17.625, 243.04
    g = math.log(max(1e-3, rh / 100.0)) + a * t_c / (b + t_c)
    return b * g / (a - g)


def to_mos(per_station, target: datetime.date):
    """转成 mos.csv 的列名。与 gfs_local_build.agg 必须完全同口径 ——
    训练用那个、预测用这个，差一列就是静默降级。"""
    rows = []
    for s, ser in sorted(per_station.items()):
        t2m = {h: v - 273.15 for h, v in ser.get("t2m", {}).items()}
        if len(t2m) < 12:
            continue
        day = [h for h in t2m if PEAK0 <= h <= PEAK1]
        if len(day) < 8:
            continue
        f = {"station": s, "date": target.isoformat(), "lead": 1}

        def put(pref, vals, scale=1.0, off=0.0):
            if not vals:
                return
            a = [v * scale + off for v in vals.values()]
            p = [vals[h] * scale + off for h in vals if PEAK0 <= h <= PEAK1]
            f[f"{pref}_max"] = round(max(a), 3)
            if p:
                f[f"{pref}_peakmean"] = round(sum(p) / len(p), 3)

        put("temperature_2m", ser.get("t2m", {}), 1.0, -273.15)
        put("relative_humidity_2m", ser.get("r2", {}))
        put("cloud_cover", ser.get("tcc", {}))
        put("shortwave_radiation", ser.get("dswrf", {}))
        put("surface_pressure", ser.get("sp", {}), 0.01)
        put("boundary_layer_height", ser.get("hpbl", {}))
        rh = ser.get("r2", {})
        dp = {h: dewpoint(t2m[h], rh[h]) for h in t2m
              if h in rh and dewpoint(t2m[h], rh[h]) is not None}
        put("dew_point_2m", dp)
        u, v = ser.get("u10", {}), ser.get("v10", {})
        ws = {h: math.hypot(u[h], v[h]) for h in u if h in v}
        put("wind_speed_10m", ws)
        mx = max(t2m.values())
        f["t2m_range"] = round(mx - min(t2m.values()), 3)
        f["t2m_peak_h"] = min(h for h, x in t2m.items() if x >= mx - 1e-9)
        a12, a16 = t2m.get(12), t2m.get(16)
        f["t2m_slope_pm"] = None if (a12 is None or a16 is None) else round((a16 - a12) / 4, 3)
        late = [t2m[h] for h in t2m if 16 <= h <= 18]
        pm = f.get("temperature_2m_peakmean")
        f["t2m_late_minus_peak"] = (None if (not late or pm is None)
                                    else round(sum(late) / len(late) - pm, 3))
        # 边界层廓线。**必须与 gfs_local_build.agg 一字不差** ——
        # 训练算一套、预测算另一套，就是静默降级
        hp = ser.get("hpbl", {})
        if len(hp) >= 12:
            hm = max(hp.values())
            f["hpbl_peak_h"] = min(h for h, x in hp.items() if x >= hm - 1e-9)
            b12, b18 = hp.get(12), hp.get(18)
            f["hpbl_slope_pm"] = (None if (b12 is None or b18 is None)
                                  else round((b18 - b12) / 6, 3))
            if hm > 1:
                aft = sorted(h for h in hp if h > f["hpbl_peak_h"] and hp[h] < hm * 0.5)
                f["hpbl_half_h"] = aft[0] if aft else None
        rows.append(f)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, required=True, help="起报时刻(北京时整点)")
    ap.add_argument("--date", default="", help="目标日，默认今天")
    ap.add_argument("--out", default="gfs_now.csv")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="只打印会选哪一轮")
    ap.add_argument("--any-lead", action="store_true",
                    help="允许用没有训练数据的时效(如 18Z 的 12h)。"
                         "**除非你知道自己在做什么，否则别开** —— 会造成训练/预测口径不一致")
    a = ap.parse_args()
    tgt = (datetime.date.fromisoformat(a.date) if a.date
           else datetime.datetime.now(CST).date())
    cand = pick_run(a.cutoff, tgt, match_lead=not a.any_lead)
    if not cand:
        print(f"[error] {a.cutoff}:15 起报时没有任何一轮已落地", file=sys.stderr)
        return 1
    print(f"目标日 {tgt}  {a.cutoff}:15 起报，可用轮次（新到旧）:", file=sys.stderr)
    for init, desc in cand:
        print(f"    {desc}", file=sys.stderr)
    if a.dry_run:
        return 0
    for init, desc in cand:
        print(f"  取 {desc} …", file=sys.stderr, flush=True)
        per = fetch_run(init, tgt, a.jobs)
        if not per:
            print(f"  [warn] {desc} 取不到，降级到上一轮", file=sys.stderr)
            continue
        rows = to_mos(per, tgt)
        if len(rows) < len(STATIONS):
            print(f"  [warn] 只成了 {len(rows)}/{len(STATIONS)} 站，降级", file=sys.stderr)
            continue
        cols = ["station", "date", "lead"]
        cols += sorted({k for r in rows for k in r} - set(cols))
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n写出 {a.out}: {len(rows)} 站 × {len(cols)} 列   用的是 {desc}",
              file=sys.stderr)
        return 0
    print("[error] 所有轮次都取不到", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

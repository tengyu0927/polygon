#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gfs_local_build.py — 把本地 GFS 归档抽出的站点序列转成 mos.csv 格式

    python3 gfs_local_build.py --gz gfs_stations.csv.gz --lead 6  --out mos_local6.csv
    python3 gfs_local_build.py --gz gfs_stations.csv.gz --lead 18 --out mos_local18.csv

为什么要转成 mos.csv 的列名: 这样 `train_nowcast.py --nwp-csv`、
`backtest_nowcast.py --nwp-csv`、`train_mos.py` 全都不用改一行代码，
A/B 就是「同一套代码 + 不同的 csv」，没有任何实现差异混进来。

--lead 的含义: 起报到目标日 14 时（北京）的小时数。
    6h  = 当天 00Z（北京 08:00 起报，约 12:00 落地）-> 12:15~15:15 起报可用
    18h = 前一天 12Z（北京 20:00 起报，约 00:00 落地）-> 9:15~11:15 可用
    30h = 前一天 00Z —— 这就是现在 Open-Meteo previous_day1 的口径，用作基线

实测原始模式日最高温误差（6484 个四档齐全的站日，完全配对）:
    6h  MAE 1.854    18h 1.979    30h 2.092    42h 2.166
    6h vs 30h  -0.238 (-11%)  P=100%   八个站全部改善

比 Open-Meteo 多出来的东西:
  - hpbl（边界层高度）: 见顶发生在混合层塌缩的午后转换时刻，
    这是该过程的直接度量。Open-Meteo 的 boundary_layer_height
    历史归档全空，一直拿不到。
比 Open-Meteo 少的:
  - wind_gusts_10m（阵风）: 归档里没有，留空，靠缺测指示位处理
  - cloud_cover / surface_pressure 在 2024-05-01 之前也没有
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import gzip
import math
import sqlite3
import sys

CST = datetime.timezone(datetime.timedelta(hours=8))
PEAK0, PEAK1 = 10, 19            # 午后峰值时段（北京时），与 build_mos_dataset 一致

# 本地变量名 -> mos.csv 的前缀。None 表示要派生，不直接映射
DIRECT = {
    "t2m": "temperature_2m",
    "r2": "relative_humidity_2m",
    "tcc": "cloud_cover",
    "dswrf": "shortwave_radiation",
    "sp": "surface_pressure",
    "hpbl": "boundary_layer_height",
}


def dewpoint(t_c, rh):
    """Magnus 公式。GFS 只给相对湿度，而 mos.csv 的列要露点。"""
    if rh is None or rh <= 0:
        return None
    a, b = 17.625, 243.04
    g = math.log(max(1e-3, rh / 100.0)) + a * t_c / (b + t_c)
    return b * g / (a - g)


def load(gz, lead, verbose=True):
    """读抽取的 csv.gz，按 (站, 目标日) 聚出指定时效那一轮的逐时序列。"""
    want_lead = lead
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    n = 0
    with gzip.open(gz, "rt") as fh:
        for r in csv.DictReader(fh):
            n += 1
            if verbose and n % 2000000 == 0:
                print(f"  读了 {n:,} 行…", file=sys.stderr, flush=True)
            v = datetime.datetime.fromisoformat(r["valid_utc"]).astimezone(CST)
            init = datetime.datetime.fromisoformat(r["init_utc"])
            day = v.strftime("%Y-%m-%d")
            peak = (datetime.datetime.strptime(day, "%Y-%m-%d")
                    .replace(tzinfo=CST) + datetime.timedelta(hours=14))
            if round((peak - init).total_seconds() / 3600) != want_lead:
                continue
            out[(r["station"], day)][r["var"]][v.hour] = float(r["val"])
    return out


def agg(series):
    """把逐时序列聚成 mos.csv 的那套列。"""
    f = {}
    t2m = {h: v - 273.15 for h, v in series.get("t2m", {}).items()}
    if len(t2m) < 12:                       # 白天覆盖不全的一律丢掉
        return None
    day = [h for h in t2m if PEAK0 <= h <= PEAK1]
    if len(day) < 8:
        return None

    def put(pref, vals, scale=1.0, off=0.0):
        if not vals:
            return
        a = [v * scale + off for v in vals.values()]
        p = [vals[h] * scale + off for h in vals if PEAK0 <= h <= PEAK1]
        f[f"{pref}_max"] = round(max(a), 3)
        if p:
            f[f"{pref}_peakmean"] = round(sum(p) / len(p), 3)

    put("temperature_2m", series.get("t2m", {}), 1.0, -273.15)
    put("relative_humidity_2m", series.get("r2", {}))
    put("cloud_cover", series.get("tcc", {}))
    put("shortwave_radiation", series.get("dswrf", {}))
    put("surface_pressure", series.get("sp", {}), 0.01)      # Pa -> hPa
    put("boundary_layer_height", series.get("hpbl", {}))

    # 露点由 t2m + r2 反算（GFS 归档没有 d2m）
    rh = series.get("r2", {})
    dp = {h: dewpoint(t2m[h], rh[h]) for h in t2m if h in rh and dewpoint(t2m[h], rh[h]) is not None}
    put("dew_point_2m", dp)

    # 风速由 u/v 合成
    u, v = series.get("u10", {}), series.get("v10", {})
    ws = {h: math.hypot(u[h], v[h]) for h in u if h in v}
    put("wind_speed_10m", ws)
    # wind_gusts_10m 归档里没有 —— 留空，让缺测指示位去处理，别拿风速冒充

    # 温度廓线（与 build_mos_dataset.daily_features 同口径）
    mx = max(t2m.values())
    f["t2m_range"] = round(mx - min(t2m.values()), 3)
    f["t2m_peak_h"] = min(h for h, x in t2m.items() if x >= mx - 1e-9)
    a12, a16 = t2m.get(12), t2m.get(16)
    f["t2m_slope_pm"] = None if (a12 is None or a16 is None) else round((a16 - a12) / 4, 3)
    late = [t2m[h] for h in t2m if 16 <= h <= 18]
    pm = f.get("temperature_2m_peakmean")
    f["t2m_late_minus_peak"] = (None if (not late or pm is None)
                                else round(sum(late) / len(late) - pm, 3))

    # 边界层廓线。见顶时刻由「混合层什么时候塌」决定，所以峰值时刻和
    # 午后衰减速率比单一均值更有针对性 —— 这是 Open-Meteo 给不了的部分
    hp = series.get("hpbl", {})
    if len(hp) >= 12:
        hm = max(hp.values())
        f["hpbl_peak_h"] = min(h for h, x in hp.items() if x >= hm - 1e-9)
        b12, b18 = hp.get(12), hp.get(18)
        f["hpbl_slope_pm"] = (None if (b12 is None or b18 is None)
                              else round((b18 - b12) / 6, 3))
        if hm > 1:
            # 塌到峰值一半的时刻 ≈ 湍流混合停止、气温开始回落
            aft = sorted(h for h in hp if h > f["hpbl_peak_h"] and hp[h] < hm * 0.5)
            f["hpbl_half_h"] = aft[0] if aft else None
    return f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gz", default="gfs_stations.csv.gz")
    ap.add_argument("--lead", type=int, default=6,
                    help="起报到目标日 14 时（北京）的小时数: 6/18/30/42")
    ap.add_argument("--obs-db", default="cn.sqlite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bias-days", type=int, default=7)
    a = ap.parse_args()

    print(f"读 {a.gz}，取时效 {a.lead}h 那一轮…", file=sys.stderr, flush=True)
    raw = load(a.gz, a.lead)
    print(f"  {len(raw):,} 个 (站, 目标日)", file=sys.stderr)

    conn = sqlite3.connect(a.obs_db)
    obs = {(s, d): v for s, d, v in conn.execute(
        "SELECT station, local_date, MAX(temp_c) FROM obs "
        "WHERE local_date >= '2024-03-01' GROUP BY 1, 2 HAVING COUNT(*) >= 18")}
    print(f"  实况 {len(obs):,} 站日", file=sys.stderr)

    feats = {}
    for k, ser in raw.items():
        g = agg(ser)
        if g:
            feats[k] = g
    print(f"  白天覆盖达标 {len(feats):,}", file=sys.stderr)

    rows = []
    for (stn, d), f in sorted(feats.items()):
        y = obs.get((stn, d))
        if y is None or f.get("temperature_2m_max") is None:
            continue
        dt = datetime.date.fromisoformat(d)
        # 因果特征: 发布日 = 目标日 - ceil(lead/24)，只回看那天及更早
        issue = dt - datetime.timedelta(days=max(1, math.ceil(a.lead / 24)) if a.lead > 14 else 0)
        prev = obs.get((stn, (dt - datetime.timedelta(days=1)).isoformat()))
        resid, back = [], 1
        while len(resid) < a.bias_days and back <= a.bias_days * 3:
            bd = (issue - datetime.timedelta(days=back)).isoformat()
            o, m = obs.get((stn, bd)), feats.get((stn, bd), {}).get("temperature_2m_max")
            if o is not None and m is not None:
                resid.append(o - m)
            back += 1
        bias = (round(sum(resid) / len(resid), 3)
                if len(resid) >= a.bias_days // 2 else None)
        doy = dt.timetuple().tm_yday
        rec = {"station": stn, "date": d, "lead": 1, "y_tmax": y,
               "prev_tmax": prev, "recent_bias": bias,
               "doy_sin1": round(math.sin(2 * math.pi * doy / 365.25), 4),
               "doy_cos1": round(math.cos(2 * math.pi * doy / 365.25), 4),
               "doy_sin2": round(math.sin(4 * math.pi * doy / 365.25), 4),
               "doy_cos2": round(math.cos(4 * math.pi * doy / 365.25), 4)}
        rec.update(f)
        rows.append(rec)

    if not rows:
        print("[!] 一行都没配上。检查站号和日期范围。", file=sys.stderr)
        return 1
    cols = ["station", "date", "lead", "y_tmax", "prev_tmax", "recent_bias",
            "doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
    cols += sorted({k for r in rows for k in r} - set(cols))
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n写出 {a.out}: {len(rows):,} 行 × {len(cols)} 列", file=sys.stderr)
    d = [r["date"] for r in rows]
    print(f"  {min(d)} ~ {max(d)}   有 recent_bias 的 "
          f"{sum(1 for r in rows if r['recent_bias'] is not None):,} 行", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_mos_dataset.py — 站点日最高温 MOS 的训练集构建

分三步，可分别执行:
    python3 build_mos_dataset.py probe                     # 探测各变量的可用起始年份
    python3 build_mos_dataset.py fetch --start 2021-03-01  # 拉固定时效因子
    python3 build_mos_dataset.py build --obs-db ~/cn.sqlite --out mos.csv

为什么用 Previous Runs API 而不是 Historical Forecast API:
    后者是把每次运行的前几小时拼成连续序列，值里含有"预报时刻拿不到"的信息。
    用它训练偏差订正，离线指标会好看，上线就崩。
    Previous Runs 按固定时效对齐: _previous_day1 = 提前 24h 预报出的值。

时效方向:
    目标是次日 14 时前后的峰值。若业务上 20:00 发布，实际提前约 18h，
    而训练用的是 24h 时效 —— 上线时信息比训练时更多，是安全方向。
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

CST = timezone(timedelta(hours=8))
API = "https://previous-runs-api.open-meteo.com/v1/forecast"
UA = "mos-dataset/1.0 (station Tmax post-processing research)"

STATIONS = {
    "ZBAA": (40.0801, 116.5846), "ZSPD": (31.1434, 121.8052),
    "ZGGG": (23.3924, 113.2988), "ZGSZ": (22.6393, 113.8107),
    "ZUUU": (30.5785, 103.9471), "ZUCK": (29.7192, 106.6417),
    "ZHHH": (30.7838, 114.2081), "ZSQD": (36.3661, 120.0864),
}
# 注意: Previous Runs 端点不支持气压层变量（temperature_850hPa 等会报
# SurfacePressureAndHeightVariable 解析错误）。要高空因子得走 Single Runs API。
VARS = ["temperature_2m", "cloud_cover", "shortwave_radiation",
        "wind_speed_10m", "dew_point_2m", "relative_humidity_2m",
        "surface_pressure", "wind_gusts_10m"]
LEADS = [1, 2]                      # D+1 / D+2
PEAK_H0, PEAK_H1 = 10, 19           # 午后峰值时段(北京时)

DDL = """
CREATE TABLE IF NOT EXISTS fcst (
    station TEXT, valid_cst TEXT, var TEXT, lead INTEGER, val REAL,
    PRIMARY KEY (station, valid_cst, var, lead));
"""


def _fetch(params: dict, timeout: int = 120, retries: int = 3) -> list[dict]:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for a in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read().decode("utf-8"))
            return d if isinstance(d, list) else [d]
        except urllib.error.HTTPError as e:
            try:
                why = json.loads(e.read().decode("utf-8")).get("reason", "")
            except Exception:
                why = str(e.reason)
            if e.code == 429 and a < retries:
                time.sleep(60 * a)
                continue
            raise RuntimeError(f"HTTP {e.code}: {why}") from None
        except Exception:
            if a < retries:
                time.sleep(10 * a)
                continue
            raise
    return []


# ============================================================ probe

def cmd_probe(args):
    """逐变量试探: 哪些变量在哪些年份真的有非空数据。"""
    lat, lon = STATIONS["ZSPD"]
    print("变量 × 年份 可用性（√=有数据, ·=全空, ✗=接口拒绝）\n")
    years = list(range(2021, datetime.now(CST).year + 1))
    print(f"  {'变量':<24}" + "".join(f"{y:>7}" for y in years))
    for v in VARS:
        row = []
        for y in years:
            try:
                d = _fetch({"latitude": lat, "longitude": lon,
                            "hourly": f"{v}_previous_day1", "models": "gfs_global",
                            "start_date": f"{y}-06-01", "end_date": f"{y}-06-05",
                            "timezone": "Asia/Shanghai"})[0]
                vals = [x for x in d.get("hourly", {}).get(f"{v}_previous_day1", [])
                        if x is not None]
                row.append("     √" if vals else "     ·")
            except Exception:
                row.append("     ✗")
            time.sleep(0.4)
        print(f"  {v:<24}" + "".join(f"{c:>7}" for c in row))
    print("\n注: · 表示接口接受但该年无归档。选起始年份时以最短的那个变量为准，")
    print("    或者放弃它换取更长的训练期 —— 样本量和特征数之间要权衡。")


# ============================================================ fetch

def cmd_fetch(args):
    conn = sqlite3.connect(args.db)
    conn.executescript(DDL)
    ids = list(STATIONS)
    variables = [s.strip() for s in args.vars.split(",") if s.strip()]
    hourly = [f"{v}_previous_day{d}" for v in variables for d in LEADS]

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else \
        datetime.now(CST).date() - timedelta(days=1)

    chunks, cur = [], start
    while cur <= end:
        nxt = min(cur + timedelta(days=args.chunk - 1), end)
        chunks.append((cur, nxt))
        cur = nxt + timedelta(days=1)

    # 先逐个校验变量，避免一个坏变量拖垮整次请求（报错信息不会告诉你是哪个）
    print("校验变量…", file=sys.stderr)
    ok = []
    for v in variables:
        try:
            _fetch({"latitude": STATIONS[ids[0]][0], "longitude": STATIONS[ids[0]][1],
                    "hourly": f"{v}_previous_day1", "models": args.model,
                    "start_date": start.isoformat(),
                    "end_date": (start + timedelta(days=2)).isoformat(),
                    "timezone": "Asia/Shanghai"})
            ok.append(v)
            print(f"  ✓ {v}", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ {v}  {str(e)[:90]}", file=sys.stderr)
        time.sleep(0.4)
    if not ok:
        print("没有可用变量，终止。", file=sys.stderr)
        return 1
    hourly = [f"{v}_previous_day{d}" for v in ok for d in LEADS]

    print(f"\n{len(chunks)} 个时间块 × {len(ids)} 站 × {len(hourly)} 序列", file=sys.stderr)
    total = 0
    for i, (a, b) in enumerate(chunks, 1):
        try:
            data = _fetch({
                "latitude": ",".join(f"{STATIONS[s][0]:.4f}" for s in ids),
                "longitude": ",".join(f"{STATIONS[s][1]:.4f}" for s in ids),
                "hourly": ",".join(hourly), "models": args.model,
                "start_date": a.isoformat(), "end_date": b.isoformat(),
                "timezone": "Asia/Shanghai",
            })
        except Exception as e:
            print(f"  [warn] {a}~{b}: {e}", file=sys.stderr)
            if i == 1:                                 # 首块就失败，后面不必再试
                print("首个时间块失败，终止。先解决上面的报错。", file=sys.stderr)
                return 1
            continue
        n = 0
        for stn, loc in zip(ids, data):
            h = loc.get("hourly", {})
            times = h.get("time", [])
            for key, series in h.items():
                if key == "time":
                    continue
                base, _, tag = key.rpartition("_previous_day")
                if not tag.isdigit():
                    continue
                lead = int(tag)
                for t, v in zip(times, series):
                    if v is None:
                        continue
                    n += conn.execute("INSERT OR IGNORE INTO fcst VALUES (?,?,?,?,?)",
                                      (stn, t, base, lead, float(v))).rowcount
        conn.commit()
        total += n
        print(f"  {i}/{len(chunks)}  {a}~{b}  +{n}  累计 {total}", file=sys.stderr)
        time.sleep(args.sleep)
    conn.close()
    print(f"完成，共 {total} 条", file=sys.stderr)


# ============================================================ build

def load_obs_tmax(path, table, scol, tcol, vcol, tz) -> dict:
    """从实况库聚合北京时日最高温，带峰值时段完整性过滤。"""
    conn = _check_db(path)
    if conn is None:
        raise SystemExit(1)
    daily = defaultdict(list)
    for stn, ts, v in conn.execute(
            f"SELECT {scol},{tcol},{vcol} FROM {table} WHERE {vcol} IS NOT NULL"):
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc if tz == "utc" else CST)
        loc = dt.astimezone(CST)
        daily[(stn, loc.strftime("%Y-%m-%d"))].append((loc.hour, float(v)))
    conn.close()
    out = {}
    for k, vals in daily.items():
        if sum(1 for h, _ in vals if PEAK_H0 <= h <= PEAK_H1) < 6:
            continue                              # 峰值时段缺测则丢弃该日
        out[k] = max(v for _, v in vals)
    return out


def daily_features(conn, lead) -> dict:
    """把逐时预报聚合成目标日的日尺度特征。"""
    rows = conn.execute(
        "SELECT station, valid_cst, var, val FROM fcst WHERE lead = ?", (lead,))
    acc = defaultdict(lambda: defaultdict(list))
    for stn, t, var, v in rows:
        d, hh = t[:10], int(t[11:13])
        acc[(stn, d)][var].append((hh, v))

    out = {}
    for key, byvar in acc.items():
        f = {}
        for var, pts in byvar.items():
            peak = [v for h, v in pts if PEAK_H0 <= h <= PEAK_H1]
            allv = [v for _, v in pts]
            if not allv:
                continue
            f[f"{var}_max"] = max(allv)
            f[f"{var}_peakmean"] = sum(peak) / len(peak) if peak else None
            if var == "temperature_2m":
                f["t2m_range"] = max(allv) - min(allv)
                # 午后廓线。之前只留 max 和 10-19 时均值，等于把模式给的 24 个
                # 小时值压成两个数 —— 峰值出现在几点、午后是继续升还是很快回落，
                # 这些信息全丢了。而晚见顶正是当前最大的弱点
                # （重庆 46% 的日子 16-17 时才见顶，站级 ME -1.40）
                by_h = {}
                for h, v in pts:
                    by_h[h] = max(v, by_h.get(h, -99))
                mx = max(allv)
                pk = [h for h, v in by_h.items() if v >= mx - 1e-9]
                f["t2m_peak_h"] = min(pk) if pk else None
                a = by_h.get(12)
                b = by_h.get(16)
                # 12->16 时的升温速率: 正得多说明午后还在猛升、峰值偏晚
                f["t2m_slope_pm"] = None if (a is None or b is None) else (b - a) / 4
                late = [v for h, v in by_h.items() if 16 <= h <= 18]
                pm = f.get("temperature_2m_peakmean")
                # 傍晚相对午后均值: 正说明高温维持到傍晚
                f["t2m_late_minus_peak"] = (None if (not late or pm is None)
                                            else sum(late) / len(late) - pm)
            # 逐时廓线特征（2026-08-01 加）。见顶时刻由「能量输入什么时候停」
            # 决定，而这件事在**短波辐射**的逐时廓线里最直接 —— 云量常年饱和在
            # 100%（广州 2026-07-20 整个下午都是 100，毫无信息），
            # 而同一天辐射从 13 时的 744 骤降到 14 时 471、15 时 282。
            # 之前只喂 cloud_cover_peakmean / shortwave_radiation_peakmean 两个
            # 时段均值，等于把决定见顶时刻的那条曲线压成了一个数。
            if var == "shortwave_radiation":
                by_h = {}
                for h, v in pts:
                    by_h[h] = max(v, by_h.get(h, -99))
                day = {h: v for h, v in by_h.items() if 8 <= h <= 19}
                if day:
                    mx = max(day.values())
                    f["swrad_peak_h"] = min(h for h, v in day.items()
                                            if v >= mx - 1e-9)
                    if mx > 1:
                        # 辐射跌到峰值一半的时刻 ≈ 加热停止的时刻
                        aft = sorted(h for h in day if h > f["swrad_peak_h"]
                                     and day[h] < mx * 0.5)
                        f["swrad_half_h"] = aft[0] if aft else None
                        late = [day[h] for h in day if 15 <= h <= 17]
                        f["swrad_late_frac"] = (sum(late) / len(late) / mx
                                                if late else None)
                    a, b = day.get(12), day.get(16)
                    f["swrad_slope_pm"] = (None if (a is None or b is None)
                                           else (b - a) / 4)
            if var == "cloud_cover":
                by_h = {}
                for h, v in pts:
                    by_h[h] = max(v, by_h.get(h, -99))
                aft = sorted(h for h in by_h if 10 <= h <= 19 and by_h[h] >= 70)
                f["cld_onset_h"] = aft[0] if aft else None
                a, b = by_h.get(12), by_h.get(16)
                f["cld_slope_pm"] = (None if (a is None or b is None)
                                     else (b - a) / 4)
        if f.get("temperature_2m_max") is not None:
            out[key] = f
    return out


def load_daily_tmax(path, table, min_obs) -> dict:
    """注意: 会剔除「还没过完的今天」。半点报的站到上午就攒够 18 条观测，
    min_obs 拦不住它 —— 当天的 tmax 是截断值（实测北京标签 29℃ 而实际已达 31℃
    且未到峰值），混进训练集就是在教模型一个冷偏差，且每次重建都会发生一次。

    直接读预聚合好的日表，比扫逐时表快得多。"""
    conn = _check_db(path)
    if conn is None:
        raise SystemExit(1)
    cols = {c[1] for c in conn.execute(f"PRAGMA table_info({table})")}
    need = {"station", "date", "tmax"}
    if not need <= cols:
        print(f"[error] {table} 缺字段，需要 {need}，实际 {sorted(cols)}", file=sys.stderr)
        raise SystemExit(1)
    has_n = "n_obs" in cols
    sql = (f"SELECT station, date, tmax{', n_obs' if has_n else ''} "
           f"FROM {table} WHERE tmax IS NOT NULL")
    today = datetime.now(CST).date().isoformat()
    out, dropped, today_drop = {}, 0, 0
    for row in conn.execute(sql):
        if has_n and row[3] is not None and row[3] < min_obs:
            dropped += 1                          # 当日观测过少，Tmax 不可信
            continue
        if row[1] >= today:                       # 今天还没过完，tmax 是截断值
            today_drop += 1
            continue
        out[(row[0], row[1])] = float(row[2])
    conn.close()
    if dropped:
        print(f"  按 n_obs >= {min_obs} 过滤掉 {dropped} 个站日", file=sys.stderr)
    if today_drop:
        print(f"  剔除未过完的当天 {today_drop} 个站日", file=sys.stderr)
    return out


def cmd_build(args):
    conn = sqlite3.connect(args.db)
    if args.daily_table:
        obs = load_daily_tmax(args.obs_db, args.daily_table, args.min_obs)
    else:
        obs = load_obs_tmax(args.obs_db, args.obs_table, args.obs_station_col,
                            args.obs_time_col, args.obs_temp_col, args.obs_tz)
    print(f"实况: {len(obs)} 个站日", file=sys.stderr)

    rows = []
    for lead in LEADS:
        feats = daily_features(conn, lead)
        print(f"lead={lead}: {len(feats)} 个站日有预报", file=sys.stderr)
        for (stn, d), f in feats.items():
            y = obs.get((stn, d))
            if y is None:
                continue
            dt = datetime.strptime(d, "%Y-%m-%d").date()

            # ---- 因果特征: 只用发布时刻之前已知的信息 ----
            # 发布日 = 目标日 - lead。该日的实况 Tmax 当晚 20:00 已知
            issue_day = dt - timedelta(days=lead)
            prev = obs.get((stn, issue_day.isoformat()))

            # 近期"实况−模式"偏差：捕捉天气型相关的系统误差。
            # 只回看发布日及更早，且模式值取同一 lead，保证同口径
            resid, k = [], 0
            back = 1
            while k < args.bias_days and back <= args.bias_days * 3:
                bd = issue_day - timedelta(days=back - 1)
                o = obs.get((stn, bd.isoformat()))
                m = feats.get((stn, bd.isoformat()), {}).get("temperature_2m_max")
                if o is not None and m is not None:
                    resid.append(o - m)
                    k += 1
                back += 1
            bias = sum(resid) / len(resid) if len(resid) >= args.bias_days // 2 else None

            # 多窗口偏差（2026-08-01 加）。recent_bias 是 D+1 模型里权重最高的
            # 特征（标准化系数 +0.631），但它只有 7 天一个窗口、一个简单均值。
            # 天气型的持续时间不是固定 7 天 —— 短窗口跟得快但噪声大，
            # 长窗口稳但滞后。多给几个让模型自己挑，再给两个「这个偏差可不可靠」
            # 的量: 趋势（在扩大还是收敛）和离散度（最近几天错得一致不一致）。
            resid_long = []
            back = 1
            while len(resid_long) < 30 and back <= 60:
                bd = issue_day - timedelta(days=back - 1)
                o = obs.get((stn, bd.isoformat()))
                m = feats.get((stn, bd.isoformat()), {}).get("temperature_2m_max")
                if o is not None and m is not None:
                    resid_long.append(o - m)
                back += 1
            bw = {}
            for w in (3, 14, 30):
                v = resid_long[:w]
                bw[f"recent_bias_{w}"] = (round(sum(v) / len(v), 3)
                                          if len(v) >= max(2, w // 2) else None)
            b3, b14 = bw.get("recent_bias_3"), bw.get("recent_bias_14")
            # 趋势: 近 3 天比近 14 天更偏，说明偏差正在扩大
            bw["bias_trend"] = (None if (b3 is None or b14 is None)
                                else round(b3 - b14, 3))
            v7 = resid_long[:7]
            # 离散度: 最近几天的误差方向一致吗？大 = 这个偏差不可信
            bw["bias_sd_7"] = (round((sum((x - sum(v7) / len(v7)) ** 2
                                          for x in v7) / len(v7)) ** .5, 3)
                               if len(v7) >= 4 else None)

            doy = dt.timetuple().tm_yday
            rec = {
                "station": stn, "date": d, "lead": lead, "y_tmax": y,
                "prev_tmax": prev,
                "recent_bias": None if bias is None else round(bias, 3),
                **bw,
                "doy_sin1": round(math.sin(2 * math.pi * doy / 365.25), 4),
                "doy_cos1": round(math.cos(2 * math.pi * doy / 365.25), 4),
                "doy_sin2": round(math.sin(4 * math.pi * doy / 365.25), 4),
                "doy_cos2": round(math.cos(4 * math.pi * doy / 365.25), 4),
            }
            rec.update({k2: (None if v is None else round(v, 3))
                        for k2, v in f.items()})
            rows.append(rec)

    if not rows:
        print("没有配上的样本。检查站号写法和日期范围是否重叠。", file=sys.stderr)
        return

    cols = ["station", "date", "lead", "y_tmax", "prev_tmax", "recent_bias",
            "doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"]
    cols += sorted({k for r in rows for k in r} - set(cols))
    rows.sort(key=lambda r: (r["date"], r["station"], r["lead"]))

    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})

    dates = sorted({r["date"] for r in rows})
    print(f"\n写出 {args.out}: {len(rows)} 行 × {len(cols)} 列")
    print(f"日期 {dates[0]} ~ {dates[-1]}  站点 {len({r['station'] for r in rows})}")
    for lead in LEADS:
        sub = [r for r in rows if r["lead"] == lead]
        nb = sum(1 for r in sub if r["recent_bias"] is not None)
        print(f"  lead={lead}: {len(sub)} 行，其中 {nb} 行有 recent_bias")

    n_tr = int(len(dates) * 0.7)
    n_va = int(len(dates) * 0.85)
    print(f"\n建议的时间切分（绝对不要随机划分，相邻日高度相关）:")
    print(f"  训练 {dates[0]} ~ {dates[n_tr-1]}")
    print(f"  验证 {dates[n_tr]} ~ {dates[n_va-1]}")
    print(f"  测试 {dates[n_va]} ~ {dates[-1]}")
    print("\n必须打赢的三条基线: prev_tmax(持续性)、temperature_2m_max(模式原始)、TAF")
    conn.close()


def _check_db(path):
    """sqlite3.connect 遇到不存在的文件会静默建空库，必须先自己查。"""
    import os
    if not os.path.exists(path):
        print(f"[error] 文件不存在: {path}", file=sys.stderr)
        return None
    if os.path.getsize(path) < 4096:
        print(f"[error] {path} 只有 {os.path.getsize(path)} 字节，"
              f"多半是被 sqlite 静默创建的空库，删掉它", file=sys.stderr)
        return None
    conn = sqlite3.connect(path)
    tabs = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    if not tabs:
        print(f"[error] {path} 里没有任何表", file=sys.stderr)
        conn.close()
        return None
    return conn


def cmd_inspect(args):
    """列出实况库里的表和字段，用来确定 --obs-table / --obs-*-col 该填什么。"""
    conn = _check_db(args.obs_db)
    if conn is None:
        return 1
    for (name,) in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        cols = [(c[1], c[2]) for c in conn.execute(f"PRAGMA table_info({name})")]
        n = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
        print(f"\n{name}  ({n} 行)")
        for cn, ct in cols:
            print(f"    {cn:<24}{ct}")
        try:
            row = conn.execute(f"SELECT * FROM {name} LIMIT 1").fetchone()
            if row:
                print(f"    样例: {row}")
        except Exception:
            pass
    conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe")

    ins = sub.add_parser("inspect")
    ins.add_argument("--obs-db", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("--db", default="mos_fcst.sqlite")
    f.add_argument("--start", default="2021-03-01")
    f.add_argument("--end", default="")
    f.add_argument("--vars", default=",".join(VARS))
    f.add_argument("--model", default="gfs_global")
    f.add_argument("--chunk", type=int, default=180, help="每块天数")
    f.add_argument("--sleep", type=float, default=2.0)

    b = sub.add_parser("build")
    b.add_argument("--db", default="mos_fcst.sqlite")
    b.add_argument("--obs-db", required=True, help="你的实况库，如 ~/cn.sqlite")
    b.add_argument("--obs-table", default="obs")
    b.add_argument("--obs-station-col", default="station")
    b.add_argument("--obs-time-col", default="ts")
    b.add_argument("--obs-temp-col", default="tmpc")
    b.add_argument("--obs-tz", choices=["utc", "local"], default="utc")
    b.add_argument("--daily-table", default="",
                   help="直接用预聚合日表(如 daily)，比扫逐时表快得多")
    b.add_argument("--min-obs", type=int, default=18,
                   help="日表模式下 n_obs 下限，过滤缺测日")
    b.add_argument("--bias-days", type=int, default=7, help="近期偏差回看天数")
    b.add_argument("--out", default="mos.csv")

    args = ap.parse_args()
    return {"probe": cmd_probe, "inspect": cmd_inspect,
            "fetch": cmd_fetch, "build": cmd_build}[args.cmd](args) or 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run0_probe.py — 前瞻记录「起报时刻实际能取到的最新一轮模式」

    python3 run0_probe.py --log            # 每轮起报时调一次，落盘
    python3 run0_probe.py --analyze        # 攒够数据后分析

为什么需要这个（2026-08-01）:

现在训练和预测都用 `previous_day1`（约 24 小时时效的那一轮）。
实测原始模式 tmax 的误差:

    previous_day2 (~48h)              MAE 2.146
    previous_day1 (~24h，现在用的)      MAE 1.994
    无后缀 previous_day0（最新一轮）     MAE 1.770

**更新的一轮明显更准**（0.89 倍），且逐小时比值 0.82-0.92 无断崖，
说明它是真正的预报而不是分析场（分析场会让清晨那几个小时准得反常）。

但有一件事**无法从历史数据判定**: `previous_day0` 对某个历史日期返回的
究竟是哪一轮？如果是当天 06Z（北京时 14 点）或 12Z（20 点）的轮次，
那么 9:15 甚至 13:15 起报时它根本还没出来 —— 拿它训练就是用未来信息，
上线后会当场失效（训练时以为有、预报时取不到或取到旧的）。

**唯一严谨的判定办法是前瞻记录**: 在每个起报时刻真实地调一次接口，
把当时**实际拿到的** previous_day0 存下来；过一两周后，再用历史接口
拉同样日期的 previous_day0，两者比对：

- 若逐时次都一致 -> 历史接口给的就是当时可得的那一轮，可以放心用来训练
- 若历史接口的值更「新」（与实况更接近） -> 它含有当时还没出来的轮次，
  直接用来训练就是泄漏，只能改用 previous_day1

在判定清楚之前**不要**把 previous_day0 接进模型。

判定结果（2026-08-22，前瞻记录 22 天 / 32832 条）:

    当时实际取到的 previous_day0  vs  事后用历史接口拉的同一天
      完全一致 14790/32832 (45.0%)   最大差 5.5℃

**45% 远低于判据要求的 98% —— 历史接口确实含有当时还没出来的轮次。**
所以「回测一周出结论」那条路是死的，只剩文档里写的退路: **每轮起报时
自己存一份预报当训练集**，攒够天数再训。

于是 2026-08-22 把采集面补齐（见 run0h 表）。原来只存了 `gfs_global` 的
`temperature_2m` —— 而模型实际要的是**逐模式**的 tmax / cloud_peak /
swrad_peak（9 时 91 个 NWP 特征、12 时 41 个），且 README 自己写着
「那 10 个百分点是模式给的 —— 模式对**下午云量与辐射**的预报」。
照原样攒十个月，攒完会发现建不出特征。

    run0    旧表，只有 gfs_global 的 t2m。**保留** —— 上面那个判定靠它，
            别删，否则结论不可复现。
    run0h   新表，逐 (模式 × 站 × 小时) 一行，11 个变量各一列。
            覆盖生产在用的 7 个 Open-Meteo 模式。本地 GFS 不在内 ——
            它走自己的归档，且 gfs_live.pick_run 本来就取最新轮次。

量级: 10 站 × 24 时 × 7 模式 × 9 轮/天 = 15120 行/天，攒 300 天约 4.5M 行。

    python3 run0_probe.py --fields        # 查存的字段够不够建模型要的特征
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

CST = timezone(timedelta(hours=8))
API = "https://previous-runs-api.open-meteo.com/v1/forecast"
UA = "run0-probe/1.0 (station Tmax post-processing research)"
import stations as _S  # 站点清单唯一真相源
STATIONS = _S.COORD

# 与 build_mos_dataset.VARS 同一套，外加 train_nowcast 的对流三项
# （PLOYGON_CONV）。**顺序即 run0h 的列序，改了要一起改 DDL。**
VARS = ["temperature_2m", "cloud_cover", "shortwave_radiation",
        "wind_speed_10m", "dew_point_2m", "relative_humidity_2m",
        "surface_pressure", "wind_gusts_10m",
        "precipitation", "cape", "lifted_index"]

# 生产在用的追加模式，顺序必须与 run_hourly.sh 的 MODELS 一致 ——
# 对不上的话将来建出来的 m2_/m3_ 各列会错配到别的模式。
# 本地 GFS 不在这里: 它走自己的归档，pick_run 已经在取最新轮次。
MODELS = ["gfs_global", "ecmwf_ifs025", "cma_grapes_global", "icon_global",
          "jma_gsm", "gem_global", "ecmwf_aifs025_single"]

DDL = """CREATE TABLE IF NOT EXISTS run0 (
  probed_at TEXT, cutoff INT, station TEXT, target_date TEXT,
  hour INT, run0 REAL, run1 REAL,
  PRIMARY KEY (target_date, cutoff, station, hour));
CREATE TABLE IF NOT EXISTS run0h (
  probed_at TEXT, cutoff INT, model TEXT, station TEXT, target_date TEXT,
  hour INT,
  temperature_2m REAL, cloud_cover REAL, shortwave_radiation REAL,
  wind_speed_10m REAL, dew_point_2m REAL, relative_humidity_2m REAL,
  surface_pressure REAL, wind_gusts_10m REAL,
  precipitation REAL, cape REAL, lifted_index REAL,
  PRIMARY KEY (target_date, cutoff, model, station, hour));"""


def fetch(model="gfs_global", day=None):
    day = (day or datetime.now(CST).date()).isoformat()
    ids = list(STATIONS)
    url = ("https://previous-runs-api.open-meteo.com/v1/forecast?"
           + urllib.parse.urlencode({
               "latitude": ",".join(f"{STATIONS[s][0]:.4f}" for s in ids),
               "longitude": ",".join(f"{STATIONS[s][1]:.4f}" for s in ids),
               "hourly": "temperature_2m,temperature_2m_previous_day1",
               "models": model, "start_date": day, "end_date": day,
               "timezone": "Asia/Shanghai"}))
    data = json.load(urllib.request.urlopen(url, timeout=60))
    out = []
    for stn, loc in zip(ids, data):
        h = loc.get("hourly", {})
        for i, t in enumerate(h.get("time", [])):
            a = h["temperature_2m"][i]
            b = h["temperature_2m_previous_day1"][i]
            if a is None and b is None:
                continue
            out.append((stn, t[:10], int(t[11:13]), a, b))
    return out


def fetch_full(model, day):
    """某个模式在**此刻**能取到的最新一轮（无后缀 = run0），逐时全变量。

    只取目标日当天。返回 [(station, date, hour, [各变量值...]), ...]
    """
    ids = list(STATIONS)
    url = (API + "?" + urllib.parse.urlencode({
        "latitude": ",".join(f"{STATIONS[s][0]:.4f}" for s in ids),
        "longitude": ",".join(f"{STATIONS[s][1]:.4f}" for s in ids),
        "hourly": ",".join(VARS), "models": model,
        "start_date": day, "end_date": day, "timezone": "Asia/Shanghai"}))
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    data = json.load(urllib.request.urlopen(req, timeout=90))
    if not isinstance(data, list):
        data = [data]
    out = []
    for stn, loc in zip(ids, data):
        h = loc.get("hourly", {})
        for i, t in enumerate(h.get("time", [])):
            vals = [(h.get(v) or [None] * (i + 1))[i] for v in VARS]
            if all(v is None for v in vals):
                continue
            out.append((stn, t[:10], int(t[11:13]), vals))
    return out


def log_full(args, now, conn):
    """逐模式抓一遍存进 run0h。**任何一个模式失败都不许影响其余模式，
    更不许影响预报** —— 调用方在 run_hourly.sh 里是 `|| true`。"""
    cut = args.cutoff or now.hour
    day = now.date().isoformat()
    ok = bad = 0
    for mdl in MODELS:
        try:
            rows = fetch_full(mdl, day)
        except Exception as e:                             # noqa: BLE001
            print(f"[run0_probe] {mdl} 取数失败: {str(e)[:80]}", file=sys.stderr)
            bad += 1
            continue
        conn.executemany(
            "INSERT OR REPLACE INTO run0h VALUES ("
            + ",".join("?" * (6 + len(VARS))) + ")",
            [(now.isoformat(timespec="seconds"), cut, mdl, s_, d_, h_, *v)
             for s_, d_, h_, v in rows])
        ok += len(rows)
    conn.commit()
    print(f"[run0_probe] run0h 写入 {ok} 行"
          + (f"（{bad} 个模式失败）" if bad else ""), file=sys.stderr)


def log(args):
    now = datetime.now(CST)
    conn = sqlite3.connect(args.db)
    conn.executescript(DDL)
    rows = fetch(day=now.date())
    conn.executemany(
        "INSERT OR REPLACE INTO run0 VALUES (?,?,?,?,?,?,?)",
        [(now.isoformat(timespec="seconds"), args.cutoff or now.hour,
          s, d, h, a, b) for s, d, h, a, b in rows])
    conn.commit()
    print(f"[run0_probe] {now:%m-%d %H:%M} 记录 {len(rows)} 条", file=sys.stderr)
    # 全变量 × 全模式。失败不影响上面那张旧表（判定结论要靠它复现）
    try:
        log_full(args, now, conn)
    except Exception as e:                                 # noqa: BLE001
        print(f"[run0_probe] run0h 落盘失败: {str(e)[:80]}", file=sys.stderr)
    return 0


def analyze(args):
    if not os.path.exists(args.db):
        print(f"还没有 {args.db}，先让 run_hourly.sh 跑几天。", file=sys.stderr)
        return 1
    conn = sqlite3.connect(args.db)
    got = list(conn.execute(
        "SELECT target_date, cutoff, station, hour, run0 FROM run0 "
        "WHERE run0 IS NOT NULL"))
    if not got:
        print("库里还没有数据。", file=sys.stderr)
        return 1
    days = sorted({r[0] for r in got})
    print(f"前瞻记录 {len(days)} 天（{days[0]} ~ {days[-1]}），{len(got)} 条\n")

    # 用历史接口重新拉同样的日期，与当时实际记录的比
    import statistics as st
    same = diff = 0
    dmax = 0.0
    for d in days:
        try:
            now = {(s, h): a for s, _, h, a, _ in fetch(day=datetime.strptime(
                d, "%Y-%m-%d").date()) if a is not None}
        except Exception as e:
            print(f"  [warn] {d}: {e}", file=sys.stderr)
            continue
        for td, cut, s, h, v in got:
            if td != d or (s, h) not in now:
                continue
            if abs(now[(s, h)] - v) < 1e-6:
                same += 1
            else:
                diff += 1
                dmax = max(dmax, abs(now[(s, h)] - v))
    tot = same + diff
    if not tot:
        print("没有可比对的样本。", file=sys.stderr)
        return 1
    print(f"  当时实际取到的 previous_day0  vs  事后用历史接口拉的同一天")
    print(f"    完全一致 {same}/{tot} ({same/tot:.1%})   不一致 {diff}   最大差 {dmax:.1f}℃\n")
    if same / tot > 0.98:
        print("  → 历史接口给的就是当时可得的那一轮，**可以**用 previous_day0 训练。")
        print("     下一步: 在 build_mos_dataset.py 里加 previous_day0，做 A/B 回测。")
    else:
        print("  → 历史接口含有当时还没出来的轮次，用它训练就是泄漏。")
        print("     只能继续用 previous_day1，或改成每轮起报时自己存一份预报当训练集。")
    print(f"\n  逐时次一致率（看早时次是不是更容易对不上）")
    by = {}
    for d in days:
        pass
    for cut in sorted({r[1] for r in got}):
        n = sum(1 for r in got if r[1] == cut)
        print(f"    {cut:>2} 时  记录 {n} 条")
    return 0


def fields(args):
    """存的字段够不够把线上模型要的 NWP 特征建出来。

    这条检查存在的理由: 采集要攒十个月才够训练，而「字段少了」这件事
    十个月后才发现就全白攒了。2026-08-22 就是这么发现旧表只有 t2m 的。
    """
    import json as _j
    ok = True

    # 模型点名要的 NWP 列（nwp_* 走主模式，m2_..m8_ 走追加模式）
    try:
        import build_mos_dataset as B
        import train_nowcast as N
    except Exception as e:                                 # noqa: BLE001
        print(f"[warn] 读不到训练端定义（{e}），只能查表结构", file=sys.stderr)
        B = N = None

    print("\n[1] 变量覆盖")
    if B is not None:
        need = set(B.VARS)
        # daily_features 里 NWP_COLS 点名的原始变量
        for col in N.NWP_COLS.values():
            for v in B.VARS:
                if col.startswith(v):
                    need.add(v)
        miss = sorted(need - set(VARS))
        print(f"  训练端需要 {len(need)} 个原始变量，run0h 存了 {len(VARS)} 个")
        if miss:
            ok = False
            print(f"  ✗ 缺: {miss}")
        else:
            print(f"  ✓ 全覆盖（另多存 "
                  f"{sorted(set(VARS) - need)} 供对流因子用）")

    print("\n[2] 模式覆盖")
    print(f"  run0h 采 {len(MODELS)} 个: {', '.join(MODELS)}")
    print(f"  本地 GFS 不采 —— 它走自己的归档，pick_run 已取最新轮次")

    print("\n[3] 已攒到多少")
    if not os.path.exists(args.db):
        print("  库还不存在")
        return 0 if ok else 1
    conn = sqlite3.connect(args.db)
    try:
        n, nd, nm = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT target_date), "
            "COUNT(DISTINCT model) FROM run0h").fetchone()
    except sqlite3.OperationalError:
        print("  run0h 还没建（下一轮 run_hourly 会建）")
        return 0 if ok else 1
    print(f"  run0h: {n} 行 / {nd} 天 / {nm} 个模式")
    if nd:
        print(f"  按 300 天目标，还差 {max(0, 300 - nd)} 天")
        print(f"\n  逐模式逐时次的完整度（应为 10 站 × 24 时 = 240 行）")
        for mdl, cut, c in conn.execute(
                "SELECT model, cutoff, COUNT(*)/COUNT(DISTINCT target_date) "
                "FROM run0h GROUP BY 1,2 ORDER BY 1,2"):
            if c < 200:
                ok = False
                print(f"    ✗ {mdl} {cut} 时: 平均只有 {c} 行/天")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="run0_probe.sqlite")
    ap.add_argument("--cutoff", type=int, default=0)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--fields", action="store_true",
                    help="查存的字段够不够建模型要的特征")
    args = ap.parse_args()
    if args.fields:
        return fields(args)
    if args.analyze:
        return analyze(args)
    if args.log:
        return log(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

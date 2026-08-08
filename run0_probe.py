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
import stations as _S  # 站点清单唯一真相源
STATIONS = _S.COORD

DDL = """CREATE TABLE IF NOT EXISTS run0 (
  probed_at TEXT, cutoff INT, station TEXT, target_date TEXT,
  hour INT, run0 REAL, run1 REAL,
  PRIMARY KEY (target_date, cutoff, station, hour));"""


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="run0_probe.sqlite")
    ap.add_argument("--cutoff", type=int, default=0)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        return analyze(args)
    if args.log:
        return log(args)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

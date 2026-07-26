#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pair_taf_model.py — TAF 与模式在「同一批站日、可比时效」上的配对比较

    python3 pair_taf_model.py --taf-db taf_bias.sqlite --mos mos.csv

前面那次比较有两个不可比:
  1. 时效不同 —— TAF 的 1.26 是 15.9h 时效，模式 previous_day1 是 24h
  2. 样本不同 —— n=34 和 n=104 不是同一批站日
本脚本只保留两边都有的 (站, 日期)，并按 TAF 轮次分别对齐。
"""

from __future__ import annotations

import argparse
import csv
import math
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone

CST = timezone(timedelta(hours=8))
PEAK_H0, PEAK_H1 = 10, 19


def load_taf(db):
    """按 taf_bias 的口径: 排除边界 TX、去重搬运、归轮次。"""
    conn = sqlite3.connect(db)
    rows = list(conn.execute(
        "SELECT station, issue_utc, valid_utc, local_date, temp_c "
        "FROM raw_tx WHERE at_edge = 0 AND local_hour BETWEEN ? AND ?",
        (PEAK_H0, PEAK_H1)))
    conn.close()
    first = {}
    for s, iss, _, ld, tc in rows:
        k = (s, ld, tc)
        first[k] = min(first.get(k, iss), iss)

    out, kept = {}, set()
    for s, iss, val, ld, tc in rows:
        if iss != first[(s, ld, tc)] or (s, ld, tc) in kept:
            continue
        kept.add((s, ld, tc))
        ei = datetime.fromisoformat(iss)
        h = ei.astimezone(CST).hour
        cyc = ((h - 5) % 24) // 6 * 6 + 5
        day = (datetime.strptime(ld, "%Y-%m-%d").date() - ei.astimezone(CST).date()).days
        if day != 1:                                   # 只要 D+1
            continue
        lead = (datetime.fromisoformat(val) - ei).total_seconds() / 3600
        out.setdefault((s, ld), {})[cyc] = (tc, lead)
    return out


def load_mos(path):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r["lead"] != "1" or not r["temperature_2m_max"]:
            continue
        out[(r["station"], r["date"])] = (
            float(r["temperature_2m_max"]), float(r["y_tmax"]),
            float(r["recent_bias"]) if r["recent_bias"] else None)
    return out


def sc(e):
    n = len(e)
    if not n:
        return None
    me = sum(e) / n
    return (n, me, sum(abs(x) for x in e) / n,
            math.sqrt(sum(x * x for x in e) / n),
            100 * sum(1 for x in e if abs(x) <= 1) / n)


def show(tag, s, lead=None):
    if not s:
        print(f"  {tag:<26} (无样本)")
        return
    lt = f"{lead:5.1f}h" if lead is not None else "     "
    print(f"  {tag:<26} n={s[0]:>4} {lt}  ME={s[1]:+5.2f}  MAE={s[2]:5.2f}  "
          f"RMSE={s[3]:5.2f}  ±1℃={s[4]:4.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taf-db", default="taf_bias.sqlite")
    ap.add_argument("--mos", default="mos.csv")
    args = ap.parse_args()

    taf, mos = load_taf(args.taf_db), load_mos(args.mos)
    common = sorted(set(taf) & set(mos))
    if not common:
        print("两边没有共同的站日。检查日期范围是否重叠。")
        return
    dates = sorted({d for _, d in common})
    print(f"共同站日 {len(common)} 个 | {dates[0]} ~ {dates[-1]}\n")

    for cyc in (5, 11, 17, 23):
        pairs = [(k, taf[k][cyc]) for k in common if cyc in taf[k]]
        if len(pairs) < 10:
            continue
        print(f"═══ TAF {cyc:02d}时轮 vs 模式（限于这 {len(pairs)} 个共同站日）")
        lead = sorted(p[1][1] for p in pairs)[len(pairs) // 2]
        show("TAF", sc([v[0] - mos[k][1] for k, v in pairs]), lead)
        show("模式原始 (24h)", sc([mos[k][0] - mos[k][1] for k, _ in pairs]), 24)
        e = [mos[k][0] + mos[k][2] - mos[k][1] for k, _ in pairs if mos[k][2] is not None]
        show("模式+在线订正 (24h)", sc(e), 24)

        # 配对差值: 同一站日上两者绝对误差之差
        d = [abs(v[0] - mos[k][1]) -
             abs(mos[k][0] + (mos[k][2] or 0) - mos[k][1])
             for k, v in pairs if mos[k][2] is not None]
        if len(d) >= 10:
            m = sum(d) / len(d)
            se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d))
            print(f"  配对差 ΔMAE(TAF−订正模式) = {m:+.3f} ± {1.96*se:.3f}"
                  f"   {'模式更准' if m - 1.96*se > 0 else '差异不显著' if m + 1.96*se > 0 else 'TAF 更准'}")
        print()


if __name__ == "__main__":
    main()
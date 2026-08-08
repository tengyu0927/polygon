#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_hit_table.py — 由回测结果生成「预期命中率」查表。

    python3 build_hit_table.py bt_cur.csv bt_cur15.csv --out hit_table.json

为什么是「剩余升幅」这一个量:

两周生产实测（752 条）+ 15 个月回测都指向同一条主线 —— 决定准不准的不是
「今天什么天气」，而是「起报那一刻这一天还剩多少没发生」:

    起报时剩余升幅   MAE    完全命中   偏差
    <=0.5 度        0.32     72%    +0.32
    0.5-2 度        0.66     40%    -0.38
    2-4 度          1.14     28%    -0.94
    >4 度           1.42     21%    -1.28

挨个查过露点/湿度/气压/云量/对流/边界层，**没有任何单要素能独立预测误差**。
反例就在 2026-08-08: 上海风速异常 +5.3σ（全场最大）却 9 档全对；
成都零异常却误差 1.43（全场最差）。

**这个表不改任何预报值**，只是把「这个数该不该信」量化出来印在旁边。
预报值仍以「预报」那一列为准。

表按「起报时次 × 剩余升幅档」统计，因为同样剩 2 度，9 时和 13 时的把握不同。
样本不足的格子回退到该时次的整体命中率，再不够就回退到全局。
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys

# 剩余升幅的分档上界。最后一档用 inf。
# 分档边界。实测 0.5 度以下还有很强梯度，而且六个时次高度一致:
#   <=0.1 度 89-96% | 0.1-0.25 73-83% | 0.25-0.5 67-72% | >0.5 32-44%
# 所以细分低端、粗分高端 —— 高端各档之间差别很小（32~44%），细分没有意义。
EDGES = [0.1, 0.25, 0.5, 1.0, 2.0, float("inf")]
MIN_N = 40          # 格子样本数下限，不够就回退


def bucket(rise):
    for i, e in enumerate(EDGES):
        if rise <= e:
            return i
    return len(EDGES) - 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="backtest_nowcast --csv-out 的输出")
    ap.add_argument("--out", default="hit_table.json")
    a = ap.parse_args()

    rows = []
    for p in a.csvs:
        rows += list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        print("[error] 没读到数据", file=sys.stderr)
        return 1

    cells = collections.defaultdict(list)      # (cutoff, bucket) -> [hit]
    per_cut = collections.defaultdict(list)
    allhit = []
    for r in rows:
        c = int(r["cutoff"])
        pred, act = float(r["pred_mean"]), float(r["actual"])
        rise = float(r["pred_raw"]) - float(r["so_far"])
        hit = 1.0 if pred == act else 0.0
        cells[(c, bucket(rise))].append(hit)
        per_cut[c].append(hit)
        allhit.append(hit)

    g = sum(allhit) / len(allhit)
    table = {"edges": EDGES[:-1], "global": round(g, 4), "cells": {}, "per_cutoff": {}}
    for c, v in sorted(per_cut.items()):
        table["per_cutoff"][str(c)] = round(sum(v) / len(v), 4)
    n_fallback = 0
    for (c, b), v in sorted(cells.items()):
        if len(v) >= MIN_N:
            table["cells"][f"{c}|{b}"] = [round(sum(v) / len(v), 4), len(v)]
        else:
            n_fallback += 1
    json.dump(table, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    lbl = ["<=0.1", "0.1-.25", ".25-0.5", "0.5-1", "1-2", ">2"]
    print(f"  {'时次':<7}" + "".join(x.rjust(11) for x in lbl) + f"{'整体':>8}")
    for c in sorted(per_cut):
        line = f"  {c} 时{'':<3}"
        for b in range(len(EDGES)):
            v = table["cells"].get(f"{c}|{b}")
            line += (f"{v[0]:.0%}/{v[1]}" if v else "-").rjust(11)
        line += f"{table['per_cutoff'][str(c)]:>8.0%}"
        print(line)
    print(f"\n  全局 {g:.0%}，样本不足回退的格子 {n_fallback} 个 -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

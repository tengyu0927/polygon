#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_exceed_table.py — 「更高?」列的查表: 报出去的这个数，有多大概率还会被超过。

    python3 build_exceed_table.py --db cn.sqlite --out exceed_table.json

**为什么要按站分。** 2026-08-17 查今天北京/郑州/重庆七个时次全线报低时发现的:
原表只按 `时次|剩余升幅档` 分格，15 时低升幅那一格全站共用 4.4%。而各站
「15 时之后还会再升」的真实频率差一个数量级 ——

    成都 29% | 重庆 23% | 武汉 18% | 郑州 13% | 济南 9% | 广州 8%
    深圳 7% | 北京 6% | 青岛 3% | 上海 2%

上海被高估两倍、成都被低估七倍。物理上讲得通: 盆地站午后对流触发晚，
沿海站海风一到就封顶。夏季（6-9 月）各站还要再翻近一倍（成都到 45%）。

**偏差稳不稳 —— 这是能不能按站分的唯一判据。** 标准窗口对半切，
前半段各站概率 vs 后半段: **r = +0.970**。后半段 Brier 分数从 0.1078
（单一常数）降到 0.0996，**改善 7.6%**。是站点固有属性，不是噪声。

**这一列不改任何预报值**，只给「这个数能不能放心买」一个按站校准的刻度。

样本不足的格子逐级回退: 站×季×档 -> 站×档 -> 档（原来的全局值）。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import stations as _S                      # noqa: E402

EDGES = [0.15, 0.3, 0.5]                   # 剩余升幅分档，与原表一致
CUTS = [9, 10, 11, 12, 13, 14, 15]
WARM = {"06", "07", "08", "09"}            # 暖季: 对流活跃、见顶偏晚
MIN_N = 60                                 # 一格至少这么多站日才用它自己的值


def rows_for(csvs, cut: int):
    """某个截止时次的样本: (站, 日期, 剩余升幅档, 是否还会更高)。

    **分档必须用「模型判的剩余升幅」，不能用实际升幅。** 消费端
    `predict_nowcast._exceed(cutoff, rise)` 传进来的 rise 就是模型判的那个数；
    建表若拿实际升幅分档，档 0（升幅<0.15）里「跳了整数」几乎恒为假，整张表
    全是 0 —— 2026-08-16 首版就这么错的。所以数据源用回测序列
    （`bt_c*.csv` 的 `pred_raw - so_far`），与生产同一条路径产出。
    """
    out = []
    for p in csvs:
        if not os.path.exists(p):
            continue
        import csv as _csv
        for r in _csv.DictReader(open(p, encoding="utf-8")):
            if int(r["cutoff"]) != cut:
                continue
            rise = float(r["pred_raw"]) - float(r["so_far"])
            i = next((k for k, e in enumerate(EDGES) if rise < e), None)
            if i is None:                  # 升幅超过最后一档: 原表也不印
                continue
            up = round(float(r["actual"])) > round(float(r["so_far"]))
            out.append((r["station"], r["date"], i, up))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt", nargs="+", default=[
        "bt_c9.csv", "bt_c1011.csv", "bt_c12.csv", "bt_c13.csv",
        "bt_c14.csv", "bt_c15.csv"],
        help="backtest_nowcast --csv-out 的输出，各时次按其部署口径跑的那批")
    ap.add_argument("--out", default="exceed_table.json")
    a = ap.parse_args()

    cells: dict = {}                       # 档: "cut|i"        （原格式，回退用）
    per_stn: dict = {}                     # 站档: "cut|stn|i"
    per_sw: dict = {}                      # 站季档: "cut|stn|w|i"
    for cut in CUTS:
        rs = rows_for(a.bt, cut)
        agg = collections.defaultdict(lambda: [0, 0])
        for s, d, i, up in rs:
            w = 1 if d[5:7] in WARM else 0
            for key, tgt in ((f"{cut}|{i}", cells),
                             (f"{cut}|{s}|{i}", per_stn),
                             (f"{cut}|{s}|{w}|{i}", per_sw)):
                agg[(id(tgt), key)][0] += up
                agg[(id(tgt), key)][1] += 1
        for (tid, key), (u, n) in agg.items():
            tgt = {id(cells): cells, id(per_stn): per_stn, id(per_sw): per_sw}[tid]
            if n >= (1 if tgt is cells else MIN_N):
                tgt[key] = [round(u / n, 4), n]
        print(f"  {cut:>2} 时: {len(rs)} 样本", file=sys.stderr)

    tbl = {"edges": EDGES, "cells": cells, "per_stn": per_stn, "per_sw": per_sw,
           "warm_months": sorted(WARM), "min_n": MIN_N}
    json.dump(tbl, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n查表已存 {a.out}: 档 {len(cells)} 格 / 站档 {len(per_stn)} 格 / "
          f"站季档 {len(per_sw)} 格")

    print("\n  15 时低升幅档，按站（对比原来全站共用的一个值）:")
    base = cells.get("15|0")
    print(f"    {'站点':<16}{'全年':>8}{'暖季':>8}{'冷季':>8}{'原表':>8}")
    for s in sorted(_S.ICAOS):
        g = per_stn.get(f"15|{s}|0")
        w = per_sw.get(f"15|{s}|1|0")
        c = per_sw.get(f"15|{s}|0|0")
        f = lambda x: f"{x[0]:.0%}" if x else "--"
        print(f"    {s} {_S.NAMES.get(s, '')[:6]:<10}{f(g):>8}{f(w):>8}{f(c):>8}"
              f"{f(base):>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

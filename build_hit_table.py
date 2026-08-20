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

# 「晚见顶概率」分档（2026-08-19 加）。回测 CSV 的 peak_p 列，来自本时次自己的
# 见顶时刻判别器（train_nowcast.fit_peak_prob）。
#
# **为什么值得加一维**: 同样的剩余升幅，今天这个站是「早早定下来」还是
# 「拖到 16 点还在升」，把握完全不同。实测 15 时按 9 时判的晚见顶概率五等分，
# 命中率从 99.6% 一路到 45.7% —— 54 个百分点的跨度。
#
# 查表分格改善（13/14/15 合计，后半 7007 站日验证）:
#     + 9 时判的概率      +0.21%   几乎为零
#     + 本时次自己判的     +3.24%
# 三种切分 × 三个时次 = 9/9 全为正: 13 时 +1.6~2.0% / 14 时 +6.9~7.5% /
# 15 时 +0.6~1.7%。**必须用本时次自己的判别器。**
#
# 教训: 首测「9 时概率」得到 +6.62%，是假的 —— 概率取自已部署的判别器，
# 而它在 tr+va 全量上训过、包含用来验证的后半段。换交叉折后只剩 +0.21%。
# **边界按分位数从数据算，不写死。** 2026-08-19 首版写死 [0.10,0.22,0.38,0.58]，
# 结果各时次有 47~56% 的行挤在最低档里、档内差异被抹平 —— 独立复核只剩 +0.05%，
# 而按分位数分档验证时是 +3.24%。分档也**逐时次算** —— 14 时 peak_p 中位数
# 0.046、9 时 0.125，一套边界套不住。
PEAK_NQ = 5        # 分几档（按分位数）
# **只有这两个时次用三维。** 四种切分（前半→后半 / 后半→前半 / 奇→偶 / 偶→奇）
# 逐时次验证，只有 13/14 时四次全正:
#   13 时 +1.09/+1.65/+0.46/+2.42  平均 +1.4%
#   14 时 +2.34/+7.19/+5.28/+5.78  平均 +5.1%
#   15 时 +1.48/-0.01/+0.45/+0.58  有一次≈0，量级也小
#   9/11/12 时正负混杂，10 时四次全正但只有 +0.15~0.61%
# 其余时次走二维（cells3 里没有它们的格子，_exp_hit 自动回退）。
#
# **离线验证曾给出 +3.24%，是乐观的。** 那次判别器在半数数据（4 万多样本、
# 含全部历史）上一次训成；生产路径是逐块重训、只用有 NWP 的行（每块几千），
# 概率噪声大得多。用生产路径的 peak_p 复核只剩 +0.24%（全时次合计）。
PEAK_CUTOFFS = {13, 14}


def bucket(rise):
    for i, e in enumerate(EDGES):
        if rise <= e:
            return i
    return len(EDGES) - 1


def pbucket(p, edges):
    """晚见顶概率分档。没有 peak_p 或该时次没算出边界，返回 None -> 走二维。"""
    if p in (None, "") or not edges:
        return None
    p = float(p)
    for i, e in enumerate(edges):
        if p <= e:
            return i
    return len(edges)


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
    cells3 = collections.defaultdict(list)     # (cutoff, bucket, pbucket) -> [hit]
    prep = []                                  # 先攒着，边界要先按分位数算出来
    per_cut = collections.defaultdict(list)
    allhit = []
    for r in rows:
        c = int(r["cutoff"])
        pred, act = float(r["pred_mean"]), float(r["actual"])
        rise = float(r["pred_raw"]) - float(r["so_far"])
        hit = 1.0 if pred == act else 0.0
        b = bucket(rise)
        cells[(c, b)].append(hit)
        per_cut[c].append(hit)
        allhit.append(hit)
        prep.append((c, b, r.get("peak_p"), hit))

    g = sum(allhit) / len(allhit)
    table = {"edges": EDGES[:-1], "peak_edges": {},
             "global": round(g, 4), "cells": {}, "cells3": {}, "per_cutoff": {}}
    for c, v in sorted(per_cut.items()):
        table["per_cutoff"][str(c)] = round(sum(v) / len(v), 4)
    n_fallback = 0
    for (c, b), v in sorted(cells.items()):
        if len(v) >= MIN_N:
            table["cells"][f"{c}|{b}"] = [round(sum(v) / len(v), 4), len(v)]
        else:
            n_fallback += 1
    # 逐时次按分位数定边界，再分格
    pe = {}
    byc = collections.defaultdict(list)
    for c, b, p, h in prep:
        if p not in (None, ""):
            byc[c].append(float(p))
    for c, v in byc.items():
        if len(v) < 500 or c not in PEAK_CUTOFFS:
            continue
        v = sorted(v)
        e = [v[int(len(v) * k / PEAK_NQ)] for k in range(1, PEAK_NQ)]
        if len(set(e)) == len(e):              # 边界重复说明概率高度集中，放弃
            pe[c] = [round(x, 4) for x in e]
    table["peak_edges"] = {str(k): v for k, v in sorted(pe.items())}
    for c, b, p, h in prep:
        pb = pbucket(p, pe.get(c))
        if pb is not None:
            cells3[(c, b, pb)].append(h)
    n3 = 0
    for (c, b, pb), v in sorted(cells3.items()):
        if len(v) >= MIN_N:
            table["cells3"][f"{c}|{b}|{pb}"] = [round(sum(v) / len(v), 4), len(v)]
            n3 += 1
    json.dump(table, open(a.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"  三维格子（时次 × 升幅 × 晚见顶概率）: {n3} 个够样本"
          f"，其余自动回退到二维", file=sys.stderr)

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

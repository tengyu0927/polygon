#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""resid_feat.py — 把「模型自己最近几天报偏了多少」做成特征表。

    python3 resid_feat.py --out resid_feat.csv

**动机。** 2026-08-17 量到: 临近模型自己的签名残差**有时间持续性**，
而且越早的时次越强（前 5 天平均残差与今天残差的相关）:

    9 时 0.275 | 10 时 0.139 | 11 时 0.110 | 12 时 0.117 | 15 时 0.037

模型现在有「最近实际升幅相对气候的异常」(`rise_anom_3d/7d`)，但**没有
「我自己的残差」** —— 那是另一回事: 前者是天气漂移，后者是模型在当前天气型
下的失准。r=0.28 说明模型确实没把它吃掉。

**为什么要做成特征而不是直接减。** 直接减（pred - α·近期残差）实测:

    时次   前半选参数     后半基线 -> 订正后        P
    9 时  k=30 α=0.4   35.72% -> 36.67% +0.95pt  91.1%
    10 时 k=5  α=0.3   38.36% -> 38.88% +0.52pt  80.4%
    11 时                          -0.60pt
    12 时                          -0.35pt

差一口气，而且**它每天都按同一个比例订正**，学不会「什么时候这个偏差成立」。
分站各调各的更差（9 时 +0.6 vs 统一 +1.0）—— 每站只有 ~230 天，挑 k 和 α
两个参数挑到的是噪声。同一个教训在档位表、已见顶判别器上都出现过。
做成特征，模型才能跟云量/风速/剩余升幅一起决定今天调不调。

**无泄漏。** 数据源是 `backtest_nowcast --csv-out` 的滚动样本外序列
（每块只用块之前的数据训练），且 D 日的特征只取 D 之前的残差。生产端用
自己历史日志里的预报值算同一个量 —— 与 `mosd_*` 用 mos_rolling.py 是同一
个套路（见 mos_rolling.py 顶部对训练/预测口径必须一致的说明）。
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys

WINDOWS = (3, 7, 20)
FEAT_NAMES = [f"rs_bias_{k}d" for k in WINDOWS] + ["rs_sign_7d", "rs_n_7d"]

DEFAULT_BT = ["bt_c9.csv", "bt_c1011.csv", "bt_c12.csv", "bt_c13.csv",
              "bt_c14.csv", "bt_c15.csv"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bt", nargs="+", default=DEFAULT_BT,
                    help="各时次按其部署口径跑出来的回测序列")
    ap.add_argument("--out", default="resid_feat.csv")
    a = ap.parse_args()

    # (时次, 站) -> [(日期, 残差)]，残差 = 未取整预报 - 取整实况
    seq: dict = collections.defaultdict(list)
    for p in a.bt:
        if not os.path.exists(p):
            print(f"  [warn] 缺 {p}", file=sys.stderr)
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            seq[(int(r["cutoff"]), r["station"])].append(
                (r["date"], float(r["pred_raw"]) - round(float(r["actual"]))))

    rows = []
    for (cut, stn), v in seq.items():
        v.sort()
        for i in range(len(v)):
            d = v[i][0]
            f = {"station": stn, "date": d, "cutoff": cut}
            for k in WINDOWS:                       # **只取 i 之前**
                w = [e for _, e in v[max(0, i - k):i]]
                f[f"rs_bias_{k}d"] = (round(sum(w) / len(w), 4)
                                      if len(w) >= max(2, k // 2) else "")
            w7 = [e for _, e in v[max(0, i - 7):i]]
            # 同号比例: 偏差是稳定一边倒还是来回摆。0.5=完全无方向
            f["rs_sign_7d"] = (round(max(sum(1 for x in w7 if x > 0),
                                         sum(1 for x in w7 if x < 0)) / len(w7), 4)
                               if len(w7) >= 4 else "")
            f["rs_n_7d"] = len(w7)
            rows.append(f)

    rows.sort(key=lambda r: (r["date"], r["cutoff"], r["station"]))
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["station", "date", "cutoff"] + FEAT_NAMES)
        w.writeheader()
        w.writerows(rows)
    days = sorted({r["date"] for r in rows})
    print(f"残差特征表已存 {a.out}: {len(rows)} 行, {len(days)} 天, "
          f"{days[0] if days else '-'} ~ {days[-1] if days else '-'}")
    for cut in sorted({r["cutoff"] for r in rows}):
        n = sum(1 for r in rows if r["cutoff"] == cut and r["rs_bias_7d"] != "")
        print(f"  {cut:>2} 时: {n} 行有 7 天残差", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

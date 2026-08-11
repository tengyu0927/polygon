#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mos_rolling.py — 按月滚动重训 D+1 MOS，导出**无泄漏且与生产同口径**的预报序列。

    python3 mos_rolling.py mos_multi.csv --start 2025-05 --end 2026-08 --out mos_roll.csv

为什么需要这个:

临近模型想把 D+1 那条链路的**订正后**输出当特征（DEB 偏差订正 + ridge/GBM
融合的结果，`mosd_*` 三项）。2026-08-11 的 A/B 显示 9/10 时确实有增益
（ΔMAE −0.0186 / −0.0231，P=99.4% / 99.9%，三个时间窗口符号一致），
但那次训练用的是 `train_mos --pred` 的单次切分测试期预报 —— 模型只见过
2025-10-22 之前的数据，而**生产用的是每月重训的模型**。两端口径不同，
学到的「该调多少」在生产上就会偏。这是本项目最常犯的静默错配那一类
（`stn_id` 训练端设了预测端没设、XSTN 训练/预测值对不上，都是同一类病）。

这个脚本让训练端的序列按生产方式生成: 对每个月 M，只用 M 之前的数据训练，
预测整个 M。拼起来就是一条逐月滚动的样本外序列。

验证集: 训练段末尾 15% 的日期，与 train_mos 默认的 70/15/15 结构一致。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import train_mos as T                      # noqa: E402


def months(a: str, b: str):
    y, m = int(a[:4]), int(a[5:7])
    y2, m2 = int(b[:4]), int(b[5:7])
    while (y, m) <= (y2, m2):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="mos_multi.csv")
    ap.add_argument("--lead", type=int, default=1)
    ap.add_argument("--start", required=True, help="第一个预测月，如 2025-05")
    ap.add_argument("--end", required=True, help="最后一个预测月，如 2026-08")
    ap.add_argument("--out", default="mos_roll.csv")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.3, 1, 3, 10, 30, 100])
    ap.add_argument("--obs-pen", nargs="+", type=float, default=[1, 3, 10])
    ap.add_argument("--no-gbm", action="store_true")
    a = ap.parse_args()
    a.coef = False

    rows = T.load(a.csv)
    got = []
    for mo in months(a.start, a.end):
        nxt = f"{int(mo[:4]) + (mo[5:7] == '12'):04d}-{1 if mo[5:7] == '12' else int(mo[5:7]) + 1:02d}"
        # 训练段 = mo 之前全部；验证 = 训练段末 15% 的日期；测试 = 这个月
        hist = sorted({r["date"] for r in rows
                       if r["lead"] == a.lead and r["date"] < mo + "-01"})
        if len(hist) < 400:
            print(f"  {mo}: 历史仅 {len(hist)} 天，跳过", file=sys.stderr)
            continue
        c1 = hist[int(len(hist) * 0.85)]
        c2 = mo + "-01"
        sub = [r for r in rows if r["date"] < nxt + "-01"]
        res = T.run_lead(sub, a.lead, a, cuts=(c1, c2))
        if not res:
            continue
        n = 0
        for stn, d, pred, pr, obs in res["pred"]:
            if d[:7] != mo:
                continue
            got.append((a.lead, stn, d, pred, pr, obs))
            n += 1
        print(f"  {mo}: 训练 <{c1} | 验证 {c1}~ | 测试 {mo}  -> {n} 条",
              file=sys.stderr, flush=True)

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lead", "station", "date", "pred", "pred_round", "obs"])
        w.writerows(got)
    days = sorted({x[2] for x in got})
    print(f"\n滚动样本外序列已存 {a.out}: {len(got)} 行, "
          f"{len(days)} 天, {days[0] if days else '-'} ~ {days[-1] if days else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

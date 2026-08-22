#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""late_call.py — 16/17 时那两轮。**不跑模型，直接报截止此刻的最高温。**

    python3 late_call.py --cutoff 16
    python3 late_call.py --cutoff 17 --date 2026-08-20

为什么没有模型。标准窗口 4533 站日，「直接报已达」的完全命中率:

    时次    9    10    11    12    13    14    15    16    17    18
    命中  9.6% 12.6% 19.0% 31.2% 48.8% 69.0% 87.9% 96.6% 99.5% 99.9%

到 16 时，96.6% 的站日当天最高温已经出现了 —— 剩下的是「读温度计」，
不是预报。实测（82,524 站日、全部历史、前半训后半验、纯观测 28 项特征）:

    时次    报已达     训个模型   模型+已见顶覆盖   完美判别上界
    15 时  89.66%   89.01%      89.44%        89.65%
    16 时  97.45%   97.31%      97.37%        97.44%
    17 时  99.50%   99.50%      99.50%        99.50%

**16/17 时训模型比报已达更差**，而报已达已经等于完美判别的上界 —— 那时
「还会不会再升」这个问题几乎没有剩余不确定性可供预测。所以这里一行模型
代码都不该有。15 时保持现状（那里模型仍在 nowcast_late.json 里，且
89.01% < 89.66% 的差距在标准窗口内不显著，不动它）。

**为什么补这两轮。** 用户的目标是「十个站里至少九个对」。按日统计:

    时次          15 时    16 时    17 时
    >=9 站对      67.2%   95.6%   100.0%
    10 站全对     34.5%   72.9%    95.0%

流水线原来只跑到 15 时（cron `15 9-15`），16/17 这两轮从来没有过。
目标落在 16 时，不在模型里。

「更高?」列沿用 exceed_table.json；那张表按 15 时以下的时次建，
16/17 时没有格子，于是直接用这两个时次的实测频率（见 EXCEED）。
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import stations as _S                      # noqa: E402
import tablefmt as F                       # noqa: E402  显示宽度对齐（中文双宽）

CST = timezone(timedelta(hours=8))
HERE = os.path.dirname(os.path.abspath(__file__)) or "."

# 「报出去这个数还会被超过」的实测频率，按 (时次, 已降多少度) 分。
# 口径与 build_exceed_table 的 DROP_EDGES 一致: 没降 / 降 0.5-2 / 已降 >=2。
# 数字由 exceed_late.json 生成（build_late_exceed.py），这里只做兜底默认值。
EXCEED_FALLBACK = {16: 0.034, 17: 0.005}


def hourly(db: str, d: str):
    """{站: {小时: 温度}}，只取目标日。"""
    conn = sqlite3.connect(os.path.join(HERE, db))
    out: dict = {}
    for s, h, t in conn.execute(
            """SELECT station,
                 CAST(strftime('%H', datetime(obs_time_utc,'+8 hours')) AS INT),
                 MAX(temp_c)
               FROM obs WHERE local_date = ? AND temp_c IS NOT NULL
               GROUP BY 1, 2""", (d,)):
        out.setdefault(s, {})[h] = t
    conn.close()
    return out


def exceed(cut: int, drop: float) -> float:
    """还会更高的概率。有 exceed_late.json 就查表，否则用兜底常数。"""
    p = os.path.join(HERE, "exceed_late.json")
    if os.path.exists(p):
        import json
        tbl = json.load(open(p, encoding="utf-8"))
        di = 0 if drop < 0.5 else (1 if drop < 2.0 else 2)
        cell = tbl.get(f"{cut}|{di}") or tbl.get(str(cut))
        if cell:
            return cell[0] if isinstance(cell, list) else cell
    return EXCEED_FALLBACK.get(cut, 0.05)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, required=True, choices=(16, 17, 18))
    ap.add_argument("--date", default=None)
    ap.add_argument("--db", default="cn.sqlite")
    ap.add_argument("--stations", default=None, help="逗号分隔，默认全部")
    a = ap.parse_args()

    tgt = a.date or datetime.now(CST).strftime("%Y-%m-%d")
    want = ([s.strip() for s in a.stations.split(",")] if a.stations
            else sorted(_S.ICAOS))
    H = hourly(a.db, tgt)

    print(f"临近预报  目标日 {tgt}  截止 {a.cutoff:02d} 时（北京时）  直接报已达")
    # 列宽按**显示列**算（中文双宽），表头与数据行共用。见 tablefmt.py。
    # 这几个宽度与 predict_nowcast.py 主表同名列一致 —— 两张表要能上下对齐。
    W_STN, W_VAL, W_P90, W_RISE, W_UP, W_RDY, W_OBS = 16, 7, 8, 10, 7, 8, 9
    print("\n  " + F.L("站点", W_STN) + F.R("预报", W_VAL)
          + F.R("不排除", W_P90) + F.R("已达", W_VAL) + F.R("预计再升", W_RISE)
          + F.R("更高?", W_UP) + F.R("可下手", W_RDY) + F.R("实况", W_OBS)
          + "   备注")

    n_stale = 0
    for s in want:
        v = {h: t for h, t in (H.get(s) or {}).items() if h <= a.cutoff}
        if len(v) < 5:
            print("  " + F.L(f"{s} {_S.NAMES.get(s, '')}", W_STN)
                  + "".join(F.R("--", x) for x in
                            (W_VAL, W_P90, W_VAL, W_RISE, W_UP, W_RDY, W_OBS))
                  + "   实况不足")
            continue
        last = max(v)
        msf = max(v.values())
        pred = round(msf)
        p_up = exceed(a.cutoff, msf - v[last])
        # 「不排除」: 超过概率 >= 10% 才给高一档，否则与预报同值。
        # 规则与 predict_nowcast 的自洽约束一致 —— 见那里 906 行起的说明。
        hi = pred + 1 if p_up >= 0.10 else pred
        note = ""
        if last < a.cutoff - 1:
            note = f"实况滞后 {a.cutoff - last} 小时"
            n_stale += 1
        # 「可下手」与 predict_nowcast 同口径: 1-超越概率 >= 0.95 就标 ✓。
        # 16 时约八成站、17 时几乎全部站会到 ✓。
        rdy = 1.0 - p_up
        rc = F.R(f"✓ {rdy:.0%}" if rdy >= 0.95 else f"{rdy:.0%}", W_RDY)
        print("  " + F.L(f"{s} {_S.NAMES.get(s, '')}", W_STN)
              + F.R(pred, W_VAL) + F.R(hi, W_P90) + F.R(f"{msf:.0f}", W_VAL)
              + F.R("0.0", W_RISE) + F.R(f"{p_up:.0%}", W_UP) + rc
              + F.R(f"{v[last]:.0f}", W_OBS) + f"   {note}")

    print(f"\n  {a.cutoff} 时不跑模型 —— 那时「还会不会再升」已几乎没有不确定性，"
          f"报已达即为完美上界。")
    print(f"  实测完全命中: 16 时 96.6% / 17 时 99.5%（标准窗口 4533 站日）。")
    if n_stale:
        print(f"  [warn] {n_stale} 个站实况滞后，报出去的已达可能还没包含峰值。",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

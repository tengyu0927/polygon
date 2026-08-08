#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nowcast_potential.py — 用已有 30 年逐时实况，量化「知道上午实况后还剩多少不确定性」

在动手建模之前先回答: 截到 9 点 / 12 点 / 13 点时，当天日最高温还有多大的
剩余不确定性？这决定了临近预报的精度上限，也决定值不值得做。

    python3 nowcast_potential.py --db cn.sqlite
    python3 nowcast_potential.py --db cn.sqlite --cutoffs 9 11 12 13 14
    python3 nowcast_potential.py --db cn.sqlite --month 7      # 只看 7 月

最简预报器: Tmax_hat = 截止时刻的当日已达最高 + 该站该月的平均剩余升温
气候平均在训练年份上估，在测试年份上评估 —— 不在同一批数据上估完再减。

关键输出「已见顶比例」: 当日最高温在截止时刻之前就已出现的天数占比。
这类日子的「预报」其实是已知事实，会把准确率虚高，必须单独看。
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
CST = timezone(timedelta(hours=8))
import stations as _S  # 站点清单唯一真相源
NAMES = _S.NAMES


def load_days(db, table="obs", min_peak=6):
    """按 (站, 北京时日期) 聚合逐时观测。返回 {key: {hour: temp}}。"""
    conn = sqlite3.connect(db)
    cols = {c[1] for c in conn.execute(f"PRAGMA table_info({table})")}
    tcol = "valid_time_gmt" if "valid_time_gmt" in cols else "obs_time_utc"
    days = defaultdict(dict)
    q = f"SELECT station, {tcol}, temp_c FROM {table} WHERE temp_c IS NOT NULL"
    for stn, ts, v in conn.execute(q):
        try:
            if tcol == "valid_time_gmt":
                dt = datetime.fromtimestamp(int(ts), UTC).astimezone(CST)
            else:
                s = str(ts).replace("Z", "+00:00").replace(" ", "T")
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                dt = dt.astimezone(CST)
        except (ValueError, OSError, TypeError):
            continue
        d = days[(stn, dt.strftime("%Y-%m-%d"))]
        h = dt.hour
        # 同一小时多条(半点报)取较高值，与日最高温口径一致
        if h not in d or v > d[h]:
            d[h] = float(v)
    conn.close()
    # 午后时段缺测过多的日子丢弃，否则「真值」本身不可信
    return {k: v for k, v in days.items()
            if sum(1 for h in v if 10 <= h <= 19) >= min_peak}


def stats(e):
    n = len(e)
    if not n:
        return None
    me = sum(e) / n
    return {"n": n, "me": me, "mae": sum(abs(x) for x in e) / n,
            "rmse": math.sqrt(sum(x * x for x in e) / n),
            "p05": 100 * sum(1 for x in e if abs(x) <= 0.5) / n,
            "p1": 100 * sum(1 for x in e if abs(x) <= 1) / n,
            "p2": 100 * sum(1 for x in e if abs(x) <= 2) / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cn.sqlite")
    ap.add_argument("--table", default="obs")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[9, 12, 13])
    ap.add_argument("--split-year", type=int, default=2024,
                    help="该年起为测试期，之前为训练期（估气候平均）")
    ap.add_argument("--month", type=int, default=0, help="只看某月，0=全年")
    args = ap.parse_args()

    print("读取逐时实况…", file=sys.stderr)
    days = load_days(args.db, args.table)
    print(f"  {len(days)} 个可用站日", file=sys.stderr)
    if not days:
        return 1

    for cut in args.cutoffs:
        recs = []
        for (stn, d), hrs in days.items():
            if args.month and int(d[5:7]) != args.month:
                continue
            before = {h: t for h, t in hrs.items() if h <= cut}
            # 要求截止前有足够观测，且最后一条离截止时刻不超过 2 小时
            if len(before) < 5 or max(before) < cut - 2:
                continue
            so_far = max(before.values())
            tmax = max(hrs.values())
            recs.append({"stn": stn, "date": d, "month": int(d[5:7]),
                         "year": int(d[:4]), "so_far": so_far, "tmax": tmax,
                         "rise": tmax - so_far,
                         "peaked": tmax <= so_far + 1e-9})
        if not recs:
            print(f"\n{cut} 时截止: 无可用样本")
            continue

        tr = [r for r in recs if r["year"] < args.split_year]
        te = [r for r in recs if r["year"] >= args.split_year]
        clim = defaultdict(list)
        for r in tr:
            clim[(r["stn"], r["month"])].append(r["rise"])
        cm = {k: sum(v) / len(v) for k, v in clim.items() if len(v) >= 30}

        print(f"\n{'='*78}")
        print(f"截止 {cut:02d} 时（北京时）  训练 {len(tr)} 站日 / 测试 {len(te)} 站日")

        ev = [r for r in te if (r["stn"], r["month"]) in cm]
        e_all = [r["so_far"] + cm[(r["stn"], r["month"])] - r["tmax"] for r in ev]
        s = stats(e_all)
        print(f"\n  最简预报器 = 已达最高 + 该站该月平均剩余升温")
        print(f"    n={s['n']}  MAE={s['mae']:.2f}  RMSE={s['rmse']:.2f}  "
              f"ME={s['me']:+.2f}  ±0.5℃={s['p05']:.0f}%  ±1℃={s['p1']:.0f}%  "
              f"±2℃={s['p2']:.0f}%")

        pk = [r for r in ev if r["peaked"]]
        npk = [r for r in ev if not r["peaked"]]
        print(f"\n  已见顶比例 {100*len(pk)/len(ev):.0f}%  "
              f"（这些日子最高温已发生，「预报」实为已知事实，会虚高指标）")
        if npk:
            e2 = [r["so_far"] + cm[(r["stn"], r["month"])] - r["tmax"] for r in npk]
            s2 = stats(e2)
            print(f"    仅未见顶的日子: n={s2['n']}  MAE={s2['mae']:.2f}  "
                  f"±1℃={s2['p1']:.0f}%   ← 这才是真正要预报的部分")

        rises = sorted(r["rise"] for r in ev)
        q = lambda p: rises[int(p * (len(rises) - 1))]
        print(f"\n  剩余升温分布: 中位 {q(.5):.1f}℃  "
              f"10%~90% 分位 {q(.1):.1f} ~ {q(.9):.1f}℃  最大 {rises[-1]:.1f}℃")
        print(f"    分布越宽，说明上午实况留下的不确定性越大，模型可发挥空间也越大")

        print(f"\n  分站（测试期，仅未见顶的日子）")
        print(f"    {'站点':<14}{'n':>6}{'MAE':>7}{'±1℃':>7}{'平均剩余':>9}{'剩余标准差':>11}")
        for stn in sorted({r["stn"] for r in ev}):
            sub = [r for r in npk if r["stn"] == stn]
            if len(sub) < 20:
                continue
            e3 = [r["so_far"] + cm[(r["stn"], r["month"])] - r["tmax"] for r in sub]
            s3 = stats(e3)
            rs = [r["rise"] for r in sub]
            mu = sum(rs) / len(rs)
            sd = math.sqrt(sum((x - mu) ** 2 for x in rs) / len(rs))
            print(f"    {stn} {NAMES.get(stn,''):<9}{s3['n']:>6}{s3['mae']:>7.2f}"
                  f"{s3['p1']:>6.0f}%{mu:>9.1f}{sd:>11.1f}")

    print(f"\n{'='*78}")
    print("对照: D+1 模型（24h 时效）取整后 MAE 1.15；TAF 11 时轮 D+0（4h 时效）MAE 1.07")
    print("这里的最简预报器只用了「已达最高 + 气候平均」，没用任何当天的云、露点、")
    print("风、趋势信息，也没用模式对午后的预报。真模型应明显优于它 —— 上面的")
    print("数字是下限而非上限。「剩余标准差」那一列才是不确定性的量级参考。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""peak_hint_track.py — 逐日记账「见顶时刻」两条提示说得准不准

    python3 peak_hint_track.py --log             # 当天收盘后记一天（run_daily 调）
    python3 peak_hint_track.py --log --date 2026-08-22
    python3 peak_hint_track.py --report          # 攒够了再看

为什么要这个（2026-08-24）:

`predict_nowcast` 的见顶风险表印两条提示，用户会照着做决策:

    ⚠ 16 点后见顶风险偏高      校准后 P(>=16 时见顶) >= PEAK_WARN_TH
    基本能早早定下来            P(<13 时见顶) >= PEAK_EARLY_TH

两条都有回测数字撑着（⚠ 精度 25~30%、「早早」83~96%），但**回测是在
标准窗口上做的，生产分布未必一样**。头两天的实测就对不上:

    2026-08-22  「早早」标 11 次对 1 次（旧门槛 0.5）；新门槛 0.8 下 5 次对 0 次
    2026-08-23  「早早」标 13 次对 7 次 = 53.8%

样本还小（n=18），但都远低于回测给的 83~96%。**这类偏差只有攒日子才能判**，
而攒日子这件事必须是机器做的 —— 靠人记，两周后一定记不全。

判据（攒够 >= 200 次标注再下结论，约两周）:
  - 「早早」实测精度若稳定低于 cal_e 表给的值 10pt 以上，说明那张校准表
    在生产分布上不成立，回去查 fit_peak_prob 的训练窗口与生产是否同分布
  - ⚠ 实测精度若稳定低于 20%，PEAK_WARN_TH 该往上提

**只记账，不改任何预报值、不参与任何决策。** 失败一律不影响预报（run_daily
里是 `|| true`）。
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

CST = timezone(timedelta(hours=8))
HERE = os.path.dirname(os.path.abspath(__file__)) or "."
sys.path.insert(0, HERE)
import stations as _S                            # noqa: E402
import train_nowcast as N                        # noqa: E402

DDL = """CREATE TABLE IF NOT EXISTS hint (
  target_date TEXT, cutoff INT, station TEXT, kind TEXT,
  prob REAL, peak_h INT, correct INT,
  PRIMARY KEY (target_date, cutoff, station, kind));"""

# 见顶风险表的一行。宽度对齐前后都能匹配（一律走 \\s+，不依赖列宽）
# 百分比列数变过（2026-08-26 加了「该站常年」，从 4 个变 5 个），所以
# **不写死列数** —— 抓「ICAO + 站名 + 一串百分比 + N倍 + 提示」，百分比按
# 出现顺序取: [0]=早<13、[-2]=会改整数（原「校准后」）、[-1]=该站常年。
ROW = re.compile(r"^\s+(Z[A-Z]{3})\s+\S+\s+((?:\d+%\s+)+)[\d.]+倍\s*(.*)$")


def peaks(db: str, d: str):
    """{站: 见顶小时}。口径与 train_nowcast.make_samples 一致 = **首达**。

    当天观测没走完峰值时段就返回空 —— 拿截断值算见顶时刻是错的
    （2026-08-17 那次整张错表就是这么来的）。
    """
    conn = sqlite3.connect(os.path.join(HERE, db))
    hh = collections.defaultdict(dict)
    for s, h, t in conn.execute(
            """SELECT station,
                 CAST(strftime('%H', datetime(obs_time_utc,'+8 hours')) AS INT),
                 temp_c FROM obs WHERE local_date = ? AND temp_c IS NOT NULL""",
            (d,)):
        if h not in hh[s] or t > hh[s][h]:
            hh[s][h] = t
    conn.close()
    out = {}
    for s, v in hh.items():
        if not v or max(v) < N.PEAK_H1:
            continue                              # 当天还没过完，不记
        mx = max(v.values())
        out[s] = min(h for h, t in v.items() if t == mx)
    return out


def parse(d: str):
    """[(时次, 站, kind, 概率)]。kind: warn=⚠ / early=基本能早早定下来"""
    p = os.path.join(HERE, f"pred_{d}.log")
    if not os.path.exists(p):
        return []
    out, cut, inb = [], None, False
    for ln in open(p, encoding="utf-8", errors="replace"):
        m = re.search(r"(\d{1,2}) 时起报", ln)
        if m and "#####" in ln:
            cut, inb = int(m.group(1)), False
            continue
        # 标题在 2026-08-26 从「16 点后见顶的风险」改成「16 点后还在涨、
        # 会改掉整数的风险」（换了预测目标）。**两种都认** —— 历史日志要
        # 还能回填，而且以后再改标题也不该让记账静默断掉。
        if "16 点后见顶的风险" in ln or "16 点后还在涨" in ln:
            inb = True
            continue
        if inb and (ln.startswith("  分档") or ln.startswith("──")
                    or not ln.strip()):
            inb = False
            continue
        if not inb or cut is None:
            continue
        r = ROW.match(ln.rstrip())
        if not r:
            continue
        stn, pcts, hint = r.groups()
        nums = [int(x) for x in re.findall(r"(\d+)%", pcts)]
        if len(nums) < 4:
            continue
        early, cal = nums[0], nums[-2]
        if "⚠" in hint:          # 提示语从「见顶风险偏高」改成「整数可能改」，
                                     # 用 ⚠ 符号判，不依赖文案
            out.append((cut, stn, "warn", int(cal) / 100))
        elif "早早" in hint:
            out.append((cut, stn, "early", int(early) / 100))
    return out


def log(a) -> int:
    d = a.date or datetime.now(CST).strftime("%Y-%m-%d")
    pk = peaks(a.obs_db, d)
    if not pk:
        print(f"[hint] {d} 峰值时段还没走完（或没观测），本次不记", file=sys.stderr)
        return 0
    rows = parse(d)
    if not rows:
        print(f"[hint] {d} 日志里没有见顶风险表", file=sys.stderr)
        return 0
    conn = sqlite3.connect(os.path.join(HERE, a.db))
    conn.executescript(DDL)
    n = 0
    for cut, stn, kind, prob in rows:
        if stn not in pk:
            continue
        h = pk[stn]
        ok = (h >= 16) if kind == "warn" else (h < 13)
        conn.execute("INSERT OR REPLACE INTO hint VALUES (?,?,?,?,?,?,?)",
                     (d, cut, stn, kind, prob, h, int(ok)))
        n += 1
    conn.commit()
    w = [r for r in rows if r[2] == "warn" and r[1] in pk]
    e = [r for r in rows if r[2] == "early" and r[1] in pk]
    f = lambda L, ok: (f"{sum(1 for c, s, _, _ in L if ok(pk[s]))}/{len(L)}"
                       if L else "0/0")
    print(f"[hint] {d} 记 {n} 条  ⚠ {f(w, lambda h: h >= 16)}"
          f"  早早 {f(e, lambda h: h < 13)}", file=sys.stderr)
    conn.close()
    return 0


def report(a) -> int:
    p = os.path.join(HERE, a.db)
    if not os.path.exists(p):
        print("还没有记账库，先让 run_daily.sh 跑几天。", file=sys.stderr)
        return 1
    conn = sqlite3.connect(p)
    rows = list(conn.execute(
        "SELECT target_date, cutoff, station, kind, prob, correct FROM hint"))
    if not rows:
        print("库里还没有数据。", file=sys.stderr)
        return 1
    days = sorted({r[0] for r in rows})
    print(f"\n  见顶提示记账  {days[0]} ~ {days[-1]}（{len(days)} 天，"
          f"{len(rows)} 次标注）")
    if len(rows) < 200:
        print(f"  ⚠ 样本不足 200 次，**先别下结论**（回测口径需约两周）")

    for kind, lab, base in (("warn", "⚠ 16 点后见顶风险偏高", "回测 25~30%"),
                            ("early", "基本能早早定下来", "回测 83~96%")):
        sub = [r for r in rows if r[3] == kind]
        if not sub:
            continue
        n = len(sub)
        hit = sum(r[5] for r in sub)
        print(f"\n  {lab}   实测 {hit}/{n} = {100*hit/n:.1f}%   （{base}）")
        print(f"    {'轮次':<6}{'标了':>6}{'对':>5}{'精度':>8}{'均值把握':>10}")
        for cut in sorted({r[1] for r in sub}):
            g = [r for r in sub if r[1] == cut]
            h = sum(r[5] for r in g)
            print(f"    {cut:>2}时{'':<2}{len(g):>6}{h:>5}{100*h/len(g):>7.1f}%"
                  f"{sum(r[4] for r in g)/len(g):>9.0%}")
        # 标定: 印出去的把握 vs 实测。差得多说明校准表在生产分布上不成立
        print(f"    {'把握分档':<10}{'n':>6}{'实测':>8}{'偏差':>8}")
        for lo, hi in ((0, .3), (.3, .5), (.5, .8), (.8, .95), (.95, 1.01)):
            g = [r for r in sub if lo <= r[4] < hi]
            if len(g) < 5:
                continue
            pm = sum(r[4] for r in g) / len(g)
            ac = sum(r[5] for r in g) / len(g)
            print(f"    {lo:.0%}-{hi:.0%}{'':<4}{len(g):>6}{ac:>7.0%}"
                  f"{ac-pm:>+8.0%}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="peak_hint.sqlite")
    ap.add_argument("--obs-db", default="cn.sqlite")
    ap.add_argument("--date", default=None)
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report(a)
    if a.log:
        return log(a)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

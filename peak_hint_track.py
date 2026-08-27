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

**判据按「独立站日」，不按「标注行」**（2026-08-27 改）。同一个站同一天会被
7 个时次各标一次，但它只有一个结果 —— 按行算等于把一个事件重复计 4~5 次。
实测: ⚠ 92 行 = 20 个独立站日，行口径 33% / 站日口径 25%。旧的「攒够 200 次
标注」= 43 个站日，远不够下任何结论，却读起来像快到了。

攒多少才够（双侧 alpha=.05, power=.8，每天约 0.64 个 ⚠ 站日）:
    要分辨的差            站日/组     ≈ 天
    25% vs 40%（现状 vs 承诺）  152     237
    +10pt                    375     586
    +5pt                   1,469   2,295
    +2pt（换目标那次的量级）   9,031  14,110
**+2pt 级的改动在实盘上永远验证不了** —— 别拿回测的 +2pt 当上线理由，
也别指望记账器能事后确认它。只有 >=10pt 的改动才有希望在一年内看出来。

结论怎么下:
  - 「早早」实测精度若稳定低于 cal_e 表给的值 10pt 以上，说明那张校准表
    在生产分布上不成立，回去查 fit_peak_prob 的训练窗口与生产是否同分布
  - ⚠ 实测精度若稳定低于 20%，PEAK_WARN_TH 该往上提
  - 任何一条提示，若「对」的站日有一半以上来自同一个站，它多半在复读
    那个站的气候而不是在预测 —— 报告里会自动标出来

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


def _migrate(conn):
    """2026-08-27 加 y_true = 「整数真的改了没有」。

    ⚠ 的训练目标在 2026-08-26（47abe27）从「首达 >= 16 时」换成了「16 时后
    还在涨、把报出去的整数改掉了」，**记账器当时漏改**，一直拿旧标签给新
    模型打分。两者差 17.3%（92703 站日实测），只是头 84 条恰好没分歧。
    """
    cols = {r[1] for r in conn.execute("PRAGMA table_info(hint)")}
    if "y_true" not in cols:
        conn.execute("ALTER TABLE hint ADD COLUMN y_true INT")


def _wilson(k, n, z=1.96):
    """比例的 Wilson 区间。n 只有二十几时正态近似会给出负下界，用它。"""
    if not n:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / d
    return max(0.0, c - h), min(1.0, c + h)

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


def int_changed(db: str, d: str):
    """{站: 0/1} —— 「16 时之后还在涨、把报出去的整数改掉了」。

    **口径与 train_nowcast.fit_peak_prob2 的训练标签逐字一致**:
        int(round(max(全天)) > round(max(h <= 15)))
    这是 ⚠ 现在真正在预测的东西。用 peak_h >= 16 打分是旧标签，会把
    「见顶确实晚、但只多涨不到半度」记成 ⚠ 说对了 —— 那种日子对整数
    盘口毫无影响，本来就不该报警。

    与 peaks() 同样的守门: 当天没走完峰值时段就不给，拿截断值算是错的。
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
        pre = [t for h, t in v.items() if h <= 15]
        if not v or max(v) < N.PEAK_H1 or not pre:
            continue
        out[s] = int(round(max(v.values())) > round(max(pre)))
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
        # **必须连 `──` 一起判。** 只看关键词会把数据行当成标题:
        # 2026-08-26 起 ⚠ 的提示文案是「⚠ 16 点后还在涨，整数可能改」，
        # 与段落标题「── 16 点后还在涨、会改掉整数的风险」撞词，于是**每一条
        # ⚠ 行都被当成标题跳过** —— 从那天起记账器一条 ⚠ 都没记上，而且不
        # 报错。2026-08-27 查出来时已经漏了两天。
        # 标题一定以 `──` 开头，数据行一定以空格 + ICAO 开头，用这个分。
        if ln.startswith("──") and ("16 点后见顶的风险" in ln
                                    or "16 点后还在涨" in ln):
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
    ic = int_changed(a.obs_db, d)
    conn = sqlite3.connect(os.path.join(HERE, a.db))
    conn.executescript(DDL)
    _migrate(conn)
    n = 0
    for cut, stn, kind, prob in rows:
        if stn not in pk:
            continue
        h, y = pk[stn], ic.get(stn)
        # ⚠ 判「整数真的改了没有」（= 它的训练目标），不是「首达 >= 16」。
        # 「早早」的目标没变过，仍判首达 < 13。
        ok = (y == 1) if kind == "warn" else (h < 13)
        conn.execute(
            "INSERT OR REPLACE INTO hint"
            " (target_date,cutoff,station,kind,prob,peak_h,correct,y_true)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (d, cut, stn, kind, prob, h, int(ok), y))
        n += 1
    conn.commit()
    # 印站日数，不印行数 —— 行数会把一个事件重复计 4~5 次
    w = {s for c, s, k, _ in rows if k == "warn" and s in pk}
    e = {s for c, s, k, _ in rows if k == "early" and s in pk}
    f = lambda L, ok: f"{sum(1 for s in L if ok(s))}/{len(L)}" if L else "0/0"
    print(f"[hint] {d} 记 {n} 行  ⚠ {f(w, lambda s: ic.get(s) == 1)} 站日"
          f"  早早 {f(e, lambda s: pk[s] < 13)} 站日", file=sys.stderr)
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

    # **一个站一天 = 一个事件。** 它会被 7 个时次各标一次，但只有一个结果，
    # 所以精度、n、区间一律按站日算。行口径只保留在下面两张诊断表里
    # （「哪个时次标得准」「印出去的把握准不准」），那两个问题本来就是逐行的。
    sd = collections.defaultdict(list)
    for d, cut, stn, kind, prob, ok in rows:
        sd[(kind, d, stn)].append(ok)

    print(f"\n  见顶提示记账  {days[0]} ~ {days[-1]}（{len(days)} 天，"
          f"{len(rows)} 行 = {len(sd)} 个独立站日）")

    for kind, lab, base in (("warn", "⚠ 16 点后还在涨、会改整数", "回测 34~40%"),
                            ("early", "基本能早早定下来", "回测 83~96%")):
        g = {k: v for k, v in sd.items() if k[0] == kind}
        if not g:
            continue
        sub = [r for r in rows if r[3] == kind]
        n = len(g)
        hit = sum(1 for v in g.values() if max(v))
        lo, hi = _wilson(hit, n)
        print(f"\n  {lab}   实测 {hit}/{n} 站日 = {100 * hit / n:.0f}%"
              f"   95%CI [{100 * lo:.0f}%, {100 * hi:.0f}%]   （{base}）")
        print(f"    （{len(sub)} 个标注行 —— 每个事件重复计 "
              f"{len(sub) / n:.1f} 次，别拿行数当样本量）")

        # 分站。一条提示若「对」的站日大半来自同一个站，它在复读那站的气候。
        bys = collections.defaultdict(lambda: [0, 0])
        for (_, d, stn), v in g.items():
            bys[stn][0] += 1
            bys[stn][1] += int(bool(max(v)))
        print(f"    {'站':<10}{'标了':>6}{'对':>5}{'精度':>8}")
        for stn, (c, h) in sorted(bys.items(), key=lambda x: (-x[1][1], -x[1][0])):
            print(f"    {_S.NAMES.get(stn, stn):<10}{c:>6}{h:>5}{100 * h / c:>7.0f}%")
        if hit:
            top, (tc, th) = max(bys.items(), key=lambda x: x[1][1])
            if th * 2 >= hit and n - tc > 0:
                rn, rh = n - tc, hit - th
                print(f"    ⚠ 对的里面 {th}/{hit} 是{_S.NAMES.get(top, top)}。"
                      f"去掉它 {rh}/{rn} = {100 * rh / rn:.0f}% —— "
                      f"这条提示的价值几乎全押在一个站上")

        # 以下两张按行 —— 问的是逐行的问题，不是「这个事件判对没有」
        print(f"    {'轮次':<6}{'标了行':>7}{'对':>5}{'精度':>8}{'均值把握':>10}"
              f"   （行口径）")
        for cut in sorted({r[1] for r in sub}):
            gg = [r for r in sub if r[1] == cut]
            h = sum(r[5] for r in gg)
            print(f"    {cut:>2}时{'':<2}{len(gg):>7}{h:>5}{100 * h / len(gg):>7.1f}%"
                  f"{sum(r[4] for r in gg) / len(gg):>9.0%}")
        print(f"    {'把握分档':<10}{'n行':>6}{'实测':>8}{'偏差':>8}")
        for lo_, hi_ in ((0, .3), (.3, .5), (.5, .8), (.8, .95), (.95, 1.01)):
            gg = [r for r in sub if lo_ <= r[4] < hi_]
            if len(gg) < 5:
                continue
            pm = sum(r[4] for r in gg) / len(gg)
            ac = sum(r[5] for r in gg) / len(gg)
            print(f"    {lo_:.0%}-{hi_:.0%}{'':<4}{len(gg):>6}{ac:>7.0%}"
                  f"{ac - pm:>+8.0%}")

    # 够不够下结论。**按 ⚠ 站日算**，因为它是稀缺的那一条。
    wk = [k for k in sd if k[0] == "warn"]
    nw = len(wk)
    if nw and nw < 152:
        hw = sum(1 for k in wk if max(sd[k]))
        # 攒速按**记账库自己观测到的**站日/天算，不按设计标注率 ——
        # 两者对不上（8% 设计是 0.64 个/天，实测远高），用设计值会低估天数
        rate = nw / len(days)          # 分母是记账的总天数，不是「有 ⚠ 的天数」
        print(f"\n  ⚠ 只有 {nw} 个独立 ⚠ 站日（{rate:.1f} 个/天，"
              f"还需约 {(152 - nw) / rate:.0f} 天）。**先别下结论。**")
        print(f"    要把「现状 {100 * hw / nw:.0f}%」与「承诺 40%」分开需 ~152 站日；"
              f"分辨 ±2pt 的改动需 ~9000 站日（≈{9000 / rate / 365:.0f} 年）——")
        print(f"    也就是说**回测里 +2pt 的改动，实盘上永远验证不了**。"
              f"别拿它当上线理由。")
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify.py — 每天把「预报 vs 实况」按站累计，找出站级系统性偏差

    python3 verify.py                    # 解析日志、回填实况、入库（幂等）
    python3 verify.py --report           # 看分站分档的偏差
    python3 verify.py --report --days 14 # 只看最近 14 天

为什么需要: 通用判读规则（模式离散、预计再升等）在三天真实样本上表现参差，
但**站级偏差稳定得多** —— 重庆连续三天被报低 2-3 度、青岛连续偏高 1-2 度。
这类偏差人工记不住也记不准，交给脚本每天累计。

攒够两周后，如果某站某档的平均偏差稳定偏离 0（且置信区间不含 0），
就可以考虑在输出层加分站常数订正，或者把它作为重训的信号。

数据来源:
  临近  — cron_hourly.log / pred_YYYY-MM-DD.log 里的逐档预报
  D+1/2 — cron_daily.log / pred_mos_YYYY-MM-DD.log
  实况  — cn.sqlite 的 obs 表（按北京时日界取当日最高）
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

CST = timezone(timedelta(hours=8))
NAMES = {"ZBAA": "北京首都", "ZSPD": "上海浦东", "ZGGG": "广州白云",
         "ZGSZ": "深圳宝安", "ZUUU": "成都双流", "ZUCK": "重庆江北",
         "ZHHH": "武汉天河", "ZSQD": "青岛胶东"}

DDL = """
CREATE TABLE IF NOT EXISTS fc (
    station TEXT, target_date TEXT,
    kind TEXT,            -- 'nowcast' 或 'mos'
    slot INTEGER,         -- 临近=截止时刻; MOS=lead
    pred INTEGER, p90 INTEGER, so_far REAL,
    spread REAL, clim REAL, rise REAL,
    obs REAL,
    PRIMARY KEY (station, target_date, kind, slot));
"""


def parse_nowcast(paths):
    """临近日志。同一 (站,日,档) 出现多次时取最后一次（补跑覆盖原轮）。"""
    out = {}
    for p in paths:
        txt = open(p, encoding="utf-8", errors="replace").read()
        # 块头行尾也有 10 个 #，所以必须行首锚定，否则块内容会被截断
        for m in re.finditer(
                r'^#{10} (\d{4}-\d{2}-\d{2}) \S+\s+(\d+) 时起报.*?$(.*?)(?=^#{10}|\Z)',
                txt, re.S | re.M):
            day, slot, body = m.group(1), int(m.group(2)), m.group(3)
            for r in re.finditer(
                    r'^  (Z\w{3})\s+\S+\s+(\d+)\s+(\d+)\s+(\d+)\s+([-+][\d.]+)\s*(.*)$',
                    body, re.M):
                note = r.group(6)
                g = lambda pat: (float(re.search(pat, note).group(1))
                                 if re.search(pat, note) else None)
                out[(r.group(1), day, "nowcast", slot)] = dict(
                    pred=int(r.group(2)), p90=int(r.group(3)),
                    so_far=float(r.group(4)), rise=float(r.group(5)),
                    spread=g(r'模式离散 ([\d.]+)'), clim=g(r'气候升温 ([\d.]+)'))
    return out


def parse_mos(paths):
    out = {}
    for p in paths:
        txt = open(p, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r'目标日 (\d{4}-\d{2}-\d{2})（北京时）(.*?)(?=目标日 |\Z)',
                             txt, re.S):
            day, seg = m.group(1), m.group(2)
            for mm in re.finditer(r'── D\+(\d)（.*?\n(.*?)(?=── D\+|\Z)', seg, re.S):
                lead, body = int(mm.group(1)), mm.group(2)
                for r in re.finditer(r'^  (Z\w{3})\s+\S+\s+(\d+)\s+([\d.]+)',
                                     body, re.M):
                    out[(r.group(1), day, "mos", lead)] = dict(
                        pred=int(r.group(2)), p90=None, so_far=None,
                        rise=None, spread=None, clim=None)
    return out


def cmd_collect(args):
    files_n = sorted(glob.glob("cron_hourly.log") + glob.glob("pred_2*.log"))
    files_m = sorted(glob.glob("cron_daily.log") + glob.glob("pred_mos_2*.log"))
    recs = {}
    recs.update(parse_nowcast(files_n))
    recs.update(parse_mos(files_m))
    if not recs:
        print("没解析到任何预报。确认日志文件存在。", file=sys.stderr)
        return 1

    oc = sqlite3.connect(args.obs_db)
    obs = {(s, d): v for s, d, v in oc.execute(
        "SELECT station, local_date, MAX(temp_c) FROM obs GROUP BY 1, 2")}
    # 只回填「已经过完」的日子，当天还在走的不算
    today = datetime.now(CST).date().isoformat()
    oc.close()

    conn = sqlite3.connect(args.db)
    conn.executescript(DDL)
    n = pend = 0
    for (stn, day, kind, slot), v in recs.items():
        o = obs.get((stn, day)) if day < today else None
        if o is None:
            pend += 1
        conn.execute(
            "INSERT OR REPLACE INTO fc VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (stn, day, kind, slot, v["pred"], v["p90"], v["so_far"],
             v["spread"], v["clim"], v["rise"], o))
        n += 1
    conn.commit()
    tot = conn.execute("SELECT COUNT(*), SUM(obs IS NOT NULL) FROM fc").fetchone()
    conn.close()
    print(f"入库 {n} 条（其中 {pend} 条当日未结束、暂无实况）")
    print(f"累计 {tot[0]} 条，已配实况 {tot[1]} 条 -> {args.db}")
    return 0


def cmd_report(args):
    if not os.path.exists(args.db):
        print(f"还没有 {args.db}，先跑一次 verify.py", file=sys.stderr)
        return 1
    conn = sqlite3.connect(args.db)
    conn.executescript(DDL)
    cut = (datetime.now(CST).date() - timedelta(days=args.days)).isoformat()
    rows = list(conn.execute(
        "SELECT station, kind, slot, pred, obs FROM fc "
        "WHERE obs IS NOT NULL AND target_date >= ?", (cut,)))
    days = [r[0] for r in conn.execute(
        "SELECT DISTINCT target_date FROM fc WHERE obs IS NOT NULL "
        "AND target_date >= ? ORDER BY 1", (cut,))]
    conn.close()
    if not rows:
        print("还没有配上实况的样本。当天的要等第二天才会回填。")
        return 0
    print(f"验证期 {days[0]} ~ {days[-1]}（{len(days)} 天），共 {len(rows)} 条\n")

    import collections
    g = collections.defaultdict(list)
    for stn, kind, slot, pred, obs in rows:
        g[(stn, kind, slot)].append(pred - obs)

    for kind, label in (("nowcast", "临近（按截止时刻）"), ("mos", "MOS（按时效）")):
        slots = sorted({k[2] for k in g if k[1] == kind})
        if not slots:
            continue
        print(f"── {label}   表内为平均偏差 ME（正=偏高），括号内 MAE")
        print(f"  {'站点':<14}" + "".join(f"{('D+' if kind=='mos' else '')}{s}{'' if kind=='mos' else '时':>0}"
                                          .rjust(13) for s in slots))
        for stn in sorted(NAMES):
            line = f"  {stn} {NAMES[stn]:<9}"
            for s in slots:
                e = g.get((stn, kind, s))
                if not e:
                    line += f"{'--':>13}"
                    continue
                me = sum(e) / len(e)
                mae = sum(abs(x) for x in e) / len(e)
                line += f"{me:>+7.2f}({mae:.2f})"
            print(line)
        # 站级总偏差 + 显著性（t 检验的粗略版: |ME| > 2*SE 即认为稳定）
        print(f"\n  站级合计（跨全部档）")
        print(f"  {'站点':<14}{'n':>5}{'ME':>8}{'MAE':>8}{'2×标准误':>10}  判定")
        for stn in sorted(NAMES):
            e = [x for k, v in g.items() if k[0] == stn and k[1] == kind for x in v]
            if len(e) < 5:
                continue
            me = sum(e) / len(e)
            sd = math.sqrt(sum((x - me) ** 2 for x in e) / max(1, len(e) - 1))
            se2 = 2 * sd / math.sqrt(len(e))
            verdict = ("稳定偏高" if me > se2 else "稳定偏低" if me < -se2
                       else "无稳定偏差")
            print(f"  {stn} {NAMES[stn]:<9}{len(e):>5}{me:>+8.2f}"
                  f"{sum(abs(x) for x in e)/len(e):>8.2f}{se2:>10.2f}  {verdict}")
        print()
    _blend_report(args)
    _slot_report(args)
    print("注意: 同一天同一站的多个档高度相关，标准误偏乐观。"
          "有效样本约等于「天数×站数」，两周以上再下结论。")
    return 0


# 日常判读规则: 这两个站看「不排除」，其余看「预报」。
# 依据是它们的 w 在每次留一验证里都稳定 >=0.5（见 _blend_report），
# 物理上对应「晚见顶站」—— 重庆 46%、武汉 31% 的日子 16-17 时才见顶
LATE_PEAK = {"ZUCK", "ZHHH"}


def _slot_report(args):
    """按起报时刻看命中率，回答「该主要盯哪一轮」。"""
    conn = sqlite3.connect(args.db)
    rows = list(conn.execute(
        "SELECT station, target_date, slot, pred, p90, obs FROM fc "
        "WHERE kind='nowcast' AND obs IS NOT NULL AND p90 IS NOT NULL"))
    conn.close()
    if len(rows) < 40:
        return
    import collections
    val = lambda s, p, q: q if s in LATE_PEAK else p
    G = collections.defaultdict(list)
    for s, d, slot, p, q, o in rows:
        G[(d, slot)].append(abs(val(s, p, q) - o))
    days = sorted({d for d, _ in G})
    print(f"\n── 按起报时刻（规则: {'/'.join(sorted(LATE_PEAK))} 看不排除，其余看预报）")
    print(f"  {'起报':<7}{'天数':>5}{'平均命中站数':>13}{'8站全对':>10}{'8站全±1℃':>11}{'MAE':>8}")
    for slot in range(9, 16):
        ds = [d for d in days if len(G.get((d, slot), [])) >= 8]
        if not ds:
            continue
        avg = sum(sum(1 for x in G[(d, slot)] if x == 0) for d in ds) / len(ds)
        allhit = sum(1 for d in ds if all(x == 0 for x in G[(d, slot)]))
        allw1 = sum(1 for d in ds if all(x <= 1 for x in G[(d, slot)]))
        mae = sum(x for d in ds for x in G[(d, slot)]) / sum(len(G[(d, slot)]) for d in ds)
        print(f"  {slot} 时{'':<3}{len(ds):>5}{avg:>11.1f}/8{allhit:>7}/{len(ds):<3}"
              f"{allw1:>8}/{len(ds):<3}{mae:>8.3f}")

    # 待验观察: 晚见顶站在 15 时可能该改看「预报」（那时多半已见顶，
    # 用不排除会多报 1 度）。2026-07-30 首次出现，样本太少，先跟踪
    print(f"\n  晚见顶站在各时次: 用「不排除」 vs 用「预报」哪个好")
    print(f"  {'起报':<7}{'n':>4}{'用不排除MAE':>13}{'用预报MAE':>12}")
    for slot in range(9, 16):
        sub = [r for r in rows if r[2] == slot and r[0] in LATE_PEAK]
        if len(sub) < 4:
            continue
        a = sum(abs(r[4] - r[5]) for r in sub) / len(sub)
        b = sum(abs(r[3] - r[5]) for r in sub) / len(sub)
        mark = "  <- 预报更好" if b < a else ""
        print(f"  {slot} 时{'':<3}{len(sub):>4}{a:>13.3f}{b:>12.3f}{mark}")


def _blend_report(args):
    """「预报 + w×(不排除 − 预报)」的最优 w，并用留一天交叉验证给出诚实估计。

    直接在全部数据上选 w 再在同一批数据上报效果 = 过拟合。
    实测差别很大: 同批数据上重庆「改善 0.958」，留一天验证只剩 +0.05 且不稳定。
    """
    import math
    conn = sqlite3.connect(args.db)
    rows = list(conn.execute(
        "SELECT station, slot, pred, p90, obs, target_date FROM fc "
        "WHERE kind='nowcast' AND obs IS NOT NULL AND p90 IS NOT NULL"))
    conn.close()
    if len(rows) < 40:
        return
    days = sorted({r[5] for r in rows})
    mean = lambda v: sum(v) / len(v)

    def fit_w(sub):
        best = (9e9, 0.0)
        for i in range(11):
            w = i / 10
            m = mean([abs(round(p + w * (q - p)) - o) for _, _, p, q, o, _ in sub])
            if m < best[0]:
                best = (m, w)
        return best[1]

    print(f"\n── 混合权重 w: 最终值 = 预报 + w×(不排除 − 预报)")
    print(f"  {'站点':<14}{'n':>5}{'全量拟合w':>11}" +
          "".join(f"{'留出'+d[5:]:>9}" for d in days))
    stable = []
    for stn in sorted(NAMES):
        sub = [r for r in rows if r[0] == stn]
        if len(sub) < 10:
            continue
        line = f"  {stn} {NAMES[stn]:<9}{len(sub):>5}{fit_w(sub):>11.1f}"
        ws = []
        for d in days:
            tr = [r for r in sub if r[5] != d]
            ws.append(fit_w(tr) if len(tr) >= 8 else 0.0)
            line += f"{ws[-1]:>9.1f}"
        print(line)
        if min(ws) >= 0.5:                 # 每一次留一都稳定要求高 w 才算数
            stable.append(stn)

    # 留一天交叉验证: 用其余天定 w，在留出那天检验
    base, new = [], []
    for d in days:
        tr = [r for r in rows if r[5] != d]
        te = [r for r in rows if r[5] == d]
        W = {}
        for stn in NAMES:
            sub = [r for r in tr if r[0] == stn]
            W[stn] = fit_w(sub) if len(sub) >= 10 else 0.0
        base += [abs(p - o) for s, _, p, q, o, _ in te]
        new += [abs(round(p + W[s] * (q - p)) - o) for s, _, p, q, o, _ in te]
    f1 = lambda v: 100 * sum(1 for x in v if x <= 1) / len(v)
    print(f"\n  留一天交叉验证（这才是诚实的估计）: n={len(base)}")
    print(f"    MAE   {mean(base):.3f} -> {mean(new):.3f}   ({mean(base)-mean(new):+.3f})")
    print(f"    ±1℃   {f1(base):.1f}% -> {f1(new):.1f}%")
    if stable:
        print(f"  每次留一都稳定要求 w>=0.5 的站: "
              f"{'、'.join(NAMES[s] for s in stable)} -> 这些站往「不排除」靠")
    else:
        print("  暂无站点在每次留一里都稳定要求高 w，先别用这个公式")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="verify.sqlite")
    ap.add_argument("--obs-db", default="cn.sqlite")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    return cmd_report(args) if args.report else cmd_collect(args)


if __name__ == "__main__":
    sys.exit(main())

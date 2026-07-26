#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
taf_compare.py — 每天把 MOS 预报与当日 TAF 报文的 TX 并排记下来，攒分歧样本

    python3 taf_compare.py --pred pred_mos.csv          # 抓当前 TAF 并入库
    python3 taf_compare.py --analyze                    # 回填实况，看谁准
    python3 taf_compare.py --analyze --min-gap 2        # 只看分歧 >=2 度的日子

为什么值得攒: TAF 是民航业务在用的预报，MOS 在整体 MAE 上已经赢它
（七模式 D+1 1.08 vs TAF 1.26-1.53），但**两者分歧大的日子谁对**是另一个问题。
如果 TAF 在分歧日有独立价值，那它就该进模型（当特征）；如果没有，
那这条对照可以彻底关掉。这个问题只能靠前瞻样本回答，回算做不到 ——
AWC 只存 15 天 TAF，历史补不回来。

口径与 taf_bias.py 一致:
  - 只取落在午后峰值时段（10-19 时北京时）的 TX
  - 排除压在有效期边界上的 TX（那是「窗口内最大」，不是日最高温预报）
  - 同一 (站,目标日,TX值) 只记首次出现，AMD 里的搬运不重复计数
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import taf_bias as TB

UTC = timezone.utc
CST = timezone(timedelta(hours=8))
PEAK_H0, PEAK_H1 = 10, 19

DDL = """
CREATE TABLE IF NOT EXISTS pair (
    station TEXT, target_date TEXT, lead INTEGER,
    mos REAL, mos_round INTEGER, model_raw REAL,
    taf_tx REAL, taf_issue TEXT, recorded_utc TEXT,
    obs REAL,
    PRIMARY KEY (station, target_date, lead));
"""


def fetch_taf_tx(ids):
    """抓当前 TAF，解析出 (站, 目标日) -> (TX, 发报时刻)。取峰值时段内最高的那条。"""
    import re
    recs = TB._get("taf", {"ids": ",".join(ids), "format": "json"})
    out = {}
    for t in recs:
        stn, raw = t.get("icaoId"), t.get("rawTAF", "")
        issue = TB._norm(t.get("issueTime") or "")
        if not (stn and raw and issue):
            continue
        vf, vt = TB._iso(t.get("validTimeFrom")), TB._iso(t.get("validTimeTo"))
        ref = datetime.fromisoformat(issue)
        for neg, val, dd, hh in re.findall(r"\bTX(M?)(\d{2})/(\d{2})(\d{2})Z", raw):
            tv = TB._resolve(int(dd), int(hh), ref)
            if tv is None:
                continue
            # 压在有效期边界上的 TX 含义不同，丢掉
            if any(x and abs((tv - datetime.fromisoformat(x)).total_seconds()) < 60
                   for x in (vf, vt)):
                continue
            loc = tv.astimezone(CST)
            if not (PEAK_H0 <= loc.hour <= PEAK_H1):
                continue
            v = -float(val) if neg else float(val)
            k = (stn, loc.strftime("%Y-%m-%d"))
            if k not in out or v > out[k][0]:
                out[k] = (v, issue)
    return out


def cmd_record(args):
    if not os.path.exists(args.pred):
        print(f"[error] 找不到 {args.pred}，先跑 "
              f"predict_mos.py --csv-out {args.pred}", file=sys.stderr)
        return 1
    preds = list(csv.DictReader(open(args.pred, encoding="utf-8")))
    ids = sorted({r["station"] for r in preds})

    print(f"抓取 {len(ids)} 个站的当前 TAF…", file=sys.stderr)
    try:
        tx = fetch_taf_tx(ids)
    except Exception as e:
        print(f"[warn] TAF 取数失败: {e}", file=sys.stderr)
        return 0                              # 不该因为对照失败而让主流程报错
    print(f"  解析出 {len(tx)} 个 (站, 目标日) 的 TX", file=sys.stderr)

    conn = sqlite3.connect(args.db)
    conn.executescript(DDL)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    today = datetime.now(CST).date().isoformat()
    n = 0
    print(f"\n{'='*70}")
    print(f"MOS vs TAF（{now[:16]}Z 记录）")
    print(f"  {'站点':<8}{'目标日':<12}{'时效':>5}{'MOS':>7}{'TAF TX':>9}{'分歧':>7}")
    skipped = 0
    for r in preds:
        k = (r["station"], r["date"])
        t = tx.get(k)
        # 目标日已过（或就是今天且已过午后）时，当前 TAF 的有效期早就不覆盖
        # 那天的峰值窗口，只会记一行空值。跳过，别把样本库填满 NULL
        if t is None and r["date"] <= today:
            skipped += 1
            continue
        gap = "" if t is None else f"{int(r['pred_round']) - t[0]:+.0f}"
        print(f"  {r['station']:<8}{r['date']:<12}D+{r['lead']:<3}"
              f"{r['pred_round']:>7}{('--' if t is None else f'{t[0]:.0f}'):>9}"
              f"{gap:>7}")
        conn.execute(
            "INSERT OR REPLACE INTO pair (station,target_date,lead,mos,mos_round,"
            "model_raw,taf_tx,taf_issue,recorded_utc,obs) VALUES (?,?,?,?,?,?,?,?,?,"
            "COALESCE((SELECT obs FROM pair WHERE station=? AND target_date=? "
            "AND lead=?), NULL))",
            (r["station"], r["date"], int(r["lead"]), float(r["pred"]),
             int(r["pred_round"]), float(r["model_raw"]),
             None if t is None else t[0], None if t is None else t[1], now,
             r["station"], r["date"], int(r["lead"])))
        n += 1
    conn.commit()
    conn.close()
    msg = f"\n已记录 {n} 条到 {args.db}"
    if skipped:
        msg += f"（跳过 {skipped} 条: 目标日已过，当前 TAF 不覆盖其峰值窗口）"
    print(msg + "。攒够样本后跑 --analyze")
    return 0


def cmd_analyze(args):
    if not os.path.exists(args.db):
        print(f"[error] 还没有 {args.db}，先跑几天记录", file=sys.stderr)
        return 1
    conn = sqlite3.connect(args.db)
    conn.executescript(DDL)

    # 回填实况
    if os.path.exists(args.obs_db):
        oc = sqlite3.connect(args.obs_db)
        obs = {(s, d): v for s, d, v in oc.execute(
            "SELECT station, date, tmax FROM daily WHERE tmax IS NOT NULL")}
        oc.close()
        n = 0
        for stn, d, lead in conn.execute(
                "SELECT station, target_date, lead FROM pair WHERE obs IS NULL"
        ).fetchall():
            v = obs.get((stn, d))
            if v is not None:
                conn.execute("UPDATE pair SET obs=? WHERE station=? AND "
                             "target_date=? AND lead=?", (v, stn, d, lead))
                n += 1
        conn.commit()
        if n:
            print(f"回填了 {n} 条实况", file=sys.stderr)

    rows = list(conn.execute(
        "SELECT station,target_date,lead,mos_round,taf_tx,obs FROM pair "
        "WHERE obs IS NOT NULL AND taf_tx IS NOT NULL ORDER BY target_date"))
    conn.close()
    if not rows:
        print("还没有「实况 + TAF 都齐全」的样本。TAF 只存 15 天，"
              "要每天跑 record 才攒得起来。")
        return 0

    def block(sub, tag):
        n = len(sub)
        em = [abs(r[3] - r[5]) for r in sub]
        et = [abs(r[4] - r[5]) for r in sub]
        wm = sum(1 for a, b in zip(em, et) if a < b)
        wt = sum(1 for a, b in zip(em, et) if a > b)
        print(f"\n── {tag}  n={n}")
        print(f"  MOS  MAE={sum(em)/n:.2f}  ±1℃={100*sum(1 for x in em if x<=1)/n:.0f}%")
        print(f"  TAF  MAE={sum(et)/n:.2f}  ±1℃={100*sum(1 for x in et if x<=1)/n:.0f}%")
        print(f"  逐条胜负: MOS 赢 {wm}  TAF 赢 {wt}  平 {n-wm-wt}")
        if n >= 10:
            import train_mos as T
            r = T.paired_boot(et, em, [x[1] for x in sub])
            if r:
                d, lo, hi = r
                v = ("MOS 显著更优" if lo > 0 else
                     "TAF 显著更优" if hi < 0 else "无显著差异")
                print(f"  配对检验 ΔMAE={d:+.3f} [{lo:+.3f}, {hi:+.3f}]  {v}")

    for lead in sorted({r[2] for r in rows}):
        sub = [r for r in rows if r[2] == lead]
        block(sub, f"D+{lead} 全部")
        gap = [r for r in sub if abs(r[3] - r[4]) >= args.min_gap]
        if gap:
            block(gap, f"D+{lead} 分歧 >= {args.min_gap:g}℃（关键子集）")

    print(f"\n分歧子集才是重点: 如果 TAF 在这里赢或打平，说明预报员看到了模型没有的"
          f"\n信息，那 TAF 就该进模型当特征；如果一直输，这条对照可以关掉。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="taf_compare.sqlite")
    ap.add_argument("--pred", default="pred_mos.csv")
    ap.add_argument("--obs-db", default="cn.sqlite")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--min-gap", type=float, default=2.0,
                    help="算「分歧日」的阈值（度）")
    args = ap.parse_args()
    return cmd_analyze(args) if args.analyze else cmd_record(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""live_check.py — 随时跑，看当天各站「实况 vs 各时效预报」的横向对比。

    python3 live_check.py                  # 今天，先刷新实况
    python3 live_check.py --date 2026-08-19
    python3 live_check.py --no-fetch       # 不刷新，直接读库（快）

每行一个站，列依次是:
    实况    截止此刻的当日最高温（不是最终日最高温，见下）
    D+2     前天晚上那轮 MOS
    D+1     昨天晚上那轮 MOS
    9~17    当天各时次临近预报（16/17 时是 late_call.py 的「报已达」，不跑模型）
    命中    上面这些预报里，有几个等于当前实况（如 2/5）

没出的预报印 `-`。12:10 跑就只有 D+2/D+1/9/10/11 五个值 —— 12:15 那轮还没跑。

**「实况」是截止此刻的累计最高，不是最终值。** 峰值时段没过完时它只会往上走，
所以那一列「命中」是**暂定**的: 现在算错的，可能只是还没升到；现在算对的，
也可能后面被顶掉。脚本会在表头标出当前是否已过峰值时段（`train_nowcast.PEAK_H1`），
过了才可以当定论看。2026-08-17 就是拿截断值当日最高温做出整张错表的（库只到
15 时，而北京 15:30、郑州 16:00 才见顶）。

数据源:
  实况    cn.sqlite（默认先跑 iem_multi + wu_obs 刷新最近 1 天）
  D+1/2   pred_mos.csv（run_daily 每天重写）+ verify.sqlite 兜底
  9~15    当天的 pred_YYYY-MM-DD.log

**日志解析只取主预报表**（表头之后、第一个 `──` 之前）。2026-08-19 踩过:
「见顶时刻概率」「档位配置建议」「近期偏差订正」各自也是「两空格 + ICAO +
站名 + 数字」的格式，不截住就会把三张表都当成预报读进来。
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import re
import sqlite3
import subprocess
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import stations as _S                      # noqa: E402
import train_nowcast as N                  # noqa: E402
from tablefmt import L, R, w                # noqa: E402  显示宽度对齐，见那里

HERE = os.path.dirname(os.path.abspath(__file__)) or "."


CUTS = [9, 10, 11, 12, 13, 14, 15, 16, 17]
HDR = re.compile(r"^\s*站点\s+预报\s+不排除\s+已达")
ROW = re.compile(r"^\s{2,}(Z[A-Z]{3})\s+\S+\s+(-?\d+)\s")


def refresh(db: str) -> None:
    """按生产同一条路径刷新实况。失败不致命，只是数据旧一点。"""
    stns = ",".join(sorted(_S.ICAOS))
    for cmd in ([sys.executable, "iem_multi.py", "--db", db, "--stations", stns,
                 "--recent-days", "1", "--timeout", "90"],
                [sys.executable, "wu_obs.py", "--db", db, "--update", "--days", "1"]):
        try:
            subprocess.run(cmd, cwd=HERE, capture_output=True, timeout=300)
        except Exception as e:                          # noqa: BLE001
            print(f"[warn] {cmd[1]} 刷新失败（{e}），用库里现有数据",
                  file=sys.stderr)


def observed(db: str, d: str):
    """{站: (截止此刻最高温, 最后一条观测的小时)}"""
    conn = sqlite3.connect(os.path.join(HERE, db))
    out = {}
    for s, mx, lh in conn.execute(
            """SELECT station, MAX(temp_c),
                 MAX(CAST(strftime('%H', datetime(obs_time_utc,'+8 hours')) AS INT))
               FROM obs WHERE local_date = ? AND temp_c IS NOT NULL GROUP BY 1""",
            (d,)):
        out[s] = (round(mx), lh)
    conn.close()
    return out


def mos_preds(d: str):
    """{lead: {站: 预报}}。pred_mos.csv 只有当前那一轮，历史轮次走 verify.sqlite。"""
    out = {1: {}, 2: {}}
    p = os.path.join(HERE, "pred_mos.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("date") == d and r.get("lead") in ("1", "2"):
                v = r.get("pred_round") or r.get("pred")
                if v not in (None, ""):
                    out[int(r["lead"])][r["station"]] = round(float(v))
    vp = os.path.join(HERE, "verify.sqlite")
    if os.path.exists(vp):
        conn = sqlite3.connect(vp)
        try:
            for s, sl, pv in conn.execute(
                    "SELECT station, slot, pred FROM fc "
                    "WHERE target_date=? AND kind='mos'", (d,)):
                if pv is not None and s not in out.get(sl, {}):
                    out.setdefault(sl, {})[s] = round(float(pv))
        except sqlite3.OperationalError:
            pass
        conn.close()
    return out


def nowcasts(d: str):
    """{(时次, 站): 预报}。同一时次有两个批次（:15 主批 / :50 的 WU 站）。"""
    out = {}
    p = os.path.join(HERE, f"pred_{d}.log")
    if not os.path.exists(p):
        return out
    cut, inb = None, False
    for ln in open(p, encoding="utf-8", errors="replace"):
        m = re.search(r"(\d{1,2}) 时起报", ln)
        if m and "#####" in ln:
            cut, inb = int(m.group(1)), False
            continue
        if HDR.match(ln):
            inb = True
            continue
        if inb and ln.lstrip().startswith(("──", "关于", "「")):
            inb = False
            continue
        if inb and cut is not None:
            r = ROW.match(ln)
            if r:
                out[(cut, r.group(1))] = int(r.group(2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--db", default="cn.sqlite")
    ap.add_argument("--no-fetch", action="store_true", help="不刷新实况，直接读库")
    a = ap.parse_args()

    if not a.no_fetch and a.date == date.today().isoformat():
        refresh(a.db)

    obs = observed(a.db, a.date)
    if not obs:
        print(f"[error] {a.date} 没有任何观测。库是空的，或日期写错了",
              file=sys.stderr)
        return 1
    mos = mos_preds(a.date)
    nc = nowcasts(a.date)

    lh = max(v[1] for v in obs.values())
    done = lh >= N.PEAK_H1
    print(f"\n  {a.date}   实况截止 {lh} 时"
          f"   {'✓ 已过峰值时段，命中可当定论' if done else f'⚠ 未过峰值时段（须 >= {N.PEAK_H1} 时），实况还会涨，命中是暂定'}")
    print(f"  跑于 {datetime.now():%H:%M}\n")

    # 列宽（显示列）: 站点 16 / 实况 6 / 每个预报列 7 / 命中 9
    CW, OW, PW, HW = 16, 6, 7, 9
    head = ("  " + L("站点", CW) + R("实况", OW) + R("D+2", PW) + R("D+1", PW)
            + "".join(R(f"{c}时", PW) for c in CUTS) + R("命中", HW))
    print(head)
    print("  " + "-" * (w(head) - 2))

    tot_h = tot_n = 0
    for s in sorted(_S.ICAOS):
        if s not in obs:
            continue
        act = obs[s][0]
        cells, h, n = [], 0, 0
        for v in ([mos.get(2, {}).get(s), mos.get(1, {}).get(s)]
                  + [nc.get((c, s)) for c in CUTS]):
            if v is None:
                cells.append(R("-", PW))
            else:
                n += 1
                h += (v == act)
                cells.append(R(("✓" + str(v)) if v == act else str(v), PW))
        tot_h += h
        tot_n += n
        print("  " + L(f"{s} {_S.NAMES.get(s, '')}", CW) + R(str(act), OW)
              + "".join(cells) + R(f"{h}/{n}", HW))

    print("  " + "-" * (w(head) - 2))
    print("  " + L("合计", CW) + R("", OW)
          + "".join(R(_col(mos, nc, obs, i), PW)
                    for i in range(2 + len(CUTS)))
          + R(f"{tot_h}/{tot_n}", HW))
    if not done:
        print(f"\n  注: 「实况」是截止 {lh} 时的累计最高，峰值时段（到 "
              f"{N.PEAK_H1} 时）还没过完 —— 现在算错的可能只是还没升到。")
    return 0


def _col(mos, nc, obs, i):
    """某一列的「命中/有值」，列序: D+2, D+1, 9..15。"""
    h = n = 0
    for s, (act, _) in obs.items():
        v = (mos.get(2, {}).get(s) if i == 0 else
             mos.get(1, {}).get(s) if i == 1 else
             nc.get((CUTS[i - 2], s)))
        if v is not None:
            n += 1; h += (v == act)
    return f"{h}/{n}" if n else "-"


if __name__ == "__main__":
    sys.exit(main())

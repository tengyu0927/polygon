#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
taf_bias.py — 回补 15 天 TAF + METAR，分站分轮次算偏差，并交叉验证偏差订正是否真的有效

单文件、只用标准库、不依赖任何其他脚本或已有数据库。新机器直接跑。

    python3 taf_bias.py                    # 回补 + 分析（约 8 分钟）
    python3 taf_bias.py --days 7           # 只回补 7 天，先试试
    python3 taf_bias.py --analyze-only     # 已回补过，只重跑分析

**ZGSZ 注意**: 本脚本用 AWC 的深圳宝安 METAR，这是**对的** ——
TAF 本来就是为宝安机场发的，拿宝安实况对它才有意义。
但要清楚: 主预报链路（predict_nowcast / live_tmax / cn.sqlite 里的 ZGSZ）
已改用 WU 的香港流浮山序列（见 wu_obs.py），因为最终打分以 WU 为准。
所以本脚本的 ZGSZ 数字与主链路的不是同一个站，别混着比。

核心问题: TAF 的 TX 相对实况日最高温，偏差是不是稳定的？
         如果稳定，减掉它能不能让预报更准？

关键方法: 留一交叉验证。在同一批数据上算偏差再减掉，MAE 必然下降，
         那是过拟合。只有「用其他日子估的偏差，去订正今天」才说明问题。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

UTC = timezone.utc
CST = timezone(timedelta(hours=8))
BASE = "https://aviationweather.gov/api/data"
UA = "taf-bias/1.0 (station Tmax verification research)"

STATIONS = {
    "ZBAA": "北京首都", "ZSPD": "上海浦东", "ZGGG": "广州白云", "ZGSZ": "深圳宝安",
    "ZUUU": "成都双流", "ZUCK": "重庆江北", "ZHHH": "武汉天河", "ZSQD": "青岛胶东",
}
PEAK_H0, PEAK_H1 = 10, 19          # 午后峰值时段(北京时)
MIN_PEAK_OBS = 6                   # 峰值时段最少观测条数，不足则该日不参与

DDL = """
CREATE TABLE IF NOT EXISTS raw_taf (
    station TEXT, issue_utc TEXT, valid_from TEXT, valid_to TEXT,
    report_type TEXT, raw TEXT,
    PRIMARY KEY (station, issue_utc, raw));
CREATE TABLE IF NOT EXISTS raw_tx (
    station TEXT, issue_utc TEXT, valid_utc TEXT, local_date TEXT,
    local_hour INTEGER, temp_c REAL, at_edge INTEGER,
    PRIMARY KEY (station, issue_utc, valid_utc));
CREATE TABLE IF NOT EXISTS raw_obs (
    station TEXT, obs_utc TEXT, temp_c REAL,
    PRIMARY KEY (station, obs_utc));
"""


# ============================================================ 取数

def _get(kind: str, params: dict, timeout: int = 30, retries: int = 3):
    url = f"{BASE}/{kind}?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for a in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if r.status == 204:
                    return []
                d = json.loads(r.read().decode("utf-8", "replace"))
            return d.get("data", d) if isinstance(d, dict) else d
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(60 * a)
                continue
            if e.code >= 500 and a < retries:
                time.sleep(5 * a)
                continue
            raise
        except Exception:
            if a < retries:
                time.sleep(5 * a)
                continue
            raise
    return []


def _norm(ts) -> str | None:
    """AWC 的时间戳格式不统一: '2026-07-24 09:01:00'、'...00.000Z'、带偏移的都有。"""
    if not ts:
        return None
    import re as _re
    s = str(ts).strip().replace(" ", "T")
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    if not _re.search(r"[+-]\d{2}:\d{2}$", s):
        s += "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    # 规整成唯一形式，否则主键去重会失效
    return dt.replace(microsecond=0).astimezone(UTC).isoformat()


def _iso(epoch) -> str | None:
    if epoch in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(int(epoch), UTC).isoformat()
    except (ValueError, TypeError, OSError):
        return None


def backfill_taf(conn, ids, days, sleep):
    """date 参数逐小时回溯。端点只返回每站最新一条，走一遍时间轴才能拿全序列。"""
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    steps = list(range(0, days * 24))
    print(f"回补 TAF: {len(steps)} 次请求，约 {len(steps)*sleep/60:.1f} 分钟", file=sys.stderr)
    seen = 0
    for i, back in enumerate(steps, 1):
        when = now - timedelta(hours=back)
        try:
            recs = _get("taf", {"ids": ",".join(ids), "format": "json",
                                "date": when.strftime("%Y-%m-%dT%H:%M:%SZ")})
        except Exception as e:
            print(f"  [warn] {when:%m-%d %HZ}: {e}", file=sys.stderr)
            continue
        for t in recs:
            try:
                seen += store_taf(conn, t)
            except Exception as e:                     # 单条坏记录不该毁掉整轮
                print(f"  [skip] {t.get('icaoId')} {t.get('issueTime')}: {e}",
                      file=sys.stderr)
        if i % 24 == 0:
            conn.commit()
            print(f"  {i}/{len(steps)}  累计 {seen} 条 TX", file=sys.stderr)
        time.sleep(sleep)
    conn.commit()
    print(f"  完成，共 {seen} 条 TX", file=sys.stderr)


def store_taf(conn, t) -> int:
    stn, raw = t.get("icaoId"), t.get("rawTAF", "")
    vf, vt = _iso(t.get("validTimeFrom")), _iso(t.get("validTimeTo"))
    issue = t.get("issueTime")
    if not (stn and issue and raw):
        return 0
    issue = _norm(issue)
    if not issue:
        return 0
    rtype = "TAF AMD" if " AMD " in f" {raw} " else ("TAF COR" if " COR " in f" {raw} " else "TAF")
    conn.execute("INSERT OR IGNORE INTO raw_taf VALUES (?,?,?,?,?,?)",
                 (stn, issue, vf, vt, rtype, raw))

    import re
    n = 0
    ref = datetime.fromisoformat(issue)
    for neg, val, dd, hh in re.findall(r"\bTX(M?)(\d{2})/(\d{2})(\d{2})Z", raw):
        tv = _resolve(int(dd), int(hh), ref)
        if tv is None:
            continue
        loc = tv.astimezone(CST)
        edge = int(any(x and abs((tv - datetime.fromisoformat(x)).total_seconds()) < 60
                       for x in (vf, vt)))
        n += conn.execute(
            "INSERT OR IGNORE INTO raw_tx VALUES (?,?,?,?,?,?,?)",
            (stn, issue, tv.isoformat(), loc.strftime("%Y-%m-%d"), loc.hour,
             -float(val) if neg else float(val), edge)).rowcount
    return n


def _resolve(day, hour, ref):
    extra = 1 if hour >= 24 else 0
    hour = hour - 24 if hour >= 24 else hour
    best = None
    for d in (-1, 0, 1):
        y, m = ref.year, ref.month + d
        if m == 0:
            y, m = y - 1, 12
        elif m == 13:
            y, m = y + 1, 1
        try:
            c = datetime(y, m, day, hour, tzinfo=UTC) + timedelta(days=extra)
        except ValueError:
            continue
        if best is None or abs(c - ref) < abs(best - ref):
            best = c
    return best


def backfill_obs(conn, ids, days, sleep):
    """METAR 端点支持 hours 参数，先试；只回来一条就退化成逐日 date 回溯。"""
    print("回补 METAR 实况…", file=sys.stderr)
    try:
        probe = _get("metar", {"ids": ids[0], "format": "json", "hours": "24"})
        bulk = len(probe) > 3
    except Exception:
        bulk = False
    print(f"  hours 参数{'可用' if bulk else '不可用，改逐时回溯'}", file=sys.stderr)

    n = 0
    if bulk:
        for d in range(days):
            when = datetime.now(UTC) - timedelta(days=d)
            try:
                recs = _get("metar", {"ids": ",".join(ids), "format": "json",
                                      "hours": "24",
                                      "date": when.strftime("%Y-%m-%dT%H:%M:%SZ")})
            except Exception as e:
                print(f"  [warn] {when:%m-%d}: {e}", file=sys.stderr)
                continue
            try:
                n += store_obs(conn, recs)
            except Exception as e:
                print(f"  [skip] {when:%m-%d}: {e}", file=sys.stderr)
            conn.commit()
            print(f"  {d+1}/{days} 天  累计 {n} 条", file=sys.stderr)
            time.sleep(sleep)
    else:
        now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
        steps = list(range(0, days * 24))
        for i, back in enumerate(steps, 1):
            when = now - timedelta(hours=back)
            try:
                recs = _get("metar", {"ids": ",".join(ids), "format": "json",
                                      "date": when.strftime("%Y-%m-%dT%H:%M:%SZ")})
            except Exception:
                continue
            try:
                n += store_obs(conn, recs)
            except Exception:
                pass
            if i % 24 == 0:
                conn.commit()
                print(f"  {i}/{len(steps)}  累计 {n} 条", file=sys.stderr)
            time.sleep(sleep)
    conn.commit()
    print(f"  完成，共 {n} 条观测", file=sys.stderr)


def store_obs(conn, recs) -> int:
    n = 0
    for m in recs:
        stn, temp = m.get("icaoId"), m.get("temp")
        t = _iso(m.get("obsTime")) or _norm(m.get("reportTime"))
        if not (stn and t and temp is not None):
            continue
        n += conn.execute("INSERT OR IGNORE INTO raw_obs VALUES (?,?,?)",
                          (stn, t, float(temp))).rowcount
    return n


# ============================================================ 分析

def load(conn):
    # 实况日最高温（北京时日界），带峰值时段完整性过滤
    daily = defaultdict(list)
    for stn, t, v in conn.execute("SELECT station, obs_utc, temp_c FROM raw_obs"):
        loc = datetime.fromisoformat(t).astimezone(CST)
        daily[(stn, loc.strftime("%Y-%m-%d"))].append((loc.hour, v))
    obs = {}
    for k, vals in daily.items():
        if sum(1 for h, _ in vals if PEAK_H0 <= h <= PEAK_H1) < MIN_PEAK_OBS:
            continue
        obs[k] = max(v for _, v in vals)

    # 预报：排除边界 TX，只留峰值时段内的
    rows = list(conn.execute(
        "SELECT station, issue_utc, valid_utc, local_date, local_hour, temp_c "
        "FROM raw_tx WHERE at_edge = 0 AND local_hour BETWEEN ? AND ?",
        (PEAK_H0, PEAK_H1)))

    # 同一 (站,目标日,温度值) 取最早发报时刻 —— AMD 常是搬运，不是新预报
    first = {}
    for stn, iss, _, ld, _, tc in rows:
        k = (stn, ld, tc)
        first[k] = min(first.get(k, iss), iss)

    # 去重: 同一 (站,目标日,TX值) 只保留首次出现那条。
    # 否则同一条预报被后续轮次搬运时会重复计数，n 虚高、置信区间虚窄、
    # 留一验证也失效（留出的"今天"和估偏差的"其他日子"里有同一预报的副本）
    out, kept = [], set()
    for stn, iss, val, ld, lh, tc in rows:
        eff = first[(stn, ld, tc)]
        if iss != eff:
            continue
        key = (stn, ld, tc)
        if key in kept:
            continue
        kept.add(key)
        o = obs.get((stn, ld))
        if o is None:
            continue
        ei = datetime.fromisoformat(eff)
        lead = (datetime.fromisoformat(val) - ei).total_seconds() / 3600
        h = ei.astimezone(CST).hour
        out.append({
            "station": stn, "date": ld,
            "cycle": ((h - 5) % 24) // 6 * 6 + 5,          # 归到 05/11/17/23 轮
            "lead_h": lead,
            # 目标日相对起报日: 0=当天, 1=次日
            "day": (datetime.strptime(ld, "%Y-%m-%d").date()
                    - ei.astimezone(CST).date()).days,
            "fcst": tc, "obs": o, "err": tc - o,
        })
    return out, obs


def stats(errs):
    n = len(errs)
    if not n:
        return None
    me = sum(errs) / n
    mae = sum(abs(e) for e in errs) / n
    sd = math.sqrt(sum((e - me) ** 2 for e in errs) / n) if n > 1 else 0.0
    return {"n": n, "me": me, "mae": mae, "sd": sd,
            "se": sd / math.sqrt(n) if n else 0.0,
            "p1": 100 * sum(1 for e in errs if abs(e) <= 1) / n}


def loo_cv(samples):
    """
    留一交叉验证：用其他日子估的偏差订正今天，看 MAE 到底降不降。
    在同一批数据上估完再减，MAE 必然下降，那个数字没有意义。
    """
    by_day = defaultdict(list)
    for s in samples:
        by_day[s["date"]].append(s)
    days = sorted(by_day)
    if len(days) < 4:
        return None
    raw, corr = [], []
    for d in days:
        others = [s["err"] for dd in days if dd != d for s in by_day[dd]]
        if not others:
            continue
        bias = sum(others) / len(others)          # 只用其他日子估偏差
        for s in by_day[d]:
            raw.append(s["err"])
            corr.append(s["err"] - bias)
    if not raw:
        return None
    m0 = sum(abs(e) for e in raw) / len(raw)
    m1 = sum(abs(e) for e in corr) / len(corr)
    return {"mae_raw": m0, "mae_corr": m1, "gain": m0 - m1, "n": len(raw)}


def report(samples):
    if not samples:
        print("没有可用样本。检查回补是否成功、日期范围是否重叠。")
        return
    days = sorted({s["date"] for s in samples})
    print(f"样本 {len(samples)} 条（已去重：同一预报被后续轮次搬运时只计一次）")
    print(f"{len(days)} 天 ({days[0]} ~ {days[-1]}) | "
          f"{len({s['station'] for s in samples})} 站")
    for d in sorted({s["day"] for s in samples}):
        sub = [s for s in samples if s["day"] == d]
        lead = sorted(s["lead_h"] for s in sub)[len(sub) // 2]
        print(f"  D+{d}: {len(sub)} 条，中位时效 {lead:.1f}h")
    print()

    print("=" * 78)
    print("整体")
    s = stats([x["err"] for x in samples])
    print(f"  n={s['n']}  ME={s['me']:+.2f}  MAE={s['mae']:.2f}  "
          f"标准差={s['sd']:.2f}  ±1℃={s['p1']:.0f}%")
    print(f"  偏差/散度 = {abs(s['me'])/s['mae']:.2f}  "
          f"（接近 1 说明误差主要是系统性的，订正有效；接近 0 说明主要是随机的）")
    lo, hi = s["me"] - 1.96 * s["se"], s["me"] + 1.96 * s["se"]
    print(f"  ME 的 95% 区间 [{lo:+.2f}, {hi:+.2f}]"
          f"{'  ← 跨 0，偏差跟 0 无法区分' if lo < 0 < hi else '  ← 偏差显著'}")

    print("\n" + "=" * 78)
    d0 = [x for x in samples if x["day"] == 0]
    if d0:
        print("仅 D+0（当日预报，口径最干净）")
        st = stats([x["err"] for x in d0])
        cv = loo_cv(d0)
        print(f"  n={st['n']}  ME={st['me']:+.2f}  MAE={st['mae']:.2f}  "
              f"±1℃={st['p1']:.0f}%"
              + (f"  | LOO 订正后 {cv['mae_corr']:.2f} ({cv['gain']:+.2f})" if cv else ""))
        print("\n" + "=" * 78)
    print("分站（各轮次合并）")
    print(f"  {'站点':<14}{'n':>4}{'ME':>8}{'MAE':>7}{'标准差':>8}{'±1℃':>7}   LOO 订正后 MAE")
    for stn in sorted({x["station"] for x in samples}):
        sub = [x for x in samples if x["station"] == stn]
        st = stats([x["err"] for x in sub])
        cv = loo_cv(sub)
        tail = (f"{cv['mae_corr']:.2f}  ({cv['gain']:+.2f})" if cv else "样本不足")
        print(f"  {stn} {STATIONS.get(stn,''):<9}{st['n']:>4}{st['me']:>+8.2f}"
              f"{st['mae']:>7.2f}{st['sd']:>8.2f}{st['p1']:>6.0f}%   {tail}")

    print("\n" + "=" * 78)
    print("分起报轮次 × 目标日（D+0 和 D+1 必须分开，否则各轮的样本构成不同、无法比较）")
    for d in sorted({x["day"] for x in samples}):
        print(f"  ── D+{d} ──")
        for cyc in sorted({x["cycle"] for x in samples if x["day"] == d}):
            sub = [x for x in samples if x["cycle"] == cyc and x["day"] == d]
            st = stats([x["err"] for x in sub])
            if not st or st["n"] < 3:
                continue
            lead = sorted(x["lead_h"] for x in sub)[len(sub) // 2]
            print(f"    {cyc:02d}时轮  n={st['n']:>3}  时效{lead:5.1f}h  "
                  f"ME={st['me']:+.2f}  MAE={st['mae']:.2f}  ±1℃={st['p1']:.0f}%")

    print("\n" + "=" * 78)
    cv = loo_cv(samples)
    print("偏差订正到底有没有用？（留一交叉验证，全样本）")
    if cv:
        print(f"  订正前 MAE {cv['mae_raw']:.3f}  →  订正后 {cv['mae_corr']:.3f}"
              f"   净收益 {cv['gain']:+.3f}℃")
        if cv["gain"] <= 0.02:
            print("  收益微乎其微：误差以随机成分为主，单纯加常数偏差没用。")
            print("  下一步应该找条件偏差 —— 分晴雨、分温度区间、分季节看，")
            print("  或者直接上多元回归（模式输出 + 实况轨迹 + TAF 一起当输入）。")
        else:
            print("  订正有效。但注意样本期短，季节变化会让偏差漂移，")
            print("  正式使用时用滑动窗口（如最近 30-60 天）重估，别用固定常数。")
    else:
        print("  样本天数不足，先多攒几天。")

    print("\n  提醒：METAR 气温是整数度，TAF 的 TX 也是整数度。")
    print("  四舍五入本身就给 MAE 设了约 0.25℃ 的地板，别追一个达不到的目标。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="taf_bias.sqlite")
    ap.add_argument("--days", type=int, default=15, help="回补天数，AWC 上限 15")
    ap.add_argument("--sleep", type=float, default=1.0, help="请求间隔秒")
    ap.add_argument("--ids", default=",".join(STATIONS))
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    ids = [s.strip().upper() for s in args.ids.split(",") if s.strip()]
    conn = sqlite3.connect(args.db)
    conn.executescript(DDL)

    if not args.analyze_only:
        backfill_taf(conn, ids, min(args.days, 15), args.sleep)
        backfill_obs(conn, ids, min(args.days, 15), args.sleep)

    samples, _ = load(conn)
    print()
    report(samples)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
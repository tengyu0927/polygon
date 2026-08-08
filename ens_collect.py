#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ens_collect.py — 每轮抓一次 Open-Meteo 集合预报，聚合后入库。**只攒数据，不进模型。**

    python3 ens_collect.py --db ens.sqlite                # 抓今天+明天
    python3 ens_collect.py --db ens.sqlite --report       # 看攒了多少

为什么要自己攒:

Open-Meteo 的集合 API **拿不到历史**（2026-08-04 实测）:
  - `past_days=92` 时间轴有 93 天，但 89 天全是 null，实际只有 3 天有值
  - `temperature_2m_previous_day1` 参数被接受但返回全空 —— **没有固定时效**
  - `historical-forecast-api` / `archive-api` 对集合模式只返回 1 个变量列

所以时效只能自己保证: 记下**抓取时刻**，由它和目标日推出 lead。
这正是本项目最忌讳的「训练/预测时效不一致」的防线 —— 训练时用哪个 lead 的
行，预测时就必须用同一个 lead。

为什么值得攒:

现在的「模式离散度」是 6 个**不同模式**互相分歧算出来的，是个很差的不确定性
代理 —— 实测对大升幅日完全没有区分度（10 时剩余升幅 >=2 度的样本按离散度
四等分，命中率 37%/38%/30%/32%）。真正的工具是**单一模式扰动初值的集合**:
ECMWF 51 + GFS 31 + ICON 40 + GEM 21 = 143 个成员。

**攒够约 3 个月（~720 站日）才够做 A/B。在那之前它不进任何模型。**

一次请求可以带多个站（逗号分隔，按顺序返回），所以每轮只要 4 次请求。
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import sqlite3
import statistics as st
import sys
import urllib.parse
import urllib.request

CST = datetime.timezone(datetime.timedelta(hours=8))
UTC = datetime.timezone.utc

import stations as _S  # 站点清单唯一真相源
STATIONS = _S.COORD
MODELS = ["ecmwf_ifs025", "gfs05", "icon_seamless", "gem_global"]
API = "https://ensemble-api.open-meteo.com/v1/ensemble"

DDL = """
CREATE TABLE IF NOT EXISTS ens (
  station TEXT, target_date TEXT, model TEXT,
  fetch_utc TEXT,          -- 抓取时刻，lead 由它推出
  cutoff INTEGER,          -- 抓取时的北京时整点（对应哪一轮临近预报）
  lead_h REAL,             -- 抓取时刻 -> 目标日 14 时（北京）的小时数
  n INTEGER,               -- 有效成员数
  tmax_mean REAL, tmax_sd REAL, tmax_min REAL, tmax_max REAL,
  tmax_p10 REAL, tmax_p25 REAL, tmax_p50 REAL, tmax_p75 REAL, tmax_p90 REAL,
  peak_h_mean REAL, peak_h_sd REAL,
  PRIMARY KEY (station, target_date, model, cutoff))
"""


def http_json(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "ploygon/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def quant(v, q):
    """线性插值分位数。成员数不多，不引 numpy。"""
    if not v:
        return None
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    i = q * (len(s) - 1)
    lo = int(i)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def fetch_model(model, days, timeout):
    """一次请求拿全部 8 个站。返回 {station: {date: {member_idx: {hour: t}}}}"""
    stns = list(STATIONS)
    q = urllib.parse.urlencode({
        "latitude": ",".join(f"{STATIONS[s][0]}" for s in stns),
        "longitude": ",".join(f"{STATIONS[s][1]}" for s in stns),
        "models": model, "hourly": "temperature_2m",
        "forecast_days": days, "timezone": "Asia/Shanghai"})
    d = http_json(f"{API}?{q}", timeout=timeout)
    if not isinstance(d, list):
        d = [d]
    out = {}
    for stn, blk in zip(stns, d):
        h = blk.get("hourly") or {}
        cols = [k for k in h if k != "time"]
        per = collections.defaultdict(lambda: collections.defaultdict(dict))
        for i, t in enumerate(h.get("time", [])):
            day, hh = t[:10], int(t[11:13])
            for m, c in enumerate(cols):
                v = h[c][i]
                if v is not None:
                    per[day][m][hh] = v
        out[stn] = per
    return out


def agg(members):
    """members: {member_idx: {hour: temp}} -> 聚合量。白天覆盖不全的丢掉。"""
    tmax, peak = [], []
    for hrs in members.values():
        day = {x: t for x, t in hrs.items() if 6 <= x <= 21}
        if len(day) < 14:                 # 白天时次太少，这个成员不算
            continue
        mx = max(day.values())
        tmax.append(mx)
        peak.append(min(x for x, t in day.items() if t == mx))
    if len(tmax) < 10:                    # 成员太少，整条不要
        return None
    return {
        "n": len(tmax),
        "tmax_mean": round(st.mean(tmax), 3),
        "tmax_sd": round(st.pstdev(tmax), 3),
        "tmax_min": round(min(tmax), 2), "tmax_max": round(max(tmax), 2),
        "tmax_p10": round(quant(tmax, 0.10), 3),
        "tmax_p25": round(quant(tmax, 0.25), 3),
        "tmax_p50": round(quant(tmax, 0.50), 3),
        "tmax_p75": round(quant(tmax, 0.75), 3),
        "tmax_p90": round(quant(tmax, 0.90), 3),
        "peak_h_mean": round(st.mean(peak), 3),
        "peak_h_sd": round(st.pstdev(peak), 3),
    }


def report(db):
    c = sqlite3.connect(db)
    c.execute(DDL)
    n, d0, d1 = c.execute(
        "SELECT COUNT(*), MIN(target_date), MAX(target_date) FROM ens").fetchone()
    print(f"  共 {n} 行   目标日 {d0} ~ {d1}")
    if not n:
        return 0
    print(f"\n  {'模式':<16}{'行数':>8}{'站日':>8}{'平均成员':>10}{'平均离散':>10}")
    for m, k, sd, nm in c.execute(
            "SELECT model, COUNT(*), AVG(tmax_sd), AVG(n) FROM ens GROUP BY 1"):
        sj = c.execute("SELECT COUNT(DISTINCT station||target_date) FROM ens "
                       "WHERE model=?", (m,)).fetchone()[0]
        print(f"  {m:<16}{k:>8}{sj:>8}{nm:>10.0f}{sd:>10.2f}")
    print(f"\n  {'时效档':<12}{'行数':>8}")
    for lo, hi in ((0, 9), (9, 15), (15, 27), (27, 60)):
        k = c.execute("SELECT COUNT(*) FROM ens WHERE lead_h>=? AND lead_h<?",
                      (lo, hi)).fetchone()[0]
        print(f"  {f'{lo}-{hi}h':<12}{k:>8}")
    sj = c.execute("SELECT COUNT(DISTINCT station||target_date) FROM ens").fetchone()[0]
    print(f"\n  独立站日 {sj}   做 A/B 大约需要 700+（约 3 个月）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="ens.sqlite")
    ap.add_argument("--days", type=int, default=2, help="今天 + 未来几天")
    ap.add_argument("--timeout", type=int, default=45)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report(a.db)

    now = datetime.datetime.now(UTC)
    now_cst = now.astimezone(CST)
    c = sqlite3.connect(a.db)
    c.execute(DDL)

    n_ok = n_skip = 0
    for model in MODELS:
        try:
            got = fetch_model(model, a.days, a.timeout)
        except Exception as e:                        # noqa: BLE001
            print(f"[warn] {model} 取数失败: {str(e)[:80]}", file=sys.stderr)
            continue
        for stn, per in got.items():
            for day, members in per.items():
                g = agg(members)
                if not g:
                    n_skip += 1
                    continue
                peak = datetime.datetime.combine(
                    datetime.date.fromisoformat(day),
                    datetime.time(14), CST)
                lead = round((peak - now).total_seconds() / 3600, 2)
                if lead <= 0:                          # 目标日的峰值已经过去
                    continue
                c.execute(
                    "INSERT OR REPLACE INTO ens VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (stn, day, model, now.isoformat(timespec="seconds"),
                     now_cst.hour, lead, g["n"],
                     g["tmax_mean"], g["tmax_sd"], g["tmax_min"], g["tmax_max"],
                     g["tmax_p10"], g["tmax_p25"], g["tmax_p50"],
                     g["tmax_p75"], g["tmax_p90"],
                     g["peak_h_mean"], g["peak_h_sd"]))
                n_ok += 1
    c.commit()
    tot = c.execute("SELECT COUNT(*) FROM ens").fetchone()[0]
    sj = c.execute("SELECT COUNT(DISTINCT station||target_date) FROM ens").fetchone()[0]
    print(f"  集合: 写入 {n_ok} 行（跳过 {n_skip}），累计 {tot} 行 / {sj} 站日",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

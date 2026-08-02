#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wu_check.py — 核对我们的实测与 Weather Underground 是否一致

    python3 wu_check.py                      # 最近 60 天，全部 8 站
    python3 wu_check.py --start 2026-01-01 --end 2026-07-31
    python3 wu_check.py --stations ZGSZ --hourly   # 单站逐时明细

为什么需要这个脚本:

最终的对错以 WU 页面上的日最高温为准。我们训练和检验用的是 IEM ASOS 归档的
METAR。两者必须完全一致，否则模型学到的是 A 的规律、被 B 打分，会留下改不掉的
系统偏差。

2026-08-01 首次核对（7 个月 212 天）的结果:

    站      完全相同   平均差
    北京     100%     +0.02
    上海     100%     -0.00
    广州     100%     +0.00
    成都      99%     -0.01
    武汉     100%     +0.00
    重庆     100%     +0.00
    青岛     100%     -0.01
    深圳      29%     +0.42    <- 有问题

**深圳的根因**: WU 的 `ZGSZ:9:CN` 返回的 `obs_name` 是 **Lau Fau Shan（流浮山）**，
即香港天文台在新界后海湾的站（WMO 45035），离深圳宝安机场约 30 公里，
不是 ZGSZ 的 METAR。试过 `VHHH:9:HK` / `59493:9:CN` / `CHGD9820:1:CH`，
WU 没有任何 id 能给出深圳宝安的观测。

差异不是常数（标准差 1.34℃，25% 的日子差 ≥2℃），减一个固定偏差修不掉。

用的是 wunderground.com 页面自身内嵌的公开 apiKey，与浏览器打开那个页面
拿到的是同一份数据。
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import statistics as st
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta

API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
NAMES = {"ZBAA": "北京首都", "ZSPD": "上海浦东", "ZGGG": "广州白云",
         "ZGSZ": "深圳宝安", "ZUUU": "成都双流", "ZUCK": "重庆江北",
         "ZHHH": "武汉天河", "ZSQD": "青岛胶东"}

# WU 的 ZGSZ 挂的是香港流浮山，不是深圳宝安。核对时单独标注，别当成我们的 bug。
MISMATCHED = {"ZGSZ": "WU 挂的是 Lau Fau Shan（香港流浮山，WMO 45035）"}


def wu_obs(station, start, end, retries=3):
    """取 WU 某站某段时间的逐条观测。start/end 为 YYYYMMDD。"""
    url = ("https://api.weather.com/v1/location/"
           + urllib.parse.quote(f"{station}:9:CN")
           + "/observations/historical.json?"
           + urllib.parse.urlencode({"apiKey": API_KEY, "units": "m",
                                     "startDate": start, "endDate": end}))
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.load(urllib.request.urlopen(req, timeout=60)).get("observations", [])
        except Exception as e:
            if i == retries - 1:
                print(f"  [warn] {station} {start}: {type(e).__name__} {str(e)[:60]}",
                      file=sys.stderr)
                return []
            time.sleep(2 * (i + 1))
    return []


def months(a: date, b: date):
    """按自然月切块 —— 接口一次最多给一个月。"""
    cur = a.replace(day=1)
    while cur <= b:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield max(cur, a), min(nxt - timedelta(days=1), b)
        cur = nxt


def fetch(stations, a: date, b: date, sleep=0.35):
    """返回 (日最高温 {(站,日): 值}, 逐时 {(站,日,时): 值}, 站名 {站: obs_name})"""
    daily, hourly, oname = {}, {}, {}
    for s in stations:
        for x0, x1 in months(a, b):
            for x in wu_obs(s, x0.strftime("%Y%m%d"), x1.strftime("%Y%m%d")):
                if x.get("obs_name") and s not in oname:
                    oname[s] = x["obs_name"]
                if x.get("temp") is None:
                    continue
                t = time.gmtime(x["valid_time_gmt"] + 8 * 3600)   # 北京时
                d = time.strftime("%Y-%m-%d", t)
                k = (s, d)
                daily[k] = x["temp"] if k not in daily else max(daily[k], x["temp"])
                hk = (s, d, t.tm_hour)
                hourly[hk] = x["temp"] if hk not in hourly else max(hourly[hk], x["temp"])
            time.sleep(sleep)
    return daily, hourly, oname


def ours(db, stations, a: date, b: date):
    conn = sqlite3.connect(db)
    q = ("SELECT station, local_date, MAX(temp_c) FROM obs "
         "WHERE local_date BETWEEN ? AND ? AND temp_c IS NOT NULL "
         f"AND station IN ({','.join('?' * len(stations))}) GROUP BY 1, 2")
    return {(s, d): v for s, d, v in
            conn.execute(q, [a.isoformat(), b.isoformat(), *stations])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cn.sqlite")
    ap.add_argument("--stations", default=",".join(NAMES))
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--days", type=int, default=60, help="没给 --start 时回看几天")
    ap.add_argument("--hourly", action="store_true", help="逐时明细（配合单站用）")
    args = ap.parse_args()

    stations = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    b = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    a = date.fromisoformat(args.start) if args.start else b - timedelta(days=args.days - 1)

    print(f"核对 {a} ~ {b}，{len(stations)} 站", file=sys.stderr)
    W, WH, oname = fetch(stations, a, b)
    O = ours(args.db, stations, a, b)

    print(f"\n日最高温: WU vs 我们（{a} ~ {b}）\n")
    print(f"  {'站':<10}{'n':>5}{'完全相同':>10}{'差1度':>8}{'差≥2度':>9}"
          f"{'平均差':>9}{'标准差':>8}{'最大差':>8}")
    # 判据检的是「会毁掉训练的东西」，不是「完全一样」。会毁掉训练的只有两种:
    #   1. 固定偏差 —— 训练标签比打分标签系统性高/低，模型学进去就改不掉
    #   2. 大幅分歧 —— 挂错站（2026-08-01 的 ZGSZ 就是: 30% 相同、26% 差≥2度）
    # 差 1 度的零星分歧不属于任何一种: ZGSZ 改用 WU 之后 30 天 93% 相同、
    # 7% 差 1 度、平均差 +0.00 —— 那是 WU 的日最高按次时次观测算、我们按逐时
    # 序列算的取样差，对称无偏，消不掉也不需要消。按老判据（相同 <95%）它天天
    # 报警，而一个天天响的警告等于没有警告（mos_multi.csv 停更两天没人发现，
    # 就是被这类噪声淹的）。相同率仍然打印，只是不再升级成 WARN。
    BIAS_MAX, GROSS_MAX = 0.30, 0.10
    bad, noisy, absent = [], [], []
    for s in stations:
        ks = sorted(d for (ss, d) in W if ss == s and (s, d) in O)
        if not ks:
            print(f"  {NAMES.get(s, s):<9} 无数据")
            absent.append(s)
            continue
        df = [W[(s, d)] - O[(s, d)] for d in ks]
        same = sum(1 for x in df if x == 0) / len(ks)
        gross = sum(1 for x in df if abs(x) >= 2) / len(ks)
        if abs(st.mean(df)) >= BIAS_MAX or gross >= GROSS_MAX:
            bad.append(s)
        elif same < 0.95:
            noisy.append(s)
        print(f"  {NAMES.get(s, s):<9}{len(ks):>5}{same:>10.0%}"
              f"{sum(1 for x in df if abs(x) == 1) / len(ks):>8.0%}"
              f"{sum(1 for x in df if abs(x) >= 2) / len(ks):>9.0%}"
              f"{st.mean(df):>+9.2f}{st.pstdev(df):>8.2f}{max(abs(x) for x in df):>8.0f}")

    # 只有对不上的站才提示站名。WU 给的名字本来就是自己的写法
    # （Beijing/Capital、Qingdao/Liuting 等），名字不同不代表站不同 ——
    # 判据是上面的「完全相同」比例，不是站名。
    for s in stations:
        if s in oname and (s in bad or s in MISMATCHED):
            note = MISMATCHED.get(s, "与我们的实测对不上，很可能不是同一个站")
            print(f"\n  [注意] WU 的 {s} 返回站名 “{oname[s]}” —— {note}")

    if bad:
        print(f"\n  不一致的站: {', '.join(NAMES.get(s, s) for s in bad)}")
        print(f"  这些站训练用的实况与打分用的实况不是一回事，模型会留下改不掉的偏差。")
    elif absent:
        # 取不到数不等于对得上。WU 连着问会限流，整批空返回时老版本照样打印
        # 「全部一致」—— 那是把取数失败当成了全清。
        print(f"\n  没核上（WU 取不到数，多半是限流）: "
              f"{', '.join(NAMES.get(s, s) for s in absent)}")
        print(f"  这轮没有结论，不是「一致」。")
    else:
        print(f"\n  全部一致。训练/检验用的实况与 WU 打分口径相同。")
    if noisy:
        # 无偏的取样噪声。不影响训练，也不用处理，但保留可见性 ——
        # 万一哪天它开始往一边偏，上面的判据会接手。
        print(f"  [注] 有零星 1 度分歧但无系统偏差: "
              f"{', '.join(NAMES.get(s, s) for s in noisy)}（取样差，不影响训练）")

    if args.hourly:
        for s in stations:
            ks = sorted(d for (ss, d) in W if ss == s and (s, d) in O)
            conn = sqlite3.connect(args.db)
            oh = {(d, h): t for d, h, t in conn.execute(
                "SELECT local_date, CAST(strftime('%H', datetime(obs_time_utc,'+8 hours')) AS INT),"
                " MAX(temp_c) FROM obs WHERE station=? AND local_date BETWEEN ? AND ?"
                " AND temp_c IS NOT NULL GROUP BY 1,2", (s, a.isoformat(), b.isoformat()))}
            pair = [(WH[(s, d, h)], oh[(d, h)]) for d in ks for h in range(24)
                    if (s, d, h) in WH and (d, h) in oh]
            if not pair:
                continue
            df = [x - y for x, y in pair]
            print(f"\n  {NAMES.get(s, s)} 逐时  n={len(pair)}  "
                  f"完全相同 {sum(1 for x in df if x == 0) / len(df):.0%}  "
                  f"平均差 {st.mean(df):+.2f}  平均绝对差 {st.mean(abs(x) for x in df):.2f}")
            c = collections.Counter(df)
            for v in sorted(c):
                print(f"    差 {v:+.0f}℃  {c[v]:>5} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())

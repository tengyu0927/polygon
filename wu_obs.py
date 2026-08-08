#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wu_obs.py — 把 Weather Underground 的观测写进 cn.sqlite

    python3 wu_obs.py --migrate            # 首次: 备份 ZGSZ 的 METAR 并换成 WU 序列
    python3 wu_obs.py --update             # 每天: 增量刷新最近几天（含当天）
    python3 wu_obs.py --update --days 30
    python3 wu_obs.py --restore            # 回滚: 换回原来的 METAR

为什么要这么做（2026-08-01）:

**最终对错以 WU 页面上的日最高温为准。** 而 WU 的 `ZGSZ:9:CN` 返回的
`obs_name` 是 **Lau Fau Shan（流浮山）** —— 香港天文台在新界后海湾的站
（WMO 45035），离深圳宝安机场约 30 公里，根本不是 ZGSZ 的 METAR。
试过 `VHHH:9:HK` / `59493:9:CN` / `CHGD9820:1:CH`，WU 没有任何 id 能给出
深圳宝安的观测（见 `wu_check.py` 的说明）。

后果: 我们对深圳宝安报得几乎完美（13 时 MAE 0.267、±1℃ 100%），
但按 WU 打分只有 MAE 1.133、±1℃ 65% —— 因为在给另一个站打分。
差异不是常数（标准差 1.38℃、25% 的日子差 ≥2℃），减固定偏差修不掉。

**为什么直接换实测，而不是学一个「ZGSZ -> 流浮山」的转换**：
实测对比（624 天，2024-07~2026-07，按时间 7:3 切分），目标=流浮山日最高温：

| 13 时方案 | MAE | ±1℃ |
|---|---|---|
| ZGSZ **真实** tmax + 常数偏移（转换法的理论上限）| 1.030 | 57% |
| 只用 ZGSZ 上午实测 | 1.015 | 55% |
| **只用流浮山自己上午实测** | **0.631** | **89%** |

用目标站自己的上午实测，比转换法的天花板还好 0.4 度。这与整个项目的
主线结论一致: **只有站点自己的温度计知道这个站点**。

**代价（必须知道）**:

1. WU 不给 `clds` / `wx_phrase` / `vis`，所以深圳的 `cld_now`、`cld_mean_am`、
   `ts_am`、`ra_am`、`obsc_am` 五个特征全为空，靠 `matrix()` 的中位数填补
   ＋缺测指示位参与。
2. 流浮山的归档密度: 2024-07 前是 8 条/天（三小时一次），临近预报一天都用不了；
   2024-07 起 19-24 条/天。**所以深圳只有约 2 年可用历史，其余 7 站有 30 年。**
   `--migrate` 默认只写密度 >= MIN_OBS 的日子。
3. 原 METAR 全部备份在 `obs_zgsz_metar` 表里，`--restore` 可完整回滚。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone

API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
CST = timezone(timedelta(hours=8))
# 2026-08-08: 从只服务 ZGSZ 改成多站。站点清单取自 stations.WU_STATIONS。
#   ZGSZ 深圳: WU 挂的是流浮山，归档 2024-07-10 起变密（之前 8 条/天）
#   ZSJN 济南: **IEM 上根本没有逐时数据**（42-81 条/月，每天 1-2 条），
#              只能整站走 WU；实测 2020 年起逐时齐全
# 每站的起始日与最小条数不同，所以按站配置，别再用一个全局常量。
import stations as _S                      # noqa: E402
STATION = "ZGSZ"                           # 兼容旧调用；实际按 --stations 走
BACKUP_TABLE = "obs_zgsz_metar"            # 只有 ZGSZ 有 METAR 需要备份
MIN_OBS = 18          # 一天少于这么多条就不收 —— 三小时一次的日子上午只有 2 条
FIRST_DAY = "2024-07-10"   # 流浮山归档变密的起点，之前是 8 条/天
PER_STATION = {
    "ZGSZ": {"first": "2024-07-10", "min_obs": 18},
    "ZSJN": {"first": "2020-01-01", "min_obs": 18},
}


def wu_obs(station, start, end, retries=3):
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
                print(f"  [warn] {station} {start}~{end}: {type(e).__name__} {str(e)[:70]}",
                      file=sys.stderr)
                return []
            time.sleep(2 * (i + 1))
    return []


def months(a: date, b: date):
    cur = a.replace(day=1)
    while cur <= b:
        nxt = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield max(cur, a), min(nxt - timedelta(days=1), b)
        cur = nxt


def to_rows(obs, station=None):
    """WU 观测 -> obs 表的行。返回 {北京时日期: [行, ...]}。

    **station 必须传。** 2026-08-08 踩过: 这个函数原来把模块级常量 STATION
    （="ZGSZ"）硬写进第一个字段，泛化 ingest 支持多站时漏了这里，于是灌济南
    时把济南的观测按 ZGSZ 写进 obs 表 —— 深圳 2020-01~2024-07 被写进 25344 行
    济南数据（1 月均温 -1℃ 而深圳应是 16℃），直到撞上主键冲突才崩。
    默认值只为兼容旧调用，新代码一律显式传。

    WU 不提供 skyc1 / wxcodes / vis_km，这几列写 NULL；
    train_nowcast.matrix() 会用中位数填补并加缺测指示位。
    """
    stn = station or STATION
    by_day = {}
    for x in obs:
        if x.get("temp") is None:
            continue
        ts = x["valid_time_gmt"]
        dt = datetime.fromtimestamp(ts, timezone.utc)
        d = dt.astimezone(CST).strftime("%Y-%m-%d")
        by_day.setdefault(d, []).append((
            stn,
            ts,                                   # valid_time_gmt
            dt.strftime("%Y-%m-%dT%H:%M:%SZ"),    # obs_time_utc
            d,                                    # local_date
            float(x["temp"]),
            None if x.get("dewPt") is None else float(x["dewPt"]),
            None if x.get("rh") is None else float(x["rh"]),
            None if x.get("wdir") is None else float(x["wdir"]),
            None if x.get("wspd") is None else round(float(x["wspd"]) / 3.6, 2),   # km/h -> m/s
            None if x.get("gust") is None else round(float(x["gust"]) / 3.6, 2),
            None if x.get("pressure") is None else float(x["pressure"]),
            None,        # precip_mm  —— WU 的 precip_hrly 对该站恒为空
            None,        # vis_km
            None,        # skyc1
            None,        # skyl1
            None,        # wxcodes
            None if x.get("feels_like") is None else float(x["feels_like"]),
            None,        # metar
        ))
    return by_day


COLS = ("station, valid_time_gmt, obs_time_utc, local_date, temp_c, dewp_c, rh, "
        "drct, wspd_ms, gust_ms, pres_hpa, precip_mm, vis_km, skyc1, skyl1, "
        "wxcodes, feel_c, metar")


def ingest(conn, a: date, b: date, min_obs=MIN_OBS, sleep=0.35, station=None):
    """抓 [a,b] 的 WU 观测，替换 obs 表里该站对应日期的行。"""
    STATION = station or globals()["STATION"]
    total = skipped = 0
    for x0, x1 in months(a, b):
        obs = wu_obs(STATION, x0.strftime("%Y%m%d"), x1.strftime("%Y%m%d"))
        by_day = to_rows(obs, STATION)
        for d, rows in sorted(by_day.items()):
            # 当天还没走完，条数天然少，不能按 min_obs 卡掉
            today = datetime.now(CST).strftime("%Y-%m-%d")
            if len(rows) < min_obs and d != today:
                skipped += 1
                continue
            conn.execute("DELETE FROM obs WHERE station=? AND local_date=?", (STATION, d))
            conn.executemany(
                f"INSERT INTO obs ({COLS}) VALUES ({','.join('?' * 18)})", rows)
            total += len(rows)
        conn.commit()
        print(f"  {STATION} {x0}~{x1}  写入 "
              f"{sum(len(v) for v in by_day.values()):>5} 条", file=sys.stderr)
        time.sleep(sleep)
    return total, skipped


def migrate(conn, args):
    n = conn.execute("SELECT COUNT(*) FROM obs WHERE station=?", (STATION,)).fetchone()[0]
    have = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                        (BACKUP_TABLE,)).fetchone()[0]
    if have:
        m = conn.execute(f"SELECT COUNT(*) FROM {BACKUP_TABLE}").fetchone()[0]
        print(f"备份表 {BACKUP_TABLE} 已存在（{m:,} 条），跳过备份。", file=sys.stderr)
    else:
        print(f"备份 {STATION} 的 METAR（{n:,} 条）到 {BACKUP_TABLE} …", file=sys.stderr)
        conn.execute(f"CREATE TABLE {BACKUP_TABLE} AS SELECT * FROM obs WHERE station=?",
                     (STATION,))
        conn.commit()
    conn.execute("DELETE FROM obs WHERE station=?", (STATION,))
    conn.commit()
    a = date.fromisoformat(args.first)
    b = date.today()
    print(f"抓 WU（Lau Fau Shan）{a} ~ {b} …", file=sys.stderr)
    total, skipped = ingest(conn, a, b, args.min_obs)
    print(f"\n写入 {total:,} 条，跳过 {skipped} 个密度不足的日子", file=sys.stderr)
    return 0


def restore(conn, args):
    have = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                        (BACKUP_TABLE,)).fetchone()[0]
    if not have:
        print(f"没有 {BACKUP_TABLE}，无法回滚。", file=sys.stderr)
        return 1
    n = conn.execute(f"SELECT COUNT(*) FROM {BACKUP_TABLE}").fetchone()[0]
    conn.execute("DELETE FROM obs WHERE station=?", (STATION,))
    conn.execute(f"INSERT INTO obs ({COLS}) SELECT {COLS} FROM {BACKUP_TABLE}")
    conn.commit()
    print(f"已回滚: 恢复 {n:,} 条 METAR。", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cn.sqlite")
    ap.add_argument("--migrate", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--days", type=int, default=5, help="--update 回看几天（含当天）")
    ap.add_argument("--first", default=FIRST_DAY)
    ap.add_argument("--min-obs", type=int, default=MIN_OBS)
    ap.add_argument("--stations", default=",".join(sorted(_S.WU_STATIONS)),
                    help="逗号分隔；默认 stations.WU_STATIONS")
    ap.add_argument("--backfill", action="store_true",
                    help="按 PER_STATION 的起始日灌全历史（新站首次用）")
    args = ap.parse_args()
    stns = [x.strip().upper() for x in args.stations.split(",") if x.strip()]

    conn = sqlite3.connect(args.db)
    if args.migrate:
        return migrate(conn, args)
    if args.restore:
        return restore(conn, args)
    if args.backfill:
        for s in stns:
            cfg = PER_STATION.get(s, {})
            a = date.fromisoformat(cfg.get("first", args.first))
            b = datetime.now(CST).date()
            n, sk = ingest(conn, a, b, cfg.get("min_obs", args.min_obs), station=s)
            print(f"[{s}] 全历史 {a} ~ {b}: 写入 {n} 条，跳过 {sk} 天",
                  file=sys.stderr)
        return 0
    if args.update:
        b = datetime.now(CST).date()
        a = b - timedelta(days=args.days - 1)
        tot = 0
        for s in stns:
            n, sk = ingest(conn, a, b,
                           PER_STATION.get(s, {}).get("min_obs", args.min_obs),
                           station=s)
            tot += n
        print(f"更新 {a} ~ {b}: {len(stns)} 个站共写入 {tot} 条", file=sys.stderr)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

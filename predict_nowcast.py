#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_nowcast.py — 用当天上午实况预报今日最高温

    python3 predict_nowcast.py --cutoff 12              # 从 cn.sqlite 读
    python3 predict_nowcast.py --auto --live            # 按当前时刻自动选，实时取数
    python3 predict_nowcast.py --cutoff 12 --compare pred_mos.txt

数据源两种，各有取舍:
  默认读 cn.sqlite —— 与训练同源同口径，但需要先跑
      python3 iem_multi.py --db cn.sqlite --stations ... --update
  --live 从 AWC 取实时 METAR —— 不用等 IEM 归档，但单位口径可能与训练源
      略有出入（风速节->米/秒、气压可能是英寸汞柱），脚本会自动换算并提示。
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import stations as _S                     # noqa: E402  站点清单唯一真相源
import train_mos as T                      # noqa: E402
import train_nowcast as N                  # noqa: E402


def fetch_nwp(stations, tgt, gfs_model="gfs_global"):
    """模型若含 NWP 特征，取当天的 GFS 固定时效预报（与训练同口径）。"""
    import build_mos_dataset as B
    import predict_mos as P
    conn, n = P.fetch_window(stations, tgt, tgt, gfs_model, B.VARS)
    daily = B.daily_features(conn, 1)
    return {stn: daily.get((stn, tgt.isoformat()), {}) for stn in stations}


# 特殊的「模式名」: 不走 Open-Meteo，而是实时从 NCEP 取当时能拿到的最新一轮
# GFS（gfs_live.py）。作为第七个集合成员加入 —— **是加，不是换**。
#
# 实测（15 个月滚动回测，按各时次真实可用的时效）:
#   11 时  0.7682 -> 0.7524  -0.0157  P=98%   第七成员用前一天 12Z（18h）
#   13 时  0.4807 -> 0.4685  -0.0122  P=98%   第七成员用当天 00Z（6h）
#    9 时  0.9409 -> 0.9371  -0.0038  P=67%   未过线
# 对照「把 GFS 成员换掉」的方案: 三档全不过线（61%/85%/89%）。**加比换好。**
LOCAL_GFS = "local_gfs"


def fetch_m2(stations, tgt, models, cutoff=None):
    """追加模式的当天预报。models 的顺序必须与训练时 --nwp-csv2 的顺序一致，
    否则 m2_/m3_ 各列会对错模式，模型系数全部错位。"""
    out = []
    for mdl in models:
        daily = {}
        if mdl == LOCAL_GFS:
            try:
                import gfs_live as GL
                cand = GL.pick_run(cutoff if cutoff is not None else 13, tgt)
                for init, desc in cand:
                    per = GL.fetch_run(init, tgt, jobs=10, verbose=False)
                    if not per:
                        continue
                    rows = GL.to_mos(per, tgt)
                    if len(rows) < len(stations):
                        continue
                    daily = {r["station"]: r for r in rows}
                    print(f"  本地 GFS: {desc}", file=sys.stderr)
                    break
                if not daily:
                    print("[warn] 本地 GFS 全部轮次取不到，该成员留空",
                          file=sys.stderr)
            except Exception as e:
                print(f"[warn] 本地 GFS 取数失败: {e}", file=sys.stderr)
        else:
            try:
                daily = fetch_nwp(stations, tgt, mdl)
            except Exception as e:
                print(f"[warn] {mdl} 取数失败: {e}", file=sys.stderr)
        out.append({stn: {k: daily.get(stn, {}).get(c)
                          for k, c in N.M2_COLS.items()} for stn in stations})
    return out

UTC = timezone.utc
CST = timezone(timedelta(hours=8))
AWC = "https://aviationweather.gov/api/data/metar"
UA = "nowcast/1.0 (station Tmax research)"

# 这些站的实况以 Weather Underground 为准，不能用 AWC/IEM 的 METAR。
# **清单取自 stations.py，别在这里再抄一份** —— 2026-08-10 踩过: 加济南时
# stations.WU_STATIONS 加上了 ZSJN，这里却还写死 {"ZGSZ"}，于是济南走不到
# WU 实时取数那条路、只能读库，而库里当天只到 11 时 -> 12 时起直接「无今日观测」。
# 与 to_rows 硬写 STATION、stn_id 训练端设了预测端没设是同一类病。
WU_STATIONS = _S.WU_STATIONS


def _overwrite_from_wu(days, stations, tgt, db):
    """把 days 里这些站的观测换成 WU 的。取不到就退回读库，绝不静默用 METAR。"""
    import wu_obs as WO
    for s in stations:
        got = {}
        for k in range(11):
            d = tgt - timedelta(days=k)
            try:
                rows = WO.to_rows(WO.wu_obs(s, d.strftime("%Y%m%d"),
                                            d.strftime("%Y%m%d"), retries=2), s)
            except Exception as e:
                print(f"[warn] WU 取 {s} {d} 失败: {str(e)[:60]}", file=sys.stderr)
                rows = {}
            for dd, rr in rows.items():
                for r in rr:
                    dt = datetime.fromtimestamp(r[1], UTC).astimezone(CST)
                    cur = got.setdefault((s, dd), {})
                    if dt.hour in cur and cur[dt.hour]["t"] >= r[4]:
                        continue
                    cur[dt.hour] = {"t": r[4], "dewp": r[5], "rh": r[6],
                                    "wspd": r[8], "pres": r[10], "cld": None,
                                    "ts": 0.0, "ra": 0.0, "obsc": 0.0, "drct": r[7]}
        if not got.get((s, tgt.isoformat())):
            print(f"[warn] WU 没给 {s} 当天数据，退回读 {db}（口径仍是 WU 的归档）",
                  file=sys.stderr)
            fb = from_db(db, "obs", [s],
                         {(tgt - timedelta(days=k)).isoformat() for k in range(11)})
            got.update(fb)
        for k in [k for k in days if k[0] == s]:
            days.pop(k)
        days.update(got)
        n = len(got.get((s, tgt.isoformat()), {}))
        site = "Lau Fau Shan" if s == "ZGSZ" else "WU 站"
        print(f"  {s} 改用 WU 实况（{site}），当天 {n} 个小时", file=sys.stderr)


def from_db(db, table, stations, dates):
    """从 cn.sqlite 读指定日期的逐时观测，结构与训练一致。"""
    days = defaultdict(dict)
    if not os.path.exists(db):
        print(f"[error] 找不到实况库 {db}。新机器请先跑 ./bootstrap.sh 建数据",
              file=sys.stderr)
        raise SystemExit(1)
    conn = sqlite3.connect(db)
    if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' "
                        "AND name=?", (table,)).fetchone():
        print(f"[error] {db} 里没有 {table} 表 —— 数据库是空的或建歪了。\n"
              f"        新机器请跑 ./bootstrap.sh；已有库请跑\n"
              f"        python3 iem_multi.py --db {db} --stations ... --backfill",
              file=sys.stderr)
        raise SystemExit(1)
    cols = {c[1] for c in conn.execute(f"PRAGMA table_info({table})")}
    tcol = "valid_time_gmt" if "valid_time_gmt" in cols else "obs_time_utc"
    get = lambda c: c if c in cols else "NULL"
    ph = ",".join("?" * len(stations))
    q = (f"SELECT station, {tcol}, temp_c, {get('dewp_c')}, {get('rh')}, "
         f"{get('wspd_ms')}, {get('pres_hpa')}, {get('skyc1')}, {get('wxcodes')} "
         f"FROM {table} WHERE temp_c IS NOT NULL AND station IN ({ph})")
    params = list(stations)
    # 只要两天数据，别扫全表（248 万行）。local_date 上有索引，能走索引扫描
    if "local_date" in cols:
        dl = sorted(dates)
        q += f" AND local_date BETWEEN ? AND ?"
        params += [min(dl), max(dl)]
    for r in conn.execute(q, params):
        stn, ts = r[0], r[1]
        try:
            if tcol == "valid_time_gmt":
                dt = datetime.fromtimestamp(int(ts), UTC).astimezone(CST)
            else:
                s = str(ts).replace("Z", "+00:00").replace(" ", "T")
                dt = datetime.fromisoformat(s)
                dt = (dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt).astimezone(CST)
        except (ValueError, OSError, TypeError):
            continue
        d = dt.strftime("%Y-%m-%d")
        if d not in dates:
            continue
        tsf, raf, obf = N.wx_flags(r[8])
        cur = days[(stn, d)]
        h = dt.hour
        if h in cur and cur[h]["t"] >= float(r[2]):
            continue
        cur[h] = {"t": float(r[2]),
                  "dewp": None if r[3] is None else float(r[3]),
                  "rh": None if r[4] is None else float(r[4]),
                  "wspd": None if r[5] is None else float(r[5]),
                  "pres": None if r[6] is None else float(r[6]),
                  "cld": N.cloud_frac(r[7]), "ts": tsf, "ra": raf, "obsc": obf}
    conn.close()
    return days


def _rh(t, td):
    """AWC 的 METAR JSON 不给相对湿度，用 Magnus 公式从温度和露点算。
    否则实时模式下 rh_now 永远缺测，与训练口径不一致，精度会静默下降。"""
    if t is None or td is None:
        return None
    import math as _m
    es = lambda x: 6.112 * _m.exp(17.67 * x / (x + 243.5))
    return max(0.0, min(100.0, 100.0 * es(td) / es(t)))


def from_awc(stations, hours=30, retries=3):
    """实时 METAR。单位换算: 风速节->米/秒；气压若是英寸汞柱则换算成百帕。

    AWC 会间歇性 504（实测 2026-07-28 14 时那轮），所以要重试。
    重试用完仍失败时抛出，由调用方决定怎么降级。
    """
    import time as _time
    url = AWC + "?" + urllib.parse.urlencode(
        {"ids": ",".join(stations), "format": "json", "hours": str(hours)})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last = None
    for a in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            break
        except Exception as e:
            last = e
            print(f"[warn] AWC 取数失败({e})，{3*a}s 后重试 {a}/{retries}",
                  file=sys.stderr)
            if a < retries:
                _time.sleep(3 * a)
    else:
        raise last
    recs = data.get("data", data) if isinstance(data, dict) else data

    days, inhg = defaultdict(dict), False
    for m in recs:
        stn, t = m.get("icaoId"), m.get("temp")
        if not stn or t is None or m.get("obsTime") is None:
            continue
        dt = datetime.fromtimestamp(int(m["obsTime"]), UTC).astimezone(CST)
        p = m.get("altim")
        if p is not None:
            p = float(p)
            if 25 < p < 35:                       # 英寸汞柱
                p, inhg = p * 33.8639, True
        cov = None
        cl = m.get("clouds") or []
        if cl and isinstance(cl[0], dict):
            cov = cl[0].get("cover")
        ws = m.get("wspd")
        tsf, raf, obf = N.wx_flags(m.get("wxString"))
        d = days[(stn, dt.strftime("%Y-%m-%d"))]
        h = dt.hour
        if h in d and d[h]["t"] >= float(t):
            continue
        dp = None if m.get("dewp") is None else float(m["dewp"])
        d[h] = {"t": float(t), "dewp": dp,
                "rh": _rh(float(t), dp),
                "wspd": None if ws is None else float(ws) * 0.514444,
                "pres": p, "cld": N.cloud_frac(cov),
                "ts": tsf, "ra": raf, "obsc": obf}
    if inhg:
        print("[note] AWC 气压为英寸汞柱，已换算为百帕", file=sys.stderr)
    return days


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nowcast.json")
    ap.add_argument("--agree-with", default="",
                    help="另一个模型 JSON，用同一批特征再算一遍，输出「一致?」列。"
                         "**不改任何预报值**，只标注两个模型是否给出同一个整数")
    ap.add_argument("--stations", default="",
                    help="只跑这些站（逗号分隔）。默认跑模型里的全部站")
    ap.add_argument("--db", default="cn.sqlite")
    ap.add_argument("--table", default="obs")
    ap.add_argument("--cutoff", type=int, default=0)
    ap.add_argument("--auto", action="store_true", help="按当前时刻自动选最接近的截止")
    ap.add_argument("--date", default="", help="目标日，默认今天")
    ap.add_argument("--live", action="store_true", help="从 AWC 实时取 METAR")
    ap.add_argument("--hurdle", action="store_true", help="用两段式而非直接回归")
    ap.add_argument("--pooled", action="store_true", help="强制用合并模型")
    ap.add_argument("--gfs-model", default="gfs_global")
    ap.add_argument("--extra-models", default="",
                    help="逗号分隔的追加模式，顺序须与训练时 --nwp-csv2 一致，"
                         "如 ecmwf_ifs025,cma_grapes_global,icon_global")
    ap.add_argument("--compare", default="", help="并排提示 D+1 结果文件")
    ap.add_argument("--p90", action="store_true",
                    help="额外给 P90 高端情景（不改点预报，只多一路「不排除冲到」）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"[error] 找不到 {args.model}，先跑 "
              f"train_nowcast.py --dump {args.model}", file=sys.stderr)
        return 1
    specs = json.load(open(args.model, encoding="utf-8"))
    avail = sorted(int(k) for k in specs)

    now = datetime.now(CST)
    if args.auto:
        ok = [c for c in avail if c <= now.hour]
        if not ok:
            print(f"[error] 当前 {now.hour} 时早于最早可用截止 {avail[0]} 时",
                  file=sys.stderr)
            return 1
        cutoff = ok[-1]
    else:
        cutoff = args.cutoff or avail[-1]
    if cutoff not in avail:
        print(f"[error] 模型里没有 {cutoff} 时截止，可用: {avail}", file=sys.stderr)
        return 1
    spec = specs[str(cutoff)]

    tgt = (datetime.strptime(args.date, "%Y-%m-%d").date() if args.date
           else now.date())
    prv = tgt - timedelta(days=1)
    stations = sorted({k.split("|")[0] for k in spec["clim_rise"]})
    # 分批跑: WU 那两个站的整点观测比 METAR 晚落地，让它们单独晚一档跑，
    # 其余 8 站不用陪着等。迟滞状态是「先读全量再逐站更新」，分批不会互相
    # 覆盖（见下方 _state 的加载）。
    if args.stations:
        want = {x.strip().upper() for x in args.stations.split(",") if x.strip()}
        bad = want - set(stations)
        if bad:
            print(f"[error] --stations 里有模型不认识的站: {sorted(bad)}",
                  file=sys.stderr)
            return 1
        stations = [s for s in stations if s in want]

    nwp = {}
    if spec.get("has_nwp"):
        print(f"模型含 NWP 特征，取当天 GFS 预报…", file=sys.stderr)
        try:
            nwp = fetch_nwp(stations, tgt, args.gfs_model)
            ok = sum(1 for v in nwp.values() if v.get("temperature_2m_max") is not None)
            print(f"  {ok}/{len(stations)} 站取到", file=sys.stderr)
        except Exception as e:
            print(f"[warn] GFS 取数失败: {e}\n"
                  f"       NWP 特征将按缺测填补，精度会明显下降", file=sys.stderr)

    m2 = []
    if spec.get("has_nwp") and any(n.startswith("m2_") for n in spec["names"]):
        want = sum(1 for n in spec["names"] if n.endswith("_minus_m1"))
        mdls = [x.strip() for x in args.extra_models.split(",") if x.strip()]
        if len(mdls) != want:
            print(f"[error] 模型需要 {want} 个追加模式，--extra-models 给了 "
                  f"{len(mdls)} 个。顺序错了系数会全部错位", file=sys.stderr)
            return 1
        print(f"取追加模式 {', '.join(mdls)}…", file=sys.stderr)
        m2 = fetch_m2(stations, tgt, mdls, cutoff)

    if args.live:
        # 兜底自己也要有兜底: AWC 挂了就退回读库。库里可能是陈旧实况，
        # 但 morning() 会判断够不够，好过整轮崩掉不出预报
        src = "实时 AWC"
        try:
            days = from_awc(stations)
        except Exception as e:
            src = f"{args.db}（AWC 不可用）"
            print(f"[warn] AWC 不可用（{e}），退回读 {args.db} 的已有实况",
                  file=sys.stderr)
            days = from_db(args.db, args.table, stations,
                           {(tgt - timedelta(days=k)).isoformat()
                            for k in range(11)})
        # ZGSZ 的模型学的是 WU 的 Lau Fau Shan（见 wu_obs.py），
        # 而 AWC 给的是深圳宝安的 METAR —— 两个站相距 30km，
        # 直接用会造成「拿 A 站的上午实测喂 B 站的模型」。必须单独覆盖。
        if WU_STATIONS & set(stations):
            _overwrite_from_wu(days, WU_STATIONS & set(stations), tgt, args.db)
    else:
        src = args.db
        # 多取 10 天: rise_anom_3d/7d 要回看前几个可用日的实际升幅
        days = from_db(args.db, args.table, stations,
                       {(tgt - timedelta(days=k)).isoformat() for k in range(11)})
        have = max((d for (_, d) in days), default="")
        if have < tgt.isoformat():
            print(f"[warn] {args.db} 里没有 {tgt} 的观测（最新 {have or '无'}）。"
                  f"\n       先跑 iem_multi.py --update，或加 --live 实时取数",
                  file=sys.stderr)

    if now.hour < cutoff and tgt == now.date():
        print(f"[warn] 当前 {now.hour}:{now.minute:02d} 早于截止 {cutoff} 时，"
              f"特征会不完整", file=sys.stderr)

    print(f"\n{'='*70}")
    print(f"临近预报  目标日 {tgt}  截止 {cutoff:02d} 时（北京时）  "
          f"{src}")
    p90_col = f"{'不排除':>8}" if args.p90 else ""
    _hdr_eh = f"{'预期命中':>7}" if os.path.exists(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hit_table.json")) else ""
    print(f"\n  {'站点':<14}{'预报':>7}{p90_col}{'已达':>7}{'预计再升':>10}"
          f"{_hdr_eh}{'更高?':>7}{'一致?':>7}{'实况':>9}   备注")

    # 迟滞用的上一轮报出去的整数。报的是整数，连续值在 x.5 附近抖 0.02 度就会
    # 让整数翻个个儿 —— 15 个月回测里 61% 的逐小时改动是这种空转（动出去又回来）。
    # 规则: 上次报了 k 就一直报 k，直到连续值离 k 超过 0.5+HYST 才改。
    #
    # 只在 9-12 时启用。回测（完全命中率口径，3494 站日）:
    #   全时段 0.2   空转 -52%  命中 54.43% -> 53.59%（-0.84pt）
    #   只 9-12 0.2  空转 -32%  命中 54.43% -> 54.34%（-0.09pt）  <- 采用
    # 损失全部发生在 13-15 时: 那时值在收敛，压住真实变动就是纯损失。
    HYST, HYST_CUTOFFS = 0.20, (9, 10, 11, 12)
    # 可改道 —— 一致性检查实跑会真的出预报，不能让它覆盖生产的迟滞状态
    _st_path = os.environ.get("PLOYGON_STATE") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nowcast_state.json")
    _state = {}
    if os.path.exists(_st_path):
        try:
            _j = json.load(open(_st_path, encoding="utf-8"))
            if _j.get("date") == tgt.isoformat():
                _state = _j.get("last", {})
        except Exception:                             # noqa: BLE001
            _state = {}

    # GBM 边文件。缺 sklearn / 缺文件 / 读不出来都自动降级成纯线性，不报错 ——
    # 与 predict_mos.py 同一约定（<model>.gbm.pkl）。
    _GBM, _NP = {}, None
    _gp = args.model + ".gbm.pkl"
    if os.path.exists(_gp):
        try:
            import pickle
            import numpy as _NP                        # noqa: N813
            _GBM = {int(k): v for k, v in pickle.load(open(_gp, "rb")).items()}
        except Exception as e:                          # noqa: BLE001
            print(f"[warn] GBM 没读成（{str(e)[:60]}），本轮只用线性", file=sys.stderr)
            _GBM, _NP = {}, None

    # 对照模型（--agree-with）。**只标注，不改预报值。**
    # 2026-08-12 实测: 两个模型给出同一个整数时 9 时命中 37.9%，分歧时 31.1%
    # （差 6.8pt），而且**在剩余升幅分档内部依然全正**（+1.7~+15.6pt），
    # 说明这是独立于「预期命中」的第二个维度。四变体只多 0.9pt，两个就够。
    # 生产上 9 时本来就在跑不含 AIFS 的影子对照，等于免费。
    _alt = _altgbm = None
    if args.agree_with and os.path.exists(args.agree_with):
        try:
            _alt = json.load(open(args.agree_with, encoding="utf-8")).get(str(cutoff))
            _ap = args.agree_with + ".gbm.pkl"
            if _alt and os.path.exists(_ap) and _NP is not None:
                import pickle as _pk
                _altgbm = _pk.load(open(_ap, "rb")).get(cutoff)
        except Exception as e:                         # noqa: BLE001
            print(f"[warn] 对照模型 {args.agree_with} 读不出来（{e}），不输出一致性",
                  file=sys.stderr)
            _alt = None

    def _alt_int(f, msf, stn):
        """对照模型在同一行特征上的整数结果。任何异常都返回 None（列留空）。"""
        if not _alt:
            return None
        try:
            an, am = _alt["names"], _alt["median"]
            per = (_alt.get("per_station") or {}).get(stn)
            if per and not args.pooled:
                mr, mh, ms = per["ridge"], per.get("hurdle"), per["median"]
            else:
                mr, mh, ms = _alt["ridge"], _alt.get("hurdle"), am
            X, _ = N.matrix([{"f": f}], ms, an)
            r = (N.pred_hurdle(mh, X)[0] if (args.hurdle and mh)
                 else max(0.0, T.ridge_pred(mr, X)[0]))
            gw = _alt.get("gbm_w")
            if _altgbm is not None and gw is not None and _NP is not None:
                Xg, _ = N.matrix([{"f": f}], am, an)
                pg = max(0.0, float(_altgbm.predict(_NP.asarray(Xg, float))[0]))
                r = gw * r + (1 - gw) * pg
            return int(round(msf + r))
        except Exception:                              # noqa: BLE001
            return None

    # 超出概率查表（exceed_table.json）。**不改任何预报值**，只把
    # 「这个数会不会还往上走」量化出来印在旁边 —— 晚见顶的站在 15 时收敛之后
    # 原本一点提示都没有，这一列就是补那个缺口。
    _exc = {}
    _ep = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exceed_table.json")
    if os.path.exists(_ep):
        try:
            _exc = json.load(open(_ep, encoding="utf-8"))
        except Exception:                              # noqa: BLE001
            _exc = {}

    def _exceed(cut, rise):
        if not _exc or rise >= _exc["edges"][-1]:
            return ""
        i = next((k for k, e in enumerate(_exc["edges"]) if rise < e), None)
        c = _exc["cells"].get(f"{cut}|{i}")
        return "" if not c else f"↑{c[0]:.0%}"
    # 高优势清单（edge_table.json + 生产已有的 pred_mos.csv）。
    # **不改任何预报值。** 回答的是「什么时候我们对而隔夜预报错」——
    # 盘口大概率锚定在隔夜预报上，所以优势 = 我们的把握 × 隔夜的错误。
    #
    # 实测（标准窗口，隔夜用 mos_rolling.py 的滚动样本外 D+1，MAE 0.99）:
    #   12 时 与隔夜差 1 度 且 剩余<0.25   447 站日  我们 72% / 隔夜 16%  +57pt
    #   12 时 与隔夜差 >=2 度 且 剩余<0.25  233 站日  我们 65% / 隔夜  4%  +61pt
    #   13 时 与隔夜差 >=2 度 且 剩余<0.25  460 站日  我们 69% / 隔夜  3%  +65pt
    #   与隔夜一致时优势恒为 0（同一个数，谁也不比谁强）
    # 这两类合计约占站日的 15%，每天约 1.5 个站。
    _EDGE = {}
    _ep2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edge_table.json")
    if os.path.exists(_ep2):
        try:
            _EDGE = json.load(open(_ep2, encoding="utf-8"))
        except Exception:                              # noqa: BLE001
            _EDGE = {}
    # 隔夜 D+1: run_daily.sh 每天 23:59 生成 pred_mos.csv，直接读，不新增取数
    _D1 = {}
    _mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pred_mos.csv")
    if os.path.exists(_mp):
        try:
            for _r in csv_mod.DictReader(open(_mp, encoding="utf-8")):
                if _r.get("lead") == "1" and _r.get("date") == tgt.isoformat():
                    _D1[_r["station"]] = int(round(float(_r["pred"])))
        except Exception as e:                         # noqa: BLE001
            print(f"[warn] pred_mos.csv 读不出来（{e}），本轮无高优势清单",
                  file=sys.stderr)

    def _edge(stn, rise, shown):
        d1 = _D1.get(stn)
        if d1 is None or not _EDGE:
            return None
        dd = abs(shown - d1)
        db = 0 if dd == 0 else (1 if dd == 1 else 2)
        rb = next((i for i, e in enumerate(_EDGE["rise_edges"]) if rise < e),
                  len(_EDGE["rise_edges"]))
        c = _EDGE["cells"].get(f"{cutoff}|{db}|{rb}")
        if not c or db == 0:
            return None
        return (d1, c[0], c[1], c[2], c[0] - c[2])

    # 档位配置建议（bucket_table.json）。**不改任何预报值**，只回答
    # 「这个站今天该买几档」—— 盘口是分档的，而误差系统性偏低（今天所有
    # 量化都指向这一点: 标了把握的预报错时 100% 是报低）。
    # 实测 12 时: 只买点预报 47-48%，买「点预报+上一档」71-73%（双向验证
    # 72.7%/71.2%），买 ±1 三档 90%。**第二档必须往上买** —— +1 比 -1 多
    # 7 个百分点。学习型判上下反而更差（69.8% vs 72.7%），一律 +1 最优。
    _BK = {}
    _bp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bucket_table.json")
    if os.path.exists(_bp):
        try:
            _BK = json.load(open(_bp, encoding="utf-8"))
        except Exception:                              # noqa: BLE001
            _BK = {}

    def _bucket_advice(cut, rise, shown):
        """返回 (建议档位串, 历史覆盖率串)。样本不足就返回空。"""
        if not _BK:
            return "", ""
        i = next((k for k, e in enumerate(_BK["edges"]) if rise < e), len(_BK["edges"]))
        c = _BK["cells"].get(f"{cut}|{i}")
        if not c:
            return "", ""
        one, two, three = c[0], c[1], c[2]
        # 一档已经够好（>=85%）就买一档；否则两档；两档还不到 70% 就提三档
        if one >= 0.85:
            return f"{shown}", f"{one:.0%}"
        if two >= 0.70 or three - two < 0.10:
            return f"{shown},{shown+1}", f"{two:.0%}"
        return f"{shown},{shown+1}｜或 ±1 三档", f"{two:.0%}｜{three:.0%}"

    # 自身近期偏差订正。**只在 9 时。**
    #
    # 2026-08-17 量到临近模型自己的签名残差有时间持续性，越早的时次越强
    # （前 5 天平均残差 vs 今天残差）: 9 时 0.275 / 10 时 0.139 / 12 时 0.117 /
    # 15 时 0.037。物理上讲得通 —— 实况信息越少越依赖模式，模式的系统偏差
    # 就越持续。
    #
    # **双向验证**（一半定参数、另一半验，标准窗口）:
    #   前半定→后半验  k=30 α=0.4  35.72% -> 36.67%  +0.95pt  P=91.0%
    #   后半定→前半验  k=5  α=0.4  34.38% -> 35.33%  +0.95pt  P=89.2%
    # 两个方向效果量完全一致，但都过不了 95%。真正的证据是**不敏感**:
    # α=0.4 固定时 k 从 3 到 30 全部两半为正（+0.82%~+1.86%），k=14 固定时
    # α 从 0.2 到 0.6 全部两半为正。取 k=10 / α=0.4 —— 两个平台的中间，
    # 不取 k=7 那个 +1.74% 的峰值。
    #
    # **试过更好看的做法，都更差**（同一批数据、同样双向验证）:
    #   做成特征喂模型 9 时 -0.38% / 10 时 -1.22% / 11 时 -1.37%
    #     —— 信号只有 r=0.28，而模型已有 130 项特征，再塞 4 个弱特征，
    #        学到的噪声比信号多。硬减是把结构先验地强加进去，模型学不到但我们知道对。
    #   分站各调 k,α   9 时 +0.6% vs 全站统一 +1.0%（每站 230 天撑不起两个参数）
    #   只在偏差大/同号时才订正   越挑越差（挑中的是噪声被吹大的日子）
    #   10/11/12 时   +0.52%(不稳) / 反号 / 负
    RESID_CUTOFF, RESID_K, RESID_ALPHA = 9, 10, 0.4
    _HIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "nowcast_hist.json")

    def _recent_bias(stn):
        """该站前 RESID_K 个可用日的平均残差（未取整预报 − 取整实况）。

        **与训练端同口径** —— resid_feat.py 用的就是 `pred_raw - round(actual)`。
        历史预报从 nowcast_hist.json 取（本文件每轮写），实况从库里取。
        样本不足一半就返回 None（宁可不订正）。
        """
        if cutoff != RESID_CUTOFF or not _hist:
            return None
        got = []
        for d in sorted(_hist, reverse=True):
            if d >= tgt.isoformat():
                continue
            p = _hist[d].get(stn)
            a = _past_max.get((stn, d))
            if p is None or a is None:
                continue
            got.append(p - round(a))
            if len(got) >= RESID_K:
                break
        return sum(got) / len(got) if len(got) >= max(2, RESID_K // 2) else None

    _hist, _past_max = {}, {}
    if cutoff == RESID_CUTOFF:
        try:
            if os.path.exists(_HIST_PATH):
                _hist = json.load(open(_HIST_PATH, encoding="utf-8"))
        except Exception:                              # noqa: BLE001
            _hist = {}
        if _hist:
            import sqlite3 as _sq3
            _c3 = _sq3.connect(args.db)
            for _s, _d, _t in _c3.execute(
                    "SELECT station, local_date, MAX(temp_c) FROM obs "
                    "WHERE local_date >= ? AND local_date < ? AND temp_c IS NOT NULL "
                    "GROUP BY 1,2", (min(_hist), tgt.isoformat())):
                _past_max[(_s, _d)] = _t
            _c3.close()

    # 「见顶时刻」判别器（<model>.peak.pkl）。**不改任何预报值** ——
    # 只回答「今天这个站可能几点见顶」，让人知道 12 时看到的数在 15 时后
    # 还有多大概率被顶掉。见 train_nowcast.fit_peak_prob。
    # 概率一律走训练时存的经验频率回填（原始概率偏散: 报 75% 实测 61%）。
    _PEAK = None
    _pkp = args.model + ".peak.pkl"
    if os.path.exists(_pkp) and _NP is not None:
        try:
            import pickle as _pk3
            _PEAK = _pk3.load(open(_pkp, "rb")).get(cutoff)
        except Exception as e:                         # noqa: BLE001
            print(f"[warn] {_pkp} 读不出来（{e}），本轮不出见顶时刻概率",
                  file=sys.stderr)

    def _peak_prob(f, med_, names_):
        """返回 (早, 正常, 晚, 校准后的晚)。缺模型/缺 sklearn 返回 None。"""
        if _PEAK is None:
            return None
        Xp, _ = N.matrix([{"f": f}], med_, names_)
        p = _PEAK["clf"].predict_proba(_NP.asarray(Xp, float))[0]
        i = next((k for k, e in enumerate(_PEAK["edges"]) if p[2] < e),
                 len(_PEAK["edges"]))
        c = _PEAK["cal"][i] if i < len(_PEAK["cal"]) else None
        return float(p[0]), float(p[1]), float(p[2]), c

    # 「已见顶」判别器（<model>.settled.pkl）。判定天已过完就把预报改成
    # 已达值 —— 见 train_nowcast.fit_settled。只在 10-14 时有，9/15 时没有。
    # 缺文件/缺 sklearn 自动跳过，不报错。
    _SETTLED = None
    _sp = args.model + ".settled.pkl"
    if os.path.exists(_sp) and _NP is not None:
        try:
            import pickle as _pk2
            _SETTLED = _pk2.load(open(_sp, "rb")).get(cutoff)
        except Exception as e:                         # noqa: BLE001
            print(f"[warn] {_sp} 读不出来（{e}），本轮不做已见顶覆盖",
                  file=sys.stderr)


    # 「预期命中率」查表（build_hit_table.py 生成）。**不改任何预报值**，
    # 只是把「这个数该不该信」量化出来印在旁边。两周生产 752 条 + 15 个月回测
    # 都指向同一条主线: 决定准不准的是「起报那一刻还剩多少没发生」，不是
    # 「今天什么天气」—— 剩 <=0.5 度时完全命中 72%、剩 >4 度时只有 21%。
    # 读不到表就整列不显示，绝不影响预报。
    _hit = {}
    try:
        _hp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hit_table.json")
        if os.path.exists(_hp):
            _hit = json.load(open(_hp, encoding="utf-8"))
    except Exception:                                  # noqa: BLE001
        _hit = {}

    def _exp_hit(cut, rise):
        if not _hit:
            return None
        b = 0
        for i, e in enumerate(_hit.get("edges", []) + [float("inf")]):
            if rise <= e:
                b = i
                break
        v = _hit.get("cells", {}).get(f"{cut}|{b}")
        if v:
            return v[0]
        return _hit.get("per_cutoff", {}).get(str(cut), _hit.get("global"))

    names, med = spec["names"], spec["median"]
    warned_p90 = False
    rows_out = []
    _raw_today, _shadow = {}, []       # 9 时: 订正前的值 / 被订正改掉的站
    _peaks = []                        # 见顶时刻概率（每站一行）
    _advice = []
    _edges = []
    # 每个站用的是哪个实况源、最新一条是几点。**只是标识，不改任何预报值。**
    # 起因: 滞后 1-2 小时的实况会让模型抓不住当下的升温趋势，而 morning() 的
    # 硬防护只挡 >2 小时（train_nowcast.py 的 `max(o) < cutoff - 2`），
    # 1-2 小时这一档是放行的，且日志里从来没记过观测时刻，事后查不了。
    def _obs_tag(stn, hrs):
        s = "WU" if stn in WU_STATIONS else ("AWC" if args.live else "库")
        hh = [h for h in (hrs or {}) if h <= cutoff]
        if not hh:
            return s, None, f"{s + ' --':>9}"
        last = max(hh)
        lag = cutoff - last
        return s, lag, f"{s + ' ' + str(last) + '时' + ('!' if lag >= 1 else ''):>9}"
    stale = []
    for stn in stations:
        hrs = days.get((stn, tgt.isoformat()))
        _s, _lag, _otag = _obs_tag(stn, hrs)
        if _lag is not None and _lag >= 1:
            stale.append((stn, _s, cutoff - _lag, _lag))
        if not hrs:
            print(f"  {stn} {N.NAMES.get(stn,''):<9}{'--':>7}{'--':>7}{'--':>10}"
                  f"{_otag}   无今日观测")
            continue
        o = N.morning(hrs, cutoff)
        if o is None:
            n = len([h for h in hrs if h <= cutoff])
            print(f"  {stn} {N.NAMES.get(stn,''):<9}{'--':>7}{'--':>7}{'--':>10}"
                  f"{_otag}   截止前仅 {n} 条观测，不足")
            continue

        ph = days.get((stn, prv.isoformat()))
        mo = tgt.month
        cr = spec["clim_rise"].get(f"{stn}|{mo}")
        cp = spec["clim_peak"].get(f"{stn}|{mo}")
        prev = (max(v["t"] for v in ph.values()) if ph else None, cr)
        # stn_id 必须与 make_samples 一样设上 —— 训练端设了、预测端不设，
        # 就是本项目最常犯的静默错配（2026-08-07 加这个特征时差点又踩一次）
        f, msf = N.build_feats(o, cutoff, prev, cr, cp,
                               tgt.timetuple().tm_yday, nwp.get(stn))
        f["stn_id"] = N.STN_IDX.get(stn)

        # 近期升幅异常。训练侧在 make_samples 里算，这条推理路径不走那里，
        # 必须在这补齐 —— 否则这两项永远缺测，与训练口径不一致。
        hist = []
        for k in range(1, 11):
            d0 = (tgt - timedelta(days=k)).isoformat()
            hrs0 = days.get((stn, d0))
            if not hrs0 or N.morning(hrs0, cutoff) is None:
                continue
            c0 = spec["clim_rise"].get(f"{stn}|{int(d0[5:7])}")
            if c0 is None:
                continue
            t0 = max(v["t"] for v in hrs0.values())
            s0 = max(v["t"] for v in N.morning(hrs0, cutoff).values())
            hist.append(t0 - s0 - c0)
        for k in (3, 7):
            f[f"rise_anom_{k}d"] = (sum(hist[:k]) / k if len(hist) >= k else None)

        if m2:
            N.add_m2_feats(f, msf, [mm.get(stn) for mm in m2])

        per = (spec.get("per_station") or {}).get(stn)
        if per and not args.pooled:
            mr, mh, ms, base = per["ridge"], per.get("hurdle"), per["median"], "分站"
            mo, mq = per.get("ordinal"), per.get("q90")
        else:
            mr, mh, ms, base = spec["ridge"], spec.get("hurdle"), med, "合并"
            mo, mq = spec.get("ordinal"), spec.get("q90")
        X, _ = N.matrix([{"f": f}], ms, names)

        if args.hurdle and mh:
            rise, tag = N.pred_hurdle(mh, X)[0], base + "两段式"
        else:
            rise, tag = max(0.0, T.ridge_pred(mr, X)[0]), base + "岭回归"

        # 非线性一路（只有 9/10 时训了，见 train_nowcast.NLIN_CUTOFFS）。
        # GBM 是在**合并**样本上训的，所以喂它的矩阵要用合并的 median，
        # 不能用上面分站那套 —— 列的标准化基准不一样，混用就是喂错数。
        gw = spec.get("gbm_w")
        gmod = _GBM.get(cutoff)
        if gmod is not None and gw is not None and _NP is not None:
            Xg, _ = N.matrix([{"f": f}], med, names)
            pg = max(0.0, float(gmod.predict(_NP.asarray(Xg, float))[0]))
            rise = gw * rise + (1 - gw) * pg
            tag += "+GBM"
        if _SETTLED is not None:
            Xs, _ = N.matrix([{"f": f}], med, names)
            if float(_SETTLED.predict_proba(_NP.asarray(Xs, float))[0][1]) >= N.SETTLED_TH:
                rise, tag = 0.0, tag + "|已见顶"
        fin = msf + rise
        fin_raw = fin                                  # 影子对照用: 订正前的值
        b = _recent_bias(stn)
        if b is not None:
            fin -= RESID_ALPHA * b
            tag += f"|近期偏差{-RESID_ALPHA * b:+.2f}"

        p90 = None
        if args.p90:
            if mq:
                # 分位数回归优先。序贯分类的 PMF 是线性概率模型，尾部标定差
                p90 = msf + max(0.0, T.ridge_pred(mq, X)[0])
            elif mo:
                p90 = msf + N.rise_quantile(N.rise_pmf(mo, X), 0.90)[0]
            elif not warned_p90:
                print(f"[warn] {args.model} 里没有序贯分类模型，无法给 P90。"
                      f"用新版 train_nowcast.py 重训即可", file=sys.stderr)
                warned_p90 = True

        # 模式特征缺失会让预报静默退化成「纯实况」水平（9 时 MAE 0.87 -> 1.37）。
        # 只在 stderr 警告一句太容易漏，这里逐站打上可见标记
        need_nwp = [n for n in names if N.is_nwp_feat(n)]
        got = sum(1 for n in need_nwp if f.get(n) is not None)
        degraded = need_nwp and got < len(need_nwp) * 0.5

        note = ""
        if degraded:
            note = f"⚠模式特征缺 {len(need_nwp)-got}/{len(need_nwp)}，精度下降  "
        if args.verbose:
            note += (f"{tag} | 气候升温 {cr:.1f}" if cr is not None else tag)
            if f.get("cld_mean_am") is not None:
                note += f" | 上午云量 {f['cld_mean_am']:.2f}"
            if f.get("nwp_tmax") is not None:
                note += f" | GFS {f['nwp_tmax']:.1f}"
                if f.get("nwp_cloud_peak") is not None:
                    note += f" 云{f['nwp_cloud_peak']:.0f}%"
            if f.get("ens_spread") is not None:
                note += f" | 模式离散 {f['ens_spread']:.1f}"
            if f.get("ts_am"):
                note += " | 上午有雷暴"
        elif f.get("ts_am"):
            note += "上午有雷暴"
        # 点预报与 P90 是两个独立拟合的模型，逻辑上不保证 P90 >= 点预报。
        # 回测里 0/2184 出现过，但那是运气，不是保证 —— 硬约束住
        if p90 is not None:
            p90 = max(p90, fin)
        # 迟滞: 只有连续值离上次报的整数超过 0.5+HYST 才改口
        prev = _state.get(stn)
        if cutoff in HYST_CUTOFFS and prev is not None and abs(fin - prev) <= 0.5 + HYST:
            shown = int(prev)
        else:
            shown = int(round(fin))
        _state[stn] = shown

        # 把握: 判据是「预计还要升多少」。回测（3494 站日）实测的完全命中率 ——
        #   剩余<0.15  9时88% 11时83% 13时93%（覆盖 3%/4%/15%）
        #   剩余<0.5   9时75% 11时75% 13时79%（覆盖 6%/11%/43%）
        #   剩余>=0.5  不管几点都只有 31-50%
        # 命中率几乎不随时次变化，变的只有覆盖率 —— 预报不会随时间"变准"，
        # 时间只是把不确定的情况物理性地消解掉。
        # 「把握」改印**超出概率**，不再用「已定/大致」这种安慰性的词。
        # 2026-08-12 实测（15 个月回测）: 只要 rise<0.5，报错时 **100% 是报低**
        #   15 时 <0.15  命中 96%  报低 4%  报高 0%
        #   15 时 .15-.5 命中 76%  报低 24% 报高 0%
        #   14 时 .15-.5 命中 72%  报低 28% 报高 0%
        #   13 时 .15-.5 命中 69%  报低 31% 报高 0%
        # 这是结构性的: 「已达」是硬底，模型说「几乎不会再升」时唯一的出错
        # 方式就是它还是升了。所以这一列印「↑N%」= 有 N% 概率比这个数更高、
        # 不会更低。生产上 12 条「已定」报错的记录也是 12/12 全部报低。
        #
        # rise>=0.5 时误差是双向的（14 时报高占 41%），不印 —— 那时以「预报」
        # 和「不排除」两列为准。
        conf = _exceed(cutoff, rise)
        eh = _exp_hit(cutoff, rise)
        ehs = f"{eh:>7.0%}" if eh is not None else ""
        _ai = _alt_int(f, msf, stn)
        agr = "" if _ai is None else ("一致" if _ai == shown else f"分歧{_ai}")
        pc = (f"{round(p90):>8}" if p90 is not None
              else (f"{'--':>8}" if args.p90 else ""))
        print(f"  {stn} {N.NAMES.get(stn,''):<9}{shown:>7}{pc}{msf:>7.0f}"
              f"{rise:>+10.1f}{ehs}{conf:>7}{agr:>7}{_otag}   {note}")
        rows_out.append((stn, shown, msf, rise))
        _pp = _peak_prob(f, med, names)
        if _pp is not None:
            _peaks.append((stn, *_pp))
        if cutoff == RESID_CUTOFF:
            _raw_today[stn] = round(fin_raw, 4)        # **订正前**的值，见落盘处
            if b is not None and round(fin_raw) != shown:
                _shadow.append((stn, round(fin_raw), shown))
        _e = _edge(stn, rise, shown)
        if _e and _e[4] >= 0.25:
            _edges.append((stn, shown, _e))
        _adv, _cov = _bucket_advice(cutoff, rise, shown)
        if _adv:
            _advice.append((stn, _adv, _cov, rise))

    if _edges:
        print(f"\n── 高优势清单（我们有把握、而隔夜预报大概率错）")
        print(f"  {'站点':<14}{'我们':>5}{'隔夜':>6}{'一档':>7}{'两档':>7}{'隔夜命中':>9}{'优势':>7}")
        for stn, sh, (d1, o1, o2, dh, adv) in sorted(_edges, key=lambda x: -x[2][4]):
            print(f"  {stn} {N.NAMES.get(stn,''):<9}{sh:>5}{d1:>6}{o1:>7.0%}{o2:>7.0%}"
                  f"{dh:>9.0%}{adv:>+7.0%}")
        print(f"  盘口若锚定隔夜预报，这些站的价格最可能是错的。"
              f"没列出的站要么与隔夜一致（无优势），要么我们自己也没把握。")

    if _advice:
        print(f"\n── 档位配置建议（盘口分档时用；**不改上面的预报值**）")
        print(f"  {'站点':<14}{'买哪几档':<18}{'历史覆盖':<12}备注")
        for stn, adv, cov, rise in _advice:
            note = ("已定，一档足够" if "," not in adv else
                    ("剩余升幅大，可考虑三档" if "三档" in adv else "误差偏低，第二档往上买"))
            print(f"  {stn} {N.NAMES.get(stn,''):<9}{adv:<18}{cov:<12}{note}")
        print(f"  依据: 12 时实测 只买点预报 47-48%，买「点预报+上一档」71-73%，"
              f"±1 三档 90%。第二档往上买比往下买多 7 个百分点。")

    if stale:
        print(f"\n[warn] 实况滞后（截止 {cutoff} 时，标 ! 的站）:", file=sys.stderr)
        for stn, s, last, lag in sorted(stale, key=lambda x: -x[3]):
            print(f"       {stn} {N.NAMES.get(stn,'')} 最新 {last} 时（{s}），"
                  f"滞后 {lag} 小时", file=sys.stderr)
        print(f"       滞后 >2 小时的站会被 morning() 挡掉出 --，1-2 小时照报，"
              f"抓不住当下升温趋势", file=sys.stderr)

    if _peaks:
        print(f"\n── 见顶时刻概率（**不改上面的预报值**，只说今天几点可能到顶）")
        print(f"  {'站点':<16}{'早<13时':>9}{'正常13-14':>11}{'晚>=15时':>10}"
              f"{'晚·校准':>9}   提示")
        for _s, _e, _n, _l, _c in sorted(_peaks, key=lambda x: -x[3]):
            _use = _c if _c is not None else _l
            _hint = ("⚠ 大概率拖到下午晚些，15 时的数仍可能被顶掉" if _use >= 0.5
                     else ("留意晚见顶" if _use >= 0.33 else
                           ("基本能早早定下来" if _e >= 0.5 else "")))
            print(f"  {_s} {N.NAMES.get(_s, '')[:6]:<10}{_e:>9.0%}{_n:>11.0%}"
                  f"{_l:>10.0%}{_use:>9.0%}   {_hint}")
        print(f"  分档: 早=<13 时见顶 / 正常=13-14 时 / 晚=>=15 时。"
              f"「晚·校准」是按训练集经验频率回填的值，比原始概率准 —— "
              f"实测按它五等分，实际晚见顶率 10%/20%/33%/42%/61%，单调无反转。")
        print(f"  站内 AUC 平均 0.678（济南/成都/青岛/重庆/武汉 0.73~0.75 最好，"
              f"北京/深圳 0.58 几乎只能给基础率）。各站基础率差很多: "
              f"晚见顶占比 上海 6% / 青岛 13% / 北京 30% / 武汉 43% / 成都 53%。")

    if _shadow:
        print(f"\n── 近期偏差订正改掉的站（前 {RESID_K} 天平均残差 × {RESID_ALPHA}）")
        print(f"  {'站点':<16}{'订正前':>8}{'订正后':>8}")
        for _s, _a, _b2 in _shadow:
            print(f"  {_s} {N.NAMES.get(_s, '')[:6]:<10}{_a:>8}{_b2:>8}")
        print(f"  依据: 双向验证各 +0.95pt（P=91.0%/89.2%），"
              f"k 从 3 到 30、α 从 0.2 到 0.6 两半全部为正。**只在 9 时**，"
              f"10-15 时实测不稳或为负。")

    if args.compare and os.path.exists(args.compare):
        print(f"\n（另见 {args.compare} 的 D+1 结果，可并排比较）")
    try:
        json.dump({"date": tgt.isoformat(), "cutoff": cutoff, "last": _state},
                  open(_st_path, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception as e:                            # noqa: BLE001
        print(f"[warn] 迟滞状态没写成 ({e})，下一轮会退回普通四舍五入", file=sys.stderr)

    # 9 时**订正前**的未取整预报值落盘，供后续日子算近期残差。
    # 存订正前的值 —— 否则订正会自我叠加（今天减了，明天算残差时又把减过的
    # 值当预报，等于连着减两次）。只保留最近 90 天。
    if cutoff == RESID_CUTOFF and _raw_today:
        try:
            _h = {}
            if os.path.exists(_HIST_PATH):
                _h = json.load(open(_HIST_PATH, encoding="utf-8"))
            _h[tgt.isoformat()] = {**_h.get(tgt.isoformat(), {}), **_raw_today}
            for _d in sorted(_h)[:-90]:
                _h.pop(_d, None)
            json.dump(_h, open(_HIST_PATH, "w", encoding="utf-8"), ensure_ascii=False)
        except Exception as e:                        # noqa: BLE001
            print(f"[warn] 9 时预报历史没写成 ({e})，近期偏差订正下轮会跳过",
                  file=sys.stderr)

    if any(r[3] < 0.5 for r in rows_out):
        print("\n「更高?」= 有多大概率比报出去的这个数**更高**。"
              "实测只要标了这一列，错的时候 100% 是报低、从来没报高过 —— "
              "「已达」是硬底，模型说不再升时唯一的出错方式就是它还是升了。"
              "\n没标的（预计再升 >= 0.5 度）误差是双向的，以「预报」「不排除」两列为准。")
        print(f""
              f"「已定」实测完全命中 83~93%、「大致」75~79%，"
              f"没标的不管几点都只有 31~50%。")
        print("**但「已定」不等于不再改。** 回测里标了「已定」之后仍有 ~22% "
              "会在后面的时次改动，平均改 1.2~1.5 度 —— "
              "而那些改动 **100% 是改对方向的，一次都没改坏过**。")
        print("所以口径是: 永远以最新一轮为准。锁死 9 时的值最终只有 75% 命中，"
              "跟着更新是 94%。")

    print(f"\n**要报的数就是「预报」这一列。** 已达 = 截止时刻当日累计最高。")
    if args.p90:
        print("「不排除」不是备选预报值，别拿它当每天要报的那个数 ——")
        print("  它是升幅的条件 P90，实测覆盖 94%（比目标 90% 还保守），"
              "作用只有一个: 极端日心里有数、别被打懵。")
        print("  15 个月回测（11064 条）实测: 全用「预报」MAE 0.730 / ±1℃ 86%，"
              "全用「不排除」1.325 / 64%。")
        print("  按站、按时次、按见顶类型穷举 72 个子集，只有 6 个（8%）"
              "用「不排除」更好 —— 是多重比较的产物，不是规律。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

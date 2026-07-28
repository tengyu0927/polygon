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
import json
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import train_mos as T                      # noqa: E402
import train_nowcast as N                  # noqa: E402


def fetch_nwp(stations, tgt, gfs_model="gfs_global"):
    """模型若含 NWP 特征，取当天的 GFS 固定时效预报（与训练同口径）。"""
    import build_mos_dataset as B
    import predict_mos as P
    conn, n = P.fetch_window(stations, tgt, tgt, gfs_model, B.VARS)
    daily = B.daily_features(conn, 1)
    return {stn: daily.get((stn, tgt.isoformat()), {}) for stn in stations}


def fetch_m2(stations, tgt, models):
    """追加模式的当天预报。models 的顺序必须与训练时 --nwp-csv2 的顺序一致，
    否则 m2_/m3_ 各列会对错模式，模型系数全部错位。"""
    out = []
    for mdl in models:
        try:
            daily = fetch_nwp(stations, tgt, mdl)
        except Exception as e:
            print(f"[warn] {mdl} 取数失败: {e}", file=sys.stderr)
            daily = {}
        out.append({stn: {k: daily.get(stn, {}).get(c)
                          for k, c in N.M2_COLS.items()} for stn in stations})
    return out

UTC = timezone.utc
CST = timezone(timedelta(hours=8))
AWC = "https://aviationweather.gov/api/data/metar"
UA = "nowcast/1.0 (station Tmax research)"


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
        m2 = fetch_m2(stations, tgt, mdls)

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
    print(f"\n  {'站点':<14}{'预报':>7}{p90_col}{'已达':>7}{'预计再升':>10}   备注")

    names, med = spec["names"], spec["median"]
    warned_p90 = False
    rows_out = []
    for stn in stations:
        hrs = days.get((stn, tgt.isoformat()))
        if not hrs:
            print(f"  {stn} {N.NAMES.get(stn,''):<9}{'--':>7}{'--':>7}{'--':>10}   无今日观测")
            continue
        o = N.morning(hrs, cutoff)
        if o is None:
            n = len([h for h in hrs if h <= cutoff])
            print(f"  {stn} {N.NAMES.get(stn,''):<9}{'--':>7}{'--':>7}{'--':>10}   "
                  f"截止前仅 {n} 条观测，不足")
            continue

        ph = days.get((stn, prv.isoformat()))
        mo = tgt.month
        cr = spec["clim_rise"].get(f"{stn}|{mo}")
        cp = spec["clim_peak"].get(f"{stn}|{mo}")
        prev = (max(v["t"] for v in ph.values()) if ph else None, cr)
        f, msf = N.build_feats(o, cutoff, prev, cr, cp,
                               tgt.timetuple().tm_yday, nwp.get(stn))

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
        fin = msf + rise

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
        pc = (f"{round(p90):>8}" if p90 is not None
              else (f"{'--':>8}" if args.p90 else ""))
        print(f"  {stn} {N.NAMES.get(stn,''):<9}{round(fin):>7}{pc}{msf:>7.0f}"
              f"{rise:>+10.1f}   {note}")
        rows_out.append((stn, round(fin), msf, rise))

    if args.compare and os.path.exists(args.compare):
        print(f"\n（另见 {args.compare} 的 D+1 结果，可并排比较）")
    print(f"\n已达 = 截止时刻当日累计最高；预报值已取整（上线口径）。")
    if args.p90:
        print("不排除 = 升幅的条件 P90（分位数回归）。回测覆盖率 93-94%，"
              "平均比点预报高 1.0-1.3℃。它不改点预报的准确率，作用是极端日不漏报。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

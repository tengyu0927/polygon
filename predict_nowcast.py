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
import tablefmt as F                       # noqa: E402  显示宽度对齐（中文双宽）
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

# ⚠「16 点后见顶风险偏高」的门槛（校准后概率）。**只有这一处定义** ——
# 见顶风险表和「档位配置建议」的抬档例外用的必须是同一个数，两边各写一个
# 就会出现「表里标了 ⚠、档位却还说买一档」。
# 阈值 20%: 每天亮 1.7 个站、精度 29%、召回 47%、漏 0.5 个/天。
# 放到 30% 只亮 0.9 个站但召回掉到 32%；40% 只剩 0.5 个站、召回 18%。
PEAK_WARN_TH = 0.20

# 「基本能早早定下来」的门槛（P(见顶<13时)，**原始概率，没有校准表**）。
#
# 2026-08-22 从 0.5 提到 0.8。原门槛太松 —— 直接读线上判别器的校准表
# （cal_e，就是按预测概率十档的样本外实测频率）就能看出来:
#      9 时  最高一档实测 84.0%，第九档 61.7%，第八档 46.1%
#     12 时  最高一档实测 88.6%，第九档 67.6%，第八档 47.1%
#     15 时  最高一档实测 99.6%，第九档 95.9%，第八档 64.7%
# 门槛 0.5 会把第八、九档一起放进来（约 20% 的站日，合并精度只有六成上下）；
# 门槛 0.8 基本只留最高那一档（约 10% 的站日，精度 84~99%）。
# 独立的样本外评估（4 折交叉）给出同样的结论: 9 时 61.7% -> 83.2%，
# 15 时 93.1% -> 95.7%，且标了之后真拖到 >=16 见顶的从 4.4~6.1% 降到 1.2~3.8%。
# 分站看，门槛 0.5 时上海只标对 74.0%，而上海本来就有 64.9% 的日子是早见顶
# （净提升 +9.1pt，几乎在复读该站气候）；提到 0.8 后上海 90.4%（+25.6pt），
# 十个站的净提升全部 >= +25.6pt、精度全部 >= 85%。
#
# 这一维从 2026-08-22 起也走经验频率回填（cal_e）。**还没按新版重训的时次
# 没有那张表**，那时退回分类器原始输出 —— 见 _peak_prob。
PEAK_EARLY_TH = 0.80


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
    # drct（风向）是 2026-08-22 加的 —— 训练端 load_hourly 一直在取，预测端
    # 这条 SELECT 里没有。check_consistency 的 [1] 逐列比对当场把它抓出来了
    # （KeyError: 'drct'）。列顺序改了就要同步改下面 r[9]。
    q = (f"SELECT station, {tcol}, temp_c, {get('dewp_c')}, {get('rh')}, "
         f"{get('wspd_ms')}, {get('pres_hpa')}, {get('skyc1')}, {get('wxcodes')}, "
         f"{get('drct')} "
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
                  "cld": N.cloud_frac(r[7]), "ts": tsf, "ra": raf, "obsc": obf,
                  "drct": None if r[9] is None else float(r[9])}
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
        wd = m.get("wdir")
        # AWC 的 wdir 在静风时给 "VRB" 或 0，非数字一律当缺测
        try:
            wd = float(wd)
        except (TypeError, ValueError):
            wd = None
        d[h] = {"t": float(t), "dewp": dp,
                "rh": _rh(float(t), dp),
                "wspd": None if ws is None else float(ws) * 0.514444,
                "pres": p, "cld": N.cloud_frac(cov),
                "ts": tsf, "ra": raf, "obsc": obf, "drct": wd}
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
    # 列宽一律按**显示列**算，表头和数据行共用同一组常量。别再用 f-string 的
    # `:<14`/`:>7` —— 那按字符数补，汉字占两列，每张表都会歪。见 tablefmt.py。
    W_STN, W_VAL, W_P90, W_RISE = 16, 7, 8, 10
    W_HIT, W_UP, W_RDY, W_AGR, W_OBS = 9, 7, 8, 7, 9
    # 「预期命中」整列的有无必须与数据格同一个判据，否则表头有列、格子空着又歪。
    _has_hit = os.path.exists(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hit_table.json"))
    print("\n  " + F.L("站点", W_STN) + F.R("预报", W_VAL)
          + (F.R("不排除", W_P90) if args.p90 else "")
          + F.R("已达", W_VAL) + F.R("预计再升", W_RISE)
          + (F.R("预期命中", W_HIT) if _has_hit else "")
          + F.R("更高?", W_UP) + F.R("可下手", W_RDY) + F.R("一致?", W_AGR)
          + F.R("实况", W_OBS) + "   备注")

    # 迟滞用的上一轮报出去的整数。报的是整数，连续值在 x.5 附近抖 0.02 度就会
    # 让整数翻个个儿 —— 15 个月回测里 61% 的逐小时改动是这种空转（动出去又回来）。
    # 规则: 上次报了 k 就一直报 k，直到连续值离 k 超过 0.5+HYST 才改。
    #
    # 只在 9-12 时启用。回测（完全命中率口径，3494 站日）:
    #   全时段 0.2   空转 -52%  命中 54.43% -> 53.59%（-0.84pt）
    #   只 9-12 0.2  空转 -32%  命中 54.43% -> 54.34%（-0.09pt）  <- 采用
    # 损失全部发生在 13-15 时: 那时值在收敛，压住真实变动就是纯损失。
    HYST, HYST_CUTOFFS = 0.20, (9, 10, 11, 12)

    # 「可下手」的门槛。见下面那一大段实测。
    READY_TH = 0.95
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

    # numpy 是**所有** pkl 边文件（gbm / peak / settled）共用的前提，必须独立
    # 拿，不能挂在 GBM 那一段里。2026-08-22 修: 原来 `_NP` 只在 gbm.pkl 读成
    # 之后才绑定，而 15 时的 nowcast_late.json **没有 GBM 边文件** —— 于是
    # `_PEAK` 的 `and _NP is not None` 恒假，训好落盘的 nowcast_late.json.peak.pkl
    # （2.1MB）生产一次都没读过，15 时既不出见顶风险表也拿不到 p_late。
    try:
        import numpy as _NP                            # noqa: N813
    except ImportError:
        _NP = None

    # GBM 边文件。缺 sklearn / 缺文件 / 读不出来都自动降级成纯线性，不报错 ——
    # 与 predict_mos.py 同一约定（<model>.gbm.pkl）。
    _GBM = {}
    _gp = args.model + ".gbm.pkl"
    if os.path.exists(_gp) and _NP is not None:
        try:
            import pickle
            _GBM = {int(k): v for k, v in pickle.load(open(_gp, "rb")).items()}
        except Exception as e:                          # noqa: BLE001
            # 只降级 GBM 本身。**不要在这里清掉 _NP** —— 见顶/已见顶判别器
            # 也靠它，GBM 坏了不该把那两个一起拖下水。
            print(f"[warn] GBM 没读成（{str(e)[:60]}），本轮只用线性", file=sys.stderr)
            _GBM = {}

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

    def _exceed(cut, rise, drop=None):
        """「更高?」= 有多大概率比报出去的这个数更高。

        **优先查「档 × 已降多少」**，缺格才回退到只按档。2026-08-20 加这一维:
        用户指出深圳 11 时 31 度、12 时 30、13 时 29，13 时起报已达仍是 31，
        于是印「不排除 32」——「明明在降温还在往上报」。实测确实错得离谱:
        13 时「已降>=2 度」的站日实际只有 5.0% 会超过已达，而原表印 24.4%；
        14 时印 20.1%、实际 1.8%。方向是系统性的（没降的组反而低估 5~7pt）。
        根因是 `sofar_minus_now` 在 REGIME_FEATS 里，而 9/12/13 时不开 REGIME。
        加这一维后前半建后半验 +4.55%，四种切分 × 五个有样本时次 20/20 全正。
        """
        if not _exc or rise >= _exc["edges"][-1]:
            return ""
        i = next((k for k, e in enumerate(_exc["edges"]) if rise < e), None)
        de = _exc.get("drop_edges")
        if drop is not None and de and _exc.get("per_drop"):
            di = next((k for k, e in enumerate(de) if drop < e), len(de))
            c3 = _exc["per_drop"].get(f"{cut}|{i}|{di}")
            if c3:
                return f"↑{c3[0]:.0%}"
        c = _exc["cells"].get(f"{cut}|{i}")
        return "" if not c else f"↑{c[0]:.0%}"

    def _exceed_p(cut, rise, drop=None):
        """同 _exceed，但返回概率本身（None = 查不到）。给 P90 的自洽约束用。"""
        if not _exc or rise >= _exc["edges"][-1]:
            return None
        i = next((k for k, e in enumerate(_exc["edges"]) if rise < e), None)
        de = _exc.get("drop_edges")
        if drop is not None and de and _exc.get("per_drop"):
            di = next((k for k, e in enumerate(de) if drop < e), len(de))
            c3 = _exc["per_drop"].get(f"{cut}|{i}|{di}")
            if c3:
                return c3[0]
        c = _exc["cells"].get(f"{cut}|{i}")
        return c[0] if c else None
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

    def _bucket_advice(cut, rise, shown, p_late=None):
        """返回 (建议档位串, 历史覆盖率串, 是否因晚见顶风险被抬档)。

        样本不足就返回空。**不改任何预报值**，只改买几档。
        """
        if not _BK:
            return "", "", False
        i = next((k for k, e in enumerate(_BK["edges"]) if rise < e), len(_BK["edges"]))
        c = _BK["cells"].get(f"{cut}|{i}")
        if not c:
            return "", "", False
        one, two, three = c[0], c[1], c[2]
        # 一档已经够好（>=85%）就买一档；否则两档；两档还不到 70% 就提三档
        if one >= 0.85:
            # **例外: ⚠ 晚见顶风险的站不许只买一档。** 2026-08-22 加。
            #
            # 这张表只按 (时次 x 剩余升幅) 分格，15 时剩余升幅几乎恒为 0，
            # 于是整格覆盖 88%、一律建议一档 —— 而那 88% 是拿全体站日平均
            # 出来的。拆开看: 15 时那一格里 ⚠ 标了的那 1/4 站日，一档只有
            # 72.7~81.4%，剩下 3/4 才是 90%+。**误差全集中在晚见顶那批**
            # （15 时按实际见顶档分: 早 99.8% / 正常 98.3% / 晚 6.3%，
            # 而晚见顶时 93.7% 是报低、83.3% 恰好低 1 度、报高 0%）。
            #
            # 双向验证（标准窗口对半切，32254 行，概率走分块交叉折）:
            #   前半定→后半验  受影响 381 行  81.4% -> 99.0%  (+17.6pt)
            #   后半定→前半验  受影响 194 行  72.7% -> 96.4%  (+23.7pt)
            # 全样本覆盖 88.32%->88.73% / 86.97%->87.26%，平均买的档数只多
            # 0.012~0.023 —— 受影响的只有 1~2% 的行，全部落在 15 时。
            #
            # **不做成表的第三维。** 试过（p_late 三分位当第三维重建整张表）:
            # 两个方向的覆盖-成本前沿与现行交叉、不占优，是分格变细后每格
            # 样本不够的典型症状。只在「本来要买一档」这一处做例外才稳。
            if p_late is not None and p_late >= PEAK_WARN_TH:
                return f"{shown},{shown+1}", f"{two:.0%}", True
            return f"{shown}", f"{one:.0%}", False
        if two >= 0.70 or three - two < 0.10:
            return f"{shown},{shown+1}", f"{two:.0%}", False
        return f"{shown},{shown+1}｜或 ±1 三档", f"{two:.0%}｜{three:.0%}", False

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
        """返回 (早, 正常, 晚, 校准后的晚)。缺模型/缺 sklearn 返回 None。

        「早」这一维**优先返回校准值**（edges_e/cal_e，2026-08-22 加）。
        重训之前的 pkl 里没有这两个键，那时退回原始概率 —— 那个数在早时次
        过度自信 10~14 个百分点，见 train_nowcast.fit_peak_prob。
        """
        if _PEAK is None:
            return None
        Xp, _ = N.matrix([{"f": f}], med_, names_)
        p = _PEAK["clf"].predict_proba(_NP.asarray(Xp, float))[0]

        def _lookup(v, ek, ck):
            eg, cl = _PEAK.get(ek), _PEAK.get(ck)
            if not eg or not cl:
                return None
            i = next((k for k, e in enumerate(eg) if v < e), len(eg))
            return cl[i] if i < len(cl) else None

        pe = _lookup(p[0], "edges_e", "cal_e")
        return (float(pe if pe is not None else p[0]), float(p[1]), float(p[2]),
                _lookup(p[2], "edges", "cal"))

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

    def _exp_hit(cut, rise, p_late=None):
        """预期命中率。**三维（时次 × 升幅 × 晚见顶概率）优先，缺格自动回退二维。**

        2026-08-19 加第三维: 同样的剩余升幅，今天这个站「早早定下来」还是
        「拖到 16 点还在升」，把握完全不同（15 时按晚见顶概率五等分，命中率
        从 99.6% 到 45.7%）。查表 Brier 改善 +3.24%，三种切分 × 三个时次
        9/9 全为正。**必须用本时次自己判的概率**，用 9 时那个只有 +0.21%。
        """
        if not _hit:
            return None
        b = 0
        for i, e in enumerate(_hit.get("edges", []) + [float("inf")]):
            if rise <= e:
                b = i
                break
        # peak_edges 是 {时次: [边界]} —— 逐时次按分位数算的，而且只有 13/14 时
        # 有（其余时次四种切分下正负混杂，见 build_hit_table.PEAK_CUTOFFS）。
        _pe = (_hit.get("peak_edges") or {}).get(str(cut))
        if p_late is not None and _pe and _hit.get("cells3"):
            pb = next((i for i, e in enumerate(_pe) if p_late <= e), len(_pe))
            v3 = _hit["cells3"].get(f"{cut}|{b}|{pb}")
            if v3:
                return v3[0]
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
            return s, None, F.R(f"{s} --", W_OBS)
        last = max(hh)
        lag = cutoff - last
        return s, lag, F.R(f"{s} {last}时{'!' if lag >= 1 else ''}", W_OBS)
    # 出不了数的行也要把每一列都占满，否则「实况」「备注」会顶到前面去
    _dash = (F.R("--", W_VAL) + (F.R("--", W_P90) if args.p90 else "")
             + F.R("--", W_VAL) + F.R("--", W_RISE)
             + (F.R("--", W_HIT) if _has_hit else "")
             + F.R("--", W_UP) + F.R("--", W_RDY) + F.R("--", W_AGR))
    stale = []
    for stn in stations:
        hrs = days.get((stn, tgt.isoformat()))
        _s, _lag, _otag = _obs_tag(stn, hrs)
        if _lag is not None and _lag >= 1:
            stale.append((stn, _s, cutoff - _lag, _lag))
        if not hrs:
            print("  " + F.L(f"{stn} {N.NAMES.get(stn,'')}", W_STN) + _dash
                  + _otag + "   无今日观测")
            continue
        o = N.morning(hrs, cutoff)
        if o is None:
            n = len([h for h in hrs if h <= cutoff])
            print("  " + F.L(f"{stn} {N.NAMES.get(stn,'')}", W_STN) + _dash
                  + _otag + f"   截止前仅 {n} 条观测，不足")
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
        _sp = None
        if _SETTLED is not None:
            Xs, _ = N.matrix([{"f": f}], med, names)
            _sp = float(_SETTLED.predict_proba(_NP.asarray(Xs, float))[0][1])
            if _sp >= N.SETTLED_TH:
                if N.settled_blocked(f):
                    tag += "|已见顶(模式不认，未覆盖)"
                else:
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
        _pp = _peak_prob(f, med, names)   # 「预期命中」要用它当第三维，须先算
        # 校准后的晚见顶概率。见顶风险表、「不排除」下限、档位建议**共用这一个数**
        _pl = None if _pp is None else (_pp[3] if _pp[3] is not None else _pp[2])
        _warn = _pl is not None and _pl >= PEAK_WARN_TH
        # sofar_minus_now 由 build_feats 一律计算（不受 flag 控制），所以
        # 9/12/13 时虽然模型里没这一项，这里照样拿得到
        conf = _exceed(cutoff, rise, f.get("sofar_minus_now"))
        eh = _exp_hit(cutoff, rise, _pp[2] if _pp is not None else None)
        ehs = (F.R(f"{eh:.0%}" if eh is not None else "", W_HIT)
               if _has_hit else "")
        _ai = _alt_int(f, msf, stn)
        agr = "" if _ai is None else ("一致" if _ai == shown else f"分歧{_ai}")
        # 「不排除」与「更高?」必须自洽: **超过已达的概率 < 10%，90 分位就不该
        # 高于已达值**。2026-08-20 加 —— 用户指出深圳降温日仍印「不排除 32」。
        # 实测 12/13 时有 92.7~97.9% 的站日「不排除」都高于已达，几乎逢报必高一档；
        # 而 P90 覆盖率是 94.6~96.5%（目标 90%），本来就偏保守报太高。
        # 压之后覆盖率 93.5~95.4%，仍全部高于 90%，不存在压过头；被压的行占
        # 5.1~14.3%（12~15 时），9-11 时一行不压（那三档 rise 几乎都 >=0.5，
        # 「更高?」本来就不印）。压到 max(已达, 点预报) —— 不能低于点预报，
        # 否则「不排除」比「预报」还小。
        _pe = _exceed_p(cutoff, rise, f.get("sofar_minus_now"))
        if (p90 is not None and _pe is not None and _pe < 0.10
                and round(p90) > round(msf)):
            p90 = max(round(msf), shown)
        # **⚠ 晚见顶风险的站不许印「不排除 == 预报」。** 2026-08-22 加。
        #
        # 「不排除」是 90 分位，契约是被实际值超过 <=10%。按分组实测（32254 行，
        # 只看印出来恰好等于「预报」的那些行）:
        #     非⚠   5567 行   实际 > 预报  7.0%   —— 在契约内
        #     ⚠      865 行   实际 > 预报 20.3%   —— **超契约两倍**
        #                     （14 时 27.8% / 11 时 27.3% / 15 时 14.7%）
        # 也就是屏幕上写着「没有上行空间」，而这批站每五次有一次真的更高。
        # 而且方向是死的: 15 时按实际见顶档分，晚见顶时 93.7% 是报低、
        # 83.3% 恰好低 1 度、报高 0% —— 抬一档正好覆盖主峰。
        #
        # 双向验证（标准窗口对半切）: ⚠ 组覆盖 92.3%->94.0% / 91.3%->93.3%，
        # 代价是 8.0~8.7% 的 ⚠ 行被白抬一度（实际没超过预报）。
        # **只抬到 预报+1，不放开上面那条压制** —— 那条是 2026-08-20 为
        # 「深圳降温日还在往上报」加的，非⚠ 组它标定得很好（7.0%），不能动。
        if p90 is not None and _warn and round(p90) <= shown:
            p90 = shown + 1
        pc = (F.R(round(p90) if p90 is not None else "--", W_P90)
              if args.p90 else "")
        # 「可下手」—— 这个站现在就能定下来吗。**不改任何预报值。**
        #
        # 把握 = max(已见顶概率, 1 - 超越概率)。两条路都指向同一件事:
        # 「今天这个站的最高温已经出现了」。已见顶判别器从特征判，超越概率
        # 从查表判（时次 x 剩余升幅 x 已降幅度 x 站 x 季），互为补充。
        #
        # 门槛 0.95。这个数是量出来的 —— 按「逐档看，头一次把握够就下手、
        # 之后不改」的规则回测 4608 站日 / 476 天:
        #
        #     门槛   平均出手时刻   完全命中   十站≥9站对   十站全对
        #     0.90     14.50 时    90.4%     76.5%      40.8%
        #     0.93     14.91 时    94.0%     88.9%      56.5%
        #     0.95     15.99 时    98.9%     99.8%      89.9%   <- 用这个
        #     0.97     16.05 时    99.1%     99.8%      91.4%
        #     对照: 全部站死等 16 时   16.00 时  96.7%  95.6%  72.9%
        #     对照: 全部站死等 17 时   17.00 时  99.5% 100.0%  95.0%
        #
        # 平均出手 15.99 时与「死等 16 时」几乎同时，但 >=9站对 95.6% -> 99.8%、
        # 全对 72.9% -> 89.9% —— 差别在于**每个站等的长短不一样**: 上海平均
        # 15.3 时、成都 16.4 时。成都是十站里最晚也最弱的（97.3%），规则自动
        # 让它多等，不必人工分站配置。
        #
        # **为什么是 min 不是 max。** 两个来源会打架（浦东那种「已见顶 91%、
        # 但超越概率还有 21%」）。2026-08-22 首版用 max，只看了整体数就上线了；
        # 按站拆开才发现它在上海漏水 —— max 让上海 13.68 时就出手，命中只有
        # 93.9%，是十站里最弱的。根因: `settled_p` 是全站合训的，对上海过度
        # 自信；而超越概率是按 站 x 季 x 已降 标定的。max 等于「哪个更乐观听哪个」。
        #
        # 四种合成方式，门槛都取 0.95（4608 站日 / 476 天）:
        #     规则           平均出手   命中    >=9站对   全对    最弱站
        #     max（首版）    15.45 时  97.4%   96.4%  78.2%  上海 93.9%
        #     只用 1-超越    15.95 时  98.8%   99.8%  88.9%  成都 97.3%
        #     均值           15.93 时  98.8%   99.8%  89.1%  成都 97.3%
        #     **min**        15.99 时  98.9%   99.8%  89.9%  成都 97.3%
        #
        # max 用 0.54 小时换掉 >=9站对 3.4pt、全对 11.7pt。对「十站至少九站对」
        # 这个目标是明显亏本。改 min。
        #
        # 门槛在 0.93 和 0.95 之间有个陡坎（超越概率查表的分档结构造成的）:
        #     min@0.93  15.51 时  96.2%  94.7%  70.0%
        #     min@0.95  15.99 时  98.9%  99.8%  89.9%   <- 用这个
        #     min@0.97  16.05 时  99.1%  99.8%  91.4%
        #
        # `_sp` 为 None 时（9/15 时没有已见顶判别器）退化成「只用 1-超越」。
        _rdy = min(_sp if _sp is not None else 1.0,
                   (1.0 - _pe) if _pe is not None else 0.0)
        rc = F.R("--" if _rdy <= 0 else
                 (f"✓ {_rdy:.0%}" if _rdy >= READY_TH else f"{_rdy:.0%}"), W_RDY)
        print("  " + F.L(f"{stn} {N.NAMES.get(stn,'')}", W_STN)
              + F.R(shown, W_VAL) + pc + F.R(f"{msf:.0f}", W_VAL)
              + F.R(f"{rise:+.1f}", W_RISE) + ehs + F.R(conf, W_UP) + rc
              + F.R(agr, W_AGR) + _otag + f"   {note}")
        rows_out.append((stn, shown, msf, rise))
        if _pp is not None:
            _peaks.append((stn, *_pp))
        if cutoff == RESID_CUTOFF:
            _raw_today[stn] = round(fin_raw, 4)        # **订正前**的值，见落盘处
            if b is not None and round(fin_raw) != shown:
                _shadow.append((stn, round(fin_raw), shown))
        _e = _edge(stn, rise, shown)
        if _e and _e[4] >= 0.25:
            _edges.append((stn, shown, _e))
        _adv, _cov, _forced = _bucket_advice(cutoff, rise, shown, _pl)
        if _adv:
            _advice.append((stn, _adv, _cov, rise, _forced))

    if _edges:
        print(f"\n── 高优势清单（我们有把握、而隔夜预报大概率错）")
        _EW = (16, 6, 6, 7, 7, 9, 7)
        print("  " + F.L("站点", _EW[0])
              + "".join(F.R(t, x) for t, x in
                        zip(("我们", "隔夜", "一档", "两档", "隔夜命中", "优势"),
                            _EW[1:])))
        for stn, sh, (d1, o1, o2, dh, adv) in sorted(_edges, key=lambda x: -x[2][4]):
            print("  " + F.L(f"{stn} {N.NAMES.get(stn,'')}", _EW[0])
                  + "".join(F.R(t, x) for t, x in
                            zip((sh, d1, f"{o1:.0%}", f"{o2:.0%}",
                                 f"{dh:.0%}", f"{adv:+.0%}"), _EW[1:])))
        print(f"  盘口若锚定隔夜预报，这些站的价格最可能是错的。"
              f"没列出的站要么与隔夜一致（无优势），要么我们自己也没把握。")

    if _advice:
        print(f"\n── 档位配置建议（盘口分档时用；**不改上面的预报值**）")
        _AW = (16, 20, 12)
        print("  " + F.L("站点", _AW[0]) + F.L("买哪几档", _AW[1])
              + F.L("历史覆盖", _AW[2]) + "备注")
        for stn, adv, cov, rise, forced in _advice:
            note = ("⚠ 晚见顶风险，第二档保底" if forced else
                    "已定，一档足够" if "," not in adv else
                    ("剩余升幅大，可考虑三档" if "三档" in adv else "误差偏低，第二档往上买"))
            print("  " + F.L(f"{stn} {N.NAMES.get(stn,'')}", _AW[0])
                  + F.L(adv, _AW[1]) + F.L(cov, _AW[2]) + note)
        print(f"  依据: 12 时实测 只买点预报 47-48%，买「点预报+上一档」71-73%，"
              f"±1 三档 90%。第二档往上买比往下买多 7 个百分点。")
        if any(a[4] for a in _advice):
            print(f"  标「⚠ 晚见顶风险」的站本来会建议一档 —— 那一格 88% 的覆盖是"
                  f"全体站日的平均值，⚠ 那 1/4 实测只有 73~81%。双向验证改两档后"
                  f"到 96.4%/99.0%（+23.7pt/+17.6pt），代价是平均多买 0.01~0.02 档。")

    if stale:
        print(f"\n[warn] 实况滞后（截止 {cutoff} 时，标 ! 的站）:", file=sys.stderr)
        for stn, s, last, lag in sorted(stale, key=lambda x: -x[3]):
            print(f"       {stn} {N.NAMES.get(stn,'')} 最新 {last} 时（{s}），"
                  f"滞后 {lag} 小时", file=sys.stderr)
        print(f"       滞后 >2 小时的站会被 morning() 挡掉出 --，1-2 小时照报，"
              f"抓不住当下升温趋势", file=sys.stderr)

    if _peaks:
        print(f"\n── 16 点后见顶的风险（**不改上面的预报值**）")
        _PW = (16, 9, 11, 10, 8, 7)
        print("  " + F.L("站点", _PW[0])
              + "".join(F.R(t, x) for t, x in
                        zip(("早<13时", "正常13-15", "晚>=16时", "校准后", "倍数"),
                            _PW[1:]))
              + "   提示")
        _BASE = 0.107          # 全站 >=16 见顶的基础率
        for _s, _e, _n, _l, _c in sorted(_peaks, key=lambda x: -x[3]):
            _use = _c if _c is not None else _l
            # 门槛见 PEAK_WARN_TH。**不说「极有可能」** —— 最高档实测也只有 38%。
            _hint = ("⚠ 16 点后见顶风险偏高" if _use >= PEAK_WARN_TH
                     else ("基本能早早定下来" if _e >= PEAK_EARLY_TH else ""))
            print("  " + F.L(f"{_s} {N.NAMES.get(_s, '')}", _PW[0])
                  + "".join(F.R(t, x) for t, x in
                            zip((f"{_e:.0%}", f"{_n:.0%}", f"{_l:.0%}",
                                 f"{_use:.0%}", f"{_use / _BASE:.1f}倍"), _PW[1:]))
                  + f"   {_hint}")
        print(f"  分档: 早=<13 时 / 正常=13-15 时 / **晚=>=16 时**。"
              f"15 点见顶不算晚 —— 15 时那轮（15:15 起报）看得到 15:00 的观测、"
              f"抓得住；16/17 点见顶时所有轮次都结束了，必然漏。")
        print(f"  全站基础率 {_BASE:.0%}（成都 29% / 重庆 23% / 武汉 19% / "
              f"上海 2% / 青岛 3%）。AUC 0.767，最高 5% 那档实测 37%、是平常 3.5 倍 "
              f"—— **不是「极有可能」，是「风险高 3 倍」**。")
        print(f"  阈值 20%: 每天平均亮 1.7 个站，其中 29% 会真的 >=16 见顶，"
              f"能抓住全部晚见顶站的 47%，**每天仍会漏掉约 0.5 个**。"
              f"每天真正 >=16 见顶的只有约 1.07 个站，漏是必然的。")
        print(f"  「基本能早早定下来」门槛 {PEAK_EARLY_TH:.0%}（原 50%，2026-08-22 提高）。"
              f"样本外实测精度: 9 时 83% / 12 时 91% / 15 时 96%，标了之后真的"
              f"拖到 >=16 见顶的只有 1.5~3.8%。**两条提示都不改预报值。**")

    if _shadow:
        print(f"\n── 近期偏差订正改掉的站（前 {RESID_K} 天平均残差 × {RESID_ALPHA}）")
        _RW = (16, 8, 8)
        print("  " + F.L("站点", _RW[0]) + F.R("订正前", _RW[1])
              + F.R("订正后", _RW[2]))
        for _s, _a, _b2 in _shadow:
            print("  " + F.L(f"{_s} {N.NAMES.get(_s, '')}", _RW[0])
                  + F.R(_a, _RW[1]) + F.R(_b2, _RW[2]))
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

    print("\n「可下手」= min(已见顶概率, 1-超越概率)，标 ✓ 表示 >= 95%，"
          "这个站现在就能定下来、后面不用再改。**它不改任何预报值。**")
    print("  用法: 每个站头一次见到 ✓ 就下手，之后不再动。回测 4608 站日/476 天，"
          "平均出手 15.99 时，完全命中 98.9%，十站里至少九站对的日子占 99.8%、"
          "\n  十站全对 89.9%。没到 ✓ 的站等下一轮，17 时那轮兜底（届时全部会到 ✓）。")

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

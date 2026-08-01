#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_nowcast.py — 临近预报: 用当天上午实况预报当日最高温

    python3 train_nowcast.py --db cn.sqlite --cutoffs 9 11 12 13
    python3 train_nowcast.py --db cn.sqlite --cutoffs 12 --dump nowcast.json
    python3 train_nowcast.py --db cn.sqlite --cutoffs 12 --coef

与 train_mos.py（D+1/D+2）的三点结构差异:

1. 目标量是「剩余升温」rise = Tmax - 截止时刻已达最高，不是 Tmax 本身。
   rise 恒 >= 0，且在 0 处有大量堆积（12 点截止时约 1/3 的日子已见顶）。
   直接回归 Tmax 会把这个尖峰抹平。

2. 两段式（hurdle）: 先用线性概率模型判「还会不会再升」，
   再在「会升」的子集上回归升幅，相乘得到期望值。
   预测再截断到 >= 0 —— Tmax >= 已达最高 是恒等式，模型不该违反。

3. 只用实况特征，可用 1995 年至今全部约 75000 站日；
   而 D+1 模型受 GFS 归档限制只有 2024 年后约 5000 条。
   代价是 9-11 点截止时纯实况打不过 NWP（实测 9 点 MAE 1.68 vs D+1 的 1.15），
   那几个时刻应把 D+1 模型的输出当特征加进来（--nwp-csv）。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import train_mos as T                      # 复用 ridge_fit / ridge_pred / sc / paired_boot

UTC = timezone.utc
CST = timezone(timedelta(hours=8))

NAMES = {"ZBAA": "北京首都", "ZSPD": "上海浦东", "ZGGG": "广州白云",
         "ZGSZ": "深圳宝安", "ZUUU": "成都双流", "ZUCK": "重庆江北",
         "ZHHH": "武汉天河", "ZSQD": "青岛胶东"}

# METAR 云量编码 -> 云量分数
CLOUD = {"CLR": 0.0, "SKC": 0.0, "NCD": 0.0, "NSC": 0.0, "CAVOK": 0.0,
         "FEW": 0.19, "SCT": 0.44, "BKN": 0.75, "OVC": 1.0, "VV": 1.0, "OVX": 1.0}

PEAK_H0, PEAK_H1 = 10, 19


def cloud_frac(s):
    if not s:
        return None
    return CLOUD.get(str(s).strip().upper()[:3])


def wx_flags(s):
    u = (s or "").upper()
    return (1.0 if "TS" in u else 0.0,
            1.0 if any(c in u for c in ("RA", "DZ", "SH", "SN")) else 0.0,
            1.0 if any(c in u for c in ("FG", "BR", "HZ")) else 0.0)


# ============================================================ 读数

def load_hourly(db, table="obs", min_peak=6):
    """返回 {(站, 北京时日期): {小时: {t, dewp, rh, wspd, pres, cld, ts, ra, obsc}}}"""
    import sqlite3
    conn = sqlite3.connect(db)
    cols = {c[1] for c in conn.execute(f"PRAGMA table_info({table})")}
    tcol = "valid_time_gmt" if "valid_time_gmt" in cols else "obs_time_utc"
    get = lambda c: c if c in cols else "NULL"
    q = (f"SELECT station, {tcol}, temp_c, {get('dewp_c')}, {get('rh')}, "
         f"{get('wspd_ms')}, {get('pres_hpa')}, {get('skyc1')}, {get('wxcodes')}, "
         f"{get('drct')} FROM {table} WHERE temp_c IS NOT NULL")
    days = defaultdict(dict)
    for stn, ts, t, dp, rh, ws, pr, sk, wx, dr in conn.execute(q):
        try:
            if tcol == "valid_time_gmt":
                dt = datetime.fromtimestamp(int(ts), UTC).astimezone(CST)
            else:
                s = str(ts).replace("Z", "+00:00").replace(" ", "T")
                dt = datetime.fromisoformat(s)
                dt = (dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt).astimezone(CST)
        except (ValueError, OSError, TypeError):
            continue
        d = days[(stn, dt.strftime("%Y-%m-%d"))]
        h = dt.hour
        # 同小时多条（半点报）取气温较高的那条，与日最高温口径一致
        if h in d and d[h]["t"] >= float(t):
            continue
        tsf, raf, obf = wx_flags(wx)
        d[h] = {"t": float(t),
                "dewp": None if dp is None else float(dp),
                "rh": None if rh is None else float(rh),
                "wspd": None if ws is None else float(ws),
                "pres": None if pr is None else float(pr),
                "cld": cloud_frac(sk), "ts": tsf, "ra": raf, "obsc": obf,
                "drct": None if dr is None else float(dr)}
    conn.close()
    # 峰值时段观测数达标还不够: 「今天」下午训练时 10-16 时已有 7 个小时、
    # 能通过 min_peak，但 tmax 仍是截断值。要求当天已过峰值时段末端才收。
    today = datetime.now(CST).date().isoformat()
    return {k: v for k, v in days.items()
            if sum(1 for h in v if PEAK_H0 <= h <= PEAK_H1) >= min_peak
            and (k[1] < today or max(v) >= PEAK_H1)}


# ============================================================ 特征

FEATS = ["max_so_far", "t_now", "trend_1h", "trend_2h", "trend_3h", "rise_since_06",
         "dewp_now", "dpd_now", "rh_now", "wspd_now", "pres_now", "pres_tend_3h",

         "cld_now", "cld_mean_am", "ts_am", "ra_am", "obsc_am",
         "prev_tmax", "prev_rise", "clim_rise", "clim_peak_h", "hours_to_peak",
         "rise_anom_3d", "rise_anom_7d",
         "doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2",
         "nwp_tmax", "nwp_minus_sofar",
         "nwp_cloud_peak", "nwp_swrad_peak", "nwp_swrad_max", "nwp_rh_peak",
         "nwp_dpd_peak", "nwp_t2m_range", "nwp_wind_peak", "nwp_gust_max"]
# 注意: INTERACTIONS 里那几个交互项实测无显著收益（见下），故不在 FEATS 里。
# 要重测就把它们加回这个列表

# 特征名 -> mos.csv / daily_features 的列名。午后云量和短波辐射直接决定还能升多少，
# 只喂 2m 温度等于扔掉模式里最相关的那部分信息。
NWP_COLS = {
    "nwp_tmax": "temperature_2m_max",
    "nwp_cloud_peak": "cloud_cover_peakmean",
    "nwp_swrad_peak": "shortwave_radiation_peakmean",
    "nwp_swrad_max": "shortwave_radiation_max",
    "nwp_rh_peak": "relative_humidity_2m_peakmean",
    "nwp_t2m_range": "t2m_range",
    "nwp_wind_peak": "wind_speed_10m_peakmean",
    "nwp_gust_max": "wind_gusts_10m_max",
}
# 对流/降水因子（A/B 测试中）。设 PLOYGON_CONV=1 打开。
# 物理动机: 午后雷暴压顶是广州这类站的主要误差来源，而现有 8 个模式变量里
# 没有任何降水或对流量。previous-runs 端点支持 precipitation / cape /
# lifted_index（能锁定起报时刻 = 不泄漏），气压层变量则不支持。
if os.environ.get("PLOYGON_CONV") == "1":
    NWP_COLS.update({
        "nwp_precip_peak": "precipitation_peakmean",
        "nwp_precip_max": "precipitation_max",
        "nwp_cape_peak": "cape_peakmean",
        "nwp_li_peak": "lifted_index_peakmean",
    })
    FEATS.extend(["nwp_precip_peak", "nwp_precip_max",
                  "nwp_cape_peak", "nwp_li_peak"])
# 模式的逐时廓线特征（A/B 测试中）。设 PLOYGON_PROF=1 打开。
# 物理动机: 见顶时刻由「能量输入什么时候停」决定，这件事在短波辐射的逐时
# 廓线里最直接 —— 云量常年饱和在 100%（广州 2026-07-20 整个下午都是 100，
# 毫无信息），而同一天辐射从 13 时的 744 骤降到 14 时 471、15 时 282。
# 原来只喂 cloud_cover_peakmean / shortwave_radiation_peakmean 两个时段均值，
# 等于把决定见顶时刻的那条曲线压成了一个数。
if os.environ.get("PLOYGON_PROF") == "1":
    NWP_COLS.update({
        "nwp_swrad_peak_h": "swrad_peak_h",
        "nwp_swrad_half_h": "swrad_half_h",
        "nwp_swrad_late_frac": "swrad_late_frac",
        "nwp_swrad_slope_pm": "swrad_slope_pm",
        "nwp_cld_onset_h": "cld_onset_h",
        "nwp_cld_slope_pm": "cld_slope_pm",
    })
    FEATS.extend(["nwp_swrad_peak_h", "nwp_swrad_half_h", "nwp_swrad_late_frac",
                  "nwp_swrad_slope_pm", "nwp_cld_onset_h", "nwp_cld_slope_pm"])

# 试过没赢: 模式的午后温度廓线（峰值时刻 t2m_peak_h、12->16 时升温速率
# t2m_slope_pm、傍晚相对午后均值 t2m_late_minus_peak，均已在
# build_mos_dataset.daily_features 里算好，csv 里有，只是不进特征表）。
# 11/13 时 ΔMAE +0.007 / +0.001，均不显著；**目标人群反而更差** ——
# 16 时后见顶的日子 11 时档 +0.078、13 时档 +0.010，重庆 MAE 纹丝不动。
# 根因: 假设「模式知道今天几点见顶，只是被聚合掉了」不成立。25km 分辨率下
# 模式对盆地站的午后廓线本身就不准，用不准的峰值时刻去预测另一个量只是引噪声。
# 与「用上午观测预报见顶时刻」那次失败同源: 不是信息被压掉了，是源头就不可靠。
# 要重测把下面三行加回 NWP_COLS 即可。
# 试过没赢: 各模式自己的 recent_bias（借鉴 PolyWeather 的 DEB）。
# 10/11/12 时 ΔMAE -0.021 / +0.001 / -0.005，全部无显著差异。
# 原因是临近路线已有 max_so_far（当天实测），对预报的锚定远强于模式 7 天偏差 ——
# recent_bias 在 D+1 时效上排第二，有了当日实况后就失效了。


# ---- 多模式 ----
# 多个模式的价值不只是各自的值，更是成员离散度: 模式吵得越凶剩余升温越不确定，
# 模型可据此把预报往气候态收。单个高分辨率模式给不了这个信息。
M2_COLS = {"tmax": "temperature_2m_max",
           "cloud_peak": "cloud_cover_peakmean",
           "swrad_peak": "shortwave_radiation_peakmean"}
ENS_FEATS = ["ens_mean", "ens_spread", "ens_mean_minus_sofar", "ens_max_minus_min"]


def m2_feature_names(n_extra):
    out = []
    for i in range(2, n_extra + 2):
        out += [f"m{i}_{k}" for k in M2_COLS] + [f"m{i}_minus_sofar",
                                                 f"m{i}_minus_m1"]
    return out + ENS_FEATS


def is_nwp_feat(name):
    if name in ("spread_x_hours", "gap_x_hours"):
        return True               # 这两个交互项依赖模式特征
    return (name.startswith("nwp") or name.startswith("ens_")
            or (name.startswith("m") and "_" in name and name[1:2].isdigit()))


def load_m2(paths):
    """读追加模式的 mos 格式 csv。返回 [{(站,日): {列: 值}}, ...]，顺序即模式顺序。"""
    maps = []
    for p in paths:
        mm = {}
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r.get("lead") == "1" and r.get("temperature_2m_max"):
                mm[(r["station"], r["date"])] = {
                    k: float(r[c]) for k, c in M2_COLS.items()
                    if r.get(c) not in (None, "")}
        maps.append(mm)
    return maps


def add_m2_feats(f, so_far, members_extra):
    """把追加模式的值写进特征字典。members_extra 是 [{列: 值}, ...]。"""
    t1 = f.get("nwp_tmax")
    members = [t1] if t1 is not None else []
    for i, g in enumerate(members_extra, start=2):
        g = g or {}
        for k in M2_COLS:
            f[f"m{i}_{k}"] = g.get(k)
        t2 = g.get("tmax")
        f[f"m{i}_minus_sofar"] = None if t2 is None else t2 - so_far
        f[f"m{i}_minus_m1"] = (None if (t2 is None or t1 is None) else t2 - t1)
        if t2 is not None:
            members.append(t2)
    if len(members) >= 2:
        mu = sum(members) / len(members)
        var = sum((x - mu) ** 2 for x in members) / len(members)
        f["ens_mean"] = mu
        f["ens_spread"] = var ** 0.5
        f["ens_mean_minus_sofar"] = mu - so_far
        f["ens_max_minus_min"] = max(members) - min(members)
    else:
        for k in ENS_FEATS:
            f[k] = None
    return f


# 岭回归是线性的，而这几条关系本质是乘性: 云的影响取决于还剩几小时能晒、
# 模式分歧的影响也随剩余时间放大。物理上有依据，但**实测无显著收益**
# （10/11/12 时 ΔMAE +0.005 / +0.004 / -0.004，区间全部跨 0），
# 所以不在 FEATS 里。函数保留，加回 FEATS 即可重测。
INTERACTIONS = ["cld_x_hours", "spread_x_hours", "gap_x_hours", "ts_x_clim",
                "dpd_x_hours"]


def add_interactions(f):
    h = f.get("hours_to_peak")
    def mul(a, b):
        x, y = f.get(a), f.get(b)
        return None if (x is None or y is None) else x * y
    f["cld_x_hours"] = mul("cld_mean_am", "hours_to_peak")
    f["spread_x_hours"] = mul("ens_spread", "hours_to_peak")
    f["gap_x_hours"] = mul("nwp_minus_sofar", "hours_to_peak")
    f["ts_x_clim"] = mul("ts_am", "clim_rise")
    f["dpd_x_hours"] = mul("dpd_now", "hours_to_peak")
    return f


def morning(hrs, cutoff):
    """截止时刻及之前的观测。要求足够密且最后一条不太旧。"""
    o = {h: v for h, v in hrs.items() if h <= cutoff}
    if len(o) < 5:
        return None
    if max(o) < cutoff - 2:
        return None
    return o


def _wdir_feats(now, am_list):
    """风向 -> sin/cos。另给上午平均风向与「风向转了多少」(0-180 度)。

    **试过没赢，不在 FEATS 里**（要重测就把 wdir_* 五项加回去）。
    信号是真的: 深圳控制住上午云量后，离岸风(W/N)比向岸风(S/SE)仍多升 0.76℃
    （n=307 vs 2042），成都 SW 比 NE 多 1.5℃。但 11/13 时 ΔMAE 0.000 / -0.004，
    均不显著。三个原因叠加:
      1. 那 0.76℃ 是全天升幅的差异，而模型预报的是「剩余升幅」，11 时已升大半
      2. 样本极不平衡 —— 深圳离岸风 307 天 vs 向岸 2042 天，而带 NWP 的训练集
         每站只有约 750 条，稀有类别学不出稳定系数
      3. 只对个别站有效: 深圳 11 时 -0.055，上海 +0.044，成都/重庆纹丝不动
    """
    import math as _m
    out = {"wdir_sin": None, "wdir_cos": None, "wdir_sin_am": None,
           "wdir_cos_am": None, "wdir_shift": None}
    if now is not None:
        r = _m.radians(now)
        out["wdir_sin"], out["wdir_cos"] = _m.sin(r), _m.cos(r)
    vals = [v for v in am_list if v is not None]
    if vals:
        # 角度平均要走向量平均，直接算术平均会在 0/360 边界翻车
        sx = sum(_m.sin(_m.radians(v)) for v in vals) / len(vals)
        cx = sum(_m.cos(_m.radians(v)) for v in vals) / len(vals)
        out["wdir_sin_am"], out["wdir_cos_am"] = sx, cx
        if now is not None:
            am_deg = _m.degrees(_m.atan2(sx, cx)) % 360
            d = abs(now - am_deg) % 360
            out["wdir_shift"] = min(d, 360 - d)
    return out


def _tend(o, lh, key, back=3):
    """某要素最近 back 小时的变化量。缺任一端就返回 None。"""
    a, b = o.get(lh), o.get(lh - back)
    if a is None or b is None:
        return None
    x, y = a.get(key), b.get(key)
    return None if (x is None or y is None) else x - y


def build_feats(o, cutoff, prev, clim_r, clim_p, doy, nwp):
    lh = max(o)
    cur = o[lh]
    msf = max(v["t"] for v in o.values())
    at = lambda h: o[h]["t"] if h in o else None

    def diff(k):
        b = at(lh - k)
        return None if b is None else cur["t"] - b

    am = [v for h, v in o.items() if h >= 6]
    cl = [v["cld"] for v in am if v["cld"] is not None]

    f = {
        "max_so_far": msf, "t_now": cur["t"],
        "trend_1h": diff(1), "trend_2h": diff(2), "trend_3h": diff(3),
        # 上午温度曲线的形态（A/B 测试中，PLOYGON_CURVE=1 打开）。
        # 现有 trend_1h/2h/3h 只覆盖最近三小时，等于把 6-13 时的曲线压成三个数。
        # trend_4/5/6h 补足更长的一段，curv_2h 是二阶差分（升温在加速还是减速）。
        "trend_4h": diff(4), "trend_5h": diff(5), "trend_6h": diff(6),
        "curv_2h": (None if (diff(1) is None or diff(2) is None)
                    else diff(1) - (diff(2) - diff(1))),
        "rise_since_06": None if at(6) is None else cur["t"] - at(6),
        "dewp_now": cur["dewp"],
        "dpd_now": None if cur["dewp"] is None else cur["t"] - cur["dewp"],
        "rh_now": cur["rh"], "wspd_now": cur["wspd"], "pres_now": cur["pres"],
        "pres_tend_3h": (None if (cur["pres"] is None or lh - 3 not in o
                                  or o[lh - 3]["pres"] is None)
                         else cur["pres"] - o[lh - 3]["pres"]),
        "cld_now": cur["cld"],
        # 上午各要素的 3 小时变化率。**试过没赢，不在 FEATS 里**（要重测就把
        # dewp_tend_3h 等加回去）。全量 6 项: 11/13 时 ΔMAE -0.010 / -0.012；
        # 只留露点+云量两项: -0.003 / -0.001。都不显著且方向偏坏。
        # 说明这些趋势的信息已被 trend_1h/2h/3h（温度自身趋势）和 cld_mean_am
        # 覆盖 —— 上午温度怎么走，本来就把水汽和云的影响都体现进去了
        **{f"{k}_tend_3h": _tend(o, lh, k) for k in
           ("dewp", "rh", "wspd", "cld")},
        "dpd_tend_3h": (None if (diff(3) is None or _tend(o, lh, "dewp") is None)
                        else diff(3) - _tend(o, lh, "dewp")),
        "dewp_mean_am": (sum(v["dewp"] for v in am if v["dewp"] is not None)
                         / max(1, sum(1 for v in am if v["dewp"] is not None))
                         if any(v["dewp"] is not None for v in am) else None),
        "cld_mean_am": (sum(cl) / len(cl)) if cl else None,
        "ts_am": max((v["ts"] for v in am), default=0.0),
        "ra_am": max((v["ra"] for v in am), default=0.0),
        "obsc_am": max((v["obsc"] for v in am), default=0.0),
        "prev_tmax": prev[0], "prev_rise": prev[1],
        "clim_rise": clim_r, "clim_peak_h": clim_p,
        "hours_to_peak": None if clim_p is None else clim_p - cutoff,
        "doy_sin1": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos1": math.cos(2 * math.pi * doy / 365.25),
        "doy_sin2": math.sin(4 * math.pi * doy / 365.25),
        "doy_cos2": math.cos(4 * math.pi * doy / 365.25),
    }
    g = nwp if isinstance(nwp, dict) else ({} if nwp is None else {"temperature_2m_max": nwp})
    for name, col in NWP_COLS.items():
        f[name] = g.get(col)
    f["nwp_minus_sofar"] = None if f["nwp_tmax"] is None else f["nwp_tmax"] - msf
    tp, dp = g.get("temperature_2m_peakmean"), g.get("dew_point_2m_peakmean")
    f["nwp_dpd_peak"] = None if (tp is None or dp is None) else tp - dp
    return f, msf


def make_samples(days, cutoff, clim_r, clim_p, nwp_map, split_year, m2_maps=()):
    out = []
    for (stn, d), hrs in sorted(days.items()):
        o = morning(hrs, cutoff)
        if o is None:
            continue
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        mo = dt.month
        pd_ = (dt - timedelta(days=1)).isoformat()
        ph = days.get((stn, pd_))
        prev = (max(v["t"] for v in ph.values()) if ph else None,
                clim_r.get((stn, mo)))
        f, msf = build_feats(o, cutoff, prev, clim_r.get((stn, mo)),
                             clim_p.get((stn, mo)), dt.timetuple().tm_yday,
                             nwp_map.get((stn, d)))
        if m2_maps:
            add_m2_feats(f, msf, [mm.get((stn, d)) for mm in m2_maps])
        add_interactions(f)
        tmax = max(v["t"] for v in hrs.values())
        # 当天实际见顶时刻。只当辅助模型的训练目标用，绝不进 FEATS ——
        # 它是未来信息，混进主模型就是泄漏
        ph_ = [h for h, v in hrs.items() if v["t"] >= tmax - 1e-9]
        out.append({"stn": stn, "date": d, "year": dt.year, "month": mo,
                    "so_far": msf, "tmax": tmax, "rise": tmax - msf,
                    "peak_h": min(ph_) if ph_ else None, "f": f})

    # 近期升幅异常: 前 3/7 个可用日的 (实际升幅 - 当月气候升幅) 均值。
    # 只回看更早的日子，预报时这些都已知。捕捉「这几天午后偏爱升得多/少」的
    # 天气型漂移 —— D+1 模型里 recent_bias 系数排第二，临近侧一直缺这个。
    by_stn = defaultdict(list)
    for r in out:
        by_stn[r["stn"]].append(r)
    for rows in by_stn.values():
        rows.sort(key=lambda r: r["date"])
        hist = []                       # [(date, anomaly)]
        for r in rows:
            # 只认 10 个日历日以内的历史。不限窗口的话，数据有缺口时
            # 「前 3 个可用日」可能是一个月前，与 predict_nowcast 的口径也对不上
            cut = (datetime.strptime(r["date"], "%Y-%m-%d").date()
                   - timedelta(days=10)).isoformat()
            recent = [a for d0, a in hist if d0 >= cut]
            for k in (3, 7):
                v = recent[-k:]
                r["f"][f"rise_anom_{k}d"] = sum(v) / len(v) if len(v) == k else None
            cr = r["f"]["clim_rise"]
            if cr is not None:
                hist.append((r["date"], r["rise"] - cr))

    if XSTN:
        add_xstn_feats(out)
    if ORACLE_PEAK:
        # **明知泄漏的先知实验**，只用来量「见顶时刻不确定」这个瓶颈值多少。
        # 把当天真实见顶时刻直接喂进特征表 —— 任何见顶时刻预测器的效果
        # 都不可能超过它。绝不可上线，也不进 check_consistency 的白名单。
        for r in out:
            ph = r.get("peak_h")
            # 可加噪：量「见顶时刻要预测到多准才有用」。噪声 0 = 完美先知。
            if ph is not None and ORACLE_NOISE > 0:
                import random as _rnd
                _rnd.seed(hash((r["stn"], r["date"], cutoff)) & 0xffffffff)
                ph = ph + _rnd.gauss(0.0, ORACLE_NOISE)
            r["f"]["oracle_peak_h"] = None if ph is None else float(ph)
            r["f"]["oracle_hours_to_peak"] = (
                None if ph is None else float(ph) - cutoff)
    return out


ORACLE_PEAK = os.environ.get("PLOYGON_ORACLE_PEAK") == "1"
ORACLE_NOISE = float(os.environ.get("PLOYGON_ORACLE_NOISE") or 0.0)
if ORACLE_PEAK:
    FEATS.extend(["oracle_peak_h", "oracle_hours_to_peak"])


# 跨站特征（A/B 测试中）。设 PLOYGON_XSTN=1 打开。
#
# 起报时刻各站的上午观测都已到手，用邻站补充本站信息不泄漏。
# 伙伴选择用 2010 起夏季日最高温距平的同日相关（30 天滑动中位去季节）:
#   广州-深圳  98km  0.720      重庆-武汉 736km  0.517
#   成都-重庆 276km  0.477      上海-武汉 726km  0.368
#   北京-广州1881km  0.004      北京-上海1100km -0.026
# 距离与相关的相关系数 -0.663，但**不是越近越有用**: 广州-深圳同日相关最高
# 却毫无增量（离线 P(更好)=8%），因为太近，邻站知道的本站自己也知道。
# 成都->重庆隔 276km 增量最大（离线 MAE 0.9031->0.8325，P=100%）。
CURVE = os.environ.get("PLOYGON_CURVE") == "1"
CURVE_FEATS = ["trend_4h", "trend_5h", "trend_6h", "curv_2h"]
if CURVE:
    FEATS.extend(CURVE_FEATS)

XSTN = os.environ.get("PLOYGON_XSTN") == "1"
XPARTNER = {                       # 站 -> 两个伙伴站（按上面的相关排序取前二）
    "ZBAA": ("ZSQD", "ZSPD"), "ZSPD": ("ZHHH", "ZUCK"),
    "ZGGG": ("ZGSZ", "ZSPD"), "ZGSZ": ("ZGGG", "ZSPD"),
    "ZUUU": ("ZUCK", "ZHHH"), "ZUCK": ("ZUUU", "ZHHH"),
    "ZHHH": ("ZUCK", "ZSPD"), "ZSQD": ("ZHHH", "ZSPD"),
}
XFEATS = ["rise_since_06", "trend_3h", "wspd_now"]


def xstn_feature_names():
    return [f"x{i}_{k}" for i in (1, 2) for k in XFEATS]


def add_xstn_feats(rows):
    """给每条样本补两个伙伴站同日同截止时刻的上午特征。"""
    idx = {(r["stn"], r["date"]): r for r in rows}
    for r in rows:
        for i, p in enumerate(XPARTNER.get(r["stn"], ()), start=1):
            src = idx.get((p, r["date"]))
            for k in XFEATS:
                r["f"][f"x{i}_{k}"] = None if src is None else src["f"].get(k)


if XSTN:
    FEATS.extend(xstn_feature_names())


def climatology(days, cutoffs, split_year):
    """只用训练年份估气候态，避免泄漏。"""
    rise = {c: defaultdict(list) for c in cutoffs}
    peak = defaultdict(list)
    for (stn, d), hrs in days.items():
        y, mo = int(d[:4]), int(d[5:7])
        if y >= split_year:
            continue
        tmax = max(v["t"] for v in hrs.values())
        ph = [h for h, v in hrs.items() if v["t"] >= tmax - 1e-9]
        peak[(stn, mo)].append(sum(ph) / len(ph))
        for c in cutoffs:
            o = morning(hrs, c)
            if o:
                rise[c][(stn, mo)].append(tmax - max(v["t"] for v in o.values()))
    R = {c: _month_est(m) for c, m in rise.items()}
    P = _month_est(peak)
    return R, P


def _month_est(m, need=20):
    """每(站,月)的均值。样本够就用本月；不够退回 ±1 月窗口。

    2026-08-01 加的兜底。深圳换用 WU 序列后只剩 2024-07 起的数据，
    按「本月 >= 20 天」只有 7 月和 12 月过关 —— 当时正是 8 月，
    clim_rise / clim_peak / hours_to_peak 三个特征全走中位数填补，
    而且**不报错、不留痕**，只有翻模型 json 才看得出来。

    只在本月不足时才放宽，所以对样本充足的 7 个站**逐位不变**。
    """
    out = {}
    for (stn, mo), v in m.items():
        if len(v) >= need:
            out[(stn, mo)] = sum(v) / len(v)
            continue
        wide = list(v)
        for d in (-1, 1):
            wide += m.get((stn, (mo + d - 1) % 12 + 1), [])
        if len(wide) >= need:
            out[(stn, mo)] = sum(wide) / len(wide)
    return out


# ============================================================ 训练

def matrix(rows, med=None, names=None):
    names = names or FEATS
    if med is None:
        med = {}
        for f in names:
            v = sorted(r["f"][f] for r in rows if r["f"].get(f) is not None)
            med[f] = v[len(v) // 2] if v else 0.0
    X = []
    for r in rows:
        row = []
        for f in names:
            v = r["f"].get(f)
            row.append(med[f] if v is None else v)
            row.append(1.0 if v is None else 0.0)
        X.append(row)
    return X, med


def fit_hurdle(rows, alphas, med, names):
    """两段式: P(还会升) × E[升幅 | 会升]。

    alphas 传单元素列表即为指定 alpha。以前这里固定取列表中位数（=10），
    与岭回归用验证集选出来的 alpha 不同 —— 导致上线的两段式模型和
    backtest_nowcast.py 里检验的那个不是同一个，回测数字对不上生产。
    """
    X, _ = matrix(rows, med, names)
    yb = [1.0 if r["rise"] > 1e-9 else 0.0 for r in rows]
    pos = [i for i, r in enumerate(rows) if r["rise"] > 1e-9]
    if len(pos) < 50:
        return None
    Xp = [X[i] for i in pos]
    yp = [rows[i]["rise"] for i in pos]
    return {"cls": T.ridge_fit(X, yb, alphas[len(alphas) // 2]),
            "reg": T.ridge_fit(Xp, yp, alphas[len(alphas) // 2])}


MAX_RISE = 8


def fit_ordinal(rows, alpha, med, names):
    """对每个 k 拟合 P(rise >= k)。rise 严格是整数度（观测就是整数），
    所以能直接给出整数升幅的分布。用途是高端情景 P90，不是改点预报 ——
    用分布改点预报已验证无效（众数、±1 窗口决策都与均值取整无显著差异）。"""
    X, _ = matrix(rows, med, names)
    out = []
    for k in range(1, min(max(int(r["rise"]) for r in rows), MAX_RISE) + 1):
        yk = [1.0 if r["rise"] >= k else 0.0 for r in rows]
        if sum(yk) < 30:
            break
        out.append(T.ridge_fit(X, yk, alpha))
    return out or None


def rise_pmf(ordm, X):
    """P(rise>=k) 逐 k 预测后强制单调，差分得每档概率。"""
    cum = [[min(1.0, max(0.0, v)) for v in T.ridge_pred(mk, X)] for mk in ordm]
    out = []
    for i in range(len(X)):
        s = [cum[k][i] for k in range(len(cum))]
        for k in range(1, len(s)):        # 各 k 独立拟合会破坏单调性
            s[k] = min(s[k], s[k - 1])
        out.append([1.0 - s[0]] + [s[k - 1] - s[k] for k in range(1, len(s))]
                   + [s[-1]])
    return out


def fit_quantile(rows, tau, alpha, med, names, iters=30):
    """升幅的条件 tau 分位（pinball 损失，IRLS 求解）。

    为什么不继续用序贯分类的 PMF: 那是线性概率模型，尾部标定差 ——
    实测 P90 在已见顶日覆盖 100%（恒等式白送），在大升幅日只有 60-88%，
    正好在最需要它的地方失效。直接回归分位数没有这个问题。

    输出格式与 ridge_fit 一致（w/mx/sx/my），推理端复用 ridge_pred，
    模型 JSON 里仍然只有权重，不需要任何机器学习库。
    """
    try:
        import numpy as np
    except ImportError:
        return None
    X, _ = matrix(rows, med, names)
    Xa = np.asarray(X, float)
    ya = np.asarray([r["rise"] for r in rows], float)
    mx, sx = Xa.mean(0), Xa.std(0)
    sx[sx < 1e-9] = 1.0
    Z = np.hstack([(Xa - mx) / sx, np.ones((len(Xa), 1))])
    beta = np.zeros(Z.shape[1])
    pen = np.eye(Z.shape[1]) * alpha
    pen[-1, -1] = 0.0                     # 截距不惩罚
    for _ in range(iters):
        r = ya - Z @ beta
        # pinball 的 IRLS 权重。分母兜底 0.1 度，避免残差趋零时权重爆炸
        w = np.where(r >= 0, tau, 1.0 - tau) / np.maximum(np.abs(r), 0.1)
        A = (Z * w[:, None]).T @ Z + pen
        try:
            beta = np.linalg.solve(A, (Z * w[:, None]).T @ ya)
        except np.linalg.LinAlgError:
            return None
    return {"w": beta[:-1].tolist(), "mx": mx.tolist(), "sx": sx.tolist(),
            "my": float(beta[-1])}


def calibrate_quantile(q90, rows_va, med, names, tau=0.90, buckets=(1, 2, 3, 4)):
    """按预测升幅分层标定分位数。**实测无用，未启用**，保留供复现。

    动机是「大升幅日覆盖只有 60-83%」，但那个诊断口径本身就是错的:
    分位数承诺的是「给定预测」的覆盖，不是「给定实际结果」的覆盖。
    按后者去调，反而把前者的标定弄差了（平均偏离 90% 从 3.0pt 涨到 4.0pt）。
    纯分位数回归（不加这层标定）在正确口径下最优、区间也最窄。

    以下为原实现。

    为什么需要: 未标定时总体覆盖 94%，但拆开看 —— 已见顶日 100%（恒等式白送），
    大升幅日只有 60-83%。总体数字把最该管的那部分完全掩盖了。
    做法是在验证集上按预测升幅分桶，逐桶算出「让该桶覆盖率达到 tau」所需的偏移。
    返回 {桶号: 偏移量}，推理时按桶加上去。
    """
    if not q90 or not rows_va:
        return None
    X, _ = matrix(rows_va, med, names)
    pred = T.ridge_pred(q90, X)
    by = defaultdict(list)
    for r, p in zip(rows_va, pred):
        b = min(int(max(0.0, p)), len(buckets))
        by[b].append(r["rise"] - p)          # 需要补多少才能盖住
    out = {}
    for b, res in by.items():
        if len(res) < 30:
            continue
        res.sort()
        off = res[min(len(res) - 1, int(len(res) * tau))]
        out[str(b)] = max(0.0, off)          # 只放宽不收紧: 高端情景宁保守
    return out or None


def apply_quantile_cal(pred_q, pred_mean, cal, buckets=4):
    """把分桶偏移加到分位数预测上。桶按未标定的分位数预测本身划分。"""
    out = []
    for q, _m in zip(pred_q, pred_mean):
        v = max(0.0, q)
        if cal:
            v += cal.get(str(min(int(v), buckets)), 0.0)
        out.append(v)
    return out


def rise_quantile(pmf, q):
    """整数升幅分布的 q 分位。高端情景「不排除冲到多少」。"""
    out = []
    for p in pmf:
        tot = sum(p) or 1.0
        acc, k = 0.0, 0
        for k in range(len(p)):
            acc += p[k] / tot
            if acc >= q:
                break
        out.append(float(k))
    return out


def pred_hurdle(m, X):
    p = [min(1.0, max(0.0, v)) for v in T.ridge_pred(m["cls"], X)]
    r = T.ridge_pred(m["reg"], X)
    return [pi * max(0.0, ri) for pi, ri in zip(p, r)]


def run_cutoff(days, cutoff, clim_r, clim_p, nwp_map, args, m2_maps=()):
    rows = make_samples(days, cutoff, clim_r, clim_p, nwp_map,
                        args.split_year, m2_maps)
    rows = [r for r in rows if r["f"]["clim_rise"] is not None]
    if len(rows) < 500:
        print(f"\n截止 {cutoff:02d} 时: 样本不足 ({len(rows)})")
        return None

    obs_names = [n for n in FEATS if not is_nwp_feat(n)]
    if args.nwp_csv:
        # NWP 只覆盖 2024 年起。若沿用年份切分，训练期该特征全缺 -> 系数恒为 0，
        # 加了等于没加。所以限定到有 NWP 的样本，并改用日期分位切分。
        rows = [r for r in rows if r["f"].get("nwp_tmax") is not None]
        if m2_maps:
            rows = [r for r in rows if r["f"].get("ens_mean") is not None]
        if len(rows) < 500:
            print(f"\n截止 {cutoff:02d} 时: NWP 覆盖的样本不足 ({len(rows)})")
            return None
        ds = sorted({r["date"] for r in rows})
        c1, c2 = ds[int(len(ds) * .70)], ds[int(len(ds) * .85)]
        tr = [r for r in rows if r["date"] < c1]
        va = [r for r in rows if c1 <= r["date"] < c2]
        te = [r for r in rows if r["date"] >= c2]
        variants = [("纯实况", obs_names), ("实况+NWP", list(FEATS))]
        split_desc = f"<{c1} / {c1}~{c2} / >={c2}"
    else:
        tr = [r for r in rows if r["year"] < args.split_year - 1]
        va = [r for r in rows if r["year"] == args.split_year - 1]
        te = [r for r in rows if r["year"] >= args.split_year]
        variants = [("纯实况", obs_names)]
        split_desc = (f"<{args.split_year-1} / {args.split_year-1} / "
                      f">={args.split_year}")
    names = variants[-1][1]
    if not (tr and va and te):
        print(f"\n截止 {cutoff:02d} 时: 切分后有空集")
        return None

    print(f"\n{'='*76}")
    print(f"截止 {cutoff:02d} 时（北京时）   训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)}")
    print(f"切分 {split_desc}")
    if args.nwp_csv:
        print(f"同一批样本上训练两个版本做配对比较（纯实况 vs 实况+NWP）")

    tmax_te = [r["tmax"] for r in te]
    npk = [i for i, r in enumerate(te) if r["rise"] > 1e-9]
    print(f"测试期已见顶 {100*(1-len(npk)/len(te)):.0f}%")

    # ---- 基线: 气候平均升温 ----
    e_clim = [r["so_far"] + r["f"]["clim_rise"] - r["tmax"] for r in te]
    print(f"\n── 基线")
    T.show("气候平均剩余升温", T.sc(e_clim))
    T.show("  仅未见顶", T.sc([e_clim[i] for i in npk]))
    e_pers = [r["so_far"] - r["tmax"] for r in te]
    T.show("已达最高（不再升）", T.sc(e_pers))

    keys_te = [(r["stn"], r["date"]) for r in te]
    err_by, rnd_by, per_all, keep = {}, {}, {}, {}

    for vtag, vnames in variants:
        suf = f"({vtag})" if len(variants) > 1 else ""
        Xtr, med = matrix(tr, None, vnames)
        Xva, _ = matrix(va, med, vnames)
        Xte, _ = matrix(te, med, vnames)
        ytr = [r["rise"] for r in tr]

        best = (args.alphas[0], 1e9, None)
        for a in args.alphas:
            m = T.ridge_fit(Xtr, ytr, a)
            pv = [max(0.0, v) for v in T.ridge_pred(m, Xva)]
            mae = sum(abs(x + r["so_far"] - r["tmax"])
                      for x, r in zip(pv, va)) / len(va)
            if mae < best[1]:
                best = (a, mae, m)
        alpha, _, mdl = best
        pr = [max(0.0, v) for v in T.ridge_pred(mdl, Xte)]
        err_by["合并岭回归" + suf] = dict(zip(keys_te,
            [x + r["so_far"] - r["tmax"] for x, r in zip(pr, te)]))
        rnd_by["合并岭回归" + suf] = dict(zip(keys_te,
            [round(x + r["so_far"]) - r["tmax"] for x, r in zip(pr, te)]))

        hm = fit_hurdle(tr, [alpha], med, vnames)
        if hm:
            ph = pred_hurdle(hm, Xte)
            err_by["合并两段式" + suf] = dict(zip(keys_te,
                [x + r["so_far"] - r["tmax"] for x, r in zip(ph, te)]))
            rnd_by["合并两段式" + suf] = dict(zip(keys_te,
                [round(x + r["so_far"]) - r["tmax"] for x, r in zip(ph, te)]))

        # 分站
        per = {}
        for stn in sorted({r["stn"] for r in rows}):
            trs = [r for r in tr if r["stn"] == stn]
            vas = [r for r in va if r["stn"] == stn]
            tes = [r for r in te if r["stn"] == stn]
            if len(trs) < 300 or not vas or not tes:
                continue
            Xs, meds = matrix(trs, None, vnames)
            Xv, _ = matrix(vas, meds, vnames)
            Xt, _ = matrix(tes, meds, vnames)
            ys = [r["rise"] for r in trs]
            bb = (args.alphas[0], 1e9, None)
            for a in args.alphas:
                m = T.ridge_fit(Xs, ys, a)
                pv = [max(0.0, v) for v in T.ridge_pred(m, Xv)]
                mae = sum(abs(x + r["so_far"] - r["tmax"])
                          for x, r in zip(pv, vas)) / len(vas)
                if mae < bb[1]:
                    bb = (a, mae, m)
            hs = fit_hurdle(trs, [bb[0]], meds, vnames)
            per[stn] = {"alpha": bb[0], "median": meds, "ridge": bb[2],
                        "hurdle": hs,
                        "ordinal": fit_ordinal(trs, bb[0], meds, vnames),
                        "q90": fit_quantile(trs, 0.90, bb[0], meds, vnames)}
            pt = [max(0.0, v) for v in T.ridge_pred(bb[2], Xt)]
            err_by.setdefault("分站岭回归" + suf, {}).update(
                {(r["stn"], r["date"]): x + r["so_far"] - r["tmax"]
                 for x, r in zip(pt, tes)})
            rnd_by.setdefault("分站岭回归" + suf, {}).update(
                {(r["stn"], r["date"]): round(x + r["so_far"]) - r["tmax"]
                 for x, r in zip(pt, tes)})
            if hs:
                p2 = pred_hurdle(hs, Xt)
                err_by.setdefault("分站两段式" + suf, {}).update(
                    {(r["stn"], r["date"]): x + r["so_far"] - r["tmax"]
                     for x, r in zip(p2, tes)})
                rnd_by.setdefault("分站两段式" + suf, {}).update(
                    {(r["stn"], r["date"]): round(x + r["so_far"]) - r["tmax"]
                     for x, r in zip(p2, tes)})
        per_all[vtag] = per
        # 按版本分别记录。导出时统一取最后一个版本，否则 names/median 会与
        # per_station 模型的列数对不上（26 项 vs 28 项 -> 52 列 vs 56 列）
        keep[vtag] = (alpha, med, mdl, hm, vnames,
                      fit_ordinal(tr, alpha, med, vnames),
                      fit_quantile(tr, 0.90, alpha, med, vnames))

        print(f"\n── {vtag}（{len(vnames)} 项特征，合并 alpha={alpha:g}）")
        for tag in ("合并岭回归", "合并两段式", "分站岭回归", "分站两段式"):
            k = tag + suf
            if k not in err_by:
                continue
            ks = sorted(err_by[k])
            T.show(tag, T.sc([err_by[k][x] for x in ks]))
            T.show("  取整后（上线口径）", T.sc([rnd_by[k][x] for x in ks]))
        al = sorted({v["alpha"] for v in per.values()})
        print(f"  各站 alpha: {al}")

    vsel = variants[-1][0]
    vsuf = f"({vsel})" if len(variants) > 1 else ""
    e_ridge = [err_by["合并岭回归" + vsuf][k] for k in keys_te]
    alpha, med, mdl, hm, names, ordm, q90 = keep[vsel]
    per = per_all[vsel]
    if len(variants) > 1:
        print(f"\n（导出的是「{vsel}」版本）")

    npk_keys = {keys_te[i] for i in npk}

    # ---- 显著性 ----
    dts = [r["date"] for r in te]
    print(f"\n── 对气候基线的配对检验")
    for tag in sorted(err_by):
        ks = sorted(err_by[tag])
        rb = T.paired_boot([e_clim[keys_te.index(k)] for k in ks],
                           [err_by[tag][k] for k in ks], [k[1] for k in ks])
        if rb:
            d, lo, hi = rb
            v = "模型显著更优" if lo > 0 else "基线更优" if hi < 0 else "无显著差异"
            print(f"  {tag:<20} ΔMAE={d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {v}")
    def cmp(a, b, label=""):
        if a not in err_by or b not in err_by:
            return
        ck = sorted(set(err_by[a]) & set(err_by[b]))
        rb = T.paired_boot([err_by[a][k] for k in ck],
                           [err_by[b][k] for k in ck], [k[1] for k in ck])
        if not rb:
            return
        d, lo, hi = rb
        v = f"{b} 更优" if lo > 0 else f"{a} 更优" if hi < 0 else "无显著差异"
        print(f"  {label or (a + ' vs ' + b):<44} ΔMAE={d:+.3f}  "
              f"[{lo:+.3f}, {hi:+.3f}]  {v}")

    arch = ["合并岭回归", "合并两段式", "分站岭回归", "分站两段式"]
    if len(variants) > 1:
        va_, vb_ = f"({variants[0][0]})", f"({variants[1][0]})"
        print(f"\n── 加 NWP 有没有用（同架构、同样本、同测试期）")
        for a in arch:
            cmp(a + va_, a + vb_, f"{a}: {variants[0][0]} vs {variants[1][0]}")
        suf = vb_
    else:
        suf = ""
    print(f"\n── 架构比较（区间跨 0 就选更简单的）")
    cmp("合并岭回归" + suf, "分站岭回归" + suf, "合并 vs 分站（岭回归）")
    cmp("合并两段式" + suf, "分站两段式" + suf, "合并 vs 分站（两段式）")
    cmp("分站岭回归" + suf, "分站两段式" + suf, "分站: 岭回归 vs 两段式")

    # ---- 分站 ----
    print(f"\n── 分站（测试期，仅未见顶的日子）")
    pkey = "分站岭回归" + (f"({variants[-1][0]})" if len(variants) > 1 else "")
    hasp = pkey in err_by
    print(f"  {'站点':<14}{'n':>6}{'气候基线':>10}{'合并':>8}"
          + (f"{'分站':>8}{'分站-合并':>11}" if hasp else ""))
    for stn in sorted({r["stn"] for r in te}):
        idx = [i for i in npk if te[i]["stn"] == stn]
        if len(idx) < 20:
            continue
        a = sum(abs(e_clim[i]) for i in idx) / len(idx)
        b = sum(abs(e_ridge[i]) for i in idx) / len(idx)
        line = f"  {stn} {NAMES.get(stn,''):<9}{len(idx):>6}{a:>10.2f}{b:>8.2f}"
        if hasp:
            ks = [keys_te[i] for i in idx if keys_te[i] in err_by[pkey]]
            if ks:
                c = sum(abs(err_by[pkey][k]) for k in ks) / len(ks)
                line += f"{c:>8.2f}{b-c:>+11.2f}"
        print(line)

    if args.coef:
        print(f"\n── 系数（标准化尺度，绝对值前 12）")
        cn = []
        for f in names:
            cn += [f, f + "__isnan"]
        for k, v in sorted(zip(cn, mdl["w"]), key=lambda x: -abs(x[1]))[:12]:
            print(f"  {k:<28}{v:+8.3f}")

    return {"cutoff": cutoff, "alpha": alpha, "names": names, "median": med,
            "ridge": mdl, "hurdle": hm, "ordinal": ordm, "q90": q90,
            "per_station": per or None,
            "has_nwp": bool(args.nwp_csv),
            "clim_rise": {f"{k[0]}|{k[1]}": v for k, v in clim_r.items()},
            "clim_peak": {f"{k[0]}|{k[1]}": v for k, v in clim_p.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cn.sqlite")
    ap.add_argument("--table", default="obs")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[9, 11, 12, 13])
    ap.add_argument("--split-year", type=int, default=2024)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1, 3, 10, 30, 100, 300])
    ap.add_argument("--nwp-csv", default="",
                    help="mos.csv 路径，把 D+1 模式预报当特征（会限制到 2024 年后）")
    ap.add_argument("--nwp-csv2", nargs="+", default=[],
                    help="追加模式的 mos 格式 csv（可多个），加多模式与集合离散度特征")
    ap.add_argument("--coef", action="store_true")
    ap.add_argument("--dump", help="存模型 JSON")
    args = ap.parse_args()

    print("读取逐时实况…", file=sys.stderr)
    days = load_hourly(args.db, args.table)
    print(f"  {len(days)} 个可用站日", file=sys.stderr)
    if not days:
        return 1

    nwp_map = {}
    if args.nwp_csv and os.path.exists(args.nwp_csv):
        want = set(NWP_COLS.values()) | {"temperature_2m_peakmean",
                                         "dew_point_2m_peakmean"}
        for r in csv.DictReader(open(args.nwp_csv, encoding="utf-8")):
            if r.get("lead") == "1" and r.get("temperature_2m_max"):
                nwp_map[(r["station"], r["date"])] = {
                    c: float(r[c]) for c in want if r.get(c) not in (None, "")}
        print(f"  NWP 特征 {len(nwp_map)} 个站日", file=sys.stderr)

    m2_maps = []
    if args.nwp_csv2:
        if not args.nwp_csv:
            print("[error] --nwp-csv2 需要 --nwp-csv", file=sys.stderr)
            return 1
        m2_maps = load_m2(args.nwp_csv2)
        for p_, mm in zip(args.nwp_csv2, m2_maps):
            print(f"  追加模式 {os.path.basename(p_)}: {len(mm)} 站日", file=sys.stderr)
        FEATS.extend(m2_feature_names(len(m2_maps)))

    clim_r, clim_p = climatology(days, args.cutoffs, args.split_year)
    out = {}
    for c in args.cutoffs:
        r = run_cutoff(days, c, clim_r.get(c, {}), clim_p, nwp_map, args,
                       m2_maps)
        if r:
            out[c] = r

    print(f"\n{'='*76}")
    print("对照: D+1 模型（24h 时效）取整后 MAE 1.15 / TAF 11 时轮 D+0（4h）1.07")

    if args.dump and out:
        json.dump({str(k): v for k, v in out.items()},
                  open(args.dump, "w"), ensure_ascii=False)
        print(f"\n模型已存 {args.dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

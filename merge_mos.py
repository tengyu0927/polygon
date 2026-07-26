#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_mos.py — 把多个模式的 mos csv 横向拼成一份多模式训练集

    python3 merge_mos.py --base mos.csv --extra mos_ecmwf.csv mos_cma.csv \\
        mos_icon.csv mos_jma_.csv mos_gem_.csv mos_ukmo.csv --out mos_multi.csv

基准列（y_tmax / temperature_2m_max）取 base 那份，因为 train_mos.py 的目标量是
残差 y - temperature_2m_max，换基准会让新旧结果不可比。追加模式的列加 m2_/m3_ 前缀，
再补一组集合统计量。

train_mos.py 的 feature_names() 自动识别数值列，所以拼完直接喂给它即可，
不用改训练代码。

缺测不丢样本: UKMO 归档只到 2025-01，早期行该模式列为空，
train_mos 会用中位数填补并加一列缺测指示。

关于 prev_tmax / doy_* 是否也加前缀（`--dup-obs`）—— 一个已解决的谜:
七模式时代（含 UKMO）加重复列能让 D+1 从 1.18 降到 1.08，当时查不出机制。
剔除 UKMO 并加入 DEB 后重测，A/B/C 三个变体全部持平
（不加 1.024 / 加 1.031 / 精确复制 1.02，配对区间跨 0）。

**结论: 那不是普遍现象，是 UKMO 浅归档的副作用。** UKMO 在 2025-01 之前没有数据，
那几组"重复列"在缺测时被填成中位数，等于给了模型一个「2025-01 之前」的时代标记，
让它对陈旧训练期区别对待。UKMO 一剔除，标记没了，效果也没了。
所以默认改成不加（少 46 列，更简单）。--dup-obs 保留供复现该实验。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

KEY = ("station", "date", "lead")
# 只有这些列在各份 csv 里逐行相同（纯实况/日历），加前缀是造重复列。
# recent_bias 绝不能列进来 —— 它是「实况 - 该模式预报」，每个模式各不相同
# （同一天 GFS 2.33、ICON 0.39），是各模式自己的漂移订正项，删了会明显变差。
OBS_COLS = {"prev_tmax", "doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"}
SKIP = set(KEY) | {"y_tmax"}


def read(path):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {(r["station"], r["date"], r["lead"]): r for r in rows}, rows


def deb_weights(hist, cur, lookback=7, decay=0.85, spread_cap=1.7):
    """DEB 的核心数学。训练端与预测端共用这一个函数，避免两处实现漂移。

    hist: {模式号: [(日期, 预报, 实况), ...]}，必须已按日期升序且只含目标日之前
    cur:  {模式号: 目标日的预报值}
    返回 (融合值, 回退系数) —— 回退系数 <1 表示成员分歧大、权重被拉向等权
    """
    errs, biases = {}, {}
    for i, past in hist.items():
        used = []
        for age, (_, f, y) in enumerate(reversed(past[-lookback * 3:])):
            if len(used) >= lookback:
                break
            used.append((abs(f - y), f - y, decay ** age))
        if len(used) >= 2:
            tw = sum(w for _, _, w in used)
            errs[i] = sum(e * w for e, _, w in used) / tw
            # 偏差样本太少不可信，直接归零（原实现的阈值是 5）
            biases[i] = (sum(b for _, b, _ in used) / len(used)
                         if len(used) >= 5 else 0.0)
    common = [i for i in cur if i in errs and cur[i] is not None]
    if len(common) < 2:
        return None, None
    inv = {i: 1.0 / (errs[i] + abs(biases[i]) * 0.5 + 0.1) for i in common}
    tot = sum(inv.values())
    w = {i: v / tot for i, v in inv.items()}
    vals = [cur[i] for i in common]
    spread = max(vals) - min(vals)
    trust = 1.0
    if spread > spread_cap:
        trust = spread_cap / spread
        w = {i: x * trust + (1 - trust) / len(w) for i, x in w.items()}
    return sum(cur[i] * w[i] for i in common), trust


def deb_columns(base_rows, extras, lookback=7, decay=0.85, spread_cap=1.7):
    """借鉴 PolyWeather (AGPL-3.0) 的 Dynamic Error Balancing:
    按各模式近期误差的倒数加权融合，误差用指数衰减加权，偏差绝对值进分母惩罚，
    成员分歧过大时向等权回退。

    为什么值得加: 岭回归学到的是**固定**线性组合，而 DEB 权重是近期误差的函数、
    每天都在变。这种随时间自适应的非线性组合，线性模型表示不出来。

    严格因果: 第 d 天的权重只用 d 之前的「实况-预报」误差。
    spread_cap 用摄氏度（原实现是 3 华氏度 ≈ 1.7 摄氏度）。
    """
    # 每个模式的历史: {(站, lead): [(日期, 预报, 实况), ...]}
    series = {}
    for i, (_, m) in enumerate([(None, {(r["station"], r["date"], r["lead"]): r
                                        for r in base_rows})] + extras):
        for (stn, d, lead), r in m.items():
            f = r.get("temperature_2m_max")
            y = r.get("y_tmax")
            if f in ("", None) or y in ("", None):
                continue
            series.setdefault((i, stn, lead), []).append((d, float(f), float(y)))
    for v in series.values():
        v.sort()

    n_models = 1 + len(extras)
    out = {}
    for r in base_rows:
        stn, d, lead = r["station"], r["date"], r["lead"]
        hist, cur = {}, {}
        for i in range(n_models):
            h = series.get((i, stn, lead)) or []
            hist[i] = [x for x in h if x[0] < d]      # 严格因果
            now = ([x for x in h if x[0] == d] or [None])[0]
            if now:
                cur[i] = now[1]
        out[(stn, d, lead)] = deb_weights(hist, cur, lookback, decay, spread_cap)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mos.csv")
    ap.add_argument("--extra", nargs="+", required=True)
    ap.add_argument("--out", default="mos_multi.csv")
    ap.add_argument("--deb", action="store_true",
                    help="加 DEB 自适应加权融合列（借鉴 PolyWeather）")
    ap.add_argument("--dup-obs", action="store_true",
                    help="给纯实况列也加前缀（造重复列）。默认不加 —— 见文件头说明")
    args = ap.parse_args()

    base_map, base_rows = read(args.base)
    if not base_rows:
        print("base 为空", file=sys.stderr)
        return 1
    extras = []
    for p in args.extra:
        if not os.path.exists(p):
            print(f"[warn] 跳过不存在的 {p}", file=sys.stderr)
            continue
        m, _ = read(p)
        extras.append((os.path.basename(p), m))
        print(f"  {os.path.basename(p)}: {len(m)} 行", file=sys.stderr)

    ecols = [c for c in base_rows[0]
             if c not in SKIP and (args.dup_obs or c not in OBS_COLS)]
    out_cols = list(base_rows[0].keys())
    for i, _ in enumerate(extras, start=2):
        out_cols += [f"m{i}_{c}" for c in ecols] + [f"m{i}_present"]
    out_cols += ["ens_mean_tmax", "ens_spread_tmax", "ens_max_minus_min_tmax",
                 "ens_mean_cloud_peak", "ens_spread_cloud_peak"]
    deb = {}
    if args.deb:
        print("  计算 DEB 自适应权重…", file=sys.stderr)
        deb = deb_columns(base_rows, extras)
        out_cols += ["deb_pred", "deb_trust", "deb_minus_gfs"]

    n_full = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols)
        w.writeheader()
        for r in base_rows:
            k = (r["station"], r["date"], r["lead"])
            out = dict(r)
            tm, cl = [], []
            for v in (r.get("temperature_2m_max"),):
                if v not in ("", None):
                    tm.append(float(v))
            if r.get("cloud_cover_peakmean") not in ("", None):
                cl.append(float(r["cloud_cover_peakmean"]))
            for i, (_, m) in enumerate(extras, start=2):
                g = m.get(k) or {}
                for c in ecols:
                    out[f"m{i}_{c}"] = g.get(c, "")
                out[f"m{i}_present"] = 1 if g else 0
                t = g.get("temperature_2m_max")
                if t not in ("", None):
                    tm.append(float(t))
                c2 = g.get("cloud_cover_peakmean")
                if c2 not in ("", None):
                    cl.append(float(c2))
            if len(tm) >= 2:
                mu = sum(tm) / len(tm)
                out["ens_mean_tmax"] = round(mu, 3)
                out["ens_spread_tmax"] = round(
                    (sum((x - mu) ** 2 for x in tm) / len(tm)) ** .5, 3)
                out["ens_max_minus_min_tmax"] = round(max(tm) - min(tm), 3)
                n_full += 1
            else:
                out["ens_mean_tmax"] = out["ens_spread_tmax"] = ""
                out["ens_max_minus_min_tmax"] = ""
            if len(cl) >= 2:
                mu = sum(cl) / len(cl)
                out["ens_mean_cloud_peak"] = round(mu, 3)
                out["ens_spread_cloud_peak"] = round(
                    (sum((x - mu) ** 2 for x in cl) / len(cl)) ** .5, 3)
            else:
                out["ens_mean_cloud_peak"] = out["ens_spread_cloud_peak"] = ""
            if args.deb:
                p_, t_ = deb.get(k, (None, None))
                out["deb_pred"] = "" if p_ is None else round(p_, 3)
                out["deb_trust"] = "" if t_ is None else round(t_, 3)
                base_t = r.get("temperature_2m_max")
                out["deb_minus_gfs"] = ("" if (p_ is None or base_t in ("", None))
                                        else round(p_ - float(base_t), 3))
            w.writerow(out)

    print(f"\n写出 {args.out}: {len(base_rows)} 行 × {len(out_cols)} 列，"
          f"其中 {n_full} 行有 >=2 个模式成员")
    return 0


if __name__ == "__main__":
    sys.exit(main())

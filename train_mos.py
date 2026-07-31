#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_mos.py — 站点日最高温 MOS 训练与检验

    python3 train_mos.py mos.csv                  # 训练全部模型并比较
    python3 train_mos.py mos.csv --lead 1         # 只做 D+1
    python3 train_mos.py mos.csv --dump model.json --pred pred.csv

设计要点（都来自前期实测，不是拍脑袋）:

1. 预报残差 y - temperature_2m_max，不是直接预报温度。
   季节循环和天气型信号已在模式输出里，让模型重学一遍是浪费样本。

2. 时间切分，绝不随机划分。相邻日天气高度相关，随机划分会把
   测试日的邻居放进训练集，指标虚高。

3. 取整只在输出层。真值是整数度，取整能让 ±1℃ 命中率白涨约 15 个
   百分点而 MAE 几乎不变 —— 那是量化 artifact，拿它调参会走偏。
   训练和选超参全用连续值。

4. 站点效应很大且方向相反（盆地站偏冷近 2℃，沿海站偏暖 1.4℃），
   所以既做「合并 + 站点哑变量」也做「分站独立建模」，取优。

5. 依赖可选: 没有 numpy/sklearn 也能跑（标准库实现岭回归）。
   装了 lightgbm 或 sklearn 会自动加上梯度提升做对比。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime

# ---- 可选依赖 ----
try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

GBM = None
try:
    import lightgbm as lgb
    GBM = "lightgbm"
except ImportError:
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        GBM = "sklearn"
    except ImportError:
        pass


# ============================================================ 数据

def load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = {"station": r["station"], "date": r["date"], "lead": int(r["lead"])}
            for k, v in r.items():
                if k in ("station", "date", "lead"):
                    continue
                d[k] = float(v) if v not in ("", None) else None
            if d.get("y_tmax") is None or d.get("temperature_2m_max") is None:
                continue                       # 没有真值或没有基准，样本无用
            rows.append(d)
    return rows


# 对流/降水因子。数据已在 mos_fcst.sqlite 里（2026-07-31 拉的 106 万条），
# 但**实测未采用**，见下方 CONV_COLS 注释。这里必须显式排除:
# 本函数自动识别 csv 列，而 run_daily.sh 每天重建 mos.csv —— 不排除的话，
# 下次每周重训会把被否决的特征悄悄吃进生产模型。设 PLOYGON_CONV=1 可重测。
#
# 实测结果（2025-04-01~2026-07-30 滚动回测，按天配对 bootstrap）:
#   临近预报 13 时  0.4859 -> 0.4933   P(更好)=1%    显著变差
#   临近预报 11 时  0.7976 -> 0.8030   P(更好)=10%
#   D+1/D+2       1.3279 -> 1.3225   P(更好)=79%   不显著
#   广州(目标站)    0.791  -> 0.800    变差
# 原因: lifted_index 与 temperature_2m_max 相关 -0.85 近乎共线；
# cape 与 dew_point_2m_peakmean 相关 0.56。信息已被已有因子覆盖，
# 加进来只增加方差。样本内 R² 确实从 0.134 涨到 0.169 —— 那是过拟合的样子。
CONV_COLS = {"precipitation_max", "precipitation_peakmean",
             "cape_max", "cape_peakmean",
             "lifted_index_max", "lifted_index_peakmean"}


def feature_names(rows: list[dict]) -> list[str]:
    """自动识别数值特征列，排除标识列和真值。"""
    skip = {"station", "date", "lead", "y_tmax"}
    if os.environ.get("PLOYGON_CONV") != "1":
        skip |= CONV_COLS
    names = sorted({k for r in rows for k in r} - skip)
    return names


def derive(r: dict) -> dict:
    """派生特征。都是物理上有理由的组合，不是穷举。"""
    out = {}
    t = r.get("temperature_2m_max")
    dp = r.get("dew_point_2m_max")
    if t is not None and dp is not None:
        out["dewpoint_depression"] = t - dp        # 干燥度: 越干日较差越大
    pv = r.get("prev_tmax")
    if t is not None and pv is not None:
        out["fcst_minus_prev"] = t - pv            # 模式预报的日际变化
    cc = r.get("cloud_cover_peakmean")
    sw = r.get("shortwave_radiation_peakmean")
    if cc is not None and sw is not None:
        out["clear_index"] = sw * (1 - cc / 100.0)  # 有效到达辐射
    return out


def build_matrix(rows, feats, stations, med=None):
    """
    返回 (X, y_resid, base, meta)。
    y_resid = 真值 - 模式原始输出，即模型要学的订正量。
    缺测用训练期中位数填补，并加一列缺测指示 —— 缺测本身可能有信息。
    """
    if med is None:
        med = {}
        for f in feats:
            vals = sorted(r[f] for r in rows if r.get(f) is not None)
            med[f] = vals[len(vals) // 2] if vals else 0.0

    X, y, base, meta = [], [], [], []
    for r in rows:
        d = dict(r)
        d.update(derive(r))
        row = []
        for f in feats:
            v = d.get(f)
            row.append(med[f] if v is None else v)
            row.append(1.0 if v is None else 0.0)       # 缺测指示
        for s in stations:                              # 站点哑变量
            row.append(1.0 if r["station"] == s else 0.0)
        X.append(row)
        y.append(r["y_tmax"] - r["temperature_2m_max"])
        base.append(r["temperature_2m_max"])
        meta.append((r["station"], r["date"], r["y_tmax"]))
    return X, y, base, meta, med


def col_names(feats, stations):
    out = []
    for f in feats:
        out += [f, f + "__isnan"]
    return out + [f"stn_{s}" for s in stations]


# ============================================================ 岭回归（标准库）

def _solve(A, b):
    """高斯消元，部分主元。A: n×n，b: n。"""
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            M[c][c] += 1e-8                       # 退化时轻微加固
            p = c
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / pv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = M[r][n] - sum(M[r][k] * x[k] for k in range(r + 1, n))
        x[r] = s / M[r][r]
    return x


def ridge_fit(X, y, alpha, pen=None):
    """标准化后解正规方程。截距通过中心化隐式处理。

    pen: 每列的惩罚倍数（默认全 1）。用来给不同来源的特征分组正则 ——
    实况类特征（recent_bias/prev_tmax）比几十列模式特征信息密度高得多，
    统一惩罚会把它们压过头。给它们更小的倍数等价于更弱的正则。
    """
    n, p = len(X), len(X[0])
    if pen is None:
        pen = [1.0] * p
    if HAS_NP:
        Xa = np.asarray(X, float)
        ya = np.asarray(y, float)
        mx, sx = Xa.mean(0), Xa.std(0)
        sx[sx < 1e-9] = 1.0
        Z = (Xa - mx) / sx
        my = ya.mean()
        A = Z.T @ Z + alpha * np.diag(np.asarray(pen, float))
        w = np.linalg.solve(A, Z.T @ (ya - my))
        return {"w": w.tolist(), "mx": mx.tolist(), "sx": sx.tolist(), "my": float(my)}

    mx = [sum(r[j] for r in X) / n for j in range(p)]
    sx = []
    for j in range(p):
        v = math.sqrt(sum((r[j] - mx[j]) ** 2 for r in X) / n)
        sx.append(v if v > 1e-9 else 1.0)
    my = sum(y) / n
    Z = [[(r[j] - mx[j]) / sx[j] for j in range(p)] for r in X]
    yc = [v - my for v in y]
    A = [[0.0] * p for _ in range(p)]
    for i in range(n):
        zi = Z[i]
        for a in range(p):
            za = zi[a]
            if za:
                Aa = A[a]
                for b in range(a, p):
                    Aa[b] += za * zi[b]
    for a in range(p):
        for b in range(a):
            A[a][b] = A[b][a]
        A[a][a] += alpha * pen[a]
    bb = [sum(Z[i][a] * yc[i] for i in range(n)) for a in range(p)]
    return {"w": _solve(A, bb), "mx": mx, "sx": sx, "my": my}


def ridge_pred(m, X):
    w, mx, sx, my = m["w"], m["mx"], m["sx"], m["my"]
    if HAS_NP:
        return (my + (np.asarray(X, float) - np.asarray(mx)) / np.asarray(sx) @ np.asarray(w)).tolist()
    return [my + sum(w[j] * (r[j] - mx[j]) / sx[j] for j in range(len(w))) for r in X]


# ============================================================ 评估

def sc(errs):
    n = len(errs)
    if not n:
        return None
    me = sum(errs) / n
    return {"n": n, "me": me, "mae": sum(abs(e) for e in errs) / n,
            "rmse": math.sqrt(sum(e * e for e in errs) / n),
            "p1": 100 * sum(1 for e in errs if abs(e) <= 1) / n,
            "p2": 100 * sum(1 for e in errs if abs(e) <= 2) / n}


def show(tag, s, mark=""):
    if not s:
        print(f"  {tag:<26} (无样本)")
        return
    print(f"  {tag:<26} MAE={s['mae']:5.2f}  RMSE={s['rmse']:5.2f}  "
          f"ME={s['me']:+5.2f}  ±1℃={s['p1']:4.1f}%  ±2℃={s['p2']:4.1f}%{mark}")


def paired_boot(e_a, e_b, dates, n_boot=2000, seed=0):
    """按天块自助: 同一天各站误差相关，必须整天一起重采样。"""
    by = defaultdict(list)
    for ea, eb, d in zip(e_a, e_b, dates):
        by[d].append((abs(ea), abs(eb)))
    days = sorted(by)
    if len(days) < 10:
        return None
    flat = lambda ds: ([x for d in ds for x, _ in by[d]], [y for d in ds for _, y in by[d]])
    A, B = flat(days)
    obs = sum(A) / len(A) - sum(B) / len(B)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        ds = [rng.choice(days) for _ in days]
        a, b = flat(ds)
        diffs.append(sum(a) / len(a) - sum(b) / len(b))
    diffs.sort()
    return obs, diffs[int(.025 * n_boot)], diffs[int(.975 * n_boot)]


# ============================================================ 主流程

def run_lead(rows, lead, args):
    sub = [r for r in rows if r["lead"] == lead]
    if len(sub) < 200:
        print(f"D+{lead} 样本不足 ({len(sub)})，跳过")
        return None

    dates = sorted({r["date"] for r in sub})
    c1, c2 = dates[int(len(dates) * .70)], dates[int(len(dates) * .85)]
    tr = [r for r in sub if r["date"] < c1]
    va = [r for r in sub if c1 <= r["date"] < c2]
    te = [r for r in sub if r["date"] >= c2]

    print(f"\n{'='*74}\nD+{lead}   训练 {len(tr)} (<{c1})   验证 {len(va)} ({c1}~)   "
          f"测试 {len(te)} (>={c2})")

    stations = sorted({r["station"] for r in sub})
    base_feats = [f for f in feature_names(sub) if f != "temperature_2m_max"]
    # temperature_2m_max 本身作为基准已减掉，但它的绝对量级仍是有用的因子
    # （高温端偏差更大），所以保留在特征里
    base_feats.append("temperature_2m_max")
    derived = sorted({k for r in sub[:50] for k in derive(r)})
    feats = base_feats + derived

    Xtr, ytr, btr, mtr, med = build_matrix(tr, feats, stations)
    Xva, yva, bva, mva, _ = build_matrix(va, feats, stations, med)
    Xte, yte, bte, mte, _ = build_matrix(te, feats, stations, med)
    names = col_names(feats, stations)
    print(f"特征 {len(names)} 列（含缺测指示与站点哑变量）")

    # 实况/日历类特征给单独的惩罚倍数。列布局是 [f, f__isnan] * len(feats) + 站点哑变量
    # 注意不含 recent_bias: 多模式版里每个模式有自己的 recent_bias 列，
    # 它们是独立信息，不该被当成一组降正则
    OBS_F = {"prev_tmax", "doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2",
             "fcst_minus_prev"}
    def make_pen(mult, feats_, stations_):
        v = []
        for f in feats_:
            x = mult if f in OBS_F else 1.0
            v += [x, x]
        return v + [1.0] * len(stations_)

    te_dates = [m[1] for m in mte]
    truth = [m[2] for m in mte]
    results = {}

    # ---------- 基线 ----------
    print(f"\n── 基线（测试期）")
    e_raw = [b - t for b, t in zip(bte, truth)]
    show("模式原始", sc(e_raw))

    prev = [r.get("prev_tmax") for r in te]
    e_pers = [p - t for p, t in zip(prev, truth) if p is not None]
    show("持续性 prev_tmax", sc(e_pers))

    pb = defaultdict(list)
    for r in tr:
        pb[r["station"]].append(r["y_tmax"] - r["temperature_2m_max"])
    pbm = {k: sum(v) / len(v) for k, v in pb.items()}
    e_stn = [b + pbm.get(m[0], 0.0) - t for b, m, t in zip(bte, mte, truth)]
    show("模式+分站常数", sc(e_stn))

    rb = [r.get("recent_bias") for r in te]
    e_rb = [b + (x or 0.0) - t for b, x, t in zip(bte, rb, truth)]
    show("模式+recent_bias", sc(e_rb))
    results["baseline_best"] = min(
        [("模式原始", e_raw), ("分站常数", e_stn), ("recent_bias", e_rb)],
        key=lambda kv: sum(abs(e) for e in kv[1]) / len(kv[1]))

    # ---------- 岭回归: 合并 ----------
    print(f"\n── 岭回归（合并，站点哑变量）")
    best = (None, 1e9, None, 1.0)
    for a in args.alphas:
        for mult in args.obs_pen:
            m = ridge_fit(Xtr, ytr, a, make_pen(mult, feats, stations))
            pv = ridge_pred(m, Xva)
            mae = sum(abs(p + b - r["y_tmax"])
                      for p, b, r in zip(pv, bva, va)) / len(va)
            if mae < best[1]:
                best = (a, mae, m, mult)
    alpha, _, mdl, obs_mult = best
    print(f"  实况类特征惩罚倍数 {obs_mult:g}（1 = 与模式特征同等正则）")
    pte = ridge_pred(mdl, Xte)
    e_ridge = [p + b - t for p, b, t in zip(pte, bte, truth)]
    key_te = [(m[0], m[1]) for m in mte]
    err_by = {"岭回归(合并)": dict(zip(key_te, e_ridge))}
    show(f"岭回归 (alpha={alpha:g})", sc(e_ridge))
    show("  取整后（上线口径）", sc([round(p + b) - t for p, b, t in zip(pte, bte, truth)]))
    results["ridge"] = (mdl, alpha, e_ridge, names)

    # ---------- 岭回归: 分站 ----------
    print(f"\n── 岭回归（分站独立建模）")
    # 合并 vs 分站要在验证集上选，不能看测试集。特征列多了以后（多模式 346 列）
    # 分站每站只有几百样本，会过拟合，架构优劣会翻转 —— 必须真的比一次。
    va_pool = (sum(abs(p + b - r["y_tmax"])
                   for p, b, r in zip(ridge_pred(mdl, Xva), bva, va)) / len(va)
               if va else None)
    va_per_err = []
    e_per, ok, per_models = [], True, {}
    for s in stations:
        trs = [r for r in tr if r["station"] == s]
        vas = [r for r in va if r["station"] == s]
        tes = [r for r in te if r["station"] == s]
        if len(trs) < 100 or not tes:
            ok = False
            continue
        Xs, ys, bs, ms, meds = build_matrix(trs, feats, [])
        Xv, yv, bv, mv, _ = build_matrix(vas, feats, [], meds) if vas else (None,) * 5
        Xt, yt, bt, mt, _ = build_matrix(tes, feats, [], meds)
        bb = (args.alphas[0], 1e9, None)
        for a in args.alphas:
            for mult in args.obs_pen:
                mm = ridge_fit(Xs, ys, a, make_pen(mult, feats, []))
                if Xv:
                    pv = ridge_pred(mm, Xv)
                    mae = sum(abs(p + b - m[2])
                              for p, b, m in zip(pv, bv, mv)) / len(mv)
                else:
                    mae = a
                if mae < bb[1]:
                    bb = (a, mae, mm)
        per_models[s] = {"alpha": bb[0], "model": bb[2], "median": meds}
        if vas:
            va_per_err += [abs(p + b - m[2]) for p, b, m
                           in zip(ridge_pred(bb[2], Xv), bv, mv)]
        pt = ridge_pred(bb[2], Xt)
        e_per += [((m[0], m[1]), p + b - m[2], round(p + b) - m[2])
                  for p, b, m in zip(pt, bt, mt)]
    if e_per:
        show("分站岭回归", sc([x[1] for x in e_per]))
        show("  取整后（上线口径）", sc([x[2] for x in e_per]))
        err_by["岭回归(分站)"] = {k: v for k, v, _ in e_per}
        results["ridge_per"] = per_models
        if va_pool is not None and va_per_err:
            va_per = sum(va_per_err) / len(va_per_err)
            results["va_pool"], results["va_per"] = va_pool, va_per
            results["prefer"] = "per_station" if va_per < va_pool else "pooled"
    elif not ok:
        print("  部分站训练样本不足，跳过")

    # ---------- 梯度提升 ----------
    if GBM and not args.no_gbm:
        print(f"\n── 梯度提升 ({GBM})")
        try:
            if GBM == "lightgbm":
                ds = lgb.Dataset(np.asarray(Xtr), label=np.asarray(ytr))
                dv = lgb.Dataset(np.asarray(Xva), label=np.asarray(yva))
                p = {"objective": "l1", "learning_rate": .05, "num_leaves": 15,
                     "min_data_in_leaf": 40, "feature_fraction": .8,
                     "bagging_fraction": .8, "bagging_freq": 1, "verbose": -1}
                bst = lgb.train(p, ds, 2000, valid_sets=[dv],
                                callbacks=[lgb.early_stopping(80, verbose=False)])
                pte2 = bst.predict(np.asarray(Xte))
                imp = sorted(zip(names, bst.feature_importance("gain")),
                             key=lambda x: -x[1])[:10]
            else:
                import numpy as _np
                Xa, ya = _np.asarray(Xtr), _np.asarray(ytr)
                Xv, bv_ = _np.asarray(Xva), _np.asarray(bva)
                yv_true = _np.asarray([r["y_tmax"] for r in va])
                # 用验证集手动早停。sklearn 的 early_stopping 只能随机切分或
                # 用训练集打分，对时间序列都不合适；且必须与岭回归同用训练集，
                # 否则 GBM 多吃验证集那部分数据，比较就不公平了
                g = HistGradientBoostingRegressor(
                    loss="absolute_error", learning_rate=.05,
                    max_leaf_nodes=15, min_samples_leaf=40,
                    early_stopping=False, warm_start=True,
                    random_state=0, max_iter=1)
                best_it, best_mae, bad = 50, 1e9, 0
                for it in range(50, 1201, 50):
                    g.set_params(max_iter=it)
                    g.fit(Xa, ya)
                    mae = float(_np.abs(g.predict(Xv) + bv_ - yv_true).mean())
                    if mae < best_mae - 1e-4:
                        best_it, best_mae, bad = it, mae, 0
                    else:
                        bad += 1
                        if bad >= 3:
                            break
                g.set_params(max_iter=best_it, warm_start=False)
                g.fit(Xa, ya)
                print(f"  早停于 {best_it} 轮（验证 MAE {best_mae:.3f}）")
                pte2 = g.predict(_np.asarray(Xte))
                imp = None
            # 融合权重在验证集上选。岭回归与 GBM 误差结构不同（GBM 的 ME 更接近 0），
            # 平均后两者都显著更优 —— 实测 D+1 1.080/1.072 -> 1.031
            pv_r = ridge_pred(mdl, Xva)
            pv_g = (bst.predict(np.asarray(Xva)) if GBM == "lightgbm"
                    else g.predict(_np.asarray(Xva)))
            yv_true2 = [r["y_tmax"] for r in va]
            bw, bmae = 1.0, 1e9
            for w in [i / 10 for i in range(11)]:
                mae = sum(abs(round(w * pr_ + (1 - w) * pg_ + b) - t)
                          for pr_, pg_, b, t in zip(pv_r, pv_g, bva, yv_true2)) / len(va)
                if mae < bmae:
                    bw, bmae = w, mae
            results["gbm_obj"] = bst if GBM == "lightgbm" else g
            results["blend_w"] = bw
            # 架构选择必须把融合也算进来。只比「合并 vs 分站」的话，
            # 分站可能以微弱优势胜出，却把明显更好的融合挡在门外
            results["va_blend"] = bmae
            pbl = [bw * pr_ + (1 - bw) * pg_ for pr_, pg_ in zip(pte, pte2)]
            show(f"融合 (岭回归 w={bw:g})",
                 sc([round(p + b) - t for p, b, t in zip(pbl, bte, truth)]), "  <- 上线用这个")
            err_by["岭回归+GBM 融合"] = dict(zip(key_te,
                [p + b - t for p, b, t in zip(pbl, bte, truth)]))

            e_gbm = [float(p) + b - t for p, b, t in zip(pte2, bte, truth)]
            show("梯度提升", sc(e_gbm))
            show("  取整后", sc([round(float(p) + b) - t
                                for p, b, t in zip(pte2, bte, truth)]))
            results["gbm"] = e_gbm
            err_by["梯度提升"] = dict(zip(key_te, e_gbm))
            if imp:
                print("  增益最高的特征:")
                for k, v in imp:
                    print(f"    {k:<32}{v:>12.0f}")
        except Exception as e:
            print(f"  失败: {type(e).__name__}: {e}")
    elif not GBM:
        print(f"\n（未装 lightgbm/scikit-learn，跳过梯度提升。"
              f"pip install lightgbm 可启用）")

    # ---------- 架构最终选择（三方，全部看验证集）----------
    cand = {}
    if results.get("va_pool") is not None:
        cand["pooled"] = results["va_pool"]
    if results.get("va_per") is not None:
        cand["per_station"] = results["va_per"]
    if results.get("va_blend") is not None:
        cand["blend"] = results["va_blend"]
    if cand:
        results["prefer"] = min(cand, key=cand.get)
        print(f"\n── 架构选择（验证集 MAE）")
        for k, v in sorted(cand.items(), key=lambda x: x[1]):
            mark = "  <- 上线用这个" if k == results["prefer"] else ""
            print(f"  {k:<12}{v:.3f}{mark}")

    # ---------- 显著性 ----------
    bl_name, bl_err = results["baseline_best"]
    print(f"\n── 对最强基线（{bl_name}）的配对检验，95% 区间跨 0 即无显著差异")
    for tag, key in (("岭回归", "ridge"), ("梯度提升", "gbm")):
        if key not in results:
            continue
        err = results[key][2] if key == "ridge" else results[key]
        r = paired_boot(bl_err, err, te_dates)
        if r:
            d, lo, hi = r
            verdict = ("模型显著更优" if lo > 0 else
                       "基线显著更优" if hi < 0 else "无显著差异")
            print(f"  {tag:<10} ΔMAE={d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}")

    ms = [k for k, v in err_by.items() if v]
    if len(ms) > 1:
        print(f"\n── 模型间配对比较（区间跨 0 就选更简单的那个）")
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                ck = sorted(set(err_by[a]) & set(err_by[b]))
                r = paired_boot([err_by[a][k] for k in ck],
                                [err_by[b][k] for k in ck], [k[1] for k in ck])
                if r:
                    d, lo, hi = r
                    v = (f"{b} 更优" if lo > 0 else
                         f"{a} 更优" if hi < 0 else "无显著差异")
                    print(f"  {a} vs {b:<14} ΔMAE={d:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {v}")

    # ---------- 分站 ----------
    print(f"\n── 分站（岭回归 vs 模式原始）")
    print(f"  {'站点':<8}{'n':>5}{'原始MAE':>10}{'岭回归':>9}{'改进':>8}")
    for s in stations:
        idx = [i for i, m in enumerate(mte) if m[0] == s]
        if not idx:
            continue
        a = sum(abs(e_raw[i]) for i in idx) / len(idx)
        b = sum(abs(e_ridge[i]) for i in idx) / len(idx)
        print(f"  {s:<8}{len(idx):>5}{a:>10.2f}{b:>9.2f}{a-b:>+8.2f}")

    # ---------- 系数 ----------
    if args.coef:
        print(f"\n── 岭回归系数（标准化尺度，绝对值前 15）")
        w = results["ridge"][0]["w"]
        for k, v in sorted(zip(names, w), key=lambda x: -abs(x[1]))[:15]:
            print(f"  {k:<34}{v:+8.3f}")

    return {"lead": lead, "model": mdl, "alpha": alpha, "names": names,
            "prefer": results.get("prefer", "per_station"),
            "gbm_obj": results.get("gbm_obj"), "blend_w": results.get("blend_w"),
            "per_station": results.get("ridge_per"),
            "feats": feats, "stations": stations, "med": med,
            "pred": [(m[0], m[1], round(p + b, 2), round(p + b), m[2])
                     for p, b, m in zip(pte, bte, mte)]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="mos.csv")
    ap.add_argument("--lead", type=int, action="append",
                    help="只跑指定时效，可重复。默认全部")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.3, 1, 3, 10, 30, 100, 300])
    ap.add_argument("--obs-pen", type=float, nargs="+",
                    default=[1.0, 0.5, 0.2, 0.14, 0.1],
                    help="实况类特征的惩罚倍数候选，在验证集上选。"
                         "多模式下模式特征多达 280 列，实况类特征会被压过头")
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--coef", action="store_true", help="打印岭回归系数")
    ap.add_argument("--dump", help="把模型存成 JSON，供上线使用")
    ap.add_argument("--pred", help="把测试期预报写成 CSV")
    args = ap.parse_args()

    rows = load(args.csv)
    if not rows:
        print("没读到样本", file=sys.stderr)
        return 1
    print(f"{len(rows)} 行 | numpy={'有' if HAS_NP else '无'} | GBM={GBM or '无'}")

    leads = args.lead or sorted({r["lead"] for r in rows})
    out = {}
    for L in leads:
        res = run_lead(rows, L, args)
        if res:
            out[L] = res

    if args.dump and out:
        j = {}
        for L, r in out.items():
            e = {"alpha": r["alpha"], "names": r["names"], "feats": r["feats"],
                 "stations": r["stations"], "median": r["med"], "model": r["model"],
                 "prefer": r["prefer"]}
            if r.get("gbm_obj") is not None:
                e["blend_w"] = r["blend_w"]
            if r.get("per_station"):
                # 分站模型不含站点哑变量，特征列即 feats（各带一列缺测指示）
                e["per_station"] = r["per_station"]
            j[str(L)] = e
        json.dump(j, open(args.dump, "w"), ensure_ascii=False, indent=1)
        print(f"\n模型已存 {args.dump}")
        # GBM 没法序列化成 JSON，单独 pickle。预测端没有 sklearn 时会自动只用岭回归
        gb = {str(L): r["gbm_obj"] for L, r in out.items() if r.get("gbm_obj") is not None}
        if gb:
            import pickle
            with open(args.dump + ".gbm.pkl", "wb") as fh:
                pickle.dump(gb, fh)
            print(f"GBM 部分已存 {args.dump}.gbm.pkl（预测端缺 sklearn 会自动降级）")

    if args.pred and out:
        with open(args.pred, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["lead", "station", "date", "pred", "pred_round", "obs"])
            for L, r in out.items():
                for p in r["pred"]:
                    w.writerow([L, *p])
        print(f"测试期预报已存 {args.pred}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
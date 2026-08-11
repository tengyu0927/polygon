#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backtest_nowcast.py — 逐日滚动回测: 分站 × 分起报时的命中率

    python3 backtest_nowcast.py --db cn.sqlite --nwp-csv mos.csv \
        --cutoffs 10 11 12 --start 2026-06-25 --end 2026-07-24

与 train_nowcast.py 的检验的区别: 那边是「一次切分、一个测试窗口」，
这边是「按块滚动重训」，每块只用块开始日之前的数据训练，
包括气候态也重算 —— 模拟真实业务里 6 月 25 日 10 点起报时能拿到的全部信息。

命中口径: round(预报) == 实际日最高温（整数度）。这是用户关心的「完全一致」。
同时给 ±1℃ 和 MAE，因为完全命中率对 0.5℃ 边界极敏感，单看它会误判优劣。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import train_mos as T
import train_nowcast as N


# 两段式的第一段: 先预报「今天几点见顶」，再把它喂给主模型。
# 动机是实测误差随见顶时刻单调上升（13 时截止: ≤13时见顶 MAE 0.28，
# 16-17时见顶 0.77），而模型只有 clim_peak_h 这个分站分月常数，
# 知道「重庆通常晚」却不知道「今天会不会晚」。
#
# **实测未采用**（--peak-feat 默认关）。七档配对检验:
#   9/10/11 时 ΔMAE -0.014 / -0.008 / -0.012（方向偏坏，均不显著）
#   12/14/15 时 +0.003 / +0.007 / +0.003（均不显著）
#   13 时     +0.019 [+0.007, +0.033] 显著 —— 但测了 7 档出现 1 档显著，
#             大概率是多重比较的产物，区间下界也只有 +0.007
# 决定性证据是目标人群没站住: 16 时后见顶的 102 天上，七档变化
# +0.010/0.000/-0.029/+0.020/-0.020/-0.020/-0.010，正负交替、无一致方向。
# 机制若成立，这一层该全线改善。开关保留，样本更多时可重测。
# 2026-08-01 重测: 上面那次失败的原因是**辅助模型太弱**。原来用岭回归 +
# 只有基础特征，实测见顶时刻 MAE 2.21h -> 2.19h（等于没预测）。
# 换成 GBM + 全部 70 项特征（含六模式的午后廓线 t2m_peak_h / t2m_slope_pm /
# t2m_late_minus_peak）后: 2.21h -> 1.93h，重庆 3.11 -> 2.49；
# 「≥15 时见顶」的分类 AUC 0.65-0.78。所以值得重开一次。
# 先知实验（PLOYGON_ORACLE_PEAK=1）给出的上限是 -26.6%。
PEAK_FEATS = ["pk_pred", "pk_minus_clim", "pk_minus_cutoff", "pk_p_late"]

try:
    from sklearn.ensemble import HistGradientBoostingRegressor as _GBR
    from sklearn.ensemble import HistGradientBoostingClassifier as _GBC
    import numpy as _np
    PEAK_GBM = True
except ImportError:
    PEAK_GBM = False


def add_peak_feats(rows, model, med, names, cutoff):
    """把辅助模型预报的见顶时刻写进特征。model 为 None 时全部置空。"""
    if model is None:
        for r in rows:
            for k in PEAK_FEATS:
                r["f"][k] = None
        return
    X, _ = N.matrix(rows, med, names)
    reg, clf = model if isinstance(model, tuple) else (model, None)
    if clf is not None or PEAK_GBM and not isinstance(reg, dict):
        Xa = _np.asarray(X, float)
        pv = reg.predict(Xa)
        pl = clf.predict_proba(Xa)[:, 1] if clf is not None else [None] * len(rows)
    else:
        pv = T.ridge_pred(reg, X)
        pl = [None] * len(rows)
    for r, p, q in zip(rows, pv, pl):
        p = min(22.0, max(8.0, float(p)))     # 见顶时刻限定在 8-22 时
        r["f"]["pk_pred"] = p
        cp = r["f"].get("clim_peak_h")
        r["f"]["pk_minus_clim"] = None if cp is None else p - cp
        r["f"]["pk_minus_cutoff"] = p - cutoff
        r["f"]["pk_p_late"] = None if q is None else float(q)


def clim_before(days, cutoffs, cut_date):
    """只用 cut_date 之前的日子估气候态。逐块重算，避免用到未来。"""
    rise = {c: defaultdict(list) for c in cutoffs}
    peak = defaultdict(list)
    for (stn, d), hrs in days.items():
        if d >= cut_date:
            continue
        mo = int(d[5:7])
        tmax = max(v["t"] for v in hrs.values())
        ph = [h for h, v in hrs.items() if v["t"] >= tmax - 1e-9]
        peak[(stn, mo)].append(sum(ph) / len(ph))
        for c in cutoffs:
            o = N.morning(hrs, c)
            if o:
                rise[c][(stn, mo)].append(tmax - max(v["t"] for v in o.values()))
    R = {c: {k: sum(v) / len(v) for k, v in m.items() if len(v) >= 20}
         for c, m in rise.items()}
    P = {k: sum(v) / len(v) for k, v in peak.items() if len(v) >= 20}
    return R, P


def blocks(start, end, days_per):
    """把评估期切成若干块，每块开始前重训一次。"""
    out, cur = [], start
    while cur <= end:
        nxt = min(cur + timedelta(days=days_per - 1), end)
        out.append((cur, nxt))
        cur = nxt + timedelta(days=1)
    return out


def fit_block(tr, names, alphas, val_days=90, peak=False, cutoff=12):
    """分站两段式 + 合并兜底。alpha 用训练期末尾 val_days 天选。"""
    ds = sorted({r["date"] for r in tr})
    if len(ds) < val_days * 2:
        vcut = ds[int(len(ds) * .8)]
    else:
        vcut = ds[-val_days]
    fit_tr = [r for r in tr if r["date"] < vcut]
    fit_va = [r for r in tr if r["date"] >= vcut]
    if not fit_tr or not fit_va:
        fit_tr, fit_va = tr, tr

    def fit_peak(rows_tr, names_):
        """辅助模型: 用上午特征预报当天见顶时刻 + 「晚见顶」概率。

        岭回归版实测等于没预测（2.21h -> 2.19h）。GBM 版 2.21h -> 1.93h，
        「>=15 时见顶」AUC 0.65-0.78，所以这里优先用 GBM。
        """
        sub = [r for r in rows_tr if r.get("peak_h") is not None]
        if len(sub) < 300:
            return None
        X, med_ = N.matrix(sub, None, names_)
        y = [float(r["peak_h"]) for r in sub]
        if not PEAK_GBM:
            return T.ridge_fit(X, y, 10.0), med_
        Xa = _np.asarray(X, float)
        reg = _GBR(max_iter=400, learning_rate=.05, max_leaf_nodes=31,
                   min_samples_leaf=20, l2_regularization=1.0,
                   random_state=0).fit(Xa, _np.asarray(y))
        lab = _np.asarray([1 if v >= 15 else 0 for v in y])
        clf = None
        if 0 < lab.sum() < len(lab):
            clf = _GBC(max_iter=300, learning_rate=.05, max_leaf_nodes=31,
                       min_samples_leaf=20, random_state=0).fit(Xa, lab)
        return (reg, clf), med_

    def one(rows_tr, rows_va):
        X, med = N.matrix(rows_tr, None, names)
        Xv, _ = N.matrix(rows_va, med, names)
        y = [r["rise"] for r in rows_tr]
        best = (alphas[0], 1e9, None)
        for a in alphas:
            m = T.ridge_fit(X, y, a)
            pv = [max(0.0, v) for v in T.ridge_pred(m, Xv)]
            mae = sum(abs(x + r["so_far"] - r["tmax"])
                      for x, r in zip(pv, rows_va)) / len(rows_va)
            if mae < best[1]:
                best = (a, mae, m)
        # 非线性一路（A/B 测试中，PLOYGON_NLIN=1 打开）。
        # 临近预报到 2026-08-05 为止是**纯线性的**（ridge + 两段式的 cls/reg
        # 都是线性），而 D+1 那边早就是 ridge+GBM 融合、融合比纯 ridge 好 0.055。
        # 线性模型表达不了交互: 「静风 × 剩余升幅大」只能给风速一个固定权重。
        # add_interactions 里手工枚举的那几个交互项实测无收益 —— 枚举本来就
        # 枚举不过来，这正是该交给树模型的地方。
        # 融合权重在验证集上选，与 train_mos 同法。
        gbm = None
        if os.environ.get("PLOYGON_NLIN") == "1" and PEAK_GBM:
            Xa = _np.asarray(X, float)
            g = _GBR(max_depth=3, max_iter=300, learning_rate=0.06,
                     min_samples_leaf=20, random_state=0).fit(Xa, _np.asarray(y, float))
            pr = [max(0.0, v) for v in T.ridge_pred(best[2], Xv)]
            pg = [max(0.0, v) for v in g.predict(_np.asarray(Xv, float))]
            bw, bm = 1.0, 1e9
            for w in (1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0):
                e = sum(abs(w * a + (1 - w) * b + r["so_far"] - r["tmax"])
                        for a, b, r in zip(pr, pg, rows_va)) / len(rows_va)
                if e < bm:
                    bw, bm = w, e
            gbm = (g, bw)

        # 直接优化命中率的一路（A/B 测试中，PLOYGON_CLS=1 打开）。
        #
        # 动机: 实况温度是整数、已达也是整数，所以 rise 恒是**非负整数** ——
        # 这本来就是分类问题，被当成回归做了。平方误差优化「离真值多远」，
        # 而用户的损失是「等不等于真值」，两者在 x.5 附近直接冲突。
        # 实测: 加了 GBM 之后 MAE 0.7200 -> 0.7042，但完全命中 44.3% -> 44.5%
        # 几乎没动 —— 误差被压缩了，没被消灭。
        #
        # 与已否决的 pred_mode 区分: 那个众数来自 fit_ordinal（**线性**概率模型
        # 逐个拟合 P(rise>=k) 再差分），代码注释自己写着「尾部标定差」，
        # 9 时 28.5% vs 均值 36.0%。0/1 损失的最优解**就是**众数，前提是概率
        # 估得准 —— 之前输在标定，不是输在众数本身。这里换成多分类 GBM 的
        # 对数损失，那是直接在优化概率标定。
        #
        # PLOYGON_CLS_TIE: 前二类概率差小于它时，倒向回归值那一侧。
        # 防的是「分布平坦时 argmax 每小时翻」——那与「别来回跳」的诉求冲突。
        cls = None
        if os.environ.get("PLOYGON_CLS") == "1" and PEAK_GBM:
            lab = [min(8, max(0, int(round(r["rise"])))) for r in rows_tr]
            if len(set(lab)) >= 3:
                cls = _GBC(max_depth=3, max_iter=300, learning_rate=0.06,
                           min_samples_leaf=20, random_state=0).fit(
                               _np.asarray(X, float), lab)

        q90 = N.fit_quantile(rows_tr, 0.90, best[0], med, names)
        # 方差膨胀用的中心: 训练集上预报升幅的均值。绕它做 pred' = mu + c*(pred-mu)，
        # 双向调整 —— 大升幅往上推、零升幅往下压，不像换 P90 那样单向抬高
        Xtr_, _ = N.matrix(rows_tr, med, names)
        ptr = N.pred_hurdle(N.fit_hurdle(rows_tr, [best[0]], med, names), Xtr_) \
            if N.fit_hurdle(rows_tr, [best[0]], med, names) else \
            [max(0.0, v) for v in T.ridge_pred(best[2], Xtr_)]
        rise_mu = sum(ptr) / len(ptr) if ptr else 0.0
        return {"median": med, "ridge": best[2], "alpha": best[0],
                "hurdle": N.fit_hurdle(rows_tr, [best[0]], med, names),
                "ordinal": N.fit_ordinal(rows_tr, best[0], med, names),
                "q90": q90, "rise_mu": rise_mu, "gbm": gbm, "cls": cls}

    # 第一段: 见顶时刻辅助模型。只用训练集拟合，再把预报值注入全部行的特征
    base_names = [n for n in names if n not in PEAK_FEATS]
    peak_model = None
    if peak:
        got = fit_peak(fit_tr, base_names)
        if got:
            peak_model, peak_med = got
            for rs in (fit_tr, fit_va):
                add_peak_feats(rs, peak_model, peak_med, base_names, cutoff)

    pooled = one(fit_tr, fit_va)
    per = {}
    for stn in sorted({r["stn"] for r in tr}):
        s_tr = [r for r in fit_tr if r["stn"] == stn]
        s_va = [r for r in fit_va if r["stn"] == stn]
        if len(s_tr) >= 300 and s_va:
            per[stn] = one(s_tr, s_va)
    if peak and peak_model is not None:
        pooled["peak"] = (peak_model, peak_med, base_names, cutoff)
        for v in per.values():
            v["peak"] = (peak_model, peak_med, base_names, cutoff)
    return pooled, per


MOS_FEATS = ["mos_d1", "mos_d1_minus_sofar"]
M2_COLS = N.M2_COLS
ENS_FEATS = N.ENS_FEATS
m2_feature_names = N.m2_feature_names


def mos_block(mos_rows, cut_date, alphas, val_days=120):
    """路线组合: 用块开始日之前的 mos.csv 训一个 D+1 MOS，输出订正后的 Tmax。

    直接把 model.json 拿来用会泄漏（它见过评估期），所以每块重训。
    目标量与 train_mos.py 一致: 残差 y - temperature_2m_max。
    返回 {(站, 日): 订正后 Tmax}，只对 cut_date 及之后的日子出预报。
    """
    sub = [r for r in mos_rows if r["lead"] == 1]
    tr = [r for r in sub if r["date"] < cut_date]
    te = [r for r in sub if r["date"] >= cut_date]
    if len(tr) < 300 or not te:
        return {}
    feats = [f for f in T.feature_names(tr) if f != "temperature_2m_max"]
    feats.append("temperature_2m_max")
    derived = sorted({k for r in tr[:50] for k in T.derive(r)})
    feats += derived

    def prep(rows):
        return [{"f": dict(r, **T.derive(r))} for r in rows]

    ds = sorted({r["date"] for r in tr})
    vcut = ds[-val_days] if len(ds) > val_days * 2 else ds[int(len(ds) * .8)]
    out = {}
    for stn in sorted({r["station"] for r in sub}):
        s_tr = [r for r in tr if r["station"] == stn]
        s_te = [r for r in te if r["station"] == stn]
        if len(s_tr) < 200 or not s_te:
            continue
        f_tr = [r for r in s_tr if r["date"] < vcut]
        f_va = [r for r in s_tr if r["date"] >= vcut]
        if not f_tr or not f_va:
            f_tr, f_va = s_tr, s_tr
        X, med = N.matrix(prep(f_tr), None, feats)
        Xv, _ = N.matrix(prep(f_va), med, feats)
        y = [r["y_tmax"] - r["temperature_2m_max"] for r in f_tr]
        best = (alphas[0], 1e9, None)
        for a in alphas:
            m = T.ridge_fit(X, y, a)
            mae = sum(abs(p + r["temperature_2m_max"] - r["y_tmax"])
                      for p, r in zip(T.ridge_pred(m, Xv), f_va)) / len(f_va)
            if mae < best[1]:
                best = (a, mae, m)
        Xt, _ = N.matrix(prep(s_te), med, feats)
        for r, p in zip(s_te, T.ridge_pred(best[2], Xt)):
            out[(r["station"], r["date"])] = p + r["temperature_2m_max"]
    return out


def mos_walkforward(mos_rows, alphas, step=30, min_train=400):
    """整段滚动生成 D+1 MOS 预报。临近模型的训练期也必须拿到样本外的 MOS 值，
    否则学到的组合权重是在「MOS 见过这些天」的前提下估的，会高估它。"""
    ds = sorted({r["date"] for r in mos_rows if r["lead"] == 1})
    if len(ds) < min_train + step:
        return {}
    out = {}
    i = min_train
    while i < len(ds):
        cut = ds[i]
        stop = ds[min(i + step, len(ds) - 1)]
        blk = mos_block(mos_rows, cut, alphas)
        for (stn, d), v in blk.items():
            if cut <= d < stop or (i + step >= len(ds) and d >= cut):
                out[(stn, d)] = v
        i += step
    return out


def predict(m, rows, names, hurdle, thresh=0.0):
    """thresh > 0 时启用「硬判定见顶」: P(会升) 低于阈值就直接输出 0。

    动机: 两段式现在输出 P(会升) × E[升幅|会升]，即使 P=0.3 也会给 0.6 度，
    **永远不会真正判定见顶**。实测「已见顶判定」准确率只有 75-83%，
    且漏判是误判的 2-3 倍（12 时 113 vs 43）—— 这种不对称说明是阈值问题，
    不是信息不足。完美判定的上界: 12 时 MAE 0.610 -> 0.430。
    """
    X, _ = N.matrix(rows, m["median"], names)
    if hurdle and m["hurdle"]:
        if thresh <= 0:
            out = N.pred_hurdle(m["hurdle"], X)
        else:
            pp = [min(1.0, max(0.0, v)) for v in T.ridge_pred(m["hurdle"]["cls"], X)]
            rr = T.ridge_pred(m["hurdle"]["reg"], X)
            out = [0.0 if p < thresh else p * max(0.0, r) for p, r in zip(pp, rr)]
    else:
        out = [max(0.0, v) for v in T.ridge_pred(m["ridge"], X)]
    g = m.get("gbm")
    if g:
        mod, w = g
        pg = [max(0.0, v) for v in mod.predict(_np.asarray(X, float))]
        out = [w * a + (1 - w) * b for a, b in zip(out, pg)]

    c = m.get("cls")
    if c is not None:
        tie = float(os.environ.get("PLOYGON_CLS_TIE") or 0.0)
        Pm = c.predict_proba(_np.asarray(X, float))
        ks = list(c.classes_)
        new = []
        for p, reg in zip(Pm, out):
            order = sorted(range(len(ks)), key=lambda i: -p[i])
            k0 = ks[order[0]]
            if tie > 0 and len(order) > 1 and p[order[0]] - p[order[1]] < tie:
                # 前二类难分时，选离回归值更近的那个 —— 平抑逐小时翻转
                k1 = ks[order[1]]
                k0 = min((k0, k1), key=lambda k: abs(k - reg))
            new.append(float(k0))
        out = new
    return out


def rise_pmf(m, rows, names):
    if not m.get("ordinal"):
        return None
    X, _ = N.matrix(rows, m["median"], names)
    return N.rise_pmf(m["ordinal"], X)


def decide(pmf, half_width):
    """给定整数升幅分布，选让 P(|误差| <= half_width) 最大的那档。
    half_width=0 就是众数（完全命中最优），=1 是 ±1℃ 命中最优。
    指标换了，决策规则就得跟着换 —— 用均值取整去追 ±1℃ 命中是错配的。"""
    out = []
    for p in pmf:
        best, bv = 0, -1.0
        for k in range(len(p)):
            v = sum(p[max(0, k - half_width):k + half_width + 1])
            if v > bv:
                best, bv = k, v
        out.append(float(best))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cn.sqlite")
    ap.add_argument("--table", default="obs")
    ap.add_argument("--cutoffs", type=int, nargs="+", default=[10, 11, 12])
    ap.add_argument("--start", required=True, help="评估期起（含）")
    ap.add_argument("--end", required=True, help="评估期止（含）")
    ap.add_argument("--retrain-days", type=int, default=30,
                    help="每隔几天重训一次（默认 30，即按月滚动）")
    ap.add_argument("--nwp-csv", default="", help="加 GFS 特征（限 2024 年后样本）")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1, 3, 10, 30, 100, 300])
    ap.add_argument("--nwp-csv2", nargs="+", default=[],
                    help="追加模式的 mos 格式 csv（可多个），加多模式与集合离散度特征")
    ap.add_argument("--hurdle-thresh", type=float, default=0.0,
                    help="两段式硬判定阈值: P(会升) 低于它就直接输出 0（已见顶）")
    ap.add_argument("--inflate", type=float, default=1.0,
                    help="方差膨胀系数。**实测未采用**: 校准确实改善"
                         "（>=3度升幅欠报 0.90->0.66），但晚见顶日 MAE 反而从"
                         "0.868 涨到 0.892、总 MAE 0.583->0.605。1.0=不启用")
    ap.add_argument("--peak-feat", action="store_true",
                    help="加「预报见顶时刻」辅助模型的输出当特征（实验）")
    ap.add_argument("--blend-mos", action="store_true",
                    help="把滚动重训的 D+1 MOS 预报当特征加进来（组合两条路线）")
    ap.add_argument("--no-hurdle", action="store_true")
    ap.add_argument("--daily", action="store_true", help="逐日逐站明细")
    ap.add_argument("--mos-oos", default="",
                    help="train_mos --pred 导出的样本外 D+1 预报 CSV")
    ap.add_argument("--csv-out", default="", help="把逐日结果写 CSV")
    ap.add_argument("--stations", default="",
                    help="逗号分隔，只用这些站训练+评估。加站时做「加站前 vs 加站后」"
                         "的对照要用它 —— 否则两臂的训练集不同，比的不是同一件事")
    args = ap.parse_args()

    s = datetime.strptime(args.start, "%Y-%m-%d").date()
    e = datetime.strptime(args.end, "%Y-%m-%d").date()

    print("读取逐时实况…", file=sys.stderr)
    days = N.load_hourly(args.db, args.table)
    if args.stations:
        _keep = {x.strip().upper() for x in args.stations.split(",") if x.strip()}
        days = {k: v for k, v in days.items() if k[0] in _keep}
        print(f"  只用 {len(_keep)} 个站: {sorted(_keep)}", file=sys.stderr)
    print(f"  {len(days)} 个可用站日", file=sys.stderr)

    nwp_map = {}
    if args.nwp_csv and os.path.exists(args.nwp_csv):
        want = set(N.NWP_COLS.values()) | {"temperature_2m_peakmean",
                                           "dew_point_2m_peakmean", "recent_bias"}
        for r in csv.DictReader(open(args.nwp_csv, encoding="utf-8")):
            if r.get("lead") == "1" and r.get("temperature_2m_max"):
                nwp_map[(r["station"], r["date"])] = {
                    c: float(r[c]) for c in want if r.get(c) not in (None, "")}
        print(f"  NWP 覆盖 {len(nwp_map)} 站日", file=sys.stderr)

    # D+1 链路的订正后输出（train_mos --pred 导出的**样本外**测试期预报）。
    # 只在 --mos-oos 给了文件时挂上；缺的站日 __mosd 为 None，走缺测路径。
    if args.mos_oos and os.path.exists(args.mos_oos):
        n_mo = 0
        for r in csv.DictReader(open(args.mos_oos, encoding="utf-8")):
            if r.get("lead") != "1" or not r.get("pred"):
                continue
            k = (r["station"], r["date"])
            if k in nwp_map:
                nwp_map[k]["__mosd"] = float(r["pred"]); n_mo += 1
        print(f"  MOS 订正输出覆盖 {n_mo} 站日", file=sys.stderr)

    names = list(N.FEATS) if args.nwp_csv else \
        [n for n in N.FEATS if not N.is_nwp_feat(n)]
    if args.peak_feat:
        names = names + PEAK_FEATS

    m2_maps = []
    if args.nwp_csv2:
        if not args.nwp_csv:
            print("[error] --nwp-csv2 需要 --nwp-csv", file=sys.stderr)
            return 1
        for path in args.nwp_csv2:
            mm = {}
            for r in csv.DictReader(open(path, encoding="utf-8")):
                if r.get("lead") == "1" and r.get("temperature_2m_max"):
                    mm[(r["station"], r["date"])] = {
                        k: float(r[c]) for k, c in M2_COLS.items()
                        if r.get(c) not in (None, "")}
            m2_maps.append(mm)
            print(f"  追加模式 {os.path.basename(path)}: {len(mm)} 站日",
                  file=sys.stderr)
        names = names + m2_feature_names(len(m2_maps))

    mos_pred = {}
    if args.blend_mos:
        if not args.nwp_csv:
            print("[error] --blend-mos 需要 --nwp-csv", file=sys.stderr)
            return 1
        print("滚动生成 D+1 MOS 预报（路线组合）…", file=sys.stderr)
        mos_pred = mos_walkforward(T.load(args.nwp_csv), args.alphas)
        print(f"  {len(mos_pred)} 个站日", file=sys.stderr)
        names = names + MOS_FEATS
    blks = blocks(s, e, args.retrain_days)
    print(f"评估期 {s} ~ {e}，{len(blks)} 个重训块，"
          f"特征 {len(names)} 项，{'两段式' if not args.no_hurdle else '直接回归'}",
          file=sys.stderr)

    # res[(stn, cutoff)] = [(date, pred, actual), ...]
    res = defaultdict(list)
    miss_nwp = 0
    for cutoff in args.cutoffs:
        for b0, b1 in blks:
            cut = b0.isoformat()
            clim_r, clim_p = clim_before(days, [cutoff], cut)
            rows = N.make_samples(days, cutoff, clim_r.get(cutoff, {}),
                                  clim_p, nwp_map, 0)
            rows = [r for r in rows if r["f"]["clim_rise"] is not None]
            if m2_maps:
                for r in rows:
                    t1 = r["f"].get("nwp_tmax")
                    members = [t1] if t1 is not None else []
                    for i, mm in enumerate(m2_maps, start=2):
                        g = mm.get((r["stn"], r["date"])) or {}
                        for k in M2_COLS:
                            r["f"][f"m{i}_{k}"] = g.get(k)
                        t2 = g.get("tmax")
                        r["f"][f"m{i}_minus_sofar"] = (None if t2 is None
                                                       else t2 - r["so_far"])
                        r["f"][f"m{i}_minus_m1"] = (None if (t2 is None
                                                    or t1 is None) else t2 - t1)
                        if t2 is not None:
                            members.append(t2)
                    if len(members) >= 2:
                        mu = sum(members) / len(members)
                        var = sum((x - mu) ** 2 for x in members) / len(members)
                        r["f"]["ens_mean"] = mu
                        r["f"]["ens_spread"] = var ** 0.5
                        r["f"]["ens_mean_minus_sofar"] = mu - r["so_far"]
                        r["f"]["ens_max_minus_min"] = max(members) - min(members)
                    else:
                        for k in ENS_FEATS:
                            r["f"][k] = None
                rows = [r for r in rows if r["f"].get("ens_mean") is not None]
            for r in rows:
                N.add_interactions(r["f"])
            if mos_pred:
                for r in rows:
                    p = mos_pred.get((r["stn"], r["date"]))
                    r["f"]["mos_d1"] = p
                    r["f"]["mos_d1_minus_sofar"] = (None if p is None
                                                    else p - r["so_far"])
                rows = [r for r in rows if r["f"]["mos_d1"] is not None]
            tr = [r for r in rows if r["date"] < cut]
            te = [r for r in rows if b0.isoformat() <= r["date"] <= b1.isoformat()]
            if args.nwp_csv:
                tr = [r for r in tr if r["f"].get("nwp_tmax") is not None]
                miss_nwp += sum(1 for r in te if r["f"].get("nwp_tmax") is None)
            if len(tr) < 500 or not te:
                print(f"[warn] 截止 {cutoff} 块 {b0}: 训练 {len(tr)} / 评估 "
                      f"{len(te)}，跳过", file=sys.stderr)
                continue
            pooled, per = fit_block(tr, names, args.alphas,
                                    peak=args.peak_feat, cutoff=cutoff)
            for stn in sorted({r["stn"] for r in te}):
                sub = [r for r in te if r["stn"] == stn]
                m = per.get(stn, pooled)
                if m.get("peak"):
                    add_peak_feats(sub, *m["peak"])
                mean_r = predict(m, sub, names, not args.no_hurdle,
                                 args.hurdle_thresh)
                if args.inflate != 1.0:
                    mu = m.get("rise_mu", 0.0)
                    mean_r = [max(0.0, mu + args.inflate * (v - mu))
                              for v in mean_r]
                pmf = rise_pmf(m, sub, names)
                mode_r = decide(pmf, 0) if pmf else mean_r
                win_r = decide(pmf, 1) if pmf else mean_r
                # 优先用分位数回归；没有就退回序贯分类的 PMF 分位。
                # PMF 是线性概率模型，尾部标定差 —— 大升幅日覆盖只有 60-88%
                if m.get("q90"):
                    Xq, _ = N.matrix(sub, m["median"], names)
                    p90_r = [max(0.0, v) for v in T.ridge_pred(m["q90"], Xq)]
                else:
                    p90_r = N.rise_quantile(pmf, 0.90) if pmf else mean_r
                for r, rm, rk, rw, rq in zip(sub, mean_r, mode_r, win_r, p90_r):
                    res[(stn, cutoff)].append(
                        (r["date"], r["so_far"] + rm, r["tmax"], r["so_far"],
                         r["so_far"] + rk, r["so_far"] + rw,
                         r["so_far"] + max(rq, rm)))   # P90 不得低于点预报

    if miss_nwp:
        print(f"[warn] 评估期有 {miss_nwp} 个站日缺 GFS 特征（mos.csv 未覆盖到），"
              f"这些日子按缺测填补，精度偏低。先跑 build_mos_dataset.py fetch",
              file=sys.stderr)

    # ---------------- 汇总 ----------------
    cuts = args.cutoffs
    stns = sorted({k[0] for k in res})
    print(f"\n{'='*84}")
    print(f"逐日滚动回测  {s} ~ {e}  重训周期 {args.retrain_days} 天  "
          f"{'含 NWP' if args.nwp_csv else '纯实况'}")
    print("命中 = round(预报) == 实际日最高温（整数度，完全一致）")

    # 三种决策，元组索引 1 / 4 / 5
    DEC = [("均值", 1), ("众数", 4), ("±1窗口", 5)]
    # 主口径必须与生产一致。predict_nowcast.py 用的是两段式期望（均值决策），
    # 所以 MAE / 完全命中 两列都按均值决策算 —— 之前设成 ±1窗口，
    # 导致汇总表的 MAE 比生产实际值虚高（14 时 0.69 vs 实际 0.39）
    MAIN = 1

    def stat(v, i):
        n = len(v)
        hit = sum(1 for r in v if round(r[i]) == round(r[2]))
        w1 = sum(1 for r in v if abs(round(r[i]) - r[2]) <= 1)
        mae = sum(abs(round(r[i]) - r[2]) for r in v) / n
        return n, hit / n, w1 / n, mae

    for cutoff in cuts:
        print(f"\n── {cutoff:02d} 时起报   主指标 = ±1℃ 命中率")
        print(f"  {'站点':<14}{'n':>5}" +
              "".join(f"{'±1℃ '+t:>12}" for t, _ in DEC) +
              f"{'完全命中':>10}{'MAE':>7}{'已见顶':>8}")
        tot = []
        for stn in stns:
            v = res.get((stn, cutoff))
            if not v:
                continue
            tot += v
            cells = "".join(f"{100*stat(v, i)[2]:>11.0f}%" for _, i in DEC)
            n, hit, w1, mae = stat(v, MAIN)
            seen = sum(1 for r in v if r[2] - r[3] < 1e-9) / len(v)
            print(f"  {stn} {N.NAMES.get(stn,''):<9}{len(v):>5}{cells}"
                  f"{100*hit:>9.0f}%{mae:>7.2f}{100*seen:>7.0f}%")
        if tot:
            cells = "".join(f"{100*stat(tot, i)[2]:>11.0f}%" for _, i in DEC)
            n, hit, w1, mae = stat(tot, MAIN)
            print(f"  {'合计':<14}{len(tot):>5}{cells}"
                  f"{100*hit:>9.0f}%{mae:>7.2f}")

    print(f"\n── 分站 ±1℃ 命中率总览（{DEC[[i for _, i in DEC].index(MAIN)][0]}决策）")
    print(f"  {'站点':<14}" + "".join(f"{str(c)+' 时':>9}" for c in cuts)
          + f"{'命中天数(最差档)':>18}")
    for stn in stns:
        line = f"  {stn} {N.NAMES.get(stn,''):<9}"
        worst = None
        for c in cuts:
            v = res.get((stn, c))
            if not v:
                line += f"{'--':>9}"
                continue
            w1 = stat(v, MAIN)[2]
            line += f"{100*w1:>8.0f}%"
            if worst is None or w1 < worst[0]:
                worst = (w1, round(w1 * len(v)), len(v))
        if worst:
            line += f"{worst[1]:>13}/{worst[2]:<4}"
        print(line)

    print(f"\n── 高端情景 P90（另给的「不排除冲到」值，不替代点预报）")
    print(f"  {'起报时':<8}{'n':>6}{'实际<=P90 占比':>16}{'P90-点预报 均值':>18}"
          f"{'大升幅日(>=4度) 点预报偏低':>26}")
    for cutoff in cuts:
        v = [r for stn in stns for r in res.get((stn, cutoff), [])]
        if not v:
            continue
        cov = sum(1 for r in v if r[2] <= round(r[6]) + 1e-9) / len(v)
        gap = sum(round(r[6]) - round(r[1]) for r in v) / len(v)
        big = [r for r in v if r[2] - r[3] >= 4]
        me = (sum(round(r[1]) - r[2] for r in big) / len(big)) if big else 0.0
        print(f"  {cutoff:<8}{len(v):>6}{100*cov:>15.0f}%{gap:>18.1f}"
              f"{me:>21.2f} (n={len(big)})")

    if args.daily:
        print(f"\n── 逐日明细（预报 / 实际 / 起报时已达）")
        for stn in stns:
            print(f"\n  {stn} {N.NAMES.get(stn,'')}")
            byd = defaultdict(dict)
            for c in cuts:
                for d, p, a, sf, pk, pw, pq in res.get((stn, c), []):
                    byd[d][c] = (pw, a, sf)
            print("    日期        " + "".join(f"{str(c)+' 时起报':>14}" for c in cuts)
                  + f"{'实际':>7}")
            for d in sorted(byd):
                row = f"    {d}"
                act = None
                for c in cuts:
                    if c in byd[d]:
                        p, a, sf = byd[d][c]
                        act = a
                        mark = "✓" if round(p) == round(a) else " "
                        row += f"{round(p):>10}{mark:>2}(+{p-sf:.1f})"[:14].rjust(14)
                    else:
                        row += f"{'--':>14}"
                row += f"{act:>7.0f}" if act is not None else f"{'--':>7}"
                print(row)

    if args.csv_out:
        with open(args.csv_out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            # pred_raw 是**未取整**的连续预报。已有的 pred_mean 等列一律取整，
            # 分析「逐小时来回跳」时就没法区分「真的变了 1 度」和「在 x.5
            # 附近抖了 0.02 度导致整数翻了个个儿」—— 后者占了大头。
            w.writerow(["station", "date", "cutoff", "pred_mean", "pred_mode",
                        "pred_win", "actual", "so_far",
                        "w1_mean", "w1_mode", "w1_win", "hit_win", "pred_p90",
                        "pred_raw"])
            for (stn, c), v in sorted(res.items()):
                for d, p, a, sf, pk, pw, pq in v:
                    w.writerow([stn, d, c, round(p), round(pk), round(pw), a, sf,
                                int(abs(round(p) - a) <= 1),
                                int(abs(round(pk) - a) <= 1),
                                int(abs(round(pw) - a) <= 1),
                                int(round(pw) == round(a)), round(pq),
                                round(p, 4)])
        print(f"\n逐日结果已写 {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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


def fit_block(tr, names, alphas, val_days=90):
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
        q90 = N.fit_quantile(rows_tr, 0.90, best[0], med, names)
        return {"median": med, "ridge": best[2], "alpha": best[0],
                "hurdle": N.fit_hurdle(rows_tr, [best[0]], med, names),
                "ordinal": N.fit_ordinal(rows_tr, best[0], med, names),
                "q90": q90}

    pooled = one(fit_tr, fit_va)
    per = {}
    for stn in sorted({r["stn"] for r in tr}):
        s_tr = [r for r in fit_tr if r["stn"] == stn]
        s_va = [r for r in fit_va if r["stn"] == stn]
        if len(s_tr) >= 300 and s_va:
            per[stn] = one(s_tr, s_va)
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


def predict(m, rows, names, hurdle):
    X, _ = N.matrix(rows, m["median"], names)
    if hurdle and m["hurdle"]:
        return N.pred_hurdle(m["hurdle"], X)
    return [max(0.0, v) for v in T.ridge_pred(m["ridge"], X)]


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
    ap.add_argument("--blend-mos", action="store_true",
                    help="把滚动重训的 D+1 MOS 预报当特征加进来（组合两条路线）")
    ap.add_argument("--no-hurdle", action="store_true")
    ap.add_argument("--daily", action="store_true", help="逐日逐站明细")
    ap.add_argument("--csv-out", default="", help="把逐日结果写 CSV")
    args = ap.parse_args()

    s = datetime.strptime(args.start, "%Y-%m-%d").date()
    e = datetime.strptime(args.end, "%Y-%m-%d").date()

    print("读取逐时实况…", file=sys.stderr)
    days = N.load_hourly(args.db, args.table)
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

    names = list(N.FEATS) if args.nwp_csv else \
        [n for n in N.FEATS if not n.startswith("nwp")]

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
            pooled, per = fit_block(tr, names, args.alphas)
            for stn in sorted({r["stn"] for r in te}):
                sub = [r for r in te if r["stn"] == stn]
                m = per.get(stn, pooled)
                mean_r = predict(m, sub, names, not args.no_hurdle)
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
            w.writerow(["station", "date", "cutoff", "pred_mean", "pred_mode",
                        "pred_win", "actual", "so_far",
                        "w1_mean", "w1_mode", "w1_win", "hit_win", "pred_p90"])
            for (stn, c), v in sorted(res.items()):
                for d, p, a, sf, pk, pw, pq in v:
                    w.writerow([stn, d, c, round(p), round(pk), round(pw), a, sf,
                                int(abs(round(p) - a) <= 1),
                                int(abs(round(pk) - a) <= 1),
                                int(abs(round(pw) - a) <= 1),
                                int(round(pw) == round(a)), round(pq)])
        print(f"\n逐日结果已写 {args.csv_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_mos.py — 用训练好的模型预报指定日期的站点日最高温

    python3 predict_mos.py                          # 预报今天
    python3 predict_mos.py --date 2026-07-25
    python3 predict_mos.py --ahead 2                # 今天起共 3 天
    python3 predict_mos.py --verbose                # 显示订正量拆解

口径说明:
  预报「今天」用的是 24 小时前起报的那一轮（训练时的 previous_day1），
  与训练完全一致。不是用最新一轮 —— 那样时效跟训练对不上，模型会失准。

  recent_bias 需要回溯发布日往前 7 个有效日的「实况 − 模式」，
  所以一次拉 30 天窗口，一个请求覆盖全部所需。

  特征构造直接复用 build_mos_dataset 和 train_mos 里的函数，
  保证与训练时逐列一致 —— 手写一遍最容易在列序或填补上错位。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import build_mos_dataset as B          # noqa: E402
import merge_mos as MG                 # noqa: E402  DEB 与训练端共用同一实现
import train_mos as T                  # noqa: E402

CST = timezone(timedelta(hours=8))

import stations as _S  # 站点清单唯一真相源
import tablefmt as F     # 显示宽度对齐（中文双宽）
NAMES = _S.NAMES


def fetch_window(stations, d0: date, d1: date, model: str, vars_: list[str]):
    """一次请求拉全部站点、全部变量、两个时效。返回内存库供聚合复用。"""
    hourly = [f"{v}_previous_day{L}" for v in vars_ for L in B.LEADS]
    data = B._fetch({
        "latitude": ",".join(f"{B.STATIONS[s][0]:.4f}" for s in stations),
        "longitude": ",".join(f"{B.STATIONS[s][1]:.4f}" for s in stations),
        "hourly": ",".join(hourly), "models": model,
        "start_date": d0.isoformat(), "end_date": d1.isoformat(),
        "timezone": "Asia/Shanghai",
    })
    conn = sqlite3.connect(":memory:")
    conn.executescript(B.DDL)
    n = 0
    for stn, loc in zip(stations, data):
        h = loc.get("hourly", {})
        times = h.get("time", [])
        for key, series in h.items():
            if key == "time":
                continue
            base, _, tag = key.rpartition("_previous_day")
            if not tag.isdigit():
                continue
            for t, v in zip(times, series):
                if v is not None:
                    conn.execute("INSERT OR IGNORE INTO fcst VALUES (?,?,?,?,?)",
                                 (stn, t, base, int(tag), float(v)))
                    n += 1
    conn.commit()
    return conn, n


def load_obs(path, table="daily", min_obs=18):
    """读预聚合日表的实况日最高温。"""
    if not os.path.exists(path):
        print(f"[warn] 实况库不存在: {path}，recent_bias 与 prev_tmax 将缺失",
              file=sys.stderr)
        return {}
    conn = sqlite3.connect(path)
    cols = {c[1] for c in conn.execute(f"PRAGMA table_info({table})")}
    if not {"station", "date", "tmax"} <= cols:
        print(f"[warn] {table} 表字段不符，跳过实况", file=sys.stderr)
        return {}
    has_n = "n_obs" in cols
    out = {}
    for row in conn.execute(
            f"SELECT station, date, tmax{', n_obs' if has_n else ''} "
            f"FROM {table} WHERE tmax IS NOT NULL"):
        if has_n and row[3] is not None and row[3] < min_obs:
            continue
        out[(row[0], row[1])] = float(row[2])
    conn.close()
    return out


def make_row(stn, target: date, lead: int, feats_daily, obs, bias_days=7):
    """构造一行样本，字段与训练用的 mos.csv 完全对应。"""
    key = (stn, target.isoformat())
    f = feats_daily.get(key)
    if not f or f.get("temperature_2m_max") is None:
        return None, "无该日预报"

    issue_day = target - timedelta(days=lead)
    prev = obs.get((stn, issue_day.isoformat()))

    resid, k, back = [], 0, 1
    while k < bias_days and back <= bias_days * 3:
        bd = issue_day - timedelta(days=back - 1)
        o = obs.get((stn, bd.isoformat()))
        m = feats_daily.get((stn, bd.isoformat()), {}).get("temperature_2m_max")
        if o is not None and m is not None:
            resid.append(o - m)
            k += 1
        back += 1
    bias = sum(resid) / len(resid) if len(resid) >= bias_days // 2 else None

    import math
    doy = target.timetuple().tm_yday
    row = {"station": stn, "date": target.isoformat(), "lead": lead,
           "y_tmax": 0.0,                      # 占位，预测时不用
           "prev_tmax": prev, "recent_bias": None if bias is None else round(bias, 3),
           "doy_sin1": round(math.sin(2 * math.pi * doy / 365.25), 4),
           "doy_cos1": round(math.cos(2 * math.pi * doy / 365.25), 4),
           "doy_sin2": round(math.sin(4 * math.pi * doy / 365.25), 4),
           "doy_cos2": round(math.cos(4 * math.pi * doy / 365.25), 4)}
    row.update({k2: v for k2, v in f.items()})
    return row, f"bias 用了 {len(resid)} 天" if resid else "无 recent_bias"


def load_gbm(model_path):
    """岭回归+GBM 融合的 GBM 部分。缺 sklearn 或缺文件就返回 None，
    预测自动降级成纯岭回归（会打提示，不静默）。"""
    p = model_path + ".gbm.pkl"
    if not os.path.exists(p):
        return None
    try:
        import pickle
        with open(p, "rb") as fh:
            return pickle.load(fh)
    except Exception as e:
        print(f"[warn] {p} 加载失败（{type(e).__name__}），降级为纯岭回归。"
              f"cron 里常见原因是用了不带 sklearn 的 python3", file=sys.stderr)
        return None


def predict(row, spec, stn, gbm=None):
    """返回 (最终预报, 模式原始, 订正量, 用的是哪个模型)。"""
    # 用哪个架构由训练时的验证集比较决定（spec["prefer"]），不能无条件用分站 ——
    # 多模式下特征列多，分站会过拟合，合并反而更好
    per = (spec.get("per_station") or {}).get(stn)
    pref = spec.get("prefer", "per_station")
    use_per = bool(per) and pref == "per_station"
    if use_per:
        mdl, med, dummies, tag = per["model"], per["median"], [], "分站"
    else:
        mdl, med, dummies, tag = spec["model"], spec["median"], spec["stations"], "合并"
    X, _, base, _, _ = T.build_matrix([row], spec["feats"], dummies, med)
    corr = T.ridge_pred(mdl, X)[0]
    w = spec.get("blend_w")
    if gbm is not None and w is not None and pref == "blend":
        # 融合只对合并模型有意义: GBM 是在带站点哑变量的合并矩阵上训的
        corr = w * corr + (1 - w) * float(gbm.predict(X)[0])
        tag += "+GBM 融合"
    return base[0] + corr, base[0], corr, tag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.json")
    ap.add_argument("--obs-db", default="cn.sqlite")
    ap.add_argument("--daily-table", default="daily")
    ap.add_argument("--date", default="", help="目标日期，默认今天（北京时）")
    ap.add_argument("--ahead", type=int, default=0, help="再往后预报几天")
    ap.add_argument("--gfs-model", default="gfs_global")
    ap.add_argument("--extra-models", default="",
                    help="逗号分隔的追加模式，顺序须与 merge_mos.py --extra 的顺序"
                         "一致，否则 m2_/m3_ 各列对错模式、系数全部错位")
    ap.add_argument("--csv-out", default="", help="把预报写成 CSV，供对照脚本使用")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"[error] 找不到 {args.model}，先跑 train_mos.py --dump {args.model}",
              file=sys.stderr)
        return 1
    spec_all = json.load(open(args.model, encoding="utf-8"))
    gbm_all = load_gbm(args.model)

    d0 = (datetime.strptime(args.date, "%Y-%m-%d").date() if args.date
          else datetime.now(CST).date())
    targets = [d0 + timedelta(days=i) for i in range(args.ahead + 1)]
    stations = list(B.STATIONS)

    obs = load_obs(args.obs_db, args.daily_table)
    if obs:
        last = max(d for _, d in obs)
        print(f"实况库最新到 {last}", file=sys.stderr)
        if last < (d0 - timedelta(days=2)).isoformat():
            print(f"[warn] 实况已滞后，recent_bias 会退化 —— 先更新 {args.obs_db}",
                  file=sys.stderr)

    lo = min(targets) - timedelta(days=30)
    hi = max(targets)
    print(f"拉取 {lo} ~ {hi} 的 GFS 固定时效预报…", file=sys.stderr)
    try:
        conn, n = fetch_window(stations, lo, hi, args.gfs_model, B.VARS)
    except Exception as e:
        print(f"[error] 取数失败: {e}", file=sys.stderr)
        return 1
    print(f"  {n} 条", file=sys.stderr)

    rows_out = []
    daily = {L: B.daily_features(conn, L) for L in B.LEADS}

    # 多模式: 每个追加模式各拉一次，特征列名与 merge_mos.py 逐列对应
    mdls = [x.strip() for x in args.extra_models.split(",") if x.strip()]
    need = max((int(n.split("_")[0][1:]) for spec in spec_all.values()
                for n in spec["feats"]
                if n.startswith("m") and n[1:2].isdigit() and "_" in n),
               default=1) - 1
    if need and len(mdls) != need:
        print(f"[error] {args.model} 需要 {need} 个追加模式，--extra-models 给了 "
              f"{len(mdls)} 个。顺序也必须与训练时一致", file=sys.stderr)
        return 1
    extra_daily = []
    for m in mdls:
        print(f"拉取 {m}…", file=sys.stderr)
        try:
            c2, n2 = fetch_window(stations, lo, hi, m, B.VARS)
            extra_daily.append({L: B.daily_features(c2, L) for L in B.LEADS})
        except Exception as e:
            print(f"[warn] {m} 取数失败: {e}，该模式各列按缺测处理", file=sys.stderr)
            extra_daily.append({L: {} for L in B.LEADS})

    for tgt in targets:
        print(f"\n{'='*66}")
        print(f"目标日 {tgt}（北京时）")
        for lead in B.LEADS:
            spec = spec_all.get(str(lead))
            if not spec:
                continue
            issue = tgt - timedelta(days=lead)
            print(f"\n── D+{lead}（{issue} 起报，时效约 {lead*24} 小时）")
            # 列宽按**显示列**算（中文双宽），表头与数据行共用。见 tablefmt.py
            W_STN, W_VAL, W_RAW, W_COR = 16, 7, 10, 8
            print("  " + F.L("站点", W_STN) + F.R("预报", W_VAL)
                  + F.R("模式原始", W_RAW) + F.R("订正", W_COR) + "   备注")
            any_ok = False
            for stn in stations:
                row, note = make_row(stn, tgt, lead, daily[lead], obs)
                if row is not None and extra_daily:
                    # 与 merge_mos.py 同构: 每个模式跑一遍 make_row 再加前缀，
                    # 这样 recent_bias 也会用该模式自己的预报算，与训练一致
                    tm = [row["temperature_2m_max"]]
                    cl = ([row["cloud_cover_peakmean"]]
                          if row.get("cloud_cover_peakmean") is not None else [])
                    for i, ed in enumerate(extra_daily, start=2):
                        r2, _ = make_row(stn, tgt, lead, ed[lead], obs)
                        for k2, v2 in (r2 or {}).items():
                            if k2 in ("station", "date", "lead", "y_tmax"):
                                continue
                            row[f"m{i}_{k2}"] = v2
                        # 与 merge_mos.py 对齐: 该模式当天有没有数据。
                        # 漏掉这列会让训练时见过的 6 个特征在上线时全按缺测填补
                        row[f"m{i}_present"] = 1 if r2 else 0
                        if r2 and r2.get("temperature_2m_max") is not None:
                            tm.append(r2["temperature_2m_max"])
                        if r2 and r2.get("cloud_cover_peakmean") is not None:
                            cl.append(r2["cloud_cover_peakmean"])
                    if len(tm) >= 2:
                        mu = sum(tm) / len(tm)
                        row["ens_mean_tmax"] = mu
                        row["ens_spread_tmax"] = (
                            sum((x - mu) ** 2 for x in tm) / len(tm)) ** .5
                        row["ens_max_minus_min_tmax"] = max(tm) - min(tm)
                    if len(cl) >= 2:
                        mu = sum(cl) / len(cl)
                        row["ens_mean_cloud_peak"] = mu
                        row["ens_spread_cloud_peak"] = (
                            sum((x - mu) ** 2 for x in cl) / len(cl)) ** .5

                    # DEB 自适应加权。训练端在 merge_mos.deb_columns 里算，
                    # 这里必须给出逐列一致的值，否则又是「训练见过、上线缺测」
                    hist, cur = {}, {}
                    for i, ed in enumerate([daily] + extra_daily):
                        h = []
                        for k in range(1, 31):
                            bd = (tgt - timedelta(days=k)).isoformat()
                            fv = ed[lead].get((stn, bd), {}).get(
                                "temperature_2m_max")
                            ov = obs.get((stn, bd))
                            if fv is not None and ov is not None:
                                h.append((bd, fv, ov))
                        hist[i] = sorted(h)
                        cv = ed[lead].get((stn, tgt.isoformat()), {}).get(
                            "temperature_2m_max")
                        if cv is not None:
                            cur[i] = cv
                    dp, dt_ = MG.deb_weights(hist, cur)
                    if dp is not None:
                        row["deb_pred"] = dp
                        row["deb_trust"] = dt_
                        row["deb_minus_gfs"] = dp - row["temperature_2m_max"]
                if row is None:
                    print("  " + F.L(f"{stn} {NAMES.get(stn, '')}", W_STN)
                          + F.R("--", W_VAL) + F.R("--", W_RAW)
                          + F.R("--", W_COR) + f"   {note}")
                    continue
                try:
                    fin, raw, corr, tag = predict(
                        row, spec, stn, (gbm_all or {}).get(str(lead)))
                except Exception as e:
                    print(f"  {stn}  预测失败: {type(e).__name__}: {e}")
                    continue
                any_ok = True
                extra = note if args.verbose else (
                    "" if row["recent_bias"] is not None else "无 recent_bias")
                if args.verbose:
                    extra += f" | {tag}模型 | rb={row['recent_bias']}"
                print("  " + F.L(f"{stn} {NAMES.get(stn, '')}", W_STN)
                      + F.R(round(fin), W_VAL) + F.R(f"{raw:.1f}", W_RAW)
                      + F.R(f"{corr:+.1f}", W_COR) + f"   {extra}")
                rows_out.append((lead, stn, tgt.isoformat(), round(fin, 2),
                                 round(fin), round(raw, 2), round(corr, 2)))
            if not any_ok:
                print("  （该时效无可用数据，可能是目标日超出归档范围）")

    if args.csv_out and rows_out:
        import csv as _csv
        with open(args.csv_out, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["lead", "station", "date", "pred", "pred_round",
                        "model_raw", "correction"])
            w.writerows(rows_out)
        print(f"\n预报已写 {args.csv_out}", file=sys.stderr)

    print(f"\n预报值已取整（真值为整数度，取整是上线口径）。"
          f"\n数据来源 Open-Meteo.com (CC BY 4.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""决定性检验: 用起报时刻能拿到的全部信息，能不能预测今天误差的正负号。

    python3 signtest.py bt_sign.csv

判据:
  AUC ~= 0.5  -> 误差里没有可利用的结构，剩下的是噪声，后处理这条线到头了
  AUC  > 0.6  -> 偏差是可识别的、模型没提取出来 -> 有救，值得继续挖

为什么这个检验能一锤定音: 「是偏差还是噪声」的操作定义就是
「能不能用事前可得的信息把它区分开」。之前那些实验（事后残差订正、
站点身份、气象异常扫描）都是在猜某一个具体机制；这个是把**全部 126 项特征**
一起交给树模型去找，找不到就是真的没有。
"""
import csv, sys, collections, statistics as st
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
# 特征表要与上线模型一致，否则「全部信息」就不全
for k in ("PLOYGON_CONV", "PLOYGON_PROF", "PLOYGON_CURVE", "PLOYGON_WINDAM",
          "PLOYGON_HPBL", "PLOYGON_REGIME"):
    os.environ[k] = "1"
import train_nowcast as N

try:
    from sklearn.ensemble import HistGradientBoostingClassifier as GBC
    from sklearn.metrics import roc_auc_score
    import numpy as np
except ImportError:
    print("需要 sklearn"); sys.exit(2)

pred = collections.defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    pred[int(r["cutoff"])][(r["station"], r["date"])] = (
        float(r["pred_raw"]), float(r["actual"]))

days = N.load_hourly("cn.sqlite")
m2 = N.load_m2(["mos_ecmwf.csv","mos_cma.csv","mos_icon.csv",
                "mos_jma_.csv","mos_gem_.csv","mos_local12.csv"])
nwp = {}
want = set(N.NWP_COLS.values()) | {"temperature_2m_peakmean","dew_point_2m_peakmean"}
for r in csv.DictReader(open("mos.csv")):
    if r.get("lead") == "1" and r.get("temperature_2m_max"):
        nwp[(r["station"], r["date"])] = {
            c: float(r[c]) for c in want if r.get(c) not in (None, "")}

for cut in sorted(pred):
    cr, cp = N.climatology(days, [cut], 2025)
    rows = N.make_samples(days, cut, cr[cut], cp, nwp, 0, m2)
    idx = {(r["stn"], r["date"]): r for r in rows}
    X, y, dates = [], [], []
    names = [n for n in N.FEATS if not n.startswith("oracle")]
    med = None
    keep = []
    for k, (p, a) in pred[cut].items():
        r = idx.get(k)
        if r is None:
            continue
        e = round(p) - a
        if e == 0:                       # 只问「错的时候偏哪边」
            continue
        keep.append((r, 1 if e > 0 else 0, k[1]))
    if len(keep) < 500:
        print(f"  {cut} 时: 样本不足"); continue
    Xall, med = N.matrix([r for r, _, _ in keep], None, names)
    y = [v for _, v, _ in keep]
    dates = [d for _, _, d in keep]
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    cutn = int(len(order) * 0.7)
    tr, te = order[:cutn], order[cutn:]
    Xa = np.asarray(Xall, float)
    g = GBC(max_depth=3, max_iter=300, learning_rate=0.06,
            min_samples_leaf=20, random_state=0).fit(Xa[tr], np.asarray(y)[tr])
    pv = g.predict_proba(Xa[te])[:, 1]
    auc = roc_auc_score(np.asarray(y)[te], pv)
    base = sum(np.asarray(y)[te]) / len(te)
    print(f"  {cut} 时  训练 {len(tr)} / 检验 {len(te)}  "
          f"检验期偏高占比 {base:.1%}  **AUC = {auc:.3f}**"
          + ("  <- 有结构" if auc > 0.6 else "  <- 接近随机"))

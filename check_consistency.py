#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_consistency.py — 训练路径与预测路径的一致性回归检查

    python3 check_consistency.py --date 2026-07-20

这个项目反复踩同一类坑: 训练时构造了某个特征，预测时那条代码路径忘了给，
于是上线后该列被中位数填补，模型静默降级、不报错、指标只是慢慢变差。
已经踩过四次（m{i}_present / rise_anom 窗口 / DEB / q90），
所以把它变成机器检查，别再靠人眼。

检查项:
  1. 临近: make_samples（训练） vs predict_nowcast 的特征构造，逐列比对
  2. MOS:  mos_multi.csv（训练） vs predict_mos 的行构造，逐列比对
  3. 模型 JSON 要求的特征列，预测端是否全部能产出
  4. 各脚本对模式列表/顺序的约定是否一致
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import train_nowcast as N                          # noqa: E402

TOL = 1e-6
FAIL = []


def rep(ok, tag, detail=""):
    print(f"  {'✓' if ok else '✗'} {tag}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(tag)


# ---------------------------------------------------------------- 临近

def check_nowcast(args):
    print("\n[1] 临近: 训练特征 vs 预测特征（逐列）")
    import predict_nowcast as P

    cutoff = int(args.cutoff)
    # 与 run_hourly.sh 同样的选择逻辑: 9-13 用六模式模型，14/15 用纯实况模型
    mpath = args.nowcast_model if cutoff <= 13 else args.nowcast_late_model
    spec_all = json.load(open(mpath, encoding="utf-8"))
    spec = spec_all.get(str(cutoff))
    print(f"  （{cutoff} 时用 {mpath}）")
    if spec is None:
        rep(False, f"模型里没有 {cutoff} 时截止")
        return
    names = spec["names"]
    stations = sorted({k.split("|")[0] for k in spec["clim_rise"]})
    tgt = datetime.strptime(args.date, "%Y-%m-%d").date()

    # ---- 训练路径 ----
    days = N.load_hourly(args.db)
    m2_maps = N.load_m2(args.nwp_csv2) if args.nwp_csv2 else []
    nwp_map = {}
    if args.nwp_csv:
        want = set(N.NWP_COLS.values()) | {"temperature_2m_peakmean",
                                           "dew_point_2m_peakmean"}
        for r in csv.DictReader(open(args.nwp_csv, encoding="utf-8")):
            if r.get("lead") == "1" and r.get("temperature_2m_max"):
                nwp_map[(r["station"], r["date"])] = {
                    c: float(r[c]) for c in want if r.get(c) not in (None, "")}
    # 必须用模型 JSON 里那份气候态。用别的重算一份，比出来的差异是检查器自己
    # 造成的，不是生产问题 —— 训练与预测在生产里用的就是同一份（dump 出来的那份）
    ck = {tuple([k.split("|")[0], int(k.split("|")[1])]): v
          for k, v in spec["clim_rise"].items()}
    cp_ = {tuple([k.split("|")[0], int(k.split("|")[1])]): v
           for k, v in spec["clim_peak"].items()}
    rows = N.make_samples(days, cutoff, ck, cp_, nwp_map, 0, m2_maps)
    train_rows = {(r["stn"], r["date"]): r for r in rows}

    # ---- 预测路径 ----
    prv = tgt - timedelta(days=1)
    pdays = P.from_db(args.db, "obs", stations,
                      {(tgt - timedelta(days=k)).isoformat() for k in range(11)})
    nwp = {stn: nwp_map.get((stn, tgt.isoformat())) for stn in stations}
    m2 = [{stn: mm.get((stn, tgt.isoformat())) for stn in stations}
          for mm in m2_maps]

    checked = miss = 0
    for stn in stations:
        key = (stn, tgt.isoformat())
        if key not in train_rows:
            continue
        hrs = pdays.get(key)
        o = N.morning(hrs, cutoff) if hrs else None
        if o is None:
            continue
        ph = pdays.get((stn, prv.isoformat()))
        mo_ = tgt.month
        cr = spec["clim_rise"].get(f"{stn}|{mo_}")
        cp = spec["clim_peak"].get(f"{stn}|{mo_}")
        prev = (max(v["t"] for v in ph.values()) if ph else None, cr)
        f, msf = N.build_feats(o, cutoff, prev, cr, cp,
                               tgt.timetuple().tm_yday, nwp.get(stn))
        # 与 predict_nowcast.py 里同一段逻辑
        hist = []
        for k in range(1, 11):
            d0 = (tgt - timedelta(days=k)).isoformat()
            h0 = pdays.get((stn, d0))
            if not h0 or N.morning(h0, cutoff) is None:
                continue
            c0 = spec["clim_rise"].get(f"{stn}|{int(d0[5:7])}")
            if c0 is None:
                continue
            t0 = max(v["t"] for v in h0.values())
            s0 = max(v["t"] for v in N.morning(h0, cutoff).values())
            hist.append(t0 - s0 - c0)
        for k in (3, 7):
            f[f"rise_anom_{k}d"] = sum(hist[:k]) / k if len(hist) >= k else None
        if m2:
            N.add_m2_feats(f, msf, [mm.get(stn) for mm in m2])

        tf = train_rows[key]["f"]
        for nm in names:
            a, b = tf.get(nm), f.get(nm)
            checked += 1
            if a is None and b is None:
                continue
            if a is None or b is None or abs(a - b) > TOL:
                miss += 1
                if miss <= 8:
                    print(f"      {stn} {nm}: 训练={a} 预测={b}")
    rep(miss == 0, f"临近 {cutoff} 时特征一致（{checked} 个值）",
      "" if miss == 0 else f"{miss} 处不一致")

    # 模型要的列，预测端一个都不能缺
    missing_names = [n for n in names if n not in f]
    rep(not missing_names, "模型要求的特征列预测端全部能产出",
      "" if not missing_names else f"缺: {missing_names[:6]}")


# ---------------------------------------------------------------- MOS

def check_mos(args):
    print("\n[2] MOS: 训练集列 vs 预测端能产出的列")
    if not os.path.exists(args.mos_csv):
        rep(False, f"找不到 {args.mos_csv}")
        return
    with open(args.mos_csv, encoding="utf-8") as fh:
        csv_cols = set(next(csv.reader(fh)))
    spec = json.load(open(args.mos_model, encoding="utf-8"))["1"]
    feats = set(spec["feats"])

    # 训练集里没有、但模型点名要的列 —— 说明 feats 与 csv 脱节
    orphan = sorted(f for f in feats if f not in csv_cols
                    and not f.startswith(("dewpoint_depression", "fcst_minus_prev",
                                          "clear_index")))
    rep(not orphan, "模型特征列都能在训练集里找到",
      "" if not orphan else f"孤儿列: {orphan[:6]}")

    # 预测端实跑一次，看构造出的行覆盖了多少特征
    out = subprocess.run(
        [sys.executable, "predict_mos.py", "--date", args.date, "--ahead", "0",
         "--extra-models", args.extra_models, "--verbose",
         "--csv-out", "/tmp/_chk_mos.csv"],
        capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        rep(False, "predict_mos.py 跑通", out.stderr.strip()[-200:])
        return
    rep(True, "predict_mos.py 跑通")
    pref = str(spec.get("prefer"))
    want = {"blend": "融合", "per_station": "分站", "pooled": "合并"}.get(pref, "")
    ok = want and want in out.stdout
    # 融合会显示成「合并+GBM 融合模型」，纯合并只显示「合并模型」，要区分开
    if pref == "pooled" and "融合" in out.stdout:
        ok = False
    rep(bool(ok), "预测端用的架构与 model.json 的 prefer 一致",
      f"prefer={pref}")

    # 被否决的实验特征不能混进上线模型。train_mos.feature_names 自动识别 csv 列，
    # 而 run_daily.sh 每天重建 mos.csv —— 只要 mos_fcst.sqlite 里有这些变量，
    # 忘了排除就会在下次重训时静默上线。2026-07-31 的对流因子实验就踩在这条线上。
    import train_mos as TM
    import train_nowcast as TN

    # 8 个站一个都不能少。2026-08-01 踩过: ZGSZ 换成 WU 序列后只剩 2024-07 起
    # 的数据，而 climatology() 用 --split-year（默认 2024）卡训练年份，
    # 导致 ZGSZ 在气候态里为空 -> 整个站被静默丢掉，预报表少一行也不报错。
    # 重训必须带 --split-year 2025，见 README「每周重训」。
    for path, tag in ((args.nowcast_model, "临近预报模型"),
                      (args.nowcast_late_model, "临近预报晚时次模型")):
        if not os.path.exists(path):
            continue
        nc = json.load(open(path, encoding="utf-8"))
        for cut, blk in sorted(nc.items()):
            if not isinstance(blk, dict) or not blk.get("clim_rise"):
                continue
            got = {k.split("|")[0] for k in blk["clim_rise"]}
            miss = sorted(set(TN.NAMES) - got)
            rep(not miss, f"{tag} {cut} 时含全部 8 站",
                "" if not miss else f"缺: {miss}（重训漏了 --split-year 2025？）")

    leaked = sorted((TM.CONV_COLS | TM.PROF_COLS) & feats)
    rep(not leaked, "被否决的实验特征没混进 MOS 上线模型",
        "" if not leaked else f"泄漏: {leaked}")
    # 临近预报侧同理。跨站特征更危险: predict_nowcast.py 根本没有算它们的代码，
    # 带着 PLOYGON_XSTN=1 训练出来的模型上线后会静默拿到 None。
    # oracle_* 是明知泄漏的先知实验特征（量「见顶时刻不确定」这个瓶颈值多少），
    # 上线就等于用未来信息预报。必须在这里挡死。
    rejected = ({"nwp_precip_peak", "nwp_precip_max", "nwp_cape_peak", "nwp_li_peak",
                 "oracle_peak_h", "oracle_hours_to_peak",
                 "nwp_swrad_peak_h", "nwp_swrad_half_h", "nwp_swrad_late_frac",
                 "nwp_swrad_slope_pm", "nwp_cld_onset_h", "nwp_cld_slope_pm",
                 "pk_p_late"}
                | set(TN.xstn_feature_names()) | set(TN.CURVE_FEATS))
    for path, tag in ((args.nowcast_model, "临近预报模型"),
                      (args.nowcast_late_model, "临近预报晚时次模型")):
        if not os.path.exists(path):
            continue
        nc = json.load(open(path, encoding="utf-8"))
        nc_feats = {f for blk in nc.values() if isinstance(blk, dict)
                    for f in (blk.get("names") or blk.get("feats") or [])}
        bad = sorted(rejected & nc_feats)
        rep(not bad, f"被否决的实验特征没混进{tag}",
            "" if not bad else f"泄漏: {bad}")


# ---------------------------------------------------------------- 入口脚本

CRON_ENV = {"PATH": "/opt/homebrew/bin:/usr/bin:/bin",
            "HOME": os.environ.get("HOME", ""),
            "PLOYGON_LOG": "/tmp/_check_run.log",
            "PLOYGON_TAF_DB": "/tmp/_check_taf.sqlite"}


def check_scripts(args):
    """在**模拟 cron 的干净环境**里实跑入口脚本。

    为什么必须这样测: 之前几次"验证通过"用的都是 --no-fetch，把最耗时、
    最容易坏的取数那段整个跳过了，结果 cron 里连着两轮崩在
    `DBS[@]: unbound variable` 和 IEM 挂死上，本地却怎么测都是好的。
    交互式 shell 的 PATH / 环境变量与 cron 完全不同，必须用 env -i 复现。
    """
    print("\n[4] 入口脚本（模拟 cron 环境实跑）")
    for lock in ("/tmp/ploygon_run_hourly.lock", "/tmp/ploygon_run_daily.lock"):
        if os.path.isdir(lock):
            import shutil
            shutil.rmtree(lock, ignore_errors=True)

    r = subprocess.run(["./run_hourly.sh", str(args.cutoff)],
                       env=CRON_ENV, capture_output=True, text=True, timeout=1800)
    rows = [l for l in r.stdout.splitlines() if l.startswith("  Z")]
    rep(r.returncode == 0 and len(rows) == 8,
        "run_hourly.sh 在 cron 环境下完整跑通",
        f"退出码 {r.returncode}，输出 {len(rows)} 个站"
        + (f"｜{r.stderr.strip().splitlines()[-1][:80]}" if r.returncode else ""))
    degraded = [l for l in rows if "⚠" in l]
    rep(not degraded, "没有站落到降级路径（模式特征齐全）",
        "" if not degraded else f"{len(degraded)} 个站缺模式特征")

    # 单实例锁: 第二个实例必须被跳过，否则会堆积僵尸进程抢 sqlite 写锁
    import threading
    res = {}

    def slow():
        subprocess.run(["./run_hourly.sh", str(args.cutoff)], env=CRON_ENV,
                       capture_output=True, text=True, timeout=1800)
    t = threading.Thread(target=slow)
    t.start()
    import time as _t
    _t.sleep(1.0)
    r2 = subprocess.run(["./run_hourly.sh", str(args.cutoff)], env=CRON_ENV,
                        capture_output=True, text=True, timeout=600)
    t.join()
    rep("[skip]" in r2.stderr, "单实例锁生效（并发时第二个被跳过）",
        r2.stderr.strip()[:60])

    if args.skip_daily:
        return
    # run_daily 也要在 cron 环境下实跑。它耗时约 5 分钟（predict_mos 要为
    # 6 个模式各拉 30 天窗口），但正是这类「只在 cron 里坏」的地方最需要覆盖
    rd = subprocess.run(["./run_daily.sh", "0"], env=CRON_ENV,
                        capture_output=True, text=True, timeout=1800)
    lines = [l for l in rd.stdout.splitlines() if l.startswith("  Z")]
    rep(rd.returncode == 0 and len(lines) >= 16,
        "run_daily.sh 在 cron 环境下完整跑通",
        f"退出码 {rd.returncode}，输出 {len(lines)} 行")
    spec1 = json.load(open(args.mos_model, encoding="utf-8"))["1"]
    if spec1.get("prefer") == "blend":
        n_blend = rd.stdout.count("融合模型")
        rep(n_blend >= 16, "D+1/D+2 都用了融合模型（sklearn 可用）",
            f"{n_blend} 行走融合。为 0 说明 cron 的 python 没有 sklearn，"
            f"crontab 里要钉 PATH")


# ---------------------------------------------------------------- 契约

def check_contracts(args):
    print("\n[3] 脚本间契约")
    src = {f: open(f, encoding="utf-8").read()
           for f in ("run_hourly.sh", "run_daily.sh", "predict_nowcast.py",
                     "predict_mos.py", "train_nowcast.py", "merge_mos.py")}

    # 两个 shell 脚本的模式列表必须一致，否则 m2_/m3_ 会对错模式
    def models_of(text):
        for line in text.splitlines():
            if line.startswith("MODELS="):
                return line.split("=", 1)[1].strip()
        return None
    a, b = models_of(src["run_hourly.sh"]), models_of(src["run_daily.sh"])
    rep(a == b and a, "run_hourly.sh 与 run_daily.sh 的模式列表一致", a or "")

    n_models = len(a.split(",")) if a else 0
    spec = json.load(open(args.nowcast_model, encoding="utf-8"))
    for k, v in spec.items():
        need = sum(1 for n in v["names"] if n.endswith("_minus_m1"))
        if need:
            rep(need == n_models, f"{k} 时模型需要的追加模式数 == 脚本给的",
              f"模型要 {need}，脚本给 {n_models}")

    # 时区: 所有取「今天/现在」的地方都必须是北京时
    bad = [f for f in ("run_hourly.sh", "run_daily.sh")
           if "date +" in src[f] and "TZ=Asia/Shanghai date +" not in src[f]]
    rep(not bad, "shell 脚本一律用北京时取日期", "" if not bad else str(bad))

    rep("deb_weights" in src["predict_mos.py"] and "deb_weights" in src["merge_mos.py"],
      "DEB 训练端与预测端共用同一实现")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="", help="用哪天做比对，默认前天")
    ap.add_argument("--db", default="cn.sqlite")
    ap.add_argument("--cutoff", type=int, default=11)
    ap.add_argument("--nowcast-model", default="nowcast_nwp.json")
    ap.add_argument("--nowcast-late-model", default="nowcast_late.json")
    ap.add_argument("--mos-model", default="model.json")
    ap.add_argument("--mos-csv", default="mos_multi.csv")
    ap.add_argument("--nwp-csv", default="mos.csv")
    ap.add_argument("--nwp-csv2", nargs="+",
                    default=["mos_ecmwf.csv", "mos_cma.csv", "mos_icon.csv",
                             "mos_jma_.csv", "mos_gem_.csv"])
    ap.add_argument("--extra-models",
                    default="ecmwf_ifs025,cma_grapes_global,icon_global,"
                            "jma_gsm,gem_global")
    ap.add_argument("--skip-mos", action="store_true", help="跳过要联网的那项")
    ap.add_argument("--skip-daily", action="store_true",
                    help="跳过 run_daily.sh 实跑（约 5 分钟）")
    ap.add_argument("--skip-scripts", action="store_true",
                    help="跳过入口脚本实跑（那项要联网、约 1 分钟）")
    args = ap.parse_args()
    if not args.date:
        args.date = (datetime.now().date() - timedelta(days=2)).isoformat()

    print(f"一致性检查  比对日期 {args.date}")
    check_nowcast(args)
    if not args.skip_mos:
        check_mos(args)
    check_contracts(args)
    if not args.skip_scripts:
        check_scripts(args)

    print(f"\n{'='*60}")
    if FAIL:
        print(f"发现 {len(FAIL)} 处问题:")
        for x in FAIL:
            print(f"  - {x}")
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

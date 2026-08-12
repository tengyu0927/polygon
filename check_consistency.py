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
import re
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
    # 与 run_hourly.sh 同样的选择逻辑: 9-14 用六模式模型，15 用纯实况模型
    mpath = args.nowcast_model if cutoff <= 14 else args.nowcast_late_model
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
        # stn_id 必须与 make_samples 一样设上 —— 训练端设了、预测端不设，
        # 就是本项目最常犯的静默错配（2026-08-07 加这个特征时差点又踩一次）
        f, msf = N.build_feats(o, cutoff, prev, cr, cp,
                               tgt.timetuple().tm_yday, nwp.get(stn))
        f["stn_id"] = N.STN_IDX.get(stn)
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

    # gfs_live 预测时选的时效，必须与训练该时次时用的时效一致。
    # 2026-08-02 踩过: lag_h 写成 4.5,12:15 被踢到 18h,而 12 时的模型
    # 是拿 6h 训的 —— 静默错配,不报错、只是慢慢变差。
    try:
        import gfs_live as GL
        import datetime as _dt
        # 2026-08-03: 9/10/11 时从 18h 切到 12h（18Z 历史回补完成，
        # 15 个月回测 Δ=-0.0062、P=95.2%）。改这张表必须同步重训，
        # 训练侧对应 gfs_local_build.py --lead 12 出的 mos_local12.csv。
        want = {9: 12, 10: 12, 11: 12, 12: 6, 13: 6, 14: 6}
        tgt = _dt.date.today() + _dt.timedelta(days=1)
        bad = []
        for cut, lead in want.items():
            cand = GL.pick_run(cut, tgt)
            got = None
            if cand:
                peak = _dt.datetime.combine(tgt, _dt.time(6), _dt.timezone.utc)
                got = round((peak - cand[0][0]).total_seconds() / 3600)
            if got != lead:
                bad.append(f"{cut}时: 预测选{got}h/训练{lead}h")
        rep(not bad, "gfs_live 各时次选的时效与训练一致",
            "" if not bad else "; ".join(bad))
    except Exception as e:                            # noqa: BLE001
        rep(False, "gfs_live 时效一致性检查", f"{type(e).__name__}: {str(e)[:60]}")

    # 站点一个都不能少。数量取自 stations.py（唯一真相源），别再写死数字 ——
    # 2026-08-08 加郑州/济南时，清单在 17 个文件里各抄一遍，靠这条守卫驱动才
    # 一处不落地改完。2026-08-01 踩过: ZGSZ 换成 WU 序列后只剩 2024-07 起
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
            rep(not miss, f"{tag} {cut} 时含全部 {len(TN.NAMES)} 站",
                "" if not miss else f"缺: {miss}（重训漏了 --split-year 2025？）")

    leaked = sorted((TM.CONV_COLS | TM.PROF_COLS | TM.BIASW_COLS) & feats)
    rep(not leaked, "被否决的实验特征没混进 MOS 上线模型",
        "" if not leaked else f"泄漏: {leaked}")
    # 临近预报侧同理。跨站特征更危险: predict_nowcast.py 根本没有算它们的代码，
    # 带着 PLOYGON_XSTN=1 训练出来的模型上线后会静默拿到 None。
    # oracle_* 是明知泄漏的先知实验特征（量「见顶时刻不确定」这个瓶颈值多少），
    # 上线就等于用未来信息预报。必须在这里挡死。
    # 2026-08-06: 对流(CONV)、辐射/云廓线(PROF)、温度曲线(CURVE)、上午风速
    # (WINDAM)、边界层(HPBL) 这五族**已从黑名单移出** —— 它们当年是在**纯线性**
    # 模型里被否的，加了 GBM 之后一起重测，9/10 时 -0.0372 / -0.0270（P 均 100%）。
    # 同一批要素单独线性加是 +0.0068(P=0%)、非线性里是负 —— 结论会随模型结构翻转。
    # 仍在黑名单里的:
    #   oracle_*  明知泄漏，永远不许上线
    #   pk_p_late 见顶时刻辅助模型的输出，未通过 A/B
    #   x1_/x2_   跨站: 训练端能算、**预测端逐站循环产不出来**，
    #             2026-08-06 被 [1] 的逐列比对当场抓到，故仍禁止
    #   mosd_*    D+1 链路的订正后输出。2026-08-12 否决 —— 首轮用单次切分的
    #             MOS 序列测出 9/10 时 P=99.4%/99.9%，换成与生产同口径的逐月
    #             滚动序列后，10 时（原本最强）直接归零（+0.0027, P=30%），
    #             11 时显著更差。详见 README。
    rejected = ({"oracle_peak_h", "oracle_hours_to_peak", "pk_p_late"}
                | set(TN.xstn_feature_names()) | set(TN.MOSF_FEATS)
                | set(TN.SONDE_FEATS))
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

# 实跑会真的出预报、真的写库，所以每个落盘的去处都必须改道到临时文件。
# PLOYGON_VERIFY_DB 原来漏了 —— 检查一跑就把预报写进真的 verify.sqlite，
# 于是「00:13 的 cron 结果」和「01:35 的检查实跑」在库里混成一份，
# 武汉 08-03 在 verify.sqlite 里是 38、在 pred_mos.csv 里是 37。
# 检验库是判断规则对错的唯一依据，绝不能掺进检查跑出来的行。
# **这里的环境必须与真实 crontab 一致**，否则这一项测的不是生产在跑的东西。
# 2026-08-11 踩过: crontab 里只有 PATH，没有 HTTP_PROXY/HTTPS_PROXY，而
# gfs_live 拉 NOAA S3 的字节范围下载在无代理下从 90 秒变成 >500 秒 ——
# 整轮从约 2 分钟拖到 20 分钟以上，:35 那批因锁被跳过五次，这一项也连着
# 几天超时。下面的 _cron_vars() 会把真实 crontab 的变量读进来，并在与当前
# shell 有出入时报出来。
def _cron_vars():
    """读真实 crontab 里的 VAR=VALUE 赋值行。取不到就返回空。"""
    try:
        out = subprocess.run(["crontab", "-l"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:                                 # noqa: BLE001
        return {}
    v = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, val = line.partition("=")
        if k.strip().isidentifier():
            v[k.strip()] = val.strip()
    return v


CRON_VARS = _cron_vars()
CRON_ENV = {"PATH": CRON_VARS.get("PATH", "/opt/homebrew/bin:/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", ""),
            "PLOYGON_LOG": "/tmp/_check_run.log",
            "PLOYGON_TAF_DB": "/tmp/_check_taf.sqlite",
            "PLOYGON_VERIFY_DB": "/tmp/_check_verify.sqlite",
            "PLOYGON_STATE": "/tmp/_check_state.json"}
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY"):
    if _k in CRON_VARS:
        CRON_ENV[_k] = CRON_VARS[_k]


def check_scripts(args):
    """在**模拟 cron 的干净环境**里实跑入口脚本。

    为什么必须这样测: 之前几次"验证通过"用的都是 --no-fetch，把最耗时、
    最容易坏的取数那段整个跳过了，结果 cron 里连着两轮崩在
    `DBS[@]: unbound variable` 和 IEM 挂死上，本地却怎么测都是好的。
    交互式 shell 的 PATH / 环境变量与 cron 完全不同，必须用 env -i 复现。
    """
    print("\n[4] 入口脚本（模拟 cron 环境实跑）")
    # 先查环境漂移: 当前 shell 有代理而 crontab 没有 -> 生产每轮都在裸连
    # NOAA S3，整轮会从约 2 分钟拖到 20 分钟以上（2026-08-11 实测 90s vs >500s）
    miss = [k for k in ("HTTP_PROXY", "HTTPS_PROXY")
            if os.environ.get(k) and k not in CRON_VARS and k.lower() not in CRON_VARS]
    rep(not miss, "crontab 的代理设置与当前 shell 一致",
        "" if not miss else f"crontab 缺 {miss}，生产会裸连 —— 加到 crontab 的 PATH 行下面")
    # 锁按批次分名（run_hourly.sh 里 LOCK 带 --stations 的取值），所以要通配
    import glob as _g
    import shutil
    for lock in (_g.glob("/tmp/ploygon_run_hourly*.lock")
                 + ["/tmp/ploygon_run_daily.lock"]):
        if os.path.isdir(lock):
            shutil.rmtree(lock, ignore_errors=True)

    r = subprocess.run(["./run_hourly.sh", str(args.cutoff)],
                       env=CRON_ENV, capture_output=True, text=True, timeout=1800)
    # 只认真正的站行: ICAO + 站名 + 数值列。光看 startswith("  Z") 会把提示行
    # 也数进去（"  ZGSZ 改用 WU 实况…" 就让这里数出 9 个站，误报了一整轮）。
    # 数值列必须同时认数字和 "--" —— 观测不足时整行是 "--"，只认数字的话
    # 半夜跑这项会数出 0 个站，等于把误报从 9 换成了 0。
    rows = [l for l in r.stdout.splitlines()
            if re.match(r"^  Z[A-Z]{3}\s+\S+\s+(-?\d|--)", l)]
    # 站数取自 stations.py。2026-08-11 踩过: 这里写死 8，扩到 10 站时漏改，
    # 于是实跑明明通过（退出码 0、10 个站、其余五项全绿）却报 ✗。
    import stations as _ST2
    rep(r.returncode == 0 and len(rows) == len(_ST2.ICAOS),
        "run_hourly.sh 在 cron 环境下完整跑通",
        f"退出码 {r.returncode}，输出 {len(rows)} 个站"
        + (f"｜{r.stderr.strip().splitlines()[-1][:80]}" if r.returncode else ""))
    degraded = [l for l in rows if "⚠" in l]
    rep(not degraded, "没有站落到降级路径（模式特征齐全）",
        "" if not degraded else f"{len(degraded)} 个站缺模式特征")

    # 第七成员（本地 GFS）必须真的取到。2026-08-03 踩过: cron 的 PATH 指向
    # /opt/homebrew/bin/python3，而那个解释器上的 eccodes 不知何时没了，
    # gfs_live 每轮都吐「需要 eccodes」-> 该成员留空。模型是**带着 m7_ 特征
    # 训练的**，预测端拿不到就是静默错配 —— 上面那两项当时全是 ✓，
    # 因为它们只看站行里有没有 ⚠，看不见某个成员整体缺失。
    blob = r.stdout + r.stderr
    m7bad = [k for k in ("本地 GFS 全部轮次取不到", "需要 eccodes") if k in blob]
    rep(not m7bad, "第七成员（本地 GFS）实时取到了",
        "" if not m7bad else f"{m7bad} —— 模型带 m7_ 特征训练，取不到就是训练/预测错配")

    # 9/10 时的模型是 ridge+GBM 融合（gbm_w=0.3）。GBM 存在 <model>.gbm.pkl，
    # 缺 sklearn / 缺文件时 predict_nowcast 会自动降级成纯线性 —— 不报错，
    # 但那是拿 0.3 权重的岭回归当全部答案，等于**静默丢掉 70% 的模型**。
    # 备注里出现 "+GBM" 才说明真的融合了。
    import json as _json
    _spec = _json.load(open(args.nowcast_model, encoding="utf-8"))
    _need = {c for c, v in _spec.items() if v.get("gbm_w") is not None}
    # 观测不足时整表是 "--"，那一轮压根没出预报，自然也没有 +GBM —— 跳过，
    # 否则半夜跑这项必然误报（与站行正则那次同一类坑）
    _live = [l for l in rows if "--" not in l]
    if _need and str(args.cutoff) in _need and _live:
        rep("+GBM" in blob, f"{args.cutoff} 时的 GBM 真的参与了融合",
            "" if "+GBM" in blob else
            "备注里没有 +GBM —— 缺 sklearn 或缺 .gbm.pkl，静默退回纯线性")

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

    # 两个脚本的模式列表: 顺序必须一致（否则 m2_/m3_ 各列错位），但不再要求
    # 完全相同 —— 临近侧多一个 local_gfs（实时从 NCEP 取最新一轮的第七成员），
    # D+1/D+2 那条链路不用它。要求 run_daily 的列表是 run_hourly 的前缀。
    def models_of(text):
        for line in text.splitlines():
            if line.startswith("MODELS="):
                return line.split("=", 1)[1].strip()
        return None
    a, b = models_of(src["run_hourly.sh"]), models_of(src["run_daily.sh"])
    la = a.split(",") if a else []
    lb = b.split(",") if b else []
    ok = bool(la) and bool(lb) and la[:len(lb)] == lb
    rep(ok, "run_daily 的模式列表是 run_hourly 的前缀（顺序一致）",
        f"临近多出: {la[len(lb):]}" if ok and len(la) > len(lb) else (a or ""))

    # 9 时另有一份清单（MODELS9，多第八成员 AIFS）。按时次取对应的那份 ——
    # 2026-08-12 加 AIFS 时这里只读 MODELS= 一行，当场报错，正是它该做的。
    def models9_of(text):
        for line in text.splitlines():
            if line.startswith("MODELS9="):
                v = line.split("=", 1)[1].strip()
                return v.replace("$MODELS", models_of(text) or "")
        return None
    a9 = models9_of(src["run_hourly.sh"])
    n_models = len(a.split(",")) if a else 0
    n_models9 = len(a9.split(",")) if a9 else n_models
    spec = json.load(open(args.nowcast_model, encoding="utf-8"))
    for k, v in spec.items():
        need = sum(1 for n in v["names"] if n.endswith("_minus_m1"))
        if need:
            want = n_models9 if k == "9" else n_models
            rep(need == want, f"{k} 时模型需要的追加模式数 == 脚本给的",
              f"模型要 {need}，脚本给 {want}")

    # 时区: 所有取「今天/现在」的地方都必须是北京时
    bad = [f for f in ("run_hourly.sh", "run_daily.sh")
           if "date +" in src[f] and "TZ=Asia/Shanghai date +" not in src[f]]
    rep(not bad, "shell 脚本一律用北京时取日期", "" if not bad else str(bad))

    rep("deb_weights" in src["predict_mos.py"] and "deb_weights" in src["merge_mos.py"],
      "DEB 训练端与预测端共用同一实现")

    # 超出概率表必须在，否则「更高?」那一列静默变空 —— 晚见顶的站在 15 时
    # 就又没有任何提示了（2026-08-12 加这一列正是为了补那个缺口）。
    rep(os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "exceed_table.json")),
      "exceed_table.json 在（「更高?」列的数据源）")

    # 站点清单只能有一个真相源 = stations.py。别处再抄一份，加站时必漏。
    # 已经犯过三次: wu_obs.to_rows 硬写 STATION="ZGSZ"（把济南观测灌进深圳
    # 25344 行）、predict_nowcast 硬写 WU_STATIONS={"ZGSZ"}（济南 12 时起
    # 「无今日观测」）、stn_id 训练端设了预测端没设。判据是**语法级**的:
    # 任何 .py 里出现「字面量 ICAO 集合/列表」的赋值，就是又抄了一份。
    # 两条判据，缺一不可:
    #   (a) 名字撞车: 赋值给 stations.py 也导出的同名变量，右边是字面量。
    #       —— 2026-08-10 两次都是这条: predict_nowcast 和 live_tmax 各抄了
    #       一份 WU_STATIONS = {"ZGSZ"}。**只含 1 个站，光看规模抓不到。**
    #   (b) 规模: 字面量里出现 3 个以上 ICAO，不管叫什么名字。
    # 确有正当理由的（如 XPARTNER 伙伴站映射），在赋值那行末尾写
    # `# stations-ok: 理由` 豁免 —— 例外必须写出来，不能靠调松阈值放过。
    import ast as _ast, glob as _glob
    import stations as _ST
    icao = set(_ST.ICAOS)
    exported = {n for n in vars(_ST) if not n.startswith("_") and n.isupper()}
    dup = []
    for f in _glob.glob("*.py"):
        if f in ("stations.py", "check_consistency.py"):
            continue
        try:
            text = open(f, encoding="utf-8").read()
            tree = _ast.parse(text)
        except SyntaxError:
            continue
        lines = text.splitlines()
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Assign):
                continue
            v = node.value
            tgt0 = node.targets[0]
            nm0 = tgt0.id if isinstance(tgt0, _ast.Name) else None
            if (nm0 in exported
                    and isinstance(v, (_ast.Set, _ast.List, _ast.Tuple,
                                       _ast.Dict, _ast.Constant))
                    and "stations-ok:" not in lines[node.lineno - 1]):
                dup.append(f"{f}:{node.lineno} {nm0} 抄了 stations.{nm0}")
                continue
            if isinstance(v, (_ast.Set, _ast.List, _ast.Tuple)):
                lits = {e.value for e in v.elts
                        if isinstance(e, _ast.Constant) and isinstance(e.value, str)}
            elif isinstance(v, _ast.Dict):
                lits = {k.value for k in v.keys
                        if isinstance(k, _ast.Constant) and isinstance(k.value, str)}
            else:
                continue
            hit = lits & icao
            # 一两个站是特例（如 LEGACY 单站修正），三个以上就是在抄清单
            if len(hit) >= 3:
                if "stations-ok:" in lines[node.lineno - 1]:
                    continue
                tgt = node.targets[0]
                nm = tgt.id if isinstance(tgt, _ast.Name) else "?"
                dup.append(f"{f}:{node.lineno} {nm}={sorted(hit)[:3]}...")
    rep(not dup, "站点清单只在 stations.py 里写死（别处不得再抄）",
      "" if not dup else " | ".join(dup))


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
    # 默认留空，parse 之后按时次补第 7 个成员（本地 GFS）—— 见下面
    ap.add_argument("--nwp-csv2", nargs="+", default=None)
    ap.add_argument("--extra-models",
                    default="ecmwf_ifs025,cma_grapes_global,icon_global,"
                            "jma_gsm,gem_global")
    ap.add_argument("--skip-mos", action="store_true", help="跳过要联网的那项")
    ap.add_argument("--skip-daily", action="store_true",
                    help="跳过 run_daily.sh 实跑（约 5 分钟）")
    ap.add_argument("--skip-scripts", action="store_true",
                    help="跳过入口脚本实跑（那项要联网、约 1 分钟）")
    args = ap.parse_args()
    if args.nwp_csv2 is None:
        # 第 7 个集合成员（本地 GFS）必须按时次配时效，和训练时一模一样:
        # 9-11 时用前一天 12Z 的 18 小时，12-13 时用当天 00Z 的 6 小时。
        # 上线时漏了这一步，检查只喂 5 个追加模式，于是天天报「缺 m7_*」——
        # 生产其实是好的（run_hourly 传了 local_gfs），是检查自己没跟上。
        # 2026-08-12 修正两处陈旧:
        #   (a) 9-11 时是 mos_local12.csv 不是 local18 —— 2026-08-03 就从 18h
        #       切到 12h 了（Δ=-0.0062、P=95.2%），本行漏改，逐列比对一直在
        #       跟一个**不上线的成员**比。上面 want={9:12,10:12,11:12} 那张表
        #       早就是对的，两处不一致本身就该被发现。
        #   (b) 9 时多第八成员 mos_aifs.csv（AIFS），与 run_hourly 的 MODELS9 对应。
        args.nwp_csv2 = ["mos_ecmwf.csv", "mos_cma.csv", "mos_icon.csv",
                         "mos_jma_.csv", "mos_gem_.csv",
                         "mos_local12.csv" if args.cutoff <= 11
                         else "mos_local6.csv"]
        if args.cutoff == 9:
            # 只动 nwp_csv2（临近侧逐列比对用它）。**别碰 args.extra_models**
            # —— 那个是给 D+1 的 predict_mos 用的，只有 5 个模式，
            # 2026-08-12 误改过一次，当场把 predict_mos 跑挂。
            args.nwp_csv2.append("mos_aifs.csv")
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

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
import collections
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
sys.path.insert(0, HERE)
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
    #   oracle_*  明知泄漏，永远不许上线。**按前缀挡** —— 写死名字的话，
    #             以后每加一族先知特征都要记得同步，早晚漏。
    #             现有两族: 见顶时刻(ORACLE_PEAK)、午后天空(ORACLE_SKY)
    #   pk_p_late 见顶时刻辅助模型的输出，未通过 A/B
    #   x1_/x2_   跨站: 训练端能算、**预测端逐站循环产不出来**，
    #             2026-08-06 被 [1] 的逐列比对当场抓到，故仍禁止
    #   mosd_*    D+1 链路的订正后输出。2026-08-12 否决 —— 首轮用单次切分的
    #             MOS 序列测出 9/10 时 P=99.4%/99.9%，换成与生产同口径的逐月
    #             滚动序列后，10 时（原本最强）直接归零（+0.0027, P=30%），
    #             11 时显著更差。详见 README。
    # **oracle_ 按前缀挡，不要写死名字。** 2026-08-22 加 ORACLE_SKY 时发现:
    # 原来这里列的是 oracle_peak_h / oracle_hours_to_peak 两个具体名字，
    # 新加一族先知特征就会直接漏网 —— 而先知特征上线等于用未来信息预报，
    # 是这份检查里后果最重的一条。
    rejected = ({"pk_p_late"} | set(TN.xstn_feature_names())
                | set(TN.MOSF_FEATS) | set(TN.SONDE_FEATS))
    _pref = ("oracle_",)
    for path, tag in ((args.nowcast_model, "临近预报模型"),
                      (args.nowcast_late_model, "临近预报晚时次模型")):
        if not os.path.exists(path):
            continue
        nc = json.load(open(path, encoding="utf-8"))
        nc_feats = {f for blk in nc.values() if isinstance(blk, dict)
                    for f in (blk.get("names") or blk.get("feats") or [])}
        bad = sorted((rejected & nc_feats)
                     | {f for f in nc_feats if f.startswith(_pref)})
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
    # **只取主预报表**: 表头（站点/预报/不排除/已达）之后、第一个 `──` 之前。
    # 2026-08-19 踩过: 新加的「见顶时刻概率」「档位配置建议」「近期偏差订正」
    # 各自也是「两空格 + ICAO + 站名 + 数字」的格式（见顶概率那行是
    # "ZUCK 重庆江北  1%  38%…"，"1%" 同样匹配数字），于是这里数出 30 个站；
    # 而见顶概率里的提示语「⚠ 大概率拖到下午晚些」又被下面 degraded 那行
    # 当成降级标记，凭空报出「1 个站缺模式特征」。**两处都是误报。**
    rows, _inblk = [], False
    for l in r.stdout.splitlines():
        if re.match(r"^\s*站点\s+预报\s+不排除\s+已达", l):
            _inblk = True
            continue
        if _inblk and l.lstrip().startswith(("──", "关于", "「")):
            _inblk = False
            continue
        if _inblk and re.match(r"^  Z[A-Z]{3}\s+\S+\s+(-?\d|--)", l):
            rows.append(l)
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
    rep(os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "bucket_table.json")),
      "bucket_table.json 在（档位配置建议的数据源）")

    # 见顶判别器 v2（分站/纯观测）。**特征函数训练端与预测端共用 peak2_feats**，
    # 但模型是按站存的 —— 站少了就整站没提示，而且不报错。所以查覆盖。
    try:
        import pickle as _pk6
        import train_nowcast as _TN6
        import stations as _ST6
        for _mp6 in ("nowcast_nwp.json", "nowcast_late.json"):
            _f6 = _mp6 + ".peak2.pkl"
            if not os.path.exists(_f6):
                rep(True, f"{os.path.basename(_f6)}", "缺（生产退回旧判别器）")
                continue
            _m6 = _pk6.load(open(_f6, "rb"))
            _cuts = sorted(_m6)
            _miss = {c: sorted(set(_ST6.ICAOS) - set(_m6[c])) for c in _cuts}
            _bad = {c: v for c, v in _miss.items() if v}
            rep(not _bad, f"{os.path.basename(_f6)} 覆盖全部 10 站",
                f"时次 {_cuts}" + ("" if not _bad else f"  -> 缺 {_bad}"))
            for c in _cuts[:1]:
                _n6 = _m6[c][sorted(_m6[c])[0]]["clf"].n_features_in_
                rep(_n6 == len(_TN6.PEAK2_ALL),
                    f"{c} 时 v2 特征数与 PEAK2_ALL 一致",
                    f"模型 {_n6} / 定义 {len(_TN6.PEAK2_ALL)}"
                    + ("" if _n6 == len(_TN6.PEAK2_ALL) else
                       "  -> 特征表改过，需重跑 fit_peak_prob2"))
    except Exception as _e6:                               # noqa: BLE001
        rep(False, "见顶判别器 v2 可检查", f"{type(_e6).__name__}: {_e6}")

    # 见顶提示记账器: 它读的是 predict_nowcast 印出来的表，**格式一变就静默
    # 解析不到**（记 0 条也不报错）。所以这里拿最近一天的日志真跑一遍解析。
    try:
        import datetime as _dt5
        import sqlite3 as _sq5
        import peak_hint_track as _H5
        # **查最近 3 天里最新的那份日志，不是「第一份能找到的」。**
        # 2026-08-26 踩过: 改了表头文案后记账器解析不到，而这条检查当时是
        # 绿的 —— 因为它从昨天开始往回找，找到的是改文案之前的旧日志。
        # 改成: 今天的日志若存在就必须能解析；否则往回最多找 3 天。
        _got, _d5 = [], None
        for _k in range(0, 4):
            _dd = (_dt5.date.today() - _dt5.timedelta(days=_k)).isoformat()
            if os.path.exists(f"pred_{_dd}.log"):
                _g = _H5.parse(_dd)
                if _d5 is None:
                    _d5, _got = _dd, _g       # 最新那份是判据
                if _g:
                    break
        rep(bool(_got), "见顶提示记账器能解析当前日志格式",
            f"{_d5} 解出 {len(_got)} 条标注"
            + ("" if _got else "  -> 表头文案改过？看 peak_hint_track 的标题匹配"))
        # **两种提示要分开查。** 只看总条数会漏掉「一类被吞、另一类还在」——
        # 2026-08-26~27 就是这样: ⚠ 的提示文案「⚠ 16 点后还在涨，整数可能改」
        # 与段落标题「── 16 点后还在涨、会改掉整数的风险」撞词，每条 ⚠ 数据行
        # 都被 parse 当成标题跳过，两天一条 ⚠ 没记上，而这条检查一直是绿的
        # （早早还解得出来，总条数不为零）。
        # 判据: 日志里 ⚠ 出现过几次，就该解出几条 warn。
        if _d5:
            _txt5 = open(f"pred_{_d5}.log", encoding="utf-8",
                         errors="replace").read()
            _want = _txt5.count("⚠ 16 点后")
            _have = sum(1 for _r in _got if _r[2] == "warn")
            rep(_want == _have, "⚠ 提示解析条数与日志一致",
                f"{_d5} 日志 {_want} 条 / 解出 {_have} 条"
                + ("" if _want == _have else
                   "  -> 提示文案与段落标题撞词？看 peak_hint_track.parse"))
        # **记账库有没有真的在长。** 上面两条只验「解析得出来」，验不了
        # 「run_daily 有没有把它记进库」—— 2026-08-31 查出记账器上线以来
        # 从 cron 里一条都没记成过: run_daily 23:59 启动，跑到那一步已过
        # 午夜，不给 --date 时 `now()` 返回「明天」，明天没数据就静默跳过，
        # 日志只留一行「峰值时段还没走完」，看着像正常拦截。库里的数据全是
        # 手动回填的，而上面两条检查一直是绿的。
        # 判据: 昨天（峰值时段必然已走完）应当在库里。
        _yst = (_dt5.date.today() - _dt5.timedelta(days=1)).isoformat()
        try:
            _c5 = _sq5.connect(os.path.join(HERE, "peak_hint.sqlite"))
            _n5 = _c5.execute(
                "SELECT count(*) FROM hint WHERE target_date = ?", (_yst,)
            ).fetchone()[0]
            _last = _c5.execute("SELECT max(target_date) FROM hint").fetchone()[0]
            _c5.close()
            _err5 = None
        except Exception as _e5b:                          # noqa: BLE001
            # **不能把「读不出来」伪装成「0 行」。** 2026-08-31 就栽在这:
            # 这段里一个 NameError 被吞掉、回退成 0 行，报出来像是真的没记上，
            # 查了半天才发现是检查器自己坏了。查静默失效的检查项自己静默失效，
            # 正是本仓库反复踩的那个形态。
            _n5, _last, _err5 = None, None, f"{type(_e5b).__name__}: {_e5b}"
        if _err5:
            rep(False, "记账库昨天有进账", f"库读不出来 —— {_err5}")
        else:
            rep(_n5 > 0, "记账库昨天有进账",
                f"{_yst} {_n5} 行，库里最新 {_last}"
                + ("" if _n5 else "  -> run_daily 那一步没记上？看是否漏传 --date"))
    except Exception as _e5:                               # noqa: BLE001
        rep(False, "见顶提示记账器可用", f"{type(_e5).__name__}: {_e5}")

    # **起报时刻的观测可得性 vs 事后重算。**
    #
    # `load_hourly` 按**整点小时**分桶、取桶内最高温，特征取 `h <= cutoff`。
    # 生产 :15 起报，看不到本小时 :30/:45 的报文；训练却把它算进了「已达」。
    # 于是训练特征里的「已达」系统性高于生产能拿到的值 —— 这是训练/线上错配，
    # 不只是回测偏乐观。2026-08-31 量到（2024 年起，66038 个站日小时）:
    #     时次    「:15 后更高」占比   期望抬升
    #      9时        17.1%        +0.204℃
    #     11时        13.8%        +0.156℃
    #     13时         9.5%        +0.106℃
    #     15时         4.8%        +0.053℃
    #     合计        11.3%        +0.128℃（发生时平均高 1.14℃）
    #
    # **上面那条 [1] 逐列比对看不见它** —— 它训练端走 load_hourly、预测端走
    # from_db，但两边都在事后读同一个 cn.sqlite，那时 :30 的报文早到齐了。
    # 唯一能看见的办法是拿**生产当时真的印出来的**「已达」跟事后重算的比，
    # 就是下面这条。
    #
    # 判据是「有没有变得更糟」，不是「有没有这个毛病」—— 毛病还在，修法要把
    # cutoff 从小时改成时刻（h:15），同时动训练和预测两端，改完所有历史回测
    # 数字失效、需重跑重训。在那之前这条至少让它每天可见。
    try:
        import glob as _g7
        _lg = sorted(_g7.glob(os.path.join(HERE, "pred_2026-*.log")))
        _row7 = re.compile(r"^\s{2,4}(Z[A-Z]{3}) \S+\s+(-?\d+)\s+(-?\d+)"
                           r"\s+(-?\d+)\s+[+-]?[\d.]+\s+\d+%")
        _n7 = _d7 = 0
        _gap = 0.0
        _worst = None
        if _lg:
            _f7 = _lg[-2] if len(_lg) >= 2 else _lg[-1]   # 用完整的前一天
            _day7 = os.path.basename(_f7)[5:15]
            _hb = collections.defaultdict(dict)
            for _s7, _h7, _t7 in _sq5.connect(
                    os.path.join(HERE, "cn.sqlite")).execute(
                    "SELECT station, CAST(strftime('%H',"
                    "datetime(valid_time_gmt,'unixepoch','+8 hours')) AS INT),"
                    " temp_c FROM obs WHERE local_date = ? AND temp_c IS NOT NULL",
                    (_day7,)):
                if _h7 not in _hb[_s7] or _t7 > _hb[_s7][_h7]:
                    _hb[_s7][_h7] = _t7
            _cut7 = None
            for _ln in open(_f7, encoding="utf-8", errors="replace"):
                _m7 = re.search(r"(\d{1,2}) 时起报", _ln)
                if _m7 and "#####" in _ln:
                    _cut7 = int(_m7.group(1))
                    continue
                _r7 = _row7.match(_ln.rstrip())
                if not _r7 or _cut7 is None or _cut7 > 15:
                    continue
                _v7 = _hb.get(_r7.group(1))
                if not _v7:
                    continue
                _post = [_t for _hh, _t in _v7.items() if _hh <= _cut7]
                if not _post:
                    continue
                _live, _now = int(_r7.group(4)), max(_post)
                _n7 += 1
                if round(_now) - _live >= 1:
                    _d7 += 1
                    _gap += _now - _live
                    if _worst is None or _now - _live > _worst[3]:
                        _worst = (_day7, _cut7, _r7.group(1), _now - _live)
        _frac = (_d7 / _n7) if _n7 else 0.0
        _mean = (_gap / _n7) if _n7 else 0.0
        rep(_n7 == 0 or (_frac <= 0.25 and _mean <= 0.40),
            "起报时「已达」与事后重算的差距没有变大",
            f"{_d7}/{_n7} 行偏低（{100 * _frac:.0f}%），平均 {_mean:+.2f}℃"
            + (f"，最大 {_worst[2]} {_worst[1]}时 {_worst[3]:+.0f}℃" if _worst else "")
            + "  [已知: load_hourly 按整点分桶，训练看得到 :15 后的报文，生产看不到]")
    except Exception as _e7:                               # noqa: BLE001
        rep(False, "起报时观测可得性可检查", f"{type(_e7).__name__}: {_e7}")

    # run0 前瞻采集的字段覆盖。**这条要攒十个月才见分晓，所以必须机器查。**
    # 2026-08-22: 采了三周才发现旧表只存了 gfs_global 的 temperature_2m，
    # 而模型要的是逐模式的 tmax/cloud_peak/swrad_peak —— 照那样攒满十个月，
    # 攒完会发现一个特征都建不出来。
    try:
        import build_mos_dataset as _B4
        import run0_probe as _R4
        _need = set(_B4.VARS)
        _miss = sorted(_need - set(_R4.VARS))
        rep(not _miss, "run0 采集覆盖训练端要的全部模式变量",
            f"需要 {len(_need)} 个，采了 {len(_R4.VARS)} 个"
            + (f"  -> 缺 {_miss}" if _miss else ""))
        # 9 时那份清单（含 AIFS）去掉 local_gfs —— 本地 GFS 走自己的归档，
        # 不从 previous-runs 接口取，所以不该要求 probe 采它。
        _want4 = [m for m in (a9 or a or "").split(",")
                  if m and m != "local_gfs"]
        _mm = [m for m in _want4 if m not in _R4.MODELS]
        rep(not _mm, "run0 采集覆盖生产在用的全部追加模式",
            f"采了 {len(_R4.MODELS)} 个" + (f"  -> 缺 {_mm}" if _mm else ""))
    except Exception as _e4:                               # noqa: BLE001
        rep(False, "run0 采集字段可检查", f"{type(_e4).__name__}: {_e4}")

    # 「见顶时刻」判别器（<model>.peak.pkl）。它用主模型的 names/median 训，
    # 主模型一变就必须重训（--peak-only），否则 matrix() 填缺值的基准对不上，
    # 而且**不会报错**，只是概率悄悄失真。这里校验特征数与主模型一致。
    # **两个模型都要查。** 2026-08-22 前这里只查 nowcast_model（9-14 时），
    # 15 时的 nowcast_late.json.peak.pkl 不在范围内 —— 于是「它训好落盘了，
    # 但生产因为缺 GBM 边文件而根本没读过」这个 bug 活了一直没被抓到。
    _any_pk = False
    for _mp in (args.nowcast_model, args.nowcast_late_model):
        _pkp = _mp + ".peak.pkl"
        if not os.path.exists(_pkp):
            continue
        _any_pk = True
        try:
            import pickle as _pk9
            _pm = _pk9.load(open(_pkp, "rb"))
        except Exception:                              # noqa: BLE001
            _pm = {}
        _spec = json.load(open(_mp, encoding="utf-8"))
        rep(bool(_pm), f"{os.path.basename(_pkp)} 读得出来", f"时次 {sorted(_pm)}")
        for _c in sorted(_pm):
            # matrix() 每个特征产两列（值 + 缺失标记），所以是 2×names
            _n_model = 2 * len(_spec.get(str(_c), {}).get("names", []))
            _n_clf = int(getattr(_pm[_c]["clf"], "n_features_in_", -1))
            rep(_n_clf == _n_model, f"{_c} 时见顶判别器特征数与主模型一致",
                f"判别器 {_n_clf} / 模型 2×{_n_model // 2}={_n_model}"
                + ("" if _n_clf == _n_model else "  -> 主模型换过，需重跑 --peak-only"))
            # 两张校准表都要单调。cal_e（早<13时）是 2026-08-22 加的 ——
            # 还没重训的时次没有这一张，那时 predict_nowcast 退回原始概率，
            # 只提示、不算错。
            for _k, _lab in (("cal", "晚>=16"), ("cal_e", "早<13")):
                _cal = _pm[_c].get(_k)
                if _cal is None:
                    rep(True, f"{_c} 时「{_lab}」校准表",
                        "缺（该时次还没按新版重训，生产退回原始概率）")
                    continue
                _ok = all(a2 is not None and b2 is not None and a2 <= b2
                          for a2, b2 in zip(_cal, _cal[1:]))
                rep(_ok, f"{_c} 时「{_lab}」见顶概率校准表单调",
                    str([None if x is None else round(x, 3) for x in _cal]))
    if not _any_pk:
        rep(False, "peak.pkl 在（见顶时刻概率的数据源）", "缺文件 -> 该段不输出")

    # 追加模式的训练文件: 站点覆盖必须与 stations.ICAOS 一致，且不能太旧。
    # 2026-08-18 加: mos_local12/6.csv 停在 08-05/08-01（13-17 天）且只有 8 站
    # （缺郑州、济南）—— 训练端那两站的 m7 恒为空，生产端 gfs_live 却实时取到
    # 真值。**逐列比对抓不到**（两端读同一个 CSV，是空对空），run_hourly 的
    # 实跑也只验「取到了」不验「训练端有没有」。
    import datetime as _dt2
    import stations as _ST3
    _today2 = _dt2.date.today()
    for _f in ("mos.csv", "mos_ecmwf.csv", "mos_cma.csv", "mos_icon.csv",
               "mos_jma_.csv", "mos_gem_.csv", "mos_local12.csv",
               "mos_local6.csv", "mos_aifs.csv"):
        if not os.path.exists(_f):
            rep(False, f"{_f} 在")
            continue
        _st3, _last3, _days3 = set(), "", set()
        for _r in csv.DictReader(open(_f, encoding="utf-8")):
            if _r.get("lead") not in (None, "", "1"):
                continue
            _st3.add(_r["station"])
            _days3.add(_r.get("date", ""))
            _last3 = max(_last3, _r.get("date", ""))
        _missing = sorted(set(_ST3.ICAOS) - _st3)
        rep(not _missing, f"{_f} 覆盖全部 {len(_ST3.ICAOS)} 站",
            "" if not _missing else f"缺 {_missing} -> 这些站训练端该成员恒为空，"
                                    f"而生产端会实时取到真值")
        _lag3 = ((_today2 - _dt2.date.fromisoformat(_last3)).days
                 if _last3 else 999)
        # 本地 GFS 靠归档回补，允许旧一些；模式 csv 由 run_daily 每天刷新
        _tol = 10 if "local" in _f else 3
        rep(_lag3 <= _tol, f"{_f} 够新", f"最后 {_last3 or '-'}，滞后 {_lag3} 天"
            + ("" if _lag3 <= _tol else f"（容忍 {_tol} 天）"))
        # **天数也要查。** 2026-08-19 踩过: 从只回补了 20/870 天的归档重建，
        # 把 mos_local6.csv 从 845 天覆盖成 20 天 —— 站覆盖满、最后日期也是今天，
        # 上面两条全绿，这个文件却已经废了。训练样本掉一个数量级不会报错。
        rep(len(_days3) >= 300, f"{_f} 天数够训练用",
            f"{len(_days3)} 天"
            + ("" if len(_days3) >= 300 else " -> 少于 300 天，八成是从半截归档重建的"))

    # 训练用的本地 GFS 时效 vs 生产 pick_run 会选的时效。
    # 2026-08-18 加: 那天照一段陈旧脚本用 mos_local12 重训了 12 时，而生产喂
    # lead 6h（两份文件 29.4% 的站日差 >=0.5 度）。原来没有任何机器检查能发现 ——
    # 逐列比对两端读同一个 CSV，是空对空。
    _spec2 = json.load(open(args.nowcast_model, encoding="utf-8"))
    _want_lead = {9: 12, 10: 12, 11: 12, 12: 6, 13: 6, 14: 6}
    for _c, _wl in sorted(_want_lead.items()):
        _rec = _spec2.get(str(_c), {}).get("local_gfs_lead")
        if _rec is None:
            rep(True, f"{_c} 时训练用的本地 GFS 时效",
                "未记录（该模型训练时还没这个字段）—— 下次重训自动落盘，暂无法校验")
        else:
            rep(_rec == _wl, f"{_c} 时训练时效与生产 pick_run 一致",
                f"训练 {_rec}h / 生产要 {_wl}h"
                + ("" if _rec == _wl else "  -> 重训时 --nwp-csv2 用错了 mos_local*.csv"))

    # 9 时的自身近期偏差订正要靠 nowcast_hist.json 存历史预报值。文件丢了、
    # 或天数不够 K//2，订正会静默跳过 —— 不报错，只是那 +0.95pt 悄悄没了。
    # 这一项就是防这个。
    _hp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "nowcast_hist.json")
    if os.path.exists(_hp):
        try:
            _h = json.load(open(_hp, encoding="utf-8"))
        except Exception:                              # noqa: BLE001
            _h = {}
        import stations as _ST3
        _ds = sorted(_h)
        rep(len(_ds) >= 5, "nowcast_hist.json 天数够 9 时偏差订正用",
            f"{len(_ds)} 天（<5 天则订正跳过）")
        _age = ((datetime.now().date() - datetime.strptime(_ds[-1], "%Y-%m-%d").date()).days
                if _ds else 999)
        rep(_age <= 2, "nowcast_hist.json 是新的",
            f"最后一天 {_ds[-1] if _ds else '-'}，距今 {_age} 天")
        _nst = len(_h[_ds[-1]]) if _ds else 0
        rep(_nst == len(_ST3.ICAOS), "最近一天的历史含全部站",
            f"{_nst}/{len(_ST3.ICAOS)}")
    else:
        rep(False, "nowcast_hist.json 在（9 时近期偏差订正的数据源）",
            "缺文件 -> 订正静默跳过")
    # 高优势清单要两样: edge_table.json + 生产每天生成的 pred_mos.csv。
    # pred_mos.csv 若陈旧（不含今天），清单会静默变空 —— 那正是最该有提示的时候。
    _et = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edge_table.json")
    rep(os.path.exists(_et), "edge_table.json 在（高优势清单的数据源）")
    _pm = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pred_mos.csv")
    fresh = False
    if os.path.exists(_pm):
        try:
            import csv as _csv
            ds = {r["date"] for r in _csv.DictReader(open(_pm, encoding="utf-8"))
                  if r.get("lead") == "1"}
            fresh = bool(ds) and max(ds) >= (datetime.now() - timedelta(days=1)).date().isoformat()
        except Exception:                              # noqa: BLE001
            fresh = False
    rep(fresh, "pred_mos.csv 是新的（高优势清单要用它当隔夜对照）",
      "" if fresh else "缺失或陈旧 —— run_daily.sh 没跑成，清单会静默变空")

    # 「一致?」列的对照模型必须在，否则那一列静默变空。同时它必须**不含**
    # AIFS —— 两个模型一样的话一致性就恒为「一致」，这一列就废了。
    _nf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "nowcast_nwp_noaifs.json")
    ok_nf = os.path.exists(_nf)
    if ok_nf:
        try:
            _n9 = json.load(open(_nf, encoding="utf-8")).get("9", {})
            ok_nf = sum(1 for x in _n9.get("names", [])
                        if x.endswith("_minus_m1")) == 6
        except Exception:                              # noqa: BLE001
            ok_nf = False
    rep(ok_nf, "「一致?」的对照模型在，且确实不含 AIFS（6 个追加模式）",
      "" if ok_nf else "缺 nowcast_nwp_noaifs.json 或它也含 AIFS，那一列会失效")

    # 「已见顶」判别器必须覆盖 10-14 时。缺了预测端会静默跳过覆盖逻辑 ——
    # 生产回测实测 12 时 +1.33pt、13 时 +1.89pt，丢了不会报错只会慢慢变差。
    import train_nowcast as _TN
    _stp = args.nowcast_model + ".settled.pkl"
    have = set()
    if os.path.exists(_stp):
        try:
            import pickle as _pk
            have = set(_pk.load(open(_stp, "rb")))
        except Exception:                              # noqa: BLE001
            have = set()
    miss_st = sorted(_TN.SETTLED_CUTOFFS - have)
    rep(not miss_st, "「已见顶」判别器覆盖 10-14 时",
      "" if not miss_st else f"缺时次 {miss_st}（{_stp}）")

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
    # 默认必须按时次给对文件。2026-08-18 踩过: 默认 None -> 逐列比对时 m2~m7
    # 两端都是 None，**空对空地「一致」**，于是漏掉了「12 时模型用 mos_local12
    # 训、生产却喂 lead 6h」这个错配（两份文件 29.4% 的站日差 >=0.5 度）。
    # 本地 GFS 的时效按时次分: 9-11 时 12h（mos_local12）、12-14 时 6h（mos_local6），
    # 与下面 want={9:12,10:12,11:12,12:6,13:6,14:6} 那张表是同一件事的两端。
    ap.add_argument("--nwp-csv2", nargs="+", default=None,
                    help="不给就按 --cutoff 自动选（9-11 用 mos_local12，12-14 用 mos_local6）")
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
        _lg = "mos_local12.csv" if int(args.cutoff) <= 11 else "mos_local6.csv"
        args.nwp_csv2 = ["mos_ecmwf.csv", "mos_cma.csv", "mos_icon.csv",
                         "mos_jma_.csv", "mos_gem_.csv", _lg]
        if int(args.cutoff) == 9:
            args.nwp_csv2.append("mos_aifs.csv")      # 9 时的第八成员
        print(f"  （--nwp-csv2 未给，按 {args.cutoff} 时自动选: "
              f"{' '.join(args.nwp_csv2)}）", file=sys.stderr)
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

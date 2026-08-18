#!/usr/bin/env bash
# retrain.sh — 全时次重训。**各时次的配置写死在这里，这是唯一真相源。**
#
# 为什么要有这个脚本: README 里那段重训命令在 2026-08 三次害人，每次都是
# 静默失效、不报错、只是精度悄悄退回:
#   1. 漏 mos_aifs.csv        -> 9 时从 130 项掉回 120，AIFS 没了
#   2. 12/13 时用 mos_local12 -> 训练用 12h 时效、生产喂 6h（两份文件 29.4%
#                               的站日差 >=0.5 度）
#   3. 12 时漏 PLOYGON_DQ=1   -> 把刚上线的去量化丢掉
# 散在文档里的命令一定会漂，写成脚本才不会。
#
#   ./retrain.sh            # 全部
#   ./retrain.sh 9 12       # 只重训这几个时次（其余保持线上原样）
#
# 配置来源（都是机器验证过的，不是抄文档）:
#   flag 组合 —— 逐组合构造 FEATS 与线上模型 names 精确比对，六个时次全中
#   本地 GFS 时效 —— gfs_live.pick_run() 生产端实际选的:
#       9/10/11 时 18Z@12h -> mos_local12.csv
#       12/13/14 时 00Z@6h -> mos_local6.csv
#   追加模式顺序 —— 必须与 run_hourly.sh 的 MODELS 一致，否则 m2_/m3_ 对错模式
set -euo pipefail
cd "$(dirname "$0")"

M5="mos_ecmwf.csv mos_cma.csv mos_icon.csv mos_jma_.csv mos_gem_.csv"
F5="PLOYGON_CONV=1 PLOYGON_PROF=1 PLOYGON_CURVE=1 PLOYGON_WINDAM=1 PLOYGON_HPBL=1"
SY=2025
WANT="${*:-9 10 11 12 13 14 15}"

has() { echo " $WANT " | grep -q " $1 "; }
say() { printf '\n\033[1m>>> %s\033[0m\n' "$*"; }

# 追加模式的 csv 必须在，且顺序固定
for f in $M5 mos.csv mos_local12.csv mos_local6.csv mos_aifs.csv; do
    [[ -f $f ]] || { echo "[error] 缺 $f，先跑 run_daily.sh / gfs_local_build.py"; exit 1; }
done

if has 9; then
    say "9 时: 五要素(无 REGIME) + 7 个追加模式(含 AIFS 当 m8) + local12"
    env $F5 python3 train_nowcast.py --db cn.sqlite --cutoffs 9 --nwp-csv mos.csv \
        --nwp-csv2 $M5 mos_local12.csv mos_aifs.csv --split-year $SY --dump nc_a.json
fi
if has 9; then
    say "9 时对照模型 nowcast_nwp_noaifs.json（**同批重训，只去掉 AIFS**）"
    # 「一致?」列拿它跟主模型比。**必须与主模型同批训** —— 版本不同的话，
    # 那一列比的就不再是「有没有 AIFS」，而混进了训练数据版本的差异。
    env $F5 python3 train_nowcast.py --db cn.sqlite --cutoffs 9 --nwp-csv mos.csv \
        --nwp-csv2 $M5 mos_local12.csv --split-year $SY --dump nowcast_nwp_noaifs.json
fi
if has 10 || has 11; then
    say "10/11 时: 五要素 + REGIME + local12"
    env $F5 PLOYGON_REGIME=1 python3 train_nowcast.py --db cn.sqlite --cutoffs 10 11 \
        --nwp-csv mos.csv --nwp-csv2 $M5 mos_local12.csv --split-year $SY --dump nc_b.json
fi
if has 12; then
    say "12 时: 纯线性 + 去量化(PLOYGON_DQ) + **local6**"
    PLOYGON_DQ=1 python3 train_nowcast.py --db cn.sqlite --cutoffs 12 --nwp-csv mos.csv \
        --nwp-csv2 $M5 mos_local6.csv --split-year $SY --dump nc_c.json
fi
if has 13; then
    say "13 时: 纯线性(无 flag) + **local6**"
    python3 train_nowcast.py --db cn.sqlite --cutoffs 13 --nwp-csv mos.csv \
        --nwp-csv2 $M5 mos_local6.csv --split-year $SY --dump nc_d.json
fi
if has 14; then
    say "14 时: 五要素 + REGIME + local6"
    env $F5 PLOYGON_REGIME=1 python3 train_nowcast.py --db cn.sqlite --cutoffs 14 \
        --nwp-csv mos.csv --nwp-csv2 $M5 mos_local6.csv --split-year $SY --dump nc_e.json
fi
if has 15; then
    say "15 时: 纯实况，无 NWP（换成六模式整体变差，P=19%）"
    python3 train_nowcast.py --db cn.sqlite --cutoffs 15 --split-year $SY \
        --dump nowcast_late.json
fi

say "合并 JSON / GBM / 已见顶判别器（只覆盖本次重训的时次）"
python3 - "$WANT" <<'PY'
import json, pickle, os, sys
want = {int(x) for x in sys.argv[1].split()}
base = json.load(open("nowcast_nwp.json", encoding="utf-8"))
for f in ("nc_a.json", "nc_b.json", "nc_c.json", "nc_d.json", "nc_e.json"):
    if os.path.exists(f):
        for k, v in json.load(open(f, encoding="utf-8")).items():
            if int(k) in want:
                base[k] = v
json.dump(base, open("nowcast_nwp.json", "w"), ensure_ascii=False, indent=1)
for suf in (".gbm.pkl", ".settled.pkl"):
    cur = {}
    if os.path.exists("nowcast_nwp.json" + suf):
        cur = pickle.load(open("nowcast_nwp.json" + suf, "rb"))
    for f in ("nc_a.json", "nc_b.json", "nc_c.json", "nc_d.json", "nc_e.json"):
        if os.path.exists(f + suf):
            for k, v in pickle.load(open(f + suf, "rb")).items():
                if int(k) in want:
                    cur[k] = v
    if cur:
        pickle.dump(cur, open("nowcast_nwp.json" + suf, "wb"))
print("  各时次特征数:", {k: len(v["names"]) for k, v in sorted(base.items(), key=lambda x: int(x[0]))})
print("  local_gfs_lead:", {k: v.get("local_gfs_lead") for k, v in sorted(base.items(), key=lambda x: int(x[0]))})
PY
rm -f nc_?.json nc_?.json.gbm.pkl nc_?.json.settled.pkl nc_?.json.peak.pkl

say "补「见顶时刻」判别器（用线上模型的 names/median，主模型不动）"
env $F5 python3 train_nowcast.py --db cn.sqlite --cutoffs 9 --nwp-csv mos.csv \
    --nwp-csv2 $M5 mos_local12.csv mos_aifs.csv --peak-only nowcast_nwp.json
cp nowcast_nwp.json.peak.pkl /tmp/_pk9.pkl
env $F5 PLOYGON_REGIME=1 python3 train_nowcast.py --db cn.sqlite --cutoffs 10 11 \
    --nwp-csv mos.csv --nwp-csv2 $M5 mos_local12.csv --peak-only nowcast_nwp.json
cp nowcast_nwp.json.peak.pkl /tmp/_pk1011.pkl
PLOYGON_DQ=1 python3 train_nowcast.py --db cn.sqlite --cutoffs 12 --nwp-csv mos.csv \
    --nwp-csv2 $M5 mos_local6.csv --peak-only nowcast_nwp.json
cp nowcast_nwp.json.peak.pkl /tmp/_pk12.pkl
python3 - <<'PY'
import pickle
m = {}
for f in ("/tmp/_pk9.pkl", "/tmp/_pk1011.pkl", "/tmp/_pk12.pkl"):
    m.update(pickle.load(open(f, "rb")))
pickle.dump(m, open("nowcast_nwp.json.peak.pkl", "wb"))
print("  见顶时刻判别器时次:", sorted(m))
PY

say "完成。**接下来必须做**: 重建 hit_table（见 rebuild_tables.sh），再跑 check_consistency"

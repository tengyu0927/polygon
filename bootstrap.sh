#!/usr/bin/env bash
# bootstrap.sh — 新机器拉下仓库后，一条命令把数据建起来
#
#   ./bootstrap.sh              # 完整重建（实况 1995 年起 + 6 个模式 2024-07 起）
#   ./bootstrap.sh --quick      # 只拉最近 2 年实况，够跑预报、不够重训长序列模型
#   ./bootstrap.sh --obs-only   # 只建实况库（12/13 时之后的纯实况模型就能用）
#
# 为什么仓库里没有数据: 实况库 510MB + 6 个模式库约 1.2GB，
# 远超 GitHub 单文件 100MB 的硬上限。全部可重建，代价是时间不是信息。
#
# 耗时参考（取决于上游快慢）:
#   实况 1995 年起          30-60 分钟
#   实况最近 2 年（--quick） 3-5 分钟
#   6 个模式 2024-07 起      50-70 分钟
# 上游偶发变慢是常态，脚本会重试；中断后重跑会跳过已完成的部分。

set -euo pipefail
cd "$(dirname "$0")"

STATIONS=ZSPD,ZUUU,ZGSZ,ZGGG,ZUCK,ZBAA,ZHHH,ZSQD
# 顺序必须与 merge_mos.py --extra、run_*.sh 的 MODELS 一致
MODELS=(
  "mos_fcst  gfs_global                      mos.csv"
  "mos_ecmwf ecmwf_ifs025                    mos_ecmwf.csv"
  "mos_cma   cma_grapes_global               mos_cma.csv"
  "mos_icon  icon_global                     mos_icon.csv"
  "mos_jma_  jma_gsm                         mos_jma_.csv"
  "mos_gem_  gem_global                      mos_gem_.csv"
)
FCST_START=2024-07-01

QUICK=""; OBS_ONLY=""
for a in "$@"; do
  [ "$a" = "--quick" ] && QUICK=1
  [ "$a" = "--obs-only" ] && OBS_ONLY=1
done

echo "===== 1/3 实况库 cn.sqlite ====="
if [ -n "$QUICK" ]; then
    THIS_Y=$(TZ=Asia/Shanghai date +%Y)
    python3 iem_multi.py --db cn.sqlite --stations "$STATIONS" \
        --backfill --min-year $((THIS_Y - 1)) --max-year "$THIS_Y"
else
    python3 iem_multi.py --db cn.sqlite --stations "$STATIONS" \
        --backfill --min-year 1995
fi

# ZGSZ 换成 WU 的序列。**这一步不能省** —— 最终对错以 WU 页面的日最高温为准，
# 而 WU 的 ZGSZ 页面挂的是香港流浮山（WMO 45035，离深圳宝安 30km）。
# 不换的话，新机器上深圳会用宝安 METAR 训练、被流浮山打分，
# 实测 624 天里 70% 的日子对不上、26% 差 >=2℃。详见 wu_obs.py
echo "--- ZGSZ 换用 WU 实况（香港流浮山）---"
python3 wu_obs.py --db cn.sqlite --migrate
python3 iem_multi.py --db cn.sqlite --daily
python3 wu_check.py --db cn.sqlite --days 30 || true

if [ -n "$OBS_ONLY" ]; then
    echo "只建了实况库。12-15 时的纯实况模型已经能用:"
    echo "  ./run_hourly.sh 13"
    exit 0
fi

echo
echo "===== 2/3 六个模式的固定时效因子 ====="
END=$(TZ=Asia/Shanghai date +%Y-%m-%d)
for row in "${MODELS[@]}"; do
    read -r db mdl out <<< "$row"
    echo "--- $mdl ---"
    python3 build_mos_dataset.py fetch --db "$db.sqlite" --model "$mdl" \
        --start "$FCST_START" --end "$END"
    python3 build_mos_dataset.py build --db "$db.sqlite" --obs-db cn.sqlite \
        --daily-table daily --out "$out"
done

echo
echo "===== 3/3 拼多模式训练集 ====="
python3 merge_mos.py --base mos.csv \
    --extra mos_ecmwf.csv mos_cma.csv mos_icon.csv mos_jma_.csv mos_gem_.csv \
    --deb --out mos_multi.csv

cat <<'TIP'

===== 完成 =====
仓库里自带的模型（model.json / nowcast_nwp.json / nowcast.json /
nowcast_late.json）现在就能用:

  ./run_hourly.sh          # 临近预报，9-15 时
  ./run_daily.sh           # D+1/D+2
  python3 check_consistency.py    # 先跑这个确认环境没问题

想用自己的数据重训（模型会随时间漂移，建议每月一次）:

  python3 train_mos.py mos_multi.csv --obs-pen 1.0 --dump model.json --pred pred.csv
  python3 train_nowcast.py --db cn.sqlite --cutoffs 9 10 11 12 13 \
      --nwp-csv mos.csv --nwp-csv2 mos_ecmwf.csv mos_cma.csv mos_icon.csv \
      mos_jma_.csv mos_gem_.csv --dump nowcast_nwp.json
  python3 train_nowcast.py --db cn.sqlite --cutoffs 14 15 --dump nowcast_late.json
TIP

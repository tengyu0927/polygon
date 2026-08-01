#!/usr/bin/env bash
# run_hourly.sh — 按当前整点跑一次临近预报，结果追加到当天的日志
#
#   ./run_hourly.sh          # 用当前小时
#   ./run_hourly.sh 11       # 指定小时（补跑）
#   ./run_hourly.sh 11 --no-fetch    # 跳过取数，只重跑模型
#
# 9-13 时用六模式模型（要联网取 6 份 GFS/ECMWF/... 预报，约 1 分钟）
# 14/15 时用纯实况模型（不联网，快）—— 那时大部分站已见顶，模式信息没有增量。

set -euo pipefail
cd "$(dirname "$0")"

export PATH="$PATH:/usr/sbin"

# 单实例锁。IEM 偶尔卡住，没有锁的话每小时叠一个僵尸进程、互相抢 cn.sqlite
# 的写锁，越堆越死（实测 14:15 那轮挂了 21 分钟没退）。
LOCK=/tmp/ploygon_run_hourly.lock
if ! mkdir "$LOCK" 2>/dev/null; then
    OLD=$(cat "$LOCK/pid" 2>/dev/null || echo "")
    if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
        echo "[skip] 上一轮 (pid $OLD) 还在跑，本轮跳过" >&2
        exit 0
    fi
    echo "[warn] 发现残留锁（进程已不在），清理后继续" >&2
    rm -rf "$LOCK"; mkdir "$LOCK" 2>/dev/null || exit 0
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT



STATIONS=ZSPD,ZUUU,ZGSZ,ZGGG,ZUCK,ZBAA,ZHHH,ZSQD
# UKMO 已剔除: 归档只到 2025-01，训练期大半缺测，实测拖累 D+2（见 README）
MODELS=ecmwf_ifs025,cma_grapes_global,icon_global,jma_gsm,gem_global
# 追加模式的顺序必须与训练时 --nwp-csv2 的顺序一致，否则 m2_/m3_ 各列对错模式

# 一律按北京时取小时。用系统本地时区的话，机器时区一变自动选档就错
HOUR=${1:-$(TZ=Asia/Shanghai date +%-H)}
[[ "${1:-}" == --* ]] && HOUR=$(TZ=Asia/Shanghai date +%-H)
NOFETCH=""
USE_LIVE=""
for a in "$@"; do [[ "$a" == "--no-fetch" ]] && NOFETCH=1; done

TODAY=$(TZ=Asia/Shanghai date +%Y-%m-%d)
LOG=${PLOYGON_LOG:-pred_${TODAY}.log}   # 一致性检查会覆盖它，避免污染真实日志

if (( HOUR < 9 || HOUR > 15 )); then
    echo "[error] 只支持 9-15 时，收到 $HOUR" >&2
    exit 1
fi

# 9-13 时都用六模式模型。12/13 时原来用纯实况长序列，实测被六模式显著打败
# （12 时 MAE 0.716 -> 0.618，13 时 0.519 -> 0.455），已切换
if (( HOUR <= 13 )); then
    MODEL=nowcast_nwp.json
    EXTRA=(--extra-models "$MODELS")
else
    MODEL=nowcast_late.json
    EXTRA=()
fi

if [[ -z "$NOFETCH" ]]; then
    echo "更新实况…" >&2
    # 增量抓最近 2 天，秒级。别用 --update —— 那是重抓两整年，
    # IEM 慢起来能挂几十分钟，而 urlopen 的 timeout 只管单次 socket 读
    if python3 iem_multi.py --db cn.sqlite --stations "$STATIONS" \
            --recent-days 2 --timeout 120 >/dev/null 2>&1; then
        # ZGSZ 的实况以 WU 为准（WU 的 ZGSZ 页面挂的是香港流浮山，
        # 打分也按它，所以模型训练在这条序列上）。IEM 抓回来的是深圳宝安
        # METAR，必须覆盖掉，否则库里混进另一个站的观测。见 wu_obs.py
        python3 wu_obs.py --db cn.sqlite --update --days 2 >/dev/null 2>&1 \
            || echo "[warn] WU 实况更新失败，ZGSZ 可能用到陈旧数据" >&2
        python3 iem_multi.py --db cn.sqlite --daily >/dev/null 2>&1 || true
    else
        echo "[warn] IEM 实况更新失败，本轮改用 AWC 实时源（--live）" >&2
        USE_LIVE=--live
    fi

    # 前瞻记录「这个时刻实际能取到的最新一轮模式」。不参与预报，
    # 只为判定 previous_day0 能不能拿来训练（见 run0_probe.py 的说明）。
    # 失败不影响本轮预报。
    python3 run0_probe.py --db run0_probe.sqlite --cutoff "$HOUR" --log >/dev/null 2>&1 || true

    # 这里**不**再重建 mos_*.csv。predict_nowcast.py 的模式特征是自己联网取的
    # （fetch_nwp / fetch_m2），压根不读 csv —— 每小时重建一遍纯属浪费:
    # 多花约 1 分钟、API 请求翻倍、对预报没有任何影响。
    # csv 只是训练集，按月重训时再更新（见 README「重训周期」）。
fi

{
    echo
    echo "########## $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')  ${HOUR} 时起报  模型 $MODEL ##########"
    python3 predict_nowcast.py --model "$MODEL" --cutoff "$HOUR" \
        --hurdle --p90 --verbose $USE_LIVE ${EXTRA[@]+"${EXTRA[@]}"}
} 2>&1 | tee -a "$LOG"

echo "已追加到 $LOG" >&2

#!/usr/bin/env bash
# run_daily.sh — 每天跑一次 D+1/D+2 MOS 预报（六模式 + DEB + 岭回归/GBM 融合）
#
#   ./run_daily.sh              # 预报今天 + 明天
#   ./run_daily.sh 2            # 预报今天 + 未来 2 天
#   ./run_daily.sh 1 --no-update    # 跳过实况更新
#
# 建议跑点: 北京时 23:00。理由见 README「什么时候跑」一节 ——
# 核心是 recent_bias（重要性排第二的特征）需要今天的日最高温已经定型，
# 而峰值最晚可以出现在 17 时（2026-07-25 成都就是 17:00 才见顶）。
# 17 点跑会把一个还没走完的今天当成已知事实，偏差直接进特征。

set -euo pipefail
cd "$(dirname "$0")"

# cron 的 PATH 里没有 /usr/sbin，joblib 找不到 sysctl 会往日志里吐一大段
# traceback（无害但淹没真正的错误）。补上并显式告诉它核数。
export PATH="$PATH:/usr/sbin"
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-$(sysctl -n hw.physicalcpu 2>/dev/null || echo 4)}"

# 单实例锁。IEM 偶尔卡住，没有锁的话每小时叠一个僵尸进程、互相抢 cn.sqlite
# 的写锁，越堆越死（实测 14:15 那轮挂了 21 分钟没退）。
LOCK=/tmp/ploygon_run_daily.lock
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
# UKMO 已剔除: 归档只到 2025-01，训练期大半缺测，实测显著拖累 D+2
MODELS=ecmwf_ifs025,cma_grapes_global,icon_global,jma_gsm,gem_global
# 顺序必须与 merge_mos.py --extra 的顺序一致，否则 m2_/m3_ 各列对错模式

AHEAD=${1:-1}
[[ "$AHEAD" == --* ]] && AHEAD=1
NOUPDATE=""
for a in "$@"; do [[ "$a" == "--no-update" ]] && NOUPDATE=1; done

TODAY=$(TZ=Asia/Shanghai date +%Y-%m-%d)
HOUR=$(TZ=Asia/Shanghai date +%-H)
LOG=${PLOYGON_LOG:-pred_mos_${TODAY}.log}   # 一致性检查会覆盖它，避免污染真实日志

if [[ -z "$NOUPDATE" ]]; then
    echo "[1/2] 更新实况（recent_bias 依赖它）…" >&2
    # 每天一次，抓最近 5 天（比每小时那条宽一点，兜住偶发缺报）
    python3 iem_multi.py --db cn.sqlite --stations "$STATIONS" \
        --recent-days 5 --timeout 180 >/dev/null 2>&1 \
        || echo "[warn] 实况增量更新失败，recent_bias 可能滞后" >&2
    # ZGSZ 覆盖成 WU（香港流浮山）—— 必须在 iem_multi 之后、建日表之前
    python3 wu_obs.py --db cn.sqlite --update --days 5 >/dev/null 2>&1 \
        || echo "[warn] WU 实况更新失败，ZGSZ 可能用到陈旧数据" >&2
    python3 iem_multi.py --db cn.sqlite --daily >/dev/null 2>&1 || true

    # 核对实测口径。最终对错以 WU 为准，训练/检验的实况必须和它一致 ——
    # WU 随时可能改站点映射（2026-08-01 就发现 ZGSZ 挂的是香港流浮山）。
    # 只打警告，不中断当天的预报。
    _wu=$(python3 wu_check.py --db cn.sqlite --days 7 2>/dev/null | tail -30)
    if echo "$_wu" | grep -q "不一致的站"; then
        echo "[WARN] 实测与 WU 对不上:" >&2
        echo "$_wu" | grep -A2 "不一致的站" >&2
    fi
fi

# 跑得太早会用到没走完的今天。18 时之前给出明确警告而不是静默出数
if (( HOUR < 18 )); then
    echo "[warn] 现在 ${HOUR} 时，今天的日最高温可能还没出现（峰值最晚见于 17 时），" >&2
    echo "       recent_bias 会偏低，预报会系统性偏冷。建议 23 时再跑。" >&2
fi

# 实况完整性检查: 各站今日观测条数，太少说明 IEM 归档还没跟上
python3 - <<PY >&2
import sqlite3
c = sqlite3.connect("cn.sqlite")
rows = list(c.execute(
    "SELECT station, COUNT(*), MAX(temp_c) FROM obs WHERE local_date=? "
    "GROUP BY station", ("$TODAY",)))
thin = [r[0] for r in rows if r[1] < 20]
if len(rows) < 8:
    print(f"[warn] 今日只有 {len(rows)}/8 个站有观测，recent_bias 会退化")
if thin:
    print(f"[warn] 观测偏少的站: {', '.join(thin)}（IEM 归档滞后）")
PY

# 刷新训练集。**这一步不能省** —— 2026-07-31 查出 mos.csv 停在 7-26 整整五天，
# 期间天气型剧变（北京 +4.4℃、成都 -8.4℃），模型权重跟不上，
# 实测重训后 13 时 MAE 0.62 -> 0.50。每天几十秒，换训练数据永远是最新的。
# 注意 predict_* 的模式特征是实时 API 取的，不读这些 csv；csv 只供训练。
if [[ -z "$NOUPDATE" ]]; then
    echo "刷新训练集（供重训用，预报本身不读 csv）…" >&2
    for row in "mos_fcst:gfs_global:mos.csv" "mos_ecmwf:ecmwf_ifs025:mos_ecmwf.csv" \
               "mos_cma:cma_grapes_global:mos_cma.csv" "mos_icon:icon_global:mos_icon.csv" \
               "mos_jma_:jma_gsm:mos_jma_.csv" "mos_gem_:gem_global:mos_gem_.csv"; do
        db=${row%%:*}; rest=${row#*:}; mdl=${rest%%:*}; out=${rest#*:}
        python3 build_mos_dataset.py fetch --db "$db.sqlite" --model "$mdl" \
            --start "$(TZ=Asia/Shanghai date -v-7d +%Y-%m-%d 2>/dev/null \
                      || date -d '7 days ago' +%Y-%m-%d)" \
            --end "$TODAY" >/dev/null 2>&1 || echo "  [warn] $mdl 训练数据刷新失败" >&2
        python3 build_mos_dataset.py build --db "$db.sqlite" --obs-db cn.sqlite \
            --daily-table daily --out "$out" >/dev/null 2>&1 || true
    done
fi

# 模型过期检查。训练数据每天更新了，但权重要人工重训 ——
# 超过 7 天不重训就大声提醒，别再让它默默漂移
python3 - <<'PYCHK' >&2
import os, time, csv, collections
try:
    age = (time.time() - os.path.getmtime("nowcast_nwp.json")) / 86400
    d = collections.Counter(r["date"] for r in csv.DictReader(open("mos_multi.csv"))
                            if r.get("lead") == "1" and r.get("y_tmax"))
    print(f"[info] 模型已训练 {age:.1f} 天，训练集最新 {max(d) if d else '?'}")
    if age > 7:
        print(f"[WARN] 模型 {age:.0f} 天没重训了。天气型漂移会让精度慢慢下降，"
              f"建议跑:\n"
              f"       python3 merge_mos.py --base mos.csv --extra mos_ecmwf.csv "
              f"mos_cma.csv mos_icon.csv mos_jma_.csv mos_gem_.csv --deb --out mos_multi.csv\n"
              f"       python3 train_mos.py mos_multi.csv --obs-pen 1.0 --dump model.json\n"
              f"       python3 train_nowcast.py --db cn.sqlite --cutoffs 9 10 11 12 13 "
              f"--nwp-csv mos.csv --nwp-csv2 mos_ecmwf.csv mos_cma.csv mos_icon.csv "
              f"mos_jma_.csv mos_gem_.csv --dump nowcast_nwp.json")
except Exception as e:
    print(f"[warn] 模型过期检查失败: {e}")
PYCHK

{
    echo
    echo "########## $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')  D+1/D+2 六模式 MOS ##########"
    # 显式传 --date。predict_mos.py 内部默认取「现在」，如果本脚本在 23:5x 起跑、
    # 取数耗时跨过午夜，目标日会整体错一天。传死日期就与起跑时刻无关了
    python3 predict_mos.py --date "$TODAY" --ahead "$AHEAD" \
        --extra-models "$MODELS" --csv-out pred_mos.csv --verbose

    # TAF 对照: 每天记一条并排样本。AWC 只存 15 天 TAF，历史补不回来，
    # 所以「分歧日谁对」这个问题只能靠每天攒前瞻样本回答
    echo
    python3 taf_compare.py --pred pred_mos.csv \
        --db "${PLOYGON_TAF_DB:-taf_compare.sqlite}" || true
    # 回填往日的实况。不做这一步，pair 表的 obs 永远是 NULL，
    # 两周后想判断「TAF 值不值得当特征」时手上没有可用数据。
    python3 taf_compare.py --analyze --obs-db cn.sqlite \
        --db "${PLOYGON_TAF_DB:-taf_compare.sqlite}" >/dev/null 2>&1 || true

    # 把当天的「预报 vs 实况」累计入库。23:59 跑，当天实况已定型。
    # 目的是找站级系统性偏差 —— 通用判读规则在小样本上表现参差，
    # 但站级偏差稳定得多（重庆连续偏低、青岛连续偏高）
    echo
    python3 verify.py --db "${PLOYGON_VERIFY_DB:-verify.sqlite}" || true
} 2>&1 | tee -a "$LOG"

echo "已追加到 $LOG" >&2

#!/usr/bin/env bash
# run_hourly.sh — 按当前整点跑一次临近预报，结果追加到当天的日志
#
#   ./run_hourly.sh          # 用当前小时
#   ./run_hourly.sh 11       # 指定小时（补跑）
#   ./run_hourly.sh 11 --no-fetch    # 跳过取数，只重跑模型
#   ./run_hourly.sh --stations ZGSZ,ZSJN --no-side   # 只跑这两站（分批用）
#
# 9-13 时用六模式模型（要联网取 6 份 GFS/ECMWF/... 预报，约 1 分钟）
# 14/15 时用纯实况模型（不联网，快）—— 那时大部分站已见顶，模式信息没有增量。
# 16/17 时**不跑模型**，直接报已达（late_call.py），见那里的实测对比。

set -euo pipefail
cd "$(dirname "$0")"

export PATH="$PATH:/usr/sbin"

# 参数解析必须在取锁之前 —— 锁名要按批次区分（见下）。
# 分批跑: WU 那两个站（深圳/济南）的整点观测比 METAR 晚落地，让它们单独
# 晚一档跑，其余 8 站不用陪着等。清单一律从 stations.py 取，别在这写死。
#   :15  ./run_hourly.sh --stations "$(non_wu)"
#   :35  ./run_hourly.sh --stations "$(wu)" --no-side
# --no-side: 跳过 run0_probe / ens_collect 这两个与站点无关的旁路任务，
#            它们一小时只该跑一次，否则 run0_probe 的逐档记录会重复。
NOFETCH=""
USE_LIVE=""
ONLY=""
NOSIDE=""
_prev=""
for a in "$@"; do
    [[ "$a" == "--no-fetch" ]] && NOFETCH=1
    [[ "$a" == "--no-side" ]] && NOSIDE=1
    [[ "$_prev" == "--stations" ]] && ONLY="$a"
    _prev="$a"
done

# 单实例锁。IEM 偶尔卡住，没有锁的话每小时叠一个僵尸进程、互相抢 cn.sqlite
# 的写锁，越堆越死（实测 14:15 那轮挂了 21 分钟没退）。
# 锁按**批次**分。两批的站点不重叠，共用一把锁的话，:15 那批慢一次就会让
# :35 那批整个被跳过 —— 2026-08-11 实际发生: :15 批连续 5 小时占锁超过 20
# 分钟（外部服务间歇性慢），深圳/济南七档只拿到 9 时和 15 时两档，
# cron_hourly.log 里 5 条 [skip] 就是这么来的。
# 锁的本意是「同一批次的上一轮没退完，本轮别叠上去」，所以按批次区分才对。
LOCK=/tmp/ploygon_run_hourly$(printf '%s' "${ONLY:-all}" | tr -c 'A-Za-z0-9' '_').lock
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



# 站点清单与 stations.py 保持一致（唯一真相源）。改站点先改那里
STATIONS=ZBAA,ZGGG,ZGSZ,ZHCC,ZHHH,ZSJN,ZSPD,ZSQD,ZUCK,ZUUU
# UKMO 已剔除: 归档只到 2025-01，训练期大半缺测，实测拖累 D+2（见 README）
# 第七个成员 local_gfs 不走 Open-Meteo,而是实时从 NCEP 取当时能拿到的
# 最新一轮 GFS(gfs_live.py)。顺序必须与训练时 --nwp-csv2 一致,放最后。
#
# 实测(15 个月滚动回测,各时次用其真实可用的时效):
#   11 时 0.7682 -> 0.7524  -0.0157  P=98%   第七成员用前一天 12Z(18h)
#   12 时 0.6532 -> 0.6307  -0.0225  P=100%  第七成员用当天 00Z(6h)
#   13 时 0.4807 -> 0.4685  -0.0122  P=98%
#    9/10 时 P=68%/63% 未过线,但方向为负、无害,一并保留以简化配置
MODELS=ecmwf_ifs025,cma_grapes_global,icon_global,jma_gsm,gem_global,local_gfs
# 追加模式的顺序必须与训练时 --nwp-csv2 的顺序一致，否则 m2_/m3_ 各列对错模式
#
# **9 时多一个第八成员 ecmwf_aifs025_single**（ECMWF 的 AI 模式 AIFS，
# 数据驱动而非物理积分，与 IFS 机理互补）。2026-08-12 实测:
#   原始 MAE D+1  AIFS 1.306 < ICON 1.502 < ECMWF 1.536 < GEM 1.661 < GFS 1.686
#   9 时完全命中  34.78% -> 37.41%（+2.63pt, P=100%），MAE -0.0389（P=100%）
# **只上 9 时**: 10 时 +0.69pt(P=80%)、11 时 +0.16pt(P=56%) 都不过线。
#
# 而且只用 2025-12 起那段过线 —— 全窗口 468 天是 +0.44pt/P=75%。2025 年中
# 那段 AIFS 反而有害（6/7/8 月 -5.9/-4.9/-5.7pt），**这个断裂没有解释**:
# 查过「早期 AIFS 版本弱」（反了，它对 ECMWF 的相对优势那时更大）和
# 「训练样本不够」（不单调）。所以配了影子对照持续验证，见 run_hourly 末尾。
MODELS9=$MODELS,ecmwf_aifs025_single

[[ -n "$ONLY" ]] && STATIONS="$ONLY"

# 一律按北京时取小时。用系统本地时区的话，机器时区一变自动选档就错
HOUR=${1:-$(TZ=Asia/Shanghai date +%-H)}
[[ "${1:-}" == --* ]] && HOUR=$(TZ=Asia/Shanghai date +%-H)
TODAY=$(TZ=Asia/Shanghai date +%Y-%m-%d)
LOG=${PLOYGON_LOG:-pred_${TODAY}.log}   # 一致性检查会覆盖它，避免污染真实日志

if (( HOUR < 9 || HOUR > 17 )); then
    echo "[error] 只支持 9-17 时，收到 $HOUR" >&2
    exit 1
fi

# 16/17 时不跑模型，直接报已达（late_call.py）。那时 96.6%/99.5% 的站日
# 当天最高温已经出现，训模型实测更差、且报已达就等于完美判别上界。
# 用户目标「十站至少九站对」正落在 16 时（按日 95.6%），15 时只有 67.2%。
LATE=""
(( HOUR >= 16 )) && LATE=1

# 9-13 时都用六模式模型。12/13 时原来用纯实况长序列，实测被六模式显著打败
# （12 时 MAE 0.716 -> 0.618，13 时 0.519 -> 0.455），已切换
#
# 2026-08-07: 14 时也切到六模式 + 非线性 + 天气型。原来 14/15 时用的
# nowcast_late.json 只有 28 项纯观测特征、纯线性、没有任何模式输入 ——
# 对「下午还会不会升」毫无信息。实测 14 时 0.3471 -> 0.3330（-0.0142,
# P=97%）、完全命中 +1.4pt。
#
# **15 时仍然不切。** 它整体反而变差（+0.0037, P=19%）: 晚见顶那 452 个
# 样本赚 0.11，但 <=15 时见顶的 3070 个样本每个亏 0.02，样本量 7 倍、一乘
# 就抵掉了。15 时现在 87.4% 完全命中是全系统最高的一档，不拿它去赌 12%。
if [[ -n "$LATE" ]]; then
    MODEL=""
    EXTRA=()
elif (( HOUR <= 14 )); then
    MODEL=nowcast_nwp.json
    if (( HOUR == 9 )); then
        EXTRA=(--extra-models "$MODELS9")
        # 「一致?」列: 用不含 AIFS 的旧模型在**同一批特征**上再算一遍。
        # 不改任何预报值，只标注两者是否给出同一个整数。实测两模型一致时
        # 9 时命中 37.9%、分歧时 31.1%（差 6.8pt），且在剩余升幅分档内部
        # 依然全正（+1.7~+15.6pt）—— 是独立于「预期命中」的第二个维度。
        [[ -f nowcast_nwp_noaifs.json ]] && EXTRA+=(--agree-with nowcast_nwp_noaifs.json)
    else EXTRA=(--extra-models "$MODELS"); fi
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

    # 这里**不**再重建 mos_*.csv。predict_nowcast.py 的模式特征是自己联网取的
    # （fetch_nwp / fetch_m2），压根不读 csv —— 每小时重建一遍纯属浪费:
    # 多花约 1 分钟、API 请求翻倍、对预报没有任何影响。
    # csv 只是训练集，按月重训时再更新（见 README「重训周期」）。
fi

{
    echo
    echo "########## $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')  ${HOUR} 时起报  模型 ${MODEL:-报已达} ##########"
    if [[ -n "$LATE" ]]; then
        python3 late_call.py --cutoff "$HOUR" ${ONLY:+--stations "$ONLY"}
    else
        python3 predict_nowcast.py --model "$MODEL" --cutoff "$HOUR" \
            --hurdle --p90 --verbose $USE_LIVE ${EXTRA[@]+"${EXTRA[@]}"} \
            ${ONLY:+--stations "$ONLY"}
    fi
} 2>&1 | tee -a "$LOG"

# 前瞻记录「这个时刻实际能取到的最新一轮模式」。**不参与预报。**
#
# 2026-08-22 从预报之前挪到预报之后。判定已经出了结果（历史接口一致率只有
# 45%，见 run0_probe.py 文档），所以采集从「1 个模式的 t2m」扩到「7 个模式
# 的 11 个变量」—— 那是**每轮 7 次 API 请求**，最坏情况超时叠加能拖十分钟。
# 它跑在预报前面就会把 :15 那一轮往后推，而它对本轮预报毫无用处。
# 放到后面，抢不到也只是今天少攒一轮数据。
if [[ -z "$NOSIDE" ]]; then
    python3 run0_probe.py --db run0_probe.sqlite --cutoff "$HOUR" --log >/dev/null 2>&1 || true
fi

# 集合预报采集。**只攒数据，不进任何模型** —— Open-Meteo 的集合 API 拿不到
# 历史（见 ens_collect.py 顶部），时效也只能靠记下抓取时刻自己保证，
# 所以要用就得从今天开始自己攒，约 3 个月（~720 站日）才够做 A/B。
# 一次请求带 8 个站，4 个模式共 4 次请求，几秒钟。失败绝不能影响预报。
if [[ -z "$NOSIDE" ]]; then
    python3 ens_collect.py --db "${PLOYGON_ENS_DB:-ens.sqlite}" --days 2 \
        2>&1 | tail -1 >&2 || echo "  [warn] 集合采集失败（不影响预报）" >&2
fi

# 影子对照: 9 时同时用**不含 AIFS**的旧模型跑一份，只落盘不打屏。
# AIFS 只在 2025-12 起那段过线、断裂原因不明，用生产数据继续验证 ——
# 按逐日命中率的离散度，约 60 天足以分辨 5 个百分点的真实差异。
# 失败绝不能影响正式预报，所以放在最后、全程 || true。
if (( HOUR == 9 )) && [[ -z "$NOSIDE" ]] && [[ -f nowcast_nwp_noaifs.json ]]; then
    {
        echo "########## $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S')  9 时影子(无AIFS) ##########"
        PLOYGON_STATE=/tmp/_shadow_state.json python3 predict_nowcast.py \
            --model nowcast_nwp_noaifs.json --cutoff 9 --hurdle --p90 \
            $USE_LIVE --extra-models "$MODELS" ${ONLY:+--stations "$ONLY"}
    } >> "${PLOYGON_SHADOW_LOG:-shadow_${TODAY}.log}" 2>&1 || true
fi

echo "已追加到 $LOG" >&2

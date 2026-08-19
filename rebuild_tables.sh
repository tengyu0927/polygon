#!/usr/bin/env bash
# rebuild_tables.sh — 重训之后必跑。**各时次按其部署口径分组回测**，再建各查表。
#
# 为什么不能一条命令跑完所有时次: 每档的 flag 和本地 GFS 时效都不同
# （9 时无 REGIME 且含 AIFS、12 时带 DQ、9-11 用 local12 而 12-14 用 local6）。
# 一把梭出来的序列不是生产在跑的东西，据此建的「预期命中」查表就是错的。
set -euo pipefail
cd "$(dirname "$0")"
M5="mos_ecmwf.csv mos_cma.csv mos_icon.csv mos_jma_.csv mos_gem_.csv"
F5="PLOYGON_CONV=1 PLOYGON_PROF=1 PLOYGON_CURVE=1 PLOYGON_WINDAM=1 PLOYGON_HPBL=1"
S=${1:-2025-05-01}; E=${2:-$(date +%F)}
say() { printf '\n\033[1m>>> %s\033[0m\n' "$*"; }

say "9 时（五要素 / 含 AIFS / local12）"
env $F5 PLOYGON_SETTLED=1 python3 backtest_nowcast.py --db cn.sqlite --cutoffs 9 \
    --start "$S" --end "$E" --nwp-csv mos.csv \
    --nwp-csv2 $M5 mos_local12.csv mos_aifs.csv --csv-out bt_c9.csv
say "10/11 时（五要素+REGIME / local12）"
env $F5 PLOYGON_REGIME=1 PLOYGON_SETTLED=1 python3 backtest_nowcast.py --db cn.sqlite \
    --cutoffs 10 11 --start "$S" --end "$E" --nwp-csv mos.csv \
    --nwp-csv2 $M5 mos_local12.csv --csv-out bt_c1011.csv
say "12 时（去量化 / local6）"
env PLOYGON_DQ=1 PLOYGON_SETTLED=1 python3 backtest_nowcast.py --db cn.sqlite --cutoffs 12 \
    --start "$S" --end "$E" --nwp-csv mos.csv --nwp-csv2 $M5 mos_local6.csv \
    --csv-out bt_c12.csv --no-baseline-check
say "13 时（纯线性 / local6）"
env PLOYGON_SETTLED=1 python3 backtest_nowcast.py --db cn.sqlite --cutoffs 13 \
    --start "$S" --end "$E" --nwp-csv mos.csv --nwp-csv2 $M5 mos_local6.csv --csv-out bt_c13.csv
say "14 时（五要素+REGIME / local6）"
env $F5 PLOYGON_REGIME=1 PLOYGON_SETTLED=1 python3 backtest_nowcast.py --db cn.sqlite \
    --cutoffs 14 --start "$S" --end "$E" --nwp-csv mos.csv --nwp-csv2 $M5 mos_local6.csv \
    --csv-out bt_c14.csv
say "15 时（纯实况）"
python3 backtest_nowcast.py --db cn.sqlite --cutoffs 15 --start "$S" --end "$E" \
    --csv-out bt_c15.csv

say "建查表"
python3 build_hit_table.py bt_c9.csv bt_c1011.csv bt_c12.csv bt_c13.csv bt_c14.csv \
    bt_c15.csv --out hit_table.json
python3 build_exceed_table.py --out exceed_table.json
say "完成。接着跑 check_consistency"

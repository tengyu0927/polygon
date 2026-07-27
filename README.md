# 站点日最高气温预报系统

8 个民航机场（ZBAA 北京首都、ZSPD 上海浦东、ZGGG 广州白云、ZGSZ 深圳宝安、
ZUUU 成都双流、ZUCK 重庆江北、ZHHH 武汉天河、ZSQD 青岛胶东）的日最高气温预报。

两条独立路线：

- **D+1/D+2 路线** —— 用 GFS 固定时效预报做 MOS 订正，提前 1-2 天
- **临近路线** —— 用当天上午实况外推当日峰值，提前 1-6 小时

---

---

## 零、新机器上怎么跑起来

```bash
git clone <你的仓库地址> && cd ploygon
pip3 install -r requirements.txt      # 可选，缺了也能跑（见下）
./bootstrap.sh                        # 建数据，完整版约 1.5-2 小时
python3 check_consistency.py          # 确认环境没问题，18 项应全过
./run_hourly.sh                       # 出预报
```

**仓库里没有数据**：实况库 510MB + 6 个模式库约 1.2GB，远超 GitHub 单文件
100MB 的硬上限。全部可由 `bootstrap.sh` 重建 —— 代价是时间不是信息。

| 命令 | 建什么 | 耗时 | 能跑什么 |
|---|---|---|---|
| `./bootstrap.sh --obs-only` | 只建实况库 | 30-60 分钟 | 12-15 时的纯实况档 |
| `./bootstrap.sh --quick` | 最近 2 年实况 + 6 模式 | 约 1 小时 | 全部预报，但重训时长序列不足 |
| `./bootstrap.sh` | 1995 年起实况 + 6 模式 | 1.5-2 小时 | 全部，含重训 |

**模型权重是随仓库走的**（`model.json`、`nowcast_nwp.json`、`nowcast.json`、
`nowcast_late.json`，共 7MB），所以数据一建好就能直接预报，不必先训练。
模型会随天气型漂移，建议按第四节的周期重训。

### 依赖

**核心推理不需要任何第三方库** —— 模型 JSON 里只有权重，纯标准库就能算。
`requirements.txt` 里两个都是「有则更好，无则自动降级」：

- `numpy`：训练提速几十倍。缺失时走纯 Python 实现，**结果一致**
- `scikit-learn`：岭回归+GBM 融合里的 GBM 部分。缺失时 `predict_mos.py`
  自动降级为纯岭回归并打警告，**D+1 会从 1.00 退到 1.08**

> ⚠️ 用 cron 时务必在 crontab 里钉住 PATH，否则会命中系统自带的
> `/usr/bin/python3`（通常没有这两个库），融合静默降级且不报错：
> ```cron
> PATH=/opt/homebrew/bin:/usr/bin:/bin
> ```

### 许可

本仓库为 **AGPL-3.0**。原因是 `merge_mos.py` 的 DEB 算法参照
[PolyWeather](https://github.com/yangyuan-zhen/PolyWeather)（AGPL-3.0）写成，
详见 `NOTICE.md`。数据来源与署名要求也在那里。


## 一、目前哪种最准

所有数字为**取整后**的 MAE（℃）。真值是整数度，取整是上线口径，
它让 ±1℃ 命中率白涨约 15 个百分点而 MAE 几乎不变。

| 时效 | 方法 | 脚本 / 模型 | MAE | ±1℃ |
|---|---|---|---|---|
| 48 小时 | 六模式 MOS D+2（DEB + 岭回归+GBM 融合） | `predict_mos.py` / `model.json` | **1.17** | 72% |
| 24 小时 | 六模式 MOS D+1（DEB + 岭回归+GBM 融合） | `predict_mos.py` / `model.json` | **1.00** | 78% |
| ~6 小时（09 时截止） | 临近 + 六模式 | `predict_nowcast.py` / `nowcast_nwp.json` | **0.87** | 80% |
| ~5 小时（10 时截止） | 临近 + 六模式 | `predict_nowcast.py` / `nowcast_nwp.json` | **0.81** | 83% |
| ~4 小时（11 时截止） | 临近 + 六模式 | `predict_nowcast.py` / `nowcast_nwp.json` | **0.71** | 86% |
| ~2 小时（12 时截止） | 临近 + 六模式 | `predict_nowcast.py` / `nowcast_nwp.json` | **0.61** | 91% |
| ~1 小时（13 时截止） | 临近 + 六模式 | `predict_nowcast.py` / `nowcast_nwp.json` | **0.46** | 96% |
| 14 时截止 | 临近 纯实况 | `predict_nowcast.py` / `nowcast_late.json` | **0.39** | 98% |
| 15 时截止 | 临近 纯实况 | `predict_nowcast.py` / `nowcast_late.json` | **0.18** | 98% |

> 临近那 7 行全部来自**同一个滚动回测**（2026-04-25~07-24，每档 728 站日，
> 均值决策 = 生产口径），可以直接横向比较。以前 14/15 时用的是年份切分
> （n≈7496）与其他行不可比、且报的是 ±1窗口决策的 MAE，现已统一。

**参照物**：GFS 原始输出（不做任何订正）D+1 是 1.73；机场 TAF 报文的 TX
在 15.9-27 小时时效上是 1.26-1.53。

### 怎么读这张表

**14/15 时那两行不要当真。** 14 时截止时 71% 的日子最高温已经出现，
15 时是 89% —— 那部分不是预报，是复述已知事实。只看真正需要预报的日子
（未见顶），14 时是 0.70、15 时是 0.85，**反而比 13 时的 0.64 更差**。
因为越晚截止，剩下的越是"峰值异常偏晚"的硬骨头。

**实用甜蜜点是 12-13 时截止。** 再往后整体数字好看，但没有增量信息。

**09 时是个分水岭。** 纯实况在 09 时只有 1.35，输给 24 小时前的 D+1（1.15）——
上午的观测告诉不了你午后云和天气系统怎么演变。加上 NWP 特征后变成 1.02，
反超。到 11 时纯实况已经能到 0.96，NWP 的边际价值降到 +0.14。

### 口径提醒

- 09/11 时那两行与 D+1/D+2 是**同一测试集**（2026-03-07 起，n=1104），可直接比
- 12/13/14/15 时用的是年份切分（测试期 2024-2026 全年，n≈7496），
  季节构成不同，跨行比较要留意
- 整数量化给 MAE 设了约 0.25℃ 的地板，别追一个物理上达不到的数

---

## 二、脚本清单

### 数据获取

| 脚本 | 用途 | 常用命令 |
|---|---|---|
| `iem_multi.py` | 从 IEM ASOS 归档抓多站逐时 METAR，写入 `cn.sqlite` | 见下 |
| `live_tmax.py` | 实时监看当日各站累计最高温（AWC 源） | `--once` |

```bash
# 增量更新最近两年（每天跑）
python3 iem_multi.py --db cn.sqlite \
  --stations ZSPD,ZUUU,ZGSZ,ZGGG,ZUCK,ZBAA,ZHHH,ZSQD --update
# 重建日表（tmax/tmin/n_obs），约 20 秒
python3 iem_multi.py --db cn.sqlite --daily
```

> **注意**：不要用 `--stations-file`。历史上的 `sites.txt` 是「中文名,ICAO」两列
> 格式，而解析器按一行一个 ICAO 读，会把整行当站号发给 IEM 导致 HTTP 422，
> 且那个文件漏了 ZSQD。该文件已移入 `archive/`，**一律用 `--stations`**。

### D+1/D+2 路线

| 脚本 | 用途 |
|---|---|
| `build_mos_dataset.py` | 拉单个模式的固定时效因子 + 拼训练集 |
| `merge_mos.py` | 把 7 个模式的 csv 横向拼成多模式训练集 |
| `run_daily.sh` | **每天跑一次 D+1/D+2**，建议 23 时，见下 |
| `check_consistency.py` | **一致性回归检查**，改完代码必跑，见附三 |
| `train_mos.py` | 训练并检验，导出 `model.json` |
| `predict_mos.py` | 用 `model.json` 预报指定日期 |

```bash
# 一次性：探测可用变量 -> 拉数据 -> 拼训练集
python3 build_mos_dataset.py probe
python3 build_mos_dataset.py inspect --obs-db cn.sqlite
python3 build_mos_dataset.py fetch --start 2024-01-01          # 约 2 分钟
python3 build_mos_dataset.py build --obs-db cn.sqlite --daily-table daily --out mos.csv

# 六模式: 每个模式各拉一份（--model / --db 换掉即可），再横向拼
python3 merge_mos.py --base mos.csv --extra mos_ecmwf.csv mos_cma.csv mos_icon.csv mos_jma_.csv mos_gem_.csv --out mos_multi.csv

# 训练（定期重跑，见第四节）
python3 train_mos.py mos_multi.csv --coef --dump model.json --pred pred.csv

# 每天预报（追加模式的顺序必须与 merge_mos.py --extra 一致！）
python3 predict_mos.py --ahead 1 --extra-models ecmwf_ifs025,cma_grapes_global,icon_global,jma_gsm,gem_global
```

**多模式把 MOS 也拉起来了**（同一测试集 n=1104，配对自助法）：

| 时效 | GFS 单模式 | **六模式** | ΔMAE |
|---|---|---|---|
| D+1 | 1.21 | **1.08** | +0.133 [+0.086, +0.180] 显著 |
| D+2 | 1.43 | **1.25** | +0.182 [+0.133, +0.231] 显著 |

收益比临近路线还大 —— 时效越长模式误差占比越高，多成员的价值也越大。

**架构会随特征数翻转。** 单模式下分站更优（验证 1.123 vs 合并 1.157），
多模式下合并更优（0.982 vs 1.038）—— 286 列除以 8 个站，分站样本不够。
`train_mos.py` 现在在**验证集**上比一次并把结论写进 `model.json` 的 `prefer` 字段，
`predict_mos.py` 照此选择，不再无条件用分站。

### 每天跑一次 D+1/D+2（什么时候跑）

```bash
./run_daily.sh          # 今天 + 明天
./run_daily.sh 2        # 今天 + 未来 2 天
```

结果追加到 `pred_mos_YYYY-MM-DD.log`。crontab：

```
0 23 * * * cd ~/projects/ploygon && ./run_daily.sh >> cron.log 2>&1
```

**结论：北京时 23:00 跑，不是 17:00。** 但理由不是 TAF 报文发布时刻 ——
本项目的预报不依赖 TAF（那只是对照基线，由 `taf_bias.py` 单独处理）。
真正的约束有两条，都实测过：

**① `recent_bias` 需要今天的日最高温已经定型。** 它是 D+1 模型里重要性排第二的
特征，定义是「过去 7 天实况 − 模式预报」的均值，其中包含**今天**。
而日最高温的峰值最晚可以出现在 17 时 —— 2026-07-25 成都双流就是 17:00 才见顶
（41℃）。17 点跑，等于把一个还没走完的今天当成已知事实，`recent_bias` 偏低，
预报会系统性偏冷。23 点跑，当天已经彻底定型。

**② IEM 归档有滞后。** 实测 2026-07-25 晚 19:00 抓取时，8 个站里有 5 个只归档了
19 条观测；到次日 00:43 补到 24 条。日最高温本身在 19:00 已经正确，
但越晚跑，`prev_tmax` 和 `recent_bias` 的输入越完整。

模式数据不是约束：实测凌晨 00:43 查明天的 `previous_day1` 已经是独立的一轮
（与 `previous_day2` 最大差 2.1℃，不是回退到同一轮），所以 17 点和 23 点在
模式侧差别不大。**卡点在实况侧。**

`run_daily.sh` 里内置了两个守卫：18 时之前跑会明确警告，各站今日观测少于 20 条
也会点名 —— 宁可提示也不静默出数。


**TAF 对照已内置在 `run_daily.sh` 里。** 每天跑完 MOS 顺手抓一次当日 TAF，
把两者并排记进 `taf_compare.sqlite`：

```
  站点      目标日          时效    MOS   TAF TX     分歧
  ZHHH    2026-07-26  D+2      36       33       +3
  ZUUU    2026-07-26  D+2      36       38       -2
```

攒够样本后（TAF 只存 15 天，历史补不回来，**只能靠每天记**）：

```bash
python3 taf_compare.py --analyze                # 回填实况，看谁准
python3 taf_compare.py --analyze --min-gap 2    # 只看分歧 >=2 度的日子
```

**要回答的问题不是「谁整体更准」** —— 那个已经知道了（六模式 D+1 1.08 vs
TAF 1.26-1.53）。要回答的是**分歧大的日子谁对**：如果 TAF 在分歧日赢或打平，
说明预报员看到了模型没有的信息，那 TAF 就该进模型当特征；如果一直输，
这条对照可以彻底关掉。`--analyze` 会分别给「全部」和「分歧子集」两组配对检验。

口径与 `taf_bias.py` 一致：只取落在午后峰值时段（10-19 时）的 TX、排除压在
有效期边界上的 TX（那是「窗口内最大」不是日最高温预报）、同一 (站,目标日,TX值)
只记首次出现（AMD 里的搬运不重复计数）。


**关键设计**：目标量是残差 `y - temperature_2m_max`，不是温度本身；
预报「今天」用的是 24 小时前起报的那一轮（`previous_day1`），与训练同口径，
**不是最新一轮**——用最新一轮会让模型面对没见过的时效。

**选型结论**：分站岭回归（`train_mos.py` 里已验证显著优于合并版 +0.048，
与梯度提升打平）。不需要 sklearn/lightgbm。

### 临近路线

| 脚本 | 用途 |
|---|---|
| `nowcast_potential.py` | 诊断工具：量化「知道上午实况后还剩多少不确定性」 |
| `train_nowcast.py` | 训练并检验，导出 `nowcast*.json` |
| `predict_nowcast.py` | 用当天上午实况预报今日最高温 |
| `backtest_nowcast.py` | 逐日滚动回测，按站 × 起报时算命中率（唯一干净的验证口径） |
| `run_hourly.sh` | **每小时跑一档临近预报**（9-15 时），自动选模型/取数/写日志 |

```bash
# 诊断（可选，不建模，只看上限）
python3 nowcast_potential.py --db cn.sqlite --cutoffs 9 11 12 13 14

# 训练：12/13 时用纯实况长序列（1995 年起，7.2 万站日）
python3 train_nowcast.py --db cn.sqlite --cutoffs 12 13 --dump nowcast.json
# 训练：09/10/11 时必须带 NWP，且用六模式（样本掉到 6 千，但那几档缺的就是模式信息）
# 追加模式的顺序必须与预报时 --extra-models 一致，否则 m2_/m3_ 各列对错模式、系数全部错位
python3 train_nowcast.py --db cn.sqlite --cutoffs 9 10 11 --nwp-csv mos.csv \
  --nwp-csv2 mos_ecmwf.csv mos_cma.csv mos_icon.csv mos_jma_.csv mos_gem_.csv --dump nowcast_nwp.json
# 训练：14/15 时（可选，见上文口径提醒）
python3 train_nowcast.py --db cn.sqlite --cutoffs 14 15 --dump nowcast_late.json

# 预报
python3 predict_nowcast.py --model nowcast_nwp.json --cutoff 11 --hurdle --p90 --verbose \
  --extra-models ecmwf_ifs025,cma_grapes_global,icon_global,jma_gsm,gem_global
python3 predict_nowcast.py --model nowcast.json     --cutoff 12 --hurdle --verbose
python3 predict_nowcast.py --auto --hurdle          # 按当前时刻自动选最接近的截止

# 滚动回测：每 30 天用「块开始日之前」的数据重训一次（气候态也重算），逐日前推
python3 backtest_nowcast.py --db cn.sqlite --nwp-csv mos.csv \
  --cutoffs 10 11 12 --start 2026-04-25 --end 2026-07-24 --csv-out backtest_3m.csv
```

### 每小时逐档预报（9-15 时，实战用法）

一条命令，自动按小时选模型、取数、输出、追加日志：

```bash
./run_hourly.sh          # 用当前整点
./run_hourly.sh 11       # 补跑 11 时
./run_hourly.sh 11 --no-fetch    # 数据已更新过，只重跑模型
```

结果同时打屏并追加到 `pred_YYYY-MM-DD.log`，一天下来就是完整的逐档演变记录。

想全自动的话，`crontab -e` 加这一行（每天 9-15 点整点跑）：

```
5 9-15 * * * cd ~/projects/ploygon && ./run_hourly.sh >> cron.log 2>&1
```

写 `5` 分而不是整点，是等 METAR 整点报进 IEM 归档。

**脚本按小时干了什么：**

| 起报时 | 模型 | 取数 | 耗时 |
|---|---|---|---|
| 9 / 10 / 11 时 | `nowcast_nwp.json`（六模式，72 项特征） | 实况 + 7 个模式今日预报 | 约 1-2 分钟 |
| 12 / 13 时 | `nowcast.json`（纯实况） | 仅实况 | 约 20 秒 |
| 14 / 15 时 | `nowcast_late.json`（纯实况） | 仅实况 | 约 20 秒 |

12 时之后不再取模式数据 —— 那几档实况信息已经够，纯实况模型用的是 1995 年起
7.2 万站日的长序列，打得过样本量只有 6 千的多模式版本。

**不想用脚本、手动跑的话**，每档等价于：

```bash
# 每档之前都要先更新实况（recent_bias / 上午观测都靠它）
python3 iem_multi.py --db cn.sqlite --stations ZSPD,ZUUU,ZGSZ,ZGGG,ZUCK,ZBAA,ZHHH,ZSQD --update
python3 iem_multi.py --db cn.sqlite --daily

# 9/10/11 时：还要拉 6 个模式的当日预报（--end 必须显式给今天！默认只到昨天）
for m in "mos_fcst gfs_global mos.csv" "mos_ecmwf ecmwf_ifs025 mos_ecmwf.csv" ...; do
  python3 build_mos_dataset.py fetch --db $DB.sqlite --model $MODEL --start <3天前> --end <今天>
  python3 build_mos_dataset.py build --db $DB.sqlite --obs-db cn.sqlite --daily-table daily --out $OUT
done
python3 predict_nowcast.py --model nowcast_nwp.json --cutoff 9 --hurdle --p90 --verbose \
  --extra-models ecmwf_ifs025,cma_grapes_global,icon_global,jma_gsm,gem_global

# 12/13 时
python3 predict_nowcast.py --model nowcast.json --cutoff 12 --hurdle --p90 --verbose
# 14/15 时
python3 predict_nowcast.py --model nowcast_late.json --cutoff 14 --hurdle --p90 --verbose
```

**晚上核对：**

```bash
python3 live_tmax.py --once        # 看当日真实最高温
grep -E "^  Z" pred_$(date +%F).log # 把各档预报排出来对比
```

> ⚠️ **别用 `predict_nowcast.py` 验证「今天」的准确率**。生产模型的训练数据
> 包含今天，跑出来偏乐观。要干净的验证口径，用
> `backtest_nowcast.py --start <当天> --end <当天>`，它只用当天之前的数据训练，
> 气候态也重算。

### 逐档预期表现（回测长期均值，别拿单天下结论）

| 起报时 | MAE | ±1℃ | 完全命中 | 起报时已见顶 |
|---|---|---|---|---|
| 9 时 | 0.87 | 80% | 39% | 6% |
| 10 时 | 0.81 | 83% | 41% | 15% |
| 11 时 | 0.71 | 86% | 46% | 23% |
| 12 时 | 0.61 | 91% | 49% | 36% |
| 13 时 | 0.46 | 96% | 58% | 51% |
| 14 时 | 0.39 | 98% | 64% | 65% |
| 15 时 | 0.18 | 98% | 85% | 82% |

14/15 时那两行不要当真（见第一节），那时大部分站最高温已经出现。


**业务指标定为 ±1℃ 命中率**（2026-04-25~07-24，728 站日/档，滚动回测实测）：

| 起报时 | ±1℃ 单 / 四 / **六模式** | MAE 单 / 四 / **六模式** | 起报时已见顶 |
|---|---|---|---|
| 10 时 | 79.8 / 81.5 / **82.7%** | 0.933 / 0.838 / **0.824** | 15% |
| 11 时 | 82.6 / 85.3 / **86.3%** | 0.810 / 0.732 / **0.732** | 23% |
| 12 时 | 90.5 / 90.9 / **90.9%** | 0.648 / 0.607 / 0.618 | 36% |

六模式分站 ±1℃（10/11/12 时）：深圳 93/95/100、上海 91/91/98、青岛 81/92/93、
武汉 86/91/89、广州 74/84/96、重庆 77/78/87、北京 76/84/89、**成都 67/71/82**。

**成员数的边际收益在 4 个之后基本饱和**：单→七显著（10 时 ±1 +2.9pt、11 时 +3.7pt），
但四→七在整体指标上三个时次全部无显著差异。**只有大升幅日例外** ——
实际升幅 ≥4 度的 408 个站日上，四→七 MAE 1.21→1.11，Δ=+0.096 [+0.046, +0.145] 显著。
所以六模式值得留，理由是极端日而不是平均表现。

**「完全命中」99-100% 物理上不可达**，别拿它当目标：真值是整数度，命中要求误差
< 0.5℃；12 时未见顶的日子平均还要涨 2-3℃，决定这几度的午后对流、海风锋、云的演变
在上午的地面观测和 25km GFS 里都没有信号。就算「是否已见顶」判得完美（上界），
10/11/12 时也只到 44%/55%/64%。

**`merge_mos.py` 那个「未解现象」已解决**：七模式时代加重复实况列能让 D+1 从
1.18 降到 1.08，机制查不出来。剔除 UKMO 并加入 DEB 后重测，加/不加/精确复制
三个变体全部持平（1.031 / 1.024 / 1.02，配对区间跨 0）。
**那不是普遍现象，是 UKMO 浅归档的副作用** —— 它在 2025-01 前没有数据，
那些"重复列"在缺测时被填成中位数，等于给了模型一个「2025-01 之前」的时代标记。
UKMO 一剔除，标记没了，效果也没了。默认已改成不加重复列（少 46 列）。

**试过、没赢的两个优化**（配对自助法，同一批站日）：

- **换决策规则**。±1℃ 命中的最优解是让 `P(rise ∈ {k-1,k,k+1})` 最大的 k，
  不是均值取整。实现了序贯分类 `P(rise≥k)` + 单调修正 + 窗口 argmax
  （`--` 三种决策并排输出）。结果 80/82/90 vs 均值 80/83/90，三个时次**全部无显著差异**。
- **组合两条路线**（`--blend-mos`）。每块滚动重训 D+1 MOS（训练期也用 walk-forward
  出样本外值，否则组合权重会高估 MOS），当特征喂给临近模型。结果 10 时
  Δ误差率 -0.019 [-0.039, -0.001]，**显著更差**；11/12 时无显著差异。
  原因大概是 GFS 的信息已经通过 8 个模式列进来了，MOS 那层订正主要是偏差校正，
  在 0-2 小时时效上没有增量，反而多 4 列噪声。

**赢了的优化：多模式集合**（`--nwp-csv2`）。ECMWF IFS 9km + CMA GRAPES 15km +
ICON 11km 接进来，除各自的 2m 温度/云量/辐射外，还给出**成员离散度**
（`ens_spread`、`ens_max_minus_min`）—— 模式吵得越凶剩余升温越不确定，
模型据此把预报往气候态收，这是单个高分辨率模式给不了的信息。
MAE 三个时次全部显著改善，1→2→4→7 个模式单调递增（4 个之后饱和，见上）。
**成都收益最大**（11 时 ±1℃ 59→68→73%）——
说明成都之前的短板不全是分辨率，也有单模式误差实现的运气成分。

**关于均态化**（多模式会不会削平极端高值）：实测**温度水平没有被削平** ——
可靠性斜率 0.986（单模式）vs 0.984（六模式），预报/实际标准差比 0.986 → 0.991，
最热 10% 日子的偏差 -0.34 → -0.21℃。结构上就削不动:
预报 = 已达最高（实测）+ 剩余升温，方差主要在实测那一截。
削平确实存在，但只在**剩余升温**上，且是 MAE 最优回归的固有性质，
**多模式反而缓解了它**：

| 实际剩余升温 | n | 单模式 ME / ±1℃ | 四模式 | 六模式 |
|---|---|---|---|---|
| 0 度 | 430 | +0.67 / 92% | +0.59 / 93% | +0.59 / 94% |
| 4 度 | 105 | -1.10 / 67% | -0.90 / 69% | -0.83 / 70% |
| ≥5 度 | 69 | **-1.88 / 29%** | -1.26 / 54% | **-1.03 / 61%** |

极端升温日往往是模式一致看好的晴热天，多成员互相印证反而敢报高。

**对策不是去修点预报**（那是零和的，会伤整体精度），而是另给高端情景:
序贯分类（`fit_ordinal`/`rise_pmf`/`rise_quantile`，模型 JSON 里的 `ordinal`）
产出完整的整数升幅分布，取 P90 作为「不排除冲到」值，`predict_nowcast.py --p90` 输出。
实测覆盖率 95%（标称 90%，略保守），只比点预报高 1.2-1.9℃。

> **`--p90` 不提升点预报准确率**，它只是多一路输出。用分布去改点预报已验证无效
> （众数决策、±1 窗口决策与均值取整三个时次全部无显著差异）。
> 它的价值是极端升温日不漏报 —— 播报形如「成都 38℃，不排除 40℃」。

**结论：分站两段式 + 均值决策 + 六模式集合 + P90 高端情景，不加 MOS 组合。**

### 岭回归 + 梯度提升融合（已上线）

单看梯度提升与岭回归仍是平手（D+1 Δ=+0.019 [-0.018,+0.056]，152 列没有翻盘，
27 列时代的结论依然成立）。**但两者误差结构不同，平均之后两个都被显著打败**：

| 时效 | 岭回归 | GBM | **融合** | 融合 vs 岭回归 |
|---|---|---|---|---|
| D+1 | 1.08 | 1.07 | **1.03** | +0.041 [+0.025, +0.058] 显著 |
| D+2 | 1.20 | 1.19 | **1.17** | +0.037 [+0.024, +0.050] 显著 |

融合权重在**验证集**上选（D+1 岭回归占 0.6，D+2 占 0.7），不是拍的。

GBM 没法序列化成 JSON，单独存 `model.json.gbm.pkl`。
**预测端缺 sklearn 会自动降级成纯岭回归并打警告**（不静默）——
cron 的 `/usr/bin/python3` 就没有 sklearn，所以 crontab 里必须钉住 PATH：

```cron
PATH=/opt/homebrew/bin:/usr/bin:/bin
15 9-15 * * * cd ~/projects/ploygon && ./run_hourly.sh >> cron_hourly.log 2>&1
59 23 * * * cd ~/projects/ploygon && ./run_daily.sh >> cron_daily.log 2>&1
```

不钉的话每天默默少 0.05℃ 精度，而且不会报错。

### 试过没赢的: 预报「今天几点见顶」

先诊断出一个真问题：**误差随实际见顶时刻单调上升**（728 站日 × 2 档）：

| 截止 | ≤13时见顶 | 14-15时 | 16-17时 | ≥18时 |
|---|---|---|---|---|
| 11 时 | 0.56 | 0.79 | 0.96 | 0.60 |
| 13 时 | **0.28** | 0.56 | **0.77** | 1.20 |

而且见顶时刻是**分站属性**（7-8 月，1995 年至今）：

| 站 | 中位见顶 | ≤13时 | 14-15时 | 16-17时 |
|---|---|---|---|---|
| 上海 | 12 时 | 84% | 15% | 1% |
| 青岛 / 深圳 | 13 时 | 67% / 53% | 26% / 37% | 7% / 10% |
| 北京 / 广州 | 14 时 | 29% / 40% | 52% / 46% | 19% / 13% |
| 武汉 | 15 时 | 23% | 45% | 31% |
| **重庆** | 15 时 | 10% | 43% | **46%** |
| **成都** | 15 时 | 16% | 43% | **37%** |

盆地站近一半的日子 16-17 时才见顶，而最晚的档是 15 时 —— 这解释了重庆为什么
连续多天被报低。模型有 `clim_peak_h`，知道「重庆通常晚」，但不知道「今天会不会晚」。

于是做了两段式：先用上午特征回归当天见顶时刻，再把
`pk_pred` / `pk_minus_clim` / `pk_minus_cutoff` 喂给主模型（`--peak-feat`）。

**结果：不采用。** 七档里只有 13 时显著（+0.019 [+0.007, +0.033]），
9/10/11 时方向偏坏，其余不显著。**测 7 档出现 1 档显著，大概率是多重比较**。
决定性证据是目标人群没站住 —— 16 时后见顶的 102 天上，七档变化正负交替
（+0.010 / 0.000 / −0.029 / +0.020 / −0.020 / −0.020 / −0.010），无一致方向。
机制若真成立，这一层该全线改善。

> 这条留作方法论提醒：**先看目标人群，再看总体显著性**。
> 只盯总体 p 值的话，13 时那个 +0.019 足以让人误判为成功。

晚见顶的问题本身是真的，剩下两条路：给盆地站加 16/17 时档（业务价值随时间递减），
或者接受它、靠 P90 高端情景兜住（已在做）。


### 试过没赢的: 交互项

给临近模型加了 5 个物理上有依据的乘性项（`cld_mean_am × hours_to_peak`、
`ens_spread × hours_to_peak`、`nwp_minus_sofar × hours_to_peak`、
`ts_am × clim_rise`、`dpd_now × hours_to_peak`）。
结果 10/11/12 时 ΔMAE +0.005 / +0.004 / −0.004，**区间全部跨 0**。
`train_nowcast.py` 里 `add_interactions()` 保留，加回 `FEATS` 即可重测。


### P90 高端情景改用分位数回归

原来的 P90 取自序贯分类的整数升幅分布（PMF）。**问题不在总覆盖率而在结构**：

| 口径 | 旧 PMF | **分位数回归** |
|---|---|---|
| 总体覆盖（标称 90%） | 94-95% | 93-94% |
| 平均比点预报高 | 1.12-1.74℃ | **0.96-1.31℃** |
| 按预测升幅分桶的平均标定偏离 | 4.2-4.6pt | **3.0-4.1pt** |

新做法是直接回归条件分位（pinball 损失，IRLS 求解），系数格式与岭回归一致，
**模型 JSON 里仍然只有权重，推理不需要任何机器学习库**。
自检：在已知真值的合成数据上，τ=0.9 实测覆盖 89.2%、τ=0.5 覆盖 50.2%。

**一个统计学要点（踩过才明白）**：诊断时先看到「实际大升幅日的覆盖只有 60-83%」，
于是做了按预测升幅分桶的偏移标定去补。结果**正确口径下的标定反而变差**
（平均偏离 90% 从 3.0pt 涨到 4.0pt）。原因是分位数承诺的是
**给定预测**的覆盖，不是**给定实际结果**的覆盖 —— 后者永远达不到 90%，
除非区间无限宽。追错口径就是这个下场。
`calibrate_quantile()` 保留在代码里并注明未启用，供以后复现。


### 借鉴 PolyWeather 的 DEB（已上线）

[PolyWeather](https://github.com/yangyuan-zhen/PolyWeather)（AGPL-3.0）做的是
Polymarket 温度结算市场的多模式融合，与本项目同域。它的核心算法
**DEB（Dynamic Error Balancing）**值得借鉴：

- 按各模式近 7 天误差的**倒数**加权，误差用指数衰减（0.85^天数）
- 各模式的 signed bias 绝对值进分母做惩罚（样本 <5 天不采信）
- 成员**分歧过大时向等权回退**（原实现阈值 3°F ≈ 1.7℃）

**为什么它能补上我们缺的东西**：岭回归学到的是**固定**线性组合，而 DEB 权重是
近期误差的函数、每天都在变。这种随时间自适应的非线性组合，线性模型表示不出来。

DEB 单独当融合器（`merge_mos.py --deb` 里的 `deb_pred`）：

| 方法 | D+1 MAE（全期 n=6027）|
|---|---|
| GFS 原始 | 1.610 |
| 等权集合平均 | 1.413 |
| **DEB 自适应加权** | **1.361** |

比等权平均好 0.05，但离训练后的 1.00 还远 —— 所以**当特征用，不当预报器用**。
加进 MOS 后（同一批 1099 站日，配对检验）：

| 时效 | 无 DEB | **有 DEB** | ΔMAE |
|---|---|---|---|
| D+1 | 1.082 | **1.031** | +0.051 [+0.010, +0.091] 显著 |
| D+2 | 1.212 | **1.169** | +0.043 [+0.001, +0.085] 显著 |

`deb_weights()` 的数学在 `merge_mos.py` 里，**训练端与预测端共用同一个函数**，
避免两处实现漂移（这个项目已经踩过一次这类坑，见 `m{i}_present`）。

**试过没赢的**：把各模式自己的 `recent_bias` 加进**临近**模型（DEB 的另一半思路）。
10/11/12 时 ΔMAE −0.021 / +0.001 / −0.005，全部无显著差异。
事后想明白: 临近路线已有 `max_so_far`（当天实测），对预报的锚定远强于模式的 7 天
偏差 —— `recent_bias` 在 D+1 时效上系数排第二，一旦有了当日实况就失效了。

### 架构选择改成三方比较

之前只比「合并 vs 分站」，融合被挡在门外: D+1 验证集上分站 0.939 微弱胜过合并 0.948
于是选了分站，而融合是 0.888。现在三方一起比，`prefer` ∈ {pooled, per_station, blend}：

```
── 架构选择（验证集 MAE）
  blend       0.888  <- 上线用这个
  per_station 0.939
  pooled      0.948
```


### 成员不是越多越好：归档深度陷阱

加模式前必须看它的归档起点。**归档只覆盖训练期尾部的模式会主动伤害精度** ——
它的系数只能从少数几个月（还是特定季节）估出来，却要用到全年。

| 配置 | D+1 | D+2 | 配对检验（对 6 模式） |
|---|---|---|---|
| 6 模式（GFS/ECMWF/CMA/ICON/JMA/GEM，均 2024-07 起） | **1.08** | **1.20** | — |
| +UKMO（2025-01 起） | 1.08 | 1.25 | D+2 Δ=-0.047 [-0.084, -0.005] **更差** |
| +UKMO+AIFS（2025-07 起） | 1.17 | 1.40 | 明显更差，ME 变成 -0.42/-0.66 的系统性冷偏 |

所以**生产配置是 6 个模式**，UKMO 与 AIFS 的数据留着但不参与训练。
等它们各自攒够一整年、能落进训练期主体时再重测 —— `merge_mos.py --extra` 加回去
即可，一条命令。

临近路线上同样的比较：去掉 UKMO 后 10/11/12 时 MAE 分别 -0.014/-0.026/-0.008，
方向一致但都不显著；既然方向一致又少拉一个模式（快约 40 秒），一并统一到 6 个。

### 12/13 时也该用模式，之前的结论已推翻

原来 12/13 时用纯实况长序列（1995 年起 7.2 万站日），理由是样本量压倒模式信息。
接入多模式后这个结论不成立了（同一批 728 站日，配对检验）：

| 起报时 | 纯实况 | **六模式** | ΔMAE |
|---|---|---|---|
| 12 时 | 0.716 | **0.618** | +0.098 [+0.043, +0.159] 显著 |
| 13 时 | 0.519 | **0.455** | +0.065 [+0.022, +0.109] 显著 |

`nowcast_nwp.json` 现在覆盖 9/10/11/12/13 五档，`run_hourly.sh` 已同步。
14/15 时仍用纯实况（`nowcast_late.json`）—— 那时大部分站已见顶，模式没有增量。


**关键设计**：目标量是「剩余升温」`Tmax - 截止时刻已达最高`，非负且在 0 处
有大量堆积（12 时截止约 1/3 已见顶）。两段式（`--hurdle`）先判「还会不会再升」
再回归升幅，四个截止时刻都显著优于直接回归。预测截断到 ≥0——
`Tmax ≥ 已达最高` 是恒等式，模型不该违反。

**选型结论**：分站两段式。分站在所有截止时刻显著获胜，且收益随截止时刻推迟
而递减（09 时 +0.103 → 13 时 +0.038）——早上信息越少，站点固有的日变化形态
就越重要。

### TAF 对照（可选）

| 脚本 | 用途 |
|---|---|
| `taf_bias.py` | 回补 15 天 TAF + METAR，分站分轮次算偏差 |
| `pair_taf_model.py` | TAF 与模式在同一批站日、可比时效上的配对比较（回算） |
| `taf_compare.py` | **每天记一条 MOS vs TAF 并排样本**，攒前瞻分歧日 |

```bash
python3 taf_bias.py --days 15                    # 回补并分析，约 8 分钟
python3 taf_bias.py --analyze-only               # 已回补过，只重跑分析
python3 pair_taf_model.py --taf-db taf_bias.sqlite --mos mos.csv
```

TAF 目前只是业务基线对照，不是训练数据来源（AWC 只存 15 天，且 TAF 的 TX
在修订报里常常只是搬运上一版的值，不是新预报）。**是否该让它进模型，
取决于 `taf_compare.py --analyze` 在分歧日上的结论** —— 见第二节。

> `Fetch_taf.py`、`push_report.py`、`notify.py`（钉钉推送那条链路）在另一台
> Linux 机器上，与本机的建模工作独立。

---

## 三、数据与模型文件

| 文件 | 内容 | 大小/覆盖 |
|---|---|---|
| `cn.sqlite` | 逐时实况 `obs` + 日表 `daily` | 510MB，8 站，1995-01 起 |
| `mos_fcst.sqlite` | GFS 固定时效因子（8 个地面变量 × D+1/D+2） | 246 万条，2024-01 起 |
| `taf_bias.sqlite` | TAF 报文与 TX 解析结果 | 最近 15 天 |
| `mos.csv` | D+1/D+2 训练集（GFS 单模式） | 14648 行 × 27 列 |
| `mos_multi.csv` | D+1/D+2 训练集（六模式横向拼） | 14696 行 × 176 列 |
| `mos_ecmwf.sqlite` / `.csv` | ECMWF IFS 9km 固定时效因子 | 202 万条，2024-07 起 |
| `mos_cma.sqlite` / `.csv` | CMA GRAPES 15km（中国自研） | 2024-07 起 |
| `mos_icon.sqlite` / `.csv` | DWD ICON 11km | 2024-07 起 |
| `mos_jma_.sqlite` / `.csv` | JMA GSM | 2024-07 起 |
| `mos_gem_.sqlite` / `.csv` | 加拿大 GEM | 2024-07 起 |
| `mos_ukmo.sqlite` / `.csv` | UKMO 10km，**已停用**（归档浅，见下） | 2025-01 起 |
| `mos_aifs.sqlite` / `.csv` | ECMWF AIFS（AI），**已停用**（同上） | 2025-07 起 |
| `model.json` | D+1/D+2 模型（六模式，含 `prefer` 与 `blend_w`） | — |
| `model.json.gbm.pkl` | 融合里的 GBM 部分（需 sklearn，缺则自动降级） | — |
| `nowcast.json` | 临近模型，纯实况 12/13 时，**已被六模式版取代**，留作对照 | — |
| `nowcast_nwp.json` | 临近模型，六模式集合，09/10/11/12/13 时截止 | — |
| `nowcast_late.json` | 临近模型，纯实况，14/15 时截止（含 P90 序贯分类） | — |

模型 JSON 只含标准化参数和权重，推理时**不需要任何机器学习库**。

---

## 四、每天的流程

### 早上 08:30（09 时截止之前）

```bash
cd ~/projects/ploygon
python3 iem_multi.py --db cn.sqlite \
  --stations ZSPD,ZUUU,ZGSZ,ZGGG,ZUCK,ZBAA,ZHHH,ZSQD --update
python3 iem_multi.py --db cn.sqlite --daily
python3 predict_mos.py --ahead 1 > pred_mos_$(date +%m%d).txt
```

`predict_mos.py` 给出今天的 D+1/D+2 和明天的预报。
**实况必须先更新**——`recent_bias` 是重要性排第二的特征，滞后两天会让
预报偏差零点几度。

### 白天分档更新（推荐直接用 `./run_hourly.sh`，见第二节）

| 时刻 | 命令 |
|---|---|
| 09:15 | `predict_nowcast.py --model nowcast_nwp.json --cutoff 9 --hurdle --p90 --extra-models ecmwf_ifs025,cma_grapes_global,icon_global,jma_gsm,gem_global` |
| 10:15 / 11:15 / 12:15 / 13:15 | 同上，换 `--cutoff` |
| 12:15 | `python3 predict_nowcast.py --model nowcast.json --cutoff 12 --hurdle` |
| 13:15 | `python3 predict_nowcast.py --model nowcast.json --cutoff 13 --hurdle` |

每档之前若 IEM 归档还没跟上，先跑一次 `iem_multi.py --update`，
或者给 `predict_nowcast.py` 加 `--live` 从 AWC 实时取数。
**日常优先用 `cn.sqlite`**（与训练同源同口径），`--live` 留给应急。

### 晚上 20:00 之后

```bash
python3 live_tmax.py --once
```

对照当天各份预报，记录真实的前瞻样本。所有离线 MAE 都是回算，
回算与实际业务之间总有落差，只有攒真实前瞻样本才知道有多大。

### 重训周期

| 模型 | 建议周期 | 命令 |
|---|---|---|
| D+1/D+2 | 每月 | 7 个模式各 `fetch/build` + `merge_mos.py` + `train_mos.py mos_multi.csv` |
| 临近（纯实况） | 每季度 | `train_nowcast.py --cutoffs 12 13` |
| 临近（四模式） | 每月 | `train_nowcast.py --cutoffs 9 10 11 --nwp-csv mos.csv --nwp-csv2 mos_ecmwf.csv mos_cma.csv mos_icon.csv mos_jma_.csv mos_gem_.csv`（四份 csv 都要先重拉重建） |

D+1 的 `recent_bias` 系数排第二，说明偏差随天气型漂移，
NWP 相关的模型对新数据更敏感，重训要勤一些。

---

## 五、已知问题

**ZGSZ 深圳宝安是唯一被模型拖累的站。** 它在临近模型的 11/12/13 时截止上
都输给「气候平均升温」这个常数基线（0.72 vs 0.67、0.61 vs 0.52、0.63 vs 0.54）。
原因大概是信号太弱：深圳 12 时截止时平均只剩 1.4℃ 要涨、标准差 0.6℃，
模型引入的估计噪声超过了它挖到的信号。**建议对该站晚截止时刻直接用气候基线，
或只用 GFS 原值**——值得单独验一次。

**`live_tmax.py` 只监看 7 个站，漏了 ZSQD。** 与 `sites.txt` 的问题同源。
而 ZSQD 恰好常常是各方法分歧最大的站，补上才能验证。

**`sites.txt` 格式与 `--stations-file` 解析器不匹配**（见第二节）。

**盆地站仍是最难的，但没原先想的那么无解。** ZUUU 成都在 D+2 上只从 2.36
改进到 2.01，是唯一没压下去的。临近侧接入四模式后成都 12 时 ±1℃ 从 78% 到 86%，
说明「25km 分辨不出成都平原地形」不是全部原因，单模式的误差实现占了一部分。
**Open-Meteo 在中国区没有 ≤5km 的可用接口**（HRRR 仅北美、ICON-D2/AROME 仅欧洲、
KMA LDPS 无数据；JMA MSM 5km 只覆盖上海和青岛，成都在域外），
所以继续挖只能靠更多模式成员、地形/边界层因子。

---

## 六、下一步可做的

1. ~~**给临近模型加更多 NWP 特征。**~~ 已做（`train_nowcast.py` 的 `NWP_COLS`）：
   加了午后云量、短波辐射、相对湿度、露点差、日较差、风速/阵风共 8 个模式列。
   同样本同测试期的配对检验：09 时分站两段式 +0.029℃ [+0.012, +0.046] 显著，
   11 时 +0.011 [-0.001, +0.022] 不显著；合并版增益大得多（+0.055 / +0.038）——
   分站模型已经把大部分信息吃掉了。09 时部分站 alpha 从 1 升到 30，
   说明特征维度已接近饱和，再加列要先减维（如只留 PCA 前几个模式主成分）。
2. **ZGSZ 单独处理**（见第五节）。
3. ~~**滚动检验。**~~ 已做，见 `backtest_nowcast.py`（每 30 天重训、测下一段，
   气候态同步重算）。
4. ~~**组合两条路线。**~~ 已试，**没用**（见第二节临近路线小节）。10 时显著更差，
   11/12 时无显著差异。`backtest_nowcast.py --blend-mos` 留着，换更长评估期可再验。
5. **攒真实前瞻样本。** 所有数字都是回算，一天都还没真正提前预报过。

---

## 附：一些反复踩到的坑

- **接口文档与实际行为常常对不上。** 这一路碰到：`hours` 参数无效、
  `recent` 参数被删、GraphCast 退役、时间戳带毫秒和 Z、
  Previous Runs 不支持气压层变量。**先用小样本跑通再上全量**。
- **SQLite 对"文件/表不存在"太宽容。** `sqlite3.connect()` 遇到不存在的文件
  会静默创建空库，`CREATE TABLE IF NOT EXISTS` 撞上异构表也不报错，
  错误往往在下游某个看不懂的地方才爆出来。相对路径的默认值会放大这个问题。
- **同一个数值在不同上下文里含义不同。** TAF 里 TX 是"窗口内最大值"，
  只有当窗口完整覆盖午后峰值时它才等于"日最高温预报"；压在有效期
  起始或末端边界上的 TX 是另一回事。
- **绝不随机划分训练/测试集。** 相邻日天气高度相关。
- **在同一批数据上估偏差再减掉，MAE 必然下降，那是自欺欺人。**
  必须留一交叉验证或时间外推。
- **配对检验才能下结论。** 同一天各站误差高度相关，
  独立算 MAE 再比大小会严重高估显著性。

---

## 附二、目录里都有什么

只保留在用的东西。被上位替代的脚本与数据都在 `archive/`，确认无碍后可整个删掉
（`rm -rf archive/`）—— **本目录不是 git 仓库，删了无法恢复，所以先归档不直删**。

**生产脚本**

| 文件 | 角色 |
|---|---|
| `run_daily.sh` / `run_hourly.sh` | 两条线的入口，日常只需跑这两个 |
| `iem_multi.py` | 实况入库 |
| `build_mos_dataset.py` / `merge_mos.py` | 模式数据入库 + 六模式拼表 |
| `train_mos.py` / `predict_mos.py` | D+1/D+2 训练与预报 |
| `train_nowcast.py` / `predict_nowcast.py` | 临近训练与预报 |
| `backtest_nowcast.py` | **唯一干净的验证口径**（严格时间切分） |
| `taf_compare.py` / `taf_bias.py` / `pair_taf_model.py` | TAF 对照（前瞻 / 回补 / 回算） |
| `live_tmax.py` / `nowcast_potential.py` | 实况监看 / 不确定性上限诊断 |

`train_mos.py` 同时是公共库 —— `ridge_fit`/`ridge_pred`/`paired_boot` 被其他脚本
import，**不能删**。`train_nowcast.py` 同理（特征构造被 predict 和 backtest 复用）。

**权重**：`model.json`（D+1/D+2 六模式）、`nowcast_nwp.json`（9/10/11 时六模式）、
`nowcast.json`（12/13 时纯实况）、`nowcast_late.json`（14/15 时纯实况）。
每个各留一份 `.bak` 作回滚点，多余的历史备份已归档。

**数据**：`cn.sqlite`（实况）、`mos_fcst/ecmwf/cma/icon/jma_/gem_/ukmo.sqlite`（六个
模式的固定时效因子，约 1.5GB）、对应的 `mos*.csv` 与拼好的 `mos_multi.csv`。
**模式 sqlite 别删** —— 每月重训靠它增量更新，删了要从 2024-07 重拉，每个约 10 分钟。

**日志**：`pred_YYYY-MM-DD.log`（临近逐档）、`pred_mos_YYYY-MM-DD.log`（D+1/D+2）、
`taf_compare.sqlite`（MOS vs TAF 并排样本）。这些是**正在积累的真实前瞻样本**，
别清。

---

## 附三、改完代码先跑一致性检查

```bash
python3 check_consistency.py                      # 全部（约 2 分钟，要联网）
python3 check_consistency.py --skip-mos --skip-scripts   # 只查特征与契约，秒级
```

**改完任何代码，这个必须通过才算完。** 它抓的是这个项目反复踩的四类坑：

| 组 | 检查项 |
|---|---|
| ① 特征一致 | 同一站日分别走 `make_samples`（训练）与 `predict_nowcast`（预测）的构造路径，**逐列比对 536 个值**；模型 JSON 点名的列预测端必须全部能产出 |
| ② MOS 对齐 | `model.json` 的 `feats` 与 `mos_multi.csv` 的列；`predict_mos.py` 实跑；实际架构与 `prefer` 相符 |
| ③ 脚本契约 | 两个 shell 的 `MODELS` 一致、与模型要求的成员数一致；一律用 `TZ=Asia/Shanghai`；DEB 训练/预测共用同一实现 |
| ④ **入口脚本实跑** | 在 `env -i PATH=/opt/homebrew/bin:/usr/bin:/bin` 的**模拟 cron 环境**里完整跑 `run_hourly.sh` 与 `run_daily.sh`，断言：输出站数正确、无降级标记、单实例锁生效、D+1/D+2 确实走了融合模型（走不了说明 cron 的 python 没 sklearn） |

两个脚本都支持 `PLOYGON_LOG` / `PLOYGON_TAF_DB` 环境变量覆盖输出路径，
所以检查器实跑不会污染真实日志和前瞻样本库。
`run_daily.sh` 那项约 5 分钟（`predict_mos` 要为 6 个模式各拉 30 天窗口），
急的话加 `--skip-daily`。

### 为什么第 ④ 组必须存在

前面几次"验证通过"用的都是 `--no-fetch`，**把最耗时、最容易坏的取数那段整个跳过了**，结果：

- 删冗余代码时补丁匹配到旧文本，`DBS` 数组被删但循环还在 → cron 里 `unbound variable` 直接崩，本地怎么测都是好的
- `iem_multi --update` 是重抓两整年（8 站约 10 万行），放进每小时任务里，IEM 慢起来挂几十分钟；`urlopen` 的 `timeout` 只管单次 socket 读，服务器涓流发数据永远不触发
- 没有单实例锁，卡住的进程每小时叠一个，互相抢 `cn.sqlite` 写锁，越堆越死

**交互式 shell 的 PATH 与环境和 cron 完全不同，必须用 `env -i` 复现。**
检查器已用注入真实 bug 的方式验证过确实能抓到（`✗ 退出码 0，输出 0 个站`）。

> 检查器自己也会有 bug。第一版报了 48 处不一致，查下来是**它**重算了一份气候态，
> 而生产用的是模型 JSON 里 dump 出来的那份。比对必须用部署的那份参数。

### 日常运维要点

- **实况增量用 `--recent-days N`，不要用 `--update`**。前者 4 秒，后者重抓两整年
- 两个入口脚本都有 `mkdir` 实现的单实例锁（macOS 没有 `flock`），带 PID 存活检测
- 实况更新失败不阻断预报，会打 `[warn]`；模式特征缺失会在**每个站的备注里**打 `⚠`
- 攒前瞻样本期间**冻结代码**。要优化就在副本上做，验证通过后择期一次性上线 ——
  边跑边改的话，日志里混的是不同代码不同模型的输出，没有分析价值
- `run_daily.sh` 单次约 5 分钟，23:59 起跑约 00:04 结束。跨午夜不影响正确性 ——
  目标日在 bash 里就取好并用 `--date` 钉死，与起跑时刻无关

### 已知的、可接受的差异

`backtest_nowcast.py` 与 `train_nowcast.py` 选 alpha 的验证窗口不同
（回测用训练期末尾 90 天，生产用日期分位 0.70~0.85）。两者都合理，
但意味着回测数字与生产模型不会逐条相等。回测是更保守的估计口径，保持现状。

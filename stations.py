#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stations.py — 站点清单的**唯一真相源**。

    import stations as S
    S.ICAOS          # ['ZBAA', 'ZGGG', ...]  排序固定
    S.NAMES["ZBAA"]  # '北京首都'
    S.COORD["ZBAA"]  # (40.0801, 116.5846)
    S.CSV            # 'ZBAA,ZGGG,...'  给 shell 用

为什么要这个文件:

2026-08-08 加郑州/济南之前清点，发现站点清单硬编码在 **17 个文件**里，
三种形式（ICAO->经纬度 7 处、ICAO->中文名 7 处、逗号串 3 处）。这类重复
已经出过两次事故:
  - ZGSZ 因 --split-year 静默从模型里掉出去，检查还是绿的
  - stn_id 训练端设了、预测端没设，靠逐列比对才抓到
而且清点时当场又发现一处**活着的不一致**: 青岛的经度在 build_mos_dataset.py
里是 120.0864（胶东机场，正确），在另外 6 个取模式数据的文件里是 120.3864 ——
差 0.3 度约 27 公里，在 0.25 度网格上是**不同格点**。也就是 Open-Meteo 六模式
取的是胶东，本地 GFS 取的是别处。120.3864 既不是胶东也不是流亭（120.3744），
像是 0864 打成 3864。流亭 2021 年已关闭，胶东才是在运行的机场。

**本文件一律用正确坐标。** 但本地 GFS 的历史归档（gfs18.csv / mos_local*.csv）
是按旧坐标抽的，直接改会让训练/预测口径错位，所以给了 LEGACY_GFS_COORD ——
在重新抽取并 A/B 验证之前，取 GFS 的那几个脚本仍用旧值。
"""

from __future__ import annotations

# ICAO -> (中文名, 纬度, 经度)
# 坐标以机场基准点为准，用于取数值模式的最近格点。
_STATIONS = {
    "ZBAA": ("北京首都", 40.0801, 116.5846),
    "ZSPD": ("上海浦东", 31.1443, 121.8083),
    "ZGGG": ("广州白云", 23.3924, 113.2988),
    "ZGSZ": ("深圳宝安", 22.6393, 113.8107),
    "ZUUU": ("成都双流", 30.5785, 103.9471),
    "ZUCK": ("重庆江北", 29.7192, 106.6417),
    "ZHHH": ("武汉天河", 30.7838, 114.2081),
    "ZSQD": ("青岛胶东", 36.3661, 120.0864),
    # 2026-08-08 新增。实况来源不同:
    #   ZHCC 郑州: IEM 有 1995 年起的逐时（720 条/月），与原 8 站同源
    #   ZSJN 济南: IEM 只有 42-81 条/月（每天 1-2 条，不是逐时），**不可用**，
    #              必须走 WU（2020 年起逐时齐全），与 ZGSZ 同一条路
    "ZHCC": ("郑州新郑", 34.5197, 113.8408),
    "ZSJN": ("济南遥墙", 36.8572, 117.2160),
}

# 本地 GFS 抽取用的旧坐标。**只在重新抽取+A/B 之前保留。**
# 见文件头说明: 青岛这个是错的，但现有 gfs18.csv / mos_local*.csv 按它抽的。
LEGACY_GFS_COORD = {
    "ZSQD": (36.3617, 120.3864),
}

ICAOS = sorted(_STATIONS)
NAMES = {k: v[0] for k, v in _STATIONS.items()}
COORD = {k: (v[1], v[2]) for k, v in _STATIONS.items()}
CSV = ",".join(ICAOS)

# 实况以 Weather Underground 为准、不能用 AWC/IEM METAR 的站。
#   ZGSZ: WU 的 ZGSZ:9:CN 返回 Lau Fau Shan（香港流浮山，WMO 45035），
#         而打分以 WU 为准，所以模型也训练在这条序列上（见 wu_obs.py）
#   ZSJN: IEM 根本没有逐时数据，只能用 WU
WU_STATIONS = {"ZGSZ", "ZSJN"}


def gfs_coord(icao):
    """取 GFS 抽取用的坐标（含历史遗留覆盖）。"""
    return LEGACY_GFS_COORD.get(icao, COORD[icao])


GFS_COORD = {k: gfs_coord(k) for k in ICAOS}

# 抽取时额外多抽的点。2026-08-08 发现青岛的 GFS 坐标是错的（见文件头），
# 但现有归档按旧坐标抽的、模型也是这么训的，不能直接换。
# 反正为新站要重下一遍 GRIB，顺便把正确坐标也抽出来（同一份缓冲，成本≈0），
# 之后用 A/B 决定要不要切。切换前它只是多一列，不进任何模型。
EXTRA_POINTS = {
    "ZSQD_NEW": COORD["ZSQD"],      # 胶东机场的正确坐标
}
GFS_COORD_EXTRACT = {**GFS_COORD, **EXTRA_POINTS}

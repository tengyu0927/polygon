#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sonde.py — 早晨探空（00Z = 北京时 08 时）的廓线解析与特征构造。

    python3 sonde.py --build --out sonde.csv        # 从 IGRA 归档建训练表
    python3 sonde.py --live ZBAA --date 2026-08-12  # 实时取一份（预测端用）

**为什么两端必须共用 `feats()`。**

日最高温的物理机制是「早晨的逆温被混合层从下往上吃掉」。经典混合层法:
从地面沿干绝热线（位温守恒）上抬，与晨间廓线的交点决定当日可达的地面温度。
这是 08 时就能算出来的量，而且是**实测**不是模式 —— 临近侧现在 120 项特征
里 81 项是原始模式值，一条实测廓线都没有。

但训练端只能用 IGRA 归档（滞后 3 天，9:15 起报用不了），预测端只能用怀俄明
实时接口。**两个来源、两种格式**。2026-08-12 刚在 D+1 订正输出那件事上栽过:
训练端和预测端用不同来源的同名量，测出来的增益是假的（10 时首轮
ΔMAE −0.0231/P=99.9%，换成同口径序列后归零）。

所以这里的设计是: 两种格式各自只负责**解析成同一个 profile 结构**
（[(气压 hPa, 高度 m, 温度 ℃, 露点 ℃), ...]），派生量一律由 `feats()`
这一个函数算。`check_consistency` 的逐列比对能当场验证两端产出一致。

IGRA2 data 格式（每次探空一个头行 `#` + 逐层）:
  头行  ID 12 位 / YEAR 14-17 / MONTH 19-20 / DAY 22-23 / HOUR 25-26 / NUMLEV 33-36
  层行  PRESS 10-15(Pa) / GPH 17-21(m) / TEMP 23-27(℃*10) / DPDP 37-41(℃*10)
        缺测为 -9999/-8888
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import urllib.request
import zipfile
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
import stations as _S                      # noqa: E402

# 机场 -> 最近的**现役**探空站（IGRA 站号）。2026-08-12 从 IGRA 站表按
# 「末年 >= 2024 且记录数 > 5000」筛出现役站后取最近的一个，距离 21-52 km。
# 深圳配的是香港京士柏 45004（52km）—— 与 WU 的流浮山同区，反倒比宝安更贴近
# 我们的训练标签（见 wu_obs.py 的说明）。
SITE = {   # stations-ok: 机场->探空站映射不是站点清单，缺站走 SITE 缺失分支
    "ZBAA": ("CHM00054511", 30),   # 北京南郊
    "ZGGG": ("CHM00059280", 42),   # 清远
    "ZGSZ": ("HKM00045004", 52),   # 香港京士柏
    "ZHCC": ("CHM00057083", 28),   # 郑州
    "ZHHH": ("CHM00057494", 25),   # 武汉
    "ZSJN": ("CHM00054727", 34),   # 章丘
    "ZSPD": ("CHM00058362", 46),   # 上海宝山
    "ZSQD": ("CHM00054857", 40),   # 青岛
    "ZUCK": ("CHM00057516", 23),   # 沙坪坝
    "ZUUU": ("CHM00056187", 21),   # 温江
}

MISS = (-9999, -8888, -99999)


# ------------------------------------------------------------ 解析

def _igra_levels(block: list[str]):
    """IGRA2 层行 -> profile。只保留气压/温度都有的层。"""
    out = []
    for l in block:
        try:
            p = int(l[9:15]); g = int(l[16:21])
            t = int(l[22:27]); dpdp = int(l[36:41])
        except (ValueError, IndexError):
            continue
        if p in MISS or t in MISS:
            continue
        temp = t / 10.0
        dew = None if dpdp in MISS else temp - dpdp / 10.0
        out.append((p / 100.0, None if g in MISS else float(g), temp, dew))
    return out


def read_igra(path: str, hour: int = 0, since: str = "2015-01-01"):
    """读一个站的 IGRA zip，返回 {日期: profile}（只取指定 UTC 时次）。"""
    got = {}
    z = zipfile.ZipFile(path)
    with z.open(z.namelist()[0]) as f:
        cur_key, block = None, []
        for raw in f:
            if raw.startswith(b"#"):
                if cur_key and block:
                    got[cur_key] = _igra_levels(block)
                block = []
                l = raw.decode("utf-8", "replace")
                try:
                    y, m, d, h = (int(l[13:17]), int(l[18:20]),
                                  int(l[21:23]), int(l[24:26]))
                except ValueError:
                    cur_key = None
                    continue
                k = f"{y:04d}-{m:02d}-{d:02d}"
                cur_key = k if (h == hour and k >= since) else None
            elif cur_key:
                block.append(raw.decode("utf-8", "replace"))
        if cur_key and block:
            got[cur_key] = _igra_levels(block)
    return {k: v for k, v in got.items() if len(v) >= 8}


def fetch_wyoming(igra_id: str, d: date, hour: int = 0, timeout: int = 40):
    """预测端: 怀俄明实时接口。解析成与 read_igra 完全相同的 profile 结构。"""
    stn = igra_id[-5:]                     # CHM00054511 -> 54511
    u = (f"https://weather.uwyo.edu/wsgi/sounding?datetime={d.isoformat()}"
         f"%20{hour:02d}:00:00&id={stn}&src=UNKNOWN&type=TEXT:LIST")
    txt = urllib.request.urlopen(
        urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}),
        timeout=timeout).read().decode("utf-8", "replace")
    out = []
    for l in txt.splitlines():
        p = l.split()
        if len(p) < 4:
            continue
        try:
            pres, hgt, temp, dew = (float(p[0]), float(p[1]),
                                    float(p[2]), float(p[3]))
        except ValueError:
            continue
        if not (1 <= pres <= 1100):
            continue
        out.append((pres, hgt, temp, dew))
    return out if len(out) >= 8 else []


# ------------------------------------------------------------ 特征

def _theta(p, t):
    """位温 (K)。"""
    return (t + 273.15) * (1000.0 / p) ** 0.286


# 固定气压网格。**两个数据源必须先插值到同一张网格再算派生量。**
# 2026-08-12 的双源比对暴露的问题: 香港京士柏 45004 在怀俄明是全分辨率
# （2292 层），IGRA 只存 18-21 个标准层；其余站怀俄明 100-193 层、IGRA
# 143-277 层。同一次探空、两种垂直分辨率，逆温识别和混合层计算就会给出
# 不同的值（实测 mix_inv 差 449 度、inv_dt 差 38 度）。插值到统一网格后
# 分辨率差异被抹平。
#
# 只到 700 hPa —— 日最高温由边界层决定，再往上与地面升温关系很弱。
GRID = [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700]


def _interp(prof):
    """插值到 GRID。返回 [(气压, 温度, 露点)]，只保留在廓线气压范围内的层。
    另返回地面层 (p0, t0, d0)。**高度一律不用** —— IGRA 很多层缺 GPH，
    用它算出来的 inv_hgt / lapse 两端对不上（20 个站日里 9 个）。"""
    prof = sorted(prof, key=lambda r: -r[0])
    if len(prof) < 8:
        return [], None
    p0, _, t0, d0 = prof[0]
    ps = [r[0] for r in prof]
    out = []
    for g in GRID:
        if g > ps[0] or g < ps[-1]:
            continue
        i = next((k for k in range(1, len(ps)) if ps[k] <= g), None)
        if i is None:
            continue
        a, b = prof[i - 1], prof[i]
        span = math.log(a[0]) - math.log(b[0])
        w = 0.0 if span <= 0 else (math.log(a[0]) - math.log(g)) / span
        t = a[2] + (b[2] - a[2]) * w
        d = (None if (a[3] is None or b[3] is None)
             else a[3] + (b[3] - a[3]) * w)
        out.append((float(g), t, d))
    return out, (p0, t0, d0)


def feats(prof) -> dict:
    """由 profile 算派生量。**训练端与预测端共用这一个函数。**

    所有特征只依赖固定气压网格，**不依赖「最低观测层」** —— 2026-08-12 的
    双源比对里，济南 08-09 的 IGRA 记录底部缺了约 20 hPa（最低层 960.8 而
    怀俄明有 980.5），把最低层当地面就整体偏 2.3 度。改成一律以 1000 hPa
    为基准后这类缺口不再传导。各站的常数偏移由模型自己学。

    地面温度/露点差也删了 —— 机场自己的 METAR 实测比 20-50 公里外探空站的
    地面层准得多，临近侧本来就有。
    """
    f = {k: None for k in FEAT_NAMES}
    lv, _ = _interp(prof)
    if not lv:
        return f
    at = {int(p): (t, d) for p, t, d in lv}

    # 混合层法，以 1000 hPa 位温表述: 整层混合到该层时能达到的温度
    for lvl, key in ((925, "sd_th925"), (850, "sd_th850")):
        if lvl in at:
            f[key] = _theta(lvl, at[lvl][0]) - 273.15
    for lvl, key in ((850, "sd_t850"), (700, "sd_t700")):
        if lvl in at:
            f[key] = at[lvl][0]

    # 逆温: 网格上自下而上第一次温度回升。厚度用气压计量
    for i in range(1, len(lv)):
        if lv[i][1] > lv[i - 1][1] + 0.1:
            j = i
            while j + 1 < len(lv) and lv[j + 1][1] >= lv[j][1]:
                j += 1
            f["sd_inv_dt"] = lv[j][1] - lv[i - 1][1]
            f["sd_inv_dp"] = lv[i - 1][0] - lv[j][0]
            f["sd_th_inv"] = _theta(lv[j][0], lv[j][1]) - 273.15
            break

    if 1000 in at and 850 in at:
        f["sd_lapse_p"] = (at[1000][0] - at[850][0]) / 150.0 * 100.0
    dpd = [t - d for p, t, d in lv if p >= 850 and d is not None]
    if dpd:
        f["sd_dpd_low"] = sum(dpd) / len(dpd)
    return f


FEAT_NAMES = ["sd_th925", "sd_th850", "sd_th_inv", "sd_inv_dt", "sd_inv_dp",
              "sd_lapse_p", "sd_dpd_low", "sd_t850", "sd_t700"]


# ------------------------------------------------------------ CLI

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="从 sonde/ 的 IGRA zip 建表")
    ap.add_argument("--live", help="实时取一个站（ICAO）")
    ap.add_argument("--date", default="")
    ap.add_argument("--since", default="2015-01-01")
    ap.add_argument("--out", default="sonde.csv")
    a = ap.parse_args()

    if a.live:
        d = (date.fromisoformat(a.date) if a.date else date.today())
        sid = SITE[a.live][0]
        prof = fetch_wyoming(sid, d)
        print(f"  {a.live} <- {sid}  {d}  00Z  {len(prof)} 层")
        for k, v in feats(prof).items():
            print(f"    {k:<14}{'--' if v is None else f'{v:.2f}'}")
        return 0

    if a.build:
        rows = []
        for icao, (sid, km) in sorted(SITE.items()):
            p = f"sonde/{sid}-data.txt.zip"
            if not os.path.exists(p):
                print(f"  [warn] 缺 {p}", file=sys.stderr)
                continue
            got = read_igra(p, hour=0, since=a.since)
            for d, prof in sorted(got.items()):
                rows.append({"station": icao, "date": d, **feats(prof)})
            print(f"  {icao} <- {sid} ({km}km): {len(got)} 天", file=sys.stderr)
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["station", "date"] + FEAT_NAMES)
            w.writeheader()
            w.writerows(rows)
        print(f"\n探空特征表已存 {a.out}: {len(rows)} 行")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

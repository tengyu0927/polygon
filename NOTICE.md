# 第三方来源与致谢

## 代码

`merge_mos.py` 里的 `deb_weights()` / `deb_columns()` 实现的
**Dynamic Error Balancing (DEB)** 算法，参照
[PolyWeather](https://github.com/yangyuan-zhen/PolyWeather)（**AGPL-3.0**）的
`src/analysis/deb_algorithm.py` 写成，沿用了它的核心设计：
按各模式近期误差倒数加权、误差指数衰减（0.85）、signed bias 绝对值进分母惩罚
（系数 0.5）、成员分歧超阈值时向等权回退。

**因此本仓库整体采用 AGPL-3.0**（见 LICENSE）。若要改用更宽松的协议，
必须先移除 DEB 相关代码与特征列，并重训模型 ——
实测代价是 D+1 从 1.00 退到 1.08、D+2 从 1.14 退到 1.21。

## 数据

| 来源 | 内容 | 许可 |
|---|---|---|
| [Open-Meteo](https://open-meteo.com/) | GFS / ECMWF IFS / CMA GRAPES / ICON / JMA GSM / GEM 的固定时效预报 | CC BY 4.0 |
| [Iowa Environmental Mesonet (ISU)](https://mesonet.agron.iastate.edu/ASOS/) | ASOS/METAR 逐时实况归档 | 公开数据 |
| [NOAA Aviation Weather Center](https://aviationweather.gov/) | 实时 METAR 与 TAF 报文 | 美国政府作品，公有领域 |

数据本身不随仓库分发，由 `bootstrap.sh` 从上述来源重建。
使用 Open-Meteo 数据时请保留 CC BY 4.0 署名。

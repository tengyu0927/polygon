#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tablefmt.py — 终端表格对齐的**唯一真相源**。中文是双宽的。

    import tablefmt as F
    F.w("北京首都")        # 8，不是 4
    F.L("站点", 16)        # 左对齐到 16 显示列
    F.R("34", 7)           # 右对齐到 7 显示列

为什么要这个文件:

f-string 的 `:<14` / `:>7` 按**字符数**补空格，终端按**显示列**排版。
汉字算 1 个字符却占 2 列，于是表头 `{'站点':<14}` 实占 16 列、数据行
`{'北京首都':<9}` 实占 18 列 —— 每张表都歪，而且歪多少取决于那一格里
有几个汉字。2026-08-22 清点，cron_hourly/cron_daily 里 8 张表**全部**中招。

规则: 凡是要对齐的格子，一律 `L()`/`R()`，**不要**再用 f-string 的宽度。
表头和数据行必须共用同一组宽度常量，别两边各写各的。

`✓ ↑ ⚠ ℃ ×` 这些是 East-Asian Ambiguous，这里按 1 列算 —— 与 iTerm2 /
macOS Terminal / VS Code 的默认设置一致（终端若设成「歧义字符按双宽」
会歪，但那不是本项目的运行环境）。
"""

from __future__ import annotations

import unicodedata


def w(s: str) -> int:
    """字符串在终端里占几列。W/F（宽/全角）算 2，其余算 1。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def L(s, n: int) -> str:
    """左对齐到 n 显示列。超宽不截断（宁可推歪一格，也别把数字切掉）。"""
    s = str(s)
    return s + " " * max(0, n - w(s))


def R(s, n: int) -> str:
    """右对齐到 n 显示列。"""
    s = str(s)
    return " " * max(0, n - w(s)) + s

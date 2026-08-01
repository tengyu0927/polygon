#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_tmax.py  —  多站点气温实时监看（不落盘，全在内存）

先说清楚一件事
    **这些站点没有 1 分钟分辨率的数据。**
    ZSPD 等民航机场发的是 METAR，例行报半小时一次（:00 / :30），
    天气剧烈变化时才临时加发 SPECI。上游数据源本身也是这个节奏：
    aviationweather.gov 的 METAR 缓存虽然每分钟刷新一次，
    但刷新的是"文件"，不是"观测"——ZSPD 半小时内不会有新观测。

    所以每分钟拉一次，你会连续看到 30 次一模一样的温度。
    脚本因此做了两件事：
      - 每行显示观测的"年龄"(age)，一眼看出这个数是几分钟前的
      - 只有 obsTime 变化时才标记 NEW，并触发日最高温度的更新
    真想要 1 分钟数据，只能上自建自动气象站，公开的 METAR 源没有。

数据源
    https://aviationweather.gov/api/data/metar?ids=ZSPD,ZBAA,...&format=json
    美国 NWS 的公开航空气象 API，支持逗号分隔一次查多站，
    所以 8 个站只需要 1 次请求。官方要求：设置自定义 User-Agent，
    请求间留间隔，每分钟最多 100 次。本脚本默认 60 秒 1 次，远低于上限。

关于"最高温度"
    API 返回的 maxT / maxT24 字段来自 METAR 备注里的温度组，
    中国的 METAR 通常不带这些组，所以基本是 null。
    脚本因此自行累计**当日实测最高值**：从启动开始，每收到一条新观测
    就更新一次。注意这是"启动以来的最高值"，不是完整的日最高，
    早上启动才能覆盖全天。按北京时 0 点自动重置。

用法
    python live_tmax.py                          # 默认 8 个站，60 秒刷新
    python live_tmax.py --interval 300           # 5 分钟一次（更合理）
    python live_tmax.py --stations ZSPD,ZBAA
    python live_tmax.py --once                   # 只打一次就退出
    python live_tmax.py --plain                  # 逐行追加，不清屏（便于重定向）

依赖：仅标准库（Python 3.7+）
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")

API = "https://aviationweather.gov/api/data/metar"
UA = "live-tmax-monitor/1.0 (station monitoring; contact: local user)"

CN_TZ = timezone(timedelta(hours=8))

# ZSQD 以前漏在这里，而它恰好常是各方法分歧最大的站，晚上核对时看不到它
DEFAULT_STATIONS = ["ZSPD", "ZUUU", "ZGSZ", "ZGGG", "ZSQD",
                    "ZUCK", "ZBAA", "ZHHH",]

CN_NAME = {
    "ZSPD": "上海浦东", "ZUUU": "成都双流", "ZGSZ": "深圳宝安",
    "ZGGG": "广州白云", "ZUCK": "重庆江北", "ZBAA": "北京首都",
    "ZSQD": "青岛胶东",
    "ZHHH": "武汉天河",
}

# ANSI 颜色，--no-color 或非 tty 时自动关闭
class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    BLUE = "\033[34m"; CYAN = "\033[36m"

    @classmethod
    def off(cls):
        for k in list(vars(cls)):
            if k.isupper():
                setattr(cls, k, "")


# --------------------------------------------------------------------------

def fetch(stations, timeout=20, retries=3, hours=None):
    """
    请求站点 METAR。
    hours=None  取各站最新一条，返回 {icao: obs}
    hours=N     取最近 N 小时的全部报文，返回 {icao: [obs, ...]}
    """
    q = {"ids": ",".join(stations), "format": "json"}
    if hours:
        q["hours"] = hours
    url = API + "?" + urllib.parse.urlencode(q)
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
            break
        except Exception as e:                       # noqa: BLE001
            last = e
            if i < retries - 1:
                time.sleep(3 * (i + 1))
    else:
        raise RuntimeError("请求失败：%s" % last)

    if isinstance(data, dict):
        data = data.get("data", [])

    out = {}
    for o in data or []:
        sid = (o.get("icaoId") or "").upper()
        if not sid:
            continue
        if hours:
            out.setdefault(sid, []).append(o)
        # 同一站可能返回多条，保留观测时间最新的
        elif sid not in out or (o.get("obsTime") or 0) > (out[sid].get("obsTime") or 0):
            out[sid] = o

    if hours:
        for sid in out:
            out[sid].sort(key=lambda x: x.get("obsTime") or 0)

    # 深圳必须换成 WU 的序列 —— 见下方 WU_STATIONS 说明
    for sid in WU_STATIONS & set(stations):
        try:
            wu = fetch_wu(sid, hours or 1)
        except Exception as e:                       # noqa: BLE001
            sys.stderr.write("  ! %s 取 WU 失败(%s)，这一行仍是宝安 METAR，"
                             "与 WU 页面会对不上\n" % (sid, e))
            continue
        if not wu:
            continue
        out[sid] = wu if hours else wu[-1]
    return out


# 这些站看的是 WU 的序列，不是 AWC 的 METAR。
# WU 的 ZGSZ:9:CN 返回的是 Lau Fau Shan（香港流浮山，WMO 45035），
# 离深圳宝安 30km。最终对错以 WU 为准，模型也训练在这条序列上（见 wu_obs.py），
# 所以这个监看窗口必须跟着换，否则你盯的当日最高温和打分用的不是一回事 ——
# 实测 624 天里 70% 的日子对不上，26% 差 >=2℃。
WU_STATIONS = {"ZGSZ"}


def fetch_wu(sid, hours):
    """取 WU 最近 hours 小时的观测，转成与 AWC 相同的字段名。"""
    import wu_obs as WO
    now = datetime.now(CN_TZ)
    days = {(now - timedelta(hours=h)).strftime("%Y%m%d")
            for h in (0, max(0, hours))}
    out = []
    for d in sorted(days):
        for x in WO.wu_obs(sid, d, d, retries=2):
            if x.get("temp") is None:
                continue
            ts = x["valid_time_gmt"]
            if ts < now.timestamp() - hours * 3600 - 60:
                continue
            out.append({"icaoId": sid, "obsTime": ts, "temp": float(x["temp"]),
                        "dewp": x.get("dewPt"),
                        "rawOb": "WU %s %s" % (x.get("obs_name") or "",
                                               x.get("wx_phrase") or "")})
    out.sort(key=lambda x: x["obsTime"])
    return out


def local_now():
    return datetime.now(CN_TZ)


def fmt_obs_time(epoch):
    if not epoch:
        return "--:--"
    return datetime.fromtimestamp(int(epoch), tz=CN_TZ).strftime("%H:%M")


# --------------------------------------------------------------------------

class Tracker:
    """内存里维护每站的当日最高温度，北京时 0 点重置。"""

    def __init__(self):
        self.day = local_now().strftime("%Y-%m-%d")
        self.state = {}     # sid -> dict
        self.seeded = False

    def maybe_rollover(self):
        today = local_now().strftime("%Y-%m-%d")
        if today != self.day:
            self.day = today
            for s in self.state.values():
                s["tmax"] = None
                s["tmax_at"] = None
            self.seeded = False
            return True
        return False

    def seed_today(self, stations, verbose=True):
        """
        启动时回补：把北京时今日 0 点至今的观测全部拉回来，
        算出**真正的**当日最高温度，而不是"启动以来"的。
        仍然不落盘，只在内存里。
        """
        now = local_now()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        hours = (now - midnight).total_seconds() / 3600.0
        # 至少回查 1 小时；API 上限按 15 天，24 小时完全在范围内
        hours = max(1.0, min(round(hours + 0.5, 1), 24.0))

        try:
            hist = fetch(stations, hours=hours)
        except Exception as e:                        # noqa: BLE001
            if verbose:
                sys.stderr.write("  ! 回补今日观测失败(%s)，"
                                 "最高温度将从本次启动开始累计\n" % e)
            return False

        cut = midnight.timestamp()
        n_used = 0
        for sid in stations:
            st = self.state.setdefault(sid, {
                "temp": None, "prev_temp": None, "obs_time": None,
                "tmax": None, "tmax_at": None, "raw": None, "is_new": False,
            })
            for o in hist.get(sid, []):
                ot, t = o.get("obsTime"), o.get("temp")
                if ot is None or t is None or ot < cut:
                    continue
                n_used += 1
                if st["tmax"] is None or t > st["tmax"]:
                    st["tmax"] = t
                    st["tmax_at"] = ot
                # 同时把最后一条当作当前观测，避免首屏空白
                if st["obs_time"] is None or ot > st["obs_time"]:
                    st["prev_temp"] = st["temp"]
                    st["temp"] = t
                    st["obs_time"] = ot
                    st["raw"] = o.get("rawOb")

        self.seeded = True
        if verbose:
            print("已回补今日 %.1f 小时内的 %d 条观测，"
                  "最高温度为真实日最高值。\n" % (hours, n_used))
        return True

    def update(self, sid, obs):
        st = self.state.setdefault(sid, {
            "temp": None, "prev_temp": None, "obs_time": None,
            "tmax": None, "tmax_at": None, "raw": None, "is_new": False,
        })
        ot = obs.get("obsTime")
        temp = obs.get("temp")
        st["is_new"] = (ot is not None and ot != st["obs_time"])

        if st["is_new"]:
            st["prev_temp"] = st["temp"]
            st["obs_time"] = ot
            st["temp"] = temp
            st["raw"] = obs.get("rawOb")
            if temp is not None and (st["tmax"] is None or temp > st["tmax"]):
                st["tmax"] = temp
                st["tmax_at"] = ot
        return st


def render(tracker, stations, color=True, plain=False):
    now = local_now()
    lines = []
    head = ("%-6s %-9s %-6s %6s %7s %8s %8s  %s"
            % ("站点", "名称", "观测", "龄min", "气温", "今日最高", "最高时刻", ""))
    lines.append(C.BOLD + head + C.RESET)
    lines.append("-" * 78)

    for sid in stations:
        st = tracker.state.get(sid)
        name = CN_NAME.get(sid, "")
        if not st or st["obs_time"] is None:
            lines.append("%-6s %-9s %-6s %6s %7s %8s %8s  %s"
                         % (sid, name, "--:--", "-", "-", "-", "-",
                            C.DIM + "无数据" + C.RESET))
            continue

        age = int((now.timestamp() - st["obs_time"]) / 60)
        t = st["temp"]
        tmax = st["tmax"]

        # 观测超过 90 分钟没更新，多半是上游断了
        age_s = "%6d" % age
        if age > 90:
            age_s = C.RED + age_s + C.RESET
        elif age > 45:
            age_s = C.YELLOW + age_s + C.RESET

        temp_s = "%6.1f°" % t if t is not None else "     -"
        if st["is_new"]:
            temp_s = C.GREEN + C.BOLD + temp_s + C.RESET

        tmax_s = "%7.1f°" % tmax if tmax is not None else "      -"
        if tmax is not None and t is not None and abs(t - tmax) < 0.05:
            tmax_s = C.RED + tmax_s + C.RESET      # 当前就是今日最高

        # 相对上一条观测的变化
        trend = ""
        if st["prev_temp"] is not None and t is not None:
            d = t - st["prev_temp"]
            if abs(d) >= 0.05:
                arrow = "↑" if d > 0 else "↓"
                col = C.RED if d > 0 else C.BLUE
                trend = "%s%s%+.1f%s" % (col, arrow, d, C.RESET)

        flag = C.GREEN + "NEW" + C.RESET if st["is_new"] else ""

        lines.append("%-6s %-9s %-6s %s %s %s %8s  %s %s"
                     % (sid, name, fmt_obs_time(st["obs_time"]), age_s,
                        temp_s, tmax_s, fmt_obs_time(st["tmax_at"]),
                        trend, flag))

    lines.append("-" * 78)
    note = ("最高温度为今日 0 时起真实值" if getattr(tracker, "seeded", False)
            else "最高温度仅为本次启动以来累计")
    lines.append(C.DIM
                 + "日期 %s (北京时)  刷新于 %s   %s"
                 % (tracker.day, now.strftime("%H:%M:%S"), note) + C.RESET)
    lines.append(C.DIM
                 + "METAR 例行报半小时一次，龄>45min 变黄、>90min 变红即上游异常"
                 + C.RESET)
    return "\n".join(lines)


# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="多站点气温与当日最高温度实时监看",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stations", default=",".join(DEFAULT_STATIONS))
    p.add_argument("--interval", type=int, default=60,
                   help="刷新间隔秒数。注意上游观测半小时才更新一次，"
                        "设成 60 只是让年龄显示更实时，不会拿到更多数据")
    p.add_argument("--once", action="store_true", help="只取一次就退出")
    p.add_argument("--plain", action="store_true",
                   help="不清屏，逐次追加输出（便于 tee / 重定向）")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--raw", action="store_true", help="同时打印原始 METAR 报文")
    p.add_argument("--no-seed", action="store_true",
                   help="不回补今日历史，最高温度只从启动时刻开始累计")
    args = p.parse_args(argv)

    if args.no_color or not sys.stdout.isatty():
        C.off()

    stations = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    if args.interval < 30 and not args.once:
        sys.stderr.write(
            "  ! 间隔小于 30 秒没有意义（上游半小时才有新观测），已调整为 30\n")
        args.interval = 30

    tracker = Tracker()
    print("监看 %d 个站：%s" % (len(stations), ",".join(stations)))
    print("提示：METAR 例行报每 30 分钟一次，因此多数刷新会看到相同数值。\n")

    if not args.no_seed:
        tracker.seed_today(stations)

    try:
        while True:
            if tracker.maybe_rollover() and not args.no_seed:
                tracker.seed_today(stations, verbose=False)
            try:
                obs = fetch(stations)
                for sid in stations:
                    if sid in obs:
                        tracker.update(sid, obs[sid])
                    elif sid in tracker.state:
                        tracker.state[sid]["is_new"] = False
                err = None
            except Exception as e:                    # noqa: BLE001
                err = str(e)

            if not args.plain and not args.once:
                os.system("cls" if os.name == "nt" else "clear")
            print(render(tracker, stations, plain=args.plain))
            if err:
                print(C.RED + "  取数失败：%s（下次重试）" % err + C.RESET)
            if args.raw:
                for sid in stations:
                    st = tracker.state.get(sid)
                    if st and st.get("raw"):
                        print(C.DIM + "  " + st["raw"] + C.RESET)

            if args.once:
                break
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止。")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
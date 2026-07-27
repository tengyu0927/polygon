#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iem_multi.py  —  多站点批量下载（IEM ASOS 归档）

相对 iem_asos.py 的改进
    1. 站点发现：从 IEM 的 network geojson 自动拉取站表（含经纬度、海拔、
       资料起止年），不用手工找 ICAO 码；
    2. 支持按经纬度框（--bbox）筛选，方便一次性取长三角 / 浙江 / 江西等区域；
    3. 关键：IEM 的 asos.py 接口**支持一次请求多个站点**。
       IEM 官方文档明确说明该服务有每 IP 1 秒的限流，并建议
       "一次把所有站点一起请求完，不要逐站高频访问"。
       所以本脚本按 (站点批 x 年份) 组织请求，而不是逐站循环；
    4. 站点元数据单独入库（经纬度、海拔、时区偏移），
       海拔是后续做 GFS 格点-站点地形订正的必需特征；
    5. 用站表里的 archive_begin / archive_end 精确界定抓取年份区间，
       不需要盲目往前试探。

依赖
    仅标准库（Python 3.7+）；导出 parquet 时需要 pandas + pyarrow。

用法
    # 1) 先看看中国有哪些站（写入 stations 表并打印）
    python iem_multi.py --db cn.sqlite --list-network CN__ASOS

    # 2) 长三角区域全部站点，全时段下载
    python iem_multi.py --db cn.sqlite --network CN__ASOS \
        --bbox 118.0,28.0,123.0,33.5 --backfill

    # 3) 指定站点
    python iem_multi.py --db cn.sqlite --stations ZSPD,ZSSS,ZSNJ,ZSHC,ZSOF --backfill

    # 4) 从文件读站点列表（每行一个）
    python iem_multi.py --db cn.sqlite --stations-file sites.txt --backfill

    # 5) 日常增量：重抓今年+去年（覆盖迟到报和订正报）
    python iem_multi.py --db cn.sqlite --stations-file sites.txt --update

    # 6) 生成逐日表并导出
    python iem_multi.py --db cn.sqlite --daily --export-daily cn_daily.parquet
"""

import argparse
import csv
import io
import json
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_GEOJSON = "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
USER_AGENT = "multi-station-archive-fetch/1.0 (research use)"

# IEM 的国际 ASOS 网络命名规则：{两位国家码}__ASOS（双下划线）
# 中国 CN__ASOS，日本 JP__ASOS，韩国 KR__ASOS，等等。
DEFAULT_NETWORK = "CN__ASOS"

MISSING = {"M", "", "None", "null"}


# --------------------------------------------------------------------------
# 单位换算
# --------------------------------------------------------------------------

def f2c(v):
    return None if v is None else round((v - 32.0) * 5.0 / 9.0, 2)


def kt2ms(v):
    return None if v is None else round(v * 0.514444, 2)


def in2mm(v):
    return None if v is None else round(v * 25.4, 2)


def mi2km(v):
    return None if v is None else round(v * 1.609344, 2)


def num(s, trace_value=0.0):
    if s is None:
        return None
    s = s.strip()
    if s == "T":
        return trace_value
    if s in MISSING:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def txt(s):
    if s is None:
        return None
    s = s.strip()
    return None if s in MISSING else s


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

def http_get(url, retries=5, timeout=300, throttle=1.2):
    """IEM 有每 IP 1 秒限流，throttle 默认留一点余量。503 要退避重试。"""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read().decode("utf-8", errors="replace")
            time.sleep(throttle)
            return data
        except Exception as e:                      # noqa: BLE001
            last = e
            code = getattr(e, "code", None)
            # 503 = 服务器繁忙，按文档要求退避
            wait = 30 if code == 503 else 5 * (attempt + 1)
            if attempt < retries - 1:
                sys.stderr.write("  ! 请求失败(%s)，%ds 后重试 %d/%d\n"
                                 % (e, wait, attempt + 2, retries))
                time.sleep(wait)
    raise RuntimeError("请求失败：%s\nURL: %s" % (last, url))


# --------------------------------------------------------------------------
# 建库
# --------------------------------------------------------------------------

OBS_COLS = ["station", "valid_time_gmt", "obs_time_utc", "local_date",
            "temp_c", "dewp_c", "rh", "drct", "wspd_ms", "gust_ms",
            "pres_hpa", "precip_mm", "vis_km", "skyc1", "skyl1",
            "wxcodes", "feel_c", "metar"]


def init_db(conn):
    conn.executescript("""
    PRAGMA journal_mode = WAL;

    -- 站点元数据：海拔是后续 GFS 地形订正的必需特征
    CREATE TABLE IF NOT EXISTS stations (
        station       TEXT PRIMARY KEY,
        name          TEXT,
        network       TEXT,
        lat           REAL,
        lon           REAL,
        elev_m        REAL,
        tz_offset_h   REAL NOT NULL DEFAULT 8,
        archive_begin TEXT,
        archive_end   TEXT,
        updated_at    TEXT
    );

    CREATE TABLE IF NOT EXISTS obs (
        station        TEXT NOT NULL,
        valid_time_gmt INTEGER NOT NULL,
        obs_time_utc   TEXT NOT NULL,
        local_date     TEXT NOT NULL,
        temp_c REAL, dewp_c REAL, rh REAL,
        drct REAL, wspd_ms REAL, gust_ms REAL,
        pres_hpa REAL, precip_mm REAL, vis_km REAL,
        skyc1 TEXT, skyl1 REAL, wxcodes TEXT, feel_c REAL,
        metar TEXT,
        PRIMARY KEY (station, valid_time_gmt)
    );
    CREATE INDEX IF NOT EXISTS idx_obs_sd ON obs(station, local_date);

    -- 抓取进度：按 (站, 年) 记录，支持任意中断后续抓
    CREATE TABLE IF NOT EXISTS fetch_log (
        station    TEXT NOT NULL,
        year       INTEGER NOT NULL,
        n_obs      INTEGER NOT NULL,
        fetched_at TEXT NOT NULL,
        PRIMARY KEY (station, year)
    );
    """)
    conn.commit()


# --------------------------------------------------------------------------
# 站点发现
# --------------------------------------------------------------------------

def load_network(conn, network, tz_offset=8.0, throttle=1.2):
    """
    从 IEM 拉取某网络的站表并写入 stations 表。
    geojson 的 properties 通常包含 sid / sname / elevation /
    archive_begin / archive_end，字段名如有变化可用 --dump-props 查看。
    """
    url = IEM_GEOJSON.format(network=urllib.parse.quote(network))
    gj = json.loads(http_get(url, throttle=throttle))
    feats = gj.get("features", [])
    rows = []
    for f in feats:
        p = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        coords = geom.get("coordinates") or [None, None]
        sid = p.get("sid") or f.get("id")
        if not sid:
            continue
        rows.append((
            sid,
            p.get("sname") or p.get("name"),
            network,
            coords[1], coords[0],
            p.get("elevation"),
            tz_offset,
            p.get("archive_begin"),
            p.get("archive_end"),
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        ))

    conn.executemany(
        "INSERT OR REPLACE INTO stations "
        "(station,name,network,lat,lon,elev_m,tz_offset_h,"
        " archive_begin,archive_end,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows)
    conn.commit()
    print("[network] %s：入库 %d 个站点" % (network, len(rows)))
    return [r[0] for r in rows]


def select_stations(conn, args):
    """按 --stations / --stations-file / --network(+--bbox) 决定要抓哪些站。"""
    if args.stations:
        sids = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    elif args.stations_file:
        with open(args.stations_file, encoding="utf-8") as f:
            sids = [ln.split("#")[0].strip().upper()
                    for ln in f if ln.split("#")[0].strip()]
    elif args.network:
        sql = "SELECT station FROM stations WHERE network=?"
        params = [args.network]
        if args.bbox:
            x1, y1, x2, y2 = [float(v) for v in args.bbox.split(",")]
            sql += " AND lon BETWEEN ? AND ? AND lat BETWEEN ? AND ?"
            params += [min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)]
        sql += " ORDER BY station"
        sids = [r[0] for r in conn.execute(sql, params)]
    else:
        sids = [r[0] for r in conn.execute(
            "SELECT station FROM stations ORDER BY station")]
    return sids


def station_tz(conn, sids, default=8.0):
    tz = {}
    for sid in sids:
        row = conn.execute(
            "SELECT tz_offset_h FROM stations WHERE station=?", (sid,)).fetchone()
        tz[sid] = row[0] if row and row[0] is not None else default
    return tz


def station_year_range(conn, sid, fallback_start, fallback_end):
    """用站表里的 archive_begin/end 确定年份区间；缺失时用 fallback。"""
    row = conn.execute(
        "SELECT archive_begin, archive_end FROM stations WHERE station=?",
        (sid,)).fetchone()
    y0, y1 = fallback_start, fallback_end
    if row:
        if row[0]:
            try:
                y0 = max(y0, int(str(row[0])[:4]))
            except ValueError:
                pass
        if row[1]:
            try:
                y1 = min(y1, int(str(row[1])[:4]))
            except ValueError:
                pass
    return y0, y1


# --------------------------------------------------------------------------
# 下载
# --------------------------------------------------------------------------

def fetch_batch(stations, year, timeout=300, throttle=1.2):
    """一次请求多个站点、整年数据。返回 CSV 文本。"""
    params = [("data", "all"), ("tz", "Etc/UTC"), ("format", "onlycomma"),
              ("latlon", "no"), ("elev", "no"), ("missing", "M"),
              ("trace", "T"), ("direct", "no"),
              ("report_type", "3"), ("report_type", "4"),
              ("sts", "%d-01-01T00:00:00Z" % year),
              ("ets", "%d-01-01T00:00:00Z" % (year + 1))]
    for s in stations:
        params.append(("station", s))
    url = IEM_ASOS + "?" + urllib.parse.urlencode(params)
    return http_get(url, timeout=timeout, throttle=throttle)


def fetch_range(stations, sts, ets, timeout=120, throttle=1.2):
    """只拉一个日期区间。--update 是重抓两整年（8 站约 10 万行），
    每小时跑一次纯属自找麻烦: IEM 慢起来单次要几十分钟，而 urlopen 的 timeout
    只管单次 socket 读、服务器涓流发数据就永远不触发。日常增量用这个。"""
    params = [("data", "all"), ("tz", "Etc/UTC"), ("format", "onlycomma"),
              ("latlon", "no"), ("elev", "no"), ("missing", "M"),
              ("trace", "T"), ("direct", "no"),
              ("report_type", "3"), ("report_type", "4"),
              ("sts", sts), ("ets", ets)]
    for s in stations:
        params.append(("station", s))
    # retries 别设小。国内直连 IEM 常见 "Connection reset by peer"，
    # 单次失败很正常，重试几乎都能成；重试用完才是真失败
    return http_get(IEM_ASOS + "?" + urllib.parse.urlencode(params),
                    retries=5, timeout=timeout, throttle=throttle)


def run_recent(conn, args, sids, days):
    """增量: 只抓最近 days 天，写入现有表。"""
    tzmap = station_tz(conn, sids, args.tz_offset)
    now = datetime.now(timezone.utc)
    sts = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    ets = (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("[recent] 抓最近 %d 天 (%s ~ %s)" % (days, sts[:10], ets[:10]),
          file=sys.stderr)
    total = 0
    for i in range(0, len(sids), args.batch_size):
        batch = sids[i:i + args.batch_size]
        text = fetch_range(batch, sts, ets, timeout=args.timeout,
                           throttle=args.throttle)
        parsed = parse_csv(text, tzmap, args.tz_offset)
        for year, rows in _split_by_year(parsed).items():
            total += store(conn, rows, year)
        conn.commit()
    print("[recent] 共 %d 条" % total, file=sys.stderr)
    return total


def _split_by_year(parsed):
    """store() 按年分表记账，跨年区间要拆开。"""
    out = {}
    for stn, rows in parsed.items():
        for r in rows:
            y = int(datetime.fromtimestamp(r[1], timezone.utc).year)
            out.setdefault(y, {}).setdefault(stn, []).append(r)
    return out


def parse_csv(csv_text, tzmap, default_tz=8.0):
    """解析 IEM CSV，返回 {station: [row, ...]}。"""
    out = {}
    if not csv_text or "station" not in csv_text[:200]:
        return out

    for r in csv.DictReader(io.StringIO(csv_text)):
        sid = txt(r.get("station"))
        valid = txt(r.get("valid"))
        if not sid or not valid:
            continue
        sid = sid.upper()
        try:
            dt_utc = datetime.strptime(valid, "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue

        off = tzmap.get(sid, default_tz)
        local_date = (dt_utc + timedelta(hours=off)).strftime("%Y-%m-%d")

        mslp = num(r.get("mslp"))
        alti = num(r.get("alti"))
        if mslp is None and alti is not None:
            mslp = round(alti * 33.8639, 2)

        out.setdefault(sid, []).append((
            sid,
            int(dt_utc.timestamp()),
            dt_utc.strftime("%Y-%m-%d %H:%M:%S"),
            local_date,
            f2c(num(r.get("tmpf"))),
            f2c(num(r.get("dwpf"))),
            num(r.get("relh")),
            num(r.get("drct")),
            kt2ms(num(r.get("sknt"))),
            kt2ms(num(r.get("gust"))),
            mslp,
            in2mm(num(r.get("p01i"), trace_value=0.01)),
            mi2km(num(r.get("vsby"))),
            txt(r.get("skyc1")),
            num(r.get("skyl1")),
            txt(r.get("wxcodes")),
            f2c(num(r.get("feel"))),
            txt(r.get("metar")),
        ))
    return out


def store(conn, by_station, year):
    sql = "INSERT OR REPLACE INTO obs (%s) VALUES (%s)" % (
        ", ".join(OBS_COLS), ", ".join("?" * len(OBS_COLS)))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    total = 0
    for sid, rows in by_station.items():
        conn.executemany(sql, rows)
        conn.execute("INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?)",
                     (sid, year, len(rows), now))
        total += len(rows)
    return total


def mark_empty(conn, sids, year):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.executemany("INSERT OR REPLACE INTO fetch_log VALUES (?,?,?,?)",
                     [(s, year, 0, now) for s in sids])


def already_done(conn, sid, year):
    row = conn.execute(
        "SELECT n_obs FROM fetch_log WHERE station=? AND year=?",
        (sid, year)).fetchone()
    return row is not None and row[0] > 0


def run_download(conn, args, sids, years, force=False):
    tzmap = station_tz(conn, sids, args.tz_offset)
    total_reqs = 0

    for year in years:
        # 该年还需要抓的站
        todo = [s for s in sids
                if force or not already_done(conn, s, year)]
        if not todo:
            continue

        # 按 archive_begin/end 裁掉不可能有数据的站
        keep = []
        for s in todo:
            y0, y1 = station_year_range(conn, s, args.min_year, args.max_year)
            if y0 <= year <= y1:
                keep.append(s)
            else:
                mark_empty(conn, [s], year)
        if not keep:
            conn.commit()
            continue

        got_year = 0
        for i in range(0, len(keep), args.batch_size):
            batch = keep[i:i + args.batch_size]
            text = fetch_batch(batch, year, timeout=args.timeout,
                               throttle=args.throttle)
            total_reqs += 1
            parsed = parse_csv(text, tzmap, args.tz_offset)
            n = store(conn, parsed, year)
            # 请求里但没返回数据的站，标记为空，避免下次重复请求
            mark_empty(conn, [s for s in batch if s not in parsed], year)
            conn.commit()
            got_year += n
            print("  %d  批 %2d/%2d (%d 站)  %7d 条"
                  % (year, i // args.batch_size + 1,
                     (len(keep) + args.batch_size - 1) // args.batch_size,
                     len(batch), n))
        print("  %d 年小计 %d 条" % (year, got_year))

    print("[download] 共发出 %d 次请求" % total_reqs)


# --------------------------------------------------------------------------
# 逐日聚合 / 导出
# --------------------------------------------------------------------------

def build_daily(conn, day_start_hour=0):
    """
    多站逐日聚合，日界按各站自己的 tz_offset_h。
    day_start_hour=0 -> 当地 00-24；=20 -> 前日 20 时至当日 20 时（国内业务口径）

    再次提醒：METAR 是定时/特选报，据此得到的 Tmax 比连续观测
    系统性偏低约 0.3-0.8 degC。若要真正的日极值，见脚本末尾 GSOD 说明。
    """
    conn.executescript("""
    DROP TABLE IF EXISTS daily;
    CREATE TABLE daily (
        station TEXT NOT NULL, date TEXT NOT NULL,
        tmax REAL, tmin REAL, tmean REAL,
        rh_mean REAL, wspd_mean REAL, pres_mean REAL,
        precip REAL, n_obs INTEGER,
        PRIMARY KEY (station, date)
    );
    """)
    conn.execute("""
    INSERT INTO daily
    SELECT o.station,
           date(datetime(o.valid_time_gmt
                         + CAST(COALESCE(s.tz_offset_h, 8) * 3600 AS INTEGER)
                         - ?, 'unixepoch')) AS d,
           MAX(o.temp_c), MIN(o.temp_c), AVG(o.temp_c),
           AVG(o.rh), AVG(o.wspd_ms), AVG(o.pres_hpa),
           SUM(COALESCE(o.precip_mm, 0)), COUNT(*)
    FROM obs o LEFT JOIN stations s ON s.station = o.station
    WHERE o.temp_c IS NOT NULL
    GROUP BY o.station, d
    """, (day_start_hour * 3600,))
    conn.commit()
    row = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT station), MIN(date), MAX(date) "
        "FROM daily").fetchone()
    print("[daily] %d 站日 / %d 个站，%s ~ %s" % row)


def export_table(conn, table, path):
    ext = os.path.splitext(path)[1].lower()
    order = "station, valid_time_gmt" if table == "obs" else "station, date"
    sql = "SELECT * FROM %s ORDER BY %s" % (table, order)
    if ext in (".parquet", ".pq"):
        import pandas as pd
        pd.read_sql_query(sql, conn).to_parquet(path, index=False)
    else:
        cur = conn.execute(sql)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([d[0] for d in cur.description])
            w.writerows(cur)
    print("[export] %s -> %s" % (table, path))


# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="IEM ASOS 多站点批量下载",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--db", default="stations.sqlite")

    g = p.add_argument_group("站点选择")
    g.add_argument("--list-network", default=None,
                   help="拉取该网络站表并入库后退出，如 CN__ASOS")
    g.add_argument("--network", default=None, help="按网络选站，如 CN__ASOS")
    g.add_argument("--bbox", default=None,
                   help="经纬度框 lonmin,latmin,lonmax,latmax（配合 --network）")
    g.add_argument("--stations", default=None, help="逗号分隔的站点列表")
    g.add_argument("--stations-file", default=None, help="站点列表文件，每行一个")

    g2 = p.add_argument_group("下载")
    g2.add_argument("--backfill", action="store_true", help="全时段回溯下载")
    g2.add_argument("--update", action="store_true", help="重抓今年与去年")
    g2.add_argument("--min-year", type=int, default=1995)
    g2.add_argument("--max-year", type=int, default=None)
    g2.add_argument("--batch-size", type=int, default=15,
                    help="每次请求打包多少个站点")
    g2.add_argument("--tz-offset", type=float, default=8.0,
                    help="默认时区偏移（小时），站表里有值则以站表为准")
    g2.add_argument("--throttle", type=float, default=1.2,
                    help="每次请求后休眠秒数（IEM 限流 1 秒/IP）")
    g2.add_argument("--timeout", type=int, default=300)
    g2.add_argument("--recent-days", type=int, default=0,
                    help="只抓最近 N 天（增量，秒级）。日常每小时跑用这个，"
                         "别用 --update（那是重抓两整年）")

    g3 = p.add_argument_group("产出")
    g3.add_argument("--daily", action="store_true")
    g3.add_argument("--day-start-hour", type=int, default=0)
    g3.add_argument("--export", default=None)
    g3.add_argument("--export-daily", default=None)

    args = p.parse_args(argv)
    this_year = datetime.now(timezone(timedelta(hours=8))).year
    if args.max_year is None:
        args.max_year = this_year

    conn = sqlite3.connect(args.db)
    try:
        init_db(conn)

        if args.list_network:
            sids = load_network(conn, args.list_network, args.tz_offset,
                                args.throttle)
            for row in conn.execute(
                    "SELECT station,name,lat,lon,elev_m,archive_begin,archive_end"
                    " FROM stations WHERE network=? ORDER BY station",
                    (args.list_network,)):
                print("  %-6s %-28s %8.3f %8.3f %7s  %s ~ %s" % (
                    row[0], (row[1] or "")[:28], row[2] or 0, row[3] or 0,
                    row[4], row[5], row[6]))
            print("\n共 %d 个站。用 --bbox 可按区域筛选。" % len(sids))
            return 0

        # 如果指定了 --network 但站表还是空的，先拉站表
        if args.network and not conn.execute(
                "SELECT 1 FROM stations WHERE network=? LIMIT 1",
                (args.network,)).fetchone():
            load_network(conn, args.network, args.tz_offset, args.throttle)

        sids = select_stations(conn, args)
        if (args.backfill or args.update) and not sids:
            print("没有选中任何站点。先跑 --list-network CN__ASOS 看看有哪些站。")
            return 1
        if sids:
            print("[选站] 共 %d 个：%s%s" % (
                len(sids), ",".join(sids[:10]),
                " ..." if len(sids) > 10 else ""))

        if args.recent_days:
            run_recent(conn, args, sids, args.recent_days)
        elif args.update:
            print("[update] 重抓 %d / %d 年 ..." % (this_year, this_year - 1))
            run_download(conn, args, sids, [this_year, this_year - 1], force=True)

        if args.backfill:
            years = list(range(args.max_year, args.min_year - 1, -1))
            print("[backfill] 年份 %d ~ %d ..." % (args.min_year, args.max_year))
            run_download(conn, args, sids, years)

        if args.daily:
            build_daily(conn, args.day_start_hour)
        if args.export:
            export_table(conn, "obs", args.export)
        if args.export_daily:
            export_table(conn, "daily", args.export_daily)

        row = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT station), MIN(local_date), "
            "MAX(local_date) FROM obs").fetchone()
        print("\n[汇总] %s 条 / %s 个站，%s ~ %s" % row)

    except KeyboardInterrupt:
        conn.commit()
        print("\n已中断，进度保存在 fetch_log，重跑同样命令即可续抓。")
        return 130
    finally:
        conn.commit()
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------
# 附：如果需要真正的日最高/最低气温（而不是由 METAR 聚合出来的）
#
# NOAA GSOD（Global Summary of the Day）直接给出每站每日 MAX/MIN 温度，
# 是台站上报的日极值，不受 METAR 采样间隔影响：
#     https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{年}/{USAF}{WBAN}.csv
# 站号对照表（含中国全部台站的 USAF/WBAN、经纬度、海拔、资料起止）：
#     https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv
#
# 另外若需要比机场站更密的站网（中国 ISD 里含大量非机场的 5xxxx 号国家站），
# 用 ISD 全球逐小时：
#     https://www.ncei.noaa.gov/data/global-hourly/access/{年}/{USAF}{WBAN}.csv
# 注意 NCEI 已用 GHCNh 作为 ISD 的下一代替代产品，长期项目建议关注 GHCNh。
# --------------------------------------------------------------------------
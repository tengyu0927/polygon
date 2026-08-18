#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gfs_backfill18.py — 从 NOAA S3 回补 18Z 历史，抽 8 站，出 gfs_extract 同格式的 CSV

    python3 gfs_backfill18.py --start 2026-06-01 --end 2026-07-31 --out gfs18.csv
    python3 gfs_backfill18.py --start 2024-04-01 --end 2026-08-01 --out gfs18.csv --jobs 12
    python3 gfs_backfill18.py --start 2026-07-01 --end 2026-07-03 --dry-run   # 只测速

为什么要 18Z:

18Z（北京时 02:00 起报、约 06:00 落地）能让 9-11 时起报用上 **12 小时时效**，
而现在只能用前一天 12Z 的 18 小时。实测（15 个月滚动回测，本地 GFS 作为
第七个集合成员）:

    起报    第七成员用           Δ MAE      P
     9 时   前一天12Z(18h)     -0.0038    68%   <- 现状，不显著
     9 时   当天00Z(6h)        -0.0249   100%   <- 拿不到，仅作上限
    11 时   前一天12Z(18h)     -0.0157    98%
    11 时   当天00Z(6h)        -0.0249   100%

18Z 的 12h 时效介于两者之间，**是唯一能把 9 时从「不显著」推到显著的办法**。

取数方式与 gfs_live.py 相同: 用 GRIB2 的 .idx 索引 + HTTP Range，
只下需要的那几个变量（约 9-12 MB/档，整文件 500 MB）。
**只落 CSV，不存 GRIB** —— 原始数据不落盘。

时效范围: 18Z 的 f000 = 目标日北京时 02:00，f022 = 24:00。
所以只取 f000~f022（23 档）就能覆盖目标日的白天，不用下到 f048。

断点续跑: 再次运行会读已有 CSV，跳过已完成的日期。
"""

from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UTC = datetime.timezone.utc
CST = datetime.timezone(datetime.timedelta(hours=8))
S3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
UA = "ploygon-nowcast/1.0 (station Tmax research)"

import stations as _S  # 站点清单唯一真相源
STATIONS = _S.GFS_COORD_EXTRACT

# (.idx 的变量段, 层次段) -> 我们的名字。与 gfs_live.WANT 保持一致
WANT = {
    ("TMP", "2 m above ground"): "t2m",
    ("RH", "2 m above ground"): "r2",
    ("UGRD", "10 m above ground"): "u10",
    ("VGRD", "10 m above ground"): "v10",
    ("DSWRF", "surface"): "dswrf",
    ("HPBL", "surface"): "hpbl",
    ("TCDC", "entire atmosphere"): "tcc",
    ("PRES", "surface"): "sp",
    ("PRMSL", "mean sea level"): "prmsl",
}
FH = list(range(0, 23))          # f000~f022 覆盖目标日北京时 02:00~24:00


def http(url, rng=None, retries=3, timeout=90):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            if rng:
                req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:                       # noqa: BLE001
            last = e
            if i < retries - 1:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError(str(last)[:100])


def one_fh(init, fh):
    """下一个时效档里我们要的那几个变量。返回 {var: bytes}。"""
    url = (f"{S3}/gfs.{init:%Y%m%d}/{init:%H}/atmos/"
           f"gfs.t{init:%H}z.pgrb2.0p25.f{fh:03d}")
    try:
        lines = http(url + ".idx", timeout=40).decode().splitlines()
    except Exception:
        return fh, None
    offs = [int(l.split(":")[1]) for l in lines]
    out = {}
    for i, l in enumerate(lines):
        f = l.split(":")
        if len(f) < 5:
            continue
        key = WANT.get((f[3], f[4]))
        if not key or key in out:
            continue
        a = int(f[1])
        b = (offs[i + 1] - 1) if i + 1 < len(offs) else a + 4_000_000
        try:
            out[key] = http(url, (a, b))
        except Exception:
            continue
    return fh, (out or None)


def do_day(init, jobs, ec):
    """下一整轮，抽 8 站，返回 [(station, init, fhour, valid, var, val), ...]"""
    got = {}
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for fh, per in ex.map(lambda h: one_fh(init, h), FH):
            if per:
                got[fh] = per
    if not got:
        return []
    rows = []
    tmp = f"/tmp/_bf18_{os.getpid()}.grb2"
    for fh, per in sorted(got.items()):
        valid = init + datetime.timedelta(hours=fh)
        for key, buf in per.items():
            with open(tmp, "wb") as f:
                f.write(buf)
            with open(tmp, "rb") as f:
                gid = ec.codes_grib_new_from_file(f)
                if gid is None:
                    continue
                try:
                    for s, (la, lo) in STATIONS.items():
                        nr = ec.codes_grib_find_nearest(gid, la, lo % 360.0)
                        if nr and nr[0].value == nr[0].value:
                            rows.append((s, init.isoformat(), fh,
                                         valid.isoformat(), key,
                                         round(float(nr[0].value), 3)))
                finally:
                    ec.codes_release(gid)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="gfs18.csv")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--dry-run", action="store_true", help="只跑头几天测速")
    ap.add_argument("--init", type=int, default=18, choices=[0, 6, 12, 18],
                    help="起报轮次 UTC 小时。2026-08-18 加: 原来写死 18Z，"
                         "而 12-14 时用的是当天 00Z@6h（mos_local6.csv）—— "
                         "那份归档只有 8 站、缺郑州和济南，训练端这两站的 m7 "
                         "恒为空，生产端却实时取到真值，是训练/预测错配")
    a = ap.parse_args()
    try:
        import eccodes as ec
    except ImportError:
        print("[error] 需要 eccodes: pip install eccodes", file=sys.stderr)
        return 2

    d0 = datetime.date.fromisoformat(a.start)
    d1 = datetime.date.fromisoformat(a.end)
    days = [d0 + datetime.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    done = set()
    new = not os.path.exists(a.out)
    if not new:
        try:
            with open(a.out, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    done.add(r["init_utc"][:10])
            print(f"已有 {a.out}，{len(done)} 天已完成，跳过", file=sys.stderr)
        except Exception:
            new = True
    todo = [d for d in days if d.isoformat() not in done]
    if a.dry_run:
        todo = todo[:3]
    if not todo:
        print("没有要处理的。", file=sys.stderr)
        return 0
    print(f"待回补 {len(todo)} 天（每天 {len(FH)} 档 × {len(WANT)} 变量）",
          file=sys.stderr, flush=True)

    fh_out = open(a.out, "a" if not new else "w", newline="", encoding="utf-8")
    w = csv.writer(fh_out)
    if new:
        w.writerow(["station", "init_utc", "fhour", "valid_utc", "var", "val"])
    t0 = time.time()
    n_ok = n_bad = n_row = 0
    for i, d in enumerate(todo, 1):
        init = datetime.datetime.combine(d, datetime.time(a.init), UTC)
        try:
            rows = do_day(init, a.jobs, ec)
        except Exception as e:                       # noqa: BLE001
            print(f"  [warn] {d}: {str(e)[:70]}", file=sys.stderr)
            rows = []
        if rows:
            n_ok += 1
            w.writerows(rows)
            n_row += len(rows)
        else:
            n_bad += 1
        if i % 5 == 0 or i == len(todo):
            fh_out.flush()
            el = time.time() - t0
            eta = el / i * (len(todo) - i)
            print(f"  {i}/{len(todo)}  成功 {n_ok} 失败 {n_bad}  {n_row} 行  "
                  f"用时 {el/60:.0f} 分  预计还要 {eta/3600:.1f} 小时",
                  file=sys.stderr, flush=True)
    fh_out.close()
    print(f"\n完成: {a.out}  {n_row} 行  "
          f"{os.path.getsize(a.out)/1048576:.1f} MB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

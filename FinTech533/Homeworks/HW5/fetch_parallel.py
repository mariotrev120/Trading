"""
Parallel fetcher: run up to N tickers concurrently via ThreadPoolExecutor.
shinybroker opens its own socket per call and each uses a distinct client_id
so concurrency is safe.
"""
from __future__ import annotations

import signal, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from fetch_data import (
    UNIVERSE, MACRO, SECTOR_ETFS,
    DATA_DIR,
    fetch_ticker,
)

WORKERS = 4   # concurrent fetches
PER_TICKER_TIMEOUT = 240  # wall-clock seconds per ticker


def _fetch_one(sym: str, sec: str, exch: str, base_cid: int) -> tuple:
    import shinybroker as sb
    cache = DATA_DIR / f"{sym}.parquet"
    if cache.exists():
        return sym, "skip", "already cached"
    t0 = time.time()
    try:
        df = fetch_ticker(sb, sym, base_cid, sec_type=sec, exchange=exch)
        df.to_parquet(cache, index=False)
        return sym, "ok", f"{len(df)} rows {df['timestamp'].min().date()}..{df['timestamp'].max().date()} ({time.time()-t0:.0f}s)"
    except Exception as e:
        return sym, "fail", f"{type(e).__name__}: {str(e)[:80]} ({time.time()-t0:.0f}s)"


def main() -> int:
    fetch_list = (
        [(s, "STK", "SMART") for s, _ in UNIVERSE]
        + MACRO
        + SECTOR_ETFS
    )
    todo = [(s, sec, exch) for (s, sec, exch) in fetch_list if not (DATA_DIR / f"{s}.parquet").exists()]
    print(f"[parallel] already have {len(fetch_list) - len(todo)} parquets; {len(todo)} to fetch with {WORKERS} workers", flush=True)

    results = []
    base_cid = 12000
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {
            ex.submit(_fetch_one, sym, sec, exch, base_cid + i * 500): (sym, sec)
            for i, (sym, sec, exch) in enumerate(todo)
        }
        for fut in as_completed(futures, timeout=None):
            sym, tag, info = fut.result()
            print(f"{sym:6s} {tag:<6s}  {info}", flush=True)
            results.append((sym, tag))

    ok   = [s for s, t in results if t == "ok"]
    fail = [s for s, t in results if t == "fail"]
    print("-" * 60, flush=True)
    print(f"ok:    {len(ok):>2d}  {', '.join(ok)}", flush=True)
    print(f"fail:  {len(fail):>2d}  {', '.join(fail)}", flush=True)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())

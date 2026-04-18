"""
Resume the 5yr fetch: only pull symbols that don't already have a parquet.
Enforces a hard 120-second timeout per ticker so no single hung read can
block the whole pipeline.
"""
from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

from fetch_data import (
    UNIVERSE, MACRO, SECTOR_ETFS,
    DATA_DIR, HOST, PORT,
    fetch_ticker,
)


class TimeoutError_(Exception): pass


def _alarm_handler(signum, frame):
    raise TimeoutError_("ticker fetch exceeded timeout")


def main() -> int:
    import shinybroker as sb

    # Full plan: equities + macro + sectors (symbol, sec_type, exchange)
    fetch_list = (
        [(s, "STK", "SMART") for s, _ in UNIVERSE]
        + MACRO
        + SECTOR_ETFS
    )

    todo = []
    for sym, sec, exch in fetch_list:
        p = DATA_DIR / f"{sym}.parquet"
        if p.exists():
            continue
        todo.append((sym, sec, exch))
    print(f"[resume] already have {len(fetch_list) - len(todo)} parquets; {len(todo)} to fetch")

    signal.signal(signal.SIGALRM, _alarm_handler)
    ok, partial, failed = [], [], []
    base_cid = 8000
    for sym, sec, exch in todo:
        print(f"[resume] fetching {sym} [{sec}] ...", flush=True)
        cache = DATA_DIR / f"{sym}.parquet"
        signal.alarm(120)   # hard 2-minute ceiling per ticker
        try:
            df = fetch_ticker(sb, sym, base_cid, sec_type=sec, exchange=exch)
            signal.alarm(0)
            df.to_parquet(cache, index=False)
            base_cid += 50
            first = df['timestamp'].min().date()
            last  = df['timestamp'].max().date()
            tag = "ok" if len(df) >= 500 else "partial"
            print(f"{sym:6s} [{sec}] {tag:<8s} {len(df):>5d}  {first}..{last}", flush=True)
            (ok if tag == "ok" else partial).append(sym)
        except TimeoutError_ as e:
            signal.alarm(0)
            print(f"{sym:6s} [{sec}] {'TIMEOUT':<8s}    -   exceeded 120s", flush=True)
            failed.append(sym)
        except Exception as e:
            signal.alarm(0)
            print(f"{sym:6s} [{sec}] {'FAIL':<8s}    -   {type(e).__name__}: {str(e)[:60]}", flush=True)
            failed.append(sym)
        time.sleep(0.5)

    print("-" * 60, flush=True)
    print(f"ok:      {len(ok):>2d}  {', '.join(ok)}", flush=True)
    print(f"partial: {len(partial):>2d}  {', '.join(partial)}", flush=True)
    print(f"failed:  {len(failed):>2d}  {', '.join(failed)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

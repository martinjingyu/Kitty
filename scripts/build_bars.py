"""
Fetch minute data from Polygon.io and build all four bar types for a ticker.
Can be run directly or imported as a function by run_pipeline.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collector import fetch_minute_aggs
from data.bars import volume, dollar, runs, imbalance

BARS_PER_DAY = 20


ROLLING_WINDOW = 20   # trailing trading days for adaptive threshold


def build_bars(
    ticker: str,
    start: str = "2021-04-29",
    end: str = "2026-04-29",
    force: bool = False,
) -> dict:
    """
    Fetch minute data and compute all 4 bar types with rolling thresholds.
    Returns dict of {bar_type: DataFrame}.
    Bars are cached as parquet under data/bars/processed/.

    force – if True, ignore cached parquet files and rebuild from scratch.
    """
    import pandas as pd

    out_dir = Path("data/bars/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fetch_minute_aggs(ticker=ticker, start=start, end=end)
    trading_days = df["timestamp"].dt.date.nunique()
    print(f"\n{ticker} raw data: {len(df):,} rows | "
          f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()} "
          f"({trading_days} trading days)")
    print(f"Target: ~{BARS_PER_DAY} bars/day  "
          f"(rolling window: {ROLLING_WINDOW} days)\n")

    tasks = [
        ("volume",    lambda: volume.compute(df,    bars_per_day=BARS_PER_DAY,
                                             window_days=ROLLING_WINDOW)),
        ("dollar",    lambda: dollar.compute(df,    bars_per_day=BARS_PER_DAY,
                                             window_days=ROLLING_WINDOW)),
        ("runs",      lambda: runs.compute(df,      bars_per_day=BARS_PER_DAY,
                                           window_days=ROLLING_WINDOW)),
        ("imbalance", lambda: imbalance.compute(df, bars_per_day=BARS_PER_DAY,
                                                window_days=ROLLING_WINDOW)),
    ]

    result = {}
    for name, fn in tasks:
        path = out_dir / f"{ticker}_{name}_bars.parquet"
        if path.exists() and not force:
            bar_df = pd.read_parquet(path)
            print(f"  {name}: loaded from cache ({len(bar_df):,} bars)  "
                  f"[use --force to rebuild]")
        else:
            bar_df = fn()
            bar_df.to_parquet(path, index=False)
            actual = len(bar_df) / trading_days
            print(f"  {name}: {len(bar_df):,} bars | {actual:.1f}/day → {path.name}")
        result[name] = bar_df

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start",  default="2021-04-29")
    parser.add_argument("--end",    default="2026-04-29")
    parser.add_argument("--force",  action="store_true",
                        help="Delete cached bars and rebuild from raw data")
    args = parser.parse_args()
    build_bars(args.ticker, args.start, args.end, force=args.force)

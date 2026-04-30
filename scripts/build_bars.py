"""
Fetch SPY minute data from Polygon.io and build all four bar types.
Output saved to data/bars/processed/.

BARS_PER_DAY controls target density for all four bar types.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collector import fetch_minute_aggs
from data.bars import volume, dollar, runs, imbalance

OUT_DIR = Path(__file__).parent.parent / "data" / "bars" / "processed"
OUT_DIR.mkdir(exist_ok=True)

BARS_PER_DAY = 20


def main():
    df = fetch_minute_aggs()
    trading_days = df["timestamp"].dt.date.nunique()
    print(f"\nRaw data: {len(df):,} rows | "
          f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()} "
          f"({trading_days} trading days)")
    print(f"Target: ~{BARS_PER_DAY} bars/day "
          f"→ expected ~{trading_days * BARS_PER_DAY:,} bars\n")

    tasks = [
        ("volume",    lambda: volume.compute(df,    bars_per_day=BARS_PER_DAY)),
        ("dollar",    lambda: dollar.compute(df,    bars_per_day=BARS_PER_DAY)),
        ("runs",      lambda: runs.compute(df,      bars_per_day=BARS_PER_DAY)),
        ("imbalance", lambda: imbalance.compute(df, bars_per_day=BARS_PER_DAY)),
    ]

    for name, fn in tasks:
        print(f"Building {name} bars ...")
        result = fn()
        out = OUT_DIR / f"SPY_{name}_bars.parquet"
        result.to_parquet(out, index=False)
        actual_per_day = len(result) / trading_days
        print(f"  {len(result):,} bars | {actual_per_day:.1f}/day | "
              f"avg {result['ticks'].mean():.0f} ticks/bar\n")


if __name__ == "__main__":
    main()

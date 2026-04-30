"""
End-to-end pipeline for any ticker:
  1. Fetch minute data from Polygon.io
  2. Build 4 bar types
  3. Train weekly + monthly models

Usage:
  python3 scripts/run_pipeline.py --tickers SNDK CRWV
  python3 scripts/run_pipeline.py --tickers SPY SNDK CRWV
"""
import sys
import argparse
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from scripts.build_bars    import build_bars
from scripts.build_and_train import train_ticker

# date range: fetch as far back as Polygon allows (up to 2 years)
END   = str(date.today())
START = "2021-01-01"   # Polygon free tier returns what it has


def run(ticker: str):
    print(f"\n{'▓'*60}")
    print(f"  PIPELINE: {ticker}")
    print(f"{'▓'*60}\n")

    # step 1 & 2: fetch + build bars
    bars = build_bars(ticker, start=START, end=END)

    # step 3: train models
    train_ticker(ticker, bars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True,
                        help="List of tickers, e.g. --tickers SNDK CRWV")
    args = parser.parse_args()

    for ticker in args.tickers:
        run(ticker.upper())

    print("\nAll tickers done.")


if __name__ == "__main__":
    main()

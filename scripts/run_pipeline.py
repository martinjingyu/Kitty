"""
End-to-end pipeline for a pooled multi-ticker model:
  1. Fetch minute data from Polygon.io
  2. Build 4 bar types
  3. Train one pooled model per regime across all tickers

Usage:
  python3 scripts/run_pipeline.py
  python3 scripts/run_pipeline.py --tickers AAPL MSFT NVDA SPY
  python3 scripts/run_pipeline.py --tickers AAPL MSFT NVDA SPY --force
"""
import sys
import argparse
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_bars    import build_bars
from scripts.build_and_train import (
    CONFIGS,
    DEFAULT_TICKERS,
    INTRADAY_BOUNDARY_PATH,
    load_intraday_boundaries,
    run_regime,
    select_regime_configs,
)

# date range: fetch as far back as Polygon allows (up to 2 years)
END   = str(date.today())
START = "2021-01-01"   # Polygon free tier returns what it has


def run(
    tickers: list[str],
    force: bool = False,
    intraday_boundaries: str | None = str(INTRADAY_BOUNDARY_PATH),
    regimes: list[str] | None = None,
):
    print(f"\n{'▓'*60}")
    print(f"  PIPELINE: {', '.join(tickers)}")
    print(f"{'▓'*60}\n")

    all_bars = {}
    for ticker in tickers:
        # step 1 & 2: fetch + build bars
        all_bars[ticker] = build_bars(ticker, start=START, end=END, force=force)

    # step 3: train pooled models, one per regime
    configs = select_regime_configs(regimes)
    if "intraday" in configs:
        configs["intraday"]["boundary_overrides"] = load_intraday_boundaries(intraday_boundaries)
    print(f"Training regimes: {', '.join(configs.keys())}")
    for regime, cfg in configs.items():
        run_regime(regime, cfg, all_bars, tickers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="List of tickers; default trains all configured tickers")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached bars and rebuild from raw data")
    parser.add_argument("--intraday-boundaries", default=str(INTRADAY_BOUNDARY_PATH),
                        help="CSV from sweep_intraday_boundaries.py; use 'none' to disable")
    parser.add_argument("--regimes", nargs="+", choices=list(CONFIGS.keys()),
                        default=list(CONFIGS.keys()),
                        help="Regimes to train; default trains all")
    args = parser.parse_args()

    tickers = [ticker.upper() for ticker in args.tickers]
    boundary_path = None if args.intraday_boundaries.lower() == "none" else args.intraday_boundaries
    run(tickers, force=args.force, intraday_boundaries=boundary_path, regimes=args.regimes)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

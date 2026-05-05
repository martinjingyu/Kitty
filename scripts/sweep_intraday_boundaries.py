"""
Sweep asymmetric intraday triple-barrier settings per ticker.

The script labels each ticker's dollar bars with separate upside/downside
barriers, then ranks boundary combinations by a simple "tradeable sample"
score.  It does not train a model; it is a fast pre-check for choosing sane
intraday label settings before retraining.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from data.labels import _daily_vol
from scripts.build_and_train import DEFAULT_TICKERS


BARS_DIR = Path("data/bars/processed")
OUT_DIR = Path("models/saved")


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _first_touch_labels(
    bars: pd.DataFrame,
    *,
    h_up: float,
    h_down: float,
    min_up: float,
    min_down: float,
    max_hold: int,
    vol_lookback: int,
    stride: int,
) -> pd.DataFrame:
    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    ts = bars["timestamp"].to_numpy()
    vol_series = _daily_vol(bars["close"], lookback=vol_lookback).to_numpy()

    records = []
    for t in range(vol_lookback, len(bars) - 1, stride):
        vol_t = vol_series[t]
        if np.isnan(vol_t) or vol_t <= 0:
            continue

        entry = close[t]
        up_dist = max(h_up * vol_t, min_up)
        down_dist = max(h_down * vol_t, min_down)
        upper = entry * (1 + up_dist)
        lower = entry * (1 - down_dist)
        horizon = min(t + max_hold, len(bars) - 1)

        label = 0
        t_exit = horizon
        for i in range(t + 1, horizon + 1):
            if high[i] >= upper:
                label, t_exit = 1, i
                break
            if low[i] <= lower:
                label, t_exit = -1, i
                break

        signed_ret = 0.0
        if label:
            signed_ret = label * float(np.log(close[t_exit] / entry))

        records.append({
            "timestamp": ts[t],
            "label": label,
            "t": t,
            "t_exit": t_exit,
            "hold_bars": t_exit - t,
            "signed_ret": signed_ret,
            "up_dist": up_dist,
            "down_dist": down_dist,
        })

    return pd.DataFrame(records)


def _summarize(
    ticker: str,
    labels: pd.DataFrame,
    *,
    h_up: float,
    h_down: float,
    min_up: float,
    min_down: float,
    min_events: int,
) -> dict:
    total = len(labels)
    if total == 0:
        directional = labels
    else:
        directional = labels[labels["label"] != 0]

    n_dir = len(directional)
    n_long = int((labels["label"] == 1).sum()) if total else 0
    n_short = int((labels["label"] == -1).sum()) if total else 0
    n_zero = int((labels["label"] == 0).sum()) if total else 0
    coverage = n_dir / total if total else 0.0
    long_share = n_long / n_dir if n_dir else 0.0
    short_share = n_short / n_dir if n_dir else 0.0
    balance = min(n_long, n_short) / max(n_long, n_short) if max(n_long, n_short) else 0.0

    avg_up = float(labels["up_dist"].mean()) if total else 0.0
    avg_down = float(labels["down_dist"].mean()) if total else 0.0
    rr = avg_up / avg_down if avg_down > 0 else 0.0
    mean_signed_ret = float(directional["signed_ret"].mean()) if n_dir else 0.0
    median_hold = float(directional["hold_bars"].median()) if n_dir else 0.0

    sample_score = min(n_dir / min_events, 1.0) if min_events > 0 else 1.0
    coverage_score = min(coverage / 0.75, 1.0)
    balance_score = balance ** 1.5
    rr_score = max(0.1, 1 - abs(rr - 1.6) / 1.6)
    score = sample_score * coverage_score * balance_score * rr_score

    return {
        "ticker": ticker,
        "h_up": h_up,
        "h_down": h_down,
        "min_up_%": min_up * 100,
        "min_down_%": min_down * 100,
        "avg_up_%": avg_up * 100,
        "avg_down_%": avg_down * 100,
        "rr": rr,
        "events": total,
        "directional": n_dir,
        "coverage": coverage,
        "long": n_long,
        "short": n_short,
        "long_share": long_share,
        "short_share": short_share,
        "zero": n_zero,
        "balance": balance,
        "median_hold_bars": median_hold,
        "mean_signed_ret_%": mean_signed_ret * 100,
        "score": score,
    }


def sweep_ticker(
    ticker: str,
    *,
    h_ups: list[float],
    h_downs: list[float],
    min_ups: list[float],
    min_downs: list[float],
    max_hold: int,
    vol_lookback: int,
    stride: int,
    min_events: int,
) -> pd.DataFrame:
    path = BARS_DIR / f"{ticker}_dollar_bars.parquet"
    if not path.exists():
        print(f"skip {ticker}: missing {path}")
        return pd.DataFrame()

    bars = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    rows = []
    for h_up in h_ups:
        for h_down in h_downs:
            for min_up in min_ups:
                for min_down in min_downs:
                    labels = _first_touch_labels(
                        bars,
                        h_up=h_up,
                        h_down=h_down,
                        min_up=min_up,
                        min_down=min_down,
                        max_hold=max_hold,
                        vol_lookback=vol_lookback,
                        stride=stride,
                    )
                    rows.append(_summarize(
                        ticker,
                        labels,
                        h_up=h_up,
                        h_down=h_down,
                        min_up=min_up,
                        min_down=min_down,
                        min_events=min_events,
                    ))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--h-up", default="0.6,0.8,1.0,1.2")
    parser.add_argument("--h-down", default="0.6,0.8,1.0,1.2")
    parser.add_argument("--min-up", default="0.01,0.0125,0.015,0.02")
    parser.add_argument("--min-down", default="0.008,0.01,0.0125,0.015")
    parser.add_argument("--max-hold", type=int, default=20)
    parser.add_argument("--vol-lookback", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-events", type=int, default=500)
    parser.add_argument("--min-balance", type=float, default=0.55,
                        help="Minimum LONG/SHORT balance for top selection")
    parser.add_argument("--max-short-share", type=float, default=0.65,
                        help="Maximum SHORT share among directional labels")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--out", default=str(OUT_DIR / "intraday_boundary_sweep.csv"))
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    all_rows = []
    for ticker in tickers:
        print(f"sweeping {ticker} ...")
        result = sweep_ticker(
            ticker,
            h_ups=_parse_float_list(args.h_up),
            h_downs=_parse_float_list(args.h_down),
            min_ups=_parse_float_list(args.min_up),
            min_downs=_parse_float_list(args.min_down),
            max_hold=args.max_hold,
            vol_lookback=args.vol_lookback,
            stride=args.stride,
            min_events=args.min_events,
        )
        if not result.empty:
            all_rows.append(result)

    if not all_rows:
        raise RuntimeError("No sweep results; check ticker list and processed dollar bars.")

    sweep = pd.concat(all_rows, ignore_index=True)
    sweep = sweep.sort_values(["ticker", "score"], ascending=[True, False])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(out_path, index=False)

    eligible = sweep[
        (sweep["balance"] >= args.min_balance) &
        (sweep["short_share"] <= args.max_short_share)
    ]
    if eligible.empty:
        print("No rows passed balance filters; falling back to unfiltered top rows.")
        eligible = sweep
    top = eligible.groupby("ticker", group_keys=False).head(args.top_n)
    top_path = out_path.with_name(out_path.stem + "_top.csv")
    top.to_csv(top_path, index=False)

    cols = [
        "ticker", "score", "h_up", "h_down", "min_up_%", "min_down_%",
        "avg_up_%", "avg_down_%", "rr", "directional", "coverage",
        "long", "short", "balance", "median_hold_bars",
    ]
    print("\nTop settings per ticker:")
    print(top[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nFull sweep saved to {out_path}")
    print(f"Top settings saved to {top_path}")


if __name__ == "__main__":
    main()

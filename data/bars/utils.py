"""Shared utilities for bar construction."""
import numpy as np
import pandas as pd


def tick_rule(closes: pd.Series) -> np.ndarray:
    """
    Assign direction b_t ∈ {+1, -1} to each row using the tick rule:
      - b_t = +1 if close_t > close_{t-1}  (uptick → buyer-initiated)
      - b_t = -1 if close_t < close_{t-1}  (downtick → seller-initiated)
      - b_t = b_{t-1} if close_t == close_{t-1}  (zero tick → inherit previous)
    """
    diff = closes.diff()
    directions = np.zeros(len(closes), dtype=np.float64)
    last = 1.0
    for i, d in enumerate(diff):
        if np.isnan(d) or d == 0:
            directions[i] = last
        elif d > 0:
            directions[i] = 1.0
            last = 1.0
        else:
            directions[i] = -1.0
            last = -1.0
    return directions


def close_bar(rows: list, ts) -> dict:
    """Aggregate a list of raw rows into a single OHLCV bar."""
    opens  = [r["open"]   for r in rows]
    highs  = [r["high"]   for r in rows]
    lows   = [r["low"]    for r in rows]
    closes = [r["close"]  for r in rows]
    vols   = [r["volume"] for r in rows]
    return {
        "timestamp": ts,
        "open":   opens[0],
        "high":   max(highs),
        "low":    min(lows),
        "close":  closes[-1],
        "volume": sum(vols),
        "dollar": sum(c * v for c, v in zip(closes, vols)),
        "ticks":  len(rows),
    }

"""
Triple-barrier labeling (Lopez de Prado, AFML Ch.3)

For each bar t, three barriers are set:
  Upper  (+1): close_t × (1 + h × vol_t)   — profit target
  Lower  (-1): close_t × (1 - h × vol_t)   — stop loss
  Vertical(0): t + max_hold bars            — time limit

Label = whichever barrier is touched first.

vol_t is the rolling daily return volatility, making barriers
adaptive to market regime (wider in volatile periods).
"""
import numpy as np
import pandas as pd


def _daily_vol(close: pd.Series, lookback: int = 20) -> pd.Series:
    """
    Rolling volatility of daily returns, aligned to bar index.
    Uses log returns and an EWM for smoothness.
    """
    log_ret = np.log(close).diff()
    return log_ret.ewm(span=lookback).std()


def label_triple_barrier(
    bars: pd.DataFrame,
    h: float = 1.0,
    max_hold: int = 10,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    bars        : DataFrame with at least [timestamp, open, high, low, close]
    h           : barrier width multiplier (barriers = h × rolling_vol)
    max_hold    : max bars to hold before vertical barrier triggers
    vol_lookback: EWM span for volatility estimate

    Returns
    -------
    DataFrame with columns:
      t           : entry bar index
      t_exit      : exit bar index (where barrier was touched)
      timestamp   : entry timestamp
      timestamp_exit
      label       : +1 / -1 / 0
      ret         : actual log return from entry to exit
      vol         : volatility at entry
    """
    close = bars["close"].values
    high  = bars["high"].values
    low   = bars["low"].values
    ts    = bars["timestamp"].values
    n     = len(bars)

    vol_series = _daily_vol(bars["close"], lookback=vol_lookback).values

    records = []
    for t in range(vol_lookback, n - 1):
        vol_t = vol_series[t]
        if np.isnan(vol_t) or vol_t == 0:
            continue

        entry   = close[t]
        upper   = entry * (1 + h * vol_t)
        lower   = entry * (1 - h * vol_t)
        horizon = min(t + max_hold, n - 1)

        label  = 0
        t_exit = horizon
        for i in range(t + 1, horizon + 1):
            if high[i] >= upper:
                label, t_exit = 1, i
                break
            if low[i] <= lower:
                label, t_exit = -1, i
                break

        ret = np.log(close[t_exit] / entry)
        records.append({
            "t":              t,
            "t_exit":         t_exit,
            "timestamp":      ts[t],
            "timestamp_exit": ts[t_exit],
            "label":          label,
            "ret":            ret,
            "vol":            vol_t,
        })

    df = pd.DataFrame(records)

    # sample weights: downweight events whose label windows overlap
    # weight_i ∝ 1 / (number of co-occurring events)
    df["weight"] = _sample_weights(df["t"].values, df["t_exit"].values)

    return df


def _sample_weights(t_enter: np.ndarray, t_exit: np.ndarray) -> np.ndarray:
    """
    Weight_i = 1 / avg_concurrent_events over event i's window.
    Uses a difference array to count concurrent events in O(n + T).
    """
    if len(t_enter) == 0:
        return np.array([])

    max_t = int(t_exit.max()) + 2
    counts = np.zeros(max_t, dtype=np.float64)
    for a, b in zip(t_enter, t_exit):
        counts[int(a)] += 1
        if int(b) + 1 < max_t:
            counts[int(b) + 1] -= 1
    concurrent = np.cumsum(counts)  # concurrent[b] = # active events at bar b

    weights = np.empty(len(t_enter))
    for i, (a, b) in enumerate(zip(t_enter, t_exit)):
        avg = concurrent[int(a): int(b) + 1].mean()
        weights[i] = 1.0 / max(avg, 1.0)
    return weights

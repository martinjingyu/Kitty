"""
Triple-barrier labeling (Lopez de Prado, AFML Ch.3)

For each sampled bar t (sampled every `stride` bars), three barriers are set:
  Upper  (+1): close_t × (1 + h × vol_t)   — profit target
  Lower  (-1): close_t × (1 - h × vol_t)   — stop loss
  Vertical(0): t + max_hold bars            — time limit

Label = whichever barrier is touched first.

stride aligns the prediction frequency to the holding period:
  Weekly  model → stride = 20  (predict once per trading day)
  Monthly model → stride = 100 (predict once per trading week)

This ensures feature lookback ≈ label horizon ≈ stride × N, and
drastically reduces label window overlap.
"""
import numpy as np
import pandas as pd


def _daily_vol(close: pd.Series, lookback: int = 20) -> pd.Series:
    log_ret = np.log(close).diff()
    return log_ret.ewm(span=lookback).std()


def label_triple_barrier(
    bars: pd.DataFrame,
    h: float = 1.0,
    max_hold: int = 100,
    vol_lookback: int = 20,
    stride: int = 1,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    bars        : DataFrame with [timestamp, open, high, low, close]
    h           : barrier width multiplier  (barrier = h × rolling_vol)
    max_hold    : max bars before vertical barrier triggers
    vol_lookback: EWM span for volatility
    stride      : sample every N bars (1 = every bar, 20 = once/day,
                  100 = once/week).  Larger stride → less label overlap.

    Returns
    -------
    DataFrame: t, t_exit, timestamp, timestamp_exit, label, ret, vol, weight
    """
    close = bars["close"].values
    high  = bars["high"].values
    low   = bars["low"].values
    ts    = bars["timestamp"].values
    n     = len(bars)

    vol_series = _daily_vol(bars["close"], lookback=vol_lookback).values

    # candidate entry bars: start after warmup, step by stride
    candidates = range(vol_lookback, n - 1, stride)

    records = []
    for t in candidates:
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
    df["weight"] = _sample_weights(df["t"].values, df["t_exit"].values)
    return df


def _sample_weights(t_enter: np.ndarray, t_exit: np.ndarray) -> np.ndarray:
    """
    Weight_i = 1 / avg_concurrent_events over event i's window.
    With large stride this is near 1.0 for most events (little overlap).
    """
    if len(t_enter) == 0:
        return np.array([])

    max_t    = int(t_exit.max()) + 2
    counts   = np.zeros(max_t, dtype=np.float64)
    for a, b in zip(t_enter, t_exit):
        counts[int(a)] += 1
        if int(b) + 1 < max_t:
            counts[int(b) + 1] -= 1
    concurrent = np.cumsum(counts)

    weights = np.empty(len(t_enter))
    for i, (a, b) in enumerate(zip(t_enter, t_exit)):
        avg = concurrent[int(a): int(b) + 1].mean()
        weights[i] = 1.0 / max(avg, 1.0)
    return weights

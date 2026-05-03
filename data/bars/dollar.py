from typing import Optional
import numpy as np
import pandas as pd
from .utils import close_bar


def _rolling_threshold(df: pd.DataFrame, bars_per_day: int, window_days: int) -> dict:
    """
    Compute a per-day threshold using a trailing `window_days` rolling mean of
    daily dollar volume.  Using the trailing window (shift-1) avoids lookahead:
    day t's threshold is based on days t-window..t-1.

    Falls back to the first available mean for the earliest days.
    """
    dates    = df["timestamp"].dt.date
    daily_dv = df.groupby(dates).apply(lambda g: (g["close"] * g["volume"]).sum())

    # shift(1) → exclude current day; min_periods=1 → no NaN at startup
    rolled   = daily_dv.shift(1).rolling(window=window_days, min_periods=1).mean()
    # the very first day has NaN after shift; fill with the first valid mean
    rolled   = rolled.bfill()
    thr_map  = (rolled / bars_per_day).to_dict()
    return thr_map


def compute(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    bars_per_day: int = 20,
    window_days: int = 20,
) -> pd.DataFrame:
    """
    Dollar bars: close a bar every time cumulative dollar value >= threshold.

    threshold   – fixed value (legacy / engine use); if None, a rolling threshold
                  is derived from `window_days` trailing days so that bars/day
                  stays ~bars_per_day regardless of changing market volume.
    window_days – trailing window for rolling mean (default 20 trading days).
    """
    if threshold is None:
        thr_map    = _rolling_threshold(df, bars_per_day, window_days)
        fixed_thr  = None
    else:
        thr_map    = {}
        fixed_thr  = threshold

    bars, bucket = [], []
    cum_dollar   = 0.0
    cur_thr      = fixed_thr  # updated each row if rolling

    for row in df.itertuples(index=False):
        d = row.timestamp.date()
        if thr_map:
            cur_thr = thr_map.get(d, cur_thr)

        bucket.append(row._asdict())
        cum_dollar += row.close * row.volume

        if cum_dollar >= cur_thr:
            bars.append(close_bar(bucket, row.timestamp))
            bucket, cum_dollar = [], 0.0

    return pd.DataFrame(bars)

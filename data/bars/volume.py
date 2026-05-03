from typing import Optional
import pandas as pd
from .utils import close_bar


def _rolling_threshold(df: pd.DataFrame, bars_per_day: int, window_days: int) -> dict:
    dates      = df["timestamp"].dt.date
    daily_vol  = df.groupby(dates)["volume"].sum()
    rolled     = daily_vol.shift(1).rolling(window=window_days, min_periods=1).mean()
    rolled     = rolled.bfill()
    return (rolled / bars_per_day).to_dict()


def compute(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    bars_per_day: int = 20,
    window_days: int = 20,
) -> pd.DataFrame:
    """
    Volume bars: close a bar every time cumulative volume >= threshold.

    threshold   – fixed value; if None, derived from rolling `window_days`
                  trailing daily volume so that bars/day ≈ bars_per_day.
    """
    if threshold is None:
        thr_map   = _rolling_threshold(df, bars_per_day, window_days)
        fixed_thr = None
    else:
        thr_map   = {}
        fixed_thr = threshold

    bars, bucket = [], []
    cum_vol      = 0.0
    cur_thr      = fixed_thr

    for row in df.itertuples(index=False):
        d = row.timestamp.date()
        if thr_map:
            cur_thr = thr_map.get(d, cur_thr)

        bucket.append(row._asdict())
        cum_vol += row.volume

        if cum_vol >= cur_thr:
            bars.append(close_bar(bucket, row.timestamp))
            bucket, cum_vol = [], 0.0

    return pd.DataFrame(bars)

"""
Dollar Imbalance Bars (Lopez de Prado, AFML Ch.2)

With balanced ticks, θ is a random walk → E[|θ|] ≈ √T × avg_dvt.
To close every T ticks on average: threshold = √T × avg_dvt
                                             = √(ticks_per_bar) × avg_dvt

Rolling threshold uses trailing window_days to track:
  - rolling_ticks_per_bar = rolling_daily_ticks / bars_per_day
  - rolling_avg_dvt       = rolling_daily_dv    / rolling_daily_ticks

  threshold = √(rolling_ticks_per_bar) × rolling_avg_dvt
            = rolling_daily_dv / √(rolling_daily_ticks × bars_per_day)
"""
import numpy as np
import pandas as pd
from .utils import tick_rule, close_bar


def _rolling_threshold(df: pd.DataFrame, bars_per_day: int, window_days: int) -> dict:
    dates       = df["timestamp"].dt.date
    daily_dv    = df.groupby(dates).apply(lambda g: (g["close"] * g["volume"]).sum())
    daily_ticks = df.groupby(dates).size().astype(float)

    roll_dv     = daily_dv.shift(1).rolling(window=window_days, min_periods=1).mean().bfill()
    roll_ticks  = daily_ticks.shift(1).rolling(window=window_days, min_periods=1).mean().bfill()

    # threshold = roll_dv / sqrt(roll_ticks * bars_per_day)
    thr_series  = roll_dv / np.sqrt(roll_ticks * bars_per_day)
    return thr_series.to_dict()


def compute(df: pd.DataFrame, bars_per_day: int = 20, window_days: int = 20) -> pd.DataFrame:
    directions      = tick_rule(df["close"])
    dollar_per_tick = (df["close"] * df["volume"]).values

    thr_map = _rolling_threshold(df, bars_per_day, window_days)
    cur_thr = next(iter(thr_map.values()))   # seed with first day

    bars, bucket = [], []
    theta = 0.0

    for i, row in enumerate(df.itertuples(index=False)):
        d       = row.timestamp.date()
        cur_thr = thr_map.get(d, cur_thr)

        b   = directions[i]
        dvt = dollar_per_tick[i]
        bucket.append(row._asdict())
        theta += b * dvt

        if abs(theta) >= cur_thr:
            bar = close_bar(bucket, row.timestamp)
            bar["theta"] = theta
            bars.append(bar)

            bucket = []
            theta  = 0.0

    return pd.DataFrame(bars)

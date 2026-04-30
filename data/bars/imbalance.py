"""
Dollar Imbalance Bars (Lopez de Prado, AFML Ch.2)

Threshold derived analytically:
  With balanced ticks, θ is a random walk → E[|θ|] ≈ √T × avg_dvt.
  To close every T ticks: threshold = √T × avg_dvt_per_tick.
"""
import numpy as np
import pandas as pd
from .utils import tick_rule, close_bar


def compute(df: pd.DataFrame, bars_per_day: int = 20) -> pd.DataFrame:
    directions = tick_rule(df["close"])
    dollar_per_tick = (df["close"] * df["volume"]).values

    trading_days   = df["timestamp"].dt.date.nunique()
    ticks_per_bar  = len(df) / (trading_days * bars_per_day)
    avg_dvt        = float(np.mean(dollar_per_tick))
    threshold = np.sqrt(ticks_per_bar) * avg_dvt

    bars, bucket = [], []
    theta = 0.0

    for i, row in enumerate(df.itertuples(index=False)):
        b   = directions[i]
        dvt = dollar_per_tick[i]
        bucket.append(row._asdict())
        theta += b * dvt

        if abs(theta) >= threshold:
            bar = close_bar(bucket, row.timestamp)
            bar["theta"] = theta
            bars.append(bar)

            bucket = []
            theta = 0.0

    return pd.DataFrame(bars)

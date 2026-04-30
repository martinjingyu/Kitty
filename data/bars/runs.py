"""
Dollar Runs Bars (Lopez de Prado, AFML Ch.2)

Threshold derived analytically:
  With p≈0.5, the winning side accumulates ≈ 0.5 × n × avg_dvt after n ticks.
  To close every T ticks: threshold = 0.5 × T × avg_dvt = daily_dv / (2 × N)
"""
import numpy as np
import pandas as pd
from .utils import tick_rule, close_bar


def compute(df: pd.DataFrame, bars_per_day: int = 20) -> pd.DataFrame:
    directions = tick_rule(df["close"])
    dollar_per_tick = (df["close"] * df["volume"]).values

    trading_days  = df["timestamp"].dt.date.nunique()
    avg_daily_dv  = float((df["close"] * df["volume"]).sum()) / trading_days
    threshold = avg_daily_dv / (bars_per_day * 2)

    bars, bucket  = [], []
    cum_buy_dv = cum_sell_dv = 0.0

    for i, row in enumerate(df.itertuples(index=False)):
        b   = directions[i]
        dvt = dollar_per_tick[i]
        bucket.append(row._asdict())

        if b > 0:
            cum_buy_dv  += dvt
        else:
            cum_sell_dv += dvt

        if max(cum_buy_dv, cum_sell_dv) >= threshold:
            bar = close_bar(bucket, row.timestamp)
            bar["buy_dollar"]  = cum_buy_dv
            bar["sell_dollar"] = cum_sell_dv
            bars.append(bar)

            bucket = []
            cum_buy_dv = cum_sell_dv = 0.0

    return pd.DataFrame(bars)

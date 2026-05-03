"""
Dollar Runs Bars (Lopez de Prado, AFML Ch.2)

With balanced ticks (p ≈ 0.5), the winning side accumulates
  ≈ 0.5 × n × avg_dvt  after n ticks.
To close every T ticks on average: threshold = 0.5 × T × avg_dvt
                                             = daily_dv / (2 × bars_per_day)

Rolling threshold: use trailing window_days days so that bars/day stays
stable even when market dollar volume drifts over time.
"""
import numpy as np
import pandas as pd
from .utils import tick_rule, close_bar


def _rolling_threshold(df: pd.DataFrame, bars_per_day: int, window_days: int) -> dict:
    dates    = df["timestamp"].dt.date
    daily_dv = df.groupby(dates).apply(lambda g: (g["close"] * g["volume"]).sum())
    rolled   = daily_dv.shift(1).rolling(window=window_days, min_periods=1).mean()
    rolled   = rolled.bfill()
    return (rolled / (bars_per_day * 2)).to_dict()


def compute(df: pd.DataFrame, bars_per_day: int = 20, window_days: int = 20) -> pd.DataFrame:
    directions      = tick_rule(df["close"])
    dollar_per_tick = (df["close"] * df["volume"]).values

    thr_map = _rolling_threshold(df, bars_per_day, window_days)
    cur_thr = next(iter(thr_map.values()))   # seed with first day

    bars, bucket     = [], []
    cum_buy_dv = cum_sell_dv = 0.0

    for i, row in enumerate(df.itertuples(index=False)):
        d       = row.timestamp.date()
        cur_thr = thr_map.get(d, cur_thr)

        b   = directions[i]
        dvt = dollar_per_tick[i]
        bucket.append(row._asdict())

        if b > 0:
            cum_buy_dv  += dvt
        else:
            cum_sell_dv += dvt

        if max(cum_buy_dv, cum_sell_dv) >= cur_thr:
            bar = close_bar(bucket, row.timestamp)
            bar["buy_dollar"]  = cum_buy_dv
            bar["sell_dollar"] = cum_sell_dv
            bars.append(bar)

            bucket = []
            cum_buy_dv = cum_sell_dv = 0.0

    return pd.DataFrame(bars)

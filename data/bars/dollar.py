from typing import Optional
import pandas as pd
from .utils import close_bar


def compute(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    bars_per_day: int = 20,
) -> pd.DataFrame:
    """
    Dollar bars: close a bar every time cumulative dollar value >= threshold.
    If threshold is None, it is derived from bars_per_day (default 20).
    """
    if threshold is None:
        daily_dv = df.groupby(df["timestamp"].dt.date).apply(
            lambda g: (g["close"] * g["volume"]).sum()
        )
        threshold = float(daily_dv.mean()) / bars_per_day

    bars, bucket = [], []
    cum_dollar = 0.0

    for row in df.itertuples(index=False):
        bucket.append(row._asdict())
        cum_dollar += row.close * row.volume

        if cum_dollar >= threshold:
            bars.append(close_bar(bucket, row.timestamp))
            bucket, cum_dollar = [], 0.0

    return pd.DataFrame(bars)

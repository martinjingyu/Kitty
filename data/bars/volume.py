from typing import Optional
import pandas as pd
from .utils import close_bar


def compute(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    bars_per_day: int = 20,
) -> pd.DataFrame:
    """
    Volume bars: close a bar every time cumulative volume >= threshold.
    If threshold is None, it is derived from bars_per_day (default 20).
    """
    if threshold is None:
        daily_vol = df.groupby(df["timestamp"].dt.date)["volume"].sum()
        threshold = float(daily_vol.mean()) / bars_per_day

    bars, bucket = [], []
    cum_vol = 0.0

    for row in df.itertuples(index=False):
        bucket.append(row._asdict())
        cum_vol += row.volume

        if cum_vol >= threshold:
            bars.append(close_bar(bucket, row.timestamp))
            bucket, cum_vol = [], 0.0

    return pd.DataFrame(bars)

"""
Feature engineering for SPY bar dataset.

Primary axis: dollar bars.
Other bar types (volume, runs, imbalance) contribute cross-bar features
that capture information density and order-flow directionality.
"""
import numpy as np
import pandas as pd


def bar_features(df: pd.DataFrame, prefix: str = "d_") -> pd.DataFrame:
    c   = df["close"]
    o   = df["open"]
    h   = df["high"]
    l   = df["low"]
    rng = (h - l).replace(0, np.nan)

    out = pd.DataFrame(index=df.index)
    out[f"{prefix}log_ret"]    = np.log(c / o)
    out[f"{prefix}body_ratio"] = (c - o).abs() / rng
    out[f"{prefix}upper_wick"] = (h - c.clip(lower=o)) / rng
    out[f"{prefix}lower_wick"] = (c.clip(upper=o) - l) / rng
    out[f"{prefix}log_volume"] = np.log1p(df["volume"])
    out[f"{prefix}log_dollar"] = np.log1p(df["dollar"])
    out[f"{prefix}ticks"]      = df["ticks"]

    if "buy_dollar" in df.columns:
        total = df["buy_dollar"] + df["sell_dollar"]
        out[f"{prefix}buy_ratio"] = df["buy_dollar"] / total.replace(0, np.nan)

    if "theta" in df.columns:
        out[f"{prefix}theta_norm"] = df["theta"] / df["dollar"].replace(0, np.nan)

    return out.fillna(0)


def rolling_features(
    df: pd.DataFrame,
    windows: list,
    prefix: str = "d_",
) -> pd.DataFrame:
    log_ret = np.log(df["close"] / df["close"].shift(1))
    out = pd.DataFrame(index=df.index)

    for w in windows:
        out[f"{prefix}mom_{w}"]       = log_ret.rolling(w).sum()
        out[f"{prefix}vol_{w}"]       = log_ret.rolling(w).std()
        out[f"{prefix}vol_ratio_{w}"] = (
            log_ret.rolling(w).std() / log_ret.rolling(w * 4).std()
        )
        out[f"{prefix}vol_trend_{w}"] = (
            np.log1p(df["volume"]) - np.log1p(df["volume"].rolling(w).mean())
        )
        out[f"{prefix}autocorr_{w}"]  = (
            log_ret.rolling(w + 1).apply(
                lambda x: x[:-1].corr(pd.Series(x[1:])), raw=False
            )
        )

    return out.fillna(0)


def info_density_feature(
    dollar_bars: pd.DataFrame,
    other_bars: pd.DataFrame,
    other_name: str,
    window_minutes: int,
    prefix: str = "",
) -> pd.Series:
    dollar_ts = pd.to_datetime(dollar_bars["timestamp"])
    other_ts  = pd.to_datetime(other_bars["timestamp"])

    counts = []
    for ts in dollar_ts:
        window_start = ts - pd.Timedelta(minutes=window_minutes)
        n = ((other_ts >= window_start) & (other_ts <= ts)).sum()
        counts.append(n)

    return pd.Series(counts, index=dollar_bars.index,
                     name=f"{prefix}density_{other_name}")


def build_features(
    dollar: pd.DataFrame,
    volume: pd.DataFrame,
    runs: pd.DataFrame,
    imbalance: pd.DataFrame,
    windows: list = [5, 10, 20],
    density_window_min: int = 60,
    prefix: str = "d_",
) -> pd.DataFrame:
    parts = [
        bar_features(dollar, prefix=prefix),
        rolling_features(dollar, windows=windows, prefix=prefix),
    ]

    for name, other in [("vol", volume), ("runs", runs), ("imb", imbalance)]:
        density = info_density_feature(
            dollar, other, name, density_window_min, prefix=prefix
        )
        parts.append(density.to_frame())

    feat = pd.concat(parts, axis=1)

    for name in ["vol", "runs", "imb"]:
        col = f"{prefix}density_{name}"
        feat[f"{col}_trend"] = feat[col] - feat[col].rolling(windows[-1]).mean()

    return feat.fillna(0)

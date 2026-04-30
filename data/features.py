"""
Feature engineering for SPY bar dataset.

Primary axis: dollar bars.
Other bar types (volume, runs, imbalance) contribute cross-bar features
that capture information density and order-flow directionality.
"""
import numpy as np
import pandas as pd


# ── single-bar structural features ───────────────────────────────────────────

def bar_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """
    Per-bar features derived from OHLCV structure.
    Works on any bar type.
    """
    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    rng = (h - l).replace(0, np.nan)

    out = pd.DataFrame(index=df.index)
    p = prefix

    out[f"{p}log_ret"]     = np.log(c / o)                   # intra-bar return
    out[f"{p}body_ratio"]  = (c - o).abs() / rng             # candle body size
    out[f"{p}upper_wick"]  = (h - c.clip(lower=o)) / rng     # upper wick ratio
    out[f"{p}lower_wick"]  = (c.clip(upper=o) - l) / rng     # lower wick ratio
    out[f"{p}log_volume"]  = np.log1p(df["volume"])
    out[f"{p}log_dollar"]  = np.log1p(df["dollar"])
    out[f"{p}ticks"]       = df["ticks"]

    # runs bar: buy/sell imbalance ratio (order flow direction)
    if "buy_dollar" in df.columns:
        total = df["buy_dollar"] + df["sell_dollar"]
        out[f"{p}buy_ratio"] = df["buy_dollar"] / total.replace(0, np.nan)

    # imbalance bar: signed theta normalised by dollar volume
    if "theta" in df.columns:
        out[f"{p}theta_norm"] = df["theta"] / df["dollar"].replace(0, np.nan)

    return out.fillna(0)


# ── rolling window features ───────────────────────────────────────────────────

def rolling_features(
    df: pd.DataFrame,
    windows: list = [5, 10, 20],
    prefix: str = "",
) -> pd.DataFrame:
    """
    Momentum, volatility, and volume trend over multiple bar windows.
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    out = pd.DataFrame(index=df.index)
    p = prefix

    for w in windows:
        out[f"{p}mom_{w}"]    = log_ret.rolling(w).sum()                     # momentum
        out[f"{p}vol_{w}"]    = log_ret.rolling(w).std()                     # volatility
        out[f"{p}vol_ratio_{w}"] = (                                          # vol regime
            log_ret.rolling(w).std() / log_ret.rolling(w * 4).std()
        )
        out[f"{p}vol_trend_{w}"] = (                                          # volume trend
            np.log1p(df["volume"]) - np.log1p(df["volume"].rolling(w).mean())
        )
        out[f"{p}autocorr_{w}"] = (                                           # return autocorr
            log_ret.rolling(w + 1).apply(
                lambda x: x[:-1].corr(pd.Series(x[1:])), raw=False
            )
        )

    return out.fillna(0)


# ── cross-bar information density feature ────────────────────────────────────

def info_density_feature(
    dollar_bars: pd.DataFrame,
    other_bars: pd.DataFrame,
    other_name: str,
    window_minutes: int = 60,
) -> pd.Series:
    """
    For each dollar bar, count how many 'other' bars formed in the same
    60-minute rolling window. High count = more information events = likely
    informed trading activity.

    Returns a Series aligned to dollar_bars.index.
    """
    dollar_ts = pd.to_datetime(dollar_bars["timestamp"])
    other_ts  = pd.to_datetime(other_bars["timestamp"])

    counts = []
    for ts in dollar_ts:
        window_start = ts - pd.Timedelta(minutes=window_minutes)
        n = ((other_ts >= window_start) & (other_ts <= ts)).sum()
        counts.append(n)

    return pd.Series(counts, index=dollar_bars.index, name=f"density_{other_name}")


# ── main entry point ──────────────────────────────────────────────────────────

def build_features(
    dollar: pd.DataFrame,
    volume: pd.DataFrame,
    runs: pd.DataFrame,
    imbalance: pd.DataFrame,
    windows: list = [5, 10, 20],
    density_window_min: int = 60,
) -> pd.DataFrame:
    """
    Assemble the full feature matrix on the dollar bar time axis.

    Returns a DataFrame aligned to dollar bars, ready to merge with labels.
    """
    parts = [
        bar_features(dollar,    prefix="d_"),
        rolling_features(dollar, windows=windows, prefix="d_"),
    ]

    # information density: how many other-type bars formed per 60-min window
    for name, other in [("vol", volume), ("runs", runs), ("imb", imbalance)]:
        density = info_density_feature(dollar, other, name, density_window_min)
        parts.append(density.to_frame())

    # rolling density trends
    feat = pd.concat(parts, axis=1)
    for name in ["vol", "runs", "imb"]:
        col = f"density_{name}"
        feat[f"{col}_trend"] = feat[col] - feat[col].rolling(20).mean()

    return feat.fillna(0)

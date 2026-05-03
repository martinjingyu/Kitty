"""
Regime-specific labeling functions.

Intraday : exit-on-first-touch triple-barrier  →  binary LONG / SHORT
Weekly   : asymmetric momentum-filtered barriers →  LONG / SHORT / CONDOR
Monthly  : price-area integral                  →  LONG / SHORT / CONDOR
"""
import numpy as np
import pandas as pd


# ── shared helpers ────────────────────────────────────────────────────────────

def _daily_vol(close: pd.Series, lookback: int = 20) -> pd.Series:
    return np.log(close).diff().ewm(span=lookback).std()


def _path_er(close_arr: np.ndarray, t: int, t_exit: int) -> float:
    path  = close_arr[t: t_exit + 1]
    moves = np.abs(np.diff(path))
    total = moves.sum()
    return 1.0 if total == 0 else abs(path[-1] - path[0]) / total


def _sample_weights(t_enter: np.ndarray, t_exit: np.ndarray) -> np.ndarray:
    if len(t_enter) == 0:
        return np.array([])
    max_t  = int(t_exit.max()) + 2
    counts = np.zeros(max_t, dtype=np.float64)
    for a, b in zip(t_enter, t_exit):
        counts[int(a)] += 1
        if int(b) + 1 < max_t:
            counts[int(b) + 1] -= 1
    concurrent = np.cumsum(counts)
    weights = np.empty(len(t_enter))
    for i, (a, b) in enumerate(zip(t_enter, t_exit)):
        avg = concurrent[int(a): int(b) + 1].mean()
        weights[i] = 1.0 / max(avg, 1.0)
    return weights


def _make_record(t, t_exit, ts, close, vol_t, label):
    return {
        "t":              t,
        "t_exit":         t_exit,
        "timestamp":      ts[t],
        "timestamp_exit": ts[t_exit],
        "label":          label,
        "ret":            float(np.log(close[t_exit] / close[t])),
        "vol":            float(vol_t),
        "er":             _path_er(close, t, t_exit),
    }


# ── INTRADAY: exit-on-first-touch, binary ────────────────────────────────────

def label_intraday(
    bars: pd.DataFrame,
    h: float = 0.5,
    max_hold: int = 20,
    vol_lookback: int = 20,
    stride: int = 1,
) -> pd.DataFrame:
    """
    Classic triple-barrier, exit on first barrier touch.
    Labels: +1 (LONG) / -1 (SHORT) / 0 (organic vertical — very rare, dropped later)
    """
    close = bars["close"].values
    high  = bars["high"].values
    low   = bars["low"].values
    ts    = bars["timestamp"].values
    n     = len(bars)

    vol_series = _daily_vol(bars["close"], lookback=vol_lookback).values

    records = []
    for t in range(vol_lookback, n - 1, stride):
        vol_t = vol_series[t]
        if np.isnan(vol_t) or vol_t == 0:
            continue

        entry   = close[t]
        upper   = entry * (1 + h * vol_t)
        lower   = entry * (1 - h * vol_t)
        horizon = min(t + max_hold, n - 1)

        label  = 0
        t_exit = horizon
        for i in range(t + 1, horizon + 1):
            if high[i] >= upper:
                label, t_exit = 1, i
                break
            if low[i] <= lower:
                label, t_exit = -1, i
                break

        records.append(_make_record(t, t_exit, ts, close, vol_t, label))

    df = pd.DataFrame(records)
    if len(df) > 0:
        df["weight"] = _sample_weights(df["t"].values, df["t_exit"].values)
    return df


# ── WEEKLY: asymmetric barrier, momentum-filtered ─────────────────────────────

def label_weekly_reversal(
    bars: pd.DataFrame,
    h_target: float = 1.5,       # profit-side barrier multiplier (wider)
    h_stop: float   = 0.75,      # loss-side barrier multiplier   (tighter)
    max_hold: int   = 100,
    vol_lookback: int = 100,
    stride: int     = 2,
    mom_window: int = 20,         # bars for prior momentum
    entry_threshold: float = 1.0, # |vol-adj momentum| to trigger dip/top sampling
    condor_threshold: float = 0.3,# |vol-adj momentum| below which condor is sampled
    condor_area_thr: float = 0.3, # max |area_norm| to keep as CONDOR
) -> pd.DataFrame:
    """
    Asymmetric barrier labeling for weekly reversals.

    Event classification by prior vol-adjusted momentum (mom_std):
      mom_std < -entry_threshold  →  dip buy:  h_up=h_target, h_down=h_stop
      mom_std > +entry_threshold  →  top sell: h_up=h_stop,   h_down=h_target
      |mom_std| < condor_threshold→  condor:   kept if area_norm < condor_area_thr

    Labels: +1 (LONG) / -1 (SHORT) / 0 (CONDOR)
    Directional events that hit the vertical barrier (no touch) are skipped.
    """
    close = bars["close"].values
    high  = bars["high"].values
    low   = bars["low"].values
    ts    = bars["timestamp"].values
    n     = len(bars)

    vol_series = _daily_vol(bars["close"], lookback=vol_lookback).values
    log_close  = np.log(close)
    start_bar  = vol_lookback + mom_window

    records = []
    for t in range(start_bar, n - 1, stride):
        vol_t = vol_series[t]
        if np.isnan(vol_t) or vol_t == 0:
            continue

        prior_ret = log_close[t] - log_close[t - mom_window]
        mom_std   = prior_ret / (vol_t * np.sqrt(mom_window))

        entry   = close[t]
        horizon = min(t + max_hold, n - 1)

        # ── condor candidate ─────────────────────────────────────────────────
        if abs(mom_std) < condor_threshold:
            t_exit = horizon
            future = close[t + 1: horizon + 1]
            if len(future) == 0:
                continue
            area = float(np.sum((future - entry) / entry))
            denom = vol_t * max_hold
            area_norm = area / denom if denom > 0 else 0.0
            if abs(area_norm) > condor_area_thr:
                continue  # net drift too large — not a true condor
            records.append(_make_record(t, t_exit, ts, close, vol_t, 0))
            continue

        # ── directional candidate ─────────────────────────────────────────────
        if mom_std < -entry_threshold:
            h_up, h_down = h_target, h_stop   # dip buy: wide profit, tight stop
        elif mom_std > entry_threshold:
            h_up, h_down = h_stop, h_target   # top sell: tight stop, wide profit
        else:
            continue  # intermediate zone — skip

        upper = entry * (1 + h_up   * vol_t)
        lower = entry * (1 - h_down * vol_t)

        label  = 0
        t_exit = horizon
        for i in range(t + 1, horizon + 1):
            if high[i] >= upper:
                label, t_exit = 1, i
                break
            if low[i] <= lower:
                label, t_exit = -1, i
                break

        if label == 0:
            continue  # vertical barrier exit from directional candidate — skip

        records.append(_make_record(t, t_exit, ts, close, vol_t, label))

    df = pd.DataFrame(records)
    if len(df) > 0:
        df["weight"] = _sample_weights(df["t"].values, df["t_exit"].values)
    return df


# ── MONTHLY: price-area integral ──────────────────────────────────────────────

def label_monthly_area(
    bars: pd.DataFrame,
    max_hold: int     = 400,
    vol_lookback: int = 100,
    stride: int       = 5,
    long_thr: float   =  5.0,  # area_norm > long_thr  → LONG  (absolute, regime-agnostic)
    short_thr: float  = -5.0,  # area_norm < short_thr → SHORT
    condor_thr: float =  2.0,  # |area_norm| < condor_thr → CONDOR
) -> pd.DataFrame:
    """
    Monthly labeling via cumulative price-area integral with FIXED thresholds.

        area      = Σ (close[t+i] - close[t]) / close[t]   for i in [1, max_hold]
        area_norm = area / (vol_t × max_hold)               (vol-standardized)

    Labels use absolute thresholds so meaning is regime-agnostic:
        +1 (LONG)  : area_norm ≥ long_thr  — price consistently above entry
        -1 (SHORT) : area_norm ≤ short_thr — price consistently below entry
         0 (CONDOR): |area_norm| ≤ condor_thr — price oscillated near entry
        skip       : intermediate zone (ambiguous, not sampled)

    This ensures SHORT always means area is truly negative, regardless of
    whether the overall market is in a bull or bear regime.
    """
    close = bars["close"].values
    ts    = bars["timestamp"].values
    n     = len(bars)

    vol_series = _daily_vol(bars["close"], lookback=vol_lookback).values

    records = []
    for t in range(vol_lookback, n - max_hold - 1, stride):
        vol_t = vol_series[t]
        if np.isnan(vol_t) or vol_t == 0:
            continue

        entry  = close[t]
        t_exit = t + max_hold
        future = close[t + 1: t_exit + 1]

        area      = float(np.sum((future - entry) / entry))
        denom     = vol_t * max_hold
        area_norm = area / denom if denom > 0 else 0.0

        if area_norm >= long_thr:
            label = 1
        elif area_norm <= short_thr:
            label = -1
        elif abs(area_norm) <= condor_thr:
            label = 0
        else:
            continue  # intermediate zone — skip

        records.append({
            "t":              t,
            "t_exit":         t_exit,
            "timestamp":      ts[t],
            "timestamp_exit": ts[t_exit],
            "label":          label,
            "area_norm":      area_norm,
            "ret":            float(np.log(close[t_exit] / entry)),
            "vol":            float(vol_t),
            "er":             _path_er(close, t, t_exit),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    dist = df["label"].value_counts().sort_index()
    print(f"    fixed thr: short≤{short_thr}  |condor|≤{condor_thr}  long≥{long_thr}  "
          f"→ {len(df):,} samples  "
          f"(-1:{dist.get(-1,0):,}  0:{dist.get(0,0):,}  +1:{dist.get(1,0):,})")

    df["weight"] = _sample_weights(df["t"].values, df["t_exit"].values)
    return df


# ── legacy shim (keeps any external caller working) ───────────────────────────

def label_triple_barrier(bars, h=1.0, max_hold=100, vol_lookback=20,
                         stride=1, neutral_frac=0.0, **_):
    return label_intraday(bars, h=h, max_hold=max_hold,
                          vol_lookback=vol_lookback, stride=stride)

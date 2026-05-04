"""
Real-time signal engine.

Every minute:
  1. Fetch latest 1-min bars via yfinance for each ticker
  2. Feed into DollarBarAccumulator
  3. When a dollar bar closes → compute features → predict_proba
  4. If prob > threshold → emit Signal

Barriers (target / stop) are derived from the same rolling volatility
the model was trained with, so they are consistent with triple-barrier labels.
"""
import os
import pickle
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# Polygon constants kept for pipeline scripts that import them
API_KEY  = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"

# ── per-ticker signal config ──────────────────────────────────────────────────
# (regime, threshold, h, vol_lookback, max_hold_bars)
SIGNAL_CONFIG = {
    # (regime, thr_long, thr_short, h, vol_lookback, max_hold_bars)
    # intraday  thr=0.65: class-balanced model, precision ~72%
    # weekly    thr=0.50: full coverage optimal, precision ~60%
    # monthly   thr=0.54: 3-class model, precision ~53%, CONDOR → no signal
    "SPY":  [("intraday", 0.65, 0.65, 0.5, 20,  20 ),
             ("weekly",   0.50, 0.50, 1.5, 100, 100),
             ("monthly",  0.54, 0.54, 1.5, 100, 400)],
    "AAPL": [("intraday", 0.65, 0.65, 0.5, 20,  20 ),
             ("weekly",   0.50, 0.50, 1.5, 100, 100),
             ("monthly",  0.54, 0.54, 1.5, 100, 400)],
    "NVDA": [("intraday", 0.65, 0.65, 0.5, 20,  20 ),
             ("weekly",   0.50, 0.50, 1.5, 100, 100),
             ("monthly",  0.54, 0.54, 1.5, 100, 400)],
    "TSLA": [("intraday", 0.65, 0.65, 0.5, 20,  20 ),
             ("weekly",   0.50, 0.50, 1.5, 100, 100),
             ("monthly",  0.54, 0.54, 1.5, 100, 400)],
    "SNDK": [("intraday", 0.65, 0.65, 0.5, 20,  20 ),
             ("weekly",   0.50, 0.50, 1.5, 100, 100),
             ("monthly",  0.54, 0.54, 1.5, 100, 400)],
    "CRWV": [("intraday", 0.65, 0.65, 0.5, 20,  20 ),
             ("weekly",   0.50, 0.50, 1.5, 100, 100),
             ("monthly",  0.54, 0.54, 1.5, 100, 400)],
}

MODEL_DIR = Path("models/saved")
BARS_DIR  = Path("data/bars/processed")
HISTORY_LEN = 500   # dollar bars to keep in memory for feature computation


# ── data classes ─────────────────────────────────────────────────────────────

@dataclass
class Signal:
    ticker:        str
    regime:        str
    direction:     str          # "LONG" or "SHORT"
    entry_price:   float
    target_price:  float
    stop_price:    float
    target_pct:    float        # % gain to target
    stop_pct:      float        # % loss to stop
    confidence:    float        # predict_proba
    vol:           float        # current rolling vol (daily)
    timeframe_days: int
    timestamp:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProbaScan:
    ticker:        str
    regime:        str
    price:         float
    proba_long:    float
    proba_short:   float
    proba_neutral: float        # 0.0 for binary (intraday) models
    direction:     str          # "LONG" | "SHORT" | "NEUTRAL"
    confidence:    float
    timestamp:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── yfinance helpers ─────────────────────────────────────────────────────────

def _yf_to_internal(ts_idx, row: "pd.Series") -> dict:
    """Convert a yfinance DataFrame row to the engine's internal minute-bar dict."""
    utc_dt = ts_idx.tz_convert("UTC").to_pydatetime()
    return {
        "timestamp":   utc_dt,
        "open":        float(row["Open"]),
        "high":        float(row["High"]),
        "low":         float(row["Low"]),
        "close":       float(row["Close"]),
        "volume":      float(row["Volume"]),
        "dollar":      float(row["Close"]) * float(row["Volume"]),
        "ticks":       1,
        "buy_dollar":  0.0,
        "sell_dollar": 0.0,
        "theta":       0.0,
    }


def _fetch_yf(ticker: str, since_utc: Optional[datetime] = None) -> "pd.DataFrame":
    """
    Fetch 1-min bars from yfinance.
    Uses period-based requests to avoid timezone issues with yfinance's start param.
    since_utc filtering is done on the returned DataFrame index.
    """
    try:
        # warmup may need several days of history; poll only needs today
        period = "7d" if since_utc is None or (
            datetime.now(timezone.utc) - since_utc
        ).days >= 1 else "1d"
        df = yf.Ticker(ticker).history(period=period, interval="1m")
        if df.empty:
            return df
        if since_utc is not None:
            since_epoch = since_utc.timestamp()
            df = df[df.index.tz_convert("UTC").map(lambda t: t.timestamp()) > since_epoch]
        return df
    except Exception as e:
        log.warning(f"yfinance fetch failed for {ticker}: {e}")
        return pd.DataFrame()


# ── dollar bar accumulator ────────────────────────────────────────────────────

class DollarBarAccumulator:
    """
    Ingests 1-minute OHLCV rows and emits a dollar bar dict
    whenever cumulative dollar volume >= threshold.
    """
    def __init__(self, threshold: float):
        self.threshold = threshold
        self._bucket: list[dict] = []
        self._cum_dollar = 0.0

    def add(self, row: dict) -> Optional[dict]:
        """
        row must have keys: timestamp, open, high, low, close, volume.
        Returns a closed bar dict or None.
        """
        self._bucket.append(row)
        self._cum_dollar += row["close"] * row["volume"]

        if self._cum_dollar >= self.threshold:
            bar = self._close()
            return bar
        return None

    def _close(self) -> dict:
        b = self._bucket
        bar = {
            "timestamp": b[-1]["timestamp"],
            "open":   b[0]["open"],
            "high":   max(r["high"]   for r in b),
            "low":    min(r["low"]    for r in b),
            "close":  b[-1]["close"],
            "volume": sum(r["volume"] for r in b),
            "dollar": self._cum_dollar,
            "ticks":  len(b),
            "buy_dollar":  0.0,   # not tracked in real-time
            "sell_dollar": 0.0,
            "theta":       0.0,
        }
        self._bucket   = []
        self._cum_dollar = 0.0
        return bar


# ── feature computation ───────────────────────────────────────────────────────

def _compute_features(
    history: deque,
    feat_cols: list[str],
    cfg: dict,
    zscore_stats: dict,
) -> Optional[np.ndarray]:
    """
    Build features for the latest dollar bar using the rolling history.
    Applies per-ticker z-score normalization to absolute-scale features.
    Returns a (1, n_features) array or None if history too short.
    """
    from data.features import build_features

    min_len = max(cfg["feat_windows"]) + 5
    if len(history) < min_len:
        return None

    df = pd.DataFrame(list(history))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # density features require other bar types — use zero approximation
    # (models have low density-feature importance, impact is minimal)
    dummy = df[["timestamp"]].copy()

    feats = build_features(
        dollar=df,
        volume=dummy.assign(timestamp=df["timestamp"]),
        runs=dummy.assign(timestamp=df["timestamp"]),
        imbalance=dummy.assign(timestamp=df["timestamp"]),
        windows=cfg["feat_windows"],
        density_window_min=cfg["density_win_min"],
        prefix=cfg["prefix"],
    )

    # apply same z-score normalization used during training
    for col, (mean, std) in zscore_stats.items():
        if col in feats.columns:
            feats[col] = (feats[col] - mean) / std

    last_row = feats.iloc[[-1]][feat_cols].values
    return last_row


def _rolling_vol(history: deque, lookback: int) -> float:
    """EWM std of log returns over the last `lookback` bars."""
    if len(history) < 5:
        return 0.01
    closes = pd.Series([r["close"] for r in history])
    log_ret = np.log(closes).diff()
    return float(log_ret.ewm(span=lookback).std().iloc[-1])


# ── signal engine ─────────────────────────────────────────────────────────────

class SignalEngine:
    def __init__(self):
        self.tickers      = list(SIGNAL_CONFIG.keys())
        self.models       = {}    # {ticker: {regime: cfg_dict}}
        self.accumulators = {}    # {ticker: DollarBarAccumulator}
        self.histories    = {}    # {ticker: deque of dollar bar dicts}
        self.last_ts      = {}    # {ticker: last processed minute timestamp}
        self.last_signal_dir: dict[str, str] = {}  # {f"{ticker}_{regime}": direction}

        self._load_models()
        self._init_histories()
        self._init_accumulators()

    def _load_models(self):
        for ticker in self.tickers:
            self.models[ticker] = {}
            for (regime, thr_long, thr_short, h, vol_lb, max_hold) in SIGNAL_CONFIG[ticker]:
                # prefer the shared multi-ticker model; fall back to per-ticker
                path = MODEL_DIR / f"multi_xgb_{regime}.pkl"
                if not path.exists():
                    path = MODEL_DIR / f"{ticker}_xgb_{regime}.pkl"
                if not path.exists():
                    path = MODEL_DIR / f"{ticker}_rf_{regime}.pkl"
                if not path.exists():
                    log.warning(f"Model not found: {path}")
                    continue
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                cfg = obj["config"]
                zscore_stats = obj.get("zscore_stats", {}).get(ticker, {})
                self.models[ticker][regime] = {
                    "model":        obj["model"],
                    "feat_cols":    obj["features"],
                    "thr_long":     thr_long,
                    "thr_short":    thr_short,
                    "h":            h,
                    "vol_lb":       vol_lb,
                    "max_hold":     max_hold,
                    "zscore_stats": zscore_stats,
                    **cfg,
                }
            log.info(f"Loaded models for {ticker}: {list(self.models[ticker].keys())}")

    def _init_histories(self):
        for ticker in self.tickers:
            path = BARS_DIR / f"{ticker}_dollar_bars.parquet"
            if path.exists():
                df   = pd.read_parquet(path)
                rows = df.tail(HISTORY_LEN).to_dict("records")
                self.histories[ticker] = deque(rows, maxlen=HISTORY_LEN)
            else:
                self.histories[ticker] = deque(maxlen=HISTORY_LEN)
            self.last_ts[ticker] = None

    def _init_accumulators(self, recent_days: int = 20):
        """
        Threshold is derived from the trailing `recent_days` trading days of
        the saved bar history, matching the rolling threshold used when building
        the bars offline.  This avoids anchoring on stale / low-volume periods.
        """
        for ticker in self.tickers:
            path = BARS_DIR / f"{ticker}_dollar_bars.parquet"
            if path.exists():
                df    = pd.read_parquet(path)
                dates = df["timestamp"].apply(
                    lambda t: t.date() if hasattr(t, "date") else pd.Timestamp(t).date()
                )
                all_dates    = sorted(dates.unique())
                window_dates = set(all_dates[-recent_days:])
                recent       = df[dates.isin(window_dates)]
                n_recent     = len(window_dates)
                avg_daily_dv = float(recent["dollar"].sum()) / max(n_recent, 1)
                threshold    = avg_daily_dv / 20
            else:
                threshold = 1e8   # fallback
            self.accumulators[ticker] = DollarBarAccumulator(threshold)

    def _last_bar_utc(self, ticker: str) -> Optional[datetime]:
        """Return the UTC datetime of the most recent dollar bar in history."""
        hist = self.histories.get(ticker)
        if not hist:
            return None
        ts = hist[-1]["timestamp"]
        if isinstance(ts, pd.Timestamp):
            return ts.tz_convert("UTC").to_pydatetime()
        if isinstance(ts, datetime):
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        return None

    def _save_bars(self, ticker: str, new_bars: list):
        """Append new dollar bars to the parquet file on disk."""
        path = BARS_DIR / f"{ticker}_dollar_bars.parquet"
        new_df = pd.DataFrame(new_bars)
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_parquet(path, index=False)
        else:
            new_df.to_parquet(path, index=False)
        log.info(f"{ticker}: saved {len(new_bars)} new dollar bars to parquet")

    def _ingest_df(self, ticker: str, df: pd.DataFrame) -> list:
        """
        Feed a yfinance minute-bar DataFrame into the accumulator.
        Skips rows already seen (based on last_ts).
        Returns list of newly closed dollar bars.
        """
        new_dollar_bars = []
        for ts_idx, row in df.iterrows():
            epoch_ms = int(ts_idx.tz_convert("UTC").timestamp() * 1000)
            if self.last_ts[ticker] and epoch_ms <= self.last_ts[ticker]:
                continue
            self.last_ts[ticker] = epoch_ms
            bar_row = _yf_to_internal(ts_idx, row)
            new_bar = self.accumulators[ticker].add(bar_row)
            if new_bar:
                self.histories[ticker].append(new_bar)
                new_dollar_bars.append(new_bar)
        return new_dollar_bars

    def warmup(self):
        """
        Gap-fill: fetch minute bars from the last saved dollar bar up to now,
        close new dollar bars, and persist them to parquet.
        Called once at startup in a thread executor.
        """
        for ticker in self.tickers:
            try:
                last_utc = self._last_bar_utc(ticker)
                df = _fetch_yf(ticker, since_utc=last_utc)
                if df.empty:
                    log.info(f"{ticker}: warmup — no new bars since {last_utc}")
                    continue
                new_bars = self._ingest_df(ticker, df)
                if new_bars:
                    self._save_bars(ticker, new_bars)
                latest = self.histories[ticker][-1]
                log.info(f"{ticker}: warmup — {len(new_bars)} new dollar bars, "
                         f"histories={len(self.histories[ticker])}, "
                         f"latest close={latest['close']:.2f} @ {latest['timestamp']}")
            except Exception as e:
                log.warning(f"{ticker}: warmup failed: {e}", exc_info=True)

    # ── public API ────────────────────────────────────────────────────────────

    def poll(self) -> list[Signal]:
        """
        Fetch new 1-min bars since last_ts for each ticker.
        Closes new dollar bars, saves them to parquet, and returns any Signals.
        """
        log.info("poll() called")
        all_events = []
        for ticker in self.tickers:
            try:
                last_utc = (datetime.fromtimestamp(self.last_ts[ticker] / 1000, tz=timezone.utc)
                            if self.last_ts[ticker] else None)
                df = _fetch_yf(ticker, since_utc=last_utc)
                if df.empty:
                    continue
                new_bars = self._ingest_df(ticker, df)
                if new_bars:
                    self._save_bars(ticker, new_bars)
                    log.info(f"{ticker}: poll — {len(new_bars)} new dollar bars, "
                             f"latest close={new_bars[-1]['close']:.2f}")
                    for bar in new_bars:
                        all_events.extend(self._predict(ticker, bar))
            except Exception as e:
                log.warning(f"{ticker}: poll failed: {e}", exc_info=True)
        return all_events

    def _predict(self, ticker: str, bar: dict) -> list[Signal]:
        results = []
        for regime, cfg in self.models.get(ticker, {}).items():
            feats = _compute_features(
                self.histories[ticker], cfg["feat_cols"], cfg,
                zscore_stats=cfg.get("zscore_stats", {}),
            )
            if feats is None:
                continue

            # LabelEncoder maps original labels in sorted order:
            # binary [-1,+1] → [0,1]:  proba[:,1]=P(LONG), proba[:,0]=P(SHORT)
            # 3-class [-1,0,+1] → [0,1,2]: proba[:,-1]=P(LONG), proba[:,0]=P(SHORT)
            proba_arr   = cfg["model"].predict_proba(feats)[0]
            proba_long  = float(proba_arr[-1])   # highest encoded = original +1
            proba_short = float(proba_arr[0])    # lowest  encoded = original -1

            if proba_long > cfg["thr_long"]:
                direction  = "LONG"
                confidence = proba_long
            elif proba_short > cfg["thr_short"]:
                direction  = "SHORT"
                confidence = proba_short
            else:
                continue

            entry    = bar["close"]
            vol      = _rolling_vol(self.histories[ticker], cfg["vol_lb"])
            h_target = cfg.get("h_target", cfg.get("h", 1.0))
            h_stop   = cfg.get("h_stop",   cfg.get("h", 1.0))

            if direction == "LONG":
                target = entry * (1 + h_target * vol)
                stop   = entry * (1 - h_stop   * vol)
            else:
                target = entry * (1 - h_target * vol)
                stop   = entry * (1 + h_stop   * vol)

            target_pct = (target / entry - 1) * 100
            stop_pct   = (stop   / entry - 1) * 100
            days = int(cfg["max_hold"] / 20)  # 20 dollar bars ≈ 1 trading day

            # Only emit a signal when the recommended direction changes
            sig_key = f"{ticker}_{regime}"
            if self.last_signal_dir.get(sig_key) == direction:
                continue
            self.last_signal_dir[sig_key] = direction

            results.append(Signal(
                ticker=ticker,
                regime=regime,
                direction=direction,
                entry_price=round(entry, 2),
                target_price=round(target, 2),
                stop_price=round(stop, 2),
                target_pct=round(target_pct, 2),
                stop_pct=round(stop_pct, 2),
                confidence=round(confidence * 100, 1),
                vol=round(vol * 100, 3),
                timeframe_days=days,
            ))

        return results

    def get_proba(self, ticker: str, regime: str) -> tuple[float, float, float] | None:
        """
        Return (proba_long, proba_short, proba_condor) for one ticker/regime,
        or None if history is too short.  CPU-bound — call in executor.
        """
        cfg = self.models.get(ticker, {}).get(regime)
        if cfg is None or not self.histories.get(ticker):
            return None
        feats = _compute_features(
            self.histories[ticker], cfg["feat_cols"], cfg,
            zscore_stats=cfg.get("zscore_stats", {}),
        )
        if feats is None:
            return None
        proba = cfg["model"].predict_proba(feats)[0]
        return (float(proba[-1]),
                float(proba[0]),
                float(proba[1]) if len(proba) > 2 else 0.0)

    def scan(self) -> list[ProbaScan]:
        """
        Return current predict_proba for every ticker/regime without threshold
        filtering.  Called by the 30-min scan loop for a market-wide snapshot.
        """
        results = []
        for ticker in self.tickers:
            if not self.histories[ticker]:
                continue
            price = self.histories[ticker][-1]["close"]

            for regime, cfg in self.models.get(ticker, {}).items():
                feats = _compute_features(
                    self.histories[ticker], cfg["feat_cols"], cfg,
                    zscore_stats=cfg.get("zscore_stats", {}),
                )
                if feats is None:
                    continue

                proba_arr = cfg["model"].predict_proba(feats)[0]
                # proba columns: sorted encoded labels → [P(SHORT), ...mid..., P(LONG)]
                proba_long    = float(proba_arr[-1])
                proba_short   = float(proba_arr[0])
                proba_neutral = float(proba_arr[1]) if len(proba_arr) > 2 else 0.0

                if proba_long >= proba_short and proba_long >= proba_neutral:
                    direction  = "LONG"
                    confidence = proba_long
                elif proba_short >= proba_long and proba_short >= proba_neutral:
                    direction  = "SHORT"
                    confidence = proba_short
                else:
                    direction  = "NEUTRAL"
                    confidence = proba_neutral

                results.append(ProbaScan(
                    ticker=ticker,
                    regime=regime,
                    price=round(price, 2),
                    proba_long=round(proba_long, 3),
                    proba_short=round(proba_short, 3),
                    proba_neutral=round(proba_neutral, 3),
                    direction=direction,
                    confidence=round(confidence, 3),
                ))
        return results

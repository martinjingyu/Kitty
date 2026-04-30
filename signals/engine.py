"""
Real-time signal engine.

Every minute:
  1. Fetch latest 1-min bar from Polygon for each ticker
  2. Feed into DollarBarAccumulator
  3. When a dollar bar closes → compute features → predict_proba
  4. If prob > threshold → emit Signal

Barriers (target / stop) are derived from the same rolling volatility
the model was trained with, so they are consistent with triple-barrier labels.
"""
import os
import pickle
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

API_KEY  = os.getenv("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"

# ── per-ticker signal config ──────────────────────────────────────────────────
# (regime, threshold, h, vol_lookback, max_hold_bars)
SIGNAL_CONFIG = {
    "SPY":  [
        ("weekly",  0.63, 1.0, 100, 100),
        ("monthly", 0.65, 2.0, 100, 400),
    ],
    "SNDK": [
        ("weekly",  0.63, 1.0, 100, 100),
    ],
    "CRWV": [
        ("weekly",  0.60, 1.0, 100, 100),
        ("monthly", 0.63, 2.0, 100, 400),
    ],
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


# ── polygon REST helper ───────────────────────────────────────────────────────

def _fetch_latest_minute(ticker: str, n_bars: int = 3) -> list:
    """Fetch the last n_bars 1-minute aggregates for ticker."""
    now   = datetime.now(timezone.utc)
    start = (now - timedelta(minutes=n_bars + 5)).strftime("%Y-%m-%d")
    end   = now.strftime("%Y-%m-%d")
    url   = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}"
    try:
        resp = requests.get(url, params={
            "adjusted": "true", "sort": "asc",
            "limit": n_bars + 5, "apiKey": API_KEY,
        }, timeout=10)
        data = resp.json()
        return data.get("results", [])[-n_bars:]
    except Exception as e:
        log.warning(f"Polygon fetch failed for {ticker}: {e}")
        return []


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

def _compute_features(history: deque, feat_cols: list[str], cfg: dict) -> Optional[np.ndarray]:
    """
    Build features for the latest dollar bar using the rolling history.
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
        self.models       = {}    # {ticker: {regime: (model, feat_cols, cfg)}}
        self.accumulators = {}    # {ticker: DollarBarAccumulator}
        self.histories    = {}    # {ticker: deque of dollar bar dicts}
        self.last_ts      = {}    # {ticker: last processed minute timestamp}

        self._load_models()
        self._init_histories()
        self._init_accumulators()

    def _load_models(self):
        for ticker in self.tickers:
            self.models[ticker] = {}
            for (regime, thr, h, vol_lb, max_hold) in SIGNAL_CONFIG[ticker]:
                path = MODEL_DIR / f"{ticker}_rf_{regime}.pkl"
                if not path.exists():
                    log.warning(f"Model not found: {path}")
                    continue
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                cfg = obj["config"]
                self.models[ticker][regime] = {
                    "model":     obj["model"],
                    "feat_cols": obj["features"],
                    "threshold": thr,
                    "h":         h,
                    "vol_lb":    vol_lb,
                    "max_hold":  max_hold,
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

    def _init_accumulators(self):
        for ticker in self.tickers:
            path = BARS_DIR / f"{ticker}_dollar_bars.parquet"
            if path.exists():
                df  = pd.read_parquet(path)
                n_days = df["timestamp"].apply(
                    lambda t: t.date() if hasattr(t, "date") else pd.Timestamp(t).date()
                ).nunique()
                avg_daily_dv  = float((df["dollar"]).sum()) / max(n_days, 1)
                threshold     = avg_daily_dv / 20
            else:
                threshold = 1e8   # fallback
            self.accumulators[ticker] = DollarBarAccumulator(threshold)

    # ── public API ────────────────────────────────────────────────────────────

    def poll(self) -> list[Signal]:
        """
        Fetch latest minute bars for all tickers.
        Returns a (possibly empty) list of Signal objects.
        """
        signals = []
        for ticker in self.tickers:
            raw_bars = _fetch_latest_minute(ticker, n_bars=3)
            for rb in raw_bars:
                ts = rb["t"]   # epoch ms
                if self.last_ts[ticker] and ts <= self.last_ts[ticker]:
                    continue   # already processed
                self.last_ts[ticker] = ts

                row = {
                    "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                    "open":   rb["o"],
                    "high":   rb["h"],
                    "low":    rb["l"],
                    "close":  rb["c"],
                    "volume": rb["v"],
                    "dollar": rb["c"] * rb["v"],
                    "ticks":  rb.get("n", 1),
                    "buy_dollar": 0.0, "sell_dollar": 0.0, "theta": 0.0,
                }

                new_bar = self.accumulators[ticker].add(row)
                if new_bar is None:
                    continue

                self.histories[ticker].append(new_bar)
                sigs = self._predict(ticker, new_bar)
                signals.extend(sigs)

        return signals

    def _predict(self, ticker: str, bar: dict) -> list[Signal]:
        results = []
        for regime, cfg in self.models.get(ticker, {}).items():
            feats = _compute_features(
                self.histories[ticker], cfg["feat_cols"], cfg
            )
            if feats is None:
                continue

            proba = cfg["model"].predict_proba(feats)[0, 1]   # P(Long)
            thr   = cfg["threshold"]

            if proba > thr:
                direction = "LONG"
            elif proba < (1 - thr):
                direction = "SHORT"
                proba     = 1 - proba   # flip to confidence in direction
            else:
                continue

            entry = bar["close"]
            vol   = _rolling_vol(self.histories[ticker], cfg["vol_lb"])
            h     = cfg["h"]

            target = entry * (1 + h * vol)
            stop   = entry * (1 - h * vol)
            if direction == "SHORT":
                target, stop = stop, target

            target_pct = (target / entry - 1) * 100
            stop_pct   = (stop   / entry - 1) * 100

            days = int(cfg["max_hold"] / 20)  # 20 dollar bars ≈ 1 trading day

            results.append(Signal(
                ticker=ticker,
                regime=regime,
                direction=direction,
                entry_price=round(entry, 2),
                target_price=round(target, 2),
                stop_price=round(stop, 2),
                target_pct=round(target_pct, 2),
                stop_pct=round(stop_pct, 2),
                confidence=round(proba * 100, 1),
                vol=round(vol * 100, 3),
                timeframe_days=days,
            ))

        return results

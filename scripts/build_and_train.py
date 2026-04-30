"""
Build datasets and train models for two holding-period regimes:

  weekly  — max_hold = 100 bars (~1 week,  5 trading days × 20 bars/day)
  monthly — max_hold = 400 bars (~1 month, 20 trading days × 20 bars/day)

Each regime gets its own:
  - triple-barrier labels (different max_hold, h, vol_lookback)
  - feature rolling windows scaled to the holding period
  - RandomForest model + threshold sweep
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data.labels   import label_triple_barrier
from data.features import build_features
from models.cv     import PurgedKFold

BARS_DIR  = Path("data/bars/processed")
OUT_DIR   = Path("data/dataset")
MODEL_DIR = Path("models/saved")
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_RATIO = 0.7
THRESHOLDS  = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.63, 0.65]

META_COLS = {"t", "t_exit", "timestamp", "timestamp_exit",
             "label", "ret", "vol", "weight"}

CONFIGS = {
    "weekly": {
        # horizon: 5 days × 20 bars/day = 100 bars
        # stride=20 → predict once per trading day → ~500 samples
        # features look back 1 month (400 bars) to give weekly predictions
        # a broader market context, matching the monthly model's input depth
        "max_hold":         100,
        "stride":           20,
        "h":                1.0,
        "vol_lookback":     100,
        "feat_windows":     [50, 100, 400],  # short/mid/long = 2.5d/1w/1mo
        "density_win_min":  1950,
        "prefix":           "d_",
    },
    "monthly": {
        # horizon: 20 days × 20 bars/day = 400 bars
        # stride=20 → predict once per trading day → ~500 samples
        # features still look back 1 month — monthly signal in daily frequency
        "max_hold":         400,
        "stride":           20,
        "h":                2.0,
        "vol_lookback":     100,
        "feat_windows":     [100, 200, 400],  # lookback ≈ 1 month
        "density_win_min":  1950,
        "prefix":           "d_",
    },
}


# ── helpers ──────────────────────────────────────────────────────────────────

def sharpe(returns: pd.Series, stride: int = 20) -> float:
    # periods_per_year = trading days per year / (stride / bars_per_day)
    periods_per_year = int(252 * 20 / stride)
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def threshold_sweep(model, X, y, ret, stride: int) -> pd.DataFrame:
    proba = model.predict_proba(X)[:, 1]
    rows  = []
    for thr in THRESHOLDS:
        long_mask  = proba >  thr
        short_mask = proba < (1 - thr)
        mask       = long_mask | short_mask
        n = mask.sum()
        if n == 0:
            rows.append({"thr": thr, "n_trades": 0, "coverage": 0,
                         "precision": np.nan, "sharpe": np.nan, "cum_ret_%": np.nan})
            continue
        direction = np.where(long_mask, 1, np.where(short_mask, -1, 0))
        strat_ret = pd.Series(direction[mask] * ret.values[mask])
        correct   = direction[mask] == np.where(y[mask] == 1, 1, -1)
        rows.append({
            "thr":        thr,
            "n_trades":   n,
            "coverage":   round(n / len(proba), 3),
            "precision":  round(correct.mean(), 4),
            "sharpe":     round(sharpe(strat_ret, stride=stride), 3),
            "cum_ret_%":  round((np.exp(strat_ret.sum()) - 1) * 100, 2),
        })
    return pd.DataFrame(rows)


def cv_score(model, df, feat_cols):
    X = df[feat_cols].values
    y = (df["label"].values + 1) // 2
    t, t_exit, w = df["t"].values, df["t_exit"].values, df["weight"].values
    pkf = PurgedKFold(n_splits=5, embargo=10)
    accs = []
    for fold, (tr, te) in enumerate(pkf.split(t, t_exit)):
        model.fit(X[tr], y[tr], sample_weight=w[tr])
        acc = (model.predict(X[te]) == y[te]).mean()
        accs.append(acc)
        print(f"    fold {fold+1}: acc={acc:.3f}  "
              f"(train={len(tr):,}  test={len(te):,})")
    print(f"    CV mean: {np.mean(accs):.3f} ± {np.std(accs):.3f}")


# ── main ─────────────────────────────────────────────────────────────────────

def run_regime(regime: str, cfg: dict, bars: dict, ticker: str = "SPY"):
    print(f"\n{'█'*60}")
    print(f"  {ticker}  ·  REGIME: {regime.upper()}"
          f"  (max_hold={cfg['max_hold']} bars, h={cfg['h']})")
    print(f"{'█'*60}")

    dollar = bars["dollar"]

    # ── labels ───────────────────────────────────────────────────────────────
    print("\n[1/4] Labelling ...")
    labels = label_triple_barrier(
        dollar,
        h=cfg["h"],
        max_hold=cfg["max_hold"],
        vol_lookback=cfg["vol_lookback"],
        stride=cfg["stride"],
    )
    labels = labels[labels["label"] != 0].copy()
    dist   = labels["label"].value_counts().sort_index()
    print(f"  events: {len(labels):,}  "
          f"| -1: {dist.get(-1,0):,}  0: {dist.get(0,0):,}  +1: {dist.get(1,0):,}")

    # ── features ─────────────────────────────────────────────────────────────
    print("\n[2/4] Building features ...")
    feats = build_features(
        dollar=dollar,
        volume=bars["volume"],
        runs=bars["runs"],
        imbalance=bars["imbalance"],
        windows=cfg["feat_windows"],
        density_window_min=cfg["density_win_min"],
        prefix=cfg["prefix"],
    )
    print(f"  {feats.shape[1]} features")

    # ── merge + split ─────────────────────────────────────────────────────────
    dataset   = labels.join(feats, on="t").dropna()
    feat_cols = [c for c in dataset.columns if c not in META_COLS]
    split     = int(len(dataset) * TRAIN_RATIO)
    train     = dataset.iloc[:split]
    test      = dataset.iloc[split:]
    print(f"  train: {len(train):,} "
          f"({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"  test:  {len(test):,}  "
          f"({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")

    dataset.to_parquet(OUT_DIR / f"{ticker}_{regime}_dataset.parquet", index=False)

    MIN_TRAIN = 80
    if len(train) < MIN_TRAIN:
        print(f"\n  ⚠ Only {len(train)} train samples (< {MIN_TRAIN}) — skipping model training")
        return

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Purged 5-Fold CV ...")
    # fewer trees / shallower for small datasets to reduce overfitting
    n_est = 300 if len(train) >= 300 else 100
    model = RandomForestClassifier(
        n_estimators=n_est, max_depth=5, min_samples_leaf=max(10, len(train)//30),
        max_features="sqrt", n_jobs=-1, random_state=42,
    )
    cv_score(model, train, feat_cols)

    X_train = train[feat_cols].values
    y_train = (train["label"].values + 1) // 2
    w_train = train["weight"].values
    model.fit(X_train, y_train, sample_weight=w_train)

    # ── threshold sweep ───────────────────────────────────────────────────────
    print(f"\n[4/4] Threshold sweep (test set) ...")
    X_test   = test[feat_cols].values
    y_test   = (test["label"].values + 1) // 2
    ret_test = test["ret"].reset_index(drop=True)

    sweep = threshold_sweep(model, X_test, y_test, ret_test, stride=cfg["stride"])
    print(f"\n  {'thr':>5}  {'trades':>7}  {'cov':>6}  "
          f"{'prec':>6}  {'sharpe':>7}  {'cum_ret%':>9}")
    print(f"  {'─'*53}")
    for _, row in sweep.iterrows():
        marker = "  ◄" if row["sharpe"] == sweep["sharpe"].max() else ""
        print(f"  {row['thr']:>5.2f}  {int(row['n_trades']):>7,}  "
              f"{row['coverage']:>6.1%}  {row['precision']:>6.3f}  "
              f"{row['sharpe']:>7.3f}  {row['cum_ret_%']:>8.2f}%{marker}")

    # ── save ──────────────────────────────────────────────────────────────────
    with open(MODEL_DIR / f"{ticker}_rf_{regime}.pkl", "wb") as f:
        pickle.dump({"model": model, "features": feat_cols, "config": cfg,
                     "ticker": ticker}, f)
    sweep.to_csv(MODEL_DIR / f"{ticker}_rf_{regime}_sweep.csv", index=False)
    print(f"\n  Model saved → models/saved/{ticker}_rf_{regime}.pkl")


def train_ticker(ticker: str, bars: dict):
    for regime, cfg in CONFIGS.items():
        run_regime(regime, cfg, bars, ticker=ticker)
    print(f"\n{'═'*60}")
    print(f"  {ticker} — done.")
    print(f"{'═'*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="SPY")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    print(f"Loading bars for {ticker} ...")
    bars = {
        "dollar":    pd.read_parquet(BARS_DIR / f"{ticker}_dollar_bars.parquet"),
        "volume":    pd.read_parquet(BARS_DIR / f"{ticker}_volume_bars.parquet"),
        "runs":      pd.read_parquet(BARS_DIR / f"{ticker}_runs_bars.parquet"),
        "imbalance": pd.read_parquet(BARS_DIR / f"{ticker}_imbalance_bars.parquet"),
    }
    for k, v in bars.items():
        print(f"  {k}: {len(v):,} bars")

    train_ticker(ticker, bars)


if __name__ == "__main__":
    main()

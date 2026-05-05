"""
Build datasets and train ONE XGBoost model per regime across all tickers.

  intraday — exit-on-first-touch triple-barrier, binary LONG/SHORT
  weekly   — asymmetric momentum-filtered barriers, 3-class LONG/SHORT/CONDOR
  monthly  — price-area integral labeling,         3-class LONG/SHORT/CONDOR
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from data.labels import (
    label_intraday,
    label_weekly_reversal,
    label_monthly_area,
)
from data.features import build_features
from models.cv     import PurgedKFold

BARS_DIR  = Path("data/bars/processed")
OUT_DIR   = Path("data/dataset")
MODEL_DIR = Path("models/saved")
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
INTRADAY_BOUNDARY_PATH = MODEL_DIR / "intraday_boundary_sweep_top.csv"

TRAIN_RATIO = 0.7
THRESHOLDS_BINARY = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.63, 0.65]
THRESHOLDS_MULTI  = [0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.54]

META_COLS = {"t", "t_exit", "timestamp", "timestamp_exit",
             "label", "ret", "vol", "er", "weight", "ticker",
             "t_seq", "t_exit_seq", "area_norm"}

# absolute-scale features that differ by ticker and must be z-scored per ticker
ABS_FEATURES = ["d_log_volume", "d_log_dollar"]
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "TSLA", "WMT", "MS", "JPM", "BE", "PLTR",
    "SPY", "IBM", "IWM", "MU", "SNDK", "CRWV",
    "NBIS", "INTC", "AMD", "ORCL", "COIN", "MSTR",
]

CONFIGS = {
    "intraday": {
        # exit-on-first-touch, binary LONG/SHORT
        "label_fn":         "intraday",
        "max_hold":         20,
        "stride":           1,
        "h":                0.5,
        "min_target_return": 0.01,
        "vol_lookback":     20,
        "feat_windows":     [20, 50, 100, 400],
        "density_win_min":  60,
        "prefix":           "d_",
        "directional_only": True,   # drop rare organic 0s
    },
    "weekly": {
        # asymmetric momentum-filtered triple-barrier, binary LONG/SHORT
        # condor handled by monthly only
        "label_fn":          "weekly",
        "max_hold":          100,
        "stride":            2,
        "h_target":          1.5,   # profit barrier (wide side)
        "h_stop":            0.75,  # stop barrier   (tight side)
        "min_target_return": 0.05,
        "vol_lookback":      100,
        "mom_window":        20,
        "entry_threshold":   1.0,
        "condor_threshold":  0.3,
        "condor_area_thr":   0.3,
        "feat_windows":      [50, 100, 400],
        "density_win_min":   1950,
        "prefix":            "d_",
        "directional_only":  True,
    },
    "monthly": {
        # price-area integral, fixed thresholds, 3-class
        "label_fn":          "monthly",
        "max_hold":          400,
        "stride":            5,
        "vol_lookback":      100,
        "long_thr":          5.0,   # area_norm > +5 → LONG
        "short_thr":        -5.0,   # area_norm < -5 → SHORT
        "condor_thr":        2.0,   # |area_norm| < 2 → CONDOR
        "min_target_return": 0.10,
        "feat_windows":      [100, 200, 400],
        "density_win_min":   1950,
        "prefix":            "d_",
        "directional_only":  False,
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _assign_seq_indices(dataset: pd.DataFrame, timestamp_exit_col: str = "timestamp_exit") -> pd.DataFrame:
    """
    Add t_seq (row position) and t_exit_seq (row position of exit timestamp)
    to a time-sorted combined dataset.  Using row indices keeps embargo in
    bar units, matching max_hold, regardless of variable dollar-bar duration.
    """
    ts     = dataset["timestamp"].values
    ts_exit = pd.to_datetime(dataset[timestamp_exit_col]).values.astype("int64")
    ts_int  = pd.to_datetime(ts).values.astype("int64")

    t_seq      = np.arange(len(dataset))
    # for each event, find how many rows have entry timestamp <= its exit timestamp
    t_exit_seq = np.searchsorted(ts_int, ts_exit, side="right") - 1
    t_exit_seq = np.clip(t_exit_seq, 0, len(dataset) - 1)

    out = dataset.copy()
    out["t_seq"]      = t_seq
    out["t_exit_seq"] = t_exit_seq
    return out


def load_intraday_boundaries(path: str | Path | None) -> dict[str, dict]:
    """
    Load per-ticker asymmetric intraday barriers from a sweep top CSV.

    Expected columns come from scripts/sweep_intraday_boundaries.py:
    ticker,h_up,h_down,min_up_%,min_down_%.
    The first row per ticker is used, so pass the *_top.csv output sorted by
    score descending.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        print(f"  Intraday boundary file not found: {path} — using default config")
        return {}

    df = pd.read_csv(path)
    required = {"ticker", "h_up", "h_down", "min_up_%", "min_down_%"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    boundaries = {}
    for _, row in df.drop_duplicates("ticker", keep="first").iterrows():
        ticker = str(row["ticker"]).upper()
        boundaries[ticker] = {
            "h_up": float(row["h_up"]),
            "h_down": float(row["h_down"]),
            "min_up_return": float(row["min_up_%"]) / 100,
            "min_down_return": float(row["min_down_%"]) / 100,
        }
    print(f"  Loaded intraday boundary overrides for {len(boundaries)} ticker(s) from {path}")
    return boundaries


def _intraday_boundary_for_ticker(cfg: dict, ticker: str) -> dict:
    overrides = cfg.get("boundary_overrides", {})
    override = overrides.get(ticker.upper(), {})
    return {
        "h_up": override.get("h_up", cfg.get("h_up", cfg["h"])),
        "h_down": override.get("h_down", cfg.get("h_down", cfg["h"])),
        "min_up_return": override.get(
            "min_up_return",
            cfg.get("min_up_return", cfg["min_target_return"]),
        ),
        "min_down_return": override.get(
            "min_down_return",
            cfg.get("min_down_return", cfg["min_target_return"]),
        ),
    }


def select_regime_configs(regimes: list[str] | None = None) -> dict[str, dict]:
    """Return copied configs for requested regimes, preserving CONFIGS order."""
    if not regimes:
        requested = list(CONFIGS.keys())
    else:
        requested = [r.lower() for r in regimes]
    unknown = sorted(set(requested) - set(CONFIGS.keys()))
    if unknown:
        raise ValueError(f"Unknown regime(s): {unknown}. Valid: {list(CONFIGS.keys())}")
    return {name: CONFIGS[name].copy() for name in CONFIGS if name in requested}


def sharpe(returns: pd.Series, stride: int = 20) -> float:
    periods_per_year = int(252 * 20 / stride)
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def threshold_sweep(model, X, y, ret, stride: int) -> pd.DataFrame:
    """
    Sweep confidence thresholds for directional signals (LONG / SHORT).

    Model classes are encoded by LabelEncoder (smallest label → 0, largest → n-1).
    Original label ordering: -1 < 0 < 1, so:
      proba[:, 0]  = P(SHORT)  (encoded class 0 = original -1)
      proba[:, -1] = P(LONG)   (encoded class n-1 = original +1)
    """
    proba_all   = model.predict_proba(X)
    n_cls       = proba_all.shape[1]
    proba_long  = proba_all[:, -1]   # P(LONG):  highest encoded = original +1
    proba_short = proba_all[:, 0]    # P(SHORT): lowest  encoded = original -1
    thresholds  = THRESHOLDS_MULTI if n_cls > 2 else THRESHOLDS_BINARY

    # encoded LONG = n_cls-1, encoded SHORT = 0
    true_dir_all = np.where(y == n_cls - 1, 1, np.where(y == 0, -1, 0))

    rows = []
    for thr in thresholds:
        long_mask  = proba_long  > thr
        short_mask = proba_short > thr
        mask       = long_mask | short_mask
        n = int(mask.sum())
        if n == 0:
            rows.append({"thr": thr, "n_trades": 0, "coverage": 0,
                         "precision": np.nan, "sharpe": np.nan, "mean_ret_%": np.nan})
            continue
        direction = np.where(long_mask[mask], 1, -1)
        strat_ret = pd.Series(direction * ret.values[mask])
        correct   = direction == true_dir_all[mask]
        mean_ret = round(float(strat_ret.mean()) * 100, 4)
        rows.append({
            "thr":          thr,
            "n_trades":     n,
            "coverage":     round(n / len(proba_long), 3),
            "precision":    round(float(correct.mean()), 4),
            "sharpe":       round(sharpe(strat_ret, stride=stride), 3),
            "mean_ret_%":   mean_ret,
        })
    return pd.DataFrame(rows)


def cv_score(model, df, feat_cols, max_hold: int, le: LabelEncoder):
    X = df[feat_cols].values
    y = le.transform(df["label"].values)
    w = df["weight"].values
    t      = df["t_seq"].values
    t_exit = df["t_exit_seq"].values
    pkf = PurgedKFold(n_splits=5, embargo=max_hold)
    accs = []
    for fold, (tr, te) in enumerate(pkf.split(t, t_exit)):
        w_tr = _class_balanced_weights(y[tr], w[tr])
        model.fit(X[tr], y[tr], sample_weight=w_tr)
        acc = (model.predict(X[te]) == y[te]).mean()
        accs.append(acc)
        print(f"    fold {fold+1}: acc={acc:.3f}  "
              f"(train={len(tr):,}  test={len(te):,})")
    print(f"    CV mean: {np.mean(accs):.3f} ± {np.std(accs):.3f}")


def _class_balanced_weights(y: np.ndarray, base_weights: np.ndarray) -> np.ndarray:
    """Multiply overlap weights by class-frequency inverse so minority classes
    aren't swamped by the majority.  Output is normalized to mean=1."""
    classes, counts = np.unique(y, return_counts=True)
    inv_freq = dict(zip(classes, counts.sum() / (len(classes) * counts)))
    cls_w = np.array([inv_freq[yi] for yi in y])
    combined = base_weights * cls_w
    combined /= combined.mean()
    return combined


def _make_xgb(n_samples: int, n_classes: int) -> XGBClassifier:
    extra = {}
    if n_classes > 2:
        extra["objective"] = "multi:softprob"
        extra["num_class"] = n_classes
    else:
        extra["objective"] = "binary:logistic"
    return XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=max(3, n_samples // 300),   # was //60 → too high for minority classes
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        **extra,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def run_regime(regime: str, cfg: dict, all_bars: dict, tickers: list):
    if regime == "intraday":
        h_str = "asymmetric" if cfg.get("boundary_overrides") else f"h={cfg['h']}"
    else:
        h_str = f"h={cfg['h']}" if 'h' in cfg else f"h_target={cfg.get('h_target','?')}"
    print(f"\n{'█'*60}")
    print(f"  REGIME: {regime.upper()}  (max_hold={cfg['max_hold']} bars, {h_str})")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"{'█'*60}")

    # ── per-ticker labels + features ─────────────────────────────────────────
    print("\n[1/4] Labelling + features ...")
    ticker_datasets = []
    zscore_stats    = {}   # {ticker: {col: (mean, std)}}

    for ticker in tickers:
        dollar = all_bars[ticker]["dollar"]

        label_fn = cfg["label_fn"]
        boundary = None
        if label_fn == "intraday":
            boundary = _intraday_boundary_for_ticker(cfg, ticker)
            labels = label_intraday(
                dollar,
                h=cfg["h"],
                h_up=boundary["h_up"],
                h_down=boundary["h_down"],
                max_hold=cfg["max_hold"],
                vol_lookback=cfg["vol_lookback"], stride=cfg["stride"],
                min_target_return=cfg["min_target_return"],
                min_up_return=boundary["min_up_return"],
                min_down_return=boundary["min_down_return"],
            )
        elif label_fn == "weekly":
            labels = label_weekly_reversal(
                dollar,
                h_target=cfg["h_target"], h_stop=cfg["h_stop"],
                max_hold=cfg["max_hold"], vol_lookback=cfg["vol_lookback"],
                stride=cfg["stride"], mom_window=cfg["mom_window"],
                entry_threshold=cfg["entry_threshold"],
                condor_threshold=cfg["condor_threshold"],
                condor_area_thr=cfg["condor_area_thr"],
                min_target_return=cfg["min_target_return"],
            )
        elif label_fn == "monthly":
            labels = label_monthly_area(
                dollar,
                max_hold=cfg["max_hold"], vol_lookback=cfg["vol_lookback"],
                stride=cfg["stride"], long_thr=cfg["long_thr"],
                short_thr=cfg["short_thr"], condor_thr=cfg["condor_thr"],
                min_target_return=cfg["min_target_return"],
            )
        else:
            raise ValueError(f"Unknown label_fn: {label_fn}")

        dist = labels["label"].value_counts().sort_index()
        boundary_str = ""
        if boundary:
            boundary_str = (
                f" | up: h={boundary['h_up']:.2f}, min={boundary['min_up_return']:.2%}"
                f"  down: h={boundary['h_down']:.2f}, min={boundary['min_down_return']:.2%}"
            )
        print(f"  {ticker}: {len(labels):,} events  "
              f"| -1: {dist.get(-1,0):,}  0: {dist.get(0,0):,}  +1: {dist.get(1,0):,}"
              f"{boundary_str}")

        feats = build_features(
            dollar=dollar,
            volume=all_bars[ticker]["volume"],
            runs=all_bars[ticker]["runs"],
            imbalance=all_bars[ticker]["imbalance"],
            windows=cfg["feat_windows"],
            density_window_min=cfg["density_win_min"],
            prefix=cfg["prefix"],
        )

        # z-score absolute-scale features so they're comparable across tickers
        stats = {}
        for col in ABS_FEATURES:
            if col in feats.columns:
                mean = float(feats[col].mean())
                std  = float(feats[col].std())
                std  = std if std > 1e-8 else 1.0
                feats[col] = (feats[col] - mean) / std
                stats[col] = (mean, std)
        zscore_stats[ticker] = stats

        ticker_ds = labels.join(feats, on="t").dropna()
        ticker_ds = ticker_ds.assign(ticker=ticker)
        ticker_datasets.append(ticker_ds)

    # ── combine + time-sort + sequential indices ──────────────────────────────
    dataset   = (pd.concat(ticker_datasets, ignore_index=True)
                   .sort_values("timestamp")
                   .reset_index(drop=True))
    dataset   = _assign_seq_indices(dataset)
    feat_cols = [c for c in dataset.columns if c not in META_COLS]

    # intraday: keep only LONG/SHORT (binary classifier — faster intraday signals)
    if cfg.get("directional_only", False):
        dataset = dataset[dataset["label"].isin([-1, 1])].reset_index(drop=True)
        dataset = _assign_seq_indices(dataset)

    # fit label encoder on all data so classes are stable across train/test splits
    le = LabelEncoder()
    le.fit(dataset["label"].values)   # [-1,+1]→[0,1] or [-1,0,+1]→[0,1,2]
    n_classes = len(le.classes_)
    print(f"\n  Classes: {le.classes_} → {list(range(n_classes))}")
    print(f"  Combined: {len(dataset):,} samples, {len(feat_cols)} features")

    split = int(len(dataset) * TRAIN_RATIO)
    train = dataset.iloc[:split]
    test  = dataset.iloc[split:]
    print(f"  train: {len(train):,} "
          f"({pd.to_datetime(train['timestamp'].min()).date()} → "
          f"{pd.to_datetime(train['timestamp'].max()).date()})")
    print(f"  test:  {len(test):,}  "
          f"({pd.to_datetime(test['timestamp'].min()).date()} → "
          f"{pd.to_datetime(test['timestamp'].max()).date()})")

    dataset.to_parquet(OUT_DIR / f"multi_{regime}_dataset.parquet", index=False)

    MIN_TRAIN = 80
    if len(train) < MIN_TRAIN:
        print(f"\n  ⚠ Only {len(train)} train samples (< {MIN_TRAIN}) — skipping")
        return

    # ── purged CV ─────────────────────────────────────────────────────────────
    print(f"\n[3/4] Purged 5-Fold CV ...")
    model = _make_xgb(len(train), n_classes=n_classes)
    cv_score(model, train, feat_cols, max_hold=cfg["max_hold"], le=le)

    # ── fit on train, sweep thresholds on test ────────────────────────────────
    X_train = train[feat_cols].values
    y_train = le.transform(train["label"].values)
    w_train = _class_balanced_weights(y_train, train["weight"].values)
    model.fit(X_train, y_train, sample_weight=w_train)

    print(f"\n[4/4] Threshold sweep (test set) ...")
    X_test   = test[feat_cols].values
    y_test   = le.transform(test["label"].values)
    ret_test = test["ret"].reset_index(drop=True)

    sweep = threshold_sweep(model, X_test, y_test, ret_test, stride=cfg["stride"])
    print(f"\n  {'thr':>5}  {'trades':>7}  {'cov':>6}  "
          f"{'prec':>6}  {'sharpe':>7}  {'mean_ret%':>10}")
    print(f"  {'─'*57}")
    for _, row in sweep.iterrows():
        marker = "  ◄" if row["mean_ret_%"] == sweep["mean_ret_%"].max() else ""
        print(f"  {row['thr']:>5.2f}  {int(row['n_trades']):>7,}  "
              f"{row['coverage']:>6.1%}  {row['precision']:>6.3f}  "
              f"{row['sharpe']:>7.3f}  {row['mean_ret_%']:>9.4f}%{marker}")

    # ── refit on full dataset ─────────────────────────────────────────────────
    print(f"\n  Refitting on full dataset ({len(dataset):,} samples) ...")
    final_model = _make_xgb(len(dataset), n_classes=n_classes)
    X_all = dataset[feat_cols].values
    y_all = le.transform(dataset["label"].values)
    w_all = _class_balanced_weights(y_all, dataset["weight"].values)
    final_model.fit(X_all, y_all, sample_weight=w_all)

    MIN_TRADES = max(5, int(len(test) * 0.03))
    eligible   = sweep[sweep["n_trades"] >= MIN_TRADES]
    if eligible.empty:
        eligible = sweep
    valid = eligible.dropna(subset=["mean_ret_%"])
    if valid.empty:
        print(f"  ⚠ No tradeable signals in threshold sweep — model saved with thr=0.50")
        best_thr = 0.50
    else:
        best_row = valid.loc[valid["mean_ret_%"].idxmax()]
        best_thr = float(best_row["thr"])
        print(f"  Best threshold (min_trades≥{MIN_TRADES}): {best_thr:.2f}  "
              f"mean_ret={best_row['mean_ret_%']:.4f}%  precision={best_row['precision']:.3f}")

    # ── save ──────────────────────────────────────────────────────────────────
    out_path = MODEL_DIR / f"multi_xgb_{regime}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "model":        final_model,
            "features":     feat_cols,
            "config":       cfg,
            "tickers":      tickers,
            "best_thr":     best_thr,
            "zscore_stats": zscore_stats,   # {ticker: {col: (mean, std)}}
            "label_encoder": le,            # maps original labels → model class indices
        }, f)
    sweep.to_csv(MODEL_DIR / f"multi_xgb_{regime}_sweep.csv", index=False)
    print(f"\n  Model saved → {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                        help="One or more tickers to train on together")
    parser.add_argument("--intraday-boundaries", default=str(INTRADAY_BOUNDARY_PATH),
                        help="CSV from sweep_intraday_boundaries.py; use 'none' to disable")
    parser.add_argument("--regimes", nargs="+", choices=list(CONFIGS.keys()),
                        default=list(CONFIGS.keys()),
                        help="Regimes to train; default trains all")
    args    = parser.parse_args()
    tickers = [t.upper() for t in args.tickers]

    boundary_path = None if args.intraday_boundaries.lower() == "none" else args.intraday_boundaries
    configs = select_regime_configs(args.regimes)
    if "intraday" in configs:
        configs["intraday"]["boundary_overrides"] = load_intraday_boundaries(boundary_path)
    print(f"Training regimes: {', '.join(configs.keys())}")

    print(f"Loading bars for: {', '.join(tickers)} ...")
    all_bars = {}
    for ticker in tickers:
        all_bars[ticker] = {
            "dollar":    pd.read_parquet(BARS_DIR / f"{ticker}_dollar_bars.parquet"),
            "volume":    pd.read_parquet(BARS_DIR / f"{ticker}_volume_bars.parquet"),
            "runs":      pd.read_parquet(BARS_DIR / f"{ticker}_runs_bars.parquet"),
            "imbalance": pd.read_parquet(BARS_DIR / f"{ticker}_imbalance_bars.parquet"),
        }
        for k, v in all_bars[ticker].items():
            print(f"  {ticker} {k}: {len(v):,} bars")

    for regime, cfg in configs.items():
        run_regime(regime, cfg, all_bars, tickers)

    print(f"\n{'═'*60}")
    print(f"  Done. Tickers: {', '.join(tickers)}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()

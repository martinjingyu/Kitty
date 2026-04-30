"""
Train XGBoost and RandomForest on the SPY triple-barrier dataset.
Only trade when predicted probability exceeds a confidence threshold.
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import precision_score, recall_score
from xgboost import XGBClassifier

from models.cv import PurgedKFold

DATA_DIR  = Path("data/dataset")
MODEL_DIR = Path("models/saved")
MODEL_DIR.mkdir(exist_ok=True)

META_COLS = {"t", "t_exit", "timestamp", "timestamp_exit",
             "label", "ret", "vol", "weight"}

THRESHOLDS = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.63, 0.65]


def load_data():
    train = pd.read_parquet(DATA_DIR / "SPY_train.parquet")
    test  = pd.read_parquet(DATA_DIR / "SPY_test.parquet")
    train = train[train["label"] != 0].copy()
    test  = test[test["label"]  != 0].copy()
    feat_cols = [c for c in train.columns if c not in META_COLS]
    return train, test, feat_cols


def sharpe(returns: pd.Series, periods_per_year: int = 252 * 20) -> float:
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def threshold_sweep(model, X_test, y_test, ret_test) -> pd.DataFrame:
    """
    For each confidence threshold, evaluate:
      - coverage   : % of bars we actually trade
      - precision  : accuracy on traded bars
      - sharpe     : annualised Sharpe of the selective strategy
      - cum_ret    : total log return
    """
    proba = model.predict_proba(X_test)[:, 1]   # P(Long)
    ret   = ret_test.values
    y     = y_test

    rows = []
    for thr in THRESHOLDS:
        # long signal when P(Long) > thr, short when P(Long) < (1-thr)
        long_mask  = proba >  thr
        short_mask = proba < (1 - thr)
        trade_mask = long_mask | short_mask

        n_trades = trade_mask.sum()
        coverage = n_trades / len(proba)

        if n_trades == 0:
            rows.append({"threshold": thr, "n_trades": 0, "coverage": 0,
                         "precision": np.nan, "sharpe": np.nan, "cum_ret": np.nan})
            continue

        direction = np.where(long_mask, 1, np.where(short_mask, -1, 0))
        strat_ret = pd.Series(direction[trade_mask] * ret[trade_mask])
        correct   = (direction[trade_mask] == np.where(y[trade_mask] == 1, 1, -1))

        rows.append({
            "threshold": thr,
            "n_trades":  n_trades,
            "coverage":  round(coverage, 3),
            "precision": round(correct.mean(), 4),
            "sharpe":    round(sharpe(strat_ret), 3),
            "cum_ret_%": round((np.exp(strat_ret.sum()) - 1) * 100, 2),
        })

    return pd.DataFrame(rows)


def cv_score(model, df, feat_cols):
    X      = df[feat_cols].values
    y      = (df["label"].values + 1) // 2
    t      = df["t"].values
    t_exit = df["t_exit"].values
    w      = df["weight"].values

    pkf  = PurgedKFold(n_splits=5, embargo=10)
    accs = []
    for fold, (tr, te) in enumerate(pkf.split(t, t_exit)):
        model.fit(X[tr], y[tr], sample_weight=w[tr])
        acc = (model.predict(X[te]) == y[te]).mean()
        accs.append(acc)
        print(f"    fold {fold+1}: acc={acc:.3f}  "
              f"(train={len(tr):,}  test={len(te):,})")
    print(f"    CV mean acc: {np.mean(accs):.3f} ± {np.std(accs):.3f}")


def main():
    train, test, feat_cols = load_data()
    print(f"Train: {len(train):,}  Test: {len(test):,}  Features: {len(feat_cols)}")

    X_train = train[feat_cols].values
    y_train = (train["label"].values + 1) // 2
    w_train = train["weight"].values
    X_test  = test[feat_cols].values
    y_test  = (test["label"].values + 1) // 2
    ret_test = test["ret"].reset_index(drop=True)

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0, random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20,
            max_features="sqrt", n_jobs=-1, random_state=42,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'═'*55}")
        print(f"  {name}  —  Purged 5-Fold CV")
        print(f"{'═'*55}")
        cv_score(model, train, feat_cols)

        model.fit(X_train, y_train, sample_weight=w_train)

        print(f"\n  Threshold sweep (test set) — {name}")
        print(f"  {'thr':>5}  {'trades':>7}  {'cov':>6}  "
              f"{'prec':>6}  {'sharpe':>7}  {'cum_ret%':>9}")
        print(f"  {'─'*53}")

        sweep = threshold_sweep(model, X_test, y_test, ret_test)
        for _, row in sweep.iterrows():
            print(f"  {row['threshold']:>5.2f}  {int(row['n_trades']):>7,}  "
                  f"{row['coverage']:>6.1%}  {row['precision']:>6.3f}  "
                  f"{row['sharpe']:>7.3f}  {row['cum_ret_%']:>8.2f}%")

        results[name] = sweep

        with open(MODEL_DIR / f"{name.lower()}.pkl", "wb") as f:
            pickle.dump({"model": model, "features": feat_cols}, f)

    # save sweep results
    for name, sweep in results.items():
        sweep.to_csv(MODEL_DIR / f"{name.lower()}_threshold_sweep.csv", index=False)

    # feature importance
    xgb = models["XGBoost"]
    imp = pd.Series(xgb.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"\n── Top 15 Features (XGBoost) ──")
    print(imp.head(15).to_string())
    imp.to_csv(MODEL_DIR / "feature_importance.csv")

    print(f"\nModels saved → {MODEL_DIR}/")


if __name__ == "__main__":
    main()

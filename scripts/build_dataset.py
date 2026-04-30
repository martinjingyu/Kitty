"""
Build the final ML dataset:
  1. Load four bar types from parquet
  2. Compute features on dollar bar axis
  3. Apply triple-barrier labeling
  4. Time-based train/test split (no random shuffle)
  5. Save dataset to data/dataset/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from data.features import build_features
from data.labels   import label_triple_barrier

BARS_DIR = Path("data/bars/processed")
OUT_DIR  = Path("data/dataset")
OUT_DIR.mkdir(exist_ok=True)

# triple-barrier params
H         = 1.0   # barrier width = h × daily_vol
MAX_HOLD  = 10    # max bars to hold
VOL_LB    = 20    # EWM span for volatility

TRAIN_RATIO = 0.7


def main():
    print("Loading bars ...")
    dollar   = pd.read_parquet(BARS_DIR / "SPY_dollar_bars.parquet")
    volume   = pd.read_parquet(BARS_DIR / "SPY_volume_bars.parquet")
    runs     = pd.read_parquet(BARS_DIR / "SPY_runs_bars.parquet")
    imbalance = pd.read_parquet(BARS_DIR / "SPY_imbalance_bars.parquet")

    for name, df in [("dollar", dollar), ("volume", volume),
                     ("runs", runs), ("imbalance", imbalance)]:
        print(f"  {name}: {len(df):,} bars")

    # ── features ──────────────────────────────────────────────────────────────
    print("\nBuilding features ...")
    feats = build_features(dollar, volume, runs, imbalance)
    print(f"  {feats.shape[1]} features × {len(feats):,} bars")

    # ── labels ────────────────────────────────────────────────────────────────
    print(f"\nApplying triple-barrier labels  (h={H}, max_hold={MAX_HOLD}) ...")
    labels = label_triple_barrier(dollar, h=H, max_hold=MAX_HOLD, vol_lookback=VOL_LB)
    dist = labels["label"].value_counts().sort_index()
    print(f"  total events : {len(labels):,}")
    print(f"  label dist   : -1={dist.get(-1,0):,}  0={dist.get(0,0):,}  +1={dist.get(1,0):,}")

    # ── merge ─────────────────────────────────────────────────────────────────
    dataset = labels.join(feats, on="t")
    dataset = dataset.dropna()
    print(f"\nDataset after merge + dropna: {len(dataset):,} rows, "
          f"{dataset.shape[1]} columns")

    # ── time-based split ───────────────────────────────────────────────────────
    split = int(len(dataset) * TRAIN_RATIO)
    train = dataset.iloc[:split]
    test  = dataset.iloc[split:]
    print(f"\nTrain: {len(train):,} rows  "
          f"({train['timestamp'].min().date()} → {train['timestamp'].max().date()})")
    print(f"Test:  {len(test):,} rows  "
          f"({test['timestamp'].min().date()} → {test['timestamp'].max().date()})")

    # ── save ──────────────────────────────────────────────────────────────────
    dataset.to_parquet(OUT_DIR / "SPY_dataset.parquet", index=False)
    train.to_parquet(OUT_DIR   / "SPY_train.parquet",   index=False)
    test.to_parquet(OUT_DIR    / "SPY_test.parquet",    index=False)
    print(f"\nSaved → {OUT_DIR}/")


if __name__ == "__main__":
    main()

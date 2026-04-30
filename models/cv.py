"""
Purged K-Fold cross-validation for financial time series.

Standard K-Fold leaks because label windows from the training set
can overlap with the test period. We fix this by:
  1. Splitting folds by time (no shuffle)
  2. Purging: removing train samples whose t_exit >= test fold start
  3. Embargo: additionally dropping train samples within `embargo` bars
     before the test fold (autocorrelation buffer)
"""
import numpy as np


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo: int = 10):
        self.n_splits = n_splits
        self.embargo  = embargo

    def split(self, t: np.ndarray, t_exit: np.ndarray):
        """
        Parameters
        ----------
        t      : bar index of each event's entry
        t_exit : bar index of each event's exit (triple-barrier touch)

        Yields (train_idx, test_idx) arrays.
        """
        n = len(t)
        indices = np.arange(n)
        fold_size = n // self.n_splits

        for k in range(self.n_splits):
            # test fold: [k*fold_size, (k+1)*fold_size)
            test_start = k * fold_size
            test_end   = n if k == self.n_splits - 1 else (k + 1) * fold_size
            test_idx   = indices[test_start:test_end]

            test_t_min = t[test_start]

            # purge: drop train samples whose label window bleeds into test
            # embargo: drop samples within `embargo` bars before test start
            embargo_t = test_t_min - self.embargo
            train_mask = (t_exit < embargo_t) | (t >= t[test_end - 1 if test_end < n else n - 1])
            # only keep events strictly before test fold (no future leakage)
            train_mask = t_exit < embargo_t
            train_idx  = indices[train_mask]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

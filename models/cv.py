"""
Purged K-Fold cross-validation for financial time series.

Standard K-Fold leaks because label windows from the training set
can overlap with the test period. We fix this by:
  1. Splitting folds by time (no shuffle)
  2. Purging: removing train samples whose t_exit >= test fold start
  3. Embargo: additionally dropping train samples within `embargo` units
     before the test fold (autocorrelation buffer)

t and t_exit are sequential row indices (0, 1, 2, ...) in the
time-sorted combined dataset.  embargo is in the same row units,
typically set to cfg["max_hold"] so the embargo matches the label
horizon length.
"""
import numpy as np


class PurgedKFold:
    def __init__(self, n_splits: int = 5, embargo: int = 1):
        self.n_splits = n_splits
        self.embargo  = embargo   # in the same units as t / t_exit (hours)

    def split(self, t: np.ndarray, t_exit: np.ndarray):
        """
        Parameters
        ----------
        t      : unix hours of each event's entry (int64)
        t_exit : unix hours of each event's exit

        Yields (train_idx, test_idx) arrays.
        """
        n = len(t)
        indices = np.arange(n)
        fold_size = n // self.n_splits

        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end   = n if k == self.n_splits - 1 else (k + 1) * fold_size
            test_idx   = indices[test_start:test_end]

            test_t_min = t[test_start]
            embargo_t  = test_t_min - self.embargo

            train_mask = t_exit < embargo_t
            train_idx  = indices[train_mask]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

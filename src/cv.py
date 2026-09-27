
from __future__ import annotations

import numpy as np


def expanding_window_folds(targets: np.ndarray, n_folds: int = 5, gap: int = 0):
    targets = np.asarray(targets)
    n = len(targets)
    val_size = n // (n_folds + 1)
    folds = []
    for k in range(n_folds):
        val_start = n - (n_folds - k) * val_size
        train = targets[: max(val_start - gap, 0)]
        val = targets[val_start: val_start + val_size]
        folds.append((train, val))
    return folds


"""
Chronological inner split: last ``frac`` of the training targets is used
only for early stopping, so the CV validation block never influences
training -> it is only used to *score* a configuration
    """
def early_stopping_split(train_targets: np.ndarray, frac: float = 0.15, min_es: int = 5):
    
    n = len(train_targets)
    n_es = max(int(round(frac * n)), min_es)
    return train_targets[: n - n_es], train_targets[n - n_es:]

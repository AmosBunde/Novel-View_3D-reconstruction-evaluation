"""Two ways to choose which cameras to hold out, with the same number held out by each.

``random``  the convention in the literature: test views are scattered among the
            training views, so every test camera has training cameras close by
            on all sides. The model interpolates.
``cap``     a contiguous patch of the camera hemisphere is held out, so most test
            cameras look at the scene from a direction no training camera is
            near. The model extrapolates.
"""
from __future__ import annotations

import numpy as np

from .cameras import view_directions

DESIGNS = ("random", "cap")


def split(c2w: np.ndarray, design: str, n_test: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (train indices, test indices), both sorted."""
    n = c2w.shape[0]
    rng = np.random.default_rng(seed)
    if design == "random":
        test = rng.choice(n, n_test, replace=False)
    elif design == "cap":
        dirs = view_directions(c2w)
        centre = dirs[rng.integers(n)]
        test = np.argsort(-(dirs @ centre))[:n_test]
    else:
        raise ValueError(f"unknown design {design!r}; choose from {DESIGNS}")
    test = np.sort(test)
    return np.setdiff1d(np.arange(n), test), test


def subsample(train: np.ndarray, n_keep: int, seed: int) -> np.ndarray:
    """A random subset of the training cameras, for the sparse-view experiment."""
    if n_keep >= train.size:
        return train
    return np.sort(np.random.default_rng(10_000 + seed).choice(train, n_keep, replace=False))

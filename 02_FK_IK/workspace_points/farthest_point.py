"""Farthest point sampling in normalized feature space."""

from __future__ import annotations

import numpy as np

from fkik.geometry import normalize_features


def farthest_point_sample(features: np.ndarray, n_select: int) -> np.ndarray:
    """Return row indices selected by greedy farthest point workspace_points."""

    values = np.asarray(features, dtype=float)
    n = values.shape[0]
    if n_select <= 0:
        return np.array([], dtype=int)
    if n <= n_select:
        return np.arange(n, dtype=int)

    norm = normalize_features(values)
    center = np.mean(norm, axis=0)
    first = int(np.argmax(np.sum((norm - center) ** 2, axis=1)))
    selected = np.empty(n_select, dtype=int)
    selected[0] = first
    min_dist2 = np.sum((norm - norm[first]) ** 2, axis=1)

    for i in range(1, n_select):
        next_idx = int(np.argmax(min_dist2))
        selected[i] = next_idx
        dist2 = np.sum((norm - norm[next_idx]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, dist2)
    return selected

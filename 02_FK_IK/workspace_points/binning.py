"""Workspace binning for uniform point selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fkik.geometry import RobotConfig, point_class_string


RADIAL_LABELS = ["near", "inner_mid", "outer_mid", "far", "boundary"]
HEIGHT_LABELS = ["low", "mid", "high"]
BENDING_LABELS = ["low", "mid", "high"]
POSTURE_LABELS = ["folded", "neutral", "extended"]


def _rank_bins(values: pd.Series, labels: list[str]) -> pd.Series:
    """Assign percentile/rank bins, robust to tied or constant values."""

    if len(values) == 0:
        return pd.Series(dtype=object, index=values.index)
    pct = values.rank(method="first", pct=True).to_numpy()
    edges = np.linspace(0.0, 1.0, len(labels) + 1)[1:-1]
    idx = np.searchsorted(edges, pct, side="right")
    return pd.Series(np.asarray(labels, dtype=object)[idx], index=values.index)


def _azimuth_bins(x: pd.Series, y: pd.Series, n_bins: int) -> pd.Series:
    """Assign azimuth sector labels from base-frame x/y position."""

    angle = np.arctan2(y.to_numpy(dtype=float), x.to_numpy(dtype=float))
    if n_bins == 8:
        labels = np.array(["front", "front_left", "left", "back_left", "back", "back_right", "right", "front_right"], dtype=object)
        sector = np.floor(((angle + np.pi / 8.0) % (2.0 * np.pi)) / (np.pi / 4.0)).astype(int)
        return pd.Series(labels[sector], index=x.index)

    labels4 = np.full(len(angle), "back", dtype=object)
    labels4[(angle >= -np.pi / 4.0) & (angle < np.pi / 4.0)] = "front"
    labels4[(angle >= np.pi / 4.0) & (angle < 3.0 * np.pi / 4.0)] = "left"
    labels4[(angle >= -3.0 * np.pi / 4.0) & (angle < -np.pi / 4.0)] = "right"
    return pd.Series(labels4, index=x.index)


def add_bin_labels(candidates: pd.DataFrame, config: RobotConfig) -> pd.DataFrame:
    """Attach radial, azimuth, height, bending, posture, and bin_id labels."""

    df = candidates.copy()
    x = df["p_flex_x"]
    y = df["p_flex_y"]
    z = df["p_flex_z"]
    rho = np.sqrt(x.to_numpy(dtype=float) ** 2 + y.to_numpy(dtype=float) ** 2)
    reach = np.sqrt(rho**2 + z.to_numpy(dtype=float) ** 2)

    df["radial_shell"] = _rank_bins(pd.Series(rho, index=df.index), RADIAL_LABELS)
    df["azimuth_bin"] = _azimuth_bins(x, y, config.azimuth_bins)
    df["height_bin"] = _rank_bins(z, HEIGHT_LABELS)
    df["bending_class"] = _rank_bins(df["bending_score"], BENDING_LABELS)
    df["joint_posture_class"] = _rank_bins(pd.Series(reach, index=df.index), POSTURE_LABELS)
    df["point_class"] = df.apply(point_class_string, axis=1)
    df["bin_id"] = df["point_class"]
    return df

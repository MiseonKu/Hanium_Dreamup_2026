"""Uniform bin-wise target point workspace_points."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fkik.geometry import RobotConfig
from workspace_points.binning import add_bin_labels
from workspace_points.farthest_point import farthest_point_sample


FEATURE_COLUMNS = ["p_flex_x", "p_flex_y", "p_flex_z", "q_1", "q_2", "q_3", "d_flex", "bending_score"]


def allocate_bin_counts(valid_candidates: pd.DataFrame, n_targets: int) -> dict[str, int]:
    """Allocate nearly equal target counts over valid bins."""

    counts = valid_candidates.groupby("bin_id").size().sort_values(ascending=True)
    if counts.empty:
        raise ValueError("no valid bins available for target sampling")

    n_bins = len(counts)
    base = n_targets // n_bins
    allocation = {bin_id: min(base, int(count)) for bin_id, count in counts.items()}
    selected = sum(allocation.values())
    remaining = n_targets - selected

    while remaining > 0:
        progressed = False
        for bin_id, count in counts.items():
            if remaining <= 0:
                break
            if allocation[bin_id] < int(count):
                allocation[bin_id] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return {bin_id: n for bin_id, n in allocation.items() if n > 0}


def _target_columns(selected: pd.DataFrame, config: RobotConfig) -> pd.DataFrame:
    """Project sampled candidates into the required target_points schema."""

    out = selected.copy().reset_index(drop=True)
    out.insert(0, "target_id", np.arange(len(out), dtype=int))
    out["p_target_x"] = out["p_flex_x"]
    out["p_target_y"] = out["p_flex_y"]
    out["p_target_z"] = out["p_flex_z"]
    out["q_seed_1"] = out["q_1"]
    out["q_seed_2"] = out["q_2"]
    out["q_seed_3"] = out["q_3"]
    out["q_cmd_1"] = out["q_ik_1"]
    out["q_cmd_2"] = out["q_ik_2"]
    out["q_cmd_3"] = out["q_ik_3"]
    out["repeat_count"] = config.repeats_per_point
    out["material_preset"] = config.material_preset
    out["link_length_preset"] = config.link_length_preset
    out["link_a_mm"] = config.link_a_mm
    out["link_b_mm"] = config.link_b_mm
    out["wall_count"] = config.wall_count
    out["top_bottom_wall_count"] = config.top_bottom_wall_count
    out["infill_percent"] = config.infill_percent
    out["infill_type"] = config.infill_type

    ordered = [
        "target_id",
        "point_class",
        "p_target_x",
        "p_target_y",
        "p_target_z",
        "q_seed_1",
        "q_seed_2",
        "q_seed_3",
        "p_rigid_x",
        "p_rigid_y",
        "p_rigid_z",
        "p_flex_x",
        "p_flex_y",
        "p_flex_z",
        "d_flex",
        "bending_score",
        "M_weak_1",
        "M_weak_2",
        "delta_1",
        "delta_2",
        "radial_shell",
        "azimuth_bin",
        "height_bin",
        "bending_class",
        "joint_posture_class",
        "joint_margin",
        "singularity_score",
        "ik_success",
        "round_trip_error",
        "repeat_count",
        "bin_id",
        "material_preset",
        "link_length_preset",
        "link_a_mm",
        "link_b_mm",
        "wall_count",
        "top_bottom_wall_count",
        "infill_percent",
        "infill_type",
        "q_cmd_1",
        "q_cmd_2",
        "q_cmd_3",
    ]
    return out[ordered]


def sample_uniform_targets(candidates: pd.DataFrame, config: RobotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample target points so selected bin counts are as flat as possible."""

    df = candidates.copy()
    if "bin_id" not in df.columns:
        df = add_bin_labels(df, config)
    valid = df[df["valid"]].copy()
    if valid.empty:
        raise ValueError("no valid candidates after filters")

    allocation = allocate_bin_counts(valid, config.n_targets)
    selected_parts: list[pd.DataFrame] = []
    for bin_id, n_select in allocation.items():
        group = valid[valid["bin_id"] == bin_id]
        feature = group[FEATURE_COLUMNS].to_numpy(dtype=float)
        selected_pos = farthest_point_sample(feature, n_select)
        selected_parts.append(group.iloc[selected_pos])

    selected = pd.concat(selected_parts, axis=0, ignore_index=False)
    if len(selected) < config.n_targets:
        remaining_pool = valid.drop(index=selected.index, errors="ignore")
        need = config.n_targets - len(selected)
        if len(remaining_pool) > 0 and need > 0:
            extra_pos = farthest_point_sample(remaining_pool[FEATURE_COLUMNS].to_numpy(dtype=float), min(need, len(remaining_pool)))
            selected = pd.concat([selected, remaining_pool.iloc[extra_pos]], axis=0)

    selected = selected.head(config.n_targets)
    targets = _target_columns(selected, config)
    report = point_distribution_report(targets, valid)
    return targets, report


def point_distribution_report(targets: pd.DataFrame, valid_candidates: pd.DataFrame) -> pd.DataFrame:
    """Build per-bin count report with global uniformity statistics."""

    target_counts = targets.groupby("bin_id").size().rename("target_count")
    candidate_counts = valid_candidates.groupby("bin_id").size().rename("candidate_count")
    report = pd.concat([candidate_counts, target_counts], axis=1).fillna(0).reset_index()
    report["target_count"] = report["target_count"].astype(int)
    report["candidate_count"] = report["candidate_count"].astype(int)
    selected_counts = report.loc[report["target_count"] > 0, "target_count"]
    mean = float(selected_counts.mean()) if len(selected_counts) else np.nan
    std = float(selected_counts.std(ddof=0)) if len(selected_counts) else np.nan
    report["selected_bin_min"] = int(selected_counts.min()) if len(selected_counts) else 0
    report["selected_bin_max"] = int(selected_counts.max()) if len(selected_counts) else 0
    report["selected_bin_mean"] = mean
    report["selected_bin_std"] = std
    report["uniformity_score"] = std / mean if mean and np.isfinite(mean) else np.nan
    return report.sort_values(["target_count", "candidate_count", "bin_id"], ascending=[False, True, True])

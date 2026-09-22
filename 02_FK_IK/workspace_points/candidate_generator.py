"""Joint-space candidate generation for workspace target selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fkik.flex_ik import ik_flex
from fkik.geometry import RobotConfig


def latin_hypercube_sample(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Latin Hypercube sample in [0, 1]^n_dim."""

    sample = np.empty((n_samples, n_dim), dtype=float)
    for dim in range(n_dim):
        perm = rng.permutation(n_samples)
        sample[:, dim] = (perm + rng.random(n_samples)) / n_samples
    return sample


def stratified_uniform_sample(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a lightweight stratified-uniform fallback sample."""

    bins = int(np.ceil(n_samples ** (1.0 / n_dim)))
    grid = np.stack(np.meshgrid(*([np.arange(bins)] * n_dim), indexing="ij"), axis=-1).reshape(-1, n_dim)
    if len(grid) < n_samples:
        extra = rng.integers(0, bins, size=(n_samples - len(grid), n_dim))
        grid = np.vstack([grid, extra])
    selected = grid[rng.choice(len(grid), size=n_samples, replace=False)]
    return (selected + rng.random((n_samples, n_dim))) / bins


def sample_joint_space(config: RobotConfig, n_samples: int | None = None) -> np.ndarray:
    """Sample q1/q2/q3 inside configured joint limits."""

    n = int(config.n_candidates if n_samples is None else n_samples)
    rng = np.random.default_rng(config.random_seed)
    if config.latin_hypercube:
        unit = latin_hypercube_sample(n, 3, rng)
    else:
        unit = stratified_uniform_sample(n, 3, rng)
    lower = config.joint_limits_rad[:, 0]
    upper = config.joint_limits_rad[:, 1]
    return lower + unit * (upper - lower)


def _vectorized_fk_metrics(q: np.ndarray, config: RobotConfig) -> dict[str, np.ndarray]:
    """Compute FK and filter metrics for many candidate postures."""

    q1 = q[:, 0]
    q2 = q[:, 1]
    q3 = q[:, 2]
    alpha2_rigid = q2 + q3
    dx, dz = config.tracker_offset_mm

    rho_rigid = config.L1 * np.cos(q2) + (config.L2 + dx) * np.cos(alpha2_rigid) - dz * np.sin(alpha2_rigid)
    z_rigid = config.L1 * np.sin(q2) + (config.L2 + dx) * np.sin(alpha2_rigid) + dz * np.cos(alpha2_rigid)
    p_rigid = np.column_stack([rho_rigid * np.cos(q1), rho_rigid * np.sin(q1), z_rigid])

    m1 = 0.5 * config.L1 * np.cos(q2) + config.L2 * np.cos(alpha2_rigid)
    m2 = 0.5 * config.L2 * np.cos(alpha2_rigid)
    delta1 = config.k1 * m1
    delta2 = config.k2 * m2

    alpha1_mid = q2 - delta1 / 2.0
    p1_rho = config.L1 * np.cos(alpha1_mid)
    p1_z = config.L1 * np.sin(alpha1_mid)

    alpha2 = q2 + q3 - delta1
    alpha2_mid = alpha2 - delta2 / 2.0
    p2_rho = p1_rho + config.L2 * np.cos(alpha2_mid)
    p2_z = p1_z + config.L2 * np.sin(alpha2_mid)

    offset_angle = alpha2 - delta2
    p2_rho = p2_rho + dx * np.cos(offset_angle) - dz * np.sin(offset_angle)
    p2_z = p2_z + dx * np.sin(offset_angle) + dz * np.cos(offset_angle)
    p_flex = np.column_stack([p2_rho * np.cos(q1), p2_rho * np.sin(q1), p2_z])

    d_flex = np.linalg.norm(p_flex - p_rigid, axis=1)
    w1, w2, w3 = config.bending_score_weights
    bending_score = w1 * np.abs(m1) + w2 * np.abs(m2) + w3 * d_flex
    lower = config.joint_limits_rad[:, 0]
    upper = config.joint_limits_rad[:, 1]
    joint_margin = np.min(np.minimum(q - lower, upper - q), axis=1)
    singularity = np.abs(np.sin(q3 + np.arctan2(dz, config.L2 + dx)))

    return {
        "p_rigid": p_rigid,
        "p_flex": p_flex,
        "M1_weak": m1,
        "M2_weak": m2,
        "delta1": delta1,
        "delta2": delta2,
        "d_flex": d_flex,
        "bending_score": bending_score,
        "joint_margin": joint_margin,
        "singularity_score": singularity,
    }


def _apply_preliminary_filters(df: pd.DataFrame, config: RobotConfig) -> pd.DataFrame:
    """Attach valid/invalid reason columns for non-IK safety filters."""

    valid = np.ones(len(df), dtype=bool)
    reason = np.full(len(df), "ok", dtype=object)

    checks = [
        (df["joint_margin"].to_numpy() < config.min_joint_margin_rad, "joint_margin"),
        (df["singularity_score"].to_numpy() < config.min_singularity_score, "singularity"),
        (np.abs(df["delta_1"].to_numpy()) > config.max_delta_rad, "delta_1"),
        (np.abs(df["delta_2"].to_numpy()) > config.max_delta_rad, "delta_2"),
        (df["d_flex"].to_numpy() > config.max_d_flex_mm, "d_flex"),
        (df["p_flex_z"].to_numpy() < config.floor_min_z_mm, "floor_collision_placeholder"),
    ]
    for mask, label in checks:
        update = valid & mask
        reason[update] = label
        valid &= ~mask

    df = df.copy()
    df["self_collision_placeholder"] = False
    df["base_collision_placeholder"] = False
    df["vive_occlusion_placeholder"] = False
    df["prelim_valid"] = valid
    df["invalid_reason"] = reason
    return df


def _validate_round_trip(df: pd.DataFrame, config: RobotConfig) -> pd.DataFrame:
    """Validate p_flex -> IK_flex -> FK_flex round trips."""

    df = df.copy()
    n = len(df)
    q_ik = df[["q_1", "q_2", "q_3"]].to_numpy().copy()
    ik_success = np.zeros(n, dtype=bool)
    round_trip_error = np.full(n, np.inf, dtype=float)

    prelim = df["prelim_valid"].to_numpy()
    if not config.validate_round_trip:
        ik_success[prelim] = True
        round_trip_error[prelim] = np.nan
    elif abs(config.k1) <= 1e-15 and abs(config.k2) <= 1e-15:
        ik_success[prelim] = True
        round_trip_error[prelim] = 0.0
    else:
        p_cols = ["p_flex_x", "p_flex_y", "p_flex_z"]
        for idx in np.flatnonzero(prelim):
            target = df.loc[df.index[idx], p_cols].to_numpy(dtype=float)
            result = ik_flex(target, config, q_init=df.loc[df.index[idx], ["q_1", "q_2", "q_3"]].to_numpy(dtype=float))
            ik_success[idx] = result.success
            round_trip_error[idx] = result.residual_error
            if result.q is not None:
                q_ik[idx, :] = result.q

    df["q_ik_1"] = q_ik[:, 0]
    df["q_ik_2"] = q_ik[:, 1]
    df["q_ik_3"] = q_ik[:, 2]
    df["ik_success"] = ik_success
    df["round_trip_error"] = round_trip_error
    final_valid = df["prelim_valid"].to_numpy() & ik_success & (
        np.nan_to_num(round_trip_error, nan=0.0, posinf=np.inf) <= config.round_trip_epsilon_mm
    )
    df["valid"] = final_valid
    df.loc[df["prelim_valid"] & ~df["ik_success"], "invalid_reason"] = "ik_flex_failed"
    df.loc[
        df["prelim_valid"] & df["ik_success"] & ~df["valid"],
        "invalid_reason",
    ] = "round_trip_error"
    return df


def generate_candidates(config: RobotConfig, n_candidates: int | None = None) -> pd.DataFrame:
    """Generate joint-space candidates and FK/IK metadata as a DataFrame."""

    q = sample_joint_space(config, n_candidates)
    metrics = _vectorized_fk_metrics(q, config)
    p_rigid = metrics["p_rigid"]
    p_flex = metrics["p_flex"]
    df = pd.DataFrame(
        {
            "candidate_id": np.arange(len(q), dtype=int),
            "q_1": q[:, 0],
            "q_2": q[:, 1],
            "q_3": q[:, 2],
            "p_rigid_x": p_rigid[:, 0],
            "p_rigid_y": p_rigid[:, 1],
            "p_rigid_z": p_rigid[:, 2],
            "p_flex_x": p_flex[:, 0],
            "p_flex_y": p_flex[:, 1],
            "p_flex_z": p_flex[:, 2],
            "d_flex": metrics["d_flex"],
            "bending_score": metrics["bending_score"],
            "M_weak_1": metrics["M1_weak"],
            "M_weak_2": metrics["M2_weak"],
            "delta_1": metrics["delta1"],
            "delta_2": metrics["delta2"],
            "joint_margin": metrics["joint_margin"],
            "singularity_score": metrics["singularity_score"],
        }
    )
    df = _apply_preliminary_filters(df, config)
    return _validate_round_trip(df, config)


def candidate_config_metadata(config: RobotConfig) -> dict[str, Any]:
    """Return the candidate-generation config subset that affects FK/IK output."""

    return {
        "robot_name": config.name,
        "link_length_preset": config.link_length_preset,
        "link_a_mm": config.link_a_mm,
        "link_b_mm": config.link_b_mm,
        "tracker_offset_mm": config.tracker_offset_mm.tolist(),
        "use_c_link": config.use_c_link,
        "flex_model": {
            "k1": config.k1,
            "k2": config.k2,
            "max_delta_rad": config.max_delta_rad,
            "bending_score_weights": config.bending_score_weights.tolist(),
        },
        "joint_limits_rad": config.joint_limits_rad.tolist(),
        "sampling": {
            "n_candidates": config.n_candidates,
            "random_seed": config.random_seed,
            "latin_hypercube": config.latin_hypercube,
            "min_joint_margin_rad": config.min_joint_margin_rad,
            "min_singularity_score": config.min_singularity_score,
            "max_d_flex_mm": config.max_d_flex_mm,
            "floor_min_z_mm": config.floor_min_z_mm,
            "round_trip_epsilon_mm": config.round_trip_epsilon_mm,
            "validate_round_trip": config.validate_round_trip,
        },
    }


def write_candidate_metadata(path: str | Path, config: RobotConfig, candidates: pd.DataFrame) -> None:
    """Write a sidecar JSON metadata file for candidate-cache validation."""

    metadata = candidate_config_metadata(config)
    metadata["sampling"]["n_candidates"] = int(len(candidates))
    metadata["row_count"] = int(len(candidates))
    metadata["valid_count"] = int(candidates["valid"].sum()) if "valid" in candidates else None
    meta_path = Path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def candidate_metadata_matches(path: str | Path, config: RobotConfig) -> bool:
    """Return True when a candidate-cache metadata file matches config."""

    meta_path = Path(path)
    if not meta_path.exists():
        return False
    with meta_path.open("r", encoding="utf-8") as f:
        existing = json.load(f)
    expected = candidate_config_metadata(config)
    return _metadata_equal(existing, expected)


def _metadata_equal(existing: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare metadata with float tolerance while ignoring summary fields."""

    for key, expected_value in expected.items():
        if key not in existing:
            return False
        existing_value = existing[key]
        if isinstance(expected_value, dict):
            if not isinstance(existing_value, dict) or not _metadata_equal(existing_value, expected_value):
                return False
        elif isinstance(expected_value, list):
            if not np.allclose(np.asarray(existing_value, dtype=float), np.asarray(expected_value, dtype=float), atol=1e-12):
                return False
        elif isinstance(expected_value, float):
            if not np.isclose(float(existing_value), expected_value, atol=1e-12):
                return False
        else:
            if existing_value != expected_value:
                return False
    return True


def write_round_trip_report(candidates: pd.DataFrame, path: str) -> pd.DataFrame:
    """Write round-trip summary statistics to CSV."""

    valid_errors = candidates.loc[candidates["ik_success"], "round_trip_error"].replace([np.inf, -np.inf], np.nan).dropna()
    report = pd.DataFrame(
        [
            {
                "count": int(valid_errors.size),
                "mean_mm": float(valid_errors.mean()) if valid_errors.size else np.nan,
                "std_mm": float(valid_errors.std(ddof=0)) if valid_errors.size else np.nan,
                "min_mm": float(valid_errors.min()) if valid_errors.size else np.nan,
                "max_mm": float(valid_errors.max()) if valid_errors.size else np.nan,
                "p95_mm": float(valid_errors.quantile(0.95)) if valid_errors.size else np.nan,
                "ik_success_count": int(candidates["ik_success"].sum()),
                "valid_count": int(candidates["valid"].sum()),
            }
        ]
    )
    report.to_csv(path, index=False)
    return report

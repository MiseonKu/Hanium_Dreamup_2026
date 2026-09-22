"""Balanced trial ordering for repeated target execution."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fkik.geometry import RobotConfig


def _interleaved_target_order(targets: pd.DataFrame, rng: np.random.Generator) -> list[int]:
    """Return target row indices interleaved across bins."""

    bins = list(targets.groupby("bin_id").groups.keys())
    rng.shuffle(bins)
    per_bin: dict[str, list[int]] = {}
    max_len = 0
    for bin_id in bins:
        indices = list(targets.groupby("bin_id").groups[bin_id])
        rng.shuffle(indices)
        per_bin[bin_id] = indices
        max_len = max(max_len, len(indices))

    order: list[int] = []
    for layer in range(max_len):
        active = [bin_id for bin_id in bins if layer < len(per_bin[bin_id])]
        rng.shuffle(active)
        for bin_id in active:
            order.append(per_bin[bin_id][layer])
    return order


def generate_trial_schedule(targets: pd.DataFrame, config: RobotConfig) -> pd.DataFrame:
    """Create a balanced repeated trial schedule.

    Each repeat cycle contains each target once. Within a cycle, targets are
    interleaved by bin so adjacent trials do not collapse into one workspace
    region or one bending class.
    """

    rng = np.random.default_rng(config.random_seed + 1009)
    rows: list[dict[str, object]] = []
    trial_id = 0
    for repeat_index in range(config.repeats_per_point):
        order = _interleaved_target_order(targets, rng)
        for idx in order:
            row = targets.loc[idx]
            rows.append(
                {
                    "trial_id": trial_id,
                    "cycle_id": repeat_index,
                    "target_id": int(row["target_id"]),
                    "bin_id": row["bin_id"],
                    "repeat_index": repeat_index,
                    "p_target_x": float(row["p_target_x"]),
                    "p_target_y": float(row["p_target_y"]),
                    "p_target_z": float(row["p_target_z"]),
                    "q_cmd_1": float(row["q_cmd_1"]),
                    "q_cmd_2": float(row["q_cmd_2"]),
                    "q_cmd_3": float(row["q_cmd_3"]),
                    "trajectory_type": config.trajectory_type,
                    "move_duration": config.duration_sec / max(config.speed_scale, 1e-9),
                    "material_preset": config.material_preset,
                    "link_length_preset": config.link_length_preset,
                    "link_a_mm": config.link_a_mm,
                    "link_b_mm": config.link_b_mm,
                }
            )
            trial_id += 1
    return pd.DataFrame(rows)

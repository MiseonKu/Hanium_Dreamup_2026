from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from workspace_points.pre_imu_targets import (
    PRE_IMU_TARGET_FIELDNAMES,
    compute_azimuth_deg,
    fk_stage1_rigid,
    fk_stage2_rigid,
    generate_pre_imu_candidate_targets,
    is_inside_forward_trimmed_hemisphere,
    load_pre_imu_workspace_config,
    select_bin_balanced_targets,
    write_pre_imu_targets_csv,
)


CONFIG_PATH = ROOT / "config" / "pre_imu_forward_hemisphere_workspace.json"


def test_fk_stage1_rigid_simple_forward_position() -> None:
    x, y, z = fk_stage1_rigid(0.0, 0.0, 0.233)

    assert math.isclose(x, 0.233, abs_tol=1e-12)
    assert math.isclose(y, 0.0, abs_tol=1e-12)
    assert math.isclose(z, 0.0, abs_tol=1e-12)


def test_fk_stage2_rigid_simple_forward_position() -> None:
    x, y, z = fk_stage2_rigid(0.0, 0.0, 0.0, 0.233, 0.215)

    assert math.isclose(x, 0.448, abs_tol=1e-12)
    assert math.isclose(y, 0.0, abs_tol=1e-12)
    assert math.isclose(z, 0.0, abs_tol=1e-12)


def test_forward_filter_accepts_positive_x_and_rejects_negative_x() -> None:
    config = load_pre_imu_workspace_config(CONFIG_PATH)

    assert is_inside_forward_trimmed_hemisphere((0.1, 0.0, 0.1), config, mode="pilot")
    assert not is_inside_forward_trimmed_hemisphere((-0.1, 0.0, 0.1), config, mode="pilot")


def test_pilot_azimuth_filter_rejects_outside_thirty_degrees() -> None:
    config = load_pre_imu_workspace_config(CONFIG_PATH)

    assert math.isclose(compute_azimuth_deg((1.0, 1.0, 0.0), (0.0, 0.0, 0.0)), 45.0)
    assert not is_inside_forward_trimmed_hemisphere((1.0, 1.0, 0.0), config, mode="pilot")


def test_generated_targets_are_deterministic_for_same_seed() -> None:
    config = load_pre_imu_workspace_config(CONFIG_PATH)
    kwargs = {
        "stage_name": "P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL",
        "mode": "pilot",
        "count": 12,
        "seed": 102,
        "candidate_count": 300,
    }
    candidates_a, _ = generate_pre_imu_candidate_targets(config, **kwargs)
    candidates_b, _ = generate_pre_imu_candidate_targets(config, **kwargs)
    targets_a = select_bin_balanced_targets(candidates_a, 12, seed=102, target_id_prefix="P2")
    targets_b = select_bin_balanced_targets(candidates_b, 12, seed=102, target_id_prefix="P2")

    assert targets_a == targets_b


def test_generated_csv_contains_required_columns(tmp_path) -> None:
    config = load_pre_imu_workspace_config(CONFIG_PATH)
    candidates, _ = generate_pre_imu_candidate_targets(
        config,
        "P1_ONE_LINK_RIGID_TARGET_VIVE_LABEL",
        mode="pilot",
        count=8,
        seed=101,
        candidate_count=240,
    )
    targets = select_bin_balanced_targets(candidates, 8, seed=101, target_id_prefix="P1")
    output = write_pre_imu_targets_csv(tmp_path / "targets.csv", targets)

    with output.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 8
    assert set(PRE_IMU_TARGET_FIELDNAMES).issubset(rows[0])
    assert rows[0]["q_cmd_rad_3"] == "0.0"
    assert rows[0]["link_count"] == "1"
    assert rows[0]["inside_workspace_region"] == "true"

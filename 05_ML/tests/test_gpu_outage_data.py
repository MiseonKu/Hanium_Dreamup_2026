from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import imu_vive_learning.gpu_outage_data as outage_data_module
from imu_vive_learning import source as source_module
from imu_vive_learning.features import FUSION_COLUMNS, LABEL_COLUMNS
from imu_vive_learning.gpu_outage_data import (
    CHALLENGE_ORDERED_TRIAL_UID_SHA256,
    CHALLENGE_SEGMENT_ROWS,
    IDENTITY_PROJECTION_COLUMNS,
    LABEL_PROJECTION_COLUMNS,
    PINNED_OUTAGE_V3_CONFIG_SHA256,
    SOURCE_MEMBER_SHA256,
    ChallengeData,
    GPUOutageDataError,
    ordered_trial_uid_sha256,
    outage_config_sha256,
    prepare_challenge_identity,
    prepare_challenge_labels,
    prepare_development_data,
    validate_outage_data_config,
)
from imu_vive_learning.gpu_outage_experiment_config import load_outage_gpu_config


def _uid(segment_id: int, t_start_ns: int) -> str:
    value = (
        f"{SOURCE_MEMBER_SHA256}|real_motion_demo|{segment_id}|{t_start_ns}"
    )
    return f"trial_{sha256(value.encode('utf-8')).hexdigest()}"


def _tiny_authenticated_archive(tmp_path):
    arrays = [pa.array([None], type=field.type) for field in source_module.EXPECTED_TRIALS_SCHEMA]
    table = pa.Table.from_arrays(arrays, schema=source_module.EXPECTED_TRIALS_SCHEMA)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    member_bytes = sink.getvalue().to_pybytes()
    archive_path = tmp_path / "tiny_source.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(outage_data_module.SOURCE_MEMBER, member_bytes)
    return (
        archive_path,
        sha256(archive_path.read_bytes()).hexdigest(),
        sha256(member_bytes).hexdigest(),
    )


def _source_row(
    segment_id: int,
    t_start_ns: int,
    *,
    duration_s: float = 3.0,
    tip_n: float = 30.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "segment_id": segment_id,
        "block": t_start_ns,
        "set_id": 1.0,
        "target_id": f"T{segment_id:03d}",
        "repeat_index": 0,
        "t_start_ns": t_start_ns,
        "duration_s": duration_s,
        "vive_n": 180,
        "vive_tracking_valid_ratio": 1.0,
        "encoder_n": 5.0,
        "keep": True,
        "encoder_tick_j0": 2056.0,
        "encoder_tick_j1": 2095.0,
        "encoder_tick_j2": 1973.0,
        "vive_pos_x_m": 0.1,
        "vive_pos_y_m": 0.2,
        "vive_pos_z_m": 0.3,
    }
    for sensor_index, prefix in (
        (0, "imu0_base"),
        (2, "imu2_link_mid"),
        (3, "imu3_tip"),
    ):
        base = 0.1 * (sensor_index + 1)
        for axis_index, axis in enumerate(("x", "y", "z"), start=1):
            row[f"{prefix}_euler_{axis}_rad"] = base * axis_index
            row[f"{prefix}_accel_{axis}_mps2"] = 10.0 * sensor_index + axis_index
        row[f"{prefix}_gyro_rms_rad_s"] = base / 10.0
        row[f"{prefix}_n"] = tip_n if sensor_index == 3 else 30.0
    return row


def _tiny_config(frame: pd.DataFrame, selected_rows: list[dict[str, object]]) -> dict:
    config = load_outage_gpu_config(
        "configs/endpoint_regression_gpu_outage_v3.yaml"
    )
    config = deepcopy(config)
    config["source"]["expected_rows"] = len(frame)
    counts = {
        segment: sum(int(row["segment_id"]) == segment for row in selected_rows)
        for segment in (40, 41)
    }
    ordered = sorted(
        selected_rows,
        key=lambda row: (
            int(row["segment_id"]),
            int(row["t_start_ns"]),
            str(row["target_id"]),
        ),
    )
    uids = [
        _uid(int(row["segment_id"]), int(row["t_start_ns"])) for row in ordered
    ]
    role = config["row_roles"]["challenge"]
    role["expected_rows"] = len(selected_rows)
    role["expected_segment_rows"] = counts
    role["ordered_trial_uid_sha256"] = ordered_trial_uid_sha256(uids)
    return config


def _prepare_identity(
    frame: pd.DataFrame, selected_rows: list[dict[str, object]]
):
    config = _tiny_config(frame, selected_rows)
    digest = outage_config_sha256(config)
    identity = prepare_challenge_identity(
        frame,
        source_member_sha256=SOURCE_MEMBER_SHA256,
        challenge_config=config,
        expected_config_sha256=digest,
        enforce_pinned_counts=False,
    )
    return config, digest, identity


def test_pinned_config_and_identity_projection_are_fail_closed() -> None:
    config = load_outage_gpu_config(
        "configs/endpoint_regression_gpu_outage_v3.yaml"
    )
    assert outage_config_sha256(config) == PINNED_OUTAGE_V3_CONFIG_SHA256
    assert config["row_roles"]["challenge"]["expected_segment_rows"] == (
        CHALLENGE_SEGMENT_ROWS
    )
    assert config["row_roles"]["challenge"]["ordered_trial_uid_sha256"] == (
        CHALLENGE_ORDERED_TRIAL_UID_SHA256
    )
    validate_outage_data_config(config)

    assert len(IDENTITY_PROJECTION_COLUMNS) == 38
    assert not any(name.startswith("vive_pos_") for name in IDENTITY_PROJECTION_COLUMNS)
    assert not any(
        name.startswith("imu1_link_root_") for name in IDENTITY_PROJECTION_COLUMNS
    )


def test_authenticated_parquet_decode_materializes_only_requested_projection(
    tmp_path, monkeypatch
) -> None:
    archive_path, archive_hash, member_hash = _tiny_authenticated_archive(tmp_path)
    calls: list[tuple[str, ...]] = []
    original = outage_data_module.pq.read_table

    def recording_read_table(source, *, columns=None, **kwargs):
        calls.append(tuple(columns or ()))
        return original(source, columns=columns, **kwargs)

    monkeypatch.setattr(outage_data_module.pq, "read_table", recording_read_table)
    identity = outage_data_module._load_authenticated_projection(
        archive_path,
        member=outage_data_module.SOURCE_MEMBER,
        expected_archive_sha256=archive_hash,
        expected_member_sha256=member_hash,
        expected_rows=1,
        columns=IDENTITY_PROJECTION_COLUMNS,
    )
    labels = outage_data_module._load_authenticated_projection(
        archive_path,
        member=outage_data_module.SOURCE_MEMBER,
        expected_archive_sha256=archive_hash,
        expected_member_sha256=member_hash,
        expected_rows=1,
        columns=LABEL_PROJECTION_COLUMNS,
    )

    assert calls == [IDENTITY_PROJECTION_COLUMNS, LABEL_PROJECTION_COLUMNS]
    assert tuple(identity.columns) == IDENTITY_PROJECTION_COLUMNS
    assert tuple(labels.columns) == LABEL_PROJECTION_COLUMNS
    assert not set(LABEL_PROJECTION_COLUMNS[2:]).intersection(identity.columns)
    assert not any(name.startswith("imu1_link_root_") for name in identity.columns)


def test_challenge_identity_selection_ignores_imu1_and_preserves_feature_order() -> None:
    rows = [_source_row(40, 100), _source_row(41, 200)]
    frame = pd.DataFrame(rows)
    config, digest, first = _prepare_identity(frame, rows)

    with_imu1 = frame.copy()
    for suffix, values in {
        "euler_x_rad": [math.nan, math.inf],
        "euler_y_rad": [999.0, -999.0],
        "euler_z_rad": [math.nan, math.nan],
        "accel_x_mps2": [1e30, -1e30],
        "accel_y_mps2": [math.inf, math.nan],
        "accel_z_mps2": [-5.0, 5.0],
        "gyro_rms_rad_s": [math.nan, math.inf],
        "n": [-100.0, math.nan],
    }.items():
        with_imu1[f"imu1_link_root_{suffix}"] = values
    second = prepare_challenge_identity(
        with_imu1,
        source_member_sha256=SOURCE_MEMBER_SHA256,
        challenge_config=config,
        expected_config_sha256=digest,
        enforce_pinned_counts=False,
    )

    assert first.feature_names == FUSION_COLUMNS
    assert np.array_equal(first.X, second.X, equal_nan=True)
    assert np.array_equal(first.trial_uids, second.trial_uids)
    assert np.array_equal(first.availability, np.array([[1, 0, 1, 1]] * 2))

    imu1_indices = [
        index
        for index, name in enumerate(first.feature_names)
        if name.startswith("imu1_link_root_")
    ]
    assert len(imu1_indices) == 10
    assert np.isnan(first.X[:, imu1_indices]).all()
    assert first.X[0, first.feature_names.index("encoder_delta_rad_j0")] == pytest.approx(
        2.0 * math.pi / 4096.0
    )
    assert first.X[0, first.feature_names.index("imu0_base_euler_x_rad_sin")] == (
        pytest.approx(math.sin(0.1))
    )
    assert first.X[0, first.feature_names.index("imu0_base_euler_x_rad_cos")] == (
        pytest.approx(math.cos(0.1))
    )
    assert first.X[0, first.feature_names.index("imu2_link_mid_accel_z_mps2")] == 23.0
    assert first.X[0, first.feature_names.index("imu3_tip_gyro_rms_rad_s")] == (
        pytest.approx(0.04)
    )

    forbidden = ("vive", "tracker", "planned", "target_id", "segment", "timestamp")
    assert not any(
        token in name.lower() for name in first.feature_names for token in forbidden
    )
    assert not first.X.flags.writeable
    assert not first.availability.flags.writeable
    with pytest.raises(ValueError):
        first.availability[0, 1] = 1.0


def test_label_blind_rules_exclude_bad_rows_without_using_label_values() -> None:
    valid_40 = _source_row(40, 100)
    invalid_duration = _source_row(40, 101, duration_s=3.3)
    valid_41 = _source_row(41, 200)
    invalid_tip = _source_row(41, 201, tip_n=27.0)
    frame = pd.DataFrame([valid_40, invalid_duration, valid_41, invalid_tip])
    label_free = frame.loc[:, list(IDENTITY_PROJECTION_COLUMNS)].copy()

    _, _, identity = _prepare_identity(label_free, [valid_40, valid_41])
    assert identity.rows == 2
    assert identity.segment_ids.tolist() == [40, 41]
    assert identity.trial_uids.tolist() == [_uid(40, 100), _uid(41, 200)]


def test_challenge_segment_count_drift_fails_closed() -> None:
    rows = [_source_row(40, 100), _source_row(41, 200)]
    frame = pd.DataFrame(rows)
    config = _tiny_config(frame, rows)
    config["row_roles"]["challenge"]["expected_rows"] = 3
    config["row_roles"]["challenge"]["expected_segment_rows"] = {40: 2, 41: 1}
    digest = outage_config_sha256(config)

    with pytest.raises(GPUOutageDataError, match="challenge segment counts changed"):
        prepare_challenge_identity(
            frame,
            source_member_sha256=SOURCE_MEMBER_SHA256,
            challenge_config=config,
            expected_config_sha256=digest,
            enforce_pinned_counts=False,
        )


def test_challenge_config_hash_drift_fails_before_selection() -> None:
    rows = [_source_row(40, 100), _source_row(41, 200)]
    frame = pd.DataFrame(rows)
    config = _tiny_config(frame, rows)
    digest = outage_config_sha256(config)
    config["label_blind_eligibility"]["duration_s"] = [2.7, 3.2]

    with pytest.raises(GPUOutageDataError, match="config SHA-256 mismatch"):
        prepare_challenge_identity(
            frame,
            source_member_sha256=SOURCE_MEMBER_SHA256,
            challenge_config=config,
            expected_config_sha256=digest,
            enforce_pinned_counts=False,
        )


def test_challenge_labels_are_a_separate_aligned_finite_shape_gate() -> None:
    rows = [_source_row(40, 100), _source_row(41, 200)]
    frame = pd.DataFrame(rows)
    config, digest, identity = _prepare_identity(
        frame.loc[:, list(IDENTITY_PROJECTION_COLUMNS)], rows
    )

    challenge = prepare_challenge_labels(
        frame,
        identity,
        source_member_sha256=SOURCE_MEMBER_SHA256,
        challenge_config=config,
        expected_config_sha256=digest,
        enforce_pinned_counts=False,
    )
    assert isinstance(challenge, ChallengeData)
    assert challenge.y.shape == (2, 3)
    assert np.isfinite(challenge.y).all()
    assert not challenge.y.flags.writeable
    assert "0.1" not in repr(challenge)

    broken = frame.copy()
    broken.loc[broken["segment_id"].eq(41), "vive_pos_y_m"] = math.nan
    with pytest.raises(GPUOutageDataError, match="entire challenge failed"):
        prepare_challenge_labels(
            broken,
            identity,
            source_member_sha256=SOURCE_MEMBER_SHA256,
            challenge_config=config,
            expected_config_sha256=digest,
            enforce_pinned_counts=False,
        )


def test_development_projection_uses_only_eligible_segments_36_through_39() -> None:
    rows: list[dict[str, object]] = []
    for offset, segment in enumerate((36, 37, 38, 39, 40)):
        row: dict[str, object] = {
            "train_eligible": True,
            "trial_uid": _uid(segment, 1_000 + offset),
            "segment_id": segment,
            "target_id": f"T{segment:03d}",
        }
        row.update({name: float(index + offset) for index, name in enumerate(FUSION_COLUMNS)})
        row.update({name: float(index + 1) for index, name in enumerate(LABEL_COLUMNS)})
        rows.append(row)
    development = prepare_development_data(
        pd.DataFrame(rows),
        expected_segment_rows={36: 1, 37: 1, 38: 1, 39: 1},
        expected_trial_uid_sha256=None,
    )

    assert development.rows == 4
    assert sorted(development.segment_ids.tolist()) == [36, 37, 38, 39]
    assert 40 not in development.segment_ids
    assert development.X.shape == (4, 43)
    assert development.y.shape == (4, 3)
    assert np.array_equal(development.availability, np.ones((4, 4)))
    assert not development.X.flags.writeable
    assert not development.y.flags.writeable

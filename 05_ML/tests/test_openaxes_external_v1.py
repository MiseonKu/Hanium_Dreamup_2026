from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tarfile
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
import pandas as pd
import pytest

import imu_vive_learning.openaxes_external_v1 as openaxes_module
from imu_vive_learning.openaxes_external_v1 import (
    OpenAxesExternalV1Error,
    _audit_zip_envelope,
    _closed_frame_sha256,
    _complete_payload,
    _correct_quiet_prefix_gyro,
    _declared_folds,
    _open_authenticated_source,
    _robust_affine_time,
    _safe_member_name,
    _scan_nested_alignment,
    _strict_json_load,
    _validate_active_geometry,
    reconstruct_openaxes_signed_shift_truth_v1,
    validate_openaxes_external_v1,
)
from imu_vive_learning.openaxes_external_v1_cli import build_parser
from imu_vive_learning.openaxes_external_v1_config import (
    OPENAXES_EXTERNAL_V1_CONFIG_SHA256,
    OPENAXES_SCENARIOS,
    OPENAXES_SENSITIVITY_SIGNED_SHIFTS_MS,
    OpenAxesExternalV1ConfigError,
    load_openaxes_config,
    model_feature_columns,
    openaxes_config_sha256,
    validate_openaxes_config,
)


def _config() -> dict:
    return load_openaxes_config("configs/openaxes_external_v1.yaml")


def _zip_receipt(path: Path) -> dict[str, object]:
    with ZipFile(path) as archive:
        infos = archive.infolist()
    return {
        "expected_filename": path.name,
        "expected_size": path.stat().st_size,
        "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected_entries": len(infos),
        "expected_payload_bytes": sum(info.file_size for info in infos),
    }


def test_frozen_config_closes_source_schema_accounting_and_scope() -> None:
    config = _config()
    assert openaxes_config_sha256(config) == OPENAXES_EXTERNAL_V1_CONFIG_SHA256
    assert len(model_feature_columns(config)) == 36
    assert tuple(item["id"] for item in config["scenarios"]) == OPENAXES_SCENARIOS
    assert sum(item["robot_rows"] for item in config["scenarios"]) == 39_793
    assert sum(
        item["common_aligned_robot_rows"] for item in config["scenarios"]
    ) == 39_301
    assert sum(
        item["dropped_robot_head_rows"] + item["dropped_robot_tail_rows"]
        for item in config["scenarios"]
    ) == 492
    assert sum(sum(item["imu_rows"].values()) for item in config["scenarios"]) == 166_106
    assert [
        item["id"]
        for item in config["scenarios"]
        if item.get("calibration_truth_complete", True) is False
    ] == ["circle-200"]
    sensitivity = config["alignment"]["sensitivity_recipe"]
    assert tuple(sensitivity["signed_target_shifts_ms"]) == (
        OPENAXES_SENSITIVITY_SIGNED_SHIFTS_MS
    )
    assert sensitivity["preparation_executes_recipe"] is False
    assert sensitivity["same_base_uid_pairing_required"] is True
    assert sensitivity["reporting_methods"] == [
        "frozen_comparator",
        "frozen_objective",
    ]
    assert sensitivity["paired_method_delta_sign"] == (
        "frozen_objective_minus_frozen_comparator"
    )
    assert sensitivity["model_selection_use_allowed"] is False
    assert sensitivity["claim_gate_use_allowed"] is False
    assert config["limitations"]["offline_transductive_preparation"] is True
    assert config["limitations"]["acausal_preparation"] is True
    assert config["limitations"]["causal_or_online_claim_allowed"] is False
    assert config["contract"] == {
        "name": "openaxes.external.preparation.v1",
        "research_role": "secondary_objective_portability_benchmark",
        "preparation_only": True,
        "independent_so101_confirmation": False,
        "so101_merge_allowed": False,
        "model_training_allowed": False,
        "model_selection_allowed": False,
        "challenge_access_allowed": False,
    }


def test_config_rejects_semantically_dangerous_drift() -> None:
    config = _config()
    attacks = []
    changed = deepcopy(config)
    changed["contract"]["so101_merge_allowed"] = True
    attacks.append(changed)
    changed = deepcopy(config)
    changed["alignment"]["label_assisted"] = False
    attacks.append(changed)
    changed = deepcopy(config)
    changed["alignment"]["sensitivity_recipe"]["signed_target_shifts_ms"] = [
        -10,
        10,
    ]
    attacks.append(changed)
    changed = deepcopy(config)
    changed["alignment"]["sensitivity_recipe"][
        "model_selection_use_allowed"
    ] = True
    attacks.append(changed)
    changed = deepcopy(config)
    changed["folds"]["outer_test_fitting_allowed"] = True
    attacks.append(changed)
    changed = deepcopy(config)
    changed["schema"]["sensor_channels"].append("timestamp")
    attacks.append(changed)
    changed = deepcopy(config)
    changed["schema"]["forbidden_model_inputs"].remove("tcp_x_m")
    attacks.append(changed)
    changed = deepcopy(config)
    changed["source"]["redistribution_allowed"] = True
    attacks.append(changed)
    changed = deepcopy(config)
    changed["scenarios"][0]["dropped_robot_tail_rows"] = 194
    attacks.append(changed)
    for attack in attacks:
        with pytest.raises(OpenAxesExternalV1ConfigError):
            validate_openaxes_config(attack)


def test_duplicate_yaml_key_is_rejected_before_hash_validation(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("contract: {}\ncontract: {}\n", encoding="utf-8")
    with pytest.raises(OpenAxesExternalV1ConfigError, match="duplicate"):
        load_openaxes_config(path)


def test_declared_folds_are_exact_closed_group_partitions() -> None:
    folds = _declared_folds()
    assert len(folds) == 14
    assert [fold["family"] for fold in folds].count("scenario_loso") == 8
    assert [fold["family"] for fold in folds].count("speed_pair_out") == 4
    assert [fold["family"] for fold in folds].count("shape_out") == 2
    for fold in folds:
        train = fold["train_scenarios"]
        test = fold["test_scenarios"]
        assert not set(train).intersection(test)
        assert set(train).union(test) == set(OPENAXES_SCENARIOS)
        assert len(train) == len(set(train))
        assert len(test) == len(set(test))
        assert fold["outer_test_fitting_allowed"] is False
        assert fold["inner_selection"] == "scenario_blocked_training_groups_only"


def test_zip_envelope_authenticates_then_rejects_hash_and_unsafe_path(
    tmp_path: Path,
) -> None:
    safe = tmp_path / "safe.zip"
    with ZipFile(safe, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("README.md", "safe")
        archive.writestr("data/input.csv", "a,b\n1,2\n")
    receipt = _zip_receipt(safe)
    result = _audit_zip_envelope(safe, **receipt)
    assert result["crc_verified"] is True
    assert result["unsafe_paths"] == 0
    assert result["single_no_follow_fd_bound"] is True
    assert result["same_handle_size_sha_crc_and_reads"] is True
    assert result["pre_post_fstat_stable"] is True
    assert result["post_path_binding_verified"] is True
    assert len(set(result["file_identity"][key]["inode"] for key in result["file_identity"])) == 1
    with pytest.raises(OpenAxesExternalV1Error, match="SHA-256"):
        _audit_zip_envelope(
            safe,
            **{**receipt, "expected_sha256": "0" * 64},
        )

    unsafe = tmp_path / "unsafe.zip"
    with ZipFile(unsafe, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("../escape.csv", "a\n1\n")
    with pytest.raises(OpenAxesExternalV1Error, match="unsafe"):
        _audit_zip_envelope(unsafe, **_zip_receipt(unsafe))

    with pytest.raises(OpenAxesExternalV1Error, match="filename"):
        _audit_zip_envelope(
            safe,
            **{**receipt, "expected_filename": "official.zip"},
        )


@pytest.mark.parametrize(
    "name",
    (
        "C:/escape.csv",
        "name:stream.csv",
        "a//b.csv",
        "a/./b.csv",
        "a/../b.csv",
        "a\\b.csv",
        "/absolute.csv",
        "trailing./file.csv",
    ),
)
def test_archive_member_paths_reject_windows_drives_colons_and_aliases(
    name: str,
) -> None:
    assert _safe_member_name(name) is False


def _set_encrypted_flags(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        start = 0
        found = 0
        while True:
            index = payload.find(signature, start)
            if index < 0:
                break
            flags = int.from_bytes(
                payload[index + flag_offset : index + flag_offset + 2], "little"
            )
            payload[index + flag_offset : index + flag_offset + 2] = (
                flags | 1
            ).to_bytes(2, "little")
            found += 1
            start = index + len(signature)
        assert found >= 1
    path.write_bytes(payload)


def _corrupt_first_stored_member(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    local = payload.index(b"PK\x03\x04")
    name_length, extra_length = struct.unpack_from("<HH", payload, local + 26)
    data_offset = local + 30 + name_length + extra_length
    payload[data_offset] ^= 0x01
    path.write_bytes(payload)


def test_zip_rejects_encryption_duplicates_crc_and_canonical_aliases(
    tmp_path: Path,
) -> None:
    encrypted = tmp_path / "encrypted.zip"
    with ZipFile(encrypted, "w", compression=ZIP_STORED) as archive:
        archive.writestr("data.csv", "x\n1\n")
    _set_encrypted_flags(encrypted)
    with pytest.raises(OpenAxesExternalV1Error, match="encrypted"):
        _audit_zip_envelope(encrypted, **_zip_receipt(encrypted))

    duplicate = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning):
        with ZipFile(duplicate, "w", compression=ZIP_STORED) as archive:
            archive.writestr("same.csv", "x\n1\n")
            archive.writestr("same.csv", "x\n2\n")
    with pytest.raises(OpenAxesExternalV1Error, match="duplicate"):
        _audit_zip_envelope(duplicate, **_zip_receipt(duplicate))

    corrupt = tmp_path / "corrupt.zip"
    with ZipFile(corrupt, "w", compression=ZIP_STORED) as archive:
        archive.writestr("data.csv", "x\n1\n")
    _corrupt_first_stored_member(corrupt)
    with pytest.raises(OpenAxesExternalV1Error, match="CRC|authenticate"):
        _audit_zip_envelope(corrupt, **_zip_receipt(corrupt))

    alias = tmp_path / "alias.zip"
    with ZipFile(alias, "w", compression=ZIP_STORED) as archive:
        archive.writestr("Data/value.csv", "x\n1\n")
        archive.writestr("data/value.csv", "x\n2\n")
    with pytest.raises(OpenAxesExternalV1Error, match="alias"):
        _audit_zip_envelope(alias, **_zip_receipt(alias))


def test_bound_source_rejects_atomic_path_replacement(tmp_path: Path) -> None:
    source = tmp_path / "official.zip"
    replacement = tmp_path / "replacement.zip"
    for path in (source, replacement):
        with ZipFile(path, "w", compression=ZIP_STORED) as archive:
            archive.writestr("data.csv", "x\n1\n")
    receipt = _zip_receipt(source)
    with pytest.raises(OpenAxesExternalV1Error, match="identity changed|replaced"):
        with _open_authenticated_source(source, **receipt) as bound:
            assert bound.archive.read("data.csv") == b"x\n1\n"
            os.replace(replacement, source)


def test_nested_tar_rejects_windows_drive_member(tmp_path: Path) -> None:
    nested_buffer = io.BytesIO()
    with tarfile.open(fileobj=nested_buffer, mode="w:bz2") as nested:
        payload = b"unsafe"
        info = tarfile.TarInfo("C:/escape.csv")
        info.size = len(payload)
        nested.addfile(info, io.BytesIO(payload))
    outer = tmp_path / "outer.zip"
    with ZipFile(outer, "w", compression=ZIP_STORED) as archive:
        archive.writestr("nested.tar.bz2", nested_buffer.getvalue())
    config = {
        "source": {
            "nested_tar_member": "nested.tar.bz2",
            "expected_nested_tar_entries": 1,
            "expected_nested_tar_payload_bytes": len(b"unsafe"),
        },
        "schema": {
            "aligned_header": [
                "TIME",
                "QUATERNION_0",
                "QUATERNION_1",
                "QUATERNION_2",
                "QUATERNION_3",
            ]
        },
        "scenarios": [],
    }
    with ZipFile(outer) as archive:
        with pytest.raises(OpenAxesExternalV1Error, match="unsafe nested TAR"):
            _scan_nested_alignment(archive, config=config)


def test_robust_imu_clock_fit_is_monotonic_and_rejects_bad_slope() -> None:
    index = np.arange(1000, dtype=np.int64) + 200
    jitter = 1500.0 * np.sin(np.arange(1000, dtype=np.float64) / 13.0)
    time = np.asarray(1_663_000_000_000_000 + 10_000 * np.arange(1000) + jitter, dtype=np.int64)
    time[500] += 20_000
    fitted, receipt = _robust_affine_time(
        index,
        time,
        iterations=8,
        slope_min=9500.0,
        slope_max=10500.0,
        residual_max=25_000.0,
    )
    assert np.all(np.diff(fitted) > 0)
    assert receipt["slope_us_per_index"] == pytest.approx(10_000.0, abs=2.0)
    assert receipt["absolute_residual_us"]["max"] <= 25_000.0

    bad_time = np.asarray(1_663_000_000_000_000 + 12_000 * np.arange(1000), dtype=np.int64)
    with pytest.raises(OpenAxesExternalV1Error, match="slope"):
        _robust_affine_time(
            index,
            bad_time,
            iterations=8,
            slope_min=9500.0,
            slope_max=10500.0,
            residual_max=25_000.0,
        )


def test_quiet_prefix_correction_is_label_free_and_fail_closed() -> None:
    features = np.zeros((30, 9), dtype=np.float64)
    features[:, :6] = np.arange(30, dtype=np.float64)[:, None]
    features[:, 6:9] = np.array([0.4, -0.2, 0.1])
    features[20:, 6] += 0.3
    corrected, mean, std = _correct_quiet_prefix_gyro(
        features,
        prefix_samples=20,
        max_axis_std_rad_s=0.01,
    )
    assert mean.tolist() == pytest.approx([0.4, -0.2, 0.1])
    assert std.tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-15)
    assert corrected[:20, 6:9] == pytest.approx(np.zeros((20, 3)))
    assert np.array_equal(corrected[:, :6], features[:, :6])
    assert np.array_equal(features[0, 6:9], np.array([0.4, -0.2, 0.1]))

    noisy = features.copy()
    noisy[:20, 8] = np.tile([-0.1, 0.1], 10)
    with pytest.raises(OpenAxesExternalV1Error, match="not quiet"):
        _correct_quiet_prefix_gyro(
            noisy,
            prefix_samples=20,
            max_axis_std_rad_s=0.01,
        )


def test_frozen_circle_geometry_accepts_path_and_rejects_boundary_search() -> None:
    theta = np.linspace(0.0, 6.0 * np.pi, 300)
    xyz = np.column_stack(
        (
            np.full(len(theta), -0.43194),
            -0.096892 + 0.2 * np.cos(theta),
            0.604972 + 0.2 * np.sin(theta),
        )
    )
    scenario = {
        "id": "circle-test",
        "active_trace": {
            "start_robot_row_inclusive": 0,
            "end_robot_row_inclusive": len(xyz) - 1,
            "geometry": {
                "kind": "circle_yz",
                "plane_x_m": -0.43194,
                "plane_tolerance_m": 0.001,
                "center_y_m": -0.096892,
                "center_z_m": 0.604972,
                "radius_m": 0.2,
                "path_tolerance_m": 0.0016,
            },
        },
    }
    receipt = _validate_active_geometry(xyz, scenario=scenario)
    assert receipt["source"] == "frozen_robot_truth_geometry_before_model_results"
    attacked = xyz.copy()
    attacked[120, 1] += 0.02
    with pytest.raises(OpenAxesExternalV1Error, match="frozen path"):
        _validate_active_geometry(attacked, scenario=scenario)


def test_frozen_triangle_geometry_accepts_edges_and_rejects_off_path_row() -> None:
    vertices = np.asarray(
        [[-0.3201, 0.5442], [0.36312, 0.21932], [-0.322, 0.21925]],
        dtype=np.float64,
    )
    yz = np.vstack(
        [
            np.linspace(vertices[index], vertices[(index + 1) % 3], 120)
            for index in range(3)
        ]
    )
    xyz = np.column_stack((np.full(len(yz), -0.54698), yz))
    scenario = {
        "id": "triangle-test",
        "active_trace": {
            "start_robot_row_inclusive": 0,
            "end_robot_row_inclusive": len(xyz) - 1,
            "geometry": {
                "kind": "triangle_yz",
                "plane_x_m": -0.54698,
                "plane_tolerance_m": 0.001,
                "vertices_yz_m": vertices.tolist(),
                "path_tolerance_m": 0.0021,
            },
        },
    }
    receipt = _validate_active_geometry(xyz, scenario=scenario)
    assert receipt["observed_path_error_max_m"] == pytest.approx(0.0, abs=1e-12)
    attacked = xyz.copy()
    attacked[150, 1:3] += np.array([0.02, 0.02])
    with pytest.raises(OpenAxesExternalV1Error, match="frozen path"):
        _validate_active_geometry(attacked, scenario=scenario)


def test_strict_json_rejects_duplicate_and_nonfinite_values(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(OpenAxesExternalV1Error, match="duplicate"):
        _strict_json_load(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"x":NaN}', encoding="utf-8")
    with pytest.raises(OpenAxesExternalV1Error, match="non-finite"):
        _strict_json_load(nonfinite)


def test_closed_frame_hash_is_deterministic_ordered_and_tamper_evident() -> None:
    schema = (("uid", "utf8"), ("time", "int64"), ("target", "float64"))
    frame = pd.DataFrame(
        {
            "uid": ["a", "b"],
            "time": np.asarray([10, 20], dtype=np.int64),
            "target": np.asarray([0.25, -0.5], dtype=np.float64),
        }
    )
    expected = _closed_frame_sha256(frame, schema)
    assert _closed_frame_sha256(frame.copy(), schema) == expected
    attacked = frame.copy()
    attacked.loc[0, "target"] += 1e-12
    assert _closed_frame_sha256(attacked, schema) != expected
    assert _closed_frame_sha256(frame.iloc[::-1].reset_index(drop=True), schema) != expected
    nonfinite = frame.copy()
    nonfinite.loc[0, "target"] = np.nan
    with pytest.raises(OpenAxesExternalV1Error, match="finite"):
        _closed_frame_sha256(nonfinite, schema)


def _synthetic_semantic_and_frame(config: dict) -> tuple[dict, pd.DataFrame]:
    features = list(model_feature_columns(config))
    frame = pd.DataFrame(
        {
            "row_uid": ["synthetic-0", "synthetic-1"],
            "scenario_id": ["circle-100", "circle-100"],
            "sample_index": np.asarray([0, 1], dtype=np.int64),
            "aligned_time_us": np.asarray([0, 10_000], dtype=np.int64),
            **{
                column: np.asarray([0.0, 1.0], dtype=np.float64)
                for column in features
            },
            "tcp_x_m": np.asarray([0.1, 0.2], dtype=np.float64),
            "tcp_y_m": np.asarray([0.3, 0.4], dtype=np.float64),
            "tcp_z_m": np.asarray([0.5, 0.6], dtype=np.float64),
        }
    )
    source_files = {"contract.py": {"bytes": 1, "sha256": "a" * 64}}
    semantic = {
        "contract": "openaxes.external.preparation.v1",
        "counts": {"prepared_active_rows": len(frame)},
        "provenance": {
            "contract_and_lock_sources": {
                "files": source_files,
                "dependency_lock_source": "pyproject.toml",
                "exact_dependency_lock_file_present": False,
                "source_set_sha256": "b" * 64,
            },
            "repository": {
                "git_head": "c" * 40,
                "dirty_path_snapshot_sha256": "d" * 64,
            },
            "runtime": {"python": {"version": "test"}},
            "stage": {
                "name": "pre_model_result_contract_capture",
                "preparation_only": True,
                "contract_sources_committed_at_git_head": False,
                "downstream_model_results_blocked_by_uncommitted_contract": True,
                "commit_required_before_model_results": True,
                "required_action": (
                    "commit_contract_sources_then_reprepare_before_any_model_result"
                ),
            },
        },
    }
    return semantic, frame


def _write_synthetic_run(
    root: Path, config: dict, semantic: dict, frame: pd.DataFrame
) -> None:
    root.mkdir()
    parquet = root / "prepared.parquet"
    manifest_path = root / "manifest.json"
    complete_path = root / "COMPLETE.json"
    openaxes_module.write_parquet(parquet, frame)
    manifest = deepcopy(semantic)
    manifest["files"] = {
        "prepared.parquet": {
            "path": "prepared.parquet",
            "bytes": parquet.stat().st_size,
            "sha256": openaxes_module.sha256_file(parquet),
            "rows": len(frame),
        }
    }
    openaxes_module.write_json(manifest_path, manifest)
    complete = _complete_payload(
        config=config,
        manifest=manifest,
        manifest_path=manifest_path,
        parquet_path=parquet,
    )
    openaxes_module.write_json(complete_path, complete)


def _reseal_synthetic_run(root: Path, config: dict, manifest: dict) -> None:
    manifest_path = root / "manifest.json"
    parquet = root / "prepared.parquet"
    openaxes_module.write_json(manifest_path, manifest)
    complete = _complete_payload(
        config=config,
        manifest=manifest,
        manifest_path=manifest_path,
        parquet_path=parquet,
    )
    openaxes_module.write_json(root / "COMPLETE.json", complete)


def test_unsigned_and_resigned_manifest_or_parquet_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    semantic, expected_frame = _synthetic_semantic_and_frame(config)
    monkeypatch.setattr(
        openaxes_module,
        "_prepare_in_memory",
        lambda _source, _config: (expected_frame.copy(), deepcopy(semantic)),
    )

    unsigned = tmp_path / "unsigned"
    _write_synthetic_run(unsigned, config, semantic, expected_frame)
    (unsigned / "COMPLETE.json").unlink()
    with pytest.raises(OpenAxesExternalV1Error, match="file set"):
        validate_openaxes_external_v1("unused.zip", config, unsigned)

    resigned_manifest = tmp_path / "resigned-manifest"
    _write_synthetic_run(resigned_manifest, config, semantic, expected_frame)
    manifest = _strict_json_load(resigned_manifest / "manifest.json")
    manifest["attacker_resigned"] = True
    _reseal_synthetic_run(resigned_manifest, config, manifest)
    with pytest.raises(OpenAxesExternalV1Error, match="semantic manifest replay"):
        validate_openaxes_external_v1("unused.zip", config, resigned_manifest)

    resigned_parquet = tmp_path / "resigned-parquet"
    _write_synthetic_run(resigned_parquet, config, semantic, expected_frame)
    attacked_frame = expected_frame.copy()
    attacked_frame.loc[0, "tcp_x_m"] += 1.0
    parquet = resigned_parquet / "prepared.parquet"
    openaxes_module.write_parquet(parquet, attacked_frame)
    manifest = _strict_json_load(resigned_parquet / "manifest.json")
    manifest["files"]["prepared.parquet"] = {
        "path": "prepared.parquet",
        "bytes": parquet.stat().st_size,
        "sha256": openaxes_module.sha256_file(parquet),
        "rows": len(attacked_frame),
    }
    _reseal_synthetic_run(resigned_parquet, config, manifest)
    with pytest.raises(OpenAxesExternalV1Error, match="parquet replay"):
        validate_openaxes_external_v1("unused.zip", config, resigned_parquet)


def test_standalone_cli_has_only_prepare_and_validate() -> None:
    parser = build_parser()
    prepare = parser.parse_args(
        [
            "prepare",
            "--source",
            "source.zip",
            "--config",
            "config.yaml",
            "--output",
            "prepared",
        ]
    )
    validate = parser.parse_args(
        [
            "validate",
            "--source",
            "source.zip",
            "--config",
            "config.yaml",
            "--run-dir",
            "prepared",
        ]
    )
    assert prepare.command == "prepare"
    assert validate.command == "validate"
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])


def test_preparer_contains_no_external_execution_or_archive_extraction() -> None:
    source = Path(openaxes_module.__file__).read_text(encoding="utf-8")
    forbidden_tokens = (
        "extractall(",
        "extract(\"",
        "os.system",
        "Popen(",
        "runpy",
        "exec(",
        "eval(",
        "shell=True",
    )
    assert not any(token in source for token in forbidden_tokens)
    assert source.count("subprocess.run(") == 1
    assert '["git", "-C", str(repo), *arguments]' in source
    assert "tarfile.open(fileobj=compressed, mode=\"r|bz2\")" in source
    assert "archive.read(\"LICENSE.txt\")" in source


def test_complete_contract_file_documents_limits_and_smoke_counts() -> None:
    text = Path("docs/OPENAXES_EXTERNAL_V1_CONTRACT.md").read_text(encoding="utf-8")
    required = (
        "22,482",
        "39,301",
        "492",
        "label-assisted",
        "±10",
        "±20",
        "±50",
        "[-50, -20, -10, +10, +20, +50]",
        "O_NOFOLLOW",
        "offline-transductive",
        "acausal",
        "downstream_model_results_blocked_by_uncommitted_contract=true",
        "not 14 independent replications",
        "must not be committed",
    )
    assert all(value in text for value in required)
    config = _config()
    assert json.dumps(
        config["alignment"]["sensitivity_recipe"]["signed_target_shifts_ms"]
    ) == "[-50, -20, -10, 10, 20, 50]"


def test_actual_signed_shift_truth_replay_when_smoke_artifact_is_present() -> None:
    source = Path(
        "artifacts/openaxes_external_v1/source/"
        "openaxes-example-robot-dataset.zip"
    )
    run = Path("artifacts/openaxes_external_v1/prepared_v1_hardened")
    if not source.is_file() or not run.is_dir():
        pytest.skip("ignored OpenAXES smoke artifact is not present")
    complete = _strict_json_load(run / "COMPLETE.json")
    if complete.get("config_sha256") != OPENAXES_EXTERNAL_V1_CONFIG_SHA256:
        pytest.skip("ignored OpenAXES smoke artifact predates this contract")
    config = _config()
    truth, receipt = reconstruct_openaxes_signed_shift_truth_v1(
        source, config, run
    )
    expected_columns = [
        "row_uid",
        "scenario_id",
        "base_aligned_time_us",
        "shift_ms",
        "shifted_target_time_us",
        "shifted_tcp_x_m",
        "shifted_tcp_y_m",
        "shifted_tcp_z_m",
    ]
    assert list(truth.columns) == expected_columns
    assert len(truth) == 22_482 * 6
    assert truth[["row_uid", "shift_ms"]].duplicated().sum() == 0
    assert sorted(truth["shift_ms"].unique().tolist()) == [
        -50,
        -20,
        -10,
        10,
        20,
        50,
    ]
    assert np.array_equal(
        truth["shifted_target_time_us"].to_numpy()
        - truth["base_aligned_time_us"].to_numpy(),
        truth["shift_ms"].to_numpy() * 1_000,
    )
    assert not set(model_feature_columns(config)).intersection(truth.columns)
    assert receipt["scope"] == {
        "read_only_truth_reconstruction": True,
        "features_included": False,
        "model_fit_performed": False,
        "metrics_computed": False,
        "selection_performed": False,
        "claim_gate_decided": False,
        "artifact_written": False,
        "semantic_validator_required_before_model_use": True,
    }
    assert receipt["output"]["rows_per_shift"] == {
        "-50": 22_482,
        "-20": 22_482,
        "-10": 22_482,
        "10": 22_482,
        "20": 22_482,
        "50": 22_482,
    }
    assert all(
        dropped == 0
        for scenario in receipt["output"][
            "dropped_edge_rows_by_scenario_and_shift"
        ].values()
        for dropped in scenario.values()
    )
    truth_schema = tuple(
        (name, "utf8" if name in {"row_uid", "scenario_id"} else "int64")
        if name
        in {
            "row_uid",
            "scenario_id",
            "base_aligned_time_us",
            "shift_ms",
            "shifted_target_time_us",
        }
        else (name, "float64")
        for name in expected_columns
    )
    assert receipt["output"]["long_table_sha256"] == _closed_frame_sha256(
        truth, truth_schema
    )
    assert receipt["prepared"]["stage"][
        "downstream_model_results_blocked_by_uncommitted_contract"
    ] is True

    locked_hashes = complete.get("contract_source_hashes", {})
    sources_still_match = all(
        Path(path).is_file()
        and openaxes_module.sha256_file(Path(path)) == expected_hash
        for path, expected_hash in locked_hashes.items()
    )
    if sources_still_match:
        result = validate_openaxes_external_v1(source, config, run)
        assert result["semantic_replay"] is True
    else:
        with pytest.raises(OpenAxesExternalV1Error):
            validate_openaxes_external_v1(source, config, run)

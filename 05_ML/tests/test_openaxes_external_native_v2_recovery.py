from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from imu_vive_learning import openaxes_external_native_v2_experiment as frozen
from imu_vive_learning import openaxes_external_native_v2_recovery as recovery


PUBLIC_SOURCE_SET = "a" * 64


def _tasks() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            fold_id=f"fold_{index:02d}", candidate_ordinal=0, seed_ordinal=0
        )
        for index in range(14)
    ]


def _ensembles() -> dict[tuple[str, str], pd.DataFrame]:
    result: dict[tuple[str, str], pd.DataFrame] = {}
    for candidate_ordinal, candidate_id in enumerate(frozen.CANDIDATE_IDS):
        for fold_ordinal in range(14):
            scenario_id = f"circle-{(100, 150, 200, 250)[fold_ordinal % 4]}"
            row_uid = f"uid-{fold_ordinal:02d}"
            rows = []
            for view_ordinal, view_id in enumerate(frozen.VIEW_IDS):
                target = float(fold_ordinal) / 100.0
                rows.append(
                    {
                        "row_uid": row_uid,
                        "scenario_id": scenario_id,
                        "sample_index": fold_ordinal,
                        "aligned_time_us": 1_000_000 + fold_ordinal * 10_000,
                        "view_id": view_id,
                        "prediction_x_m": target + candidate_ordinal * 0.001,
                        "prediction_y_m": target + view_ordinal * 0.001,
                        "prediction_z_m": target,
                        "tcp_x_m": target,
                        "tcp_y_m": target,
                        "tcp_z_m": target,
                    }
                )
            result[(candidate_id, f"fold_{fold_ordinal:02d}")] = pd.DataFrame(rows)
    return result


def _truth(ensembles: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    base = recovery._base_identity_from_ensembles(ensembles)
    rows: list[dict[str, object]] = []
    for row in base.to_dict(orient="records"):
        for shift_ms in recovery.SHIFT_ORDER:
            value = float(row["sample_index"]) / 100.0 + shift_ms / 100_000.0
            rows.append(
                {
                    "row_uid": row["row_uid"],
                    "scenario_id": row["scenario_id"],
                    "base_aligned_time_us": row["aligned_time_us"],
                    "shift_ms": shift_ms,
                    "shifted_target_time_us": row["aligned_time_us"]
                    + shift_ms * 1000,
                    "shifted_tcp_x_m": value,
                    "shifted_tcp_y_m": value,
                    "shifted_tcp_z_m": value,
                }
            )
    return pd.DataFrame(rows, columns=recovery.SIGNED_TRUTH_COLUMNS)


def _receipt(
    ensembles: dict[tuple[str, str], pd.DataFrame], truth: pd.DataFrame
) -> dict[str, object]:
    base = recovery._base_identity_from_ensembles(ensembles)
    base_schema = (
        ("row_uid", "utf8"),
        ("scenario_id", "utf8"),
        ("sample_index", "int64"),
        ("aligned_time_us", "int64"),
    )
    truth_schema = tuple(
        zip(
            recovery.SIGNED_TRUTH_COLUMNS,
            (
                "utf8",
                "utf8",
                "int64",
                "int64",
                "int64",
                "float64",
                "float64",
                "float64",
            ),
            strict=True,
        )
    )
    base_counts = {
        str(key): int(value)
        for key, value in base.groupby("scenario_id", sort=False).size().items()
    }
    drops = {
        scenario_id: {
            str(shift): base_rows
            - int(
                (
                    (truth["scenario_id"] == scenario_id)
                    & (truth["shift_ms"] == shift)
                ).sum()
            )
            for shift in recovery.SHIFT_ORDER
        }
        for scenario_id, base_rows in base_counts.items()
    }
    return {
        "contract": "openaxes.external.signed_shift_truth.v1",
        "preparation_contract": "openaxes.external.preparation.v1",
        "config_sha256": (
            "05c3f44b58a16cebb8c0cef1338a42ae55a0fd84dd9f87db09196a7b740b32c6"
        ),
        "scope": {
            "read_only_truth_reconstruction": True,
            "features_included": False,
            "model_fit_performed": False,
            "metrics_computed": False,
            "selection_performed": False,
            "claim_gate_decided": False,
            "artifact_written": False,
            "semantic_validator_required_before_model_use": True,
        },
        "source": {
            "filename": "source.zip",
            "size_bytes": 1,
            "sha256": "b" * 64,
            "zip_entries": 1,
            "zip_payload_bytes": 1,
            "crc_verified": True,
            "single_no_follow_fd_bound": True,
            "same_handle_size_sha_crc_and_reads": True,
            "pre_post_fstat_stable": True,
            "post_path_binding_verified": True,
        },
        "prepared": {
            "manifest_sha256": "c" * 64,
            "parquet_sha256": "d" * 64,
            "complete_sha256": "e" * 64,
            "complete_manifest_sha256": "c" * 64,
            "complete_parquet_sha256": "d" * 64,
            "base_rows": len(base),
            "base_identity_sha256": recovery._closed_frame_sha256(
                base, base_schema
            ),
            "base_exact_source_replay": True,
            "stage": {"synthetic": "authority"},
        },
        "recipe": {
            "signed_target_shifts_ms": list(recovery.SHIFT_ORDER),
            "no_extrapolation": True,
            "edge_policy": "drop_base_rows_without_shifted_target_support",
            "pairing_key": "row_uid",
            "same_base_uid_pairing_required": True,
            "model_selection_use_allowed": False,
            "claim_gate_use_allowed": False,
        },
        "output": {
            "closed_schema": list(recovery.SIGNED_TRUTH_COLUMNS),
            "row_order": "scenario_then_signed_shift_then_base_time",
            "target_columns_only_with_pairing_metadata": True,
            "rows": len(truth),
            "rows_per_shift": {
                str(shift): int((truth["shift_ms"] == shift).sum())
                for shift in recovery.SHIFT_ORDER
            },
            "dropped_edge_rows_by_scenario_and_shift": drops,
            "long_table_sha256": recovery._closed_frame_sha256(
                truth, truth_schema
            ),
        },
        "deterministic_receipt": True,
    }


def test_zero_drop_recovery_closes_exact_metric_grid() -> None:
    ensembles = _ensembles()
    truth = _truth(ensembles)
    report = recovery.aggregate_signed_shift_recovery(
        _tasks(),
        ensembles,
        truth,
        _receipt(ensembles, truth),
        expected_public_source_set_sha256=PUBLIC_SOURCE_SET,
    )
    assert report["status"] == "complete"
    assert len(report["support_cells"]) == 252
    assert len(report["rows"]) == 1260
    assert len(report["primary_robust_J"]) == 18
    recovery.validate_signed_shift_recovery_report(recovery._payload(report))


def test_nonzero_drop_is_explicit_and_computes_no_partial_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensembles = _ensembles()
    truth = _truth(ensembles).iloc[1:].reset_index(drop=True)
    monkeypatch.setattr(
        recovery.native_training,
        "fixed_rmse3d_m",
        lambda *_args, **_kwargs: pytest.fail("incomplete branch computed a metric"),
    )
    report = recovery.aggregate_signed_shift_recovery(
        _tasks(),
        ensembles,
        truth,
        _receipt(ensembles, truth),
        expected_public_source_set_sha256=PUBLIC_SOURCE_SET,
    )
    assert report["status"] == "incomplete"
    assert report["rows"] == []
    assert report["primary_robust_J"] == []
    assert report["candidate_fold_shift_view_metric_cells"] == 0
    recovery.validate_signed_shift_recovery_report(recovery._payload(report))


@pytest.mark.parametrize("mutation", ["duplicate", "time", "recipe", "receipt_hash"])
def test_canonical_truth_authority_rejects_adversarial_mutations(
    mutation: str,
) -> None:
    ensembles = _ensembles()
    truth = _truth(ensembles)
    receipt = _receipt(ensembles, truth)
    if mutation == "duplicate":
        truth = pd.concat([truth, truth.iloc[[0]]], ignore_index=True)
    elif mutation == "time":
        truth.loc[0, "shifted_target_time_us"] += 1
    elif mutation == "recipe":
        receipt["recipe"]["model_selection_use_allowed"] = True  # type: ignore[index]
    else:
        receipt["output"]["long_table_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(recovery.OpenAxesNativeV2RecoveryError):
        recovery.aggregate_signed_shift_recovery(
            _tasks(),
            ensembles,
            truth,
            receipt,
            expected_public_source_set_sha256=PUBLIC_SOURCE_SET,
        )


def test_sealed_stage_resume_never_recomputes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / recovery.STAGE_SEAL_PATH).write_bytes(b"sealed")
    stage_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
    sentinel = {"sealed": True}
    source = {
        "original_science_authority": {},
        "failed_prestate": recovery._expected_failed_prestate(),
    }
    monkeypatch.setattr(
        recovery,
        "_load_external_authorities",
        lambda *_args, **_kwargs: (source, {}, {}, {}),
    )
    monkeypatch.setattr(
        recovery,
        "validate_stage_seal",
        lambda *_args, **_kwargs: sentinel,
    )
    monkeypatch.setattr(
        recovery,
        "_build_and_seal_stage",
        lambda *_args, **_kwargs: pytest.fail("sealed resume recomputed outputs"),
    )
    try:
        assert recovery.stage_compound_recovery(
            -1,
            -1,
            stage_fd,
            -1,
            -1,
            -1,
            original_source_evidence_commit="1" * 40,
            recovery_source_evidence_commit="2" * 40,
        ) is sentinel
    finally:
        os.close(stage_fd)


def test_publish_boundary_accepts_only_exact_ordered_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    (run / "reports").mkdir(parents=True)
    staged = {path: f"{path}\n".encode() for path in recovery.OUTPUT_PATHS}
    monkeypatch.setattr(recovery, "_validate_base_tree_after_staging", lambda _fd: None)
    run_fd = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for path in recovery.OUTPUT_PATHS[:2]:
            destination = run / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(staged[path])
        assert recovery._validate_publish_boundary(run_fd, staged) == 2
        path = recovery.OUTPUT_PATHS[3]
        destination = run / path
        destination.write_bytes(staged[path])
        with pytest.raises(recovery.OpenAxesNativeV2RecoveryError):
            recovery._validate_publish_boundary(run_fd, staged)
    finally:
        os.close(run_fd)


def test_public_source_roster_accepts_canonical_json_key_order(tmp_path: Path) -> None:
    files: dict[str, dict[str, object]] = {}
    for workspace_path in recovery.V1_PUBLIC_SOURCE_PATHS:
        relative = workspace_path.removeprefix("09_SO101_IMU_VIVE_LEARNING/")
        raw = (workspace_path + "\n").encode()
        destination = tmp_path / workspace_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        files[relative] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    canonical_files = {key: files[key] for key in sorted(files)}
    manifest = {
        "provenance": {
            "contract_and_lock_sources": {
                "files": canonical_files,
                "dependency_lock_source": "synthetic",
                "exact_dependency_lock_file_present": True,
                "source_set_sha256": recovery._canonical_hash(files),
            }
        }
    }
    workspace_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert (
            recovery._validate_preparation_public_source_set(workspace_fd, manifest)
            == recovery._canonical_hash(files)
        )
        for mutation in ("missing", "extra", "replaced"):
            changed = deepcopy(manifest)
            changed_files = changed["provenance"]["contract_and_lock_sources"]["files"]
            first = next(iter(changed_files))
            if mutation == "missing":
                del changed_files[first]
            elif mutation == "extra":
                changed_files["unexpected.py"] = deepcopy(changed_files[first])
            else:
                row = changed_files.pop(first)
                changed_files["replaced.py"] = row
            with pytest.raises(recovery.OpenAxesNativeV2RecoveryError):
                recovery._validate_preparation_public_source_set(
                    workspace_fd, changed
                )
    finally:
        os.close(workspace_fd)


def test_recovery_source_has_no_scientific_execution_entrypoint() -> None:
    source_path = Path(recovery.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_attributes = {
        "fit",
        "backward",
        "step",
        "load_outer_features",
        "load_outer_labels",
        "evaluate_outer_once",
        "_run_fresh_task_children",
    }
    observed = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    observed.update(
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    )
    assert forbidden_attributes.isdisjoint(observed)
    assert "subprocess" not in source
    assert source.count("reconstruct_openaxes_signed_shift_truth_v1(") == 1
    assert source.count("load_openaxes_config(") == 1


def test_regular_fd_argument_duplicates_inherited_regular_file(
    tmp_path: Path,
) -> None:
    source = tmp_path / "authority.json"
    source.write_bytes(b"{}\n")
    inherited = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    duplicate = -1
    try:
        duplicate = recovery._regular_fd_argument(
            f"/proc/self/fd/{inherited}", name="authority"
        )
        original_info = os.fstat(inherited)
        duplicate_info = os.fstat(duplicate)
        assert (duplicate_info.st_dev, duplicate_info.st_ino) == (
            original_info.st_dev,
            original_info.st_ino,
        )
        os.close(inherited)
        inherited = -1
        assert os.read(duplicate, 3) == b"{}\n"
    finally:
        if duplicate >= 0:
            os.close(duplicate)
        if inherited >= 0:
            os.close(inherited)


def test_regular_fd_argument_rejects_directory_and_standard_stream(
    tmp_path: Path,
) -> None:
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(recovery.OpenAxesNativeV2RecoveryError):
            recovery._regular_fd_argument(
                f"/proc/self/fd/{directory_fd}", name="authority"
            )
        with pytest.raises(recovery.OpenAxesNativeV2RecoveryError):
            recovery._regular_fd_argument(
                "/proc/self/fd/0", name="authority"
            )
    finally:
        os.close(directory_fd)


def _direct_call_names(function: object) -> set[str]:
    tree = ast.parse(inspect.getsource(function))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            parts = [target.attr]
            value = target.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                parts.append(value.id)
            names.add(".".join(reversed(parts)))
    return names


def test_validate_cli_is_wired_to_all_three_validation_layers() -> None:
    main_calls = _direct_call_names(recovery.main)
    published_calls = _direct_call_names(
        recovery.validate_published_compound_recovery
    )
    terminal_calls = _direct_call_names(recovery.validate_compound_terminal)
    assert "validate_published_compound_recovery" in main_calls
    assert "validate_compound_terminal" in published_calls
    assert "frozen.validate_openaxes_external_native_v2_run" in published_calls
    assert "validate_staged_terminal_files" in terminal_calls
    assert (
        "recovery_authority.validate_compound_terminal_receipt" in terminal_calls
    )


@pytest.mark.parametrize("failed_layer", ["compound", "frozen_full"])
def test_validate_published_propagates_every_validation_failure(
    failed_layer: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {"original_science_authority": {}}
    monkeypatch.setattr(
        recovery,
        "_load_external_authorities",
        lambda *_args, **_kwargs: (source, {}, {}, {}),
    )
    monkeypatch.setattr(
        recovery,
        "_try_read_output",
        lambda *_args, **_kwargs: recovery._canonical_bytes({"terminal": True}),
    )
    if failed_layer == "compound":
        monkeypatch.setattr(
            recovery,
            "validate_compound_terminal",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                recovery.OpenAxesNativeV2RecoveryError(
                    "embedded semantics or authority rejected"
                )
            ),
        )
        monkeypatch.setattr(
            recovery.frozen,
            "validate_openaxes_external_native_v2_run",
            lambda *_args, **_kwargs: pytest.fail(
                "frozen validator ran after compound rejection"
            ),
        )
    else:
        monkeypatch.setattr(
            recovery,
            "validate_compound_terminal",
            lambda *_args, **_kwargs: {"valid": True},
        )
        monkeypatch.setattr(
            recovery.frozen,
            "validate_openaxes_external_native_v2_run",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                frozen.OpenAxesExternalNativeV2Error("frozen full rejected")
            ),
        )
    with pytest.raises(recovery.OpenAxesNativeV2RecoveryError):
        recovery.validate_published_compound_recovery(
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            recovery_source_evidence_commit="f" * 40,
        )

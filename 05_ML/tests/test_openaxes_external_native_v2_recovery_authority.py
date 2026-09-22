from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from imu_vive_learning import openaxes_external_native_v2_recovery_authority as authority


FIXED_TIME = "2026-08-26T00:00:00Z"


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _payload(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["payload_sha256"] = authority.canonical_sha256(result)
    return result


def _git(repo: Path, *args: str) -> str:
    environment = {
        "PATH": "/usr/bin:/bin",
        "LC_ALL": "C",
        "TZ": "UTC",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *args],
        env=environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _reference(path: str, raw: bytes) -> dict[str, object]:
    return {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _fake_original(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    source_raw = b"source\n"
    session_raw = b"session\n"
    index_raw = b"index\n"
    source_hash = hashlib.sha256(source_raw).hexdigest()
    session_hash = hashlib.sha256(session_raw).hexdigest()
    index_hash = hashlib.sha256(index_raw).hexdigest()
    config_hash = "1" * 64
    source_set_hash = "2" * 64
    controller_hash = "3" * 64
    monkeypatch.setattr(authority, "FIXED_CONFIG_SHA256", config_hash)
    monkeypatch.setattr(authority, "FIXED_EXECUTION_SOURCE_SET_SHA256", source_set_hash)
    monkeypatch.setattr(authority, "FIXED_CONTROLLER_SOURCE_SHA256", controller_hash)
    monkeypatch.setattr(authority, "FIXED_SOURCE_SET_RAW_SHA256", source_hash)
    monkeypatch.setattr(authority, "FIXED_EXECUTION_SESSION_RAW_SHA256", session_hash)
    monkeypatch.setattr(authority, "FIXED_TASK_INDEX_RAW_SHA256", index_hash)
    return {
        "run_id": authority.RUN_ID,
        "canonical_config_sha256": config_hash,
        "execution_source_set_sha256": source_set_hash,
        "controller_source_sha256": controller_hash,
        "source_set_reference": _reference("runtime/source_set.json", source_raw),
        "execution_session_reference": _reference(
            "runtime/execution_session.json", session_raw
        ),
        "task_index_reference": _reference("task_index.json", index_raw),
        "result_completed_reference_set_sha256": "4" * 64,
        "result_completed_count": 126,
    }


def _fake_prestate(monkeypatch: pytest.MonkeyPatch, *, tree_hash: str = "5" * 64) -> dict[str, object]:
    monkeypatch.setattr(authority, "FIXED_TREE_SHA256", tree_hash)
    monkeypatch.setattr(authority, "FIXED_REGULAR_FILE_COUNT", 17)
    monkeypatch.setattr(authority, "FIXED_REGULAR_FILE_BYTES", 41)
    return {
        "canonical_tree_sha256": tree_hash,
        "regular_file_count": 17,
        "regular_file_bytes": 41,
        "top_level_members": sorted(authority.FAILED_TOP_LEVEL_MEMBERS),
        "reports_members": [],
        "development_manifest_absent": True,
        "complete_absent": True,
        "result_completed_count": 126,
    }


def _source_audit(
    monkeypatch: pytest.MonkeyPatch,
    original: dict[str, object],
    prestate: dict[str, object],
) -> dict[str, object]:
    paths = (
        authority.RECOVERY_ERRATUM_PATH,
        authority.SIGNED_DIAGNOSTIC_RECEIPT_PATH,
        (
            "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/"
            "openaxes_external_native_v2_recovery_authority.py"
        ),
    )
    monkeypatch.setattr(authority, "RECOVERY_SOURCE_CLOSURE", paths)
    diagnostic_raw = b"diagnostic authority\n"
    monkeypatch.setattr(authority, "FIXED_DIAGNOSTIC_BYTES", len(diagnostic_raw))
    monkeypatch.setattr(
        authority, "FIXED_DIAGNOSTIC_SHA256", hashlib.sha256(diagnostic_raw).hexdigest()
    )
    raw_by_path = {
        paths[0]: b"recovery erratum\n",
        paths[1]: diagnostic_raw,
        paths[2]: b"authority source\n",
    }
    rows = [
        {
            "path": path,
            "bytes": len(raw_by_path[path]),
            "sha256": hashlib.sha256(raw_by_path[path]).hexdigest(),
            "git_blob_oid": hashlib.sha1(
                b"blob "
                + str(len(raw_by_path[path])).encode("ascii")
                + b"\0"
                + raw_by_path[path]
            ).hexdigest(),
            "git_mode": "100644",
        }
        for path in paths
    ]
    value: dict[str, object] = {
        "contract": authority.SOURCE_AUDIT_CONTRACT,
        "schema_version": 1,
        "run_id": authority.RUN_ID,
        "created_at_utc": FIXED_TIME,
        "repository": {
            "object_format": "sha1",
            "project_directory": authority.PROJECT_DIRECTORY,
            "implementation_git_commit": "a" * 40,
            "historical_run004_erratum_git_commit": authority.HISTORICAL_RUN004_ERRATUM_GIT_COMMIT,
            "preissued_recovery_erratum_git_commit": authority.PREISSUED_RECOVERY_ERRATUM_GIT_COMMIT,
            "signed_diagnostic_receipt_git_commit": authority.SIGNED_DIAGNOSTIC_RECEIPT_GIT_COMMIT,
            "source_audit_receipt_path": authority.SOURCE_AUDIT_RECEIPT_PATH,
            "source_audit_evidence_commit_required": True,
            "implementation_commit_must_be_ancestor_of_evidence_commit": True,
            "preissued_authorities_must_be_ancestors_of_implementation_commit": True,
        },
        "source_closure": {
            "path_order": list(paths),
            "files": {row["path"]: row for row in rows},
            "file_count": len(rows),
            "set_sha256": authority.canonical_sha256(rows),
        },
        "preissued_erratum_reference": {
            key: rows[0][key] for key in ("path", "bytes", "sha256")
        },
        "diagnostic_receipt_reference": {
            key: rows[1][key] for key in ("path", "bytes", "sha256")
        },
        "original_science_authority": original,
        "failed_prestate": prestate,
        "assertions": dict(authority.SOURCE_AUDIT_ASSERTIONS),
    }
    value["payload_sha256"] = authority.canonical_sha256(value)
    return value


def test_run_tree_projection_hashes_binary_bytes_without_json_decode(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write(run / "results" / "task_000" / "outer_predictions.parquet", b"\x00\xffnot-json")
    _write(run / "runtime" / "opaque.bin", b"opaque")
    descriptor = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _directories, rows = authority._scan_tree(descriptor)
    finally:
        os.close(descriptor)
    assert [row["path"] for row in rows] == [
        "results/task_000/outer_predictions.parquet",
        "runtime/opaque.bin",
    ]
    expected = hashlib.sha256(
        b"".join(
            f'{row["path"]}\t{row["bytes"]}\t{row["sha256"]}\n'.encode()
            for row in rows
        )
    ).hexdigest()
    assert authority._tree_sha256(rows) == expected


def test_frozen_run003_outcome_neutral_projection_when_present() -> None:
    project = Path(__file__).resolve().parents[1]
    run = project / authority.RUN_RELATIVE_PATH
    if not run.is_dir():
        pytest.skip("ignored RUN003 artifact is not present")
    descriptor = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        projection = authority.project_failed_prestate(descriptor)
        original = authority.project_original_science_authority(descriptor)
    finally:
        os.close(descriptor)
    authority._validate_failed_prestate(projection, fixed=True)
    authority._validate_original_science_authority(original)
    assert original["result_completed_count"] == 126


def test_recovery_closure_includes_both_authorities_public_api_docs_and_locks() -> None:
    paths = set(authority.RECOVERY_SOURCE_CLOSURE)
    required_suffixes = {
        "docs/OPENAXES_EXTERNAL_NATIVE_V2_RUN003_RECOVERY_ERRATUM.md",
        "docs/OPENAXES_EXTERNAL_NATIVE_V2_RUN004_ERRATUM.md",
        "docs/OPENAXES_EXTERNAL_NATIVE_V2_SIGNED_DIAGNOSTIC_RECEIPT.md",
        "src/imu_vive_learning/openaxes_external_native_v2_recovery.py",
        "tests/test_openaxes_external_native_v2_recovery.py",
        "src/imu_vive_learning/openaxes_external_native_v2_recovery_authority.py",
        "tests/test_openaxes_external_native_v2_recovery_authority.py",
        "src/imu_vive_learning/openaxes_external_native_v2_experiment.py",
        "src/imu_vive_learning/openaxes_external_native_v2_training.py",
        "src/imu_vive_learning/openaxes_external_v1.py",
        "src/imu_vive_learning/openaxes_external_v1_config.py",
        "src/imu_vive_learning/openaxes_external_v1_cli.py",
        "configs/openaxes_external_v1.yaml",
        "requirements-eval.lock",
        "requirements-gpu.lock",
    }
    assert {
        path.removeprefix(f"{authority.PROJECT_DIRECTORY}/") for path in paths
    }.issuperset(required_suffixes)
    assert all(path.startswith(f"{authority.PROJECT_DIRECTORY}/") for path in paths)


def test_source_audit_closed_schema_and_outcome_neutral_assertions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _fake_original(monkeypatch)
    prestate = _fake_prestate(monkeypatch)
    receipt = _source_audit(monkeypatch, original, prestate)
    assert authority.validate_source_audit_receipt(receipt) == receipt

    attacks: list[dict[str, object]] = []
    extra = deepcopy(receipt)
    extra["outcome"] = "forbidden"
    attacks.append(extra)
    reordered = deepcopy(receipt)
    reordered["source_closure"]["path_order"] = list(  # type: ignore[index]
        reversed(reordered["source_closure"]["path_order"])  # type: ignore[index]
    )
    reordered["payload_sha256"] = authority.canonical_sha256(
        {key: value for key, value in reordered.items() if key != "payload_sha256"}
    )
    attacks.append(reordered)
    enabled = deepcopy(receipt)
    enabled["assertions"]["cuda_allowed"] = True  # type: ignore[index]
    enabled["payload_sha256"] = authority.canonical_sha256(
        {key: value for key, value in enabled.items() if key != "payload_sha256"}
    )
    attacks.append(enabled)
    for attacked in attacks:
        with pytest.raises(authority.OpenAxesNativeV2RecoveryAuthorityError):
            authority.validate_source_audit_receipt(attacked)


def test_source_audit_git_blob_and_ancestry_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    project = repo / authority.PROJECT_DIRECTORY
    project.mkdir(parents=True)
    original = _fake_original(monkeypatch)
    prestate = _fake_prestate(monkeypatch)
    paths = (
        authority.RECOVERY_ERRATUM_PATH,
        authority.HISTORICAL_RUN004_ERRATUM_PATH,
        authority.SIGNED_DIAGNOSTIC_RECEIPT_PATH,
        (
            "09_SO101_IMU_VIVE_LEARNING/src/imu_vive_learning/"
            "openaxes_external_native_v2_recovery_authority.py"
        ),
    )
    monkeypatch.setattr(authority, "RECOVERY_SOURCE_CLOSURE", paths)
    diagnostic_raw = b"diagnostic authority\n"
    monkeypatch.setattr(authority, "FIXED_DIAGNOSTIC_BYTES", len(diagnostic_raw))
    monkeypatch.setattr(
        authority, "FIXED_DIAGNOSTIC_SHA256", hashlib.sha256(diagnostic_raw).hexdigest()
    )
    _write(project / "docs/OPENAXES_EXTERNAL_NATIVE_V2_RUN003_RECOVERY_ERRATUM.md", b"erratum\n")
    _write(project / "docs/OPENAXES_EXTERNAL_NATIVE_V2_RUN004_ERRATUM.md", b"historical\n")
    _write(project / "docs/OPENAXES_EXTERNAL_NATIVE_V2_SIGNED_DIAGNOSTIC_RECEIPT.md", diagnostic_raw)
    _write(
        project
        / "src/imu_vive_learning/openaxes_external_native_v2_recovery_authority.py",
        b"authority\n",
    )
    run = project / "run"
    run.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "implementation")
    implementation = _git(repo, "rev-parse", "HEAD")
    monkeypatch.setattr(authority, "HISTORICAL_RUN004_ERRATUM_GIT_COMMIT", implementation)
    monkeypatch.setattr(authority, "PREISSUED_RECOVERY_ERRATUM_GIT_COMMIT", implementation)
    monkeypatch.setattr(authority, "SIGNED_DIAGNOSTIC_RECEIPT_GIT_COMMIT", implementation)
    monkeypatch.setattr(authority, "project_failed_prestate", lambda _fd: deepcopy(prestate))
    monkeypatch.setattr(
        authority, "project_original_science_authority", lambda _fd: deepcopy(original)
    )
    repo_fd = os.open(repo, os.O_RDONLY | os.O_DIRECTORY)
    project_fd = os.open(project, os.O_RDONLY | os.O_DIRECTORY)
    run_fd = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    try:
        receipt = authority.build_source_audit_receipt(
            repo_fd,
            project_fd,
            run_fd,
            implementation_git_commit=implementation,
            created_at_utc=FIXED_TIME,
        )
        raw = authority.canonical_json_bytes(receipt, final_newline=True)
        _write(
            project / "receipts/openaxes_external_native_v2_compound_recovery_source_audit.json",
            raw,
        )
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "source audit evidence")
        evidence = _git(repo, "rev-parse", "HEAD")
        assert authority.validate_source_audit_live(
            receipt,
            receipt_raw=raw,
            repo_fd=repo_fd,
            project_fd=project_fd,
            run_fd=run_fd,
            evidence_git_commit=evidence,
        ) == receipt

        attacked_raw = raw.replace(b'"schema_version":1', b'"schema_version":2')
        with pytest.raises(authority.OpenAxesNativeV2RecoveryAuthorityError):
            authority.validate_source_audit_live(
                receipt,
                receipt_raw=attacked_raw,
                repo_fd=repo_fd,
                project_fd=project_fd,
                run_fd=run_fd,
                evidence_git_commit=evidence,
            )
    finally:
        os.close(run_fd)
        os.close(project_fd)
        os.close(repo_fd)


def _make_terminal_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    dict[str, object],
    bytes,
    dict[str, object],
    bytes,
    dict[str, object],
    int,
    int,
]:
    original = _fake_original(monkeypatch)
    run = tmp_path / "run"
    stage = tmp_path / "stage"
    for directory in ("parents", "quarantine", "reports", "results", "runtime", "tasks"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    _write(run / "PREPARED.json", b"p")
    _write(run / "prepared_manifest.json", b"m")
    _write(run / "task_index.json", b"i")
    run_fd = os.open(run, os.O_RDONLY | os.O_DIRECTORY)
    _dirs, base_rows = authority._scan_tree(run_fd)
    tree_hash = authority._tree_sha256(base_rows)
    monkeypatch.setattr(authority, "FIXED_TREE_SHA256", tree_hash)
    monkeypatch.setattr(authority, "FIXED_REGULAR_FILE_COUNT", len(base_rows))
    monkeypatch.setattr(
        authority, "FIXED_REGULAR_FILE_BYTES", sum(row["bytes"] for row in base_rows)
    )
    prestate = {
        "canonical_tree_sha256": tree_hash,
        "regular_file_count": len(base_rows),
        "regular_file_bytes": sum(row["bytes"] for row in base_rows),
        "top_level_members": sorted(authority.FAILED_TOP_LEVEL_MEMBERS),
        "reports_members": [],
        "development_manifest_absent": True,
        "complete_absent": True,
        "result_completed_count": 126,
    }
    source = _source_audit(monkeypatch, original, prestate)
    source_raw = authority.canonical_json_bytes(source, final_newline=True)
    output_refs: list[dict[str, object]] = []
    for ordinal, path in enumerate(authority.STAGED_FILE_ORDER):
        raw = authority.canonical_json_bytes({"ordinal": ordinal})
        _write(stage / path, raw)
        _write(run / path, raw)
        output_refs.append(_reference(path, raw))
    stage_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
    seal = _payload(
        {
            "contract": authority.STAGE_SEAL_CONTRACT,
            "schema_version": 1,
            "run_id": authority.RUN_ID,
            "created_at_utc": FIXED_TIME,
            "erratum_reference": source["preissued_erratum_reference"],
            "diagnostic_receipt_reference": source["diagnostic_receipt_reference"],
            "recovery_source_audit_reference": _reference(
                authority.SOURCE_AUDIT_RECEIPT_PATH, source_raw
            ),
            "original_science_authority": original,
            "failed_prestate": prestate,
            "staged_file_order": list(authority.STAGED_FILE_ORDER),
            "staged_file_references": output_refs,
            "staged_file_set_sha256": authority.canonical_sha256(output_refs),
            "signed_report_status": "complete",
            "action_accounting": dict(authority.ACTION_ACCOUNTING),
        }
    )
    seal_raw = authority.canonical_json_bytes(seal, final_newline=True)
    poststate = authority._poststate_projection(run_fd, output_refs)
    terminal = _payload(
        {
            "contract": authority.TERMINAL_CONTRACT,
            "schema_version": 1,
            "run_id": authority.RUN_ID,
            "created_at_utc": FIXED_TIME,
            "erratum_reference": source["preissued_erratum_reference"],
            "diagnostic_receipt_reference": source["diagnostic_receipt_reference"],
            "recovery_source_audit_reference": _reference(
                authority.SOURCE_AUDIT_RECEIPT_PATH, source_raw
            ),
            "original_science_authority": original,
            "failed_prestate": prestate,
            "stage_seal_reference": _reference("STAGE_SEAL.json", seal_raw),
            "staged_file_references": output_refs,
            "output_file_references": output_refs,
            "poststate": poststate,
            "existing_prestate_unchanged": True,
            "signed_report_status": "complete",
            "action_accounting": dict(authority.ACTION_ACCOUNTING),
            "semantic_validation": {
                key: True for key in authority.SEMANTIC_VALIDATION_KEYS
            },
            "compound_terminal_required_for_publication": True,
        }
    )
    return source, source_raw, seal, seal_raw, terminal, run_fd, stage_fd


def test_compound_terminal_binds_separate_source_and_exact_seven_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_raw, seal, seal_raw, terminal, run_fd, stage_fd = _make_terminal_fixture(
        tmp_path, monkeypatch
    )
    try:
        assert authority.validate_compound_terminal_receipt(
            terminal,
            source_audit=source,
            source_audit_raw=source_raw,
            stage_seal=seal,
            stage_seal_raw=seal_raw,
            run_fd=run_fd,
            stage_fd=stage_fd,
        ) == terminal

        attacked = deepcopy(terminal)
        attacked["original_science_authority"]["controller_source_sha256"] = "f" * 64  # type: ignore[index]
        attacked["payload_sha256"] = authority.canonical_sha256(
            {key: value for key, value in attacked.items() if key != "payload_sha256"}
        )
        with pytest.raises(authority.OpenAxesNativeV2RecoveryAuthorityError):
            authority.validate_compound_terminal_receipt(
                attacked,
                source_audit=source,
                source_audit_raw=source_raw,
                stage_seal=seal,
                stage_seal_raw=seal_raw,
                run_fd=run_fd,
                stage_fd=stage_fd,
            )
    finally:
        os.close(stage_fd)
        os.close(run_fd)


def test_terminal_rejects_output_mutation_and_hidden_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_raw, seal, seal_raw, terminal, run_fd, stage_fd = _make_terminal_fixture(
        tmp_path, monkeypatch
    )
    try:
        output = tmp_path / "run" / authority.STAGED_FILE_ORDER[0]
        output.write_bytes(b"changed")
        with pytest.raises(authority.OpenAxesNativeV2RecoveryAuthorityError):
            authority.validate_compound_terminal_receipt(
                terminal,
                source_audit=source,
                source_audit_raw=source_raw,
                stage_seal=seal,
                stage_seal_raw=seal_raw,
                run_fd=run_fd,
                stage_fd=stage_fd,
            )

        hidden = deepcopy(seal)
        hidden["action_accounting"]["recovery_model_forwards"] = 1  # type: ignore[index]
        hidden["payload_sha256"] = authority.canonical_sha256(
            {key: value for key, value in hidden.items() if key != "payload_sha256"}
        )
        with pytest.raises(authority.OpenAxesNativeV2RecoveryAuthorityError):
            authority.validate_stage_seal(hidden, source_audit=source)
    finally:
        os.close(stage_fd)
        os.close(run_fd)


def test_cli_paths_are_fd_or_exact_repository_relative() -> None:
    with pytest.raises(authority.OpenAxesNativeV2RecoveryAuthorityError):
        authority._relative_path("/tmp/receipt.json", name="receipt")
    with pytest.raises(authority.OpenAxesNativeV2RecoveryAuthorityError):
        authority._relative_path("receipts/../receipt.json", name="receipt")
    with pytest.raises(authority.OpenAxesNativeV2RecoveryAuthorityError):
        authority._project_local_path("receipts/receipt.json", name="receipt")
    assert authority._project_local_path(
        authority.SOURCE_AUDIT_RECEIPT_PATH, name="receipt"
    ) == "receipts/openaxes_external_native_v2_compound_recovery_source_audit.json"


def test_authority_module_has_no_scientific_or_external_runtime_import() -> None:
    source = Path(authority.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "import numpy",
        "import pandas",
        "import torch",
        "import socket",
        "import urllib",
        "import requests",
        "nvidia-smi",
    ):
        assert forbidden not in source.lower()

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from imu_vive_learning import openaxes_external_native_v2_recovery_salvage as salvage


def _write(root: Path, relative: str, raw: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    path.chmod(0o600)


def _synthetic_stage_bytes() -> tuple[dict[str, bytes], dict[str, bytes]]:
    first = {
        path: salvage._canonical_bytes({"synthetic": path})
        for path in salvage.FIRST_FOUR
    }
    summary_base = {
        "contract": "synthetic",
        "run_id": salvage.RUN_ID,
        "task_count": 126,
        "candidate_count": 3,
        "fold_count": 1,
        "seed_count": 1,
        "optimizer_updates": 0,
        "all_model_forward_calls": 0,
        "outer_label_reads": 0,
        "task_evaluator_calls": 0,
        "signed_truth_reconstruction_calls": 1,
        "signed_report_status": "complete",
        "report_references": {
            Path(path).name: salvage._reference(path, raw)
            for path, raw in first.items()
        },
        "recovery_source_audit_reference": {"path": "source", "bytes": 1, "sha256": "0" * 64},
        "diagnostic_receipt_reference": {"path": "diagnostic", "bytes": 1, "sha256": "1" * 64},
        "action_accounting": {},
        "compound_terminal_required_for_publication": True,
    }
    corrected_summary = salvage._payload(summary_base)
    corrected_summary_raw = salvage._canonical_bytes(corrected_summary)
    failed_summary = deepcopy(corrected_summary)
    failed_summary["report_references"]["development_summary.json"] = salvage._reference(
        salvage.SUMMARY_PATH, corrected_summary_raw
    )
    failed = {
        **first,
        salvage.SUMMARY_PATH: salvage._canonical_bytes(failed_summary),
        salvage.OUTPUT_PATHS[5]: salvage._canonical_bytes({"development": True}),
        salvage.OUTPUT_PATHS[6]: salvage._canonical_bytes({"complete": True}),
    }
    corrected = dict(failed)
    corrected[salvage.SUMMARY_PATH] = corrected_summary_raw
    return failed, corrected


def _refs(raw_by_path: dict[str, bytes]) -> tuple[dict[str, object], ...]:
    return tuple(
        salvage._reference(path, raw_by_path[path]) for path in salvage.OUTPUT_PATHS
    )


def _install_synthetic_authority(monkeypatch: pytest.MonkeyPatch, failed: dict[str, bytes], corrected: dict[str, bytes]) -> None:
    failed_refs = _refs(failed)
    corrected_refs = _refs(corrected)
    monkeypatch.setattr(salvage, "FAILED_REFERENCES", failed_refs)
    monkeypatch.setattr(salvage, "CORRECTED_REFERENCES", corrected_refs)
    monkeypatch.setattr(salvage, "FAILED_REFERENCE_SET_SHA256", salvage._canonical_hash(list(failed_refs)))
    monkeypatch.setattr(salvage, "CORRECTED_REFERENCE_SET_SHA256", salvage._canonical_hash(list(corrected_refs)))
    monkeypatch.setattr(salvage.prior, "_validate_zero_report", lambda value: value)
    monkeypatch.setattr(salvage.prior, "_validate_gate_report", lambda value: value)
    monkeypatch.setattr(salvage.prior, "_validate_stress_report", lambda value: value)
    monkeypatch.setattr(salvage.prior, "validate_signed_shift_recovery_report", lambda value: value)


def _make_stage(root: Path, raw_by_path: dict[str, bytes], *, seal: bytes | None = None) -> int:
    root.mkdir(mode=0o700)
    for path, raw in raw_by_path.items():
        _write(root, path, raw)
    if seal is not None:
        _write(root, salvage.prior.STAGE_SEAL_PATH, seal)
    return os.open(root, os.O_RDONLY | os.O_DIRECTORY)


def test_exact_one_key_salvage_preserves_six_raw_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failed, corrected = _synthetic_stage_bytes()
    _install_synthetic_authority(monkeypatch, failed, corrected)
    stage_fd = _make_stage(tmp_path / "stage", failed)
    try:
        observed, proof = salvage._correct_failed_outputs(stage_fd)
    finally:
        os.close(stage_fd)
    assert observed == corrected
    assert proof["failed_references"] == list(_refs(failed))
    assert proof["corrected_references"] == list(_refs(corrected))
    assert all(observed[path] == failed[path] for path in (*salvage.FIRST_FOUR, *salvage.OUTPUT_PATHS[5:]))


@pytest.mark.parametrize("mutation", ["second_summary_key", "changed_sixth_byte", "surplus_stage_file"])
def test_salvage_rejects_any_second_defect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    failed, corrected = _synthetic_stage_bytes()
    if mutation == "second_summary_key":
        value = json.loads(failed[salvage.SUMMARY_PATH])
        value["unexpected"] = True
        failed[salvage.SUMMARY_PATH] = salvage._canonical_bytes(value)
    elif mutation == "changed_sixth_byte":
        failed[salvage.OUTPUT_PATHS[5]] += b" "
    _install_synthetic_authority(monkeypatch, failed, corrected)
    stage_root = tmp_path / "stage"
    stage_fd = _make_stage(stage_root, failed)
    if mutation == "surplus_stage_file":
        _write(stage_root, "extra.json", b"{}\n")
    try:
        if mutation == "surplus_stage_file":
            with pytest.raises(salvage.OpenAxesNativeV2RecoverySalvageError):
                salvage._stage_projection(stage_fd, require_failed_path=False)
        else:
            with pytest.raises(salvage.OpenAxesNativeV2RecoverySalvageError):
                salvage._correct_failed_outputs(stage_fd)
    finally:
        os.close(stage_fd)


def test_stage_projection_rejects_hardlink_and_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failed, corrected = _synthetic_stage_bytes()
    _install_synthetic_authority(monkeypatch, failed, corrected)
    root = tmp_path / "stage"
    stage_fd = _make_stage(root, failed)
    victim = root / salvage.OUTPUT_PATHS[6]
    saved = victim.read_bytes()
    victim.unlink()
    os.link(root / salvage.OUTPUT_PATHS[5], victim)
    try:
        with pytest.raises(salvage.OpenAxesNativeV2RecoverySalvageError):
            salvage._stage_projection(stage_fd, require_failed_path=False)
    finally:
        os.close(stage_fd)
    victim.unlink()
    victim.symlink_to(root / salvage.OUTPUT_PATHS[5])
    stage_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(salvage.OpenAxesNativeV2RecoverySalvageError):
            salvage._stage_projection(stage_fd, require_failed_path=False)
    finally:
        os.close(stage_fd)
    assert saved


def test_external_projection_closes_outer_lock_stage_and_empty_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    failed, corrected = _synthetic_stage_bytes()
    _install_synthetic_authority(monkeypatch, failed, corrected)
    outer = tmp_path / "outer"
    outer.mkdir(mode=0o700)
    stage = outer / "stage"
    authority = outer / "authority"
    stage_fd = _make_stage(stage, failed)
    authority.mkdir(mode=0o700)
    (outer / salvage.LIFECYCLE_LOCK_PATH).write_bytes(b"synthetic lock\n")
    (outer / salvage.LIFECYCLE_LOCK_PATH).chmod(0o600)
    outer_fd = os.open(outer, os.O_RDONLY | os.O_DIRECTORY)
    authority_fd = os.open(authority, os.O_RDONLY | os.O_DIRECTORY)
    monkeypatch.setattr(salvage, "FAILED_OUTER_ROOT", str(outer))
    monkeypatch.setattr(salvage, "FAILED_STAGE_ROOT", str(stage))
    monkeypatch.setattr(salvage, "ORIGINAL_AUTHORITY_ROOT", str(authority))
    try:
        projection = salvage._external_projection(outer_fd, stage_fd, authority_fd)
        assert salvage._validate_external_projection_value(projection) == projection
        _write(authority, "unexpected.json", b"{}\n")
        with pytest.raises(salvage.OpenAxesNativeV2RecoverySalvageError):
            salvage._external_projection(outer_fd, stage_fd, authority_fd)
    finally:
        os.close(authority_fd)
        os.close(stage_fd)
        os.close(outer_fd)


def _synthetic_external(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed: dict[str, bytes],
) -> tuple[dict[str, object], int, int, int, Path, Path]:
    outer = tmp_path / "outer"
    outer.mkdir(mode=0o700)
    stage = outer / "stage"
    authority = outer / "authority"
    stage_fd = _make_stage(stage, failed)
    authority.mkdir(mode=0o700)
    (outer / salvage.LIFECYCLE_LOCK_PATH).write_bytes(b"synthetic lock\n")
    (outer / salvage.LIFECYCLE_LOCK_PATH).chmod(0o600)
    outer_fd = os.open(outer, os.O_RDONLY | os.O_DIRECTORY)
    authority_fd = os.open(authority, os.O_RDONLY | os.O_DIRECTORY)
    monkeypatch.setattr(salvage, "FAILED_OUTER_ROOT", str(outer))
    monkeypatch.setattr(salvage, "FAILED_STAGE_ROOT", str(stage))
    monkeypatch.setattr(salvage, "ORIGINAL_AUTHORITY_ROOT", str(authority))
    projection = salvage._external_projection(outer_fd, stage_fd, authority_fd)
    return projection, outer_fd, stage_fd, authority_fd, stage, authority


def _source_receipt(projection: dict[str, object]) -> dict[str, object]:
    rows = [
        {
            "path": path,
            "bytes": 1,
            "sha256": hashlib.sha256(path.encode()).hexdigest(),
            "git_blob_oid": hashlib.sha1(path.encode()).hexdigest(),
            "git_mode": "100644",
        }
        for path in salvage.SOURCE_CLOSURE
    ]
    return salvage._payload(
        {
            "contract": salvage.SOURCE_AUDIT_CONTRACT,
            "schema_version": 1,
            "run_id": salvage.RUN_ID,
            "created_at_utc": "2026-08-26T01:30:00Z",
            "repository": {
                "object_format": "sha1",
                "project_directory": salvage.PROJECT_DIRECTORY,
                "implementation_git_commit": "a" * 40,
                "preissue_git_commit": salvage.PREISSUE_COMMIT,
                "accounting_supplement_git_commit": salvage.SUPPLEMENT_COMMIT,
                "source_audit_receipt_path": salvage.SOURCE_AUDIT_PATH,
                "implementation_must_be_strict_ancestor_of_evidence": True,
            },
            "source_closure": {
                "path_order": list(salvage.SOURCE_CLOSURE),
                "files": {row["path"]: row for row in rows},
                "file_count": len(rows),
                "set_sha256": salvage._canonical_hash(rows),
            },
            "preissue_reference": {
                "path": salvage.PREISSUE_PATH,
                "bytes": salvage.PREISSUE_BYTES,
                "sha256": salvage.PREISSUE_SHA256,
            },
            "accounting_supplement_reference": {
                "path": salvage.SUPPLEMENT_PATH,
                "bytes": salvage.SUPPLEMENT_BYTES,
                "sha256": salvage.SUPPLEMENT_SHA256,
            },
            "erratum2_reference": {
                "path": salvage.ERRATUM2_PATH,
                "bytes": salvage.ERRATUM2_BYTES,
                "sha256": salvage.ERRATUM2_SHA256,
            },
            "prior_recovery_authority": deepcopy(salvage.PRIOR_AUTHORITY),
            "run003_prestate": dict(salvage.RUN003_PRESTATE),
            "failed_stage_references": list(salvage.FAILED_REFERENCES),
            "external_closed_world": deepcopy(salvage.EXTERNAL_CLOSED_WORLD),
            "publication_validation": dict(salvage.PUBLICATION_VALIDATION),
            "exact_action_accounting": dict(salvage.EXACT_ACTION_ACCOUNTING),
            "failed_external_projection": deepcopy(projection),
            "failed_external_projection_sha256": projection["projection_sha256"],
            "assertions": {
                "run003_mutation_allowed": False,
                "failed_stage_mutation_allowed": False,
                "scientific_recomputation_allowed": False,
                "fresh_corrected_stage_required": True,
                "separate_salvage_terminal_required": True,
            },
        }
    )


@pytest.mark.parametrize("mutation", ["valid", "extra_source_key", "extra_stage_row_key", "forged_projection", "wrong_accounting"])
def test_source_audit_receipt_closes_nested_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    failed, corrected = _synthetic_stage_bytes()
    _install_synthetic_authority(monkeypatch, failed, corrected)
    projection, outer_fd, stage_fd, authority_fd, _stage, _authority = _synthetic_external(
        tmp_path, monkeypatch, failed
    )
    receipt = _source_receipt(projection)
    if mutation == "extra_source_key":
        path = salvage.SOURCE_CLOSURE[0]
        receipt["source_closure"]["files"][path]["extra"] = True
    elif mutation == "extra_stage_row_key":
        receipt["failed_external_projection"]["failed_stage"]["rows"][0]["extra"] = True
    elif mutation == "forged_projection":
        receipt["failed_external_projection"]["lifecycle_lock"]["sha256"] = "f" * 64
    elif mutation == "wrong_accounting":
        receipt["exact_action_accounting"]["salvage_fits"] = 1
    if mutation != "valid":
        unsigned = dict(receipt)
        unsigned.pop("payload_sha256")
        receipt = salvage._payload(unsigned)
    try:
        if mutation == "valid":
            assert salvage.validate_salvage_source_audit_receipt(receipt) == receipt
        else:
            with pytest.raises(salvage.OpenAxesNativeV2RecoverySalvageError):
                salvage.validate_salvage_source_audit_receipt(receipt)
    finally:
        os.close(authority_fd)
        os.close(stage_fd)
        os.close(outer_fd)


@pytest.mark.parametrize("mutation", ["valid", "future_action_claimed", "projection_drift", "seal_drift"])
def test_salvage_terminal_binds_both_projections_and_sealed_stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    failed, corrected = _synthetic_stage_bytes()
    _install_synthetic_authority(monkeypatch, failed, corrected)
    projection, outer_fd, failed_fd, authority_fd, _stage, _authority = _synthetic_external(
        tmp_path, monkeypatch, failed
    )
    corrected_fd = _make_stage(tmp_path / "corrected", corrected, seal=b"synthetic seal\n")
    salvage_authority = tmp_path / "salvage-authority"
    salvage_authority.mkdir(mode=0o700)
    salvage_authority_fd = os.open(
        salvage_authority, os.O_RDONLY | os.O_DIRECTORY
    )
    source_ref = {"path": salvage.SOURCE_AUDIT_PATH, "bytes": 10, "sha256": "2" * 64}
    preissue_ref = {"path": salvage.PREISSUE_PATH, "bytes": salvage.PREISSUE_BYTES, "sha256": salvage.PREISSUE_SHA256}
    supplement_ref = {"path": salvage.SUPPLEMENT_PATH, "bytes": salvage.SUPPLEMENT_BYTES, "sha256": salvage.SUPPLEMENT_SHA256}
    erratum_ref = {"path": salvage.ERRATUM2_PATH, "bytes": salvage.ERRATUM2_BYTES, "sha256": salvage.ERRATUM2_SHA256}
    terminal = salvage._build_salvage_terminal(
        failed_external_pre_projection=projection,
        failed_external_post_projection=projection,
        corrected_stage_fd=corrected_fd,
        salvage_authority_binding=salvage._directory_binding(
            os.fstat(salvage_authority_fd)
        ),
        source_audit_reference=source_ref,
        prior_source_audit_reference=salvage.PRIOR_SOURCE_AUDIT_REFERENCE,
        preissue_reference=preissue_ref,
        supplement_reference=supplement_ref,
        erratum2_reference=erratum_ref,
    )
    if mutation == "future_action_claimed":
        terminal["observed_action_accounting"]["original_compound_publisher_invocations"] = 1
    elif mutation == "projection_drift":
        terminal["failed_stage"]["post_external_projection"]["lifecycle_lock"]["sha256"] = "e" * 64
    elif mutation == "seal_drift":
        terminal["corrected_stage"]["stage_seal_reference"]["sha256"] = "d" * 64
    if mutation != "valid":
        unsigned = dict(terminal)
        unsigned.pop("payload_sha256")
        terminal = salvage._payload(unsigned)
    kwargs = {
        "failed_stage_fd": failed_fd,
        "corrected_stage_fd": corrected_fd,
        "source_audit_reference": source_ref,
        "prior_source_audit_reference": salvage.PRIOR_SOURCE_AUDIT_REFERENCE,
        "preissue_reference": preissue_ref,
        "supplement_reference": supplement_ref,
        "erratum2_reference": erratum_ref,
        "expected_external_projection_sha256": projection["projection_sha256"],
        "salvage_authority_fd": salvage_authority_fd,
    }
    try:
        if mutation == "valid":
            assert salvage.validate_salvage_terminal(terminal, **kwargs) == terminal
        else:
            with pytest.raises(salvage.OpenAxesNativeV2RecoverySalvageError):
                salvage.validate_salvage_terminal(terminal, **kwargs)
    finally:
        os.close(corrected_fd)
        os.close(salvage_authority_fd)
        os.close(authority_fd)
        os.close(failed_fd)
        os.close(outer_fd)


@pytest.mark.parametrize("mutation", ["valid", "mode", "hardlink", "noncanonical"])
def test_terminal_raw_file_is_canonical_private_and_unlinked(
    tmp_path: Path, mutation: str
) -> None:
    authority = tmp_path / "authority"
    authority.mkdir(mode=0o700)
    terminal = salvage._payload(
        {
            "contract": salvage.TERMINAL_CONTRACT,
            "schema_version": 1,
            "run_id": salvage.RUN_ID,
        }
    )
    raw = salvage._canonical_bytes(terminal)
    if mutation == "noncanonical":
        raw = json.dumps(terminal, indent=2, sort_keys=True).encode() + b"\n"
    terminal_path = authority / salvage.TERMINAL_PATH
    terminal_path.write_bytes(raw)
    terminal_path.chmod(0o600 if mutation != "mode" else 0o640)
    if mutation == "hardlink":
        os.link(terminal_path, tmp_path / "terminal-hardlink.json")
    authority_fd = os.open(authority, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if mutation == "valid":
            observed, observed_raw = salvage._read_salvage_terminal(authority_fd)
            assert observed == terminal
            assert observed_raw == raw
        else:
            with pytest.raises(salvage.OpenAxesNativeV2RecoverySalvageError):
                salvage._read_salvage_terminal(authority_fd)
    finally:
        os.close(authority_fd)


@pytest.mark.parametrize(
    "left,right",
    [
        ("corrected stage", "original authority"),
        ("salvage authority", "original authority"),
        ("corrected stage", "salvage authority"),
    ],
)
def test_directory_aliases_fail_before_any_write(
    tmp_path: Path, left: str, right: str
) -> None:
    names = (
        "workspace",
        "RUN003",
        "failed outer",
        "failed stage",
        "corrected stage",
        "original authority",
        "salvage authority",
    )
    fds: dict[str, int] = {}
    try:
        for name in names:
            path = tmp_path / name.replace(" ", "-")
            path.mkdir(mode=0o700)
            fds[name] = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        os.close(fds[right])
        fds[right] = os.dup(fds[left])
        with pytest.raises(
            salvage.OpenAxesNativeV2RecoverySalvageError,
            match="directory descriptors alias",
        ):
            salvage._require_distinct_directories(fds)
        assert all(not os.listdir(tmp_path / name.replace(" ", "-")) for name in names)
    finally:
        for fd in fds.values():
            os.close(fd)


def test_original_validator_main_is_invoked_once_with_retained_fds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(salvage.prior, "main", lambda argv: calls.append(list(argv)) or 0)
    salvage._run_original_validation_main_once(
        7, 8, 9, 10, 11, 12, 13,
        original_source_evidence_commit="a" * 40,
    )
    assert len(calls) == 1
    assert calls[0][0] == "validate"
    assert [item for item in calls[0] if item.startswith("/proc/self/fd/")] == [
        f"/proc/self/fd/{fd}" for fd in range(7, 14)
    ]
    assert calls[0][-1] == salvage.PRIOR_EVIDENCE_COMMIT


def test_validate_cli_is_silent_and_runs_each_layer_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    directory_fds: list[int] = []
    regular_fds: list[int] = []
    try:
        for ordinal in range(7):
            path = tmp_path / f"dir{ordinal}"
            path.mkdir(mode=0o700)
            directory_fds.append(os.open(path, os.O_RDONLY | os.O_DIRECTORY))
        for ordinal in range(4):
            path = tmp_path / f"authority{ordinal}.json"
            path.write_bytes(b"{}\n")
            path.chmod(0o600)
            regular_fds.append(os.open(path, os.O_RDONLY))
        original_calls: list[tuple[object, ...]] = []
        salvage_calls: list[tuple[object, ...]] = []
        monkeypatch.setattr(
            salvage,
            "_run_original_validation_main_once",
            lambda *args, **kwargs: original_calls.append((*args, kwargs)),
        )
        monkeypatch.setattr(
            salvage,
            "validate_published_salvage",
            lambda *args, **kwargs: salvage_calls.append((*args, kwargs)) or {},
        )
        argv = ["validate"]
        for name, fd in zip(
            (
                "workspace-root",
                "run-dir",
                "failed-outer-dir",
                "failed-stage-dir",
                "corrected-stage-dir",
                "original-authority-dir",
                "salvage-authority-dir",
            ),
            directory_fds,
            strict=True,
        ):
            argv.extend((f"--{name}", f"/proc/self/fd/{fd}"))
        for name, fd in zip(
            (
                "prior-source-audit-fd",
                "prior-erratum-fd",
                "diagnostic-fd",
                "salvage-source-audit-fd",
            ),
            regular_fds,
            strict=True,
        ):
            argv.extend((f"--{name}", f"/proc/self/fd/{fd}"))
        argv.extend(
            (
                "--original-source-evidence-commit",
                "a" * 40,
                "--salvage-source-evidence-commit",
                "b" * 40,
            )
        )
        assert salvage.main(argv) == 0
        assert len(original_calls) == 1
        assert len(salvage_calls) == 1
        assert salvage_calls[0][-1]["original_already_validated"] is True
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
    finally:
        for fd in (*regular_fds, *directory_fds):
            os.close(fd)


def test_salvage_source_has_no_scientific_action_calls() -> None:
    source = Path(salvage.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_names = {
        "aggregate_signed_shift_once",
        "aggregate_zero_shift_development",
        "reconstruct_openaxes_signed_shift_truth_v1",
        "fit",
        "predict",
        "evaluate",
        "load_outer_features",
        "load_outer_labels",
    }
    observed: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            observed.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            observed.add(node.func.attr)
    assert observed.isdisjoint(forbidden_names)
    assert "validate_published_compound_recovery" not in observed
    assert sum(name == "main" for name in observed) == 1


def test_accounting_is_exact_supplement_partition() -> None:
    assert len(salvage.EXACT_ACTION_ACCOUNTING) == 23
    assert set(salvage.OBSERVED_ACTION_ACCOUNTING) == set(salvage.EXACT_ACTION_ACCOUNTING)
    assert set(salvage.AUTHORIZED_REMAINING_ACTION_ACCOUNTING) == set(salvage.EXACT_ACTION_ACCOUNTING)
    assert {
        key: salvage.OBSERVED_ACTION_ACCOUNTING[key]
        + salvage.AUTHORIZED_REMAINING_ACTION_ACCOUNTING[key]
        for key in salvage.EXACT_ACTION_ACCOUNTING
    } == salvage.EXACT_ACTION_ACCOUNTING
    assert salvage.AUTHORIZED_REMAINING_ACTION_ACCOUNTING == {
        key: (
            7
            if key == "run003_ordered_o_excl_file_copies"
            else 1
            if key in {"original_compound_publisher_invocations", "original_compound_terminal_writes"}
            else 0
        )
        for key in salvage.EXACT_ACTION_ACCOUNTING
    }


def test_write_output_requires_empty_private_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        salvage._write_output(fd, b"{}\n")
        assert os.fstat(fd).st_size == 3
        with pytest.raises(salvage.OpenAxesNativeV2RecoverySalvageError):
            salvage._write_output(fd, b"{}\n")
    finally:
        os.close(fd)


def test_frozen_authority_constants_and_reference_hashes() -> None:
    assert salvage.PREISSUE_COMMIT == "6a953c9996cc7ff446a1a6a3aa7e7c29ccc7044d"
    assert salvage.SUPPLEMENT_COMMIT == "4b71a1686f45638f9b4087c85401478626fc2ed7"
    assert salvage._canonical_hash(list(salvage.FAILED_REFERENCES)) == salvage.FAILED_REFERENCE_SET_SHA256
    assert salvage._canonical_hash(list(salvage.CORRECTED_REFERENCES)) == salvage.CORRECTED_REFERENCE_SET_SHA256
    assert salvage.FAILED_REFERENCES[:4] == salvage.CORRECTED_REFERENCES[:4]
    assert salvage.FAILED_REFERENCES[5:] == salvage.CORRECTED_REFERENCES[5:]

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from imu_vive_learning.openaxes_external_native_v3_config import (
    OPENAXES_NATIVE_V3_SALVAGE_INPUT_REFERENCE_ORDER,
    OPENAXES_NATIVE_V3_SALVAGE_RECORD_PATHS,
    OPENAXES_NATIVE_V3_SALVAGE_SEMANTIC_ASSERTIONS,
)
from imu_vive_learning.openaxes_external_native_v3_parent_receipt import (
    EMPTY_SHA256,
    ORIGINAL_RECOVERY_IMPLEMENTATION_COMMIT,
    ORIGINAL_RECOVERY_SOURCE_EVIDENCE_COMMIT,
    RECORD_CONTRACT,
    SALVAGE_IMPLEMENTATION_COMMIT,
    SALVAGE_SOURCE_EVIDENCE_COMMIT,
    SET_CONTRACT,
    SET_RECEIPT_PATH,
    STATIC_REFERENCES,
    OpenAxesExternalNativeV3ParentReceiptError,
    _canonical,
    _canonical_file,
    _canonical_hash,
    copy_salvage_validation_bundle,
    validate_salvage_validation_bundle,
)


def _payload(value: dict[str, object]) -> dict[str, object]:
    output = dict(value)
    output["payload_sha256"] = hashlib.sha256(_canonical(output)).hexdigest()
    return output


def _dynamic_reference(path: str) -> dict[str, object]:
    raw = f"authority:{path}".encode("ascii")
    return {"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _fixture(root: Path) -> None:
    directory = root / "parents" / "native_v2_compound_salvage_validations"
    directory.mkdir(parents=True, mode=0o700)
    common = {
        **STATIC_REFERENCES,
        "corrected_stage_seal_reference": _dynamic_reference("STAGE_SEAL.json"),
        "standard_compound_terminal_reference": _dynamic_reference("COMPOUND_TERMINAL.json"),
        "salvage_terminal_reference": _dynamic_reference("SALVAGE_TERMINAL.json"),
        "run003_complete_reference": _dynamic_reference("COMPLETE.json"),
        "run003_development_manifest_reference": _dynamic_reference("development_manifest.json"),
    }
    input_hash = _canonical_hash(
        [common[key] for key in OPENAXES_NATIVE_V3_SALVAGE_INPUT_REFERENCE_ORDER]
    )
    refs: list[dict[str, object]] = []
    for ordinal, path in enumerate(OPENAXES_NATIVE_V3_SALVAGE_RECORD_PATHS):
        for stream in ("stdout", "stderr"):
            (directory / f"validation_{ordinal:02d}.{stream}.log").write_bytes(b"")
        record = _payload(
            {
                "contract": RECORD_CONTRACT,
                "schema_version": 1,
                "run_id": "openaxes-native-v2-sealed-003",
                "validation_ordinal": ordinal,
                "invocation_uuid_v4": f"00000000-0000-4000-8000-{ordinal:012d}",
                "process_identity": {"pid": 1000 + ordinal, "proc_start_ticks": 2000 + ordinal},
                "original_recovery_implementation_git_commit": ORIGINAL_RECOVERY_IMPLEMENTATION_COMMIT,
                "original_recovery_source_evidence_git_commit": ORIGINAL_RECOVERY_SOURCE_EVIDENCE_COMMIT,
                "original_recovery_source_audit_reference": common["original_recovery_source_audit_reference"],
                "salvage_implementation_git_commit": SALVAGE_IMPLEMENTATION_COMMIT,
                "salvage_source_evidence_git_commit": SALVAGE_SOURCE_EVIDENCE_COMMIT,
                "salvage_source_audit_reference": common["salvage_source_audit_reference"],
                "v2_salvage_erratum2_reference": common["v2_salvage_erratum2_reference"],
                "v2_salvage_preissue_reference": common["v2_salvage_preissue_reference"],
                "v2_salvage_accounting_supplement_reference": common["v2_salvage_accounting_supplement_reference"],
                "corrected_stage_seal_reference": common["corrected_stage_seal_reference"],
                "standard_compound_terminal_reference": common["standard_compound_terminal_reference"],
                "salvage_terminal_reference": common["salvage_terminal_reference"],
                "run003_complete_reference": common["run003_complete_reference"],
                "run003_development_manifest_reference": common["run003_development_manifest_reference"],
                "input_reference_set_sha256": input_hash,
                "original_recovery_full_validate_exit_code": 0,
                "salvage_validate_exit_code": 0,
                "stdout_reference": {"path": f"parents/native_v2_compound_salvage_validations/validation_{ordinal:02d}.stdout.log", "bytes": 0, "sha256": EMPTY_SHA256},
                "stderr_reference": {"path": f"parents/native_v2_compound_salvage_validations/validation_{ordinal:02d}.stderr.log", "bytes": 0, "sha256": EMPTY_SHA256},
                "semantic_assertions": OPENAXES_NATIVE_V3_SALVAGE_SEMANTIC_ASSERTIONS,
                "scientific_values_disclosed": False,
            }
        )
        raw = _canonical_file(record)
        (root / path).write_bytes(raw)
        refs.append({"path": path, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    set_receipt = _payload(
        {
            "contract": SET_CONTRACT,
            "schema_version": 1,
            "run_id": "openaxes-native-v2-sealed-003",
            "record_count": 10,
            "salvage_validation_record_references": refs,
            "salvage_validation_receipt_set_sha256": _canonical_hash(refs),
            "original_recovery_source_audit_reference": common["original_recovery_source_audit_reference"],
            "original_recovery_source_evidence_git_commit": ORIGINAL_RECOVERY_SOURCE_EVIDENCE_COMMIT,
            "salvage_source_audit_reference": common["salvage_source_audit_reference"],
            "salvage_source_evidence_git_commit": SALVAGE_SOURCE_EVIDENCE_COMMIT,
            "v2_salvage_erratum2_reference": common["v2_salvage_erratum2_reference"],
            "v2_salvage_preissue_reference": common["v2_salvage_preissue_reference"],
            "v2_salvage_accounting_supplement_reference": common["v2_salvage_accounting_supplement_reference"],
            "corrected_stage_seal_reference": common["corrected_stage_seal_reference"],
            "standard_compound_terminal_reference": common["standard_compound_terminal_reference"],
            "salvage_terminal_reference": common["salvage_terminal_reference"],
            "run003_complete_reference": common["run003_complete_reference"],
            "run003_development_manifest_reference": common["run003_development_manifest_reference"],
            "scientific_values_disclosed": False,
        }
    )
    (root / SET_RECEIPT_PATH).write_bytes(_canonical_file(set_receipt))


def test_validate_and_copy_exact_ten_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    _fixture(source)
    bundle = validate_salvage_validation_bundle(source)
    assert len(bundle.records) == len(bundle.record_references) == 10
    assert bundle.receipt_set_sha256 == _canonical_hash(list(bundle.record_references))
    assert bundle.parent_fields()["v2_salvage_accounting_supplement_reference"] == STATIC_REFERENCES["v2_salvage_accounting_supplement_reference"]
    destination = tmp_path / "destination"
    destination.mkdir(mode=0o700)
    copied = copy_salvage_validation_bundle(source, destination)
    assert copied.record_raw_bytes == bundle.record_raw_bytes
    assert copied.receipt_set_raw == bundle.receipt_set_raw
    assert len(list((destination / "parents" / "native_v2_compound_salvage_validations").iterdir())) == 31
    with pytest.raises(OpenAxesExternalNativeV3ParentReceiptError, match="retry"):
        copy_salvage_validation_bundle(source, destination)


def test_record_payload_and_supplement_are_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    _fixture(source)
    path = source / OPENAXES_NATIVE_V3_SALVAGE_RECORD_PATHS[0]
    value = json.loads(path.read_text(encoding="utf-8"))
    value.pop("v2_salvage_accounting_supplement_reference")
    path.write_bytes(_canonical_file(_payload(value)))
    with pytest.raises(OpenAxesExternalNativeV3ParentReceiptError, match="keys"):
        validate_salvage_validation_bundle(source)


def test_nonempty_child_stream_and_reused_process_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    _fixture(source)
    directory = source / "parents" / "native_v2_compound_salvage_validations"
    (directory / "validation_03.stderr.log").write_bytes(b"unexpected")
    with pytest.raises(OpenAxesExternalNativeV3ParentReceiptError, match="stderr"):
        validate_salvage_validation_bundle(source)

    source2 = tmp_path / "source2"
    source2.mkdir(mode=0o700)
    _fixture(source2)
    first = json.loads((source2 / OPENAXES_NATIVE_V3_SALVAGE_RECORD_PATHS[0]).read_text())
    second_path = source2 / OPENAXES_NATIVE_V3_SALVAGE_RECORD_PATHS[1]
    second = json.loads(second_path.read_text())
    second["process_identity"] = first["process_identity"]
    second_path.write_bytes(_canonical_file(_payload({key: value for key, value in second.items() if key != "payload_sha256"})))
    with pytest.raises(OpenAxesExternalNativeV3ParentReceiptError, match="reused"):
        validate_salvage_validation_bundle(source2)

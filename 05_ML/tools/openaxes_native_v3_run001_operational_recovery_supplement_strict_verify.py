#!/usr/bin/env python3
"""Strictly verify the immutable Run 001 operational-recovery supplement.

This program is verification-only.  It cannot issue, replace, repair, or
delete a receipt.  It pins the already-issued supplement and its issuer by raw
SHA-256, validates every safety/claim boundary exactly, and recomputes the live
source, output, runtime, action-accounting, and 72-fit poststate projections.

Historical limitation: the immutable issuer used O_EXCL and verified its
created file descriptor, but it did not retain descriptor authority for every
parent path component or fsync the parent directory.  That historical action
cannot be repaired without replacing the self-referenced issuer/receipt.  This
strict verifier instead walks every receipt parent component descriptor-
relatively with O_NOFOLLOW and retains the final parent while reading.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
from typing import Any, Mapping, Sequence


PROJECT = Path("/home/shin-server/capstone-code/09_SO101_IMU_VIVE_LEARNING")
ISSUER_PATH = PROJECT / "tools/openaxes_native_v3_run001_operational_recovery_supplement.py"
RECEIPT_PATH = PROJECT / "receipts/openaxes_external_native_v3_run001_operational_recovery_supplement.json"

EXPECTED_RECEIPT_RAW_SHA256 = "c27d9dd9d41cd25842f1c8c2d672b18afb3d2ee179136f99022193408d6b2377"
EXPECTED_RECEIPT_PAYLOAD_SHA256 = "ac231af5f81179a4aaa1705165482041381ae3a417b3495404369740e03e9fc4"
EXPECTED_ISSUER_RAW_SHA256 = "d4f06e656631bcabd6614e8480529a83d4c7d8561a5d2dd21faccd8ca9e8d982"
EXPECTED_ISSUED_AT_UTC = "2026-08-26T10:35:00Z"

EXPECTED_SCOPE = {
    "authority_kind": "append_only_operational_recovery_supplement",
    "scientific_outputs_recomputed": False,
    "existing_receipts_modified": False,
    "full_42_context_grid_complete": False,
    "final_claim_eligible": False,
    "claim_authorized": False,
}
EXPECTED_SCOPE_SHA256 = "b42a6351e7c9b5108831e52d39b0ff2caea9bf028f5075ca7af4c0b0849d70a0"

EXPECTED_SOURCE_ACTION_PROJECTION_SHA256 = (
    "543e92e3169e2736f7889f4b23df2a1d66cc32bb29b22259982863881b1cdb3e"
)
EXPECTED_CLAIM_BOUNDARY = {
    "retraining_performed_field_asserted": False,
    "no_new_fit_evidence": {
        "exact_selected_attempt_count": 72,
        "exact_selected_result_count": 72,
        "unique_attempt_uuid_count": 72,
        "scientific_fit_attempt_values": [1],
        "all_scientific_member_mtimes_precede_recovery": True,
        "recovery_sources_contain_fit_or_internal_fit_invocation": False,
        "source_action_projection_sha256": EXPECTED_SOURCE_ACTION_PROJECTION_SHA256,
        "recovery_journal_contains_fit_or_internal_fit_invocation": False,
    },
    "no_retraining_conclusion": (
        "the preserved poststate and reviewed recovery code/journal contain no evidence "
        "of a new fit or internal-fit invocation; this is evidence, not a cryptographic "
        "proof of every historical process action"
    ),
    "pre_post_scientific_byte_equality_available": False,
    "pre_post_scientific_byte_equality_claimed": False,
    "prestate_limitation": (
        "no exact 72-member byte projection was sealed before recovery, so byte-for-byte "
        "pre/post equality cannot be established retrospectively; only the exact current "
        "72-member poststate, single-attempt lineage, pre-recovery mtimes, and recovery "
        "source/journal evidence are bound"
    ),
    "executed_source_byte_identity_cryptographically_proven": False,
    "scientific_claims_extended": False,
    "adaptive_followup_remains_nonconfirmatory": True,
}
EXPECTED_CLAIM_BOUNDARY_SHA256 = (
    "6933fca7045c3e63809637538b10ff8adc58e9b535732cd756ef023f1bc5aee0"
)

EXPECTED_RAW_AUTHORITY_SHA256 = (
    "5041804b9bdbd245b6768f3768335b1b1b970f65efc326e0a7dfebcfa75ddb2a"
)
RAW_CAPTURE_SPECS = {
    "five_loaded_failure_units_systemctl_capture": {
        "command_constant": "SYSTEMCTL_COMMAND",
        "bytes": 3668,
        "sha256": "0d81b1233253d9eb315f5024b69b04910b462eac3f337a656cfd823228552de7",
        "stat_sha256": "84951d8e7c7af2e63df4c8b2f29055d37ff3ba1bf51ba24a157a6041625abd33",
    },
    "original_and_first_recovery_journal_capture": {
        "command_constant": "ORIGINAL_JOURNAL_COMMAND",
        "bytes": 4249,
        "sha256": "abe08676c97a5e1f3160bc522e295cb8ff14787aad8b1eec4534bfdb96dc949e",
        "stat_sha256": "65001a6ab4ec68d19ba9f9418195d3c7727c6fb53acb3132fd8f853c5b5e4c00",
    },
    "successful_v2_transient_recovery_journal_capture": {
        "command_constant": "RECOVERY_JOURNAL_COMMAND",
        "bytes": 2252,
        "sha256": "4c5ee78a962182ddf8c3d035a36eb09ea49088d01d9464f0d35261daf20d0917",
        "stat_sha256": "fc697b7c98c9ddc0e2a87b88ca588227347cd645144425a3b829f9db245d142b",
    },
}

TOP_LEVEL_KEYS = {
    "contract",
    "schema_version",
    "run_id",
    "issued_at_utc",
    "scope",
    "raw_operational_authority",
    "source_authority",
    "post_recovery_scientific_tree",
    "runtime_status",
    "output_authority",
    "exact_action_accounting",
    "claim_boundary",
    "payload_sha256",
}
RAW_AUTHORITY_KEYS = {
    "five_loaded_failure_units_systemctl_capture",
    "five_loaded_failure_units_semantic_projection",
    "original_and_first_recovery_journal_capture",
    "successful_v2_transient_recovery_journal_capture",
    "journal_semantic_assertions",
    "successful_transient_units_currently_loaded",
    "successful_execution_evidence_is_journal_not_live_unit_state",
}
RAW_CAPTURE_KEYS = {"command", "raw_utf8", "bytes", "sha256", "capture_file_stat"}
CAPTURE_STAT_KEYS = {
    "file_type",
    "mode",
    "uid",
    "gid",
    "nlink",
    "device",
    "inode",
    "bytes",
    "mtime_ns",
    "mtime_utc",
    "ctime_ns",
    "ctime_utc",
}


class StrictVerifyError(RuntimeError):
    """The supplement is not the exact pinned and live-valid authority."""


def _fail(message: str) -> None:
    raise StrictVerifyError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StrictVerifyError("value is not finite canonical JSON") from exc


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_no_symlink_path(path: Path, *, name: str) -> tuple[bytes, os.stat_result]:
    """Read a file while retaining every path component and the final parent."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        _fail("strict no-follow descriptor support is unavailable")
    absolute = path.absolute()
    if (
        not absolute.is_absolute()
        or absolute.name in {"", ".", ".."}
        or any(component in {"", ".", ".."} for component in absolute.parts[1:])
    ):
        _fail(f"{name} path is not an absolute file path")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    parent_fd = -1
    file_fd = -1
    try:
        parent_fd = os.open("/", directory_flags)
        for component in absolute.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        parent_before = os.fstat(parent_fd)
        file_fd = os.open(absolute.name, file_flags, dir_fd=parent_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _fail(f"{name} is not a single-link regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after = os.fstat(file_fd)
        named_after = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        parent_after = os.fstat(parent_fd)
        if (
            _identity(before) != _identity(after)
            or _identity(before) != _identity(named_after)
            or _identity(parent_before) != _identity(parent_after)
            or len(raw) != before.st_size
        ):
            _fail(f"{name} or its retained parent changed while being read")
        return raw, before
    except StrictVerifyError:
        raise
    except OSError as exc:
        raise StrictVerifyError(f"cannot retain/read no-symlink {name}: {absolute}") from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _load_pinned_issuer() -> tuple[types.ModuleType, bytes, os.stat_result]:
    raw, info = _read_no_symlink_path(ISSUER_PATH, name="pinned supplement issuer")
    if _sha256(raw) != EXPECTED_ISSUER_RAW_SHA256:
        _fail("supplement issuer raw SHA-256 changed")
    if info.st_uid != os.geteuid() or info.st_gid != os.getegid():
        _fail("supplement issuer ownership changed")
    try:
        source = raw.decode("utf-8")
        code = compile(source, ISSUER_PATH.as_posix(), "exec", dont_inherit=True)
    except (UnicodeError, SyntaxError) as exc:
        raise StrictVerifyError("pinned supplement issuer is not valid UTF-8 Python") from exc
    module = types.ModuleType("_openaxes_run001_pinned_supplement_issuer")
    module.__file__ = ISSUER_PATH.as_posix()
    module.__package__ = ""
    exec(code, module.__dict__)
    return module, raw, info


def _load_receipt(path: Path) -> tuple[dict[str, Any], bytes, os.stat_result]:
    raw, info = _read_no_symlink_path(path, name="operational recovery supplement")
    if stat.S_IMODE(info.st_mode) != 0o600:
        _fail("supplement mode is not exactly 0600")
    if info.st_uid != os.geteuid() or info.st_gid != os.getegid():
        _fail("supplement uid/gid do not match the verifier identity")
    if _sha256(raw) != EXPECTED_RECEIPT_RAW_SHA256:
        _fail("supplement raw SHA-256 is not the pinned issued receipt")
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StrictVerifyError("supplement is not UTF-8 JSON") from exc
    if type(receipt) is not dict or raw != _canonical(receipt) + b"\n":
        _fail("supplement is not canonical object JSON with one final newline")
    payload = receipt.get("payload_sha256")
    body = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    computed = _sha256(_canonical(body))
    if payload != EXPECTED_RECEIPT_PAYLOAD_SHA256 or computed != EXPECTED_RECEIPT_PAYLOAD_SHA256:
        _fail("supplement payload SHA-256 is not the pinned canonical payload")
    return receipt, raw, info


def _verify_raw_capture(
    capture: Any,
    *,
    key: str,
    issuer: types.ModuleType,
) -> None:
    if type(capture) is not dict or set(capture) != RAW_CAPTURE_KEYS:
        _fail(f"raw capture schema changed: {key}")
    spec = RAW_CAPTURE_SPECS[key]
    command = getattr(issuer, spec["command_constant"], None)
    if capture.get("command") != command:
        _fail(f"raw capture command changed: {key}")
    text = capture.get("raw_utf8")
    if not isinstance(text, str) or not text.endswith("\n"):
        _fail(f"raw capture text/newline changed: {key}")
    raw = text.encode("utf-8")
    if (
        capture.get("bytes") != spec["bytes"]
        or len(raw) != spec["bytes"]
        or capture.get("sha256") != spec["sha256"]
        or _sha256(raw) != spec["sha256"]
    ):
        _fail(f"raw capture bytes/SHA-256 changed: {key}")
    capture_stat = capture.get("capture_file_stat")
    if type(capture_stat) is not dict or set(capture_stat) != CAPTURE_STAT_KEYS:
        _fail(f"raw capture stat schema changed: {key}")
    if _sha256(_canonical(capture_stat)) != spec["stat_sha256"]:
        _fail(f"raw capture historical stat changed: {key}")


def _verify_raw_authority(receipt: Mapping[str, Any], issuer: types.ModuleType) -> None:
    raw_authority = receipt.get("raw_operational_authority")
    if type(raw_authority) is not dict or set(raw_authority) != RAW_AUTHORITY_KEYS:
        _fail("raw operational authority schema changed")
    if _sha256(_canonical(raw_authority)) != EXPECTED_RAW_AUTHORITY_SHA256:
        _fail("raw operational authority exact projection changed")
    for key in RAW_CAPTURE_SPECS:
        _verify_raw_capture(raw_authority.get(key), key=key, issuer=issuer)

    systemctl_capture = raw_authority["five_loaded_failure_units_systemctl_capture"]
    try:
        parsed_units = issuer._parse_systemctl_capture(systemctl_capture)
    except issuer.SupplementError as exc:
        raise StrictVerifyError("raw systemctl semantics are invalid") from exc
    if raw_authority.get("five_loaded_failure_units_semantic_projection") != parsed_units:
        _fail("stored systemctl semantic projection does not match raw capture")
    original_journal = raw_authority["original_and_first_recovery_journal_capture"]
    recovery_journal = raw_authority["successful_v2_transient_recovery_journal_capture"]
    try:
        assertions = issuer._validate_journal_captures(original_journal, recovery_journal)
    except issuer.SupplementError as exc:
        raise StrictVerifyError("raw journal semantics are invalid") from exc
    if raw_authority.get("journal_semantic_assertions") != assertions:
        _fail("stored journal assertions do not match raw captures")
    if raw_authority.get("successful_transient_units_currently_loaded") is not False:
        _fail("successful transient-unit live-state boundary changed")
    if raw_authority.get("successful_execution_evidence_is_journal_not_live_unit_state") is not True:
        _fail("successful recovery authority-source boundary changed")


def _compute_live_projections(issuer: types.ModuleType) -> dict[str, Any]:
    try:
        source = issuer._source_authority()
        return {
            "source_authority": source,
            "post_recovery_scientific_tree": issuer._post_scientific_tree(),
            "runtime_status": issuer._runtime_status_projection(),
            "output_authority": issuer._output_authority(),
            "exact_action_accounting": issuer._exact_action_accounting(
                source["successful_action_source_projection"]
            ),
        }
    except issuer.SupplementError as exc:
        raise StrictVerifyError("strict live recomputation failed") from exc


def _verify_receipt_semantics(
    receipt: Mapping[str, Any],
    *,
    issuer: types.ModuleType,
    live: Mapping[str, Any],
) -> None:
    if type(receipt) is not dict or set(receipt) != TOP_LEVEL_KEYS:
        _fail("supplement top-level schema changed")
    if (
        receipt.get("contract") != issuer.CONTRACT
        or receipt.get("schema_version") != issuer.SCHEMA_VERSION
        or receipt.get("run_id") != issuer.RUN_ID
        or receipt.get("issued_at_utc") != EXPECTED_ISSUED_AT_UTC
    ):
        _fail("supplement identity/issuance timestamp changed")

    scope = receipt.get("scope")
    if scope != EXPECTED_SCOPE or _sha256(_canonical(scope)) != EXPECTED_SCOPE_SHA256:
        _fail("supplement exact scope/claim eligibility changed")
    _verify_raw_authority(receipt, issuer)

    claims = receipt.get("claim_boundary")
    if (
        claims != EXPECTED_CLAIM_BOUNDARY
        or _sha256(_canonical(claims)) != EXPECTED_CLAIM_BOUNDARY_SHA256
    ):
        _fail("supplement exact claim boundary/evidence/limitations changed")

    for key in (
        "source_authority",
        "post_recovery_scientific_tree",
        "runtime_status",
        "output_authority",
        "exact_action_accounting",
    ):
        if receipt.get(key) != live[key]:
            _fail(f"supplement {key} no longer matches strict live recomputation")


def verify_receipt(path: Path = RECEIPT_PATH) -> dict[str, Any]:
    receipt, raw, receipt_info = _load_receipt(path)
    issuer, issuer_raw, issuer_info = _load_pinned_issuer()
    live = _compute_live_projections(issuer)
    _verify_receipt_semantics(receipt, issuer=issuer, live=live)

    issuer_after, issuer_after_info = _read_no_symlink_path(
        ISSUER_PATH, name="pinned supplement issuer final pass"
    )
    if issuer_after != issuer_raw or _identity(issuer_after_info) != _identity(issuer_info):
        _fail("supplement issuer changed across strict verification")
    return {
        "contract": receipt["contract"],
        "receipt_bytes": len(raw),
        "receipt_raw_sha256": EXPECTED_RECEIPT_RAW_SHA256,
        "receipt_payload_sha256": EXPECTED_RECEIPT_PAYLOAD_SHA256,
        "receipt_mode": format(stat.S_IMODE(receipt_info.st_mode), "04o"),
        "receipt_nlink": receipt_info.st_nlink,
        "issuer_raw_sha256": EXPECTED_ISSUER_RAW_SHA256,
        "post_recovery_row_count": live["post_recovery_scientific_tree"]["ordered_row_count"],
        "error_json_count": live["post_recovery_scientific_tree"]["error_json_count"],
        "strict": True,
    }


def _set_nested(root: dict[str, Any], path: Sequence[str], value: Any) -> None:
    node: dict[str, Any] = root
    for component in path[:-1]:
        child = node.get(component)
        if type(child) is not dict:
            raise AssertionError(f"self-test mutation path is not a dict: {path}")
        node = child
    node[path[-1]] = value


def _repayload(receipt: dict[str, Any]) -> None:
    body = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    receipt["payload_sha256"] = _sha256(_canonical(body))


def _write_exclusive_test_file(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, mode)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise AssertionError("self-test write made no progress")
            offset += written
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _expect_rejection(action: Any, *, label: str) -> None:
    try:
        action()
    except StrictVerifyError:
        return
    raise StrictVerifyError(f"strict self-test mutation was accepted: {label}")


def self_test() -> dict[str, Any]:
    verify_receipt(RECEIPT_PATH)
    receipt, raw, _ = _load_receipt(RECEIPT_PATH)
    issuer, _, _ = _load_pinned_issuer()
    live = _compute_live_projections(issuer)

    mutations: tuple[tuple[str, tuple[str, ...], Any], ...] = (
        ("scope.final_claim_eligible", ("scope", "final_claim_eligible"), True),
        ("scope.claim_authorized", ("scope", "claim_authorized"), True),
        (
            "claim.adaptive_nonconfirmatory",
            ("claim_boundary", "adaptive_followup_remains_nonconfirmatory"),
            False,
        ),
        (
            "claim.selected_attempt_count",
            ("claim_boundary", "no_new_fit_evidence", "exact_selected_attempt_count"),
            999,
        ),
        (
            "claim.prestate_limitation",
            ("claim_boundary", "prestate_limitation"),
            "fabricated limitation",
        ),
        (
            "raw.systemctl.command",
            (
                "raw_operational_authority",
                "five_loaded_failure_units_systemctl_capture",
                "command",
            ),
            "fabricated command",
        ),
        (
            "raw.systemctl.bytes",
            (
                "raw_operational_authority",
                "five_loaded_failure_units_systemctl_capture",
                "bytes",
            ),
            1,
        ),
        (
            "raw.systemctl.sha256",
            (
                "raw_operational_authority",
                "five_loaded_failure_units_systemctl_capture",
                "sha256",
            ),
            "0" * 64,
        ),
        (
            "raw.capture_stat.mode",
            (
                "raw_operational_authority",
                "five_loaded_failure_units_systemctl_capture",
                "capture_file_stat",
                "mode",
            ),
            "0777",
        ),
        (
            "raw.semantic_projection",
            (
                "raw_operational_authority",
                "five_loaded_failure_units_semantic_projection",
            ),
            [],
        ),
        (
            "raw.journal_assertions",
            ("raw_operational_authority", "journal_semantic_assertions"),
            {},
        ),
        (
            "raw.authority_flag",
            (
                "raw_operational_authority",
                "successful_execution_evidence_is_journal_not_live_unit_state",
            ),
            False,
        ),
        (
            "action.evaluate_calls",
            ("exact_action_accounting", "successful_recovery", "evaluate_calls"),
            2,
        ),
        (
            "posttree.row_count",
            ("post_recovery_scientific_tree", "ordered_row_count"),
            71,
        ),
        (
            "output.validator_count",
            ("output_authority", "sealed_validation_bundle", "validator_process_count"),
            9,
        ),
        (
            "source.issuer_sha",
            ("source_authority", "supplement_generator_source_reference", "sha256"),
            "0" * 64,
        ),
        (
            "runtime.completed_count",
            ("runtime_status", "semantic_projection", "state_counts", "completed"),
            71,
        ),
    )

    semantic_rejections = 0
    temp_rejections = 0
    with tempfile.TemporaryDirectory(prefix="openaxes-run001-strict-selftest-") as directory:
        temp_root = Path(directory)
        for ordinal, (label, path, value) in enumerate(mutations):
            mutated = copy.deepcopy(receipt)
            _set_nested(mutated, path, value)
            _repayload(mutated)
            _expect_rejection(
                lambda mutated=mutated: _verify_receipt_semantics(
                    mutated, issuer=issuer, live=live
                ),
                label=f"semantic:{label}",
            )
            semantic_rejections += 1

            mutated_path = temp_root / f"mutated-{ordinal:02d}.json"
            _write_exclusive_test_file(mutated_path, _canonical(mutated) + b"\n")
            _expect_rejection(
                lambda mutated_path=mutated_path: verify_receipt(mutated_path),
                label=f"file:{label}",
            )
            temp_rejections += 1

        bad_mode = temp_root / "bad-mode.json"
        _write_exclusive_test_file(bad_mode, raw, mode=0o644)
        _expect_rejection(lambda: verify_receipt(bad_mode), label="stat:mode0644")

        hardlinked = temp_root / "hardlinked.json"
        alias = temp_root / "hardlinked-alias.json"
        _write_exclusive_test_file(hardlinked, raw)
        os.link(hardlinked, alias)
        _expect_rejection(lambda: verify_receipt(hardlinked), label="stat:nlink2")

        real_parent = temp_root / "real-parent"
        real_parent.mkdir(mode=0o700)
        symlink_parent = temp_root / "symlink-parent"
        symlink_parent.symlink_to(real_parent, target_is_directory=True)
        symlink_receipt = real_parent / "receipt.json"
        _write_exclusive_test_file(symlink_receipt, raw)
        _expect_rejection(
            lambda: verify_receipt(symlink_parent / "receipt.json"),
            label="stat:symlink-parent",
        )

    return {
        "canonical_receipt_verified": True,
        "semantic_mutation_labels": [label for label, _, _ in mutations],
        "semantic_mutations_rejected": semantic_rejections,
        "temporary_mutated_receipts_rejected": temp_rejections,
        "mode_0644_rejected": True,
        "nlink_2_rejected": True,
        "symlink_parent_rejected": True,
        "strict": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="verify the pinned immutable receipt")
    verify.add_argument("--receipt", default=RECEIPT_PATH.as_posix())
    subparsers.add_parser("self-test", help="run strict rejection tests in a private temp root")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        summary = verify_receipt(Path(args.receipt))
    elif args.command == "self-test":
        summary = self_test()
    else:
        _fail("unknown command")
    print((_canonical(summary) + b"\n").decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StrictVerifyError as exc:
        raise SystemExit(f"error: {exc}") from exc

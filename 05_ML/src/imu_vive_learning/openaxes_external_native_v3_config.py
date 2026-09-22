"""Frozen configuration and scheduling boundary for OpenAXES native V3.

The base scientific YAML is authenticated independently from the outcome-
neutral erratum4 parent-gate overlay and its accounting supplement.  Loader
metadata is private (underscore-prefixed) and never contributes to the frozen
base semantic digest.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import yaml


OPENAXES_EXTERNAL_NATIVE_V3_CONTRACT = (
    "openaxes.external.native_sensor_experts.v3.allmask_dmome"
)
OPENAXES_EXTERNAL_NATIVE_V3_CONFIG_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/configs/"
    "openaxes_external_native_v3_allmask_dmome.yaml"
)
OPENAXES_EXTERNAL_NATIVE_V3_ERRATUM4_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/configs/"
    "openaxes_external_native_v3_erratum4_salvage_validation.yaml"
)
OPENAXES_EXTERNAL_NATIVE_V3_ERRATUM4_SUPPLEMENT_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/receipts/"
    "openaxes_external_native_v3_preregistration_"
    "erratum4_accounting_supplement_preissue.json"
)
OPENAXES_EXTERNAL_NATIVE_V3_CONFIG_RAW_SHA256 = (
    "c9f828dfd1da82f0031c16a7802e3679081a3cd7ec37657435802e4d05a93381"
)
OPENAXES_EXTERNAL_NATIVE_V3_CONFIG_SHA256 = (
    "e6ae40b9e25061447689c9c6d534306b417563fcfd4a1af7a292ce2f92c3d793"
)
OPENAXES_EXTERNAL_NATIVE_V3_ERRATUM4_RAW_SHA256 = (
    "29b0664c782c0f782e5aff6c3e0d78297e1d701d79f73456002fab6c9b9b1ac2"
)
OPENAXES_EXTERNAL_NATIVE_V3_ERRATUM4_SHA256 = (
    "ad5bfbc6af2c3665eb27a082cb0b6e80544a4d31be0f57232474c90d178730c9"
)
OPENAXES_EXTERNAL_NATIVE_V3_ERRATUM4_SUPPLEMENT_RAW_SHA256 = (
    "7272358884667c8d898ddcc678a7977402024b10325d5cde0f0f353856e43b8f"
)
OPENAXES_EXTERNAL_NATIVE_V3_ERRATUM4_SUPPLEMENT_PAYLOAD_SHA256 = (
    "0c4876f66add3701e4b186108c94a96e8905968cbf30f6c6a23edca4ce95ccb9"
)

OPENAXES_NATIVE_V3_SENSOR_ORDER = ("humerus", "radius", "carpus", "digitus")
OPENAXES_NATIVE_V3_SEEDS = (1701, 2603, 3907)
OPENAXES_NATIVE_V3_CANDIDATE_IDS = (
    "dense_6979__allmask_cardinality_balanced_erm",
    "static_sensor_experts_6979__allmask_cardinality_balanced_erm",
    "dmome_6979__one_stage_allmask_cardinality_balanced_erm",
    "dmome_6979__two_stage_allmask_cardinality_balanced_erm",
    "dmome_6979__two_stage_allmask_cardinality_balanced_mofe",
    "dmome_6979__two_stage_allmask_cardinality_balanced_mofe_cagrad",
)
OPENAXES_NATIVE_V3_VIEW_IDS = (
    "full",
    "drop_humerus",
    "drop_radius",
    "drop_carpus",
    "drop_digitus",
    "only_humerus_radius",
    "only_humerus_carpus",
    "only_humerus_digitus",
    "only_radius_carpus",
    "only_radius_digitus",
    "only_carpus_digitus",
    "only_humerus",
    "only_radius",
    "only_carpus",
    "only_digitus",
)
OPENAXES_NATIVE_V3_MASKS = (
    (1, 1, 1, 1),
    (0, 1, 1, 1),
    (1, 0, 1, 1),
    (1, 1, 0, 1),
    (1, 1, 1, 0),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
    (0, 1, 0, 1),
    (0, 0, 1, 1),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (0, 0, 1, 0),
    (0, 0, 0, 1),
)
OPENAXES_NATIVE_V3_FOLD_COUNT = 14
OPENAXES_NATIVE_V3_CONTEXT_COUNT = 42
OPENAXES_NATIVE_V3_TASK_COUNT = 252
OPENAXES_NATIVE_V3_GPU_COUNT = 4
OPENAXES_NATIVE_V3_LOGICAL_SLOT_COUNT = 12
OPENAXES_NATIVE_V3_CONTEXTS_PER_SLOT = (4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3)
OPENAXES_NATIVE_V3_FITS_PER_SLOT = (24, 24, 24, 24, 24, 24, 18, 18, 18, 18, 18, 18)
OPENAXES_NATIVE_V3_FITS_PER_GPU = (66, 66, 60, 60)

# Compact stable aliases used by the V3 experiment/training modules.  The
# prefixed forms above remain the cross-module authority names.
CANDIDATE_IDS = OPENAXES_NATIVE_V3_CANDIDATE_IDS
SEEDS = OPENAXES_NATIVE_V3_SEEDS
VIEW_IDS = OPENAXES_NATIVE_V3_VIEW_IDS
MASKS = OPENAXES_NATIVE_V3_MASKS
FOLD_COUNT = OPENAXES_NATIVE_V3_FOLD_COUNT
CONTEXT_COUNT = OPENAXES_NATIVE_V3_CONTEXT_COUNT
TASK_COUNT = OPENAXES_NATIVE_V3_TASK_COUNT
GPU_COUNT = OPENAXES_NATIVE_V3_GPU_COUNT
LOGICAL_SLOT_COUNT = OPENAXES_NATIVE_V3_LOGICAL_SLOT_COUNT
CONTEXTS_PER_SLOT = OPENAXES_NATIVE_V3_CONTEXTS_PER_SLOT
FITS_PER_SLOT = OPENAXES_NATIVE_V3_FITS_PER_SLOT
FITS_PER_GPU = OPENAXES_NATIVE_V3_FITS_PER_GPU

OPENAXES_NATIVE_V3_SALVAGE_RECORD_PATHS = tuple(
    f"parents/native_v2_compound_salvage_validations/validation_{ordinal:02d}.json"
    for ordinal in range(10)
)
OPENAXES_NATIVE_V3_SALVAGE_RECORD_KEYS = (
    "contract", "schema_version", "run_id", "validation_ordinal",
    "invocation_uuid_v4", "process_identity",
    "original_recovery_implementation_git_commit",
    "original_recovery_source_evidence_git_commit",
    "original_recovery_source_audit_reference",
    "salvage_implementation_git_commit", "salvage_source_evidence_git_commit",
    "salvage_source_audit_reference", "v2_salvage_erratum2_reference",
    "v2_salvage_preissue_reference",
    "v2_salvage_accounting_supplement_reference",
    "corrected_stage_seal_reference", "standard_compound_terminal_reference",
    "salvage_terminal_reference", "run003_complete_reference",
    "run003_development_manifest_reference", "input_reference_set_sha256",
    "original_recovery_full_validate_exit_code", "salvage_validate_exit_code",
    "stdout_reference", "stderr_reference", "semantic_assertions",
    "scientific_values_disclosed", "payload_sha256",
)
OPENAXES_NATIVE_V3_SALVAGE_INPUT_REFERENCE_ORDER = (
    "original_recovery_source_audit_reference",
    "salvage_source_audit_reference",
    "v2_salvage_erratum2_reference",
    "v2_salvage_preissue_reference",
    "v2_salvage_accounting_supplement_reference",
    "corrected_stage_seal_reference",
    "standard_compound_terminal_reference",
    "salvage_terminal_reference",
    "run003_complete_reference",
    "run003_development_manifest_reference",
)
OPENAXES_NATIVE_V3_SALVAGE_SEMANTIC_ASSERTIONS = {
    "original_recovery_full_validate_passed": True,
    "salvage_aware_full_validate_passed": True,
    "embedded_exact_seven_file_semantics_passed": True,
    "frozen_full_run003_validator_passed": True,
    "original_recovery_authority_validator_passed": True,
    "salvage_authority_validator_passed": True,
    "dual_source_lineage_and_git_object_validation_passed": True,
    "failed_stage_and_corrected_stage_boundary_passed": True,
    "standard_and_salvage_terminal_joint_validation_passed": True,
    "run003_prestate_outside_exact_seven_outputs_unchanged": True,
    "salvage_scientific_recomputation_count_zero": True,
    "performance_values_disclosed": False,
}


@dataclass(frozen=True)
class CandidateSchedule:
    candidate_id: str
    architecture: str
    independent_expert_updates: int
    cooperative_updates: int
    objective: str

    @property
    def total_updates(self) -> int:
        return self.independent_expert_updates + self.cooperative_updates


class OpenAxesExternalNativeV3ConfigError(ValueError):
    """Raised when V3 configuration or its authority overlay drifts."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    def construct_mapping(
        self, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        self.flatten_mapping(node)
        output: dict[Any, Any] = {}
        seen: list[Any] = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if any(type(previous) is type(key) and previous == key for previous in seen):
                raise OpenAxesExternalNativeV3ConfigError(
                    f"duplicate native-V3 YAML key: {key!r}"
                )
            output[key] = self.construct_object(value_node, deep=deep)
            seen.append(key)
        return output


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OpenAxesExternalNativeV3ConfigError(
            "native-V3 authority is not finite canonical JSON"
        ) from exc


def _public(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or any(type(key) is not str for key in config):
        raise OpenAxesExternalNativeV3ConfigError(
            "native-V3 configuration must be string-keyed"
        )
    return {key: value for key, value in config.items() if not key.startswith("_")}


def openaxes_external_native_v3_config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(_public(config))).hexdigest()


def _exact_list(value: Any, expected: Sequence[Any], *, name: str) -> None:
    if type(value) is not list or tuple(value) != tuple(expected):
        raise OpenAxesExternalNativeV3ConfigError(f"{name} changed")


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise OpenAxesExternalNativeV3ConfigError(f"{name} must be string-keyed")
    return value


def validate_openaxes_external_native_v3_config(config: Mapping[str, Any]) -> None:
    public = _public(config)
    expected_root = {
        "contract", "erratum1", "erratum2", "claims", "outcome_blind_freeze",
        "lineage", "canonical_preparation_public_api_authority", "parent_gate",
        "preregistration_authority", "input", "availability_lattice", "models",
        "initialization_and_determinism", "row_order_and_reductions", "candidates",
        "objectives", "training", "folds", "metrics", "gates",
        "report_only_analyses", "execution", "validation_and_publication",
        "future_source_closure", "limitations",
    }
    if set(public) != expected_root:
        raise OpenAxesExternalNativeV3ConfigError("native-V3 root keys changed")
    contract = _mapping(public["contract"], name="contract")
    if (
        contract.get("name") != OPENAXES_EXTERNAL_NATIVE_V3_CONTRACT
        or contract.get("version") != 1
        or contract.get("erratum") != 2
    ):
        raise OpenAxesExternalNativeV3ConfigError("native-V3 contract changed")
    lattice = _mapping(public["availability_lattice"], name="availability lattice")
    rows = lattice.get("exact_order")
    if type(rows) is not list or len(rows) != 15:
        raise OpenAxesExternalNativeV3ConfigError("availability lattice changed")
    if tuple(row.get("view_id") for row in rows) != OPENAXES_NATIVE_V3_VIEW_IDS:
        raise OpenAxesExternalNativeV3ConfigError("view order changed")
    if tuple(tuple(row.get("availability", ())) for row in rows) != OPENAXES_NATIVE_V3_MASKS:
        raise OpenAxesExternalNativeV3ConfigError("availability masks changed")
    if tuple(row.get("cardinality") for row in rows) != tuple(sum(mask) for mask in OPENAXES_NATIVE_V3_MASKS):
        raise OpenAxesExternalNativeV3ConfigError("mask cardinalities changed")
    candidates = _mapping(public["candidates"], name="candidates")
    _exact_list(candidates.get("exact_order"), OPENAXES_NATIVE_V3_CANDIDATE_IDS, name="candidate order")
    training = _mapping(public["training"], name="training")
    _exact_list(training.get("seeds"), OPENAXES_NATIVE_V3_SEEDS, name="seed order")
    if training.get("exact_optimizer_updates_per_fit") != 400:
        raise OpenAxesExternalNativeV3ConfigError("update count changed")
    execution = _mapping(public["execution"], name="execution")
    expected_execution = {
        "candidate_count": 6, "fold_count": 14, "seed_count": 3,
        "context_count": 42, "exact_fit_count": 252,
        "gpu_count": 4, "logical_slot_count": 12,
    }
    for key, expected in expected_execution.items():
        if type(execution.get(key)) is not int or execution.get(key) != expected:
            raise OpenAxesExternalNativeV3ConfigError(f"execution.{key} changed")
    _exact_list(execution.get("contexts_per_slot"), OPENAXES_NATIVE_V3_CONTEXTS_PER_SLOT, name="contexts per slot")
    _exact_list(execution.get("fits_per_slot"), OPENAXES_NATIVE_V3_FITS_PER_SLOT, name="fits per slot")
    _exact_list(execution.get("fits_per_gpu"), OPENAXES_NATIVE_V3_FITS_PER_GPU, name="fits per GPU")
    if openaxes_external_native_v3_config_sha256(config) != OPENAXES_EXTERNAL_NATIVE_V3_CONFIG_SHA256:
        raise OpenAxesExternalNativeV3ConfigError("native-V3 semantic SHA-256 changed")


def _parse_yaml(raw: bytes, *, expected_raw_sha256: str, source: str) -> dict[str, Any]:
    if hashlib.sha256(raw).hexdigest() != expected_raw_sha256:
        raise OpenAxesExternalNativeV3ConfigError(f"raw authority changed: {source}")
    try:
        value = yaml.load(raw.decode("utf-8"), Loader=_UniqueKeySafeLoader)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise OpenAxesExternalNativeV3ConfigError(f"cannot parse {source}") from exc
    return deepcopy(dict(_mapping(value, name=source)))


def _validate_erratum4(value: Mapping[str, Any]) -> None:
    if hashlib.sha256(_canonical_json(value)).hexdigest() != OPENAXES_EXTERNAL_NATIVE_V3_ERRATUM4_SHA256:
        raise OpenAxesExternalNativeV3ConfigError("erratum4 semantic SHA-256 changed")
    if value.get("contract") != {
        "name": "openaxes.external.native_sensor_experts.v3.erratum4_salvage_validation",
        "version": 1,
        "status": "preregistered_not_implemented_erratum4",
        "role": "additive_outcome_neutral_dual_lineage_parent_authority_overlay",
    }:
        raise OpenAxesExternalNativeV3ConfigError("erratum4 contract changed")
    record = _mapping(value.get("validation_record"), name="erratum4 record")
    _exact_list(record.get("record_paths"), OPENAXES_NATIVE_V3_SALVAGE_RECORD_PATHS, name="record paths")
    base_keys = tuple(key for key in OPENAXES_NATIVE_V3_SALVAGE_RECORD_KEYS if key != "v2_salvage_accounting_supplement_reference")
    _exact_list(record.get("exact_keys"), base_keys, name="base erratum4 record keys")
    base_inputs = tuple(key for key in OPENAXES_NATIVE_V3_SALVAGE_INPUT_REFERENCE_ORDER if key != "v2_salvage_accounting_supplement_reference")
    _exact_list(record.get("exact_input_reference_order"), base_inputs, name="base erratum4 input order")
    if record.get("semantic_assertions_exact") != OPENAXES_NATIVE_V3_SALVAGE_SEMANTIC_ASSERTIONS:
        raise OpenAxesExternalNativeV3ConfigError("erratum4 semantic assertions changed")


def _validate_supplement(value: Mapping[str, Any]) -> None:
    payload = dict(value)
    observed = payload.pop("payload_sha256", None)
    if observed != OPENAXES_EXTERNAL_NATIVE_V3_ERRATUM4_SUPPLEMENT_PAYLOAD_SHA256 or hashlib.sha256(_canonical_json(payload)).hexdigest() != observed:
        raise OpenAxesExternalNativeV3ConfigError("erratum4 supplement payload changed")
    extension = _mapping(value.get("validation_authority_extension"), name="supplement extension")
    if (
        extension.get("effective_erratum4_record_exact_keys_add")
        != ["v2_salvage_accounting_supplement_reference"]
        or extension.get("effective_erratum4_record_input_reference_order_insert_after_v2_salvage_preissue")
        != "v2_salvage_accounting_supplement_reference"
        or extension.get("fresh_process_success_count") != 10
        or extension.get("authority_only_validation_is_sufficient") is not False
    ):
        raise OpenAxesExternalNativeV3ConfigError("effective erratum4 extension changed")


def _read_regular(path: str | Path) -> bytes:
    source = Path(path)
    info = source.stat(follow_symlinks=False)
    if source.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise OpenAxesExternalNativeV3ConfigError(f"not a regular authority: {source}")
    raw = source.read_bytes()
    if len(raw) != info.st_size:
        raise OpenAxesExternalNativeV3ConfigError(f"short authority read: {source}")
    return raw


def _read_at(root_fd: int, relative: str) -> bytes:
    parts = relative.split("/")
    if not relative or any(part in {"", ".", ".."} for part in parts):
        raise OpenAxesExternalNativeV3ConfigError("unsafe authority path")
    current = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise OpenAxesExternalNativeV3ConfigError("authority is not regular")
            raw = b""
            while True:
                chunk = os.read(fd, 1 << 20)
                if not chunk:
                    break
                raw += chunk
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or len(raw) != before.st_size:
                raise OpenAxesExternalNativeV3ConfigError("authority changed while read")
            return raw
        finally:
            os.close(fd)
    finally:
        os.close(current)


def _assemble(base_raw: bytes, erratum_raw: bytes, supplement_raw: bytes) -> dict[str, Any]:
    config = _parse_yaml(base_raw, expected_raw_sha256=OPENAXES_EXTERNAL_NATIVE_V3_CONFIG_RAW_SHA256, source="native-V3 config")
    validate_openaxes_external_native_v3_config(config)
    erratum = _parse_yaml(erratum_raw, expected_raw_sha256=OPENAXES_EXTERNAL_NATIVE_V3_ERRATUM4_RAW_SHA256, source="native-V3 erratum4")
    _validate_erratum4(erratum)
    if hashlib.sha256(supplement_raw).hexdigest() != OPENAXES_EXTERNAL_NATIVE_V3_ERRATUM4_SUPPLEMENT_RAW_SHA256:
        raise OpenAxesExternalNativeV3ConfigError("erratum4 supplement raw SHA-256 changed")
    try:
        supplement = json.loads(supplement_raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OpenAxesExternalNativeV3ConfigError("cannot parse erratum4 supplement") from exc
    _validate_supplement(_mapping(supplement, name="erratum4 supplement"))
    config["_erratum4"] = erratum
    config["_erratum4_accounting_supplement"] = supplement
    config["_effective_salvage_record_keys"] = OPENAXES_NATIVE_V3_SALVAGE_RECORD_KEYS
    config["_effective_salvage_input_reference_order"] = OPENAXES_NATIVE_V3_SALVAGE_INPUT_REFERENCE_ORDER
    return config


def load_openaxes_external_native_v3_config(
    path: str | Path = "configs/openaxes_external_native_v3_allmask_dmome.yaml",
    *,
    erratum4_path: str | Path = "configs/openaxes_external_native_v3_erratum4_salvage_validation.yaml",
    supplement_path: str | Path = "receipts/openaxes_external_native_v3_preregistration_erratum4_accounting_supplement_preissue.json",
) -> dict[str, Any]:
    return _assemble(_read_regular(path), _read_regular(erratum4_path), _read_regular(supplement_path))


def load_openaxes_external_native_v3_config_at(workspace_fd: int) -> dict[str, Any]:
    return _assemble(
        _read_at(workspace_fd, OPENAXES_EXTERNAL_NATIVE_V3_CONFIG_PATH),
        _read_at(workspace_fd, OPENAXES_EXTERNAL_NATIVE_V3_ERRATUM4_PATH),
        _read_at(workspace_fd, OPENAXES_EXTERNAL_NATIVE_V3_ERRATUM4_SUPPLEMENT_PATH),
    )


def candidate_schedule(config: Mapping[str, Any], candidate_id: str) -> CandidateSchedule:
    validate_openaxes_external_native_v3_config(config)
    if candidate_id not in OPENAXES_NATIVE_V3_CANDIDATE_IDS:
        raise OpenAxesExternalNativeV3ConfigError("unknown native-V3 candidate")
    row = _mapping(config["candidates"][candidate_id], name="candidate schedule")
    stages = _mapping(row.get("stages"), name="candidate stages")
    return CandidateSchedule(
        candidate_id=candidate_id,
        architecture=str(row["architecture"]),
        independent_expert_updates=int(stages["independent_expert_updates"]),
        cooperative_updates=int(stages["cooperative_updates"]),
        objective=str(row["objective"]),
    )


def context_ordinal(fold_ordinal: int, seed_ordinal: int) -> int:
    if type(fold_ordinal) is not int or not 0 <= fold_ordinal < OPENAXES_NATIVE_V3_FOLD_COUNT:
        raise OpenAxesExternalNativeV3ConfigError("fold ordinal out of range")
    if type(seed_ordinal) is not int or not 0 <= seed_ordinal < len(OPENAXES_NATIVE_V3_SEEDS):
        raise OpenAxesExternalNativeV3ConfigError("seed ordinal out of range")
    return fold_ordinal * len(OPENAXES_NATIVE_V3_SEEDS) + seed_ordinal


def task_ordinal(fold_ordinal: int, seed_ordinal: int, candidate_ordinal: int) -> int:
    if type(candidate_ordinal) is not int or not 0 <= candidate_ordinal < len(OPENAXES_NATIVE_V3_CANDIDATE_IDS):
        raise OpenAxesExternalNativeV3ConfigError("candidate ordinal out of range")
    return context_ordinal(fold_ordinal, seed_ordinal) * len(OPENAXES_NATIVE_V3_CANDIDATE_IDS) + candidate_ordinal


def slot_for_context(context: int) -> int:
    if type(context) is not int or not 0 <= context < OPENAXES_NATIVE_V3_CONTEXT_COUNT:
        raise OpenAxesExternalNativeV3ConfigError("context ordinal out of range")
    return context % OPENAXES_NATIVE_V3_LOGICAL_SLOT_COUNT


def gpu_for_context(context: int) -> int:
    return slot_for_context(context) % OPENAXES_NATIVE_V3_GPU_COUNT

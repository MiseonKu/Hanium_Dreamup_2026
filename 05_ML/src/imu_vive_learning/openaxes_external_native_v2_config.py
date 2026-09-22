"""Frozen, outcome-blind configuration boundary for OpenAXES native V2.

The production controller receives an inherited workspace directory descriptor.
Accordingly, this module offers an ``openat`` loader in addition to the small
pathname loader used by isolated source tests.  Neither loader follows a
symbolic-link component, and the semantic digest excludes only loader-private
keys whose names begin with an underscore.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import yaml


OPENAXES_EXTERNAL_NATIVE_V2_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2"
)
OPENAXES_EXTERNAL_NATIVE_V2_EXECUTION_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.execution.v1"
)
OPENAXES_EXTERNAL_NATIVE_V2_CONFIG_SHA256 = (
    "832a7618f0b7ce94c2813d4b5659153bb88d878d1e67f6ccba3f8f1e3555d59a"
)
OPENAXES_EXTERNAL_NATIVE_V2_CONFIG_PATH = (
    "09_SO101_IMU_VIVE_LEARNING/configs/openaxes_external_native_v2.yaml"
)
OPENAXES_EXTERNAL_NATIVE_V2_PROJECT_CONFIG_PATH = (
    "configs/openaxes_external_native_v2.yaml"
)
OPENAXES_EXTERNAL_NATIVE_V2_PARENT_CONTRACT = (
    "openaxes.external.model_benchmark.v1.terminal_for_native_v2.v1"
)
OPENAXES_EXTERNAL_NATIVE_V2_COMPARATOR_CONTRACT = (
    "openaxes.external.native_sensor_experts.v2.masked_erm_parent_bundle.v1"
)

OPENAXES_NATIVE_V2_SENSOR_GROUPS = (
    "humerus",
    "radius",
    "carpus",
    "digitus",
)
OPENAXES_NATIVE_V2_CHANNELS = (
    "accel_adxl355_x_mps2",
    "accel_adxl355_y_mps2",
    "accel_adxl355_z_mps2",
    "accel_bmi160_x_mps2",
    "accel_bmi160_y_mps2",
    "accel_bmi160_z_mps2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
)
OPENAXES_NATIVE_V2_CONTINUOUS_FEATURES = tuple(
    f"{sensor}_{channel}"
    for sensor in OPENAXES_NATIVE_V2_SENSOR_GROUPS
    for channel in OPENAXES_NATIVE_V2_CHANNELS
)
OPENAXES_NATIVE_V2_AVAILABILITY_COLUMNS = tuple(
    f"availability_{sensor}" for sensor in OPENAXES_NATIVE_V2_SENSOR_GROUPS
)
OPENAXES_NATIVE_V2_TARGET_COLUMNS = ("tcp_x_m", "tcp_y_m", "tcp_z_m")
OPENAXES_NATIVE_V2_METADATA_COLUMNS = (
    "row_uid",
    "scenario_id",
    "sample_index",
    "aligned_time_us",
)
OPENAXES_NATIVE_V2_VIEW_IDS = (
    "full",
    "outage_humerus",
    "outage_radius",
    "outage_carpus",
    "outage_digitus",
)
OPENAXES_NATIVE_V2_OUTAGE_VIEW_IDS = OPENAXES_NATIVE_V2_VIEW_IDS[1:]
OPENAXES_NATIVE_V2_CANDIDATE_IDS = (
    "dense_6979__scenario_balanced_masked_erm",
    "sensor_experts_6979__scenario_balanced_masked_erm",
    "sensor_experts_6979__scenario_balanced_cagrad",
)
OPENAXES_NATIVE_V2_SEEDS = (1701, 2603, 3907)
OPENAXES_NATIVE_V2_FOLD_FAMILIES = (
    "scenario_loso",
    "speed_pair_out",
    "shape_out",
)
OPENAXES_NATIVE_V2_FAMILY_COUNTS = {
    "scenario_loso": 8,
    "speed_pair_out": 4,
    "shape_out": 2,
}
OPENAXES_NATIVE_V2_PHASES = (
    "execution.started",
    "training.completed",
    "outer_prediction.completed",
    "outer_eval.started",
    "outer_eval.completed",
    "bundle.completed",
    "result.completed",
)
OPENAXES_NATIVE_V2_SHIFT_MS = (-50, -20, -10, 10, 20, 50)
OPENAXES_NATIVE_V2_MASK_LAW = (0.25, 0.1875, 0.1875, 0.1875, 0.1875)
OPENAXES_NATIVE_V2_FIT_COUNTS_BY_SLOT = (
    12,
    12,
    12,
    12,
    12,
    12,
    9,
    9,
    9,
    9,
    9,
    9,
)
OPENAXES_NATIVE_V2_FIT_COUNTS_BY_GPU = (33, 33, 30, 30)
OPENAXES_NATIVE_V2_GPU_INVENTORY = (
    {
        "physical_index": 0,
        "uuid": "GPU-462585f2-b4cc-3864-e27c-cbaae4ab6c9a",
        "pci_bus_id": "00000000:01:00.0",
        "name": "NVIDIA CMP 170HX",
    },
    {
        "physical_index": 1,
        "uuid": "GPU-517dfd2b-2ade-0b3f-70d7-0ea2a78f868b",
        "pci_bus_id": "00000000:2D:00.0",
        "name": "NVIDIA CMP 170HX",
    },
    {
        "physical_index": 2,
        "uuid": "GPU-b3db5781-61ae-33bb-63a0-64b14dd9ae6b",
        "pci_bus_id": "00000000:41:00.0",
        "name": "NVIDIA CMP 170HX",
    },
    {
        "physical_index": 3,
        "uuid": "GPU-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb",
        "pci_bus_id": "00000000:61:00.0",
        "name": "NVIDIA CMP 170HX",
    },
)
OPENAXES_NATIVE_V2_SOURCE_CLOSURE = (
    "configs/openaxes_external_native_v2.yaml",
    "docs/OPENAXES_EXTERNAL_NATIVE_V2_CONTRACT.md",
    "docs/OPENAXES_EXTERNAL_NATIVE_V2_PARENT_RECEIPT_CONTRACT.md",
    "requirements-eval.lock",
    "requirements-gpu.lock",
    "src/imu_vive_learning/__init__.py",
    "src/imu_vive_learning/io_utils.py",
    "src/imu_vive_learning/openaxes_external_native_v2_config.py",
    "src/imu_vive_learning/openaxes_external_native_v2_experiment.py",
    "src/imu_vive_learning/openaxes_external_native_v2_parent_receipt.py",
    "src/imu_vive_learning/openaxes_external_native_v2_training.py",
    "tests/test_openaxes_external_native_v2_config.py",
    "tests/test_openaxes_external_native_v2_experiment.py",
    "tests/test_openaxes_external_native_v2_parent_receipt.py",
    "tests/test_openaxes_external_native_v2_training.py",
)


class OpenAxesExternalNativeV2ConfigError(ValueError):
    """Raised when the frozen native-V2 configuration is not exact."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader whose mapping constructor rejects duplicate keys."""

    def construct_mapping(
        self, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[Any, Any]:
        self.flatten_mapping(node)
        result: dict[Any, Any] = {}
        seen: list[Any] = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = any(
                    type(previous) is type(key) and previous == key
                    for previous in seen
                )
            except Exception as exc:
                raise OpenAxesExternalNativeV2ConfigError(
                    "native-V2 YAML has an unsupported mapping key"
                ) from exc
            if duplicate:
                raise OpenAxesExternalNativeV2ConfigError(
                    f"duplicate native-V2 YAML key: {key!r}"
                )
            result[key] = self.construct_object(value_node, deep=deep)
            seen.append(key)
        return result


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OpenAxesExternalNativeV2ConfigError(
            "native-V2 configuration is not finite canonical JSON"
        ) from exc


def _public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping) or any(
        type(key) is not str for key in config
    ):
        raise OpenAxesExternalNativeV2ConfigError(
            "native-V2 configuration root must be a string-keyed mapping"
        )
    return {
        key: value
        for key, value in config.items()
        if type(key) is str and not key.startswith("_")
    }


def openaxes_external_native_v2_config_sha256(
    config: Mapping[str, Any],
) -> str:
    """Return the frozen semantic digest (loader-private keys excluded)."""

    return hashlib.sha256(_canonical_json(_public_config(config))).hexdigest()


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise OpenAxesExternalNativeV2ConfigError(
            f"{name} must be a string-keyed mapping"
        )
    return value


def _require_exact_list(
    value: Any, expected: Sequence[Any], *, name: str
) -> tuple[Any, ...]:
    if type(value) is not list or tuple(value) != tuple(expected):
        raise OpenAxesExternalNativeV2ConfigError(f"{name} order changed")
    return tuple(value)


def _require_exact_int(value: Any, expected: int, *, name: str) -> None:
    if type(value) is not int or value != expected:
        raise OpenAxesExternalNativeV2ConfigError(
            f"{name} must be exactly {expected}"
        )


def _require_exact_float(value: Any, expected: float, *, name: str) -> None:
    if type(value) is not float or not math.isfinite(value) or value != expected:
        raise OpenAxesExternalNativeV2ConfigError(
            f"{name} must be exactly finite {expected}"
        )


def validate_openaxes_external_native_v2_config(
    config: Mapping[str, Any],
) -> None:
    """Fail closed unless *config* is the exact preregistered semantic object."""

    public = _public_config(config)
    expected_root = {
        "contract",
        "authority_erratum1",
        "authority_erratum2",
        "lineage",
        "freeze_sequence",
        "git_preregistration",
        "parent_receipt_auditor",
        "v1_terminal_receipt",
        "schema",
        "preprocessing",
        "architectures",
        "initialization_and_determinism",
        "row_order_and_reductions",
        "environment_and_scheduler",
        "candidates",
        "training_risk",
        "cagrad",
        "optimization",
        "checkpoint",
        "folds_and_fits",
        "outer_boundary",
        "execution_state_machine",
        "runtime_closed_world",
        "outer_label_capability_acceptance",
        "aggregation",
        "primary_gates",
        "report_only_analyses",
        "exact_accounting",
        "source_closure",
        "execution_preconditions",
        "claims_and_limitations",
    }
    if set(public) != expected_root:
        raise OpenAxesExternalNativeV2ConfigError(
            "native-V2 root keys changed"
        )

    contract = _require_mapping(public["contract"], name="contract")
    if contract.get("name") != OPENAXES_EXTERNAL_NATIVE_V2_CONTRACT:
        raise OpenAxesExternalNativeV2ConfigError("native-V2 contract name changed")
    if contract.get("version") != 2 or type(contract.get("version")) is not int:
        raise OpenAxesExternalNativeV2ConfigError("native-V2 version changed")
    if contract.get("challenge_access_allowed") is not False:
        raise OpenAxesExternalNativeV2ConfigError("challenge boundary changed")
    if contract.get("so101_data_or_weights_allowed") is not False:
        raise OpenAxesExternalNativeV2ConfigError("SO101 boundary changed")

    schema = _require_mapping(public["schema"], name="schema")
    _require_exact_list(
        schema.get("sensor_groups_order"),
        OPENAXES_NATIVE_V2_SENSOR_GROUPS,
        name="sensor group order",
    )
    _require_exact_list(
        schema.get("continuous_channels_per_sensor_order"),
        OPENAXES_NATIVE_V2_CHANNELS,
        name="continuous channel order",
    )
    _require_exact_list(
        schema.get("availability_order"),
        OPENAXES_NATIVE_V2_AVAILABILITY_COLUMNS,
        name="availability order",
    )
    _require_exact_list(
        schema.get("target_columns_order"),
        OPENAXES_NATIVE_V2_TARGET_COLUMNS,
        name="target order",
    )
    _require_exact_int(schema.get("continuous_feature_count"), 36, name="features")
    _require_exact_int(schema.get("dense_input_count"), 40, name="dense input")

    architectures = _require_mapping(public["architectures"], name="architectures")
    dense = _require_mapping(architectures.get("dense_6979"), name="dense architecture")
    experts = _require_mapping(
        architectures.get("sensor_experts_6979"), name="expert architecture"
    )
    _require_exact_int(dense.get("trainable_total", dense.get("layer_parameter_arithmetic", {}).get("trainable_total")), 6979, name="dense parameters")
    _require_exact_int(experts.get("trainable_total"), 6979, name="expert parameters")

    scheduler = _require_mapping(
        public["environment_and_scheduler"], name="environment and scheduler"
    )
    triplets = _require_mapping(
        scheduler.get("context_triplet_assignment"), name="triplet assignment"
    )
    _require_exact_int(triplets.get("context_count"), 42, name="context count")
    _require_exact_list(
        triplets.get("within_triplet_candidate_order"),
        OPENAXES_NATIVE_V2_CANDIDATE_IDS,
        name="triplet candidates",
    )
    _require_exact_list(
        triplets.get("per_slot_fit_counts"),
        OPENAXES_NATIVE_V2_FIT_COUNTS_BY_SLOT,
        name="slot fit counts",
    )
    _require_exact_list(
        triplets.get("per_physical_gpu_fit_counts_in_inventory_order"),
        OPENAXES_NATIVE_V2_FIT_COUNTS_BY_GPU,
        name="GPU fit counts",
    )

    candidates = _require_mapping(public["candidates"], name="candidates")
    _require_exact_list(
        candidates.get("ordered_ids"),
        OPENAXES_NATIVE_V2_CANDIDATE_IDS,
        name="candidate order",
    )
    folds = _require_mapping(public["folds_and_fits"], name="folds and fits")
    _require_exact_list(
        folds.get("ordered_families"),
        OPENAXES_NATIVE_V2_FOLD_FAMILIES,
        name="fold family order",
    )
    _require_exact_list(folds.get("seeds"), OPENAXES_NATIVE_V2_SEEDS, name="seeds")
    for key, expected in (
        ("candidate_count", 3),
        ("total_outer_folds", 14),
        ("fits_per_candidate", 42),
        ("total_scientific_fits", 126),
    ):
        _require_exact_int(folds.get(key), expected, name=f"folds.{key}")
    if 3 * 14 * 3 != folds.get("total_scientific_fits"):
        raise OpenAxesExternalNativeV2ConfigError("126-fit grid is not closed")

    optimization = _require_mapping(public["optimization"], name="optimization")
    if optimization.get("optimizer") != "torch.optim.AdamW":
        raise OpenAxesExternalNativeV2ConfigError("optimizer changed")
    _require_exact_float(optimization.get("learning_rate"), 0.001, name="learning rate")
    _require_exact_float(optimization.get("weight_decay"), 0.001, name="weight decay")
    _require_exact_int(
        optimization.get("full_training_fold_updates_per_fit"),
        400,
        name="update count",
    )
    cagrad = _require_mapping(public["cagrad"], name="CAGrad")
    _require_exact_float(cagrad.get("coefficient_c"), 0.4, name="CAGrad c")
    _require_exact_list(
        cagrad.get("outage_task_order"),
        OPENAXES_NATIVE_V2_OUTAGE_VIEW_IDS,
        name="CAGrad task order",
    )

    state = _require_mapping(
        public["execution_state_machine"], name="execution state machine"
    )
    _require_exact_list(state.get("phase_order"), OPENAXES_NATIVE_V2_PHASES, name="phase order")
    task_spec = _require_mapping(state.get("task_spec"), name="task spec")
    if task_spec.get("task_ordinal_formula") != "context_ordinal_times_3_plus_candidate_ordinal":
        raise OpenAxesExternalNativeV2ConfigError("task ordinal formula changed")

    source = _require_mapping(public["source_closure"], name="source closure")
    _require_exact_list(
        source.get("future_v2_execution_closure_files_exact"),
        OPENAXES_NATIVE_V2_SOURCE_CLOSURE,
        name="source closure",
    )
    _require_exact_int(
        source.get("future_v2_execution_closure_file_count"),
        15,
        name="source closure count",
    )

    digest = openaxes_external_native_v2_config_sha256(config)
    if digest != OPENAXES_EXTERNAL_NATIVE_V2_CONFIG_SHA256:
        raise OpenAxesExternalNativeV2ConfigError(
            "native-V2 canonical SHA-256 changed"
        )


def _parse_yaml(raw: bytes, *, source_name: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        value = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise OpenAxesExternalNativeV2ConfigError(
            f"cannot parse native-V2 config: {source_name}"
        ) from exc
    if not isinstance(value, Mapping):
        raise OpenAxesExternalNativeV2ConfigError(
            "native-V2 configuration root must be a mapping"
        )
    result = deepcopy(dict(value))
    validate_openaxes_external_native_v2_config(result)
    return result


def _safe_parts(relative: str) -> tuple[str, ...]:
    if (
        type(relative) is not str
        or not relative
        or relative.startswith("/")
        or relative.endswith("/")
        or "\\" in relative
        or "//" in relative
    ):
        raise OpenAxesExternalNativeV2ConfigError("config path is not canonical relative POSIX")
    parts = tuple(relative.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise OpenAxesExternalNativeV2ConfigError("config path has an unsafe component")
    return parts


def _read_at(root_fd: int, relative: str) -> bytes:
    if type(root_fd) is not int or root_fd < 0:
        raise OpenAxesExternalNativeV2ConfigError("config root descriptor is invalid")
    parts = _safe_parts(relative)
    current = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            following = os.open(component, 196608, dir_fd=current)
            info = os.fstat(following)
            if not stat.S_ISDIR(info.st_mode):
                os.close(following)
                raise OpenAxesExternalNativeV2ConfigError(
                    "config path component is not a directory"
                )
            os.close(current)
            current = following
        descriptor = os.open(parts[-1], 131072, dir_fd=current)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise OpenAxesExternalNativeV2ConfigError(
                    "config path is not a regular file"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if identity_before != identity_after:
                raise OpenAxesExternalNativeV2ConfigError(
                    "config changed while read"
                )
            raw = b"".join(chunks)
            if len(raw) != before.st_size:
                raise OpenAxesExternalNativeV2ConfigError(
                    "config read was incomplete"
                )
            return raw
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise OpenAxesExternalNativeV2ConfigError(
            f"cannot open native-V2 config: {relative}"
        ) from exc
    finally:
        os.close(current)


def load_openaxes_external_native_v2_config_at(
    workspace_fd: int,
    relative: str = OPENAXES_EXTERNAL_NATIVE_V2_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the production config below an inherited workspace descriptor."""

    result = _parse_yaml(_read_at(workspace_fd, relative), source_name=relative)
    result["_config_relative_path"] = relative
    return result


def load_openaxes_external_native_v2_config(path: str | Path) -> dict[str, Any]:
    """Pathname loader reserved for source tests and non-production inspection."""

    source = Path(path)
    try:
        info = source.stat(follow_symlinks=False)
        if source.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise OpenAxesExternalNativeV2ConfigError(
                "native-V2 config is not a regular nonsymlink file"
            )
        raw = source.read_bytes()
    except OSError as exc:
        raise OpenAxesExternalNativeV2ConfigError(
            f"cannot load native-V2 config: {source}"
        ) from exc
    result = _parse_yaml(raw, source_name=str(source))
    result["_config_path"] = str(source)
    return result

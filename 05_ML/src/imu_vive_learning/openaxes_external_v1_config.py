"""Frozen configuration boundary for the OpenAXES external-v1 preparation.

This configuration describes an auxiliary public-data preparation only.  It
does not authorize model fitting, selection, challenge-label access, or any
combination with the SO101/Vive table.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml


OPENAXES_EXTERNAL_V1_CONTRACT = "openaxes.external.preparation.v1"
OPENAXES_EXTERNAL_V1_CONFIG_SHA256 = (
    "05c3f44b58a16cebb8c0cef1338a42ae55a0fd84dd9f87db09196a7b740b32c6"
)
OPENAXES_SENSITIVITY_SIGNED_SHIFTS_MS = (-50, -20, -10, 10, 20, 50)
OPENAXES_SCENARIOS = (
    "circle-100",
    "circle-150",
    "circle-200",
    "circle-250",
    "triangle-100",
    "triangle-150",
    "triangle-200",
    "triangle-250",
)
OPENAXES_SEGMENTS = ("humerus", "radius", "carpus", "digitus")


class OpenAxesExternalV1ConfigError(ValueError):
    """Raised when the frozen external-v1 preparation contract drifts."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    seen: list[Any] = []
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if any(type(old) is type(key) and old == key for old in seen):
                raise OpenAxesExternalV1ConfigError(
                    f"duplicate OpenAXES config key: {key!r}"
                )
            result[key] = loader.construct_object(value_node, deep=deep)
        except TypeError as exc:
            raise OpenAxesExternalV1ConfigError(
                "OpenAXES config contains an unsupported mapping key"
            ) from exc
        seen.append(key)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OpenAxesExternalV1ConfigError(
            "OpenAXES config is not finite canonical JSON"
        ) from exc


def openaxes_config_sha256(config: Mapping[str, Any]) -> str:
    public = {key: value for key, value in config.items() if key != "_config_path"}
    return hashlib.sha256(_canonical_json_bytes(public)).hexdigest()


def _exact_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise OpenAxesExternalV1ConfigError(f"{name} must be an exact integer")
    return value


def _exact_float(value: Any, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise OpenAxesExternalV1ConfigError(f"{name} must be a finite exact float")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise OpenAxesExternalV1ConfigError(f"{name} must be a string-keyed mapping")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], *, name: str) -> None:
    if set(value) != keys:
        raise OpenAxesExternalV1ConfigError(f"{name} keys changed")


def scenario_configs(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or not all(
        isinstance(item, Mapping) for item in scenarios
    ):
        raise OpenAxesExternalV1ConfigError("scenarios must be an ordered mapping list")
    return tuple(scenarios)


def model_feature_columns(config: Mapping[str, Any]) -> tuple[str, ...]:
    schema = _mapping(config.get("schema"), name="schema")
    segments = tuple(schema.get("segments", ()))
    channels = tuple(schema.get("sensor_channels", ()))
    if segments != OPENAXES_SEGMENTS or len(channels) != 9:
        raise OpenAxesExternalV1ConfigError("OpenAXES 4x9 feature layout changed")
    return tuple(f"{segment}_{channel}" for segment in segments for channel in channels)


def validate_openaxes_config(config: Mapping[str, Any]) -> None:
    """Validate the canonical, preparation-only OpenAXES external-v1 config."""

    public = {key: value for key, value in config.items() if key != "_config_path"}
    _exact_keys(
        public,
        {
            "contract",
            "source",
            "schema",
            "preprocessing",
            "alignment",
            "scenarios",
            "folds",
            "limitations",
        },
        name="root",
    )
    if openaxes_config_sha256(config) != OPENAXES_EXTERNAL_V1_CONFIG_SHA256:
        raise OpenAxesExternalV1ConfigError(
            "OpenAXES external-v1 config canonical SHA-256 changed"
        )

    contract = _mapping(config["contract"], name="contract")
    if contract.get("name") != OPENAXES_EXTERNAL_V1_CONTRACT:
        raise OpenAxesExternalV1ConfigError("OpenAXES contract name changed")
    required_false = (
        "independent_so101_confirmation",
        "so101_merge_allowed",
        "model_training_allowed",
        "model_selection_allowed",
        "challenge_access_allowed",
    )
    if contract.get("preparation_only") is not True or any(
        contract.get(key) is not False for key in required_false
    ):
        raise OpenAxesExternalV1ConfigError(
            "OpenAXES preparation-only or separation boundary weakened"
        )

    source = _mapping(config["source"], name="source")
    if source.get("expected_sha256") != (
        "78f31d1967842e693c2efb6ee0530e162cd509e4fa786745a485d502f2aa6da7"
    ):
        raise OpenAxesExternalV1ConfigError("official OpenAXES source hash changed")
    if source.get("filename") != "openaxes-example-robot-dataset.zip":
        raise OpenAxesExternalV1ConfigError("official OpenAXES filename changed")
    if _exact_int(source.get("expected_size_bytes"), name="source size") != 423_808_715:
        raise OpenAxesExternalV1ConfigError("official OpenAXES source size changed")
    if _exact_int(source.get("expected_zip_entries"), name="ZIP entries") != 97:
        raise OpenAxesExternalV1ConfigError("official OpenAXES ZIP count changed")
    if source.get("portal_license") != "CC-BY-SA-3.0" or source.get(
        "member_data_license"
    ) != "CC-BY-SA-4.0":
        raise OpenAxesExternalV1ConfigError("known OpenAXES license conflict changed")
    if (
        source.get("license_conflict_expected") is not True
        or source.get("redistribution_allowed") is not False
        or source.get("attribution_required") is not True
    ):
        raise OpenAxesExternalV1ConfigError("OpenAXES license safety boundary weakened")

    schema = _mapping(config["schema"], name="schema")
    if tuple(schema.get("segments", ())) != OPENAXES_SEGMENTS:
        raise OpenAxesExternalV1ConfigError("OpenAXES segment order changed")
    if len(model_feature_columns(config)) != 36:
        raise OpenAxesExternalV1ConfigError("OpenAXES model feature count is not 36")
    targets = tuple(schema.get("target_columns", ()))
    if targets != ("tcp_x_m", "tcp_y_m", "tcp_z_m"):
        raise OpenAxesExternalV1ConfigError("OpenAXES TCP target schema changed")
    forbidden = tuple(schema.get("forbidden_model_inputs", ()))
    if set(model_feature_columns(config)).intersection(forbidden):
        raise OpenAxesExternalV1ConfigError("allowed and forbidden features overlap")
    required_forbidden = {
        "scenario_id",
        "aligned_time_us",
        "shape",
        "nominal_speed_mm_s",
        "MAC",
        "IMU",
        "TIME",
        "INDEX",
        "timestamp",
        "tcp_x",
        "tcp_y",
        "tcp_z",
        "tcp_x_m",
        "tcp_y_m",
        "tcp_z_m",
        "normalized_tcp_x",
        "normalized_tcp_y",
        "normalized_tcp_z",
        "normalized_tcp_x_m",
        "normalized_tcp_y_m",
        "normalized_tcp_z_m",
        "tcp_x_normalized",
        "tcp_y_normalized",
        "tcp_z_normalized",
        "q0",
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "tcp_dx",
        "tcp_dy",
        "tcp_dz",
    }
    if not required_forbidden.issubset(forbidden):
        raise OpenAxesExternalV1ConfigError("forbidden model-input boundary weakened")

    preprocessing = _mapping(config["preprocessing"], name="preprocessing")
    if _exact_int(preprocessing.get("grid_hz"), name="grid_hz") != 100:
        raise OpenAxesExternalV1ConfigError("OpenAXES grid must remain 100 Hz")
    if (
        preprocessing.get("active_trace_only") is not True
        or preprocessing.get("allow_extrapolation") is not False
        or preprocessing.get("gyro_correction")
        != "subtract_quiet_prefix_axis_mean"
    ):
        raise OpenAxesExternalV1ConfigError("OpenAXES preprocessing boundary weakened")

    alignment = _mapping(config["alignment"], name="alignment")
    sensitivity = _mapping(
        alignment.get("sensitivity_recipe"), name="alignment sensitivity recipe"
    )
    if (
        alignment.get("label_assisted") is not True
        or alignment.get("fixed_before_model_results") is not True
        or alignment.get("per_model_or_prediction_realign_allowed") is not False
    ):
        raise OpenAxesExternalV1ConfigError("OpenAXES alignment disclosure changed")
    expected_sensitivity = {
        "stage": "downstream_evaluation_required",
        "preparation_executes_recipe": False,
        "signed_target_shifts_ms": list(OPENAXES_SENSITIVITY_SIGNED_SHIFTS_MS),
        "zero_shift_baseline_required": True,
        "sign_convention": (
            "target_lookup_time_us_equals_base_aligned_time_us_plus_"
            "shift_ms_times_1000"
        ),
        "positive_shift_meaning": (
            "sample_target_later_than_the_base_prediction_time"
        ),
        "no_extrapolation": True,
        "edge_policy": "drop_base_rows_without_shifted_target_support",
        "pairing_key": "row_uid",
        "same_base_uid_pairing_required": True,
        "reporting_granularity": (
            "per_signed_shift_per_outer_fold_and_family_summary"
        ),
        "reporting_fold_families": [
            "scenario_loso",
            "speed_pair_out",
            "shape_out",
        ],
        "primary_scenario_loso_folds": 8,
        "stress_reporting_required": True,
        "required_metrics": [
            "rmse3d_m",
            "robust_J_m",
            "delta_rmse3d_m_shift_minus_zero",
            "delta_robust_J_m_shift_minus_zero",
        ],
        "metric_definitions": {
            "rmse3d_m": "sqrt_mean_squared_euclidean_tcp_xyz_error_in_metres",
            "robust_J_m": (
                "arithmetic_mean_of_A_mean_context_WS_worst_segment_and_"
                "WM_worst_mask_rmse3d_m"
            ),
            "shift_delta": (
                "signed_shift_metric_minus_zero_shift_metric_on_same_base_"
                "uid_pairs"
            ),
        },
        "reporting_methods": ["frozen_comparator", "frozen_objective"],
        "paired_method_delta_required": True,
        "paired_method_delta_sign": (
            "frozen_objective_minus_frozen_comparator"
        ),
        "model_selection_use_allowed": False,
        "claim_gate_use_allowed": False,
    }
    if dict(sensitivity) != expected_sensitivity:
        raise OpenAxesExternalV1ConfigError(
            "OpenAXES signed alignment-sensitivity recipe changed"
        )

    scenarios = scenario_configs(config)
    if tuple(item.get("id") for item in scenarios) != OPENAXES_SCENARIOS:
        raise OpenAxesExternalV1ConfigError("OpenAXES scenario order changed")
    total_robot = total_common = total_unmatched = 0
    for item in scenarios:
        scenario_id = item.get("id")
        robot_rows = _exact_int(item.get("robot_rows"), name=f"{scenario_id} robot_rows")
        common_rows = _exact_int(
            item.get("common_aligned_robot_rows"),
            name=f"{scenario_id} common rows",
        )
        head = _exact_int(
            item.get("dropped_robot_head_rows"), name=f"{scenario_id} head rows"
        )
        tail = _exact_int(
            item.get("dropped_robot_tail_rows"), name=f"{scenario_id} tail rows"
        )
        if head + common_rows + tail != robot_rows:
            raise OpenAxesExternalV1ConfigError(
                f"{scenario_id} common-overlap accounting is not closed"
            )
        imu_rows = _mapping(item.get("imu_rows"), name=f"{scenario_id} imu_rows")
        if tuple(imu_rows) != OPENAXES_SEGMENTS or any(
            type(value) is not int or value <= 0 for value in imu_rows.values()
        ):
            raise OpenAxesExternalV1ConfigError(
                f"{scenario_id} IMU row contract changed"
            )
        active = _mapping(item.get("active_trace"), name=f"{scenario_id} active trace")
        start = _exact_int(
            active.get("start_robot_row_inclusive"), name=f"{scenario_id} active start"
        )
        end = _exact_int(
            active.get("end_robot_row_inclusive"), name=f"{scenario_id} active end"
        )
        if not 0 <= start < end < robot_rows:
            raise OpenAxesExternalV1ConfigError(
                f"{scenario_id} active trace bounds are invalid"
            )
        geometry = _mapping(active.get("geometry"), name=f"{scenario_id} geometry")
        if geometry.get("kind") not in {"circle_yz", "triangle_yz"}:
            raise OpenAxesExternalV1ConfigError(f"{scenario_id} geometry kind changed")
        total_robot += robot_rows
        total_common += common_rows
        total_unmatched += head + tail
    if (total_robot, total_common, total_unmatched) != (39_793, 39_301, 492):
        raise OpenAxesExternalV1ConfigError(
            "OpenAXES global 39,793/39,301/492 accounting changed"
        )

    folds = _mapping(config["folds"], name="folds")
    if (
        folds.get("total_declared_outer_folds") != 14
        or folds.get("outer_test_fitting_allowed") is not False
        or folds.get("row_random_split_allowed") is not False
        or folds.get("repeated_loops_are_independent_folds") is not False
    ):
        raise OpenAxesExternalV1ConfigError("OpenAXES blocked-fold boundary changed")
    families = folds.get("families")
    if not isinstance(families, list) or [item.get("expected_folds") for item in families] != [8, 4, 2]:
        raise OpenAxesExternalV1ConfigError("OpenAXES 8+4+2 fold families changed")

    limitations = _mapping(config["limitations"], name="limitations")
    if (
        limitations.get("physical_recording_sessions") != 1
        or limitations.get("effective_scenario_clusters") != 8
        or limitations.get("same_installation") is not True
        or limitations.get("observed_real_sensor_outage") is not False
        or limitations.get("simulated_sensor_dropout_only") is not True
        or limitations.get("confirmatory_claim_allowed") is not False
        or limitations.get("publisher_reconstructed_tcp_allowed") is not False
        or limitations.get("held_out_full_recording_clock_calibration") is not True
        or limitations.get("offline_transductive_preparation") is not True
        or limitations.get("acausal_preparation") is not True
        or limitations.get("causal_or_online_claim_allowed") is not False
    ):
        raise OpenAxesExternalV1ConfigError("OpenAXES limitation disclosure weakened")


def load_openaxes_config(path: str | Path) -> dict[str, Any]:
    """Load the one canonical external-v1 YAML with duplicate-key rejection."""

    config_path = Path(path).resolve()
    try:
        payload = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise OpenAxesExternalV1ConfigError(
            f"cannot load OpenAXES config: {config_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise OpenAxesExternalV1ConfigError("OpenAXES config root must be a mapping")
    config = deepcopy(dict(payload))
    config["_config_path"] = str(config_path)
    validate_openaxes_config(config)
    return config

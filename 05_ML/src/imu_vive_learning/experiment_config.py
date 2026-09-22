"""Frozen configuration loader for endpoint-regression experiment v1."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .config import canonical_json_bytes


EXPERIMENT_CONTRACT = "alphaf.endpoint_regression.experiment.v1"
PARENT_CONTRACT = "alphaf.ml_preparation.v1"
PINNED_RELEASE_SHA256 = (
    "94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930"
)
PINNED_EXPERIMENT_CONFIG_SHA256 = (
    "1ae48cdd2508d0bf9daf3b0092598b49e3aab1951ce5054d3c5b3e6f85d4f5c3"
)


class ExperimentConfigError(ValueError):
    """Raised when the pre-registered v1 experiment config has drifted."""


def _mapping(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ExperimentConfigError(f"missing mapping section: {name}")
    return value


def public_experiment_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return only stable, serialisable fields included in the config hash."""

    return deepcopy({key: value for key, value in config.items() if not key.startswith("_")})


def experiment_config_sha256(config: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(public_experiment_config(config))).hexdigest()


def validate_experiment_config(config: dict[str, Any]) -> None:
    """Fail closed if a scientifically material frozen v1 setting changes."""

    contract = _mapping(config, "contract")
    release = _mapping(config, "release")
    splits = _mapping(config, "splits")
    features = _mapping(config, "features")
    matrix = _mapping(config, "experiment_matrix")
    models = _mapping(config, "models")
    preprocessing = _mapping(config, "preprocessing")
    recipes = _mapping(config, "recipes")
    selection = _mapping(config, "selection")
    metrics = _mapping(config, "metrics")
    gate = _mapping(config, "augmentation_gate")
    uncertainty = _mapping(config, "uncertainty")
    final_model = _mapping(config, "final_model")
    runtime = _mapping(config, "runtime")

    if contract.get("name") != EXPERIMENT_CONTRACT:
        raise ExperimentConfigError(f"contract.name must be {EXPERIMENT_CONTRACT!r}")
    if contract.get("parent_contract") != PARENT_CONTRACT:
        raise ExperimentConfigError(f"contract.parent_contract must be {PARENT_CONTRACT!r}")
    if int(contract.get("base_seed", -1)) != 20260818:
        raise ExperimentConfigError("contract.base_seed must remain 20260818")
    if contract.get("research_status") != "exploratory_v1":
        raise ExperimentConfigError("experiment v1 must remain explicitly exploratory")
    if contract.get("label_frame") != "vive_world" or contract.get("label_unit") != "mm":
        raise ExperimentConfigError("labels must remain raw vive_world XYZ in millimetres")
    if contract.get("independent_test_available") is not False:
        raise ExperimentConfigError("v1 cannot claim an independent test set")

    if release.get("archive_sha256") != PINNED_RELEASE_SHA256:
        raise ExperimentConfigError("release.archive_sha256 differs from the pinned preparation ZIP")
    if release.get("expected_contract") != PARENT_CONTRACT:
        raise ExperimentConfigError("release.expected_contract changed")
    expected_counts = {
        "expected_canonical_rows": 7707,
        "expected_train_eligible_rows": 7702,
        "expected_strict_qc_rows": 7670,
    }
    for key, expected in expected_counts.items():
        if int(release.get(key, -1)) != expected:
            raise ExperimentConfigError(f"release.{key} must remain {expected}")

    if int(splits.get("expected_contexts", -1)) != 50:
        raise ExperimentConfigError("splits.expected_contexts must remain 50")
    if int(splits.get("expected_assignment_rows", -1)) != 385100:
        raise ExperimentConfigError("splits.expected_assignment_rows must remain 385100")
    schemes = _mapping(splits, "schemes")
    expected_groups = {
        "segment_loso": [36, 37, 38, 39, 40],
        "target_group": [0, 1, 2, 3, 4],
    }
    for scheme, groups in expected_groups.items():
        section = _mapping(schemes, scheme)
        if [int(item) for item in section.get("outer_groups", [])] != groups:
            raise ExperimentConfigError(f"{scheme} outer groups changed")
        if int(section.get("outer_contexts", -1)) != 5 or int(
            section.get("inner_contexts", -1)
        ) != 20:
            raise ExperimentConfigError(f"{scheme} must declare five outer and twenty inner contexts")

    if list(features.get("label_columns", [])) != [
        "vive_pos_raw_x_mm",
        "vive_pos_raw_y_mm",
        "vive_pos_raw_z_mm",
    ]:
        raise ExperimentConfigError("raw Vive label columns changed")
    if features.get("primary_views") != {
        "encoder": "encoder_only_v1",
        "four_imu": "four_imu_endpoint_v1",
        "fusion": "fusion_v1",
    }:
        raise ExperimentConfigError("primary feature aliases changed")

    required_matrix_sections = {
        "baseline",
        "sensor_comparison",
        "augmentation_comparison",
        "shortcut_diagnostics",
        "deployment_exclusions",
    }
    if set(matrix) != required_matrix_sections:
        raise ExperimentConfigError("experiment matrix sections changed")
    if set(models) != {"dummy_mean", "ridge", "extra_trees", "xgboost", "mlp"}:
        raise ExperimentConfigError("model family set changed")
    if list(models["ridge"].get("alpha", [])) != [0.1, 1.0, 10.0, 100.0]:
        raise ExperimentConfigError("Ridge grid changed")
    if int(models["extra_trees"].get("n_estimators", -1)) != 400:
        raise ExperimentConfigError("ExtraTrees tree count changed")
    if int(models["xgboost"].get("n_estimators", -1)) != 500:
        raise ExperimentConfigError("XGBoost tree count changed")
    if int(models["mlp"].get("ensemble_seed_count", -1)) != 3:
        raise ExperimentConfigError("MLP paired ensemble must contain three seeds")

    if preprocessing.get("fit_scope") != "real_train_only":
        raise ExperimentConfigError("preprocessing must fit on real train rows only")
    if preprocessing.get("target_scale") != "one_isotropic_scale":
        raise ExperimentConfigError("target scaling must preserve equal XYZ weighting")
    if set(recipes) != {
        "real_only",
        "replay",
        "local_interp_v2",
        "cmixup_offline_v1",
        "outer_refit_materialized_from_release",
    }:
        raise ExperimentConfigError("training recipe set changed")
    for name in ("replay", "local_interp_v2", "cmixup_offline_v1"):
        recipe = _mapping(recipes, name)
        if float(recipe.get("real_weight", -1)) != 0.5 or float(
            recipe.get("added_weight", -1)
        ) != 0.5:
            raise ExperimentConfigError(f"{name} weights must remain 0.5/0.5")
        if int(recipe.get("epoch_budget", -1)) != 200:
            raise ExperimentConfigError(f"{name} MLP epoch budget must remain 200")
    if int(_mapping(recipes, "real_only").get("epoch_budget", -1)) != 400:
        raise ExperimentConfigError("real-only MLP epoch budget must remain 400")

    if selection.get("score") != "macro_group_rmse_3d_mm":
        raise ExperimentConfigError("selection score must remain macro group RMSE_3D")
    if metrics.get("primary") != "rmse_3d_mm":
        raise ExperimentConfigError("primary metric must remain RMSE_3D")
    if metrics.get("legacy_axis_pooled_name") != "rmse_xyz_mm":
        raise ExperimentConfigError("legacy axis-pooled metric name changed")
    if float(gate.get("minimum_relative_rmse_improvement", -1)) != 0.10:
        raise ExperimentConfigError("augmentation relative gate changed")
    if float(gate.get("minimum_absolute_rmse_improvement_mm", -1)) != 3.0:
        raise ExperimentConfigError("augmentation absolute gate changed")
    if float(gate.get("maximum_relative_p95_degradation", -1)) != 0.05:
        raise ExperimentConfigError("augmentation P95 gate changed")
    if uncertainty.get("method") != "deterministic_multiway_bootstrap" or int(
        uncertainty.get("replicates", -1)
    ) != 10000:
        raise ExperimentConfigError("paired multiway bootstrap contract changed")
    if final_model.get("selection_scheme") != "segment_loso":
        raise ExperimentConfigError("final selection must use the segment nested contexts")
    if final_model.get("independent_test_score_available") is not False:
        raise ExperimentConfigError("final all-data model cannot claim a test score")
    if int(runtime.get("estimator_threads", -1)) != 1:
        raise ExperimentConfigError("estimator internal threads must remain one")

    # The checks above provide useful field-level diagnostics.  The canonical
    # digest is the final fail-closed guard: every scientifically material v1
    # field, including grids, budgets, gates, matrix membership, and runtime
    # policy, must remain byte-semantically identical to the pre-registration.
    digest = experiment_config_sha256(config)
    if digest != PINNED_EXPERIMENT_CONFIG_SHA256:
        raise ExperimentConfigError(
            "resolved endpoint-regression v1 config differs from the pinned "
            f"canonical SHA-256 {PINNED_EXPERIMENT_CONFIG_SHA256}; received {digest}"
        )


def load_experiment_config(
    path: str | Path, *, release_override: str | Path | None = None
) -> dict[str, Any]:
    config_path = Path(path).resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExperimentConfigError(f"config file does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ExperimentConfigError(f"invalid YAML config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExperimentConfigError("config root must be a mapping")
    payload["_config_path"] = str(config_path)
    payload["_workspace_root"] = str(config_path.parent.parent)
    if release_override is not None:
        payload["_release_override"] = str(Path(release_override).resolve())
    validate_experiment_config(payload)
    return payload


def resolve_release_path(config: dict[str, Any]) -> Path:
    if "_release_override" in config:
        return Path(str(config["_release_override"])).resolve()
    configured = Path(str(config["release"]["archive"]))
    if configured.is_absolute():
        return configured.resolve()
    return (Path(str(config["_workspace_root"])) / configured).resolve()


__all__ = [
    "EXPERIMENT_CONTRACT",
    "ExperimentConfigError",
    "PARENT_CONTRACT",
    "PINNED_EXPERIMENT_CONFIG_SHA256",
    "PINNED_RELEASE_SHA256",
    "experiment_config_sha256",
    "load_experiment_config",
    "public_experiment_config",
    "resolve_release_path",
    "validate_experiment_config",
]

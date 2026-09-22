"""Fail-closed validation for endpoint-regression experiment v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .augmentation import (
    CMIXUP_METHOD,
    LOCAL_METHOD,
    cmixup_offline_v1,
    local_interp_v2,
)
from .experiment_augmentation import BASE_SEED, AugmentationSettings, context_seed
from .experiment_config import (
    EXPERIMENT_CONTRACT,
    PINNED_RELEASE_SHA256,
    experiment_config_sha256,
    validate_experiment_config,
)
from .experiment_ensemble import SerializableEndpointEnsemble
from .experiment_io import (
    ENDPOINT_EVAL_PACKAGE_VERSIONS,
    environment_payload,
    read_json_object,
    sha256_file,
    stable_hash,
    validate_endpoint_eval_environment,
    verify_file_manifest,
)
from .experiment_matrix import (
    build_experiment_cells,
    resolve_feature_columns,
    validate_cell_feature_policy,
)
from .experiment_metrics import paired_group_effects, regression_metrics
from .experiment_models import (
    augmentation_gate,
    model_parameter_grid,
    select_best_configuration,
)
from .experiment_runner import (
    AUGMENTATION_RECIPES,
    PREDICTION_COLUMNS,
    SCHEMES,
    TRUE_COLUMNS,
    _aggregate_inner_selection,
)
from .experiment_splits import (
    EXPERIMENT_ASSIGNMENT_ROWS,
    assignment_sha256,
    validate_experiment_assignments,
)
from .features import (
    ENCODER_COLUMNS,
    LABEL_COLUMNS,
    PLANNED_COLUMNS,
    feature_views,
)


EXPECTED_ENVIRONMENT_VERSIONS = ENDPOINT_EVAL_PACKAGE_VERSIONS


class ExperimentValidationError(ValueError):
    """Raised when a run violates the frozen evaluation or inference contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExperimentValidationError(message)


def _uid_hash(values: Any) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_environment(
    environment: dict[str, Any],
    manifest: dict[str, Any],
    *,
    current_environment: dict[str, Any] | None = None,
) -> None:
    current = environment_payload() if current_environment is None else current_environment
    _require(
        current == environment,
        "current validation environment differs from the recorded run environment",
    )
    runtime = manifest.get("runtime", {})
    _require(
        runtime.get("environment") == environment,
        "manifest runtime environment differs from environment.json",
    )
    digest = stable_hash(environment)
    _require(
        runtime.get("environment_sha256") == digest,
        "runtime environment SHA-256 changed",
    )
    try:
        validate_endpoint_eval_environment(environment)
    except ValueError as exc:
        raise ExperimentValidationError(str(exc)) from exc
    _require(runtime.get("estimator_threads") == 1, "estimator thread count changed")
    _require(
        isinstance(runtime.get("context_jobs"), int)
        and 1 <= int(runtime["context_jobs"]) <= 32,
        "context job count is invalid",
    )


def _validate_eligible_audit(
    audit: pd.DataFrame,
    assignments: pd.DataFrame,
    manifest: dict[str, Any],
    allowlists: dict[str, Any],
) -> None:
    audit_features = list(
        dict.fromkeys(
            str(column)
            for columns in allowlists.values()
            for column in columns
        )
    )
    required = {
        "trial_uid",
        "segment_id",
        "target_id",
        "target_group",
        "set_id",
        "repeat_index",
        "strict_qc_eligible",
        *PLANNED_COLUMNS,
        *ENCODER_COLUMNS,
        *LABEL_COLUMNS,
        *audit_features,
    }
    _require(required <= set(audit.columns), "eligible-real audit schema is incomplete")
    _require(len(audit) == 7_702, "eligible-real audit row count changed")
    _require(
        audit["trial_uid"].astype(str).nunique() == len(audit),
        "eligible-real audit trial_uid is not unique",
    )
    _require(
        set(audit["trial_uid"].astype(str))
        == set(assignments["trial_uid"].astype(str)),
        "eligible-real audit does not cover assignment trials exactly",
    )
    numeric_columns = list(
        dict.fromkeys(
            [*PLANNED_COLUMNS, *ENCODER_COLUMNS, *LABEL_COLUMNS, *audit_features]
        )
    )
    numeric = audit.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    _require(np.isfinite(numeric).all(), "eligible-real audit contains non-finite values")
    expected_target_group = (
        audit["target_id"].astype(str).str[1:].astype(int).to_numpy() % 5
    )
    _require(
        np.array_equal(audit["target_group"].to_numpy(int), expected_target_group),
        "eligible-real audit target-group derivation changed",
    )
    _require(
        int(audit["strict_qc_eligible"].astype(bool).sum()) == 7_670,
        "eligible-real audit strict-QC count changed",
    )
    _require(
        manifest.get("counts", {}).get("train_eligible_real") == len(audit),
        "manifest eligible-real audit count changed",
    )


def _validate_feature_allowlists(
    payload: dict[str, Any], cells: list[Any], config: dict[str, Any]
) -> None:
    release_views = feature_views()
    expected: dict[str, list[str]] = {}
    for cell in cells:
        columns = resolve_feature_columns(
            cell.feature_view,
            config=config,
            release_feature_sets=release_views,
        )
        try:
            validate_cell_feature_policy(cell, columns)
        except ValueError as exc:
            raise ExperimentValidationError(
                f"feature policy cannot be reproduced for {cell.cell_id}: {exc}"
            ) from exc
        expected[cell.cell_id] = columns
    _require(payload == expected, "feature allow-lists differ from the frozen feature views")


def _validate_preprocessor_state(
    metadata: dict[str, Any],
    cohort: pd.DataFrame,
    *,
    feature_columns: list[str],
    scale_features: bool,
    expected_uid_hash: str,
    split_context: str,
    forbidden_uid_count: int,
) -> None:
    _require(
        metadata.get("feature_names") == feature_columns,
        "preprocessor feature allow-list changed",
    )
    X = cohort.loc[:, feature_columns].to_numpy(dtype=np.float64)
    y = cohort.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    observed_mean = np.mean(X, axis=0, dtype=np.float64)
    observed_scale = np.std(X, axis=0, ddof=0, dtype=np.float64)
    expected_feature_mean = (
        observed_mean if scale_features else np.zeros(X.shape[1], dtype=np.float64)
    )
    expected_feature_scale = (
        np.where(observed_scale > 0.0, observed_scale, 1.0)
        if scale_features
        else np.ones(X.shape[1], dtype=np.float64)
    )
    target_mean = np.mean(y, axis=0, dtype=np.float64)
    target_scale = float(np.sqrt(np.mean(np.square(y - target_mean))))
    expected_target_scale = target_scale if target_scale > 0.0 else 1.0
    _require(
        np.allclose(
            np.asarray(metadata.get("feature_mean"), dtype=np.float64),
            expected_feature_mean,
            rtol=1e-12,
            atol=1e-12,
        )
        and np.allclose(
            np.asarray(metadata.get("feature_scale"), dtype=np.float64),
            expected_feature_scale,
            rtol=1e-12,
            atol=1e-12,
        )
        and np.allclose(
            np.asarray(metadata.get("target_mean"), dtype=np.float64),
            target_mean,
            rtol=1e-12,
            atol=1e-12,
        )
        and np.isclose(
            float(metadata.get("target_scale")),
            expected_target_scale,
            rtol=1e-12,
            atol=1e-12,
        ),
        "preprocessor feature/target statistics cannot be reproduced from real train rows",
    )
    leakage = metadata.get("leakage_audit", {})
    _require(
        leakage.get("fit_scope") == "real_train_only"
        and leakage.get("split_context") == split_context
        and int(leakage.get("fit_row_count", -1)) == len(cohort)
        and leakage.get("fit_trial_uid_sha256") == expected_uid_hash
        and int(leakage.get("forbidden_uid_count", -1)) == forbidden_uid_count
        and int(leakage.get("forbidden_uid_overlap_count", -1)) == 0
        and bool(leakage.get("feature_scaling_enabled")) is scale_features
        and leakage.get("target_scale_kind") == "single_isotropic_xyz_scale",
        "preprocessor leakage/scaling contract changed",
    )


def _load_required_tables(run: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "assignments": "splits/evaluation_assignments.parquet",
        "trace": "selection/inner_selection_trace.parquet",
        "final_trace": "selection/final_selection_trace.parquet",
        "seed_oof": "predictions/oof_seed.parquet",
        "ensemble_oof": "predictions/oof_ensemble.parquet",
        "folds": "metrics/fold_metrics.parquet",
        "cells": "metrics/cell_metrics.parquet",
        "aggregate": "metrics/aggregate_metrics.parquet",
        "strict_oof": "sensitivity/strict_oof_metrics.parquet",
        "gates": "statistics/augmentation_gates.parquet",
        "paired": "statistics/paired_comparisons.parquet",
        "augmentation": "augmentation/context_provenance.parquet",
        "augmentation_rows": "augmentation/row_provenance.parquet",
        "eligible_audit": "analysis/eligible_real_audit.parquet",
    }
    result: dict[str, pd.DataFrame] = {}
    for name, relative in paths.items():
        path = run / relative
        try:
            result[name] = pd.read_parquet(path)
        except FileNotFoundError as exc:
            raise ExperimentValidationError(f"missing required Parquet: {relative}") from exc
    return result


def _validate_assignments(
    assignments: pd.DataFrame, identity: dict[str, Any], manifest: dict[str, Any]
) -> None:
    try:
        validate_experiment_assignments(assignments)
    except ValueError as exc:
        raise ExperimentValidationError(f"evaluation assignments changed: {exc}") from exc
    digest = assignment_sha256(assignments)
    _require(identity.get("sha256") == digest, "assignment identity SHA-256 changed")
    _require(identity.get("rows") == EXPERIMENT_ASSIGNMENT_ROWS, "assignment identity rows changed")
    _require(identity.get("contexts") == 50, "assignment identity context count changed")
    _require(
        manifest.get("counts", {}).get("evaluation_assignments")
        == EXPERIMENT_ASSIGNMENT_ROWS,
        "manifest assignment count changed",
    )


def _validate_oof(
    tables: dict[str, pd.DataFrame],
    assignments: pd.DataFrame,
    cells: list[Any],
    manifest: dict[str, Any],
    eligible_audit: pd.DataFrame,
) -> None:
    ensemble = tables["ensemble_oof"]
    seeds = tables["seed_oof"]
    expected = 7_702 * len(cells) * len(SCHEMES)
    _require(len(ensemble) == expected, f"ensemble OOF rows changed: {len(ensemble)}")
    _require(
        manifest.get("counts", {}).get("ensemble_oof_predictions") == expected,
        "manifest ensemble OOF count changed",
    )
    required = {
        "scheme",
        "outer_fold",
        "cell_id",
        "trial_uid",
        "prediction_kind",
        "seed_index",
        "strict_qc_eligible",
        *TRUE_COLUMNS,
        *PREDICTION_COLUMNS,
    }
    _require(required <= set(ensemble.columns), "ensemble OOF schema is incomplete")
    numeric = ensemble.loc[:, [*TRUE_COLUMNS, *PREDICTION_COLUMNS, "error_3d_mm"]].to_numpy(float)
    _require(np.isfinite(numeric).all(), "ensemble OOF contains non-finite values")
    _require(ensemble["prediction_kind"].eq("ensemble").all(), "ensemble OOF kind changed")
    _require(ensemble["seed_index"].eq(-1).all(), "ensemble OOF seed marker changed")
    _require(
        not ensemble.duplicated(["scheme", "cell_id", "trial_uid"]).any(),
        "ensemble OOF contains duplicate scheme/cell/trial keys",
    )

    cell_ids = {cell.cell_id for cell in cells}
    _require(set(ensemble["cell_id"].astype(str)) == cell_ids, "ensemble OOF cell set changed")
    for (scheme, cell_id), frame in ensemble.groupby(["scheme", "cell_id"], sort=False):
        _require(len(frame) == 7_702, f"OOF cell {scheme}/{cell_id} is incomplete")
        _require(frame["trial_uid"].astype(str).nunique() == 7_702, "OOF trial coverage changed")
        _require(frame["outer_fold"].nunique() == 5, "OOF cell lacks five outer folds")

    refit = assignments.loc[assignments["inner_fold"].eq(-1), [
        "scheme", "outer_fold", "trial_uid", "role"
    ]]
    joined = ensemble.merge(
        refit,
        on=["scheme", "outer_fold", "trial_uid"],
        how="left",
        validate="many_to_one",
    )
    _require(joined["role"].eq("test").all(), "OOF row is not from its declared outer test role")

    real_index = eligible_audit.set_index(
        eligible_audit["trial_uid"].astype(str), drop=False
    )
    real = real_index.loc[ensemble["trial_uid"].astype(str)]
    _require(
        np.array_equal(
            ensemble.loc[:, list(TRUE_COLUMNS)].to_numpy(float),
            real.loc[:, list(LABEL_COLUMNS)].to_numpy(float),
        ),
        "OOF truth differs from eligible measured-real labels",
    )
    _require(
        np.array_equal(
            ensemble["segment_id"].to_numpy(int), real["segment_id"].to_numpy(int)
        )
        and np.array_equal(
            ensemble["target_id"].astype(str).to_numpy(),
            real["target_id"].astype(str).to_numpy(),
        )
        and np.array_equal(
            ensemble["strict_qc_eligible"].astype(bool).to_numpy(),
            real["strict_qc_eligible"].astype(bool).to_numpy(),
        ),
        "OOF real-row metadata differs from eligible audit",
    )

    truth_variants = ensemble.groupby(["scheme", "trial_uid"], observed=True)[
        list(TRUE_COLUMNS)
    ].nunique()
    _require((truth_variants <= 1).all().all(), "truth changed across OOF cells")
    relation = np.sqrt(3.0) * tables["trace"]["rmse_xyz_mm"].to_numpy(float)
    _require(
        np.allclose(tables["trace"]["rmse_3d_mm"].to_numpy(float), relation, rtol=1e-12, atol=1e-12),
        "RMSE_3D != sqrt(3) * rmse_xyz_mm",
    )

    cell_map = {cell.cell_id: cell for cell in cells}
    _require(seeds["prediction_kind"].eq("seed").all(), "seed OOF kind changed")
    _require(
        not seeds.duplicated(["scheme", "cell_id", "trial_uid", "seed_index"]).any(),
        "seed OOF duplicate key detected",
    )
    seed_counts = (
        seeds.groupby(["scheme", "cell_id", "trial_uid"], sort=False, observed=True)
        .size()
        .reset_index(name="seed_count")
    )
    expected_seed_counts = seed_counts["cell_id"].astype(str).map(
        {cell_id: (3 if cell.model == "mlp" else 1) for cell_id, cell in cell_map.items()}
    )
    _require(
        np.array_equal(seed_counts["seed_count"].to_numpy(int), expected_seed_counts.to_numpy(int)),
        "seed OOF count changed for one or more scheme/cell/trial groups",
    )
    seed_means = (
        seeds.groupby(
            ["scheme", "outer_fold", "cell_id", "trial_uid"],
            sort=False,
            observed=True,
        )[list(PREDICTION_COLUMNS)]
        .mean()
        .reset_index()
    )
    ensemble_join = ensemble.merge(
        seed_means,
        on=["scheme", "outer_fold", "cell_id", "trial_uid"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_seed_mean"),
    )
    for column in PREDICTION_COLUMNS:
        _require(
            np.allclose(
                ensemble_join[column].to_numpy(float),
                ensemble_join[f"{column}_seed_mean"].to_numpy(float),
                rtol=1e-12,
                atol=1e-12,
            ),
            "ensemble OOF prediction is not the mean of its seed predictions",
        )


def _validate_selection(
    trace: pd.DataFrame,
    cells: list[Any],
    assignments: pd.DataFrame,
    ensemble_oof: pd.DataFrame,
    seed_oof: pd.DataFrame,
    outer_preprocessing: dict[str, Any],
    config: dict[str, Any],
    eligible_audit: pd.DataFrame,
    allowlists: dict[str, Any],
) -> None:
    _require(set(trace["scheme"].astype(str)) == set(SCHEMES), "inner trace scheme set changed")
    _require(trace["validation_rows"].gt(0).all(), "inner trace has empty validation rows")
    _require(not any("trial_uid" in column for column in trace.columns), "selection trace exposes trial IDs")
    required = {
        "scheme",
        "outer_fold",
        "inner_fold",
        "split_context",
        "cell_id",
        "model",
        "recipe",
        "config_id",
        "params_json",
        "prediction_kind",
        "real_train_rows",
        "additional_rows",
        "preprocessor_fit_uid_sha256",
    }
    _require(required <= set(trace.columns), "selection trace audit schema is incomplete")
    context_expectations: dict[tuple[str, int, int], dict[str, Any]] = {}
    for (scheme, outer, inner), frame in assignments.groupby(
        ["scheme", "outer_fold", "inner_fold"], sort=False
    ):
        train_ids = frame.loc[frame["role"].eq("train"), "trial_uid"].astype(str)
        forbidden_ids = frame.loc[~frame["role"].eq("train"), "trial_uid"].astype(str)
        context_expectations[(str(scheme), int(outer), int(inner))] = {
            "uid_hash": _uid_hash(train_ids),
            "rows": int(len(train_ids)),
            "train_ids": tuple(sorted(train_ids)),
            "forbidden_rows": int(len(forbidden_ids)),
            "split_context": str(frame["split_context"].iloc[0]),
        }
    for (scheme, outer, inner), frame in trace.groupby(
        ["scheme", "outer_fold", "inner_fold"], sort=False
    ):
        expectation = context_expectations[(str(scheme), int(outer), int(inner))]
        _require(
            frame["split_context"].astype(str).eq(expectation["split_context"]).all(),
            "selection trace split_context differs from assignments",
        )
        _require(
            frame["real_train_rows"].astype(int).eq(expectation["rows"]).all(),
            "selection trace real-train count differs from assignments",
        )
        _require(
            frame["preprocessor_fit_uid_sha256"]
            .astype(str)
            .eq(expectation["uid_hash"])
            .all(),
            "inner preprocessing fit UID hash differs from assignment train role",
        )
        expected_added = np.where(
            frame["recipe"].astype(str).eq("real_only"),
            0,
            expectation["rows"],
        )
        _require(
            np.array_equal(frame["additional_rows"].to_numpy(int), expected_added),
            "selection trace recipe is not 1:1 with its real train role",
        )
    seed_trace = trace.loc[trace["prediction_kind"].eq("seed")]
    paired_inner_seed_counts = seed_trace.groupby(
        ["scheme", "outer_fold", "inner_fold", "seed_index"],
        sort=False,
        observed=True,
    )["model_seed"].nunique()
    _require(
        paired_inner_seed_counts.eq(1).all(),
        "inner model seeds are not paired across models/configurations/recipes",
    )
    paired_outer_seed_counts = seed_oof.groupby(
        ["scheme", "outer_fold", "seed_index"], sort=False, observed=True
    )["model_seed"].nunique()
    _require(
        paired_outer_seed_counts.eq(1).all(),
        "outer OOF model seeds are not paired across model/recipe cells",
    )

    cell_map = {cell.cell_id: cell for cell in cells}
    selected_by_outer: dict[tuple[str, int, str], dict[str, Any]] = {}
    for (scheme, outer, cell_id), group in trace.loc[
        trace["prediction_kind"].eq("ensemble")
    ].groupby(["scheme", "outer_fold", "cell_id"], sort=False):
        cell = cell_map[str(cell_id)]
        grid = model_parameter_grid(cell.model)
        _require(group["inner_fold"].nunique() == 4, "selection cell lacks four inner groups")
        _require(group["config_id"].nunique() == len(grid), "selection grid coverage changed")
        candidates = []
        for (config_id, params_json), candidate_rows in group.groupby(
            ["config_id", "params_json"], sort=False
        ):
            _require(len(candidate_rows) == 4, "candidate lacks four equal-weight inner scores")
            candidates.append(
                {
                    "params": json.loads(str(params_json)),
                    "inner_scores": candidate_rows["rmse_3d_mm"].astype(float).tolist(),
                }
            )
        selected = select_best_configuration(
            cell.model,
            candidates,
            tie_tolerance=float(config["selection"]["parameter_tie_tolerance"]),
        )
        key = (str(scheme), int(outer), str(cell_id))
        selected_by_outer[key] = selected
        outer_ensemble = ensemble_oof.loc[
            ensemble_oof["scheme"].eq(scheme)
            & ensemble_oof["outer_fold"].eq(outer)
            & ensemble_oof["cell_id"].eq(cell_id)
        ]
        outer_seeds = seed_oof.loc[
            seed_oof["scheme"].eq(scheme)
            & seed_oof["outer_fold"].eq(outer)
            & seed_oof["cell_id"].eq(cell_id)
        ]
        expected_config = str(selected["config_id"])
        expected_params = json.dumps(
            selected["params"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        _require(
            len(outer_ensemble) > 0
            and outer_ensemble["config_id"].astype(str).eq(expected_config).all()
            and outer_seeds["config_id"].astype(str).eq(expected_config).all(),
            "OOF predictions do not use the inner-selected configuration",
        )
        _require(
            outer_ensemble["params_json"].astype(str).eq(expected_params).all()
            and outer_seeds["params_json"].astype(str).eq(expected_params).all(),
            "OOF prediction parameters differ from inner selection",
        )

    outer_rows = outer_preprocessing.get("outer_refits")
    _require(isinstance(outer_rows, list), "outer preprocessing audit is malformed")
    _require(len(outer_rows) == len(cells) * len(SCHEMES) * 5, "outer preprocessing row count changed")
    observed_keys: set[tuple[str, int, str]] = set()
    audit_index = eligible_audit.set_index(
        eligible_audit["trial_uid"].astype(str), drop=False
    )
    for row in outer_rows:
        key = (str(row["scheme"]), int(row["outer_fold"]), str(row["cell_id"]))
        _require(key not in observed_keys, "outer preprocessing audit key is duplicated")
        observed_keys.add(key)
        selected = selected_by_outer.get(key)
        _require(selected is not None, "outer preprocessing key is absent from selection trace")
        expectation = context_expectations[(key[0], key[1], -1)]
        _require(
            int(row["real_train_rows"]) == expectation["rows"]
            and str(row["real_train_uid_sha256"]) == expectation["uid_hash"],
            "outer preprocessing fit UID hash differs from assignment train role",
        )
        _require(
            str(row["selected_config_id"]) == str(selected["config_id"])
            and dict(row["selected_params"]) == dict(selected["params"]),
            "outer preprocessing metadata differs from inner-selected config",
        )
        _require(
            np.isclose(
                float(row["selection_score_rmse_3d_mm"]),
                float(selected["macro_group_rmse_3d_mm"]),
                rtol=1e-12,
                atol=1e-12,
            ),
            "outer preprocessing selection score changed",
        )
        cell = cell_map[key[2]]
        models = row.get("models")
        expected_seed_count = 3 if cell.model == "mlp" else 1
        _require(
            isinstance(models, list) and len(models) == expected_seed_count,
            "outer preprocessing ensemble member count changed",
        )
        for model in models:
            metadata = model.get("metadata", {})
            _require(
                metadata.get("split_context") == expectation["split_context"],
                "outer fitted model split_context differs from assignments",
            )
            if cell.model == "dummy_mean":
                _require(
                    int(metadata.get("fit_row_count", -1)) == expectation["rows"],
                    "dummy outer fit count differs from assignment train role",
                )
                continue
            preprocessing = metadata.get("preprocessor", {})
            _validate_preprocessor_state(
                preprocessing,
                audit_index.loc[list(expectation["train_ids"])].reset_index(drop=True),
                feature_columns=list(allowlists[key[2]]),
                scale_features=cell.model in {"ridge", "mlp"},
                expected_uid_hash=str(expectation["uid_hash"]),
                split_context=str(expectation["split_context"]),
                forbidden_uid_count=int(expectation["forbidden_rows"]),
            )
    _require(
        observed_keys == set(selected_by_outer),
        "outer preprocessing audit does not cover every selected outer cell",
    )


def _validate_metrics(tables: dict[str, pd.DataFrame]) -> None:
    ensemble = tables["ensemble_oof"]
    seeds = tables["seed_oof"]
    folds = tables["folds"]
    cells = tables["cells"]
    aggregate = tables["aggregate"]
    strict_oof = tables["strict_oof"]
    _require(len(folds) == 31 * 2 * 5, "fold metric row count changed")
    _require(len(aggregate) == 31 * 2, "aggregate metric row count changed")
    cell_key = [
        "scheme",
        "outer_fold",
        "cell_id",
        "segment_id",
        "target_id",
    ]
    _require(
        not cells.duplicated(cell_key).any(),
        "segment-target cell metrics contain duplicate keys",
    )
    expected_cell_keys = ensemble.loc[:, cell_key].drop_duplicates()
    observed_cell_keys = cells.loc[:, cell_key].drop_duplicates()
    key_join = expected_cell_keys.merge(
        observed_cell_keys,
        on=cell_key,
        how="outer",
        indicator=True,
    )
    _require(
        key_join["_merge"].eq("both").all(),
        "segment-target cell metric coverage differs from OOF predictions",
    )
    for (scheme, outer, cell_id), frame in ensemble.groupby(
        ["scheme", "outer_fold", "cell_id"], sort=False
    ):
        truth = frame.loc[:, list(TRUE_COLUMNS)].to_numpy(float)
        prediction = frame.loc[:, list(PREDICTION_COLUMNS)].to_numpy(float)
        expected = regression_metrics(truth, prediction)
        row = folds.loc[
            folds["scheme"].eq(scheme)
            & folds["outer_fold"].eq(outer)
            & folds["cell_id"].eq(cell_id)
        ]
        _require(len(row) == 1, "fold metric key is missing or duplicated")
        for key in ("rmse_3d_mm", "rmse_xyz_mm", "mean_3d_error_mm", "p95_3d_error_mm"):
            _require(
                np.isclose(float(row.iloc[0][key]), float(expected[key]), rtol=1e-12, atol=1e-12),
                f"fold metric {key} cannot be reproduced",
            )
        detail = cells.loc[
            cells["scheme"].eq(scheme)
            & cells["outer_fold"].eq(outer)
            & cells["cell_id"].eq(cell_id)
        ]
        _require(
            int(detail["row_count"].sum()) == len(frame),
            "segment-target cell row counts do not cover the fold",
        )
        _require(
            int(row.iloc[0]["cell_count"]) == len(detail),
            "fold cell count differs from detailed cell metrics",
        )
        balanced_rmse = float(
            np.sqrt(np.mean(np.square(detail["rmse_3d_mm"].to_numpy(float))))
        )
        _require(
            np.isclose(
                float(row.iloc[0]["cell_balanced_rmse_3d_mm"]),
                balanced_rmse,
                rtol=1e-12,
                atol=1e-12,
            ),
            "fold cell-balanced RMSE cannot be reproduced from detailed cells",
        )
        _require(
            np.isclose(
                float(row.iloc[0]["macro_cell_rmse_3d_mm"]),
                float(detail["rmse_3d_mm"].mean()),
                rtol=1e-12,
                atol=1e-12,
            ),
            "fold macro cell RMSE cannot be reproduced from detailed cells",
        )
        _require(
            np.isclose(
                float(row.iloc[0]["macro_cell_mean_3d_error_mm"]),
                float(detail["mean_3d_error_mm"].mean()),
                rtol=1e-12,
                atol=1e-12,
            ),
            "fold macro cell mean error cannot be reproduced from detailed cells",
        )
        for (segment_id, target_id), cell_frame in frame.groupby(
            ["segment_id", "target_id"], sort=False
        ):
            cell_row = detail.loc[
                detail["segment_id"].eq(segment_id)
                & detail["target_id"].astype(str).eq(str(target_id))
            ]
            _require(len(cell_row) == 1, "detailed segment-target metric key is missing")
            cell_truth = cell_frame.loc[:, list(TRUE_COLUMNS)].to_numpy(float)
            cell_prediction = cell_frame.loc[:, list(PREDICTION_COLUMNS)].to_numpy(float)
            expected_cell = regression_metrics(cell_truth, cell_prediction)
            for key in (
                "row_count",
                "rmse_3d_mm",
                "rmse_xyz_mm",
                "mean_3d_error_mm",
                "median_3d_error_mm",
                "p90_3d_error_mm",
                "p95_3d_error_mm",
            ):
                _require(
                    np.isclose(
                        float(cell_row.iloc[0][key]),
                        float(expected_cell[key]),
                        rtol=1e-12,
                        atol=1e-12,
                    ),
                    f"detailed segment-target metric {key} cannot be reproduced",
                )
    for (scheme, cell_id), frame in folds.groupby(["scheme", "cell_id"], sort=False):
        row = aggregate.loc[
            aggregate["scheme"].eq(scheme) & aggregate["cell_id"].eq(cell_id)
        ]
        _require(len(row) == 1, "aggregate metric key is missing or duplicated")
        macro_fields = {
            "macro_rmse_3d_mm": "rmse_3d_mm",
            "macro_rmse_xyz_mm": "rmse_xyz_mm",
            "macro_mean_3d_error_mm": "mean_3d_error_mm",
            "macro_median_3d_error_mm": "median_3d_error_mm",
            "macro_p90_3d_error_mm": "p90_3d_error_mm",
            "macro_p95_3d_error_mm": "p95_3d_error_mm",
            "macro_cell_balanced_rmse_3d_mm": "cell_balanced_rmse_3d_mm",
        }
        for aggregate_field, fold_field in macro_fields.items():
            _require(
                np.isclose(
                    float(row.iloc[0][aggregate_field]),
                    float(frame[fold_field].mean()),
                    rtol=1e-12,
                    atol=1e-12,
                ),
                f"aggregate {aggregate_field} is not an equal-fold mean",
            )
        oof = ensemble.loc[
            ensemble["scheme"].eq(scheme) & ensemble["cell_id"].eq(cell_id)
        ]
        pooled = regression_metrics(
            oof.loc[:, list(TRUE_COLUMNS)].to_numpy(float),
            oof.loc[:, list(PREDICTION_COLUMNS)].to_numpy(float),
        )
        for key, expected in pooled.items():
            _require(
                np.isclose(
                    float(row.iloc[0][f"pooled_{key}"]),
                    float(expected),
                    rtol=1e-12,
                    atol=1e-12,
                ),
                f"aggregate pooled_{key} cannot be reproduced",
            )
        seed_scores: list[float] = []
        seed_subset = seeds.loc[
            seeds["scheme"].eq(scheme) & seeds["cell_id"].eq(cell_id)
        ]
        for _, seed_frame in seed_subset.groupby("seed_index", sort=True):
            fold_scores = []
            for _, seed_fold in seed_frame.groupby("outer_fold", sort=True):
                fold_scores.append(
                    float(
                        regression_metrics(
                            seed_fold.loc[:, list(TRUE_COLUMNS)].to_numpy(float),
                            seed_fold.loc[:, list(PREDICTION_COLUMNS)].to_numpy(float),
                        )["rmse_3d_mm"]
                    )
                )
            seed_scores.append(float(np.mean(fold_scores)))
        _require(
            np.isclose(
                float(row.iloc[0]["seed_macro_rmse_mean_mm"]),
                float(np.mean(seed_scores)),
                rtol=1e-12,
                atol=1e-12,
            )
            and np.isclose(
                float(row.iloc[0]["seed_macro_rmse_std_mm"]),
                float(np.std(seed_scores, ddof=0)),
                rtol=1e-12,
                atol=1e-12,
            ),
            "aggregate seed variability cannot be reproduced",
        )
    _require(len(strict_oof) == 31 * 2, "strict-QC OOF metric row count changed")
    for (scheme, cell_id), frame in ensemble.loc[
        ensemble["strict_qc_eligible"].eq(True)
    ].groupby(["scheme", "cell_id"], sort=False):
        _require(len(frame) == 7_670, "strict-QC OOF trial coverage changed")
        truth = frame.loc[:, list(TRUE_COLUMNS)].to_numpy(float)
        prediction = frame.loc[:, list(PREDICTION_COLUMNS)].to_numpy(float)
        expected = regression_metrics(truth, prediction)
        row = strict_oof.loc[
            strict_oof["scheme"].eq(scheme) & strict_oof["cell_id"].eq(cell_id)
        ]
        _require(len(row) == 1, "strict-QC OOF metric key is missing or duplicated")
        for key in (
            "row_count",
            "rmse_3d_mm",
            "rmse_xyz_mm",
            "mean_3d_error_mm",
            "median_3d_error_mm",
            "p90_3d_error_mm",
            "p95_3d_error_mm",
        ):
            _require(
                np.isclose(
                    float(row.iloc[0][key]),
                    float(expected[key]),
                    rtol=1e-12,
                    atol=1e-12,
                ),
                f"strict-QC OOF metric {key} cannot be reproduced",
            )


def _validate_statistics(tables: dict[str, pd.DataFrame]) -> None:
    gates = tables["gates"]
    paired = tables["paired"]
    aggregate = tables["aggregate"]
    ensemble = tables["ensemble_oof"]
    _require(len(gates) == 8, "augmentation gate row count changed")
    _require(set(gates["recipe"]) == {"local_interp_v2", "cmixup_offline_v1"}, "gate recipe set changed")
    expected_pass = gates["rmse_pass"].astype(bool) & gates["p95_pass"].astype(bool)
    _require(gates["passed"].astype(bool).equals(expected_pass), "gate conjunction changed")
    segment = aggregate.loc[aggregate["scheme"].eq("segment_loso")]
    for row in gates.itertuples(index=False):
        reference = segment.loc[segment["cell_id"].eq(str(row.reference_cell_id))]
        candidate = segment.loc[segment["cell_id"].eq(str(row.candidate_cell_id))]
        _require(len(reference) == len(candidate) == 1, "gate metric cell is missing")
        reference_row = reference.iloc[0]
        candidate_row = candidate.iloc[0]
        recalculated = augmentation_gate(
            reference_macro_rmse_3d_mm=float(reference_row["macro_rmse_3d_mm"]),
            candidate_macro_rmse_3d_mm=float(candidate_row["macro_rmse_3d_mm"]),
            reference_macro_p95_3d_error_mm=float(reference_row["macro_p95_3d_error_mm"]),
            candidate_macro_p95_3d_error_mm=float(candidate_row["macro_p95_3d_error_mm"]),
        )
        metric_fields = {
            "reference_macro_rmse_3d_mm": float(reference_row["macro_rmse_3d_mm"]),
            "candidate_macro_rmse_3d_mm": float(candidate_row["macro_rmse_3d_mm"]),
            "reference_macro_p95_3d_error_mm": float(reference_row["macro_p95_3d_error_mm"]),
            "candidate_macro_p95_3d_error_mm": float(candidate_row["macro_p95_3d_error_mm"]),
            **recalculated,
        }
        for field, expected in metric_fields.items():
            actual = getattr(row, field)
            if isinstance(expected, bool):
                _require(bool(actual) is expected, f"augmentation gate {field} changed")
            else:
                _require(
                    np.isclose(float(actual), float(expected), rtol=1e-12, atol=1e-12),
                    f"augmentation gate {field} cannot be reproduced",
                )
    _require(len(paired) == 24, "paired comparison row count changed")
    _require(not paired["p_value_reported"].astype(bool).any(), "paired report contains a p-value claim")
    _require(paired["bootstrap_replicates"].eq(10_000).all(), "bootstrap replicate count changed")
    _require(paired["trial_count"].eq(7_702).all(), "paired trial coverage changed")
    for row in paired.itertuples(index=False):
        reference = ensemble.loc[
            ensemble["scheme"].eq(str(row.scheme))
            & ensemble["cell_id"].eq(str(row.reference_cell_id))
        ].sort_values("trial_uid", kind="mergesort")
        candidate = ensemble.loc[
            ensemble["scheme"].eq(str(row.scheme))
            & ensemble["cell_id"].eq(str(row.candidate_cell_id))
        ].sort_values("trial_uid", kind="mergesort")
        _require(
            len(reference) == len(candidate) == 7_702
            and np.array_equal(
                reference["trial_uid"].astype(str).to_numpy(),
                candidate["trial_uid"].astype(str).to_numpy(),
            )
            and np.array_equal(
                reference.loc[:, list(TRUE_COLUMNS)].to_numpy(float),
                candidate.loc[:, list(TRUE_COLUMNS)].to_numpy(float),
            ),
            "paired comparison trials are not exactly aligned",
        )
        truth = reference.loc[:, list(TRUE_COLUMNS)].to_numpy(float)
        reference_prediction = reference.loc[:, list(PREDICTION_COLUMNS)].to_numpy(float)
        candidate_prediction = candidate.loc[:, list(PREDICTION_COLUMNS)].to_numpy(float)
        reference_metrics = regression_metrics(truth, reference_prediction)
        candidate_metrics = regression_metrics(truth, candidate_prediction)
        _require(
            np.isclose(
                float(row.rmse_observed_delta_mm),
                float(candidate_metrics["rmse_3d_mm"])
                - float(reference_metrics["rmse_3d_mm"]),
                rtol=1e-12,
                atol=1e-12,
            )
            and np.isclose(
                float(row.p95_observed_delta_mm),
                float(candidate_metrics["p95_3d_error_mm"])
                - float(reference_metrics["p95_3d_error_mm"]),
                rtol=1e-12,
                atol=1e-12,
            ),
            "paired observed effect cannot be reproduced from OOF predictions",
        )
        effects = paired_group_effects(
            truth,
            reference_prediction,
            candidate_prediction,
            reference["outer_fold"].to_numpy(),
        )
        expected_effects = json.dumps(
            effects, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        _require(
            str(row.fold_effects_json) == expected_effects,
            "paired fold effects cannot be reproduced from OOF predictions",
        )


def _recompute_cmixup_sigma(train: pd.DataFrame) -> float:
    planned = train.loc[:, list(PLANNED_COLUMNS)].to_numpy(dtype=np.float64)
    labels = train.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    joints = train.loc[:, list(ENCODER_COLUMNS)].to_numpy(dtype=np.float64)
    tree = cKDTree(planned)
    pools = tree.query_ball_point(planned, r=100.0 + 1e-9, workers=1)
    nearest: list[float] = []
    for anchor, pool_values in enumerate(pools):
        pool = np.asarray(
            [int(value) for value in pool_values if int(value) != anchor],
            dtype=np.int64,
        )
        if len(pool):
            planned_distance = np.linalg.norm(planned[pool] - planned[anchor], axis=1)
            joint_distance = np.rad2deg(
                np.max(np.abs(joints[pool] - joints[anchor]), axis=1)
            )
            pool = pool[
                (planned_distance <= 100.0 + 1e-12)
                & (joint_distance <= 20.0 + 1e-12)
            ]
        if len(pool):
            nearest.append(
                float(np.min(np.linalg.norm(labels[pool] - labels[anchor], axis=1)))
            )
    _require(nearest, "C-Mixup train-only sigma has no admissible anchors")
    return max(float(np.median(np.asarray(nearest, dtype=np.float64))), 1e-9)


_SEEDED_PROVENANCE_COLUMNS = (
    "synthetic_id",
    "is_synthetic",
    "augmentation_method",
    "split_scheme",
    "outer_fold",
    "inner_fold",
    "split_context",
    "anchor_trial_uid",
    "anchor_replica_index",
    "replacement_for_missing_anchor",
    "parent_a_trial_uid",
    "parent_b_trial_uid",
    "mix_lambda",
    "planned_parent_distance_mm",
    "label_parent_distance_mm",
    "generator_joint_parent_distance_deg",
    "cmixup_sigma_mm",
    "derived_seed",
    "source_hash",
    "config_hash",
    "sample_weight",
)


def _regenerate_seeded_augmentation_provenance(
    train: pd.DataFrame,
    *,
    scheme: str,
    outer_fold: int,
    inner_fold: int,
    recipe: str,
    augmentation_seed: int,
    source_hash: str,
    generator_config_hash: str,
) -> pd.DataFrame:
    """Replay only the frozen pairing/provenance layer from measured real rows.

    The packaged eligible-real audit intentionally omits raw Euler channels and
    synthetic feature values.  Pair selection needs only stable IDs, grouping,
    planned XYZ, encoder deltas, and raw-Vive labels, so replay the production
    generator with no extra linear/Euler outputs.  This preserves the exact
    candidate ordering, coverage schedule, RNG draws, IDs, and provenance while
    avoiding any new public package fields.
    """

    settings = AugmentationSettings().validated()
    shared: dict[str, Any] = {
        "planned_columns": PLANNED_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "linear_columns": (),
        "euler_columns": (),
        "split_scheme": str(scheme),
        "fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "base_seed": int(augmentation_seed),
        "source_hash": str(source_hash),
        "config_hash": str(generator_config_hash),
        "planned_max_distance_mm": settings.planned_max_distance_mm,
        "sample_weight": 0.5,
    }
    if recipe == LOCAL_METHOD:
        expected = local_interp_v2(
            train,
            neighbor_count=settings.local_neighbor_count,
            lambdas=settings.local_lambdas,
            **shared,
        )
    elif recipe == CMIXUP_METHOD:
        expected = cmixup_offline_v1(
            train,
            joint_rad_columns=ENCODER_COLUMNS,
            joint_max_distance_deg=settings.joint_max_distance_deg,
            alpha=settings.cmixup_alpha,
            **shared,
        )
    else:
        raise ExperimentValidationError(
            f"cannot replay unsupported augmentation recipe {recipe!r}"
        )
    expected = expected.copy()
    expected["generator_joint_parent_distance_deg"] = expected[
        "joint_parent_distance_deg"
    ]
    return expected.loc[:, list(_SEEDED_PROVENANCE_COLUMNS)]


def _validate_seeded_augmentation_provenance(
    rows: pd.DataFrame,
    train: pd.DataFrame,
    *,
    scheme: str,
    outer_fold: int,
    inner_fold: int,
    recipe: str,
    augmentation_seed: int,
    source_hash: str,
    generator_config_hash: str,
    generator_joint_column: str,
) -> None:
    """Require provenance to equal a deterministic production-generator replay."""

    expected = _regenerate_seeded_augmentation_provenance(
        train,
        scheme=scheme,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        recipe=recipe,
        augmentation_seed=augmentation_seed,
        source_hash=source_hash,
        generator_config_hash=generator_config_hash,
    )
    _require(
        generator_joint_column in rows.columns,
        "augmentation generator joint-distance provenance is missing",
    )
    actual = rows.copy()
    actual["generator_joint_parent_distance_deg"] = actual[generator_joint_column]
    missing = sorted(set(_SEEDED_PROVENANCE_COLUMNS) - set(actual.columns))
    _require(not missing, f"seeded augmentation provenance is incomplete: {missing}")
    actual = actual.loc[:, list(_SEEDED_PROVENANCE_COLUMNS)]
    expected.sort_values("synthetic_id", kind="mergesort", inplace=True)
    actual.sort_values("synthetic_id", kind="mergesort", inplace=True)
    expected.reset_index(drop=True, inplace=True)
    actual.reset_index(drop=True, inplace=True)
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_exact=True,
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as exc:
        raise ExperimentValidationError(
            "seeded augmentation provenance does not reproduce the frozen "
            f"generator for {scheme}/outer={outer_fold}/inner={inner_fold}/{recipe}"
        ) from exc


def _validate_augmentation(
    contexts: pd.DataFrame,
    rows: pd.DataFrame,
    eligible_audit: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict[str, Any],
    release_identity: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    context_key = ["scheme", "outer_fold", "inner_fold", "recipe"]
    _require(len(contexts) == 100, "augmentation context index must contain 100 method-context rows")
    _require(
        not contexts.duplicated(context_key).any(),
        "augmentation context index contains duplicate keys",
    )
    _require(
        set(contexts["recipe"]) == set(AUGMENTATION_RECIPES),
        "augmentation context methods changed",
    )
    assignment_contexts = assignments.loc[
        :, ["scheme", "outer_fold", "inner_fold"]
    ].drop_duplicates()
    expected_context_keys = {
        (str(row.scheme), int(row.outer_fold), int(row.inner_fold), recipe)
        for row in assignment_contexts.itertuples(index=False)
        for recipe in AUGMENTATION_RECIPES
    }
    observed_context_keys = {
        (str(row.scheme), int(row.outer_fold), int(row.inner_fold), str(row.recipe))
        for row in contexts.itertuples(index=False)
    }
    _require(
        observed_context_keys == expected_context_keys,
        "augmentation context index differs from assignment contexts",
    )
    _require(
        contexts["real_train_rows"].eq(contexts["additional_rows"]).all(),
        "augmentation context is not 1:1",
    )
    _require(
        np.allclose(
            contexts["objective_mass"].to_numpy(float),
            contexts["real_train_rows"].to_numpy(float),
        ),
        "augmentation objective mass changed",
    )
    required_rows = {
        "synthetic_id",
        "augmentation_method",
        "split_scheme",
        "outer_fold",
        "inner_fold",
        "split_context",
        "anchor_trial_uid",
        "parent_a_trial_uid",
        "parent_b_trial_uid",
        "mix_lambda",
        "planned_parent_distance_mm",
        "label_parent_distance_mm",
        "joint_parent_distance_deg",
        "generator_joint_parent_distance_deg",
        "joint_constraint_applied",
        "cmixup_sigma_mm",
        "derived_seed",
        "source_hash",
        "config_hash",
        "experiment_config_sha256",
        "sample_weight",
        "context_task_hash",
        "context_parquet_sha256",
        "context_real_train_uid_sha256",
    }
    _require(required_rows <= set(rows.columns), "augmentation row provenance schema is incomplete")
    provenance_key = [
        "split_scheme",
        "outer_fold",
        "inner_fold",
        "augmentation_method",
        "synthetic_id",
    ]
    _require(not rows.duplicated(provenance_key).any(), "augmentation provenance key is duplicated")
    _require(rows["is_synthetic"].astype(bool).all(), "augmentation provenance contains a real row")
    _require(
        (rows["parent_a_trial_uid"].astype(str) != rows["parent_b_trial_uid"].astype(str)).all(),
        "augmentation provenance contains a self-pair",
    )
    _require(
        rows["anchor_trial_uid"].astype(str).eq(rows["parent_a_trial_uid"].astype(str)).all(),
        "augmentation anchor differs from parent A",
    )
    _require(
        np.isfinite(rows["mix_lambda"].to_numpy(float)).all()
        and rows["mix_lambda"].astype(float).between(0.0, 1.0, inclusive="neither").all(),
        "augmentation lambda is outside (0, 1)",
    )
    derived_seeds = rows["derived_seed"].to_numpy(dtype=np.int64)
    _require(
        np.all(derived_seeds >= 0),
        "augmentation derived seed is outside the uint63 contract",
    )
    _require(
        np.allclose(rows["sample_weight"].to_numpy(float), 0.5, rtol=0.0, atol=0.0),
        "synthetic sample weights differ from 0.5",
    )
    _require(
        rows["source_hash"].astype(str).eq(str(release_identity["source_member_sha256"])).all(),
        "augmentation source hash changed",
    )
    config_hash = experiment_config_sha256(config)
    _require(
        rows["experiment_config_sha256"].astype(str).eq(config_hash).all(),
        "augmentation experiment config hash changed",
    )
    inner_mask = rows["inner_fold"].astype(int).ne(-1)
    _require(
        rows.loc[inner_mask, "config_hash"].astype(str).eq(config_hash).all(),
        "inner augmentation generator config is not the experiment config",
    )
    _require(
        rows.loc[~inner_mask, "config_hash"]
        .astype(str)
        .eq(str(release_identity["preparation_config_sha256"]))
        .all(),
        "outer augmentation generator config is not the pinned preparation config",
    )
    _require(
        manifest.get("counts", {}).get("augmentation_contexts") == len(contexts)
        and manifest.get("counts", {}).get("augmentation_provenance_rows") == len(rows),
        "manifest augmentation counts changed",
    )

    audit_index = eligible_audit.set_index(
        eligible_audit["trial_uid"].astype(str), drop=False
    )
    assignment_groups = {
        (str(scheme), int(outer), int(inner)): frame
        for (scheme, outer, inner), frame in assignments.groupby(
            ["scheme", "outer_fold", "inner_fold"], sort=False
        )
    }
    cmixup_sigma_cache: dict[tuple[str, int, int], float] = {}
    total_rows = 0
    for context in contexts.itertuples(index=False):
        scheme = str(context.scheme)
        outer = int(context.outer_fold)
        inner = int(context.inner_fold)
        recipe = str(context.recipe)
        assignment = assignment_groups[(scheme, outer, inner)]
        train_ids = set(
            assignment.loc[assignment["role"].eq("train"), "trial_uid"].astype(str)
        )
        expected_hash = _uid_hash(train_ids)
        split_context = str(assignment["split_context"].iloc[0])
        frame = rows.loc[
            rows["split_scheme"].astype(str).eq(scheme)
            & rows["outer_fold"].astype(int).eq(outer)
            & rows["inner_fold"].astype(int).eq(inner)
            & rows["augmentation_method"].astype(str).eq(recipe)
        ].copy()
        total_rows += len(frame)
        _require(
            len(frame) == len(train_ids) == int(context.additional_rows),
            "augmentation row provenance is not 1:1 with assignment train role",
        )
        _require(
            frame["split_context"].astype(str).eq(split_context).all(),
            "augmentation split_context differs from assignments",
        )
        _require(
            frame["context_real_train_uid_sha256"].astype(str).eq(expected_hash).all()
            and str(context.real_train_uid_sha256) == expected_hash,
            "augmentation real-train UID hash differs from assignments",
        )
        _require(
            frame["context_task_hash"].astype(str).eq(str(context.task_hash)).all()
            and frame["context_parquet_sha256"].astype(str).eq(str(context.parquet_sha256)).all(),
            "augmentation row provenance differs from its context index",
        )
        expected_task_hash = stable_hash(
            {
                "contract": EXPERIMENT_CONTRACT,
                "release_sha256": str(release_identity["archive_sha256"]),
                "config_sha256": config_hash,
                "scheme": scheme,
                "outer_fold": outer,
                "inner_fold": inner,
                "recipe": recipe,
            }
        )
        _require(
            str(context.task_hash) == expected_task_hash,
            "augmentation context task hash cannot be reproduced",
        )
        train = audit_index.loc[sorted(train_ids)].copy()
        generator_config_hash = (
            str(release_identity["preparation_config_sha256"])
            if inner == -1
            else config_hash
        )
        augmentation_seed = (
            BASE_SEED
            if inner == -1
            else context_seed(
                scheme=scheme,
                outer_fold=outer,
                inner_fold=inner,
                recipe=recipe,
                base_seed=BASE_SEED,
            )
        )
        _validate_seeded_augmentation_provenance(
            frame,
            train,
            scheme=scheme,
            outer_fold=outer,
            inner_fold=inner,
            recipe=recipe,
            augmentation_seed=augmentation_seed,
            source_hash=str(release_identity["source_member_sha256"]),
            generator_config_hash=generator_config_hash,
            generator_joint_column="generator_joint_parent_distance_deg",
        )
        parents_a = frame["parent_a_trial_uid"].astype(str)
        parents_b = frame["parent_b_trial_uid"].astype(str)
        _require(
            set(parents_a).issubset(train_ids) and set(parents_b).issubset(train_ids),
            "augmentation parent is not in its assignment train role",
        )
        _require(
            str(context.parent_uid_sha256)
            == _uid_hash(
                frame.loc[:, ["parent_a_trial_uid", "parent_b_trial_uid"]]
                .astype(str)
                .to_numpy()
                .ravel()
                .tolist()
            ),
            "augmentation context parent UID hash cannot be reproduced",
        )
        try:
            parent_a = audit_index.loc[parents_a]
            parent_b = audit_index.loc[parents_b]
        except KeyError as exc:
            raise ExperimentValidationError(
                "augmentation parent is absent from eligible-real audit"
            ) from exc
        planned_distance = np.linalg.norm(
            parent_a.loc[:, list(PLANNED_COLUMNS)].to_numpy(float)
            - parent_b.loc[:, list(PLANNED_COLUMNS)].to_numpy(float),
            axis=1,
        )
        label_distance = np.linalg.norm(
            parent_a.loc[:, list(LABEL_COLUMNS)].to_numpy(float)
            - parent_b.loc[:, list(LABEL_COLUMNS)].to_numpy(float),
            axis=1,
        )
        joint_distance = np.rad2deg(
            np.max(
                np.abs(
                    parent_a.loc[:, list(ENCODER_COLUMNS)].to_numpy(float)
                    - parent_b.loc[:, list(ENCODER_COLUMNS)].to_numpy(float)
                ),
                axis=1,
            )
        )
        _require(
            np.allclose(
                frame["planned_parent_distance_mm"].to_numpy(float),
                planned_distance,
                rtol=1e-12,
                atol=1e-9,
            )
            and np.allclose(
                frame["label_parent_distance_mm"].to_numpy(float),
                label_distance,
                rtol=1e-12,
                atol=1e-9,
            )
            and np.allclose(
                frame["joint_parent_distance_deg"].to_numpy(float),
                joint_distance,
                rtol=1e-12,
                atol=1e-9,
            ),
            "augmentation parent distances cannot be reproduced from measured real rows",
        )
        _require(
            np.all(planned_distance <= 100.0 + 1e-12),
            "augmentation planned-distance constraint was violated",
        )
        if recipe == "local_interp_v2":
            context_sigma = context.cmixup_sigma_mm
            if isinstance(context_sigma, np.ndarray):
                context_sigma = context_sigma.tolist()
            _require(
                set(np.unique(frame["mix_lambda"].to_numpy(float))).issubset(
                    {0.25, 0.5, 0.75}
                ),
                "local interpolation lambda set changed",
            )
            _require(
                not frame["joint_constraint_applied"].astype(bool).any()
                and frame["generator_joint_parent_distance_deg"].isna().all()
                and frame["cmixup_sigma_mm"].isna().all(),
                "local interpolation incorrectly claims C-Mixup constraints",
            )
            _require(
                isinstance(context_sigma, list) and len(context_sigma) == 0,
                "local interpolation context unexpectedly records C-Mixup sigma",
            )
            for column in ("segment_id", "set_id", "repeat_index"):
                _require(
                    np.array_equal(
                        parent_a[column].to_numpy(), parent_b[column].to_numpy()
                    ),
                    f"local interpolation parent {column} group changed",
                )
            grouped_ids = {
                key: group["trial_uid"].astype(str).tolist()
                for key, group in train.groupby(
                    ["segment_id", "set_id", "repeat_index"], sort=False
                )
            }
            allowed_partners: dict[str, set[str]] = {}
            for anchor_id in sorted(set(parents_a)):
                anchor = audit_index.loc[anchor_id]
                group_key = (
                    anchor["segment_id"],
                    anchor["set_id"],
                    anchor["repeat_index"],
                )
                candidates = [
                    value for value in grouped_ids[group_key] if value != anchor_id
                ]
                candidate_frame = audit_index.loc[candidates]
                distances = np.linalg.norm(
                    candidate_frame.loc[:, list(PLANNED_COLUMNS)].to_numpy(float)
                    - anchor.loc[list(PLANNED_COLUMNS)].to_numpy(float),
                    axis=1,
                )
                ranked = sorted(
                    (
                        (str(candidate), float(distance))
                        for candidate, distance in zip(
                            candidates, distances, strict=True
                        )
                        if distance <= 100.0 + 1e-12
                    ),
                    key=lambda item: (item[1], item[0]),
                )[:5]
                allowed_partners[anchor_id] = {item[0] for item in ranked}
            _require(
                all(
                    partner in allowed_partners[anchor]
                    for anchor, partner in zip(parents_a, parents_b, strict=True)
                ),
                "local interpolation parent is outside the five nearest admissible neighbours",
            )
        else:
            _require(
                frame["joint_constraint_applied"].astype(bool).all(),
                "C-Mixup joint constraint marker changed",
            )
            _require(
                np.all(joint_distance <= 20.0 + 1e-12)
                and np.allclose(
                    frame["generator_joint_parent_distance_deg"].to_numpy(float),
                    joint_distance,
                    rtol=1e-12,
                    atol=1e-9,
                ),
                "C-Mixup joint-distance constraint was violated",
            )
            sigma_key = (scheme, outer, inner)
            if sigma_key not in cmixup_sigma_cache:
                cmixup_sigma_cache[sigma_key] = _recompute_cmixup_sigma(
                    train.reset_index(drop=True)
                )
            sigma = cmixup_sigma_cache[sigma_key]
            context_sigma = context.cmixup_sigma_mm
            if isinstance(context_sigma, np.ndarray):
                context_sigma = context_sigma.tolist()
            _require(
                np.allclose(
                    frame["cmixup_sigma_mm"].to_numpy(float),
                    sigma,
                    rtol=1e-12,
                    atol=1e-9,
                ),
                "C-Mixup sigma is not reproducible from assignment train rows",
            )
            _require(
                isinstance(context_sigma, list)
                and len(context_sigma) == 1
                and np.isclose(
                    float(context_sigma[0]), sigma, rtol=1e-12, atol=1e-9
                ),
                "augmentation context C-Mixup sigma changed",
            )
    _require(total_rows == len(rows), "augmentation row provenance has an unknown context")


def _validate_final_selection(
    trace: pd.DataFrame,
    saved_trace: pd.DataFrame,
    gates: pd.DataFrame,
    cells: list[Any],
    config: dict[str, Any],
    final_metadata: dict[str, Any],
) -> None:
    expected_trace, expected_selection = _aggregate_inner_selection(
        trace, cells, gates, config
    )
    try:
        pd.testing.assert_frame_equal(
            saved_trace.reset_index(drop=True),
            expected_trace.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as exc:
        raise ExperimentValidationError(
            f"final selection trace cannot be reproduced: {exc}"
        ) from exc
    _require(
        final_metadata.get("selection") == expected_selection,
        "final model selection cannot be reproduced from inner scores and gates",
    )


def _validate_final_synthetic(
    frame: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    recipe: str,
    config_hash: str,
    source_hash: str,
    scheme: str,
    outer_fold: int,
    inner_fold: int,
    seed_inner_fold: int,
) -> None:
    required = {
        "synthetic_id",
        "augmentation_method",
        "parent_a_trial_uid",
        "parent_b_trial_uid",
        "mix_lambda",
        "planned_parent_distance_mm",
        "label_parent_distance_mm",
        "joint_parent_distance_deg",
        "cmixup_sigma_mm",
        "source_hash",
        "config_hash",
        "experiment_config_sha256",
        "sample_weight",
    }
    _require(required <= set(frame.columns), "final synthetic provenance schema is incomplete")
    _require(len(frame) == len(cohort), "final synthetic data is not 1:1")
    _require(
        frame["synthetic_id"].astype(str).nunique() == len(frame),
        "final synthetic IDs are not unique",
    )
    parent_a_ids = frame["parent_a_trial_uid"].astype(str)
    parent_b_ids = frame["parent_b_trial_uid"].astype(str)
    cohort_ids = set(cohort["trial_uid"].astype(str))
    _require(
        set(parent_a_ids).issubset(cohort_ids) and set(parent_b_ids).issubset(cohort_ids),
        "final synthetic parent is outside its measured-real cohort",
    )
    _require(
        (parent_a_ids != parent_b_ids).all(),
        "final synthetic data contains a self-pair",
    )
    _require(
        frame["augmentation_method"].astype(str).eq(recipe).all()
        and frame["source_hash"].astype(str).eq(source_hash).all()
        and frame["config_hash"].astype(str).eq(config_hash).all()
        and frame["experiment_config_sha256"].astype(str).eq(config_hash).all(),
        "final synthetic method/source/config provenance changed",
    )
    _validate_seeded_augmentation_provenance(
        frame,
        cohort,
        scheme=scheme,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        recipe=recipe,
        augmentation_seed=context_seed(
            scheme=scheme,
            outer_fold=outer_fold,
            inner_fold=seed_inner_fold,
            recipe=recipe,
            base_seed=BASE_SEED,
        ),
        source_hash=source_hash,
        generator_config_hash=config_hash,
        generator_joint_column="joint_parent_distance_deg",
    )
    _require(
        np.allclose(frame["sample_weight"].to_numpy(float), 0.5, rtol=0.0, atol=0.0),
        "final synthetic weights differ from 0.5",
    )
    _require(
        np.isfinite(frame["mix_lambda"].to_numpy(float)).all()
        and frame["mix_lambda"].astype(float).between(0.0, 1.0, inclusive="neither").all(),
        "final synthetic lambda is outside (0, 1)",
    )
    index = cohort.set_index(cohort["trial_uid"].astype(str), drop=False)
    parent_a = index.loc[parent_a_ids]
    parent_b = index.loc[parent_b_ids]
    planned = np.linalg.norm(
        parent_a.loc[:, list(PLANNED_COLUMNS)].to_numpy(float)
        - parent_b.loc[:, list(PLANNED_COLUMNS)].to_numpy(float),
        axis=1,
    )
    labels = np.linalg.norm(
        parent_a.loc[:, list(LABEL_COLUMNS)].to_numpy(float)
        - parent_b.loc[:, list(LABEL_COLUMNS)].to_numpy(float),
        axis=1,
    )
    joints = np.rad2deg(
        np.max(
            np.abs(
                parent_a.loc[:, list(ENCODER_COLUMNS)].to_numpy(float)
                - parent_b.loc[:, list(ENCODER_COLUMNS)].to_numpy(float)
            ),
            axis=1,
        )
    )
    _require(
        np.allclose(frame["planned_parent_distance_mm"].to_numpy(float), planned, rtol=1e-12, atol=1e-9)
        and np.allclose(frame["label_parent_distance_mm"].to_numpy(float), labels, rtol=1e-12, atol=1e-9)
        and np.all(planned <= 100.0 + 1e-12),
        "final synthetic parent distances cannot be reproduced",
    )
    if recipe == "local_interp_v2":
        _require(
            set(np.unique(frame["mix_lambda"].to_numpy(float))).issubset(
                {0.25, 0.5, 0.75}
            ),
            "final local interpolation lambda set changed",
        )
        _require(
            frame["joint_parent_distance_deg"].isna().all()
            and frame["cmixup_sigma_mm"].isna().all(),
            "final local interpolation claims C-Mixup distances",
        )
        for column in ("segment_id", "set_id", "repeat_index"):
            _require(
                np.array_equal(parent_a[column].to_numpy(), parent_b[column].to_numpy()),
                f"final local interpolation parent {column} group changed",
            )
    else:
        _require(
            np.all(joints <= 20.0 + 1e-12)
            and np.allclose(
                frame["joint_parent_distance_deg"].to_numpy(float),
                joints,
                rtol=1e-12,
                atol=1e-9,
            ),
            "final C-Mixup joint-distance constraint changed",
        )
        sigma = _recompute_cmixup_sigma(cohort.reset_index(drop=True))
        _require(
            np.allclose(
                frame["cmixup_sigma_mm"].to_numpy(float),
                sigma,
                rtol=1e-12,
                atol=1e-9,
            ),
            "final C-Mixup sigma is not train-only reproducible",
        )


def _validate_final(
    run: Path,
    manifest: dict[str, Any],
    eligible_audit: pd.DataFrame,
    cells: list[Any],
    config: dict[str, Any],
    release_identity: dict[str, Any],
) -> dict[str, Any]:
    metadata = read_json_object(run / "final" / "model_metadata.json")
    _require(metadata.get("contract") == EXPERIMENT_CONTRACT, "final metadata contract changed")
    _require(metadata.get("independent_test_score_available") is False, "final model claims a test score")
    model_path = run / "final" / "model.joblib"
    _require(metadata["model"]["sha256"] == sha256_file(model_path), "final model hash changed")
    model = joblib.load(model_path)
    _require(isinstance(model, SerializableEndpointEnsemble), "final model wrapper type changed")
    _require(
        model.contract_fingerprint() == metadata["model"]["contract_fingerprint"],
        "final model contract fingerprint changed",
    )
    selection = metadata.get("selection", {})
    selected_cell = next(
        (cell for cell in cells if cell.cell_id == selection.get("cell_id")), None
    )
    _require(selected_cell is not None, "final selected cell is not in the experiment matrix")
    expected_features = resolve_feature_columns(
        selected_cell.feature_view,
        config=config,
        release_feature_sets=feature_views(),
    )
    _require(
        tuple(model.feature_names) == tuple(expected_features)
        and model.model_name == selected_cell.model
        and model.recipe == selected_cell.recipe,
        "final model family/recipe/feature allow-list changed",
    )
    eligible_hash = _uid_hash(eligible_audit["trial_uid"].astype(str))
    expected_members = 3 if selected_cell.model == "mlp" else 1
    _require(len(model.members) == expected_members, "final ensemble member count changed")
    final_preprocessors: list[dict[str, Any]] = []
    for member in model.members:
        _require(member.preprocessor is not None, "deployable final member lacks preprocessing audit")
        preprocessing = member.preprocessor.metadata()
        final_preprocessors.append(preprocessing)
        _validate_preprocessor_state(
            preprocessing,
            eligible_audit,
            feature_columns=expected_features,
            scale_features=selected_cell.model in {"ridge", "mlp"},
            expected_uid_hash=eligible_hash,
            split_context="final_full:all_eligible_real",
            forbidden_uid_count=0,
        )
    expected_added = len(eligible_audit) if selected_cell.recipe in AUGMENTATION_RECIPES else 0
    _require(
        int(metadata["fit"]["real_rows"]) == len(eligible_audit)
        and int(metadata["fit"]["additional_rows"]) == expected_added
        and np.isclose(float(metadata["fit"]["objective_mass"]), len(eligible_audit))
        and metadata["fit"]["feature_columns"] == expected_features,
        "final fit metadata changed",
    )
    _require(
        metadata["fit"].get("preprocessing") == final_preprocessors,
        "final fit preprocessing metadata differs from serialized model members",
    )
    fixture = pd.read_parquet(run / "final" / "inference_fixture.parquet")
    expected = pd.read_parquet(run / "final" / "inference_fixture_predictions.parquet")
    prediction, row_ids = model.predict_frame(fixture)
    _require(row_ids is not None, "inference fixture row IDs disappeared")
    _require(np.array_equal(row_ids, expected["row_id"].to_numpy()), "fixture row ID order changed")
    _require(
        np.array_equal(prediction, expected.loc[:, list(PREDICTION_COLUMNS)].to_numpy(float)),
        "loaded final model does not reproduce fixture predictions",
    )
    prediction_hash = hashlib.sha256(np.asarray(prediction, dtype="<f8").tobytes()).hexdigest()
    _require(
        prediction_hash == metadata["model"]["fixture_prediction_sha256"],
        "fixture prediction hash changed",
    )
    _require(manifest.get("final_model", {}).get("selection") == metadata["selection"], "manifest final selection changed")

    strict_cohort = eligible_audit.loc[
        eligible_audit["strict_qc_eligible"].eq(True)
    ].copy()
    strict_metadata = read_json_object(run / "sensitivity" / "strict_refit_metadata.json")
    strict_path = run / "sensitivity" / "strict_model.joblib"
    _require(
        strict_metadata.get("model_sha256") == sha256_file(strict_path),
        "strict-QC model hash changed",
    )
    strict_model = joblib.load(strict_path)
    _require(
        isinstance(strict_model, SerializableEndpointEnsemble)
        and strict_model.model_name == selected_cell.model
        and strict_model.recipe == selected_cell.recipe
        and tuple(strict_model.feature_names) == tuple(expected_features),
        "strict-QC final model contract changed",
    )
    strict_hash = _uid_hash(strict_cohort["trial_uid"].astype(str))
    for member in strict_model.members:
        _require(member.preprocessor is not None, "strict-QC member lacks preprocessing audit")
        _validate_preprocessor_state(
            member.preprocessor.metadata(),
            strict_cohort,
            feature_columns=expected_features,
            scale_features=selected_cell.model in {"ridge", "mlp"},
            expected_uid_hash=strict_hash,
            split_context="final_strict_qc:resubstitution_only",
            forbidden_uid_count=0,
        )
    strict_added = len(strict_cohort) if selected_cell.recipe in AUGMENTATION_RECIPES else 0
    _require(
        int(strict_metadata.get("real_rows", -1)) == len(strict_cohort)
        and int(strict_metadata.get("additional_rows", -1)) == strict_added
        and np.isclose(float(strict_metadata.get("objective_mass", -1.0)), len(strict_cohort))
        and strict_metadata.get("model") == selected_cell.model
        and strict_metadata.get("recipe") == selected_cell.recipe
        and strict_metadata.get("feature_columns") == expected_features,
        "strict-QC refit metadata changed",
    )
    _require(
        metadata.get("strict_qc_sensitivity") == strict_metadata,
        "final metadata strict-QC summary changed",
    )
    _require(
        int(strict_metadata.get("fit_metrics", {}).get("row_count", -1))
        == len(strict_cohort)
        and np.isfinite(
            np.asarray(
                list(strict_metadata.get("fit_metrics", {}).values()),
                dtype=np.float64,
            )
        ).all(),
        "strict-QC resubstitution metric metadata is incomplete or non-finite",
    )
    strict_prediction = strict_model.predict(
        strict_cohort.loc[:, expected_features].to_numpy(dtype=np.float64)
    )
    strict_expected_metrics = regression_metrics(
        strict_cohort.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64),
        strict_prediction,
    )
    for key, expected_value in strict_expected_metrics.items():
        _require(
            np.isclose(
                float(strict_metadata["fit_metrics"][key]),
                float(expected_value),
                rtol=1e-12,
                atol=1e-12,
            ),
            f"strict-QC refit metric {key} cannot be reproduced",
        )
    full_synthetic_path = run / "final" / "full_synthetic.parquet"
    strict_synthetic_path = run / "sensitivity" / "strict_full_synthetic.parquet"
    if selected_cell.recipe in AUGMENTATION_RECIPES:
        _require(
            full_synthetic_path.is_file() and strict_synthetic_path.is_file(),
            "selected final augmentation members are missing",
        )
        _validate_final_synthetic(
            pd.read_parquet(full_synthetic_path),
            eligible_audit,
            recipe=selected_cell.recipe,
            config_hash=experiment_config_sha256(config),
            source_hash=str(release_identity["source_member_sha256"]),
            scheme="final_full",
            outer_fold=0,
            inner_fold=-1,
            seed_inner_fold=0,
        )
        _validate_final_synthetic(
            pd.read_parquet(strict_synthetic_path),
            strict_cohort,
            recipe=selected_cell.recipe,
            config_hash=experiment_config_sha256(config),
            source_hash=str(release_identity["source_member_sha256"]),
            scheme="final_strict_qc",
            outer_fold=0,
            inner_fold=0,
            seed_inner_fold=0,
        )
    else:
        _require(
            not full_synthetic_path.exists() and not strict_synthetic_path.exists(),
            "real-only final selection unexpectedly contains synthetic training members",
        )
    return metadata


def _validate_report_boundaries(run: Path) -> None:
    report = (run / "reports" / "report.md").read_text(encoding="utf-8").lower()
    card = (run / "MODEL_CARD.md").read_text(encoding="utf-8").lower()
    combined = report + "\n" + card
    for phrase in (
        "exploratory",
        "vive_world",
        "no independent",
        "not robot-base/tcp",
        "formal shapefk",
        "realtime",
        "control",
    ):
        _require(phrase in combined, f"report/model card lacks limitation phrase: {phrase}")


def validate_experiment_run(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    _require(run.is_dir(), f"experiment run directory does not exist: {run}")
    config = read_json_object(run / "config.resolved.json")
    manifest = read_json_object(run / "manifest.json")
    release = read_json_object(run / "release_identity.json")
    assignment_identity = read_json_object(run / "splits" / "assignment_identity.json")
    environment = read_json_object(run / "environment.json")
    outer_preprocessing = read_json_object(
        run / "selection" / "outer_preprocessing.json"
    )
    allowlists = read_json_object(run / "matrix" / "feature_allowlists.json")
    try:
        validate_experiment_config(config)
    except ValueError as exc:
        raise ExperimentValidationError(f"resolved experiment config changed: {exc}") from exc
    _require(manifest.get("contract") == EXPERIMENT_CONTRACT, "manifest contract changed")
    _require(manifest.get("research_status") == "exploratory_v1", "research status changed")
    _require(manifest.get("config_sha256") == experiment_config_sha256(config), "config SHA-256 changed")
    _require(release.get("archive_sha256") == PINNED_RELEASE_SHA256, "parent release SHA-256 changed")
    _require(release.get("train_eligible_rows") == 7_702, "eligible real count changed")
    _require(release.get("strict_qc_rows") == 7_670, "strict-QC real count changed")
    _validate_environment(environment, manifest)
    verify_file_manifest(run, manifest.get("files", []))
    tables = _load_required_tables(run)
    cells = build_experiment_cells(config)
    _validate_feature_allowlists(allowlists, cells, config)
    _validate_assignments(tables["assignments"], assignment_identity, manifest)
    _validate_eligible_audit(
        tables["eligible_audit"],
        tables["assignments"],
        manifest,
        allowlists,
    )
    _validate_oof(
        tables,
        tables["assignments"],
        cells,
        manifest,
        tables["eligible_audit"],
    )
    _validate_selection(
        tables["trace"],
        cells,
        tables["assignments"],
        tables["ensemble_oof"],
        tables["seed_oof"],
        outer_preprocessing,
        config,
        tables["eligible_audit"],
        allowlists,
    )
    _validate_metrics(tables)
    _validate_statistics(tables)
    _validate_augmentation(
        tables["augmentation"],
        tables["augmentation_rows"],
        tables["eligible_audit"],
        tables["assignments"],
        config,
        release,
        manifest,
    )
    final = _validate_final(
        run,
        manifest,
        tables["eligible_audit"],
        cells,
        config,
        release,
    )
    _validate_final_selection(
        tables["trace"],
        tables["final_trace"],
        tables["gates"],
        cells,
        config,
        final,
    )
    _validate_report_boundaries(run)
    audit = read_json_object(run / "analysis" / "leakage_audit.json")
    zero_checks = [
        key
        for key in audit
        if key.endswith("rows")
        or "duplicate" in key
        or "wrong_" in key
        or "overlap" in key
        or "leakage" in key
    ]
    zero_checks = [key for key in zero_checks if not key.startswith(("expected_", "actual_", "assignment_"))]
    _require(all(audit[key] == 0 for key in zero_checks), f"leakage audit is non-zero: {audit}")
    return {
        "status": "ok",
        "contract": EXPERIMENT_CONTRACT,
        "assignment_rows": int(len(tables["assignments"])),
        "contexts": int(tables["assignments"]["split_context"].nunique()),
        "cells": len(cells),
        "ensemble_oof_rows": int(len(tables["ensemble_oof"])),
        "seed_oof_rows": int(len(tables["seed_oof"])),
        "paired_comparisons": int(len(tables["paired"])),
        "final_cell": final["selection"]["cell_id"],
        "manifest_files": len(manifest["files"]),
    }


__all__ = ["ExperimentValidationError", "validate_experiment_run"]

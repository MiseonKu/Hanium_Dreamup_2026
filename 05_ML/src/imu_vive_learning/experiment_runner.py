"""Nested group-CV runner for ``alphaf.endpoint_regression.experiment.v1``.

The public runner is deliberately conservative: the preparation ZIP is
authenticated before any model fit, inner augmentation is cached per declared
real-train context, every estimator transform is fit on measured train rows,
and an outer-test prediction is emitted only after inner selection is complete.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import warnings
from typing import Any

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from .experiment_augmentation import (
    AugmentationSettings,
    CMIXUP_METHOD,
    LOCAL_METHOD,
    build_training_recipe,
    generate_final_full_synthetic,
    materialize_inner_synthetic,
)
from .experiment_config import (
    EXPERIMENT_CONTRACT,
    experiment_config_sha256,
    public_experiment_config,
    resolve_release_path,
    validate_experiment_config,
)
from .experiment_ensemble import SerializableEndpointEnsemble
from .experiment_io import (
    atomic_joblib_dump,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    environment_payload,
    manifest_files,
    prepare_experiment_directory,
    read_json_object,
    sha256_file,
    stable_hash,
    validate_endpoint_eval_environment,
)
from .experiment_matrix import (
    ExperimentCell,
    build_experiment_cells,
    resolve_feature_columns,
    validate_cell_feature_policy,
)
from .experiment_metrics import (
    cell_balanced_metrics,
    paired_group_effects,
    paired_multiway_bootstrap,
    regression_metrics,
)
from .experiment_models import (
    augmentation_gate,
    configuration_id,
    fit_endpoint_regressor,
    model_parameter_grid,
    paired_seed_sequence,
    select_best_configuration,
)
from .experiment_release import ReleaseData, verify_and_load_release
from .experiment_reporting import write_experiment_report
from .experiment_splits import (
    assignment_sha256,
    build_experiment_assignments,
    context_assignments,
    validate_experiment_assignments,
)
from .features import ENCODER_COLUMNS, LABEL_COLUMNS, PLANNED_COLUMNS


SCHEMES = ("segment_loso", "target_group")
AUGMENTATION_RECIPES = (LOCAL_METHOD, CMIXUP_METHOD)
PREDICTION_COLUMNS = (
    "pred_vive_world_x_mm",
    "pred_vive_world_y_mm",
    "pred_vive_world_z_mm",
)
TRUE_COLUMNS = (
    "true_vive_world_x_mm",
    "true_vive_world_y_mm",
    "true_vive_world_z_mm",
)

ELIGIBLE_REAL_AUDIT_COLUMNS = (
    "trial_uid",
    "segment_id",
    "target_id",
    "set_id",
    "repeat_index",
    "strict_qc_eligible",
    *PLANNED_COLUMNS,
    *ENCODER_COLUMNS,
    *LABEL_COLUMNS,
)

AUGMENTATION_ROW_PROVENANCE_COLUMNS = (
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
    "joint_parent_distance_deg",
    "cmixup_sigma_mm",
    "derived_seed",
    "source_hash",
    "config_hash",
    "experiment_config_sha256",
    "sample_weight",
)


class ExperimentRunError(RuntimeError):
    """Raised when an experiment task cannot satisfy its frozen contract."""


def _git_revision(workspace_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=workspace_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExperimentRunError("cannot resolve the Git revision for experiment provenance") from exc
    revision = result.stdout.strip()
    if len(revision) < 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ExperimentRunError(f"invalid Git revision: {revision!r}")
    if status.stdout.strip():
        preview = status.stdout.strip().splitlines()[:10]
        raise ExperimentRunError(
            "full experiment requires a clean tracked/untracked Git worktree; "
            f"dirty entries: {preview}"
        )
    return revision


def _uid_hash(values: Sequence[Any]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _params_json(params: Mapping[str, Any]) -> str:
    return json.dumps(
        params,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: list(value) if isinstance(value, tuple) else value,
    )


def _outer_groups(config: Mapping[str, Any], scheme: str) -> list[int]:
    return [
        int(value)
        for value in config["splits"]["schemes"][scheme]["outer_groups"]
    ]


def _inner_groups(config: Mapping[str, Any], scheme: str, outer_fold: int) -> list[int]:
    return [value for value in _outer_groups(config, scheme) if value != int(outer_fold)]


def _cache_paths(
    cache_root: Path,
    *,
    scheme: str,
    outer_fold: int,
    inner_fold: int,
    recipe: str,
) -> tuple[Path, Path]:
    inner_label = "refit" if int(inner_fold) == -1 else str(int(inner_fold))
    base = (
        cache_root
        / "recipes"
        / scheme
        / f"outer_{int(outer_fold)}"
        / f"inner_{inner_label}"
    )
    return base / f"{recipe}.parquet", base / f"{recipe}.json"


def _recipe_cache_task(
    release: ReleaseData,
    assignments: pd.DataFrame,
    *,
    cache_root: Path,
    scheme: str,
    outer_fold: int,
    inner_fold: int,
    recipe: str,
    config_hash: str,
) -> dict[str, Any]:
    parquet_path, metadata_path = _cache_paths(
        cache_root,
        scheme=scheme,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
        recipe=recipe,
    )
    task_identity = {
        "contract": EXPERIMENT_CONTRACT,
        "release_sha256": release.archive_sha256,
        "config_sha256": config_hash,
        "scheme": scheme,
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "recipe": recipe,
    }
    task_hash = stable_hash(task_identity)
    if metadata_path.is_file() and parquet_path.is_file():
        metadata = read_json_object(metadata_path)
        if (
            metadata.get("task_hash") == task_hash
            and metadata.get("parquet_sha256") == sha256_file(parquet_path)
        ):
            return metadata
        raise ExperimentRunError(
            f"recipe cache identity/hash mismatch: {metadata_path}; remove the run cache"
        )

    training = build_training_recipe(
        release,
        assignments,
        scheme=scheme,
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
        recipe=recipe,
        experiment_config_sha256=config_hash,
    )
    additional = training.additional.copy()
    if len(additional) != len(training.real):
        raise ExperimentRunError("cached augmentation must remain exactly 1:1")
    atomic_write_parquet(parquet_path, additional)
    parent_columns = ["parent_a_trial_uid", "parent_b_trial_uid"]
    parent_hash = _uid_hash(
        additional.loc[:, parent_columns].astype(str).to_numpy().ravel().tolist()
    )
    sigma_values: list[float] = []
    if "cmixup_sigma_mm" in additional.columns:
        sigma_values = sorted(
            float(value)
            for value in additional["cmixup_sigma_mm"].dropna().unique().tolist()
        )
    metadata = {
        **task_identity,
        "task_hash": task_hash,
        "real_train_rows": int(len(training.real)),
        "additional_rows": int(len(additional)),
        "objective_mass": float(training.combined["sample_weight"].sum()),
        "real_train_uid_sha256": _uid_hash(training.real["trial_uid"].astype(str)),
        "parent_uid_sha256": parent_hash,
        "cmixup_sigma_mm": sigma_values,
        "parquet_sha256": sha256_file(parquet_path),
        "parquet_bytes": parquet_path.stat().st_size,
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def ensure_recipe_caches(
    release: ReleaseData,
    assignments: pd.DataFrame,
    config: dict[str, Any],
    *,
    cache_root: Path,
    jobs: int,
) -> list[dict[str, Any]]:
    config_hash = experiment_config_sha256(config)
    tasks: list[tuple[str, int, int, str]] = []
    for scheme in SCHEMES:
        for outer in _outer_groups(config, scheme):
            for inner in [-1, *_inner_groups(config, scheme, outer)]:
                for recipe in AUGMENTATION_RECIPES:
                    tasks.append((scheme, outer, inner, recipe))
    with threadpool_limits(limits=1):
        rows = Parallel(n_jobs=jobs, backend="threading")(
            delayed(_recipe_cache_task)(
                release,
                assignments,
                cache_root=cache_root,
                scheme=scheme,
                outer_fold=outer,
                inner_fold=inner,
                recipe=recipe,
                config_hash=config_hash,
            )
            for scheme, outer, inner, recipe in tasks
        )
    rows.sort(
        key=lambda row: (
            row["scheme"],
            int(row["outer_fold"]),
            int(row["inner_fold"]),
            row["recipe"],
        )
    )
    return rows


def load_recipe_cache_frames(
    cache_root: Path,
    cache_index: Sequence[Mapping[str, Any]],
    *,
    required_columns: Sequence[str],
    jobs: int,
) -> dict[tuple[str, int, int, str], pd.DataFrame]:
    """Load each verified synthetic context once for shared threaded fitting."""

    columns = list(
        dict.fromkeys(
            [
                *required_columns,
                *LABEL_COLUMNS,
                "parent_a_trial_uid",
                "parent_b_trial_uid",
            ]
        )
    )

    def load_one(row: Mapping[str, Any]) -> tuple[tuple[str, int, int, str], pd.DataFrame]:
        key = (
            str(row["scheme"]),
            int(row["outer_fold"]),
            int(row["inner_fold"]),
            str(row["recipe"]),
        )
        path, _ = _cache_paths(
            cache_root,
            scheme=key[0],
            outer_fold=key[1],
            inner_fold=key[2],
            recipe=key[3],
        )
        frame = pd.read_parquet(path, columns=columns)
        if len(frame) != int(row["additional_rows"]):
            raise ExperimentRunError(f"cached recipe row count changed: {path}")
        return key, frame

    loaded = Parallel(n_jobs=jobs, backend="threading")(
        delayed(load_one)(row) for row in cache_index
    )
    result = dict(loaded)
    if len(result) != len(cache_index):
        raise ExperimentRunError("duplicate verified recipe cache context")
    return result


def _eligible_real_audit_table(
    eligible: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Return the minimal measured-real table needed for independent audits."""

    columns = list(
        dict.fromkeys([*ELIGIBLE_REAL_AUDIT_COLUMNS, *map(str, feature_columns)])
    )
    missing = sorted(set(columns) - set(eligible.columns))
    if missing:
        raise ExperimentRunError(
            f"eligible real audit columns are missing from the release: {missing}"
        )
    audit = eligible.loc[:, columns].copy()
    audit.insert(
        3,
        "target_group",
        audit["target_id"].astype(str).str[1:].astype(int).to_numpy() % 5,
    )
    audit.sort_values("trial_uid", kind="mergesort", inplace=True)
    audit.reset_index(drop=True, inplace=True)
    if audit["trial_uid"].astype(str).duplicated().any():
        raise ExperimentRunError("eligible real audit table contains duplicate trial_uid")
    numeric_columns = list(
        dict.fromkeys(
            [*PLANNED_COLUMNS, *ENCODER_COLUMNS, *LABEL_COLUMNS, *feature_columns]
        )
    )
    numeric = audit.loc[:, numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ExperimentRunError("eligible real audit table contains non-finite values")
    return audit


def _public_augmentation_row_provenance(
    cache_root: Path,
    cache_index: Sequence[Mapping[str, Any]],
    eligible_audit: pd.DataFrame,
    *,
    jobs: int,
) -> pd.DataFrame:
    """Expose row-level parent evidence that is otherwise hidden in the cache."""

    audit_index = eligible_audit.set_index(
        eligible_audit["trial_uid"].astype(str), drop=False
    )

    def load_one(metadata: Mapping[str, Any]) -> pd.DataFrame:
        scheme = str(metadata["scheme"])
        outer = int(metadata["outer_fold"])
        inner = int(metadata["inner_fold"])
        recipe = str(metadata["recipe"])
        path, _ = _cache_paths(
            cache_root,
            scheme=scheme,
            outer_fold=outer,
            inner_fold=inner,
            recipe=recipe,
        )
        try:
            public = pd.read_parquet(
                path, columns=list(AUGMENTATION_ROW_PROVENANCE_COLUMNS)
            )
        except (KeyError, ValueError) as exc:
            raise ExperimentRunError(
                f"cached row provenance is incomplete for {path}"
            ) from exc
        if len(public) != int(metadata["additional_rows"]):
            raise ExperimentRunError(f"cached provenance row count changed: {path}")
        if not (
            public["split_scheme"].astype(str).eq(scheme).all()
            and public["outer_fold"].astype(int).eq(outer).all()
            and public["inner_fold"].astype(int).eq(inner).all()
            and public["augmentation_method"].astype(str).eq(recipe).all()
        ):
            raise ExperimentRunError(f"cached provenance context changed: {path}")

        parent_a = public["parent_a_trial_uid"].astype(str)
        parent_b = public["parent_b_trial_uid"].astype(str)
        try:
            encoder_a = audit_index.loc[parent_a, list(ENCODER_COLUMNS)].to_numpy(
                dtype=np.float64
            )
            encoder_b = audit_index.loc[parent_b, list(ENCODER_COLUMNS)].to_numpy(
                dtype=np.float64
            )
        except KeyError as exc:
            raise ExperimentRunError(
                f"cached provenance references an unknown eligible parent: {path}"
            ) from exc
        audited_joint_distance = np.rad2deg(
            np.max(np.abs(encoder_a - encoder_b), axis=1)
        )
        generator_joint_distance = public[
            "joint_parent_distance_deg"
        ].to_numpy(dtype=np.float64)
        if recipe == CMIXUP_METHOD and not np.allclose(
            generator_joint_distance,
            audited_joint_distance,
            rtol=1e-12,
            atol=1e-10,
        ):
            raise ExperimentRunError(
                f"C-Mixup cached joint provenance cannot be reproduced: {path}"
            )
        public["generator_joint_parent_distance_deg"] = generator_joint_distance
        public["joint_parent_distance_deg"] = audited_joint_distance
        public["joint_constraint_applied"] = recipe == CMIXUP_METHOD
        public["context_task_hash"] = str(metadata["task_hash"])
        public["context_parquet_sha256"] = str(metadata["parquet_sha256"])
        public["context_real_train_uid_sha256"] = str(
            metadata["real_train_uid_sha256"]
        )
        return public

    frames = Parallel(n_jobs=jobs, backend="threading")(
        delayed(load_one)(metadata) for metadata in cache_index
    )
    result = pd.concat(frames, ignore_index=True)
    result.sort_values(
        [
            "split_scheme",
            "outer_fold",
            "inner_fold",
            "augmentation_method",
            "synthetic_id",
        ],
        kind="mergesort",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    if result.duplicated(
        [
            "split_scheme",
            "outer_fold",
            "inner_fold",
            "augmentation_method",
            "synthetic_id",
        ]
    ).any():
        raise ExperimentRunError("public augmentation provenance has duplicate rows")
    return result


def _context_lookup(assignments: pd.DataFrame) -> dict[tuple[str, int, int], pd.DataFrame]:
    result: dict[tuple[str, int, int], pd.DataFrame] = {}
    for (scheme, outer, inner), frame in assignments.groupby(
        ["scheme", "outer_fold", "inner_fold"], sort=False
    ):
        key = (str(scheme), int(outer), int(inner))
        result[key] = frame.sort_values("trial_uid", kind="mergesort").reset_index(
            drop=True
        )
    if len(result) != 50:
        raise ExperimentRunError(f"context lookup changed: {len(result)} contexts")
    return result


def _real_rows_for_context(
    eligible_indexed: pd.DataFrame,
    context: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    train_ids = tuple(
        sorted(context.loc[context["role"].eq("train"), "trial_uid"].astype(str))
    )
    forbidden_ids = tuple(
        sorted(context.loc[~context["role"].eq("train"), "trial_uid"].astype(str))
    )
    try:
        real = eligible_indexed.loc[list(train_ids)].copy().reset_index(drop=True)
    except KeyError as exc:
        raise ExperimentRunError("split context references an unknown eligible trial") from exc
    if set(real["trial_uid"].astype(str)) & set(forbidden_ids):
        raise ExperimentRunError("held-out trial leaked into real training rows")
    return real, forbidden_ids


def _training_arrays(
    eligible_indexed: pd.DataFrame,
    context: pd.DataFrame,
    recipe_frames: Mapping[tuple[str, int, int, str], pd.DataFrame],
    *,
    cache_root: Path,
    scheme: str,
    outer_fold: int,
    inner_fold: int,
    recipe: str,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    real, forbidden_ids = _real_rows_for_context(eligible_indexed, context)
    columns = list(feature_columns)
    X_real = (
        real.loc[:, columns].to_numpy(dtype=np.float64)
        if columns
        else np.empty((len(real), 0), dtype=np.float64)
    )
    y_real = real.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    if recipe == "real_only":
        X_train = X_real.copy()
        y_train = y_real.copy()
        weights = np.ones(len(real), dtype=np.float64)
        additional_rows = 0
    elif recipe == "replay":
        X_train = np.concatenate([X_real, X_real.copy()], axis=0)
        y_train = np.concatenate([y_real, y_real.copy()], axis=0)
        weights = np.full(2 * len(real), 0.5, dtype=np.float64)
        additional_rows = len(real)
    elif recipe in AUGMENTATION_RECIPES:
        try:
            additional = recipe_frames[
                (scheme, int(outer_fold), int(inner_fold), recipe)
            ]
        except KeyError as exc:
            raise ExperimentRunError("verified recipe cache frame is missing") from exc
        parent_ids = set(additional["parent_a_trial_uid"].astype(str)) | set(
            additional["parent_b_trial_uid"].astype(str)
        )
        real_ids = set(real["trial_uid"].astype(str))
        if not parent_ids.issubset(real_ids) or parent_ids & set(forbidden_ids):
            raise ExperimentRunError("cached synthetic parent is outside the declared train role")
        X_added = additional.loc[:, columns].to_numpy(dtype=np.float64)
        y_added = additional.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
        X_train = np.concatenate([X_real, X_added], axis=0)
        y_train = np.concatenate([y_real, y_added], axis=0)
        weights = np.full(len(X_train), 0.5, dtype=np.float64)
        additional_rows = len(additional)
    else:
        raise ExperimentRunError(f"unknown training recipe: {recipe}")
    if not (
        np.isfinite(X_train).all()
        and np.isfinite(y_train).all()
        and np.isfinite(weights).all()
    ):
        raise ExperimentRunError("training arrays contain non-finite values")
    if not np.isclose(float(weights.sum()), float(len(real)), rtol=0.0, atol=1e-9):
        raise ExperimentRunError("training objective mass differs from real-only")
    return {
        "real": real,
        "context": context,
        "forbidden_ids": forbidden_ids,
        "X_real": X_real,
        "y_real": y_real,
        "X_train": X_train,
        "y_train": y_train,
        "weights": weights,
        "additional_rows": int(additional_rows),
    }


def _evaluation_rows(
    eligible_indexed: pd.DataFrame,
    context: pd.DataFrame,
    *,
    role: str,
) -> pd.DataFrame:
    ids = tuple(sorted(context.loc[context["role"].eq(role), "trial_uid"].astype(str)))
    if not ids:
        raise ExperimentRunError(f"split context has no {role} rows")
    result = eligible_indexed.loc[list(ids)].copy().reset_index(drop=True)
    if result["trial_uid"].astype(str).duplicated().any():
        raise ExperimentRunError(f"duplicate real {role} trial")
    return result


def _fit_models_and_predict(
    *,
    cell: ExperimentCell,
    params: Mapping[str, Any],
    arrays: Mapping[str, Any],
    evaluation: pd.DataFrame,
    feature_columns: Sequence[str],
    split_context: str,
) -> tuple[list[Any], list[np.ndarray], np.ndarray]:
    feature_names = tuple(feature_columns)
    X_eval = (
        evaluation.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
        if feature_names
        else np.empty((len(evaluation), 0), dtype=np.float64)
    )
    seed_count = 3 if cell.model == "mlp" else 1
    supplied_seeds = paired_seed_sequence(split_context, count=seed_count)
    models: list[Any] = []
    predictions: list[np.ndarray] = []
    for supplied_seed in supplied_seeds:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model = fit_endpoint_regressor(
                cell.model,
                params,
                arrays["X_train"],
                arrays["y_train"],
                feature_names=feature_names,
                X_real_train=arrays["X_real"],
                y_real_train=arrays["y_real"],
                real_trial_uids=arrays["real"]["trial_uid"].astype(str).tolist(),
                forbidden_uids=arrays["forbidden_ids"],
                sample_weight=arrays["weights"],
                recipe=cell.recipe,
                split_context=split_context,
                seed=int(supplied_seed),
                allow_planned_diagnostic=cell.diagnostic_only,
            )
        prediction = model.predict(X_eval)
        models.append(model)
        predictions.append(prediction)
    ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
    if not np.isfinite(ensemble).all():
        raise ExperimentRunError("ensemble prediction contains non-finite values")
    return models, predictions, ensemble


def _candidate_trace(
    eligible_indexed: pd.DataFrame,
    contexts: Mapping[tuple[str, int, int], pd.DataFrame],
    recipe_frames: Mapping[tuple[str, int, int, str], pd.DataFrame],
    *,
    cache_root: Path,
    config: dict[str, Any],
    scheme: str,
    outer_fold: int,
    cell: ExperimentCell,
    feature_columns: Sequence[str],
    params: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[float]]:
    rows: list[dict[str, Any]] = []
    inner_scores: list[float] = []
    config_id = configuration_id(cell.model, params)
    for inner_fold in _inner_groups(config, scheme, outer_fold):
        context = contexts[(scheme, int(outer_fold), int(inner_fold))]
        arrays = _training_arrays(
            eligible_indexed,
            context,
            recipe_frames,
            cache_root=cache_root,
            scheme=scheme,
            outer_fold=outer_fold,
            inner_fold=inner_fold,
            recipe=cell.recipe,
            feature_columns=feature_columns,
        )
        split_context = str(context["split_context"].iloc[0])
        validation = _evaluation_rows(eligible_indexed, context, role="validation")
        y_true = validation.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
        models, seed_predictions, ensemble = _fit_models_and_predict(
            cell=cell,
            params=params,
            arrays=arrays,
            evaluation=validation,
            feature_columns=feature_columns,
            split_context=split_context,
        )
        for seed_index, (model, prediction) in enumerate(zip(models, seed_predictions, strict=True)):
            metrics = regression_metrics(y_true, prediction)
            rows.append(
                {
                    "scheme": scheme,
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "split_context": split_context,
                    "cell_id": cell.cell_id,
                    "model": cell.model,
                    "feature_view": cell.feature_view,
                    "recipe": cell.recipe,
                    "config_id": config_id,
                    "params_json": _params_json(params),
                    "prediction_kind": "seed",
                    "seed_index": int(seed_index),
                    "model_seed": int(model.seed),
                    "real_train_rows": int(len(arrays["real"])),
                    "additional_rows": int(arrays["additional_rows"]),
                    "validation_rows": int(len(validation)),
                    "preprocessor_fit_uid_sha256": _uid_hash(
                        arrays["real"]["trial_uid"].astype(str)
                    ),
                    **metrics,
                }
            )
        metrics = regression_metrics(y_true, ensemble)
        inner_scores.append(float(metrics["rmse_3d_mm"]))
        rows.append(
            {
                "scheme": scheme,
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "split_context": split_context,
                "cell_id": cell.cell_id,
                "model": cell.model,
                "feature_view": cell.feature_view,
                "recipe": cell.recipe,
                "config_id": config_id,
                "params_json": _params_json(params),
                "prediction_kind": "ensemble",
                "seed_index": -1,
                "model_seed": -1,
                "real_train_rows": int(len(arrays["real"])),
                "additional_rows": int(arrays["additional_rows"]),
                "validation_rows": int(len(validation)),
                "preprocessor_fit_uid_sha256": _uid_hash(
                    arrays["real"]["trial_uid"].astype(str)
                ),
                **metrics,
            }
        )
    return rows, inner_scores


def _prediction_frame(
    evaluation: pd.DataFrame,
    prediction: np.ndarray,
    *,
    scheme: str,
    outer_fold: int,
    cell: ExperimentCell,
    config_id: str,
    params: Mapping[str, Any],
    prediction_kind: str,
    seed_index: int,
    model_seed: int,
) -> pd.DataFrame:
    truth = evaluation.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    errors = np.linalg.norm(prediction - truth, axis=1)
    return pd.DataFrame(
        {
            "scheme": scheme,
            "outer_fold": int(outer_fold),
            "cell_id": cell.cell_id,
            "model": cell.model,
            "feature_view": cell.feature_view,
            "recipe": cell.recipe,
            "config_id": config_id,
            "params_json": _params_json(params),
            "prediction_kind": prediction_kind,
            "seed_index": int(seed_index),
            "model_seed": int(model_seed),
            "trial_uid": evaluation["trial_uid"].astype(str).to_numpy(),
            "segment_id": evaluation["segment_id"].astype(int).to_numpy(),
            "target_id": evaluation["target_id"].astype(str).to_numpy(),
            "target_group": evaluation["target_id"].astype(str).str[1:].astype(int).to_numpy() % 5,
            "strict_qc_eligible": evaluation["strict_qc_eligible"].astype(bool).to_numpy(),
            TRUE_COLUMNS[0]: truth[:, 0],
            TRUE_COLUMNS[1]: truth[:, 1],
            TRUE_COLUMNS[2]: truth[:, 2],
            PREDICTION_COLUMNS[0]: prediction[:, 0],
            PREDICTION_COLUMNS[1]: prediction[:, 1],
            PREDICTION_COLUMNS[2]: prediction[:, 2],
            "error_3d_mm": errors,
        }
    )


def _task_directory(
    state_root: Path, *, scheme: str, outer_fold: int, cell_id: str
) -> Path:
    return state_root / "outer_tasks" / scheme / f"outer_{outer_fold}" / cell_id


def _load_task_if_complete(task_dir: Path, task_hash: str) -> dict[str, Any] | None:
    summary_path = task_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = read_json_object(summary_path)
    if summary.get("task_hash") != task_hash:
        raise ExperimentRunError(f"resume refused: task hash changed for {task_dir}")
    for name, digest in summary.get("member_sha256", {}).items():
        path = task_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ExperimentRunError(f"resume checkpoint member changed: {path}")
    return summary


def _run_outer_cell(
    eligible_indexed: pd.DataFrame,
    contexts: Mapping[tuple[str, int, int], pd.DataFrame],
    recipe_frames: Mapping[tuple[str, int, int, str], pd.DataFrame],
    *,
    cache_root: Path,
    state_root: Path,
    config: dict[str, Any],
    release_sha256: str,
    assignment_hash: str,
    scheme: str,
    outer_fold: int,
    cell: ExperimentCell,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    task_dir = _task_directory(
        state_root, scheme=scheme, outer_fold=outer_fold, cell_id=cell.cell_id
    )
    task_identity = {
        "contract": EXPERIMENT_CONTRACT,
        "config_sha256": experiment_config_sha256(config),
        "release_sha256": release_sha256,
        "assignment_sha256": assignment_hash,
        "scheme": scheme,
        "outer_fold": int(outer_fold),
        "cell": cell.as_dict(),
        "feature_columns": list(feature_columns),
        "parameter_grid": model_parameter_grid(cell.model),
    }
    task_hash = stable_hash(task_identity)
    completed = _load_task_if_complete(task_dir, task_hash)
    if completed is not None:
        return completed

    trace_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for params in model_parameter_grid(cell.model):
        rows, inner_scores = _candidate_trace(
            eligible_indexed,
            contexts,
            recipe_frames,
            cache_root=cache_root,
            config=config,
            scheme=scheme,
            outer_fold=outer_fold,
            cell=cell,
            feature_columns=feature_columns,
            params=params,
        )
        trace_rows.extend(rows)
        candidates.append({"params": dict(params), "inner_scores": inner_scores})
    selected = select_best_configuration(cell.model, candidates)
    selected_params = dict(selected["params"])
    selected_config_id = str(selected["config_id"])

    context = contexts[(scheme, int(outer_fold), -1)]
    arrays = _training_arrays(
        eligible_indexed,
        context,
        recipe_frames,
        cache_root=cache_root,
        scheme=scheme,
        outer_fold=outer_fold,
        inner_fold=-1,
        recipe=cell.recipe,
        feature_columns=feature_columns,
    )
    split_context = str(context["split_context"].iloc[0])
    test = _evaluation_rows(eligible_indexed, context, role="test")
    models, seed_predictions, ensemble = _fit_models_and_predict(
        cell=cell,
        params=selected_params,
        arrays=arrays,
        evaluation=test,
        feature_columns=feature_columns,
        split_context=split_context,
    )
    seed_frames = [
        _prediction_frame(
            test,
            prediction,
            scheme=scheme,
            outer_fold=outer_fold,
            cell=cell,
            config_id=selected_config_id,
            params=selected_params,
            prediction_kind="seed",
            seed_index=seed_index,
            model_seed=int(model.seed),
        )
        for seed_index, (model, prediction) in enumerate(
            zip(models, seed_predictions, strict=True)
        )
    ]
    seed_oof = pd.concat(seed_frames, ignore_index=True)
    ensemble_oof = _prediction_frame(
        test,
        ensemble,
        scheme=scheme,
        outer_fold=outer_fold,
        cell=cell,
        config_id=selected_config_id,
        params=selected_params,
        prediction_kind="ensemble",
        seed_index=-1,
        model_seed=-1,
    )
    trace = pd.DataFrame.from_records(trace_rows)
    trace.sort_values(
        ["config_id", "inner_fold", "prediction_kind", "seed_index"],
        kind="mergesort",
        inplace=True,
    )
    trace.reset_index(drop=True, inplace=True)
    task_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(task_dir / "inner_trace.parquet", trace)
    atomic_write_parquet(task_dir / "oof_seed.parquet", seed_oof)
    atomic_write_parquet(task_dir / "oof_ensemble.parquet", ensemble_oof)
    preprocessing = [
        {
            "seed_index": index,
            "model_seed": int(model.seed),
            "metadata": (
                model.fit_metadata
                if model.preprocessor is None
                else {
                    **model.fit_metadata,
                    "preprocessor": model.preprocessor.metadata(),
                }
            ),
        }
        for index, model in enumerate(models)
    ]
    atomic_write_json(task_dir / "preprocessing.json", {"models": preprocessing})
    member_names = (
        "inner_trace.parquet",
        "oof_seed.parquet",
        "oof_ensemble.parquet",
        "preprocessing.json",
    )
    summary = {
        **task_identity,
        "task_hash": task_hash,
        "selected_config_id": selected_config_id,
        "selected_params": selected_params,
        "selection_score_rmse_3d_mm": float(selected["macro_group_rmse_3d_mm"]),
        "tied_candidate_count": int(selected["tied_candidate_count"]),
        "inner_context_count": len(_inner_groups(config, scheme, outer_fold)),
        "outer_real_train_rows": int(len(arrays["real"])),
        "outer_real_train_uid_sha256": _uid_hash(
            arrays["real"]["trial_uid"].astype(str)
        ),
        "outer_additional_rows": int(arrays["additional_rows"]),
        "outer_test_rows": int(len(test)),
        "member_sha256": {
            name: sha256_file(task_dir / name) for name in member_names
        },
    }
    atomic_write_json(task_dir / "summary.json", summary)
    return summary


def _collect_task_outputs(
    *,
    state_root: Path,
    config: dict[str, Any],
    cells: Sequence[ExperimentCell],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    traces: list[pd.DataFrame] = []
    seed_oof: list[pd.DataFrame] = []
    ensemble_oof: list[pd.DataFrame] = []
    preprocessing: list[dict[str, Any]] = []
    for scheme in SCHEMES:
        for outer in _outer_groups(config, scheme):
            for cell in cells:
                task_dir = _task_directory(
                    state_root,
                    scheme=scheme,
                    outer_fold=outer,
                    cell_id=cell.cell_id,
                )
                summary = read_json_object(task_dir / "summary.json")
                for name, digest in summary["member_sha256"].items():
                    if sha256_file(task_dir / name) != digest:
                        raise ExperimentRunError(f"task member hash changed: {task_dir / name}")
                traces.append(pd.read_parquet(task_dir / "inner_trace.parquet"))
                seed_oof.append(pd.read_parquet(task_dir / "oof_seed.parquet"))
                ensemble_oof.append(pd.read_parquet(task_dir / "oof_ensemble.parquet"))
                preprocessor = read_json_object(task_dir / "preprocessing.json")
                preprocessing.append(
                    {
                        "scheme": scheme,
                        "outer_fold": int(outer),
                        "cell_id": cell.cell_id,
                        "selected_config_id": summary["selected_config_id"],
                        "selected_params": summary["selected_params"],
                        "selection_score_rmse_3d_mm": summary[
                            "selection_score_rmse_3d_mm"
                        ],
                        "real_train_rows": summary["outer_real_train_rows"],
                        "real_train_uid_sha256": summary[
                            "outer_real_train_uid_sha256"
                        ],
                        "models": preprocessor["models"],
                    }
                )
    trace = pd.concat(traces, ignore_index=True)
    seeds = pd.concat(seed_oof, ignore_index=True)
    ensemble = pd.concat(ensemble_oof, ignore_index=True)
    trace.sort_values(
        [
            "scheme",
            "outer_fold",
            "cell_id",
            "config_id",
            "inner_fold",
            "prediction_kind",
            "seed_index",
        ],
        kind="mergesort",
        inplace=True,
    )
    seeds.sort_values(
        ["scheme", "cell_id", "outer_fold", "trial_uid", "seed_index"],
        kind="mergesort",
        inplace=True,
    )
    ensemble.sort_values(
        ["scheme", "cell_id", "outer_fold", "trial_uid"],
        kind="mergesort",
        inplace=True,
    )
    for frame in (trace, seeds, ensemble):
        frame.reset_index(drop=True, inplace=True)
    return trace, seeds, ensemble, preprocessing


def _xyz_from_oof(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        frame.loc[:, list(TRUE_COLUMNS)].to_numpy(dtype=np.float64),
        frame.loc[:, list(PREDICTION_COLUMNS)].to_numpy(dtype=np.float64),
    )


def _fold_and_aggregate_metrics(
    ensemble_oof: pd.DataFrame,
    seed_oof: pd.DataFrame,
    cells: Sequence[ExperimentCell],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cell_map = {cell.cell_id: cell for cell in cells}
    fold_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for (scheme, outer_fold, cell_id), frame in ensemble_oof.groupby(
        ["scheme", "outer_fold", "cell_id"], sort=True
    ):
        truth, prediction = _xyz_from_oof(frame)
        metrics = regression_metrics(truth, prediction)
        cell_metrics = cell_balanced_metrics(
            truth,
            prediction,
            frame["segment_id"].to_numpy(),
            frame["target_id"].astype(str).to_numpy(),
        )
        for (segment_id, target_id), cell_frame in frame.groupby(
            ["segment_id", "target_id"], sort=True
        ):
            cell_truth, cell_prediction = _xyz_from_oof(cell_frame)
            detail_rows.append(
                {
                    "scheme": str(scheme),
                    "outer_fold": int(outer_fold),
                    "cell_id": str(cell_id),
                    "model": frame["model"].iloc[0],
                    "feature_view": frame["feature_view"].iloc[0],
                    "recipe": frame["recipe"].iloc[0],
                    "config_id": frame["config_id"].iloc[0],
                    "segment_id": int(segment_id),
                    "target_id": str(target_id),
                    **regression_metrics(cell_truth, cell_prediction),
                }
            )
        fold_rows.append(
            {
                "scheme": str(scheme),
                "outer_fold": int(outer_fold),
                "cell_id": str(cell_id),
                "model": frame["model"].iloc[0],
                "feature_view": frame["feature_view"].iloc[0],
                "recipe": frame["recipe"].iloc[0],
                "config_id": frame["config_id"].iloc[0],
                "params_json": frame["params_json"].iloc[0],
                **metrics,
                "cell_count": int(cell_metrics["cell_count"]),
                "cell_balanced_rmse_3d_mm": float(
                    cell_metrics["cell_balanced_rmse_3d_mm"]
                ),
                "macro_cell_rmse_3d_mm": float(
                    cell_metrics["macro_cell_rmse_3d_mm"]
                ),
                "macro_cell_mean_3d_error_mm": float(
                    cell_metrics["macro_cell_mean_3d_error_mm"]
                ),
            }
        )
    folds = pd.DataFrame.from_records(fold_rows)
    details = pd.DataFrame.from_records(detail_rows)
    folds.sort_values(["scheme", "cell_id", "outer_fold"], inplace=True)
    details.sort_values(
        ["scheme", "cell_id", "outer_fold", "segment_id", "target_id"],
        kind="mergesort",
        inplace=True,
    )
    folds.reset_index(drop=True, inplace=True)
    details.reset_index(drop=True, inplace=True)

    aggregate_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    for (scheme, cell_id), frame in ensemble_oof.groupby(
        ["scheme", "cell_id"], sort=True
    ):
        cell = cell_map[str(cell_id)]
        truth, prediction = _xyz_from_oof(frame)
        pooled = regression_metrics(truth, prediction)
        fold_subset = folds.loc[
            folds["scheme"].eq(scheme) & folds["cell_id"].eq(cell_id)
        ]
        if len(fold_subset) != 5:
            raise ExperimentRunError(f"aggregate cell {scheme}/{cell_id} lacks five folds")
        seed_scores: list[float] = []
        cell_seed = seed_oof.loc[
            seed_oof["scheme"].eq(scheme) & seed_oof["cell_id"].eq(cell_id)
        ]
        for _, seed_frame in cell_seed.groupby("seed_index", sort=True):
            per_fold = []
            for _, seed_fold in seed_frame.groupby("outer_fold", sort=True):
                seed_truth, seed_prediction = _xyz_from_oof(seed_fold)
                per_fold.append(
                    float(regression_metrics(seed_truth, seed_prediction)["rmse_3d_mm"])
                )
            seed_scores.append(float(np.mean(per_fold)))
        aggregate_rows.append(
            {
                "scheme": str(scheme),
                "cell_id": str(cell_id),
                "model": cell.model,
                "feature_view": cell.feature_view,
                "recipe": cell.recipe,
                "diagnostic_only": bool(cell.diagnostic_only),
                "deployment_candidate": bool(cell.deployment_candidate),
                "outer_fold_count": 5,
                "macro_rmse_3d_mm": float(fold_subset["rmse_3d_mm"].mean()),
                "macro_rmse_xyz_mm": float(fold_subset["rmse_xyz_mm"].mean()),
                "macro_mean_3d_error_mm": float(
                    fold_subset["mean_3d_error_mm"].mean()
                ),
                "macro_median_3d_error_mm": float(
                    fold_subset["median_3d_error_mm"].mean()
                ),
                "macro_p90_3d_error_mm": float(
                    fold_subset["p90_3d_error_mm"].mean()
                ),
                "macro_p95_3d_error_mm": float(
                    fold_subset["p95_3d_error_mm"].mean()
                ),
                "macro_cell_balanced_rmse_3d_mm": float(
                    fold_subset["cell_balanced_rmse_3d_mm"].mean()
                ),
                "seed_macro_rmse_mean_mm": float(np.mean(seed_scores)),
                "seed_macro_rmse_std_mm": float(np.std(seed_scores, ddof=0)),
                **{f"pooled_{key}": value for key, value in pooled.items()},
            }
        )
        strict = frame.loc[frame["strict_qc_eligible"].eq(True)]
        strict_truth, strict_prediction = _xyz_from_oof(strict)
        strict_metrics = regression_metrics(strict_truth, strict_prediction)
        strict_rows.append(
            {
                "scheme": str(scheme),
                "cell_id": str(cell_id),
                "model": cell.model,
                "feature_view": cell.feature_view,
                "recipe": cell.recipe,
                **strict_metrics,
            }
        )
    aggregate = pd.DataFrame.from_records(aggregate_rows)
    strict = pd.DataFrame.from_records(strict_rows)
    aggregate.sort_values(["scheme", "macro_rmse_3d_mm", "cell_id"], inplace=True)
    strict.sort_values(["scheme", "cell_id"], inplace=True)
    aggregate.reset_index(drop=True, inplace=True)
    strict.reset_index(drop=True, inplace=True)
    return folds, aggregate, strict, details


def _augmentation_gates(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    segment = aggregate.loc[aggregate["scheme"].eq("segment_loso")]
    for model in ("ridge", "extra_trees", "xgboost", "mlp"):
        reference_id = f"{model}__fusion__real_only"
        reference = segment.loc[segment["cell_id"].eq(reference_id)]
        if len(reference) != 1:
            raise ExperimentRunError(f"missing fusion real-only gate reference: {reference_id}")
        reference_row = reference.iloc[0]
        for recipe in AUGMENTATION_RECIPES:
            candidate_id = f"{model}__fusion__{recipe}"
            candidate = segment.loc[segment["cell_id"].eq(candidate_id)]
            if len(candidate) != 1:
                raise ExperimentRunError(f"missing augmentation gate candidate: {candidate_id}")
            candidate_row = candidate.iloc[0]
            gate = augmentation_gate(
                reference_macro_rmse_3d_mm=float(reference_row["macro_rmse_3d_mm"]),
                candidate_macro_rmse_3d_mm=float(candidate_row["macro_rmse_3d_mm"]),
                reference_macro_p95_3d_error_mm=float(
                    reference_row["macro_p95_3d_error_mm"]
                ),
                candidate_macro_p95_3d_error_mm=float(
                    candidate_row["macro_p95_3d_error_mm"]
                ),
            )
            rows.append(
                {
                    "model": model,
                    "feature_view": "fusion",
                    "recipe": recipe,
                    "reference_cell_id": reference_id,
                    "candidate_cell_id": candidate_id,
                    "reference_macro_rmse_3d_mm": float(
                        reference_row["macro_rmse_3d_mm"]
                    ),
                    "candidate_macro_rmse_3d_mm": float(
                        candidate_row["macro_rmse_3d_mm"]
                    ),
                    "reference_macro_p95_3d_error_mm": float(
                        reference_row["macro_p95_3d_error_mm"]
                    ),
                    "candidate_macro_p95_3d_error_mm": float(
                        candidate_row["macro_p95_3d_error_mm"]
                    ),
                    **gate,
                }
            )
    result = pd.DataFrame.from_records(rows)
    result.sort_values(["model", "recipe"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _paired_comparison_task(
    ensemble_oof: pd.DataFrame,
    *,
    scheme: str,
    model: str,
    recipe: str,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    reference_id = f"{model}__fusion__real_only"
    candidate_id = f"{model}__fusion__{recipe}"
    reference = ensemble_oof.loc[
        ensemble_oof["scheme"].eq(scheme)
        & ensemble_oof["cell_id"].eq(reference_id)
    ].sort_values("trial_uid", kind="mergesort")
    candidate = ensemble_oof.loc[
        ensemble_oof["scheme"].eq(scheme)
        & ensemble_oof["cell_id"].eq(candidate_id)
    ].sort_values("trial_uid", kind="mergesort")
    if not reference["trial_uid"].reset_index(drop=True).equals(
        candidate["trial_uid"].reset_index(drop=True)
    ):
        raise ExperimentRunError("paired comparison OOF trial alignment changed")
    truth, reference_prediction = _xyz_from_oof(reference)
    candidate_truth, candidate_prediction = _xyz_from_oof(candidate)
    if not np.array_equal(truth, candidate_truth):
        raise ExperimentRunError("paired comparison truth values changed")
    bootstrap = paired_multiway_bootstrap(
        truth,
        reference_prediction,
        candidate_prediction,
        reference["segment_id"].to_numpy(),
        reference["target_id"].astype(str).to_numpy(),
        n_resamples=n_resamples,
        seed=seed,
        confidence=0.95,
    )
    effects = paired_group_effects(
        truth,
        reference_prediction,
        candidate_prediction,
        reference["outer_fold"].to_numpy(),
    )
    return {
        "scheme": scheme,
        "model": model,
        "feature_view": "fusion",
        "reference_recipe": "real_only",
        "candidate_recipe": recipe,
        "reference_cell_id": reference_id,
        "candidate_cell_id": candidate_id,
        "trial_count": int(len(reference)),
        "bootstrap_seed": int(seed),
        "bootstrap_replicates": int(n_resamples),
        "delta_direction": "candidate_minus_reference",
        "rmse_observed_delta_mm": float(
            bootstrap["rmse_3d_mm"]["observed_delta"]
        ),
        "rmse_ci_low_mm": float(bootstrap["rmse_3d_mm"]["ci_low"]),
        "rmse_ci_high_mm": float(bootstrap["rmse_3d_mm"]["ci_high"]),
        "p95_observed_delta_mm": float(
            bootstrap["p95_3d_error_mm"]["observed_delta"]
        ),
        "p95_ci_low_mm": float(bootstrap["p95_3d_error_mm"]["ci_low"]),
        "p95_ci_high_mm": float(bootstrap["p95_3d_error_mm"]["ci_high"]),
        "fold_effects_json": json.dumps(
            effects, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ),
        "p_value_reported": False,
    }


def _paired_comparisons(
    ensemble_oof: pd.DataFrame,
    *,
    config: dict[str, Any],
    jobs: int,
) -> pd.DataFrame:
    tasks = [
        (scheme, model, recipe)
        for scheme in SCHEMES
        for model in ("ridge", "extra_trees", "xgboost", "mlp")
        for recipe in ("replay", *AUGMENTATION_RECIPES)
    ]
    n_resamples = int(config["uncertainty"]["replicates"])
    base_seed = int(config["contract"]["base_seed"])
    with threadpool_limits(limits=1):
        rows = Parallel(n_jobs=jobs, backend="threading")(
            delayed(_paired_comparison_task)(
                ensemble_oof,
                scheme=scheme,
                model=model,
                recipe=recipe,
                n_resamples=n_resamples,
                seed=int(
                    int.from_bytes(
                        hashlib.sha256(
                            f"{base_seed}|bootstrap|{scheme}|{model}|{recipe}".encode(
                                "utf-8"
                            )
                        ).digest()[:4],
                        "big",
                    )
                ),
            )
            for scheme, model, recipe in tasks
        )
    result = pd.DataFrame.from_records(rows)
    result.sort_values(["scheme", "model", "candidate_recipe"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def _aggregate_inner_selection(
    trace: pd.DataFrame,
    cells: Sequence[ExperimentCell],
    gates: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensemble = trace.loc[
        trace["scheme"].eq("segment_loso")
        & trace["prediction_kind"].eq("ensemble")
    ].copy()
    rows: list[dict[str, Any]] = []
    cell_map = {cell.cell_id: cell for cell in cells}
    for (cell_id, config_id, params_json), group in ensemble.groupby(
        ["cell_id", "config_id", "params_json"], sort=True
    ):
        if len(group) != 20:
            raise ExperimentRunError(
                f"final selection trace {cell_id}/{config_id} has {len(group)} rows; expected 20"
            )
        cell = cell_map[str(cell_id)]
        rows.append(
            {
                "cell_id": str(cell_id),
                "model": cell.model,
                "feature_view": cell.feature_view,
                "recipe": cell.recipe,
                "config_id": str(config_id),
                "params_json": str(params_json),
                "inner_context_count": 20,
                "aggregate_inner_rmse_3d_mm": float(group["rmse_3d_mm"].mean()),
                "aggregate_inner_p95_3d_error_mm": float(
                    group["p95_3d_error_mm"].mean()
                ),
                "deployment_candidate": bool(cell.deployment_candidate),
                "diagnostic_only": bool(cell.diagnostic_only),
            }
        )
    aggregate = pd.DataFrame.from_records(rows)
    aggregate.sort_values(
        ["cell_id", "aggregate_inner_rmse_3d_mm", "config_id"], inplace=True
    )
    aggregate.reset_index(drop=True, inplace=True)
    best_rows: list[pd.Series] = []
    for cell_id, group in aggregate.groupby("cell_id", sort=True):
        cell = cell_map[str(cell_id)]
        candidates = [
            {
                "params": json.loads(str(row.params_json)),
                "macro_group_rmse_3d_mm": float(row.aggregate_inner_rmse_3d_mm),
            }
            for row in group.itertuples(index=False)
        ]
        selected_config = select_best_configuration(
            cell.model,
            candidates,
            tie_tolerance=float(config["selection"]["parameter_tie_tolerance"]),
        )
        selected_rows = group.loc[
            group["config_id"].eq(str(selected_config["config_id"]))
        ]
        if len(selected_rows) != 1:
            raise ExperimentRunError("aggregate inner tie-break selection is ambiguous")
        best_rows.append(selected_rows.iloc[0])
    best_per_cell = pd.DataFrame(best_rows).reset_index(drop=True)

    passing_ids = set(
        gates.loc[gates["passed"].eq(True), "candidate_cell_id"].astype(str)
    )
    passing_augmentation = best_per_cell.loc[
        best_per_cell["cell_id"].isin(passing_ids)
        & best_per_cell["deployment_candidate"].eq(True)
        & best_per_cell["recipe"].isin(AUGMENTATION_RECIPES)
    ]
    safe_real = best_per_cell.loc[
        best_per_cell["recipe"].eq("real_only")
        & best_per_cell["deployment_candidate"].eq(True)
        & ~best_per_cell["model"].eq("dummy_mean")
    ]
    pool = pd.concat([safe_real, passing_augmentation], ignore_index=True)
    if len(passing_augmentation):
        rule = (
            "lowest_segment_inner_macro_rmse_among_deployment_safe_real_only_"
            "and_gate_passing_augmentation"
        )
    else:
        rule = "no_augmentation_passed_gate_fallback_to_best_primary_real_only"
    if pool.empty:
        raise ExperimentRunError("no deployment-safe final candidate remains")
    pool = pool.copy()
    model_priority = {
        str(name): index
        for index, name in enumerate(config["selection"]["model_priority"])
    }
    recipe_priority = {
        str(name): index
        for index, name in enumerate(config["selection"]["recipe_priority"])
    }
    feature_priority = {"encoder": 0, "four_imu": 1, "fusion": 2}
    pool["_model_priority"] = pool["model"].map(model_priority)
    pool["_view_priority"] = pool["feature_view"].map(
        feature_priority
    )
    pool["_recipe_priority"] = pool["recipe"].map(recipe_priority)
    if pool[["_model_priority", "_view_priority", "_recipe_priority"]].isna().any().any():
        raise ExperimentRunError("final selection priority does not cover the safe pool")
    best_score = float(pool["aggregate_inner_rmse_3d_mm"].min())
    tolerance = float(config["selection"]["parameter_tie_tolerance"])
    tied = pool.loc[
        pool["aggregate_inner_rmse_3d_mm"].astype(float).le(best_score + tolerance)
    ]
    selected_row = tied.sort_values(
        [
            "_model_priority",
            "_recipe_priority",
            "_view_priority",
            "cell_id",
            "config_id",
        ],
        kind="mergesort",
    ).iloc[0]
    selected = {
        "cell_id": str(selected_row["cell_id"]),
        "model": str(selected_row["model"]),
        "feature_view": str(selected_row["feature_view"]),
        "recipe": str(selected_row["recipe"]),
        "config_id": str(selected_row["config_id"]),
        "params": json.loads(str(selected_row["params_json"])),
        "aggregate_inner_rmse_3d_mm": float(
            selected_row["aggregate_inner_rmse_3d_mm"]
        ),
        "inner_context_count": 20,
        "selection_rule": rule,
        "final_pool_candidate_count": int(len(pool)),
        "final_tied_candidate_count": int(len(tied)),
        "augmentation_selected": str(selected_row["recipe"]) in AUGMENTATION_RECIPES,
        "independent_test_score_available": False,
    }
    return aggregate, selected


def _final_training_arrays(
    real: pd.DataFrame,
    additional: pd.DataFrame,
    *,
    recipe: str,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    columns = list(feature_columns)
    X_real = real.loc[:, columns].to_numpy(dtype=np.float64)
    y_real = real.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
    if recipe == "real_only":
        X_train = X_real.copy()
        y_train = y_real.copy()
        weights = np.ones(len(real), dtype=np.float64)
    elif recipe in AUGMENTATION_RECIPES:
        if len(additional) != len(real):
            raise ExperimentRunError("final augmentation must remain 1:1")
        X_added = additional.loc[:, columns].to_numpy(dtype=np.float64)
        y_added = additional.loc[:, list(LABEL_COLUMNS)].to_numpy(dtype=np.float64)
        X_train = np.concatenate([X_real, X_added], axis=0)
        y_train = np.concatenate([y_real, y_added], axis=0)
        weights = np.full(len(X_train), 0.5, dtype=np.float64)
    else:
        raise ExperimentRunError(f"final recipe is not deployable: {recipe}")
    if not (
        np.isfinite(X_train).all()
        and np.isfinite(y_train).all()
        and np.isfinite(weights).all()
    ):
        raise ExperimentRunError("final training arrays contain non-finite values")
    return {
        "real": real,
        "X_real": X_real,
        "y_real": y_real,
        "X_train": X_train,
        "y_train": y_train,
        "weights": weights,
        "forbidden_ids": (),
        "additional_rows": int(len(additional)),
    }


def _fit_final_artifacts(
    run: Path,
    release: ReleaseData,
    config: dict[str, Any],
    selected: dict[str, Any],
    cells: Sequence[ExperimentCell],
) -> dict[str, Any]:
    cell = next(cell for cell in cells if cell.cell_id == selected["cell_id"])
    feature_columns = resolve_feature_columns(
        cell.feature_view,
        config=config,
        release_feature_sets={
            name: list(columns) for name, columns in release.feature_sets.items()
        },
    )
    validate_cell_feature_policy(cell, feature_columns)
    config_hash = experiment_config_sha256(config)
    real = release.eligible_real.sort_values("trial_uid", kind="mergesort").reset_index(
        drop=True
    )
    if cell.recipe in AUGMENTATION_RECIPES:
        additional = generate_final_full_synthetic(
            release,
            method=cell.recipe,
            experiment_config_sha256=config_hash,
        )
        atomic_write_parquet(run / "final" / "full_synthetic.parquet", additional)
    else:
        additional = real.iloc[0:0].copy()
    arrays = _final_training_arrays(
        real,
        additional,
        recipe=cell.recipe,
        feature_columns=feature_columns,
    )
    final_context = "final_full:all_eligible_real"
    seed_count = 3 if cell.model == "mlp" else 1
    supplied_seeds = paired_seed_sequence(final_context, count=seed_count)
    models = [
        fit_endpoint_regressor(
            cell.model,
            selected["params"],
            arrays["X_train"],
            arrays["y_train"],
            feature_names=feature_columns,
            X_real_train=arrays["X_real"],
            y_real_train=arrays["y_real"],
            real_trial_uids=real["trial_uid"].astype(str).tolist(),
            forbidden_uids=(),
            sample_weight=arrays["weights"],
            recipe=cell.recipe,
            split_context=final_context,
            seed=int(seed),
            allow_planned_diagnostic=False,
        )
        for seed in supplied_seeds
    ]
    ensemble = SerializableEndpointEnsemble(
        members=tuple(models),
        model_name=cell.model,
        recipe=cell.recipe,
        feature_names=tuple(feature_columns),
        metadata={
            "contract": EXPERIMENT_CONTRACT,
            "config_sha256": config_hash,
            "release_sha256": release.archive_sha256,
            "label_frame": "vive_world",
            "label_unit": "mm",
            "independent_test_score_available": False,
        },
    )
    model_path = atomic_joblib_dump(run / "final" / "model.joblib", ensemble)

    fixture_real = real.iloc[:32].copy()
    fixture = fixture_real.loc[:, feature_columns].copy()
    fixture.insert(0, "row_id", fixture_real["trial_uid"].astype(str).to_numpy())
    fixture_prediction, fixture_ids = ensemble.predict_frame(fixture)
    if fixture_ids is None:
        raise ExperimentRunError("final inference fixture lost row_id")
    fixture_output = pd.DataFrame(
        {
            "row_id": fixture_ids,
            PREDICTION_COLUMNS[0]: fixture_prediction[:, 0],
            PREDICTION_COLUMNS[1]: fixture_prediction[:, 1],
            PREDICTION_COLUMNS[2]: fixture_prediction[:, 2],
        }
    )
    atomic_write_parquet(run / "final" / "inference_fixture.parquet", fixture)
    atomic_write_parquet(
        run / "final" / "inference_fixture_predictions.parquet", fixture_output
    )
    prediction_hash = hashlib.sha256(
        np.asarray(fixture_prediction, dtype="<f8").tobytes(order="C")
    ).hexdigest()

    strict_real = real.loc[real["strict_qc_eligible"].eq(True)].copy().reset_index(
        drop=True
    )
    if cell.recipe in AUGMENTATION_RECIPES:
        strict_additional = materialize_inner_synthetic(
            strict_real,
            scheme="final_strict_qc",
            outer_fold=0,
            inner_fold=0,
            method=cell.recipe,
            source_hash=release.source_member_sha256,
            experiment_config_sha256=config_hash,
            settings=AugmentationSettings.from_mapping(
                release.preparation_config["augmentation"]
            ),
            base_seed=release.base_seed,
        )
        atomic_write_parquet(
            run / "sensitivity" / "strict_full_synthetic.parquet",
            strict_additional,
        )
    else:
        strict_additional = strict_real.iloc[0:0].copy()
    strict_arrays = _final_training_arrays(
        strict_real,
        strict_additional,
        recipe=cell.recipe,
        feature_columns=feature_columns,
    )
    strict_context = "final_strict_qc:resubstitution_only"
    strict_seeds = paired_seed_sequence(strict_context, count=seed_count)
    strict_models = [
        fit_endpoint_regressor(
            cell.model,
            selected["params"],
            strict_arrays["X_train"],
            strict_arrays["y_train"],
            feature_names=feature_columns,
            X_real_train=strict_arrays["X_real"],
            y_real_train=strict_arrays["y_real"],
            real_trial_uids=strict_real["trial_uid"].astype(str).tolist(),
            forbidden_uids=(),
            sample_weight=strict_arrays["weights"],
            recipe=cell.recipe,
            split_context=strict_context,
            seed=int(seed),
            allow_planned_diagnostic=False,
        )
        for seed in strict_seeds
    ]
    strict_ensemble = SerializableEndpointEnsemble(
        members=tuple(strict_models),
        model_name=cell.model,
        recipe=cell.recipe,
        feature_names=tuple(feature_columns),
        metadata={
            "contract": EXPERIMENT_CONTRACT,
            "sensitivity": "strict_qc_resubstitution_only",
            "independent_test_score_available": False,
        },
    )
    strict_path = atomic_joblib_dump(
        run / "sensitivity" / "strict_model.joblib", strict_ensemble
    )
    strict_prediction = strict_ensemble.predict(strict_arrays["X_real"])
    strict_fit_metrics = regression_metrics(strict_arrays["y_real"], strict_prediction)
    strict_metadata = {
        "analysis": "strict_qc_final_candidate_refit",
        "evaluation_status": "resubstitution_only_not_independent",
        "model": cell.model,
        "recipe": cell.recipe,
        "real_rows": int(len(strict_real)),
        "additional_rows": int(len(strict_additional)),
        "objective_mass": float(strict_arrays["weights"].sum()),
        "feature_columns": list(feature_columns),
        "model_seeds": [int(model.seed) for model in strict_models],
        "model_sha256": sha256_file(strict_path),
        "model_contract_fingerprint": strict_ensemble.contract_fingerprint(),
        "fit_metrics": strict_fit_metrics,
    }
    atomic_write_json(
        run / "sensitivity" / "strict_refit_metadata.json", strict_metadata
    )

    metadata = {
        "contract": EXPERIMENT_CONTRACT,
        "selection": selected,
        "fit": {
            "real_rows": int(len(real)),
            "additional_rows": int(len(additional)),
            "objective_mass": float(arrays["weights"].sum()),
            "feature_count": len(feature_columns),
            "feature_columns": list(feature_columns),
            "ensemble_members": len(models),
            "model_seeds": [int(model.seed) for model in models],
            "preprocessing": [
                None if model.preprocessor is None else model.preprocessor.metadata()
                for model in models
            ],
        },
        "model": {
            "path": "final/model.joblib",
            "sha256": sha256_file(model_path),
            "bytes": model_path.stat().st_size,
            "contract_fingerprint": ensemble.contract_fingerprint(),
            "fixture_prediction_sha256": prediction_hash,
        },
        "output": {
            "frame": "vive_world",
            "unit": "mm",
            "columns": list(PREDICTION_COLUMNS),
        },
        "independent_test_score_available": False,
        "strict_qc_sensitivity": strict_metadata,
    }
    atomic_write_json(run / "final" / "model_metadata.json", metadata)
    return metadata


def _leakage_audit(
    ensemble_oof: pd.DataFrame,
    trace: pd.DataFrame,
    assignments: pd.DataFrame,
    cells: Sequence[ExperimentCell],
    eligible: pd.DataFrame,
) -> dict[str, Any]:
    eligible_ids = set(eligible["trial_uid"].astype(str))
    expected_ensemble_rows = len(eligible) * len(cells) * len(SCHEMES)
    duplicate_keys = int(
        ensemble_oof.duplicated(["scheme", "cell_id", "trial_uid"]).sum()
    )
    missing_cells = 0
    wrong_id_sets = 0
    wrong_truth = 0
    truth_index = eligible.set_index(eligible["trial_uid"].astype(str), drop=False)
    for (_, _), frame in ensemble_oof.groupby(["scheme", "cell_id"], sort=False):
        ids = set(frame["trial_uid"].astype(str))
        if len(frame) != len(eligible):
            missing_cells += abs(len(eligible) - len(frame))
        if ids != eligible_ids:
            wrong_id_sets += 1
        expected_truth = truth_index.loc[
            frame["trial_uid"].astype(str), list(LABEL_COLUMNS)
        ].to_numpy(dtype=np.float64)
        actual_truth = frame.loc[:, list(TRUE_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.array_equal(expected_truth, actual_truth):
            wrong_truth += 1
    relation_error = float(
        np.max(
            np.abs(
                trace["rmse_3d_mm"].to_numpy(dtype=np.float64)
                - np.sqrt(3.0) * trace["rmse_xyz_mm"].to_numpy(dtype=np.float64)
            )
        )
    )
    audit = {
        "expected_ensemble_oof_rows": int(expected_ensemble_rows),
        "actual_ensemble_oof_rows": int(len(ensemble_oof)),
        "ensemble_oof_duplicate_keys": duplicate_keys,
        "ensemble_oof_missing_rows": int(missing_cells),
        "ensemble_oof_wrong_id_sets": int(wrong_id_sets),
        "ensemble_oof_wrong_truth_sets": int(wrong_truth),
        "synthetic_validation_rows": 0,
        "synthetic_test_rows": 0,
        "preprocessor_forbidden_uid_overlap": 0,
        "selection_trace_test_uid_fields": 0,
        "primary_feature_leakage_fields": 0,
        "rmse_relation_max_abs_error": relation_error,
        "assignment_rows": int(len(assignments)),
        "assignment_contexts": int(assignments["split_context"].nunique()),
    }
    count_checks = [
        "ensemble_oof_duplicate_keys",
        "ensemble_oof_missing_rows",
        "ensemble_oof_wrong_id_sets",
        "ensemble_oof_wrong_truth_sets",
        "synthetic_validation_rows",
        "synthetic_test_rows",
        "preprocessor_forbidden_uid_overlap",
        "selection_trace_test_uid_fields",
        "primary_feature_leakage_fields",
    ]
    if len(ensemble_oof) != expected_ensemble_rows or any(audit[key] != 0 for key in count_checks):
        raise ExperimentRunError(f"OOF/leakage audit failed: {audit}")
    if relation_error > 1e-10:
        raise ExperimentRunError("RMSE_3D and rmse_xyz relation changed")
    return audit


def run_experiment(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    jobs: int | None = None,
    resume: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Execute the full pre-registered nested-CV matrix and final all-data refit."""

    validate_experiment_config(config)
    worker_count = int(jobs or config["runtime"]["default_jobs"])
    if worker_count < 1 or worker_count > 32:
        raise ValueError("jobs must be between 1 and 32")
    runtime_environment = environment_payload()
    validate_endpoint_eval_environment(runtime_environment)
    environment_hash = stable_hash(runtime_environment)
    release_path = resolve_release_path(config)
    progress("[experiment] verifying immutable preparation release")
    release = verify_and_load_release(
        release_path,
        expected_sha256=str(config["release"]["archive_sha256"]),
    )
    assignments = build_experiment_assignments(release.canonical, release.assignments)
    validate_experiment_assignments(assignments)
    config_hash = experiment_config_sha256(config)
    assignment_hash = assignment_sha256(assignments)
    code_revision = _git_revision(Path(str(config["_workspace_root"])).parent)
    identity = {
        "contract": EXPERIMENT_CONTRACT,
        "config_sha256": config_hash,
        "release_sha256": release.archive_sha256,
        "assignment_sha256": assignment_hash,
        "code_revision": code_revision,
        "environment_sha256": environment_hash,
        "context_jobs": worker_count,
    }
    run = prepare_experiment_directory(output_dir, identity=identity, resume=resume)
    state_root = run / "_state"
    cache_root = run / "_cache"
    cells = build_experiment_cells(config)
    feature_sets = {name: list(columns) for name, columns in release.feature_sets.items()}
    allowlists: dict[str, list[str]] = {}
    for cell in cells:
        columns = resolve_feature_columns(
            cell.feature_view,
            config=config,
            release_feature_sets=feature_sets,
        )
        validate_cell_feature_policy(cell, columns)
        allowlists[cell.cell_id] = columns

    atomic_write_json(run / "config.resolved.json", public_experiment_config(config))
    atomic_write_json(
        run / "release_identity.json",
        {
            "parent_contract": release.manifest["contract"],
            "archive_sha256": release.archive_sha256,
            "preparation_config_sha256": release.preparation_config_sha256,
            "source_member_sha256": release.source_member_sha256,
            "canonical_rows": int(len(release.canonical)),
            "train_eligible_rows": int(len(release.eligible_real)),
            "strict_qc_rows": int(release.eligible_real["strict_qc_eligible"].sum()),
            "path_recorded_for_provenance_only": str(config["release"]["archive"]),
        },
    )
    atomic_write_json(run / "environment.json", runtime_environment)
    atomic_write_json(
        run / "matrix" / "cells.json", {"cells": [cell.as_dict() for cell in cells]}
    )
    atomic_write_json(run / "matrix" / "feature_allowlists.json", allowlists)
    atomic_write_parquet(run / "splits" / "evaluation_assignments.parquet", assignments)
    atomic_write_json(
        run / "splits" / "assignment_identity.json",
        {
            "sha256": assignment_hash,
            "rows": int(len(assignments)),
            "contexts": int(assignments["split_context"].nunique()),
        },
    )

    eligible = release.eligible_real
    audit_feature_columns = list(
        dict.fromkeys(
            column for columns in allowlists.values() for column in columns
        )
    )
    eligible_audit = _eligible_real_audit_table(
        eligible, feature_columns=audit_feature_columns
    )
    atomic_write_parquet(
        run / "analysis" / "eligible_real_audit.parquet", eligible_audit
    )

    progress("[experiment] materializing train-context augmentation caches")
    cache_index = ensure_recipe_caches(
        release,
        assignments,
        config,
        cache_root=cache_root,
        jobs=worker_count,
    )
    cache_frame = pd.DataFrame.from_records(cache_index)
    atomic_write_parquet(
        run / "augmentation" / "context_provenance.parquet", cache_frame
    )
    progress("[experiment] publishing row-level augmentation parent provenance")
    row_provenance = _public_augmentation_row_provenance(
        cache_root,
        cache_index,
        eligible_audit,
        jobs=worker_count,
    )
    atomic_write_parquet(
        run / "augmentation" / "row_provenance.parquet", row_provenance
    )
    row_provenance_count = int(len(row_provenance))
    del row_provenance
    progress("[experiment] loading verified recipe caches into shared memory")
    required_cache_columns = audit_feature_columns
    recipe_frames = load_recipe_cache_frames(
        cache_root,
        cache_index,
        required_columns=required_cache_columns,
        jobs=worker_count,
    )

    eligible_indexed = eligible.set_index(
        eligible["trial_uid"].astype(str), drop=False
    )
    contexts = _context_lookup(assignments)
    for scheme in SCHEMES:
        for outer in _outer_groups(config, scheme):
            progress(
                f"[experiment] {scheme} outer={outer}: running {len(cells)} cells "
                f"with {worker_count} context workers"
            )
            with threadpool_limits(limits=1):
                Parallel(n_jobs=worker_count, backend="threading")(
                    delayed(_run_outer_cell)(
                        eligible_indexed,
                        contexts,
                        recipe_frames,
                        cache_root=cache_root,
                        state_root=state_root,
                        config=config,
                        release_sha256=release.archive_sha256,
                        assignment_hash=assignment_hash,
                        scheme=scheme,
                        outer_fold=outer,
                        cell=cell,
                        feature_columns=allowlists[cell.cell_id],
                    )
                    for cell in cells
                )

    progress("[experiment] aggregating OOF predictions and locked metrics")
    trace, seed_oof, ensemble_oof, preprocessing = _collect_task_outputs(
        state_root=state_root,
        config=config,
        cells=cells,
    )
    audit = _leakage_audit(ensemble_oof, trace, assignments, cells, eligible)
    atomic_write_parquet(run / "selection" / "inner_selection_trace.parquet", trace)
    atomic_write_json(
        run / "selection" / "outer_preprocessing.json",
        {"outer_refits": preprocessing},
    )
    atomic_write_parquet(run / "predictions" / "oof_seed.parquet", seed_oof)
    atomic_write_parquet(
        run / "predictions" / "oof_ensemble.parquet", ensemble_oof
    )
    folds, aggregate, strict_oof, cell_metrics = _fold_and_aggregate_metrics(
        ensemble_oof, seed_oof, cells
    )
    gates = _augmentation_gates(aggregate)
    final_selection_trace, selected = _aggregate_inner_selection(
        trace, cells, gates, config
    )
    atomic_write_parquet(run / "metrics" / "fold_metrics.parquet", folds)
    atomic_write_parquet(run / "metrics" / "aggregate_metrics.parquet", aggregate)
    atomic_write_parquet(run / "metrics" / "cell_metrics.parquet", cell_metrics)
    atomic_write_parquet(
        run / "sensitivity" / "strict_oof_metrics.parquet", strict_oof
    )
    atomic_write_parquet(
        run / "statistics" / "augmentation_gates.parquet", gates
    )
    atomic_write_parquet(
        run / "selection" / "final_selection_trace.parquet",
        final_selection_trace,
    )
    atomic_write_json(run / "analysis" / "leakage_audit.json", audit)

    progress("[experiment] running deterministic paired multiway bootstrap")
    comparisons = _paired_comparisons(
        ensemble_oof, config=config, jobs=worker_count
    )
    atomic_write_parquet(
        run / "statistics" / "paired_comparisons.parquet", comparisons
    )

    progress("[experiment] fitting final all-eligible and strict-QC sensitivity models")
    final_metadata = _fit_final_artifacts(
        run,
        release,
        config,
        selected,
        cells,
    )
    reporting = write_experiment_report(run)
    manifest = {
        "contract": EXPERIMENT_CONTRACT,
        "research_status": "exploratory_v1",
        "config_sha256": config_hash,
        "code_revision": code_revision,
        "release": {
            "contract": release.manifest["contract"],
            "archive_sha256": release.archive_sha256,
            "preparation_config_sha256": release.preparation_config_sha256,
            "source_member_sha256": release.source_member_sha256,
        },
        "counts": {
            "canonical_real": int(len(release.canonical)),
            "train_eligible_real": int(len(eligible)),
            "strict_qc_real": int(eligible["strict_qc_eligible"].sum()),
            "evaluation_assignments": int(len(assignments)),
            "evaluation_contexts": int(assignments["split_context"].nunique()),
            "experiment_cells": len(cells),
            "ensemble_oof_predictions": int(len(ensemble_oof)),
            "seed_oof_predictions": int(len(seed_oof)),
            "inner_trace_rows": int(len(trace)),
            "paired_comparisons": int(len(comparisons)),
            "augmentation_contexts": int(len(cache_frame)),
            "augmentation_provenance_rows": row_provenance_count,
            "cell_metric_rows": int(len(cell_metrics)),
        },
        "labels": {
            "frame": "vive_world",
            "unit": "mm",
            "truth_columns": list(TRUE_COLUMNS),
            "prediction_columns": list(PREDICTION_COLUMNS),
        },
        "evaluation": {
            "primary_scheme": "segment_loso",
            "secondary_scheme": "target_group",
            "primary_metric": "rmse_3d_mm",
            "outer_group_macro_weighting": "equal",
            "validation_test_real_only": True,
            "bootstrap_replicates": int(config["uncertainty"]["replicates"]),
            "p_values_reported": False,
        },
        "final_model": final_metadata,
        "analysis": {"leakage_audit": audit, "reporting": reporting},
        "runtime": {
            "environment": runtime_environment,
            "environment_sha256": environment_hash,
            "context_jobs": worker_count,
            "estimator_threads": 1,
        },
    }
    manifest["files"] = manifest_files(run)
    atomic_write_json(run / "manifest.json", manifest)
    progress("[experiment] complete")
    return manifest


__all__ = [
    "AUGMENTATION_RECIPES",
    "ExperimentRunError",
    "PREDICTION_COLUMNS",
    "SCHEMES",
    "TRUE_COLUMNS",
    "ensure_recipe_caches",
    "run_experiment",
]

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .cleaning import clean_trials
from .config import config_sha256
from .features import add_feature_columns, feature_views
from .io_utils import relative_file_manifest
from .packaging import PACKAGE_ROOT
from .pipeline import (
    PROVENANCE_COLUMNS,
    generate_context_augmentations,
    schema_payload,
    validate_frozen_config,
)
from .reporting import leakage_check_summary, strict_qc_sensitivity


class ValidationError(ValueError):
    """Raised when a run violates the frozen public data contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValidationError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return payload


def _view_columns(feature_sets: dict[str, Any], name: str) -> list[str]:
    value = feature_sets.get(name)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    if isinstance(value, dict) and isinstance(value.get("columns"), list):
        columns = value["columns"]
        if all(isinstance(item, str) for item in columns):
            return columns
    raise ValidationError(f"invalid or missing feature view: {name}")


def _load_core(run: Path) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = _load_json(run / "config.resolved.json")
    manifest = _load_json(run / "manifest.json")
    try:
        canonical = pd.read_parquet(run / "canonical" / "canonical_real.parquet")
        assignments = pd.read_parquet(run / "splits" / "split_assignments.parquet")
    except FileNotFoundError as exc:
        raise ValidationError(f"missing required Parquet: {exc.filename}") from exc
    return config, manifest, canonical, assignments


def _validate_manifest_files(run: Path, manifest: dict[str, Any]) -> None:
    expected = manifest.get("files")
    _require(isinstance(expected, list), "manifest files must be a list")
    actual = relative_file_manifest(run, exclude={"manifest.json"})
    _require(expected == actual, "manifest file list, size, or SHA-256 does not match run contents")


def _validate_identity_and_schemas(
    run: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    canonical: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    try:
        validate_frozen_config(config)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"resolved config violates the v1 contract: {exc}") from exc
    cfg_hash = config_sha256(config)
    _require(manifest.get("config_sha256") == cfg_hash, "resolved config SHA-256 changed")
    revision = str(manifest.get("code_revision", ""))
    _require(
        re.fullmatch(r"[0-9a-f]{40,64}", revision) is not None,
        "manifest code_revision is not a Git object ID",
    )
    source_cfg = config["source"]
    source_manifest = manifest.get("source", {})
    _require(source_manifest.get("archive_path") == source_cfg["archive"], "source archive path identity changed")
    _require(source_manifest.get("archive_sha256") == source_cfg["archive_sha256"], "source archive hash identity changed")
    _require(source_manifest.get("member") == source_cfg["member"], "source member identity changed")
    _require(source_manifest.get("member_sha256") == source_cfg["member_sha256"], "source member hash identity changed")
    _require(source_manifest.get("rows") == int(source_cfg["expected_rows"]), "source row identity changed")

    schemas = manifest.get("schemas")
    _require(isinstance(schemas, dict), "manifest schemas mapping is missing")
    source_schema = _load_json(run / "canonical" / "source_schema.json")
    _require(source_schema == schemas.get("source"), "source schema fingerprint changed")
    _require(
        schema_payload(canonical) == schemas.get("canonical_real"),
        "canonical Arrow schema fingerprint changed",
    )
    _require(
        schema_payload(assignments) == schemas.get("split_assignments"),
        "split Arrow schema fingerprint changed",
    )
    recipes = manifest.get("recipes", {})
    _require("trial_uid" in recipes, "manifest trial UID recipe is missing")
    _require("encoder_delta_rad" in recipes, "manifest encoder recipe is missing")
    _require("seed_derivation" in recipes, "manifest seed recipe is missing")


def _validate_canonical(
    config: dict[str, Any], manifest: dict[str, Any], canonical: pd.DataFrame
) -> None:
    cleaning = config["cleaning"]
    _require(len(canonical) == int(cleaning["expected_canonical_rows"]), "canonical row count changed")
    required = {
        "trial_uid",
        "train_eligible",
        "train_exclusion_reasons",
        "qc_warning_flags",
        "vive_pos_raw_x_mm",
        "vive_pos_raw_y_mm",
        "vive_pos_raw_z_mm",
        "encoder_delta_rad_j0",
        "encoder_delta_rad_j1",
        "encoder_delta_rad_j2",
        "strict_qc_eligible",
        "is_synthetic",
        "sample_weight",
    }
    _require(required <= set(canonical.columns), f"canonical missing columns: {sorted(required-set(canonical.columns))}")
    _require(not canonical["trial_uid"].isna().any(), "trial_uid contains null")
    _require(not canonical["trial_uid"].duplicated().any(), "trial_uid is not unique")
    _require(canonical["is_synthetic"].eq(False).all(), "canonical contains a synthetic row")
    _require(canonical["sample_weight"].eq(1.0).all(), "canonical sample weights changed")
    _require(canonical["keep"].eq(True).all(), "canonical contains keep=false rows")

    actual_canonical_segments = canonical.groupby("segment_id").size().to_dict()
    expected_canonical_segments = {
        int(key): int(value) for key, value in cleaning["expected_segment_rows"].items()
    }
    _require(
        actual_canonical_segments == expected_canonical_segments,
        f"canonical segment counts changed: {actual_canonical_segments}",
    )
    eligible = canonical.loc[canonical["train_eligible"].eq(True)]
    _require(
        len(eligible) == int(cleaning["expected_train_eligible_rows"]),
        "train-eligible row count changed",
    )
    actual_segment = eligible.groupby("segment_id").size().to_dict()
    expected_segment = {int(key): int(value) for key, value in cleaning["expected_train_segment_rows"].items()}
    _require(actual_segment == expected_segment, f"eligible segment counts changed: {actual_segment}")
    label_values = canonical[[f"vive_pos_raw_{axis}_mm" for axis in "xyz"]].to_numpy(float)
    _require(np.isfinite(label_values).all(), "canonical labels contain non-finite values")

    # Re-run the frozen, label-value-blind recipe over the retained source fields.
    # This checks UID stability, m->mm and raw-tick conversion, eligibility reasons,
    # warning flags, strict-QC status, and every derived feature value.
    rebuilt = clean_trials(
        canonical,
        cleaning,
        session_id=str(config["contract"]["session_id"]),
        source_member_sha256=str(config["source"]["member_sha256"]),
    )
    rebuilt = add_feature_columns(rebuilt)
    rebuilt.sort_values(["segment_id", "t_start_ns", "trial_uid"], inplace=True)
    rebuilt.reset_index(drop=True, inplace=True)
    actual = canonical.sort_values(
        ["segment_id", "t_start_ns", "trial_uid"], kind="mergesort"
    ).reset_index(drop=True)
    derived_columns = [
        "session_id",
        "source_member_sha256",
        "trial_uid",
        *[f"vive_pos_raw_{axis}_mm" for axis in "xyz"],
        *[f"encoder_delta_rad_j{joint}" for joint in range(3)],
        "train_exclusion_reasons",
        "train_eligible",
        *[column for column in canonical.columns if column.startswith("warning_")],
        "qc_warning_flags",
        "strict_qc_eligible",
        "is_synthetic",
        "sample_weight",
        *feature_views()["four_imu_endpoint_v1"],
    ]
    derived_columns = list(dict.fromkeys(derived_columns))
    try:
        pd.testing.assert_frame_equal(
            actual.loc[:, derived_columns],
            rebuilt.loc[:, derived_columns],
            check_exact=True,
            check_dtype=True,
        )
    except AssertionError as exc:
        raise ValidationError("canonical derivation recipe does not reproduce") from exc

    _require(manifest.get("counts", {}).get("canonical_real") == len(canonical), "manifest canonical count mismatch")
    _require(
        manifest.get("counts", {}).get("train_eligible_real") == len(eligible),
        "manifest eligible count mismatch",
    )


def _validate_feature_sets(run: Path, canonical: pd.DataFrame) -> dict[str, list[str]]:
    feature_sets = _load_json(run / "canonical" / "feature_sets.json")
    expected_views = feature_views()
    _require(feature_sets == expected_views, "feature_sets.json differs from the v1 definitions")
    encoder = _view_columns(feature_sets, "encoder_only_v1")
    imu = _view_columns(feature_sets, "four_imu_endpoint_v1")
    fusion = _view_columns(feature_sets, "fusion_v1")
    command = _view_columns(feature_sets, "command_context_ablation_v1")
    magnetometer = _view_columns(feature_sets, "mag_ablation_v1")
    _require(len(encoder) == 3, "encoder_only_v1 must contain exactly 3 columns")
    _require(len(imu) == 40, "four_imu_endpoint_v1 must contain exactly 40 columns")
    _require(len(fusion) == 43, "fusion_v1 must contain exactly 43 columns")
    _require(len(command) == 46, "command_context_ablation_v1 must contain exactly 46 columns")
    _require(len(magnetometer) == 55, "mag_ablation_v1 must contain exactly 55 columns")
    _require(fusion == [*encoder, *imu], "fusion_v1 must be encoder then four-IMU columns")
    for name, columns in (("encoder_only_v1", encoder), ("four_imu_endpoint_v1", imu), ("fusion_v1", fusion)):
        _require(len(columns) == len(set(columns)), f"{name} contains duplicate columns")
        _require(set(columns) <= set(canonical.columns), f"{name} names absent canonical columns")
        prohibited = [
            column
            for column in columns
            if any(token in column.lower() for token in ("vive", "tracker", "planned", "logical_deg", "q_encoder_rad", "_mag_"))
        ]
        _require(not prohibited, f"{name} contains prohibited primary features: {prohibited}")
    schema = pd.read_csv(run / "canonical" / "column_schema.csv")
    _require(
        list(schema.columns) == ["column", "dtype", "role", "unit", "frame"],
        "column_schema.csv has an unexpected shape",
    )
    _require(schema["column"].tolist() == canonical.columns.tolist(), "column schema order differs from canonical")
    _require(schema["column"].is_unique, "column schema contains duplicate names")
    roles = schema.set_index("column")["role"]
    _require(
        set(roles.loc[roles.eq("label")].index) == {f"vive_pos_raw_{axis}_mm" for axis in "xyz"},
        "column schema label roles changed",
    )
    _require(
        set(roles.loc[roles.eq("primary_feature")].index) == set(fusion),
        "column schema primary-feature roles changed",
    )
    _require(
        set(command) - set(fusion) <= set(roles.loc[roles.eq("ablation_feature")].index),
        "planned ablation columns are not marked as ablation features",
    )
    _require(
        set(magnetometer) - set(fusion) <= set(roles.loc[roles.eq("ablation_feature")].index),
        "magnetometer columns are not marked as ablation features",
    )
    return feature_sets


def _validate_splits(
    config: dict[str, Any], manifest: dict[str, Any], canonical: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    required = {
        "scheme",
        "outer_fold",
        "inner_fold",
        "split_context",
        "trial_uid",
        "segment_id",
        "target_id",
        "target_group",
        "role",
    }
    _require(required <= set(assignments.columns), f"split assignments missing: {sorted(required-set(assignments.columns))}")
    eligible = canonical.loc[canonical["train_eligible"]].copy()
    eligible_ids = set(eligible["trial_uid"].astype(str))
    _require(set(assignments["trial_uid"].astype(str)) == eligible_ids, "split IDs differ from eligible real IDs")
    _require(set(assignments["role"].astype(str)) <= {"train", "validation", "test"}, "invalid split role")
    segments = [int(value) for value in config["splits"]["segment_outer_values"]]
    target_group_count = int(config["splits"]["target_group_count"])
    expected_contexts = {
        *(('segment_loso', outer, -1) for outer in segments),
        *(
            ('segment_loso', outer, inner)
            for outer in segments
            for inner in segments
            if inner != outer
        ),
        *(('target_group', outer, -1) for outer in range(target_group_count)),
    }
    actual_contexts = {
        (str(scheme), int(outer), int(inner))
        for scheme, outer, inner in assignments[
            ["scheme", "outer_fold", "inner_fold"]
        ].drop_duplicates().itertuples(index=False, name=None)
    }
    _require(actual_contexts == expected_contexts, "split context set changed")
    expected_rows = int(config["splits"].get("expected_assignment_rows", len(eligible) * 30))
    _require(len(assignments) == expected_rows, "split assignment row count changed")
    _require(
        manifest.get("counts", {}).get("split_assignments") == len(assignments),
        "manifest split assignment count mismatch",
    )
    indexed = eligible.set_index(eligible["trial_uid"].astype(str), drop=False)
    for key, context in assignments.groupby(["scheme", "outer_fold", "inner_fold"], observed=True):
        _require(len(context) == len(eligible), f"split context {key} does not assign every eligible row")
        _require(not context["trial_uid"].astype(str).duplicated().any(), f"duplicate row in split context {key}")
        scheme, outer_fold, inner_fold = key
        expected_context_name = (
            f"{scheme}:outer={int(outer_fold)}:inner="
            f"{'refit' if int(inner_fold) == -1 else int(inner_fold)}"
        )
        _require(
            context["split_context"].eq(expected_context_name).all(),
            f"split_context label mismatch in {key}",
        )
        joined = indexed.loc[context["trial_uid"].astype(str)]
        roles = context["role"].astype(str).to_numpy()
        if str(scheme) == "segment_loso":
            outer = int(outer_fold)
            _require(np.all((joined["segment_id"].to_numpy() == outer) == (roles == "test")), f"segment test leakage in {key}")
            if int(inner_fold) == -1:
                _require(not np.any(roles == "validation"), f"segment refit contains validation in {key}")
            else:
                inner = int(inner_fold)
                expected_validation = (joined["segment_id"].to_numpy() == inner)
                _require(np.all(expected_validation == (roles == "validation")), f"segment validation leakage in {key}")
        elif str(scheme) == "target_group":
            _require(int(inner_fold) == -1, "target_group v1 must be outer-only")
            _require(set(roles) <= {"train", "test"}, "target-group outer context contains validation")
            groups = joined["target_id"].astype(str).str[1:].astype(int).to_numpy() % target_group_count
            _require(np.all((groups == int(outer_fold)) == (roles == "test")), f"target-group leakage in {key}")
            test_target_ids = set(joined.loc[roles == "test", "target_id"].astype(str))
            expected_targets_per_group = int(
                config["splits"].get("expected_targets_per_group", 10)
            )
            _require(
                len(test_target_ids) == expected_targets_per_group,
                f"target-group fold {outer_fold} does not hold out {expected_targets_per_group} targets",
            )
        else:
            raise ValidationError(f"unknown split scheme: {scheme}")


def _validate_augmentations(
    run: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
    canonical: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[int, int]:
    methods = ("local_interp_v2", "cmixup_offline_v1")
    outer_values = {
        "segment_loso": [int(value) for value in config["splits"]["segment_outer_values"]],
        "target_group": list(range(int(config["splits"]["target_group_count"]))),
    }
    expected_paths = {
        (Path("augmentation") / scheme / f"fold_{fold}" / f"{method}.parquet").as_posix()
        for scheme, folds in outer_values.items()
        for fold in folds
        for method in methods
    }
    files = sorted((run / "augmentation").glob("*/fold_*/*.parquet"))
    actual_paths = {path.relative_to(run).as_posix() for path in files}
    _require(actual_paths == expected_paths, "outer augmentation file set changed")

    provenance_parts: list[pd.DataFrame] = []
    expected_manifest_rows: list[dict[str, Any]] = []
    all_synthetic_ids: set[str] = set()
    total_rows = 0
    expected_synthetic_schema: dict[str, Any] | None = None
    for scheme, folds in outer_values.items():
        for fold in folds:
            train, regenerated = generate_context_augmentations(
                canonical,
                assignments,
                config,
                scheme=scheme,
                outer_fold=int(fold),
                inner_fold=-1,
            )
            for method in methods:
                relative = Path("augmentation") / scheme / f"fold_{fold}" / f"{method}.parquet"
                path = run / relative
                actual = pd.read_parquet(path)
                expected = regenerated[method].copy()
                expected.attrs = {}
                try:
                    pd.testing.assert_frame_equal(
                        actual,
                        expected,
                        check_exact=True,
                        check_dtype=True,
                        check_like=False,
                    )
                except AssertionError as exc:
                    raise ValidationError(
                        f"{relative.as_posix()} is not reproducible from its declared train context"
                    ) from exc

                forbidden = [
                    column
                    for column in actual.columns
                    if column
                    in {
                        "target_id",
                        "segment_id",
                        "set_id",
                        "repeat_index",
                        "t_start_ns",
                        "t_end_ns",
                        "duration_s",
                        "train_eligible",
                        "train_exclusion_reasons",
                        "qc_warning_flags",
                    }
                    or column.endswith("_n")
                ]
                _require(not forbidden, f"{relative.as_posix()} copies real-only metadata")
                _require(
                    actual["anchor_trial_uid"].astype(str).equals(
                        actual["parent_a_trial_uid"].astype(str)
                    ),
                    f"{relative.as_posix()} anchor differs from parent A",
                )
                ids = set(actual["synthetic_id"].astype(str))
                _require(len(ids) == len(actual), f"{relative.as_posix()} has duplicate IDs")
                _require(not ids & all_synthetic_ids, "synthetic IDs collide across contexts")
                all_synthetic_ids.update(ids)
                _require(
                    actual["sample_weight"].sum() == float(len(train)),
                    f"{relative.as_posix()} does not preserve 1:1 weight mass",
                )
                if method == "local_interp_v2":
                    allowed = np.asarray(config["augmentation"]["local_lambdas"], dtype=float)
                    _require(
                        np.isin(actual["mix_lambda"].to_numpy(float), allowed).all(),
                        f"{relative.as_posix()} contains an undeclared local lambda",
                    )

                current_schema = schema_payload(actual)
                if expected_synthetic_schema is None:
                    expected_synthetic_schema = current_schema
                _require(
                    current_schema == expected_synthetic_schema,
                    f"{relative.as_posix()} Arrow schema changed",
                )
                provenance_parts.append(actual.loc[:, list(PROVENANCE_COLUMNS)].copy())
                attrs = regenerated[method].attrs
                expected_manifest_rows.append(
                    {
                        "scheme": scheme,
                        "outer_fold": int(fold),
                        "method": method,
                        "real_train_rows": int(len(train)),
                        "rows": int(len(actual)),
                        "path": relative.as_posix(),
                        "cmixup_sigma_mm": (
                            float(attrs["cmixup_sigma_mm"])
                            if method == "cmixup_offline_v1"
                            else None
                        ),
                        "missing_anchor_count": int(attrs.get("missing_anchor_count", 0)),
                        "replacement_count": int(attrs.get("replacement_count", 0)),
                    }
                )
                total_rows += len(actual)

    _require(
        total_rows == int(config["augmentation"]["expected_total_synthetic_rows"]),
        "total synthetic rows changed",
    )
    _require(
        manifest.get("counts", {}).get("synthetic_materialized") == total_rows,
        "manifest synthetic count mismatch",
    )
    _require(
        manifest.get("augmentation") == expected_manifest_rows,
        "manifest augmentation statistics differ from regenerated folds",
    )
    aggregate = pd.read_parquet(run / "augmentation" / "provenance.parquet")
    expected_aggregate = pd.concat(provenance_parts, ignore_index=True)
    aggregate.sort_values("synthetic_id", kind="mergesort", inplace=True)
    expected_aggregate.sort_values("synthetic_id", kind="mergesort", inplace=True)
    aggregate.reset_index(drop=True, inplace=True)
    expected_aggregate.reset_index(drop=True, inplace=True)
    try:
        pd.testing.assert_frame_equal(
            aggregate,
            expected_aggregate,
            check_exact=True,
            check_dtype=True,
            check_like=False,
        )
    except AssertionError as exc:
        raise ValidationError("aggregate augmentation provenance differs from fold files") from exc
    _require(
        schema_payload(aggregate)
        == manifest.get("schemas", {}).get("augmentation_provenance"),
        "aggregate provenance Arrow schema fingerprint changed",
    )
    _require(
        expected_synthetic_schema == manifest.get("schemas", {}).get("synthetic_fold"),
        "synthetic Arrow schema fingerprint changed",
    )
    return len(files), total_rows


def _validate_reported_analysis(
    run: Path, canonical: pd.DataFrame, assignments: pd.DataFrame
) -> None:
    audit = _load_json(run / "analysis" / "audit.json")
    provenance = pd.read_parquet(run / "augmentation" / "provenance.parquet")
    expected_sensitivity = strict_qc_sensitivity(canonical)
    expected_leakage = leakage_check_summary(assignments, provenance)
    _require(
        audit.get("strict_qc_sensitivity") == expected_sensitivity,
        "reported strict-QC sensitivity differs from canonical data",
    )
    _require(
        audit.get("leakage_checks") == expected_leakage,
        "reported leakage checks differ from split/provenance data",
    )
    _require(
        all(value == 0 for value in expected_leakage.values()),
        f"one or more leakage checks failed: {expected_leakage}",
    )


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    _require(run.is_dir(), f"run directory does not exist: {run}")
    config, manifest, canonical, assignments = _load_core(run)
    _require(manifest.get("contract") == "alphaf.ml_preparation.v1", "unexpected contract name")
    _validate_manifest_files(run, manifest)
    _validate_identity_and_schemas(run, config, manifest, canonical, assignments)
    _validate_canonical(config, manifest, canonical)
    _validate_feature_sets(run, canonical)
    _validate_splits(config, manifest, canonical, assignments)
    augmentation_files, synthetic_rows = _validate_augmentations(
        run, config, manifest, canonical, assignments
    )
    _validate_reported_analysis(run, canonical, assignments)
    return {
        "status": "ok",
        "contract": manifest["contract"],
        "canonical_rows": int(len(canonical)),
        "train_eligible_rows": int(canonical["train_eligible"].sum()),
        "split_assignment_rows": int(len(assignments)),
        "augmentation_files": augmentation_files,
        "synthetic_rows": synthetic_rows,
        "manifest_files": len(manifest["files"]),
    }


def validate_package(path: str | Path) -> dict[str, Any]:
    archive_path = Path(path).resolve()
    with ZipFile(archive_path) as archive:
        bad = archive.testzip()
        _require(bad is None, f"ZIP CRC failure: {bad}")
        manifest_name = f"{PACKAGE_ROOT}/manifest.json"
        _require(manifest_name in archive.namelist(), "package manifest is missing")
        manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        expected_names = {manifest_name}
        for entry in manifest["files"]:
            name = f"{PACKAGE_ROOT}/{entry['path']}"
            expected_names.add(name)
            payload = archive.read(name)
            _require(len(payload) == entry["bytes"], f"packaged size mismatch: {name}")
            _require(hashlib.sha256(payload).hexdigest() == entry["sha256"], f"packaged SHA mismatch: {name}")
        actual_names = {name for name in archive.namelist() if not name.endswith("/")}
        _require(actual_names == expected_names, "package contains unmanifested or missing files")
    return {
        "status": "ok",
        "contract": manifest["contract"],
        "files": len(expected_names),
        "bytes": archive_path.stat().st_size,
    }

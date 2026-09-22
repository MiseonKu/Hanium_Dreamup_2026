from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import feature_views
from .io_utils import write_json, write_text


LABEL_COLUMNS = [
    "vive_pos_raw_x_mm",
    "vive_pos_raw_y_mm",
    "vive_pos_raw_z_mm",
]


def _flag_counts(series: pd.Series) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for raw in series.fillna("").astype(str):
        for flag in raw.replace(";", "|").split("|"):
            value = flag.strip()
            if value:
                counts[value] += 1
    return dict(sorted(counts.items()))


def _python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _python(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_python(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def target_shortcut_loso(canonical: pd.DataFrame) -> dict[str, Any]:
    eligible = canonical.loc[canonical["train_eligible"]].copy()
    fold_rows: list[dict[str, Any]] = []
    all_errors: list[np.ndarray] = []
    for segment in sorted(eligible["segment_id"].unique()):
        train = eligible.loc[eligible["segment_id"] != segment]
        test = eligible.loc[eligible["segment_id"] == segment]
        target_means = train.groupby("target_id", observed=True)[LABEL_COLUMNS].mean()
        predicted = test[["target_id"]].join(target_means, on="target_id")[LABEL_COLUMNS]
        actual = test[LABEL_COLUMNS].to_numpy(dtype=float)
        error = predicted.to_numpy(dtype=float) - actual
        if not np.isfinite(error).all():
            raise ValueError(f"target shortcut could not predict every target in segment {segment}")
        all_errors.append(error)
        norm = np.linalg.norm(error, axis=1)
        fold_rows.append(
            {
                "held_segment": int(segment),
                "rows": int(len(test)),
                "mean_3d_error_mm": float(norm.mean()),
                "median_3d_error_mm": float(np.median(norm)),
                "rmse_xyz_mm": float(np.sqrt(np.mean(np.square(error)))),
            }
        )
    pooled = np.concatenate(all_errors, axis=0)
    pooled_norm = np.linalg.norm(pooled, axis=1)
    return {
        "description": "Train-fold target-ID mean baseline; diagnostic only, never a primary feature.",
        "folds": fold_rows,
        "pooled_mean_3d_error_mm": float(pooled_norm.mean()),
        "pooled_median_3d_error_mm": float(np.median(pooled_norm)),
        "pooled_rmse_xyz_mm": float(np.sqrt(np.mean(np.square(pooled)))),
    }


def _cohort_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("quality-sensitivity cohort must not be empty")
    segment_target = frame.groupby(
        ["segment_id", "target_id"], observed=True
    )[LABEL_COLUMNS].agg(["mean", "std"])
    within = segment_target.xs("std", axis=1, level=1)
    within_3d = np.sqrt(np.square(within).sum(axis=1, min_count=3))
    centroids = segment_target.xs("mean", axis=1, level=1).reset_index()
    target_centroid_std = centroids.groupby("target_id", observed=True)[
        LABEL_COLUMNS
    ].std()
    drift_3d = np.sqrt(np.square(target_centroid_std).sum(axis=1, min_count=3))
    means = frame[LABEL_COLUMNS].mean()
    stds = frame[LABEL_COLUMNS].std()
    return _python(
        {
            "rows": int(len(frame)),
            "segment_rows": {
                str(key): int(value)
                for key, value in frame.groupby("segment_id", observed=True).size().items()
            },
            "label_mean_mm": {
                axis: float(means[column])
                for axis, column in zip("xyz", LABEL_COLUMNS, strict=True)
            },
            "label_std_mm": {
                axis: float(stds[column])
                for axis, column in zip("xyz", LABEL_COLUMNS, strict=True)
            },
            "median_within_segment_target_repeatability_std_3d_mm": float(
                within_3d.median()
            ),
            "median_between_segment_centroid_drift_std_3d_mm": float(
                drift_3d.median()
            ),
        }
    )


def strict_qc_sensitivity(canonical: pd.DataFrame) -> dict[str, Any]:
    """Compare the full eligible cohort with the warning-free strict slice."""

    full_metrics = _cohort_metrics(
        canonical.loc[canonical["train_eligible"].eq(True)]
    )
    strict_metrics = _cohort_metrics(
        canonical.loc[canonical["strict_qc_eligible"].eq(True)]
    )
    delta: dict[str, Any] = {
        key: strict_metrics[key] - full_metrics[key]
        for key in (
            "rows",
            "median_within_segment_target_repeatability_std_3d_mm",
            "median_between_segment_centroid_drift_std_3d_mm",
        )
    }
    for name in ("label_mean_mm", "label_std_mm"):
        delta[name] = {
            axis: strict_metrics[name][axis] - full_metrics[name][axis]
            for axis in "xyz"
        }
    return _python(
        {
            "description": (
                "Warning thresholds are not hard filters; this compares all "
                "train-eligible rows with the warning-free sensitivity cohort."
            ),
            "full_eligible": full_metrics,
            "strict_qc": strict_metrics,
            "delta_strict_minus_full": delta,
        }
    )


def leakage_check_summary(
    assignments: pd.DataFrame, provenance: pd.DataFrame
) -> dict[str, int]:
    """Return explicit, count-valued leakage checks for the release report."""

    duplicate_assignments = int(
        assignments.duplicated(
            ["scheme", "outer_fold", "inner_fold", "trial_uid"]
        ).sum()
    )
    primary = feature_views()["fusion_v1"]
    tracker_features = [
        column
        for column in primary
        if column.lower().startswith(("vive_", "tracker_"))
    ]
    planned_features = [column for column in primary if column.startswith("planned_")]
    if provenance.empty:
        self_pairs = 0
        duplicate_synthetic = 0
        parent_context_violations = 0
        non_refit_synthetic = 0
        synthetic_in_assignments = 0
    else:
        self_pairs = int(
            provenance["parent_a_trial_uid"].astype(str).eq(
                provenance["parent_b_trial_uid"].astype(str)
            ).sum()
        )
        duplicate_synthetic = int(provenance["synthetic_id"].duplicated().sum())
        non_refit_synthetic = int(provenance["inner_fold"].ne(-1).sum())
        synthetic_in_assignments = len(
            set(provenance["synthetic_id"].astype(str))
            & set(assignments["trial_uid"].astype(str))
        )
        parent_context_violations = 0
        for (scheme, outer_fold), rows in provenance.groupby(
            ["split_scheme", "outer_fold"], observed=True
        ):
            train_ids = set(
                assignments.loc[
                    assignments["scheme"].eq(scheme)
                    & assignments["outer_fold"].eq(int(outer_fold))
                    & assignments["inner_fold"].eq(-1)
                    & assignments["role"].eq("train"),
                    "trial_uid",
                ].astype(str)
            )
            for column in ("parent_a_trial_uid", "parent_b_trial_uid"):
                parent_context_violations += int(
                    (~rows[column].astype(str).isin(train_ids)).sum()
                )
    return {
        "split_context_duplicate_assignments": duplicate_assignments,
        "primary_tracker_feature_columns": len(tracker_features),
        "primary_planned_feature_columns": len(planned_features),
        "synthetic_self_pairs": self_pairs,
        "synthetic_id_collisions": duplicate_synthetic,
        "synthetic_parent_outside_outer_train": parent_context_violations,
        "materialized_non_refit_synthetic_rows": non_refit_synthetic,
        "synthetic_rows_present_in_real_split_assignments": synthetic_in_assignments,
    }


def build_analysis(
    source: pd.DataFrame,
    canonical: pd.DataFrame,
    assignments: pd.DataFrame,
    augmentation_rows: list[dict[str, Any]],
    provenance: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    segment_summary = (
        canonical.groupby("segment_id", observed=True)
        .agg(
            canonical_rows=("trial_uid", "size"),
            train_eligible_rows=("train_eligible", "sum"),
            vive_x_mean_mm=(LABEL_COLUMNS[0], "mean"),
            vive_y_mean_mm=(LABEL_COLUMNS[1], "mean"),
            vive_z_mean_mm=(LABEL_COLUMNS[2], "mean"),
            vive_x_std_mm=(LABEL_COLUMNS[0], "std"),
            vive_y_std_mm=(LABEL_COLUMNS[1], "std"),
            vive_z_std_mm=(LABEL_COLUMNS[2], "std"),
        )
        .reset_index()
    )
    eligible = canonical.loc[canonical["train_eligible"]]
    segment_target_summary = (
        eligible.groupby(["segment_id", "target_id"], observed=True)
        .agg(
            rows=("trial_uid", "size"),
            vive_x_centroid_mm=(LABEL_COLUMNS[0], "mean"),
            vive_y_centroid_mm=(LABEL_COLUMNS[1], "mean"),
            vive_z_centroid_mm=(LABEL_COLUMNS[2], "mean"),
            vive_x_within_std_mm=(LABEL_COLUMNS[0], "std"),
            vive_y_within_std_mm=(LABEL_COLUMNS[1], "std"),
            vive_z_within_std_mm=(LABEL_COLUMNS[2], "std"),
        )
        .reset_index()
    )
    within_std = segment_target_summary[
        [
            "vive_x_within_std_mm",
            "vive_y_within_std_mm",
            "vive_z_within_std_mm",
        ]
    ]
    segment_target_summary["within_segment_target_repeatability_std_3d_mm"] = np.sqrt(
        np.square(within_std).sum(axis=1, min_count=3)
    )

    target_summary = segment_target_summary.groupby(
        "target_id", observed=True
    ).agg(
        rows=("rows", "sum"),
        segments=("segment_id", "size"),
        vive_x_centroid_mean_mm=("vive_x_centroid_mm", "mean"),
        vive_y_centroid_mean_mm=("vive_y_centroid_mm", "mean"),
        vive_z_centroid_mean_mm=("vive_z_centroid_mm", "mean"),
        vive_x_centroid_std_mm=("vive_x_centroid_mm", "std"),
        vive_y_centroid_std_mm=("vive_y_centroid_mm", "std"),
        vive_z_centroid_std_mm=("vive_z_centroid_mm", "std"),
    ).reset_index()
    centroid_std = target_summary[
        [
            "vive_x_centroid_std_mm",
            "vive_y_centroid_std_mm",
            "vive_z_centroid_std_mm",
        ]
    ]
    target_summary["between_segment_centroid_drift_std_3d_mm"] = np.sqrt(
        np.square(centroid_std).sum(axis=1, min_count=3)
    )

    missing = source.isna().sum()
    missing = missing.loc[missing > 0].sort_values(ascending=False)
    split_summary = (
        assignments.groupby(["scheme", "outer_fold", "inner_fold", "role"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
        .to_dict(orient="records")
    )
    analysis = {
        "source_rows": int(len(source)),
        "source_columns": int(source.shape[1]),
        "source_segment_rows": {
            str(key): int(value) for key, value in source.groupby("segment_id").size().items()
        },
        "source_missing_cells_by_column": {
            str(key): int(value) for key, value in missing.items()
        },
        "canonical_rows": int(len(canonical)),
        "train_eligible_rows": int(canonical["train_eligible"].sum()),
        "train_exclusion_reasons": _flag_counts(canonical["train_exclusion_reasons"]),
        "qc_warning_flags": _flag_counts(canonical["qc_warning_flags"]),
        "strict_qc_sensitivity": strict_qc_sensitivity(canonical),
        "leakage_checks": leakage_check_summary(
            assignments,
            provenance if provenance is not None else pd.DataFrame(),
        ),
        "target_count": int(canonical["target_id"].nunique()),
        "median_within_segment_target_repeatability_std_3d_mm": float(
            segment_target_summary[
                "within_segment_target_repeatability_std_3d_mm"
            ].median()
        ),
        "median_between_segment_centroid_drift_std_3d_mm": float(
            target_summary["between_segment_centroid_drift_std_3d_mm"].median()
        ),
        "target_shortcut_segment_loso": target_shortcut_loso(canonical),
        "split_summary": split_summary,
        "augmentation": augmentation_rows,
        "limitations": [
            "Vive labels remain in vive_world; tracker-to-base/TCP calibration is unavailable.",
            "WT901C Euler-axis convention and configured full-scale were not independently verified.",
            "The four-IMU view is an endpoint-regression proxy, not formal three-IMU-per-link ShapeFK.",
            "Synthetic samples are mathematical interpolants, not additional physical measurements.",
        ],
    }
    return _python(analysis), segment_summary, segment_target_summary, target_summary


def write_analysis(
    output_dir: str | Path,
    source: pd.DataFrame,
    canonical: pd.DataFrame,
    assignments: pd.DataFrame,
    augmentation_rows: list[dict[str, Any]],
    provenance: pd.DataFrame | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    analysis, segment_summary, segment_target_summary, target_summary = build_analysis(
        source, canonical, assignments, augmentation_rows, provenance
    )
    write_json(output / "audit.json", analysis)
    segment_summary.to_csv(output / "segment_summary.csv", index=False, lineterminator="\n")
    segment_target_summary.to_csv(
        output / "segment_target_summary.csv", index=False, lineterminator="\n"
    )
    target_summary.to_csv(output / "target_summary.csv", index=False, lineterminator="\n")

    shortcut = analysis["target_shortcut_segment_loso"]
    sensitivity = analysis["strict_qc_sensitivity"]
    leakage = analysis["leakage_checks"]
    lines = [
        "# Collected Data Audit",
        "",
        f"- Source trials: {analysis['source_rows']:,}",
        f"- Canonical real trials: {analysis['canonical_rows']:,}",
        f"- Train-eligible real trials: {analysis['train_eligible_rows']:,}",
        f"- Materialized synthetic trials: {sum(row['rows'] for row in augmentation_rows):,}",
        "- Median within-(segment, target) repeatability 3D std: "
        f"{analysis['median_within_segment_target_repeatability_std_3d_mm']:.3f} mm",
        "- Median between-segment target-centroid drift 3D std: "
        f"{analysis['median_between_segment_centroid_drift_std_3d_mm']:.3f} mm",
        f"- Target-only segment-LOSO mean 3D error: {shortcut['pooled_mean_3d_error_mm']:.3f} mm",
        f"- Strict-QC sensitivity rows: {sensitivity['strict_qc']['rows']:,} "
        f"of {sensitivity['full_eligible']['rows']:,}",
        "- Strict-QC repeatability delta: "
        f"{sensitivity['delta_strict_minus_full']['median_within_segment_target_repeatability_std_3d_mm']:+.3f} mm",
        "- Strict-QC segment-centroid drift delta: "
        f"{sensitivity['delta_strict_minus_full']['median_between_segment_centroid_drift_std_3d_mm']:+.3f} mm",
        "",
        "The target-only score is a leakage diagnostic. Planned target values are excluded from primary features.",
        "Quality thresholds are label-blind; warning flags are retained for sensitivity analysis.",
        "No all-data Kabsch alignment or Tracker feedback is used.",
        "",
        "## Leakage checks",
        "",
    ]
    lines.extend(f"- {name}: {value}" for name, value in leakage.items())
    lines.extend(
        [
            "",
            "All leakage-check counts must remain zero.",
            "",
            "## Known limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in analysis["limitations"])
    write_text(output / "report.md", "\n".join(lines))
    return analysis

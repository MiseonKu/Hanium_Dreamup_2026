"""Deterministic tables, figures, report, and model card for experiment v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .experiment_io import atomic_write_text, read_json_object


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in selected.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, float):
                rendered.append(f"{value:.4f}")
            else:
                rendered.append(str(value))
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join([header, rule, *rows])


def _write_summary_plot(run: Path, aggregate: pd.DataFrame) -> Path:
    segment = aggregate.loc[
        aggregate["scheme"].eq("segment_loso")
        & aggregate["deployment_candidate"].eq(True)
    ].nsmallest(15, "macro_rmse_3d_mm")
    figure, axis = plt.subplots(figsize=(11, 6.5), dpi=120)
    labels = segment["cell_id"].tolist()[::-1]
    values = segment["macro_rmse_3d_mm"].tolist()[::-1]
    axis.barh(labels, values, color="#2864b4")
    axis.set_xlabel("5-fold macro RMSE_3D (mm; lower is better)")
    axis.set_ylabel("Pre-registered cell")
    axis.set_title("Exploratory segment-LOSO endpoint regression")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = run / "reports" / "segment_loso_macro_rmse.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=120,
        bbox_inches="tight",
        metadata={"Software": "imu-vive-learning"},
    )
    plt.close(figure)
    return path


def write_experiment_report(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir).resolve()
    aggregate = pd.read_parquet(run / "metrics" / "aggregate_metrics.parquet")
    folds = pd.read_parquet(run / "metrics" / "fold_metrics.parquet")
    gates = pd.read_parquet(run / "statistics" / "augmentation_gates.parquet")
    paired = pd.read_parquet(run / "statistics" / "paired_comparisons.parquet")
    final = read_json_object(run / "final" / "model_metadata.json")
    release = read_json_object(run / "release_identity.json")
    _write_summary_plot(run, aggregate)

    segment = aggregate.loc[
        aggregate["scheme"].eq("segment_loso")
        & aggregate["deployment_candidate"].eq(True)
    ].nsmallest(12, "macro_rmse_3d_mm")
    target = aggregate.loc[
        aggregate["scheme"].eq("target_group")
        & aggregate["deployment_candidate"].eq(True)
    ].nsmallest(12, "macro_rmse_3d_mm")
    diagnostics = aggregate.loc[aggregate["diagnostic_only"].eq(True)].sort_values(
        ["scheme", "macro_rmse_3d_mm"]
    )
    passed_count = int(gates["passed"].sum())
    selected = final["selection"]
    report = f"""# Endpoint Regression Experiment v1 — Exploratory Report

This report is generated from the locked nested group-CV outputs of
`alphaf.endpoint_regression.experiment.v1`. The prediction target is raw Vive
Tracker XYZ in the `vive_world` frame, in millimetres. This is a static 3-second
trial-summary endpoint-regression/fusion proxy. It is not robot-base/TCP
position, formal ShapeFK, the repository's `IMU-only` condition, a realtime
aggregator, or a control model.

The experiment is exploratory. The implementation and search space were
pre-frozen before this full run, but the same collected dataset had already
been used for feasibility smoke checks. No independent acquisition session is
available, so no confirmatory superiority or control-performance claim is made.

## Release and evaluation contract

- Parent release SHA-256: `{release['archive_sha256']}`
- Eligible measured trials: `{release['train_eligible_rows']}`
- Strict-QC measured trials: `{release['strict_qc_rows']}`
- Primary evaluation: five segment-LOSO outer folds, with four group-held-out
  inner contexts per outer fold.
- Secondary evaluation: five unseen-target outer groups, with four target-group
  inner contexts per outer group.
- Primary selection metric: `RMSE_3D = sqrt(mean(dx^2 + dy^2 + dz^2))`.
- Validation and test predictions contain measured real rows only.

## Primary segment-LOSO results

{_markdown_table(segment, ['cell_id', 'macro_rmse_3d_mm', 'macro_p95_3d_error_mm', 'pooled_median_3d_error_mm', 'seed_macro_rmse_std_mm'])}

![Segment LOSO macro RMSE](segment_loso_macro_rmse.png)

## Secondary unseen-target results

{_markdown_table(target, ['cell_id', 'macro_rmse_3d_mm', 'macro_p95_3d_error_mm', 'pooled_median_3d_error_mm', 'seed_macro_rmse_std_mm'])}

The two evaluation schemes are reported separately; their trials are not
pooled to inflate the effective sample size.

## Augmentation engineering gate

The frozen gate requires a reduction versus the same model's fusion real-only
cell of at least `max(10%, 3 mm)` in macro RMSE_3D, while macro P95 may not
worsen by more than 5%. `{passed_count}` of `{len(gates)}` model/augmentation
comparisons passed.

{_markdown_table(gates, ['model', 'recipe', 'actual_rmse_reduction_mm', 'required_rmse_reduction_mm', 'candidate_macro_p95_3d_error_mm', 'p95_limit_mm', 'passed'])}

Replay remains a negative exposure-control and cannot be deployed. Local
interpolation and C-Mixup remain separate hypotheses. Multiway bootstrap
intervals are paired by trial and crossed segment/target factors; no p-values
are reported because only five segment groups exist.

## Final all-data model

- Selected cell: `{selected['cell_id']}`
- Model / view / recipe: `{selected['model']} / {selected['feature_view']} / {selected['recipe']}`
- Segment-inner aggregate RMSE_3D: `{selected['aggregate_inner_rmse_3d_mm']:.4f} mm`
- Selection rule: `{selected['selection_rule']}`
- Final measured training rows: `{final['fit']['real_rows']}`
- Final synthetic training rows: `{final['fit']['additional_rows']}`
- Independent test score for this all-data refit: **none**

The serialized model carries an exact ordered feature allow-list, train-only
preprocessing, frame/unit metadata, source/config hashes, and a fixed-fixture
prediction hash. The package contains a Python/joblib estimator and must be
loaded only from this trusted, checksum-verified project release; joblib is not
a safe format for untrusted files. The strict-QC refit is a sensitivity
artifact; its reported fit error is resubstitution-only and not an independent
estimate.

## Shortcut diagnostics

{_markdown_table(diagnostics, ['scheme', 'cell_id', 'macro_rmse_3d_mm', 'macro_p95_3d_error_mm'])}

Planned coordinates and uncalibrated magnetometer channels are diagnostic
shortcuts only. Any apparent gain may encode schedule, environment, or session
identity and is excluded from final deployment selection.

## Reproducibility and limitations

All feature scaling, target centring/scaling, hyperparameter selection,
augmentation parent search, and C-Mixup bandwidth estimation use only the real
train role of the relevant context. Estimator internal thread counts are one;
only context tasks are parallel. Checkpoint reuse is accepted only when the
task, config, split, and release hashes match.

The data cover one static acquisition corpus and raw Vive world coordinates.
An online window aggregator, tracker-to-base/TCP calibration, dynamic motion
validation, external-session validation, and closed-loop safety validation do
not exist in this contract. See `docs/ENDPOINT_REGRESSION_METHODS.md` in the
source repository for the research basis and the boundary between published
methods and project-specific hypotheses.
"""
    atomic_write_text(run / "reports" / "report.md", report)

    model_card = f"""# Model Card — AlphaF Endpoint Regression Experiment v1

## Intended use

Offline exploratory prediction of raw `vive_world` XYZ millimetres from one
static trial-summary feature row that exactly matches the packaged ordered
allow-list.

## Not intended for

- realtime or closed-loop control;
- robot-base/TCP coordinates;
- formal ShapeFK or the repository's `IMU-only` condition;
- dynamic motion, safety-critical decisions, or a new hardware session;
- use of planned, target-ID, Vive, Tracker, QC, timestamp, or provenance fields
  as primary input.

## Selected system

- Cell: `{selected['cell_id']}`
- Parameters: `{json.dumps(selected['params'], sort_keys=True)}`
- Feature count: `{final['fit']['feature_count']}`
- Ensemble members: `{final['fit']['ensemble_members']}`
- Model SHA-256: `{final['model']['sha256']}`
- Contract fingerprint: `{final['model']['contract_fingerprint']}`
- Output frame/unit: `vive_world` / `mm`

## Evidence status

Exploratory nested group-CV on the same corpus used for feasibility checks.
The final all-data model has no independent test score. Segment and unseen-
target OOF estimates, every fold effect, strict-QC sensitivity, and paired
multiway-bootstrap intervals are packaged for audit. Independent acquisition-
session validation is required before any confirmatory claim.

## Input and output

`predict-endpoint` accepts only the exact ordered feature columns below, plus
an optional leading `row_id`:

```text
{chr(10).join(final['fit']['feature_columns'])}
```

Outputs are `pred_vive_world_x_mm`, `pred_vive_world_y_mm`, and
`pred_vive_world_z_mm`, together with model/package identity and frame/unit.
The package embeds a Python/joblib object. Load only the trusted Git LFS
release after checksum validation; never pass an untrusted ZIP to this CLI.
"""
    atomic_write_text(run / "MODEL_CARD.md", model_card)
    return {
        "report": "reports/report.md",
        "model_card": "MODEL_CARD.md",
        "plot": "reports/segment_loso_macro_rmse.png",
        "paired_comparison_rows": int(len(paired)),
        "fold_metric_rows": int(len(folds)),
    }


__all__ = ["write_experiment_report"]

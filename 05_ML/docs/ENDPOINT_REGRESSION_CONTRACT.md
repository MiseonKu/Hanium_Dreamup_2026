# Endpoint Regression Experiment Contract V1

## Status and scope

`alphaf.endpoint_regression.experiment.v1` is an **exploratory** offline
experiment over the immutable `alphaf.ml_preparation.v1` release. It fits
static endpoint regressors to three-second trial summaries and emits raw
`vive_world` XYZ positions in millimetres.

The experiment does not provide an independent-session test. Its nested
out-of-fold results measure generalisation within the already collected
session and must not be described as confirmatory superiority, robot-base or
TCP accuracy, online control performance, formal ShapeFK, or the repository's
three-IMU-per-link `IMU-only` condition. Vive is ground truth only; it is never
a primary input or control signal.

The parent preparation ZIP is read-only. This contract does not alter
`alphaf.ml_preparation.v1` or its members.

## Public commands

```powershell
cd "09_SO101_IMU_VIVE_LEARNING"
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install -r requirements-eval.lock
python -m pip install -e . --no-deps

python -m imu_vive_learning.cli run-experiment `
  --release ../06_simulation_and_data_collect_preparation/imu_added_experiment/ml_preparation_v1.zip `
  --config configs/endpoint_regression_v1.yaml `
  --output artifacts/endpoint_regression_v1 `
  --jobs 4 --resume

python -m imu_vive_learning.cli validate-experiment `
  --run-dir artifacts/endpoint_regression_v1

python -m imu_vive_learning.cli report-experiment `
  --run-dir artifacts/endpoint_regression_v1

python -m imu_vive_learning.cli package-experiment `
  --run-dir artifacts/endpoint_regression_v1 `
  --output artifacts/endpoint_regression_experiment_v1.zip

python -m imu_vive_learning.cli predict-endpoint `
  --model-package artifacts/endpoint_regression_experiment_v1.zip `
  --input feature_rows.parquet `
  --output predictions.parquet
```

For the exact CPython 3.12 Windows evaluation environment, install
`requirements-eval.lock` and then install the workspace editable with
`--no-deps`. The project extra uses XGBoost `>=3.3,<3.4` and the lock pins
3.3.0 exactly as pre-registered. XGBoost 3.3.0 requires Python 3.12 or newer
according to its [official PyPI metadata](https://pypi.org/project/xgboost/3.3.0/),
so evaluation must not run under Python 3.11. The preparation-only package
remains compatible with Python 3.11 because XGBoost is not a base dependency.
The convenience resolver command `python -m pip install -e ".[eval,dev]"`
remains supported under Python 3.12, but release experiments use the exact
lock sequence above.

## Fail-closed parent release

The accepted parent is exactly:

```text
06_simulation_and_data_collect_preparation/imu_added_experiment/
  ml_preparation_v1.zip
SHA-256:
  94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930
```

Before reading data, the experiment rejects a Git LFS pointer, an archive hash
mismatch, a missing or malformed parent manifest, a contract-name mismatch,
an unexpected member, a member checksum mismatch, an Arrow schema drift, or a
fixed-count drift. Expected parent counts are 7,707 canonical real rows, 7,702
train-eligible rows, and 7,670 warning-free strict-QC rows.

## Split contract and leakage barriers

The experiment extends the parent assignments with nested target validation:

- Segment LOSO is primary. Each of segments 36-40 is one complete outer test
  group; each remaining segment is validation once inside that outer fold.
- Target-group evaluation is secondary. `int(target_id[1:]) % 5` defines five
  ten-target outer groups; each remaining group is validation once inside that
  outer fold.
- Each scheme therefore has five outer-refit and twenty inner contexts. Fifty
  contexts times 7,702 eligible trials gives exactly 385,100 assignment rows.

Every context has disjoint real train, validation, and test IDs. Feature
scalers, target centring/scaling, hyperparameter selection, augmentation
candidate search, and C-Mixup bandwidth are fit only from that context's real
train rows. Inner synthetic rows are regenerated from inner real train.
Existing materialised parent-release synthetic rows may be used only for the
matching outer refit. Validation and test are measured real rows only.

Primary feature allow-lists are `encoder_only_v1`,
`four_imu_endpoint_v1`, and `fusion_v1`. They exclude Vive/Tracker values,
planned coordinates, IDs, groups, timestamps, QC flags, and provenance.
Planned and uncalibrated magnetometer values occur only in explicitly named
shortcut diagnostics and are barred from deployment selection.

## Frozen experiment matrix

The train-fold mean Dummy is the lowest baseline. Ridge, ExtraTrees, XGBoost,
and a shared-output XYZ MLP are compared on encoder, four-IMU endpoint, and
fusion views with real-only training. On fusion, the same four estimators are
compared across real-only, replay, `local_interp_v2`, and
`cmixup_offline_v1`. Ridge and ExtraTrees additionally run planned-only,
fusion-plus-planned, and fusion-plus-magnetometer real-only shortcut
diagnostics.

Replay duplicates the real training rows once and is a negative control for
exposure. It is not a deployable recipe. Local interpolation and C-Mixup stay
separate because they test different hypotheses. The Dummy, replay, planned,
and magnetometer conditions can never become the final package model.

All grids, three paired MLP seeds, model and parameter tie-break orders,
sample weights, row-visit budgets, estimator thread limits, base seed, gate,
and bootstrap settings are frozen in `configs/endpoint_regression_v1.yaml`.
Real-only rows have weight 1.0. In replay or augmentation runs, original and
added rows each have weight 0.5, so the 1:1 added-data recipe preserves total
objective mass. MLP runs use a 400 real-only versus 200 doubled-row epoch
budget.

## Metrics, gate, and final refit

The primary error is

```text
RMSE_3D = sqrt(mean(dx^2 + dy^2 + dz^2))
```

It is not the axis-element pooled `rmse_xyz_mm`. The validator checks
`RMSE_3D = sqrt(3) * rmse_xyz_mm` on the same residual rows. Five outer-group
values receive equal weight in the macro score. Reports also include pooled
3D mean/median/P90/P95, per-axis bias/MAE/RMSE/R2, segment-by-target
cell-balanced results, seed variability, and every fold effect.

An augmentation recipe passes its engineering gate for a model only when,
against that model's fusion real-only result, macro RMSE improves by at least
`max(10%, 3 mm)` and macro P95 does not worsen by more than 5%. Deterministic
10,000-replicate segment-by-target multiway bootstrap intervals describe
paired differences. No p-value claim is made because five segment groups do
not support confident CV-variance inference.

Final selection uses aggregate results from the twenty segment inner
contexts. Deployment-safe real-only primary candidates always remain in the
pool; an augmented fusion candidate may join it only after passing the frozen
gate. Thus a gate pass never forces selection of an augmentation that is worse
than the best real-only candidate. If no augmentation passes, the pool is
real-only. If augmentation wins, it is regenerated 1:1 from all 7,702 eligible
real rows before the final refit. That all-data model has no independent test
score, which must remain explicit in its model card.

Strict QC is sensitivity analysis, not a second opportunity to tune the
locked OOF models. The report evaluates locked OOF predictions on the strict
subset and separately reports a strict-only refit of the selected final
candidate.

## Package members

The deterministic experiment package contains at least:

- resolved config and parent-release identity;
- complete evaluation assignments and inner selection traces;
- the minimal eligible-real audit table and row-level augmentation parent
  provenance needed to recompute train scope, physical distances, and C-Mixup
  bandwidth without trusting a prefilled leakage flag;
- seed-level and ensemble OOF predictions with exactly one ensemble prediction
  per outer-test trial and evaluation cell;
- fold, segment-by-target cell, and aggregate metrics, paired comparisons,
  multiway bootstrap output, gate decisions, and strict-QC sensitivity;
- final preprocessing state, trusted estimator payload, feature allow-list,
  fixed-fixture prediction hash, and model card;
- environment inventory, human-readable report, and a checksum manifest for
  every other member.

The run validator replays the production augmentation generator for all 100
context/recipe combinations (50 split contexts times two methods) from their
measured real-train rows and requires exact agreement for synthetic ID,
derived seed, parent and lambda draw, replica/replacement schedule,
distances, bandwidth, source/config identity, and weight. The same replay guards
future final-augmentation outputs, including the local top-five condition. This
exhaustive CPU-bound check can take several minutes on the completed release.

Resume writes are atomic and keyed by task, config, parent, and code hashes. A
hash mismatch is an error rather than a partial-run reuse. Repeating the run
under the same lock and config must reproduce OOF, metrics, and report member
hashes. Estimator serialization is verified by loading it and comparing a
fixed-fixture prediction hash.

## Inference boundary

`predict-endpoint` accepts only the packaged model's ordered feature allow-list
plus optional `row_id`. Missing, extra, duplicate, reordered, nonnumeric, or
nonfinite inputs fail closed. It outputs
`pred_vive_world_{x,y,z}_mm`, the model hash, `vive_world`, and `mm` (and copies
`row_id` when supplied). Predictions are static summary estimates, not online
sensor aggregation or robot-control commands.

The model member uses Python/joblib serialization. It is a trusted-artifact
interface, not a safe parser for arbitrary files: inference must use only the
checksum-validated Git LFS package produced by this repository.

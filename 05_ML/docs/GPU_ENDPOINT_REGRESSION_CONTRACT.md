# GPU endpoint-regression experiment contract

## Objective

The ultimate objective is to estimate the robot-arm endpoint without a Vive
Tracker at inference time.  The inputs are the three encoder deltas and the
four-IMU endpoint feature view; the current label is the measured static
endpoint in raw `vive_world` millimetres.

This experiment is an exploratory GPU successor to the frozen endpoint
regression v1 evaluation.  It does not alter or replace that release.  It is
not evidence for robot-base/TCP coordinates, dynamic state estimation,
ShapeFK, or control safety.

## Pre-training decisions

- The only model input is the 43-column `fusion_v1` allow-list.
- Vive, Tracker, planned position, row identity, split identity, QC, and
  provenance fields are forbidden model inputs.
- Only measured real rows are training examples.  Augmentation is excluded
  because the frozen v1 experiment passed 0 of 8 augmentation gates.
- PyTorch MLP is the primary family.  CUDA XGBoost is a reference baseline,
  not a family selected after looking at outer-test scores.
- The exact grids, budgets, seeds, preprocessing, and tie order are in
  `configs/endpoint_regression_gpu_v1.yaml`.

One `segment_loso:outer=36:inner=37` MLP configuration was run as a hardware
and convergence pilot before this contract was frozen.  Its validation score
therefore is not independent confirmation.  The full study remains explicitly
exploratory and has no independent test set.

## Leakage boundary and model selection

For each outer group, every other group is used once as inner validation.  A
configuration is selected only from the equal-group mean of those inner
validation RMSE values.  Only after selection is the outer-test group read for
prediction.  Preprocessing is fitted exclusively on the real training rows in
each context.

The MLP checkpoint within an inner context is the epoch with minimum validation
RMSE3D.  Outer refits cannot early-stop on the test group; their per-seed epoch
budgets are the rounded medians of the corresponding inner best epochs.  The
all-data model uses the configuration selected across all 20 segment inner
contexts and never claims an independent test score.

## Direction checks during training

The runner records train and validation RMSE every 10 epochs, best epoch,
generalization gap, seed dispersion, finite-value checks, and group-level RMSE
and P95.  A falling train loss alone is not success.  Useful direction means
lower held-out-group RMSE/P95 without a growing train-validation gap or a
single-group/seed shortcut.

Four CMP 170HX GPUs run independent fold/configuration jobs, one process per
GPU.  The dataset is too small for efficient data-parallel training of one
model, so the architecture is not enlarged merely to increase utilization.

## Deployment boundary

Before passive real-world monitoring, the result still needs independent
collection sessions, Tracker-to-robot-base and endpoint/TCP calibration,
uncertainty calibration, OOD detection, and an abstention policy.  Any use for
robot control is a separate safety-qualified project.

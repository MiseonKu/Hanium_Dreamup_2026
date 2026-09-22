# Preregistration Template

Freeze this file before collecting real experiment data. If any item changes after data collection starts, add a dated changelog entry.

## Hypotheses

- H1: Encoder-only rigid FK has systematic bias in large-reach or near-horizontal poses.
- H2: IMU-array ShapeFK reduces end-effector RMSE under the same downstream correction rule.
- H3: The IMU-array benefit is larger in high-deflection target regions.
- H4: Kalman or Madgwick attitude processing improves low-cost IMU ShapeFK over raw gyro propagation.
- H5: WT901C internal attitude output is more stable than low-cost IMU raw processing.
- H6: Encoder + IMU fusion reduces total RMSE or repeated-trial variance.
- H7: Shape reconstruction RMSE is positively correlated with end-effector position RMSE.

## Experiment Conditions

Use condition IDs from `configs/experiment_defaults.json`.

## Robot Model

Record `long` or `short`, link lengths, and calibration version.

## Target Regions

Define low, medium, and high expected-deflection target regions before trials.

## Control Parameters

Record correction gain, DLS damping, maximum correction iterations, and convergence thresholds.

## Quality Exclusion Criteria

Record maximum jitter, dropped sample ratio, yaw drift, base motion, Tracker occlusion, voltage threshold, and motor temperature threshold.

## Filter And Fusion Parameters

Record `filter_params_id`, `fusion_params_id`, and exact values.

## ShapeFK Model

Default model: `arc_length_integration`.

## Statistical Plan

Record primary metrics, secondary metrics, repeated-measure test choice, multiple-comparison correction, effect size, confidence interval, and correlation/regression plan.

## Changelog

- `YYYY-MM-DD`: Initial preregistration draft.

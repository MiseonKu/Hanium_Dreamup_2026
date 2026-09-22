# Optimization And Randomization

This document fixes the rules for parameter search and randomized experiment order.
The purpose is to protect the study from tuning leakage, order effects, and
unrecorded degrees of freedom.

## Scope

- Optimization is offline only.
- Tune on `validation` split only.
- Never tune on `test` split or final reported trials.
- Record the exact objective weights, constraints, seed, candidate id, and selected
  configuration.
- Do not change the primary metric after viewing results.

## Default Objective

The default objective minimizes a weighted score:

- `position_rmse_mm`
- `shape_rmse_mm`
- `settling_time_s`
- `transient_rms_mm`
- `residual_vibration_rms_mm`
- `dropped_sample_ratio`

The tracked config is `configs/optimization_objective.json`. Keep it unchanged for
a reported experiment unless a new preregistration entry explains the change.

## Allowed Techniques

- Small grid search for filter and fusion parameters.
- Coordinate search when one parameter dominates runtime or stability.
- Successive halving for early synthetic screening, followed by a frozen full
  validation run.
- Robust ranking by median metric across repeated targets when outliers are
  dominated by tracker occlusion or dropped samples.

Do not introduce black-box optimizers until the split policy and candidate budget
are documented. Any adaptive search must save every candidate that was evaluated.

## Randomization Threats

- Order effect: one condition always follows another.
- Fatigue or heat: motors, rods, batteries, and sensor bias change over time.
- Warm-up drift: IMU yaw and gyro bias shift during initial operation.
- Carryover: aggressive motion changes the next trial's residual vibration.
- Target imbalance: hard targets are overrepresented in one condition.
- Operator intervention: manual resets happen more often in one condition.
- Tracker occlusion: missing ground truth is not random across poses.
- Timestamp gaps: a condition can look worse because acquisition was unstable.
- Calibration drift: one condition uses fresher calibration files.
- P-hacking: metrics, filters, or excluded trials are changed after inspection.

## Mitigations

- Use `research_host.cli.generate_trial_plan` before data collection.
- Use a blocked plan: every block contains all conditions once for the same
  robot model, target, and repeat id.
- Record `randomization_seed` and `plan_digest_sha256`.
- Rotate blocks to avoid adjacent identical conditions when possible.
- Run a warm-up period and discard it before trial logging.
- Lock calibration versions for a session.
- Record quality flags instead of silently dropping bad trials.
- Split data by session/date or target set before optimization.
- Keep tracker as evaluation-only ground truth.

## Commands

```powershell
$env:PYTHONPATH = ".\host"
python -m research_host.cli.generate_trial_plan `
  --conditions C1,C2,C6,C7 `
  --robot-models long,short `
  --targets T001,T002,T003 `
  --repeats 2 `
  --seed 101 `
  --output .\data\samples\trial_plan.csv `
  --metadata-output .\data\samples\trial_plan_metadata.json
```

```powershell
$env:PYTHONPATH = ".\host"
python -m research_host.cli.optimize_params `
  --input .\data\samples\analysis\optimization_candidates.csv `
  --objective-config .\configs\optimization_objective.json `
  --output .\data\samples\analysis\optimization_report.json
```

## Reporting Checklist

- Trial plan CSV and metadata are saved.
- Seed and digest match the data collection log.
- Candidate metrics have `candidate_id` and `split_id`.
- `split_id` is `validation` for optimization input.
- Final test trials were not used for tuning.
- Excluded trials have an explicit quality flag and reason.

# Validity Threats

This is the short list of factors that can damage randomization, repeatability,
or the interpretation of a condition comparison.

## Internal Validity

- Calibration drift between conditions.
- Tracker occlusion concentrated in high-deflection poses.
- Battery voltage or motor temperature changing with order.
- Sensor warm-up and yaw drift.
- Timestamp latency changing by serial port or host load.
- Manual intervention that resets only some conditions.
- Target order that gives one condition easier motion history.

## Construct Validity

- Treating acceleration double integration as `IMU-only`.
- Using tracker data as feedback while reporting it as ground truth.
- Comparing filtered and unfiltered signals with different latency.
- Mixing base-frame and world-frame positions.
- Reporting end-effector RMSE while ignoring shape RMSE.

## Statistical Validity

- Selecting metrics after seeing the results.
- Optimizing on the same trials used for final reporting.
- Dropping failed trials without a predeclared quality flag.
- Using one repeated target as if it were independent targets.
- Changing damping or fusion parameters per condition during collection.

## External Validity

- Long model and short model have different deflection envelopes.
- MPU6050 and WT901C have different onboard filtering and timing behavior.
- Desk-mounted tracker visibility may not represent real workspace use.
- Synthetic fixtures are contract tests, not performance evidence.

## Required Controls

- Freeze `configs/randomization_defaults.json`.
- Freeze `configs/optimization_objective.json`.
- Keep a preregistration note for any metric, split, or exclusion change.
- Record `quality_flags` instead of deleting suspicious rows.
- Review `docs/REVIEW_CHECKLIST.md` before merging analysis changes.

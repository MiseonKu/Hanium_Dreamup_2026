# Experiment Protocol

## Pre-Run Checklist

- Confirm robot model: `long` or `short`.
- Confirm condition ID and filter setting.
- Confirm all calibration versions are recorded.
- Confirm Tracker is evaluation-only and not used as controller feedback.
- Confirm output path is outside git-tracked `data/samples` unless this is a tiny fixture.

## Calibration Sequence

1. IMU bias calibration.
2. IMU-to-link alignment.
3. Base frame calibration.
4. D-H and encoder zero check.
5. Filter and fusion parameter freeze.
6. Latency/offset calibration.

## Trial Sequence

1. Move to initial pose.
2. Emit or record a synchronization event.
3. Start encoder, IMU, command, and Tracker logging.
4. Run the selected condition estimator.
5. Apply the common correction logic only after `P_hat_base` or `e_hat_base`.
6. Stop when termination criteria are met.
7. Record final Tracker measurement.
8. Validate quality flags before analysis.

## Quality Flags

Exclude or separately analyze trials with:

- timestamp gap
- dropped IMU sample
- delayed sample above threshold
- Tracker marker occlusion
- base motion above threshold
- yaw drift above threshold
- motor fault, over-temperature, or voltage drop

## Randomization

Condition order should be randomized before the experiment and recorded in preregistration. Do not decide trial order after seeing results.

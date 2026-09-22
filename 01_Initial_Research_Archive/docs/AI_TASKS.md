# AI Task Backlog

This backlog is scoped for future agents. Each item includes a safe acceptance
criterion so work can be delegated without re-deciding the research design.

## T1: Add Tracker CSV Adapter

Goal: Read offline Tracker CSV into `TrackerSample` rows.

Acceptance:

- Adapter lives under `research_host/adapters/`.
- Output uses `p_actual_base_mm` semantics.
- Missing marker rows are marked, not silently interpolated.
- Synthetic test covers normal row and occlusion row.

## T2: Add Parquet Optional Export

Goal: Add optional Parquet export without replacing CSV + JSON.

Acceptance:

- CSV + JSON remains the required baseline.
- Parquet dependency is optional and fails with a clear message if missing.
- Docs mention Parquet as an optional large-data export only.

## T3: Add Condition-Level Comparison Report

Goal: Compare C0-C8 trial summaries in one Markdown report.

Acceptance:

- Input is analysis CSV files.
- Output includes RMSE, residual RMS, dropped sample ratio, and quality flags.
- No statistical claims beyond the input data.

## T4: Add Live WT901C Adapter

Goal: Wrap existing `02_imu/2호기/IMU-v1/wt901c.py` behind the adapter Protocol.

Acceptance:

- No live serial code leaks into ShapeFK, conditions, logging, or analysis.
- Offline tests still pass without hardware.
- CLI has `--list-ports` and clear no-port failure text.

## T5: Add Preregistration Validator

Goal: Validate that `preregistration.md` has all mandatory sections before real
data collection.

Acceptance:

- Validator reports missing sections.
- It does not enforce exact prose.
- Test uses a temporary incomplete preregistration file.

## T6: Add IK Baseline Evaluator

Goal: Compare ShapeFK estimates against an offline DLS IK/FK baseline.

Acceptance:

- Uses `configs/ik_baseline.json`.
- Logs convergence and residual error columns.
- Does not change any condition definition.
- Test uses a synthetic target and verifies a non-converged case is flagged.

## T7: Add Optimization Candidate Generator

Goal: Build candidate config JSON files from a small parameter grid.

Acceptance:

- Candidate ids are deterministic.
- Candidate files include objective version and split policy.
- No candidate uses `test` split for selection.
- Test verifies generated candidate count and stable ids.

## T8: Add ROS 2 Description Package

Goal: Turn generated URDF/RViz assets into a standalone ROS 2 package.

Acceptance:

- Package name is `so101_sim_description`.
- Launch file can accept explicit `urdf_path` and `rviz_config` arguments.
- No ROS 2 dependency is required for `python -m pytest tests`.
- Docs include the exact ROS 2 distro tested.

## T9: Add MuJoCo Replay CLI

Goal: Replay processed CSV joint columns against the generated MJCF model.

Acceptance:

- Uses `q_encoder_rad_1..3` through `processed_row_to_joint_state`.
- Runs only when MuJoCo Python bindings are installed.
- Fails with a clear optional-dependency message otherwise.
- Does not use Tracker values as control input.

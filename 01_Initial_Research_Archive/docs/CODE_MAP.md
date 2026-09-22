# Code Map

Use this file to find the right module before editing.

## Public Workflow

- `README.md`: human entrypoint and quickstart.
- `../AGENTS.md`: repo-wide AI-agent dispatch and hardware safety rules.
- `AGENTS.md`: local AI-agent instructions for this research harness.
- `docs/AI_AGENT_GUIDE.md`: detailed agent workflow and test-selection guide.
- `docs/DATA_CONTRACT.md`: CSV + JSON contract.
- `docs/OPTIMIZATION_AND_RANDOMIZATION.md`: tuning, randomization, and split rules.
- `docs/SO101_IK_REFERENCE.md`: SO-101 IK reference algorithms and source links.
- `docs/SIMULATION_INTEGRATION.md`: ROS 2, RViz, Isaac Sim, and MuJoCo export policy.
- `docs/VALIDITY_THREATS.md`: factors that can damage randomization or interpretation.
- `configs/agent_context.json`: machine-readable project contract, task routing,
  and safe operation classes for agents.

## Python Package

- `host/research_host/types.py`
  Shared dataclasses and primitive aliases.
- `host/research_host/geometry.py`
  Long/short robot models, rod lengths, effective FK lengths, and IMU positions.
- `host/research_host/math3d.py`
  Vector and quaternion primitives. Do not duplicate these elsewhere.
- `host/research_host/shapefk.py`
  Three-IMU arc-length ShapeFK implementation.
- `host/research_host/conditions.py`
  Condition-level estimates: no correction, encoder-only, IMU-only, fusion.
- `host/research_host/sync.py`
  Timestamp correction, jitter, dropped-sample flags, scalar resampling.
- `host/research_host/logging.py`
  Trial metadata JSON, raw CSV, processed CSV, and analysis CSV contract.
- `host/research_host/analysis.py`
  RMSE and dynamic response metrics.
- `host/research_host/experimental_design.py`
  Deterministic blocked randomization plans for trials.
- `host/research_host/optimization.py`
  Offline validation-only parameter candidate scoring and selection.
- `host/research_host/quality.py`
  Quality flag counts, dropped sample ratio, timestamp gap checks.
- `host/research_host/calibration_contracts.py`
  Calibration JSON preflight validation.
- `host/research_host/reports.py`
  Descriptive condition comparison report generation.
- `host/research_host/sim/exporters.py`
  Dependency-free URDF, MJCF, RViz, ROS 2 launch, and Isaac manifest generation.
- `host/research_host/sim/replay.py`
  Optional-dependency ROS 2 and MuJoCo replay entrypoints.

## Adapters

- `host/research_host/adapters/base.py`
  Protocols for future sensors and offline readers.
- `host/research_host/adapters/mpu6050_csv.py`
  Offline MPU6050 CSV reader.
- `host/research_host/adapters/wt901c_csv.py`
  Offline WT901C CSV reader.
- `host/research_host/adapters/synthetic.py`
  Synthetic fixtures for tests and smoke runs.

## CLI

- `make_sample_trial.py`: generate the tracked synthetic fixture.
- `validate_dataset.py`: validate processed CSV + metadata JSON.
- `run_offline_trial.py`: print a metric summary from a processed trial.
- `analyze_trial.py`: write analysis CSV from a processed trial.
- `agent_doctor.py`: check agent-critical files, sample data, and importability.
- `generate_trial_plan.py`: generate blocked randomized trial plans with seed metadata.
- `optimize_params.py`: select a parameter candidate from validation metrics only.
- `export_sim_assets.py`: write simulator baseline assets under `simulation/generated`.
- `summarize_quality.py`: summarize processed CSV quality flags and thresholds.
- `validate_calibration.py`: validate real calibration JSONs before collection.
- `compare_conditions.py`: write descriptive Markdown condition reports.
- `replay_mujoco_trial.py`, `replay_ros2_joint_states.py`: optional simulator replay dry-runs and dependency-gated execution.

## Tests

- `tests/test_math3d.py`: quaternion/vector primitives.
- `tests/test_shapefk.py`: straight and bent link behavior.
- `tests/test_sync.py`: timestamp and resampling behavior.
- `tests/test_logging_and_analysis.py`: CSV/JSON contract and metrics.
- `tests/test_agent_context.py`: agent context and doctor CLI sanity checks.
- `tests/test_experimental_design.py`: trial plan balance, determinism, and digest.
- `tests/test_optimization.py`: objective scoring, constraints, and split guardrails.
- `tests/test_sim_exporters.py`: generated URDF/MJCF parseability and joint mapping.
- `tests/test_quality.py`: quality flags and threshold enforcement.
- `tests/test_calibration_contracts.py`: calibration preflight checks.
- `tests/test_reports.py`: descriptive report generation.
- `tests/test_sim_replay.py`: optional dependency replay behavior.
- `tests/test_validate_dataset_cli.py`: CLI-level dataset validation behavior.

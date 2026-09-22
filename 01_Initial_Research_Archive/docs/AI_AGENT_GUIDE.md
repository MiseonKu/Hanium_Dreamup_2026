# AI Agent Guide

This guide is for coding agents that need to modify the research baseline
without prior project context. It complements `AGENTS.md`: the local agent file
defines hard rules, while this guide explains how to plan and implement changes.

## Mental Model

The project is an offline-first research scaffold. It is not a realtime robot
controller. Its job is to make future hardware and analysis work consistent.

The core pipeline is:

```text
CSV or future sensor stream
  -> normalized adapter sample
  -> timestamp correction
  -> attitude or pose input
  -> ShapeFK
  -> condition-level P_hat_*_base_mm
  -> CSV + JSON logging
  -> analysis metrics
  -> quality, optimization, reports, and simulation exports
```

Two research rules dominate every implementation decision:

- Tracker is ground truth only, never controller feedback.
- IMU-only means three-IMU-per-link ShapeFK, not acceleration double
  integration.

## Source Of Truth

- Research design: root `필수 참고 사항.md` and `연구방법론+가이드라인.pdf`.
- Local agent rules: `AGENTS.md`.
- Machine-readable agent context: `configs/agent_context.json`.
- Code contracts: `docs/DATA_CONTRACT.md`.
- Architecture: `docs/ARCHITECTURE.md`.
- Implementation map: `docs/CODE_MAP.md`.
- Decisions already made: `docs/DECISION_LOG.md`.
- Randomization and optimization: `docs/OPTIMIZATION_AND_RANDOMIZATION.md`.
- IK references: `docs/SO101_IK_REFERENCE.md`.
- Simulation exports: `docs/SIMULATION_INTEGRATION.md`.
- Review expectations: `docs/REVIEW_CHECKLIST.md`.

## Change Classification

Classify the task before editing:

- Docs-only: update docs and run `agent_doctor`.
- Contract: update docs, constants, validators, fixtures, and tests together.
- Math: keep quaternion/vector primitives in `math3d.py`; do not duplicate math
  helpers elsewhere.
- ShapeFK: keep three-IMU link reconstruction semantics explicit and covered by
  synthetic tests.
- Adapter: put file readers and future live IO in `research_host/adapters/`.
- Logging: preserve CSV + JSON compatibility unless the user explicitly accepts
  a contract change.
- Analysis/reporting: keep descriptive reports separate from inferential
  statistics.
- Randomization/optimization: preserve validation-only tuning and recorded
  seeds.
- Simulation: keep exporters dependency-free and runtime replay optional.

## Implementation Rules

- Use `docs/CODE_MAP.md` to find the owner module before editing.
- Keep realtime hardware IO out of ShapeFK, analysis, logging, optimization,
  and reports.
- Keep processed CSV field names synchronized with
  `logging.PROCESSED_FIELDNAMES`.
- Keep metadata keys synchronized with validators and `docs/DATA_CONTRACT.md`.
- Keep frame and unit suffixes in public variable names, such as
  `P_hat_imu_base_mm`, `P_actual_base_mm`, and `q_imu_world`.
- Do not compare IMU-only and encoder-only using different downstream
  correction logic.
- Do not use final test trials for parameter search.
- Do not change randomized trial order after seeing results.
- Do not make simulator packages, serial devices, or vendor SDKs required for
  normal unit tests.

## Test Selection

Run the narrowest useful check while developing, then the full set before
finishing a behavior change.

Always useful:

```powershell
$env:PYTHONPATH = ".\host"
python -m research_host.cli.agent_doctor
python -m pytest tests
```

Contract and logging changes:

```powershell
python -m research_host.cli.make_sample_trial --output .\data\samples
python -m research_host.cli.validate_dataset `
  --metadata .\data\samples\raw\trial_0001_long_C2_imu_array_only_T001_rep01_2026-05-01_metadata.json `
  --processed .\data\samples\processed\trial_0001_long_C2_imu_array_only_T001_rep01_2026-05-01_processed.csv
```

Simulation changes:

```powershell
python -m research_host.cli.export_sim_assets --robot-model long --output .\simulation\generated
python -m research_host.cli.replay_mujoco_trial `
  --processed .\data\samples\processed\trial_0001_long_C2_imu_array_only_T001_rep01_2026-05-01_processed.csv `
  --model .\simulation\generated\mjcf\so101_long.xml `
  --dry-run
```

Randomization and optimization changes:

```powershell
python -m research_host.cli.generate_trial_plan `
  --conditions C1,C2,C6,C7 `
  --robot-models long,short `
  --targets T001,T002 `
  --repeats 1 `
  --seed 101 `
  --output .\data\samples\trial_plan.csv `
  --metadata-output .\data\samples\trial_plan_metadata.json
python -m research_host.cli.optimize_params `
  --input .\data\samples\analysis\optimization_candidates.csv `
  --objective-config .\configs\optimization_objective.json `
  --output .\data\samples\analysis\optimization_report.json
```

## Common Pitfalls

- Using names like `P_hat` without `_base_mm`.
- Treating quaternion columns as Euler angles.
- Writing realtime serial code directly inside ShapeFK or analysis modules.
- Comparing conditions with different downstream correction logic.
- Tuning parameters on the final test split.
- Changing randomization order after seeing condition performance.
- Adding IK outputs without logging convergence and residual error.
- Adding simulator-specific logic inside ShapeFK, analysis, or logging modules.
- Requiring ROS 2, Isaac Sim, MuJoCo, or RViz for normal unit tests.
- Running real data collection before `validate_calibration` and
  `validate_dataset` pass.
- Presenting generated comparison reports as statistical proof.
- Adding processed CSV fields without updating `logging.PROCESSED_FIELDNAMES`,
  `docs/DATA_CONTRACT.md`, validators, and tests.

## Expected Response Shape

When reporting work, summarize:

- Docs changed.
- Code changed.
- Machine-readable context changed.
- Data contract changed or unchanged.
- Randomization/optimization policy changed or unchanged.
- Simulation assets or contracts changed or unchanged.
- Commands run and result.
- Any known limitation.

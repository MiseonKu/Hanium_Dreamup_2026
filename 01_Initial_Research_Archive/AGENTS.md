# AGENTS.md

This file is the local instruction sheet for AI coding agents working inside
`01_Initial_Research_Archive`.

This folder is an archive of the project's initial IMU position-estimation
research plan and offline prototype. Treat it as reference material, not as
the current data-collection, ML-training, or robot-control workspace. Current
experiment data and code live in the numbered sibling workspaces.

## Start Here

Read these files in order before changing code:

1. `../AGENTS.md`
2. `README.md`
3. `docs/AI_AGENT_GUIDE.md`
4. `docs/CODE_MAP.md`
5. `docs/DATA_CONTRACT.md`
6. `docs/DECISION_LOG.md`
7. `docs/OPTIMIZATION_AND_RANDOMIZATION.md`
8. `docs/SO101_IK_REFERENCE.md` if touching IK, target poses, trajectories,
   robot geometry, or baseline reachability.
9. `docs/SIMULATION_INTEGRATION.md` if touching ROS 2, RViz, Isaac Sim,
   MuJoCo, URDF, MJCF, manifests, or simulator replay.
10. `configs/agent_context.json` when updating harness routing or agent-facing
    contracts.

For experiment intent, cross-check the repository-root source documents:

- `../필수 참고 사항.md`
- `../연구방법론+가이드라인.pdf`

## Mental Model

This is an offline-first research harness. It exists to freeze contracts,
generate synthetic fixtures, validate logs, and support future hardware
integration without making unit tests depend on hardware.

The core data path is:

```text
CSV or future sensor stream
  -> adapter sample
  -> timestamp correction
  -> attitude or pose input
  -> ShapeFK
  -> condition-level P_hat_*_base_mm
  -> CSV + JSON logging
  -> analysis, quality, optimization, reports, and simulator export
```

## Non-Negotiable Rules

- Tracker is ground truth only, never controller feedback.
- Do not use Tracker data as controller feedback. Tracker is ground truth only.
- Do not redefine IMU-only as acceleration double integration. IMU-only means
  three-IMU-per-link ShapeFK reconstruction.
- Do not commit real trial data under `data/raw`, `data/processed`, or
  `data/analysis`.
- Keep realtime hardware IO behind `research_host/adapters/`.
- Keep position, attitude, error, and deflection names frame-suffixed and
  unit-suffixed when units are part of the contract.
- Keep the CSV + JSON contract backward-compatible unless docs, constants, and
  tests are updated in the same change.
- Do not tune parameters on final test trials.
- Do not change randomized trial order after seeing results.
- Do not make ROS, Isaac Sim, MuJoCo, RViz, serial devices, cameras, or Tracker
  SDKs required unit-test dependencies.
- Do not present generated condition reports as inferential statistics.

## Task Routing

- Docs-only work: update the nearest doc and run `agent_doctor`.
- CSV or metadata contract work: update `docs/DATA_CONTRACT.md`,
  `host/research_host/logging.py`, validators, sample fixtures, and tests in
  the same change.
- ShapeFK math work: start in `host/research_host/shapefk.py`,
  `math3d.py`, and `geometry.py`; cover with synthetic tests.
- Condition comparison work: start in `conditions.py`, `analysis.py`, and
  `reports.py`; preserve fair downstream correction logic across conditions.
- Adapter work: place readers or future live IO under `research_host/adapters/`;
  keep the rest of the pipeline offline-testable.
- Randomization or optimization work: update
  `docs/OPTIMIZATION_AND_RANDOMIZATION.md`, configs, and deterministic tests.
- Simulation work: keep exporters dependency-free; keep runtime replay gated
  behind optional imports and dry-run behavior.
- Hardware collection work: do not implement direct live control in core math,
  logging, analysis, or report modules.

## Public Contracts

Treat these as public interfaces:

- `docs/DATA_CONTRACT.md`
- `host/research_host/logging.py`
- `host/research_host/types.py`
- `host/research_host/conditions.py`
- `host/research_host/experimental_design.py`
- `host/research_host/optimization.py`
- `host/research_host/quality.py`
- `host/research_host/calibration_contracts.py`
- `host/research_host/reports.py`
- `configs/*.json`
- `configs/agent_context.json`
- `simulation/generated/*_manifest.json`

When a public contract changes, update docs, implementation, sample data if
needed, and tests together.

## Data Policy

- `data/samples/` contains small tracked synthetic fixtures.
- `data/raw/`, `data/processed/`, and `data/analysis/` are for real experiment
  outputs and are ignored except placeholder `.gitignore` files.
- Do not add real trial captures as samples unless the user explicitly asks for
  a small anonymized fixture and approves the filename and contents.
- Do not change the randomized plan after inspecting condition performance.

## Useful Commands

PowerShell from this folder:

```powershell
$env:PYTHONPATH = ".\host"
python -m research_host.cli.agent_doctor
python -m research_host.cli.make_sample_trial --output .\data\samples
python -m research_host.cli.validate_dataset `
  --metadata .\data\samples\raw\trial_0001_long_C2_imu_array_only_T001_rep01_2026-05-01_metadata.json `
  --processed .\data\samples\processed\trial_0001_long_C2_imu_array_only_T001_rep01_2026-05-01_processed.csv
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
python -m research_host.cli.export_sim_assets `
  --robot-model long `
  --output .\simulation\generated
python -m research_host.cli.summarize_quality `
  --processed .\data\samples\processed\trial_0001_long_C2_imu_array_only_T001_rep01_2026-05-01_processed.csv
python -m research_host.cli.replay_mujoco_trial `
  --processed .\data\samples\processed\trial_0001_long_C2_imu_array_only_T001_rep01_2026-05-01_processed.csv `
  --model .\simulation\generated\mjcf\so101_long.xml `
  --dry-run
python -m pytest tests
```

## Safe First Tasks For Agents

- Improve docs without changing data contracts.
- Add tests around existing math, sync, analysis, reporting, or quality
  behavior.
- Add an offline adapter for an existing CSV format.
- Add a descriptive analysis metric with a synthetic test.
- Add deterministic randomization or optimization utilities.
- Add simulator export fields while keeping generated assets dependency-free.
- Add validation/report CLI behavior with synthetic fixtures.

## Ask Before Doing

- Changing condition IDs.
- Changing required CSV or metadata fields.
- Moving generated data policy from CSV + JSON to Parquet-only.
- Connecting live hardware control.
- Reusing Tracker values inside controller logic.
- Changing optimization objective weights for a reported experiment.
- Changing IK solver policy from offline analysis to realtime control.
- Requiring a simulator install for offline tests or agent doctor.
- Treating generated condition reports as statistical proof.
- Adding real experiment data to tracked fixtures.

## Reporting Shape

When reporting work, summarize:

- Docs changed.
- Code changed.
- Machine-readable context changed.
- Data contract changed or unchanged.
- Randomization/optimization policy changed or unchanged.
- Simulation assets or contracts changed or unchanged.
- Commands run and result.
- Any known limitation.

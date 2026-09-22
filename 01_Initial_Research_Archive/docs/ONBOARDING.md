# Onboarding

## One Line

This project tests whether a three-IMU array on each flexible carbon-rod link can reconstruct link shape and estimate end-effector position better than rigid encoder-only FK.

## Read Order

1. `README.md`
2. `../AGENTS.md`, `AGENTS.md`, and `docs/AI_AGENT_GUIDE.md` if you are an AI
   coding agent or harness automation.
3. `configs/agent_context.json` if you need machine-readable routing.
4. `docs/CODE_MAP.md`
5. `docs/ARCHITECTURE.md`
6. `docs/DATA_CONTRACT.md`
7. `docs/OPTIMIZATION_AND_RANDOMIZATION.md`
8. `docs/EXPERIMENT_PROTOCOL.md`
9. `docs/CALIBRATION_GUIDE.md`
10. `docs/GLOSSARY.md`

## Local Setup

PowerShell:

```powershell
cd "01_Initial_Research_Archive"
$env:PYTHONPATH = ".\host"
python -m research_host.cli.agent_doctor
python -m pytest tests
```

The baseline package uses only the Python standard library. Hardware adapters may later require `pyserial`, `numpy`, or vendor SDKs.

From the repository root, documentation and harness routing can also be checked
with:

```powershell
python tools\agent_doctor.py --root .
```

## Smoke Test

```powershell
$env:PYTHONPATH = ".\host"
python -m research_host.cli.make_sample_trial --output .\data\samples
python -m research_host.cli.validate_dataset `
  --metadata .\data\samples\raw\trial_0001_long_C2_imu_array_only_T001_rep01_2026-05-01_metadata.json `
  --processed .\data\samples\processed\trial_0001_long_C2_imu_array_only_T001_rep01_2026-05-01_processed.csv
```

If this passes, your environment can run the offline contract.

## Adding Work

- New sensor input: add a reader under `research_host/adapters/`.
- New ShapeFK model: add it behind `reconstruct_link_shape(..., model=...)`.
- New metric: add it to `analysis.py` and cover it with a synthetic test.
- New data column: update `docs/DATA_CONTRACT.md`, `logging.PROCESSED_FIELDNAMES`, and tests in the same change.
- New agent-facing workflow: update `AGENTS.md`, `docs/AI_AGENT_GUIDE.md`,
  `configs/agent_context.json`, and the relevant doctor/test checks together.

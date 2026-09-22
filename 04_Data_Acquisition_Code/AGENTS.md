# AGENTS

This workspace contains the completed four-WT901C, encoder, and Vive data
acquisition code and the raw-to-Parquet conversion tools.

## Start Here

Read `README.md`, `HANDOFF.md`, `experiment_scripts/README.md`,
`docs/RAW_DATA_POLICY.md`, and `docs/FIRMWARE_MOTION_HAZARDS.md` before making
changes. The real captures came from `experiment_scripts/`, not from
`SessionController`.

## Ownership Boundary

- The collected raw and derived archives are stored under
  `../03_Dataset_Collection/imu_added_experiment/`.
- The version-pinned static learning source is the
  `derived/parquet/real_motion_demo/trials.parquet` member of
  `../03_Dataset_Collection/imu_added_experiment/derived_dataset_archive.zip`.
- `../05_ML/` owns downstream data cleaning, split assignment, model training,
  validation, and packaging.
- `timeseries.parquet` is explicitly outside `alphaf.ml_preparation.v1`
  because its v1 alignment/provenance contract is not evaluation-safe.

## Non-Negotiable Rules

- Never run anything under `experiment_scripts/` or connect serial/SteamVR
  sources unless the user explicitly requests a hardware operation. Those
  scripts move the arm.
- Do not bypass `safe_engage()`, calibration, torque, range, thermal, or fault
  safeguards. Never substitute `INIT` for the documented raw-tick sequence.
- Vive is immutable ground-truth/label data, never controller feedback.
- Preserve invalid records and their flags in raw data. Do not rewrite or
  forward-fill raw streams.
- Do not call this four-IMU endpoint dataset the formal ShapeFK `IMU-only`
  condition; that definition belongs to the initial research archive.
- Do not change reported segment order, target order, raw records, or required
  acquisition fields after observing results without an explicit new contract.

## Safe Offline Checks

```powershell
python -m pytest tests -q
python tools/run_mock_session.py --duration-s 5
```

Tests that require ignored real sessions must skip cleanly when those sessions
are unavailable. No check may require hardware, SteamVR, or motor motion.

# AGENTS

This workspace owns offline preparation of the collected four-WT901C,
encoder, and Vive endpoint-regression dataset.

## Read Order

- For cleaning, split, augmentation, or preparation packaging, read
  `docs/DATA_CONTRACT.md`, `docs/DATA_PACKAGE_STRUCTURE.md`, and
  `docs/AUGMENTATION_EVIDENCE.md`.
- For model fitting, evaluation, packaging, or inference, also read
  `docs/ENDPOINT_REGRESSION_CONTRACT.md`,
  `docs/ENDPOINT_REGRESSION_METHODS.md`, and
  `docs/ENDPOINT_REGRESSION_RESULTS_V1.md`.
- Treat research citations as support for a method family. Frozen repository
  configuration and validators define the exact local implementation.

## Scope

- Read the tracked `trials.parquet` member from the derived archive in
  `../03_Dataset_Collection/`.
- Audit, clean, split, augment, validate, and package static trial-level data.
- Fit and evaluate the exploratory endpoint-regression experiment from the
  immutable `alphaf.ml_preparation.v1` package.
- Treat Vive as a label/evaluation signal only.
- Call the IMU view `four_imu_endpoint`; it is not the formal three-IMU-per-link
  ShapeFK `IMU-only` condition owned by the initial research archive.

## Non-Negotiable Rules

- Never send motor, torque, calibration, firmware-upload, Tracker SDK, or live
  serial commands from this workspace.
- Never use `timeseries.parquet` as an input to the v1 contract.
- Never fit preprocessing or generate synthetic rows using validation/test
  samples.
- Never place Vive or planned target coordinates in a primary feature view.
- Never apply all-data Kabsch alignment. The v1 label remains raw Vive world
  position in millimetres.
- Never describe `alphaf.endpoint_regression.experiment.v1` as confirmatory,
  independently tested, online, control-ready, robot-base/TCP calibrated,
  ShapeFK, or `IMU-only`.
- Never select planned-coordinate, uncalibrated-magnetometer, replay, or Dummy
  diagnostic conditions as the final inference model.
- For endpoint regression, fit every scaler, target transform, hyperparameter,
  augmentation parent search, and C-Mixup bandwidth inside the active real
  train context. Validation and test rows must remain measured real rows.
- Do not overwrite the legacy before/after/clean augmentation archives.
- Keep generated runs under `artifacts/`; only an explicitly versioned release
  package may be copied to the tracked 06 LFS archive directory.

## Required Checks

```powershell
python -m pytest tests -q
python -m imu_vive_learning.cli validate --run-dir artifacts/static_endpoint_v1
python -m imu_vive_learning.cli validate-experiment --run-dir artifacts/endpoint_regression_v1
```

Run the root agent doctor after changing workspace routing or public contract
documentation.

# Changelog

## 2026-05-19

### Added

- Added the pre-IMU `forward_trimmed_hemisphere` target-generation workflow under the 02 flexible-link FK/IK workspace package.
- Added `config/pre_imu_forward_hemisphere_workspace.json` for the latest `TPU95A_40x12_wall4_topbottom4_infill15_honeycomb` link preset.
- Added P1/P2 rigid-FK target generation:
  - `P1_ONE_LINK_RIGID_TARGET_VIVE_LABEL`
  - `P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL`
- Added `workspace_points/pre_imu_targets.py` with lightweight standard-library-only target generation, forward workspace filtering, bin classification, bin-balanced selection, CSV output, metadata JSON output, and balanced schedule helpers.
- Added CLI scripts:
  - `scripts/generate_pre_imu_targets.py`
  - `scripts/generate_pre_imu_trial_schedule.py`
- Added `docs/PRE_IMU_FORWARD_HEMISPHERE_PROTOCOL.md` documenting the pre-IMU protocol, frame convention, label policy, limitations, and commands.
- Added small synthetic sample outputs under `data/samples/`.
- Added tests for rigid FK, forward-hemisphere filtering, deterministic generation, and CSV field coverage.

### Changed

- Updated `README.md` to describe the new pre-IMU workflow, latest TPU95A 40x12 preset, commands, and document locations.
- Corrected the default material preset in `config/robot_config.yaml` and `README.md` to 15% honeycomb infill with 4 wall loops and 4 top/bottom layers.
- Added `top_bottom_wall_count` to the full-workspace target metadata so regenerated target CSVs carry the latest print preset traceability.
- Kept existing full-workspace FK/IK, workspace candidate, target sampling, randomization, trajectory, and trial schedule workflow intact.

### Notes

- Vive Tracker measured position remains label/ground-truth only and is not a runtime model input.
- IMU columns are not required for this pre-IMU workflow.
- The forward hemisphere limits are manual pilot/main workspace limits, not a calibrated Vive ROI.

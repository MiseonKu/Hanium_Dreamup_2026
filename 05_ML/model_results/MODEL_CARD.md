# Model Card — AlphaF Endpoint Regression Experiment v1

## Intended use

Offline exploratory prediction of raw `vive_world` XYZ millimetres from one
static trial-summary feature row that exactly matches the packaged ordered
allow-list.

## Not intended for

- realtime or closed-loop control;
- robot-base/TCP coordinates;
- formal ShapeFK or the repository's `IMU-only` condition;
- dynamic motion, safety-critical decisions, or a new hardware session;
- use of planned, target-ID, Vive, Tracker, QC, timestamp, or provenance fields
  as primary input.

## Selected system

- Cell: `mlp__fusion__real_only`
- Parameters: `{"alpha": 0.001, "hidden_layer_sizes": [64, 64]}`
- Feature count: `43`
- Ensemble members: `3`
- Model SHA-256: `14ddf90e009f112e85600f9346281dd91b125da10f37ca4b211c42c03ae45401`
- Contract fingerprint: `2c3445fb5e48713642413d1b4f011e83cff66bc1817cd920cb67123edb1116a5`
- Output frame/unit: `vive_world` / `mm`

## Evidence status

Exploratory nested group-CV on the same corpus used for feasibility checks.
The final all-data model has no independent test score. Segment and unseen-
target OOF estimates, every fold effect, strict-QC sensitivity, and paired
multiway-bootstrap intervals are packaged for audit. Independent acquisition-
session validation is required before any confirmatory claim.

## Input and output

`predict-endpoint` accepts only the exact ordered feature columns below, plus
an optional leading `row_id`:

```text
encoder_delta_rad_j0
encoder_delta_rad_j1
encoder_delta_rad_j2
imu0_base_euler_x_rad_sin
imu0_base_euler_x_rad_cos
imu0_base_euler_y_rad_sin
imu0_base_euler_y_rad_cos
imu0_base_euler_z_rad_sin
imu0_base_euler_z_rad_cos
imu1_link_root_euler_x_rad_sin
imu1_link_root_euler_x_rad_cos
imu1_link_root_euler_y_rad_sin
imu1_link_root_euler_y_rad_cos
imu1_link_root_euler_z_rad_sin
imu1_link_root_euler_z_rad_cos
imu2_link_mid_euler_x_rad_sin
imu2_link_mid_euler_x_rad_cos
imu2_link_mid_euler_y_rad_sin
imu2_link_mid_euler_y_rad_cos
imu2_link_mid_euler_z_rad_sin
imu2_link_mid_euler_z_rad_cos
imu3_tip_euler_x_rad_sin
imu3_tip_euler_x_rad_cos
imu3_tip_euler_y_rad_sin
imu3_tip_euler_y_rad_cos
imu3_tip_euler_z_rad_sin
imu3_tip_euler_z_rad_cos
imu0_base_accel_x_mps2
imu0_base_accel_y_mps2
imu0_base_accel_z_mps2
imu1_link_root_accel_x_mps2
imu1_link_root_accel_y_mps2
imu1_link_root_accel_z_mps2
imu2_link_mid_accel_x_mps2
imu2_link_mid_accel_y_mps2
imu2_link_mid_accel_z_mps2
imu3_tip_accel_x_mps2
imu3_tip_accel_y_mps2
imu3_tip_accel_z_mps2
imu0_base_gyro_rms_rad_s
imu1_link_root_gyro_rms_rad_s
imu2_link_mid_gyro_rms_rad_s
imu3_tip_gyro_rms_rad_s
```

Outputs are `pred_vive_world_x_mm`, `pred_vive_world_y_mm`, and
`pred_vive_world_z_mm`, together with model/package identity and frame/unit.
The package embeds a Python/joblib object. Load only the trusted Git LFS
release after checksum validation; never pass an untrusted ZIP to this CLI.

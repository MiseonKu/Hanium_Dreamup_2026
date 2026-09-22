# Calibration Guide

Calibration files live in `calibration/`. Templates live in `calibration/templates/`.

## Required Calibration Types

- `calib_imu_bias`: gyro and accelerometer bias for each IMU.
- `calib_imu_alignment`: rotation from IMU local frame to link frame.
- `calib_handeye`: optional transform between IMU frame and Tracker marker frame.
- `calib_baseframe`: world frame to robot base frame transform.
- `calib_dh_params`: SO-101-style D-H parameters and effective link lengths.
- `calib_filter_params`: Kalman and Madgwick parameters.
- `calib_fusion_params`: encoder/IMU covariance or residual fusion gain.
- `calib_latency`: sensor latency and stream offsets.

## Recalibration Triggers

- IMU detached or remounted: redo `imu_bias`, `imu_alignment`, and `handeye`.
- Carbon rod replaced: redo `imu_alignment`, `dh_params`, and modal check.
- Robot model changed between long/short: redo `dh_params`, base check, and IMU positions.
- Motor or joint reassembled: redo encoder zero and backlash check.
- Tracker moved: redo `baseframe`.
- Port, cable, sample rate, or control period changed: redo `latency`.
- New experiment day or large temperature change: at least check `imu_bias` and yaw drift.

Every trial metadata file must record the calibration versions used.

## Preflight Validation

Before collecting real data, place measured calibration files directly under
`calibration/` and run:

```powershell
$env:PYTHONPATH = ".\host"
python -m research_host.cli.validate_calibration `
  --calibration-dir .\calibration `
  --templates .\calibration\templates
```

The command fails when required calibration types are missing, when template
dates are still present, or when core matrix/vector fields are malformed.

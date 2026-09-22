# Architecture

## Pipeline

```text
sensor adapters
  -> timestamp sync
  -> attitude estimation
  -> ShapeFK
  -> condition estimator
  -> CSV/JSON logger
  -> analysis
```

## Module Roles

- `adapters/`: convert raw files or future serial streams into normalized sample dataclasses.
- `sync.py`: estimates sample time, applies latency and offset correction, resamples scalar series.
- `math3d.py`: owns quaternion/vector math. Do not duplicate quaternion code elsewhere.
- `shapefk.py`: reconstructs one flexible link from start/mid/end IMU attitudes.
- `conditions.py`: defines condition-level estimates such as encoder-only, IMU-only, and fusion.
- `logging.py`: owns the CSV + JSON file contract.
- `analysis.py`: computes RMSE, settling time, transient RMS, residual vibration RMS, and quality ratios.

## Hardware Boundary

Realtime IO is intentionally outside the baseline. ST3215, MPU6050, WT901C, and Tracker code should be connected through adapters so the offline tests remain usable without hardware.

## Frame Rule

All position and attitude names must include their frame. Examples:

- `P_hat_imu_base_mm`
- `P_actual_base_mm`
- `q_imu_world`
- `delta_i_link_mm`

Frame-free names are allowed only in local math helper internals.

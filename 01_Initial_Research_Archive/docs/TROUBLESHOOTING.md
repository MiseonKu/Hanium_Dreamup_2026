# Troubleshooting

## No Serial Port

- Check USB cable and device manager.
- Confirm the sensor has power.
- Pass the port explicitly in future realtime adapters.

## IMU Yaw Drift

- Re-run stationary bias calibration.
- Confirm IMU mounting is rigid.
- Check temperature change.
- Mark `yaw_drift_flag` if the trial exceeds the preregistered threshold.

## Timestamp Gap

- Check UART throughput.
- Reduce logging load.
- Inspect `latency_*_s` and `quality_flags`.
- Re-run latency calibration after sample-rate changes.

## Tracker Occlusion

- Move marker or camera.
- Mark `tracker_occlusion_flag`.
- Do not fill missing Tracker samples without marking them.

## ST3215 Response Failure

- Run bus scan in the SO-101 control firmware.
- Check external motor PSU.
- Check daisy-chain cable order.
- Do not start research trials until encoder/motor status is stable.

## Coordinate Sign Looks Wrong

- Check `calib_imu_alignment`.
- Verify IMU local x-axis is the rod tangent direction.
- Verify frame suffixes in the CSV column names.
- Use a one-link static pose before running the full manipulator.

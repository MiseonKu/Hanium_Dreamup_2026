# Glossary

- `IMU-only`: Link shape and end-effector position are estimated from the three-IMU arrays without encoder feedback.
- `ShapeFK`: Forward kinematics over a reconstructed flexible link shape, not rigid-link FK.
- `arc-length integration`: Integrating IMU-derived tangent directions along the physical rod length.
- `P_hat_encoder_base_mm`: Encoder-only estimated end-effector position in base frame.
- `P_hat_imu_base_mm`: IMU-array estimated end-effector position in base frame.
- `P_hat_fusion_base_mm`: Encoder + IMU fusion estimated end-effector position in base frame.
- `P_actual_base_mm`: Tracker-measured ground truth end-effector position in base frame.
- `delta_i_link_mm`: Local deflection residual for link `i`.
- `quality_flags`: Pipe-separated flags describing data quality problems. `ok` means no active flags.
- `long model`: A/B/C rod lengths 233/215/106 mm, effective FK lengths 299.5/299.5/150.3 mm.
- `short model`: A/B/C rod lengths 183/165/56 mm, effective FK lengths 249.5/249.5/100.3 mm.

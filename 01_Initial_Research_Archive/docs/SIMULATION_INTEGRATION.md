# Simulation Integration

This document prepares the research scaffold for Isaac Sim, ROS 2, RViz, and
MuJoCo without adding heavyweight simulator dependencies to the test suite.

## Official Reference Points

- NVIDIA Isaac Sim URDF importer documentation:
  `https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_urdf.html`
- ROS 2 `robot_state_publisher` URDF workflow:
  `https://docs.ros.org/en/rolling/Tutorials/Intermediate/URDF/Using-URDF-with-Robot-State-Publisher-py.html`
- MuJoCo XML / MJCF model reference:
  `https://mujoco.readthedocs.io/en/stable/XMLreference.html`

These tools should consume generated assets, not become mandatory dependencies
for the offline research package.

## Contract

- Research code stores positions in millimeters.
- Simulator assets store positions in meters.
- Angles stay in radians.
- Base frame is `base_link`.
- End-effector frame is `tool0`.
- Joint order is `joint_1`, `joint_2`, `joint_3`.
- Processed CSV joint columns map to `q_encoder_rad_1`, `q_encoder_rad_2`,
  `q_encoder_rad_3`.

The machine-readable version is `configs/simulation_targets.json`.

## Exported Assets

Run:

```powershell
$env:PYTHONPATH = ".\host"
python -m research_host.cli.export_sim_assets `
  --robot-model long `
  --output .\simulation\generated
```

Generated files:

- `urdf/so101_long.urdf`: ROS 2, RViz, and Isaac Sim baseline description.
- `mjcf/so101_long.xml`: MuJoCo baseline model.
- `rviz/so101_long.rviz`: RViz display config.
- `ros2/so101_long_display.launch.py`: ROS 2 display launch file.
- `isaac/so101_long_import.json`: Isaac import checklist and joint manifest.
- `so101_long_manifest.json`: export metadata and simulator-neutral model spec.

## Isaac Sim

Use the generated URDF as the import source. Enable the URDF importer and ROS 2
bridge extensions in Isaac Sim, then import `simulation/generated/urdf/*.urdf`.
Treat the generated mass and inertia values as placeholders.

## ROS 2 And RViz

The generated launch file assumes a future ROS 2 package named
`so101_sim_description` with `urdf/` and `rviz/` folders. Until that package is
created, pass explicit `urdf_path` and `rviz_config` launch arguments or use the
files manually with `robot_state_publisher`.

## MuJoCo

Load `simulation/generated/mjcf/so101_long.xml` or the short model equivalent.
The MJCF file contains hinge joints, capsule link geometry, IMU sites, and a
`tool0` site. Use it for kinematic visualization and algorithm replay first;
physics validation requires identified mass, inertia, damping, friction, and
actuator parameters.

Dry-run replay without MuJoCo installed:

```powershell
$env:PYTHONPATH = ".\host"
python -m research_host.cli.replay_mujoco_trial `
  --processed .\data\samples\processed\trial_0001_long_C2_imu_array_only_T001_rep01_2026-05-01_processed.csv `
  --model .\simulation\generated\mjcf\so101_long.xml `
  --dry-run
```

ROS 2 JointState replay uses the same `q_encoder_rad_1..3` mapping. Run
`replay_ros2_joint_states --dry-run` first; without `--dry-run`, the CLI requires
`rclpy` and returns a clear optional-dependency error if ROS 2 is unavailable.

## Non-Goals

- No simulator is required to run unit tests.
- No realtime control loop is introduced here.
- No generated simulator asset changes the CSV + JSON data contract.
- No simulator feedback replaces Tracker ground truth.

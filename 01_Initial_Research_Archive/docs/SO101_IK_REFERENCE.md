# SO-101 IK Reference

This folder does not implement realtime SO-101 inverse kinematics control. It
records the baseline algorithms and decision points that future work should use.

## Web-Checked References

- Hugging Face LeRobot action representations:
  `https://huggingface.co/docs/lerobot/action_representations`
- LeRobot `RobotKinematics` source:
  `https://github.com/huggingface/lerobot/blob/main/src/lerobot/model/kinematics.py`
- LeRobot SO follower kinematic processor source:
  `https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/robot_kinematic_processor.py`

As of 2026-05-01, LeRobot's public code path uses a URDF-backed
`RobotKinematics` wrapper around the Placo solver. The SO follower processor
contains `inverse_kinematics_ee_to_joints`, which converts desired
end-effector-space actions back to joint targets through that kinematics object.

## Local Repository Baselines

- `01_control/ik_시즌2_업그레이드/2호기/ik/6dof/numerical_ik_6dof.cpp`
  uses forward kinematics, a geometric Jacobian, damped least-squares normal
  equations, reference regularization, limit penalties, and waypoint planning.
- `01_control/ik_시즌2_업그레이드/1호기/ik/4dof/analytic_ik_4dof.cpp`
  provides the older 4-DOF path with joint limit checks and a damped numerical
  refinement path.

## Recommended Research Baseline

Use damped least-squares numerical IK as the offline baseline unless a later
hardware integration explicitly adopts LeRobot's URDF/Placo path.

Baseline inputs:

- Target pose in base frame: `P_target_base_mm`, `q_target_base`.
- Seed joints: current encoder estimate or nearest valid previous solution.
- Joint limits: pulled from the robot configuration used by the trial.
- Damping: fixed in config, not tuned per target after seeing results.
- Stop criteria: max iterations, position tolerance, orientation tolerance.

Baseline outputs:

- `q_ik_rad_*`
- `ik_converged`
- `ik_iteration_count`
- `ik_position_error_mm`
- `ik_orientation_error_rad`
- `ik_limit_flag`
- `ik_singularity_flag`

## What To Log

- Initial seed and final solution.
- Solver parameters and config version.
- Convergence status and residual error.
- Any joint clipping or singularity damping event.
- Whether the solution was used for analysis only or hardware command.

## Research Rule

IK may be used to generate candidate encoder trajectories or compare ShapeFK
against a model-based baseline. It must not replace the IMU-array ShapeFK
condition definition. If IK is ever used in realtime control, add a new condition
id and a hardware safety protocol before collecting data.

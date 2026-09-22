# Pre-IMU Forward Hemisphere Protocol

This protocol defines an offline target-generation workflow for the SO-101
flexible-link capstone project before IMUs are available.

## Why This Mode Exists

- IMU supply is delayed, so this phase cannot rely on IMU time-series input.
- Vive Tracker stable range, line of sight, and occlusion constraints make the
  old full 360-degree workspace unsuitable for the current desk-clamped setup.
- The pre-IMU experiment should validate a manual forward valid range first,
  then expand only after Tracker and hardware limits are measured.

This mode is intentionally separate from existing full-workspace and legacy
target-generation logic. It adds `forward_trimmed_hemisphere`; it does not
remove or redefine legacy workspace algorithms.

## Link Material Preset

Preset id:

```text
TPU95A_40x12_wall4_topbottom4_infill15_honeycomb
```

Physical link spec:

- Material: TPU 95A
- Link 1 length: 233 mm
- Link 2 length: 215 mm
- Cross section for both links: 40 mm x 12 mm
- Wall count: 4
- Top/bottom wall count: 4
- Infill: 15%
- Infill pattern: honeycomb
- Installation: both links bend toward the thin-face direction, so weak-axis
  bending is along the 12 mm side.

## Stages

### P1_ONE_LINK_RIGID_TARGET_VIVE_LABEL

- Link count: 1
- Mounted link: link 1 only, length 233 mm
- Vive Tracker mount: link 1 tip
- IMU: unavailable
- Target position: rigid FK or ROS2 rigid simulation coordinate
- Label: Vive Tracker measured settled position
- Basic pose: robot clamped on the desk, manipulator faces away from the desk,
  link 1 vertical.

The offline FK config uses `q2=90 deg` as the vertical basic pose and samples
small offsets around that pose. Verify the hardware sign convention before
executing motion.

### P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL

- Link count: 2
- Mounted links: link 1 = 233 mm, link 2 = 215 mm
- Vive Tracker mount: link 2 tip
- IMU: unavailable
- Target position: rigid FK or ROS2 rigid simulation coordinate
- Label: Vive Tracker measured settled position
- Basic pose: robot clamped on the desk, tracker faces away from the desk, arm
  starts from the most folded L-shaped pose.

The offline FK config uses `q2=90 deg`, `q3=-90 deg` as the L-shaped basic pose.
Verify the hardware sign convention before executing motion.

## Frame Definition

Frame: robot base frame `B`

- `+X_B`: forward, away from the desk, manipulator-facing direction
- `+Y_B`: lateral
- `+Z_B`: upward

For a target point `p` and workspace center `c`:

```text
v = p - c
```

Forward condition:

```text
v dot [1, 0, 0] > 0
```

Azimuth:

```text
azimuth = atan2(v_y, v_x)
```

Default pilot mode:

```text
azimuth_deg in [-30, +30]
```

Optional main mode:

```text
azimuth_deg in [-45, +45]
```

This is a manual forward valid range for the pre-IMU experiment, not a precise
Vive ROI calibration.

## Target-Generation Algorithm

1. Generate deterministic joint-space candidates from stage-specific q offsets.
2. Compute rigid FK in meters using the base yaw plus planar arm convention.
3. Filter by `forward_trimmed_hemisphere`.
4. Reject candidates outside the manual forward workspace.
5. Bin candidates by stage-specific pose and workspace metadata.
6. Select targets with approximate bin balance.
7. Save target CSV and metadata JSON.

The bending bin is a deterministic rigid-pose proxy. It is not a measured
stiffness fit and does not use IMU values.

## Rigid FK Convention

For one link:

```text
rho = L1*cos(q2)
z   = L1*sin(q2)
x   = rho*cos(q1)
y   = rho*sin(q1)
```

For two links:

```text
rho = L1*cos(q2) + L2*cos(q2 + q3)
z   = L1*sin(q2) + L2*sin(q2 + q3)
x   = rho*cos(q1)
y   = rho*sin(q1)
```

Positions are written in both meters (`_m`) and millimeters (`_mm`). Existing
processed research CSVs remain millimeter-based.

For P1, `q_cmd_rad_3` is written as `0.0` and interpreted together with
`link_count=1`.

## Label Definition

The label is the Vive Tracker measured settled position. The rigid FK target is
not the label.

Supported label definitions for later processing:

- Direct target: `p_vive_settled`
- Residual: `delta_p = p_vive_settled - p_rigid_target`

The rigid FK target is an input, reference, and baseline quantity. It is not
ground truth.

## ML Architecture Note

The current recommended structure is:

```text
ML -> optimization
```

This patch does not train a model. Since IMU is unavailable in this pre-IMU
phase, candidate inputs are limited to values such as:

- `q_cmd`
- `q_encoder`, if available later
- rigid FK position
- target position
- trajectory metadata
- workspace/bin metadata

No IMU columns are required by this pre-IMU target-generation workflow.

## Commands

PowerShell from `02_FK_IK`:

```powershell
python .\scripts\generate_pre_imu_targets.py `
  --config .\config\pre_imu_forward_hemisphere_workspace.json `
  --stage P1_ONE_LINK_RIGID_TARGET_VIVE_LABEL `
  --mode pilot `
  --count 30 `
  --seed 101 `
  --output .\data\samples\pre_imu_stage1_targets.csv `
  --metadata-output .\data\samples\pre_imu_stage1_targets_metadata.json
```

```powershell
python .\scripts\generate_pre_imu_targets.py `
  --config .\config\pre_imu_forward_hemisphere_workspace.json `
  --stage P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL `
  --mode pilot `
  --count 40 `
  --seed 102 `
  --output .\data\samples\pre_imu_stage2_targets.csv `
  --metadata-output .\data\samples\pre_imu_stage2_targets_metadata.json
```

Optional separate pre-IMU schedule:

```powershell
python .\scripts\generate_pre_imu_trial_schedule.py `
  --targets .\data\samples\pre_imu_stage2_targets.csv `
  --repeats 3 `
  --seed 202 `
  --output .\data\samples\pre_imu_stage2_schedule.csv `
  --metadata-output .\data\samples\pre_imu_stage2_schedule_metadata.json
```

The existing full-workspace target generation and `scripts/generate_trial_schedule.py`
remain unchanged. This pre-IMU workflow is a separate config, module, CLI, and
data contract inside the 02 flexible-link workspace package.

## Limitations

- This dataset validates only the forward trimmed hemisphere region.
- It does not prove generalization to the full 360-degree workspace.
- IMU-based correction is not tested until IMU is available.
- The current workspace limits are manual and must be revised after measured
  Vive stability/range data is available.
- The q sign convention must be checked before hardware execution.

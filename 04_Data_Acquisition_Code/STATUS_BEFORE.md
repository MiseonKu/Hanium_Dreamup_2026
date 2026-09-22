# STATUS_BEFORE.md

Snapshot taken before any code changes for the SO-101 5-DOF IMU/Vive raw
acquisition task (master prompt, section 0-29).

## Repository

- Remote: `https://github.com/Inha-EEE-Capstone-26/capstone-code.git`
- Local clone: `C:\Users\hsw10\Documents\한이음 실험\capstone-code`
- Read-only ML reference clone: `C:\Users\hsw10\Documents\한이음 실험\ML-reference-readonly`
  (`https://github.com/Inha-EEE-Capstone-26/ML.git`, will not be modified)

## Git state at task start

```
branch: main
HEAD:   3a0c757e913454ade5007cbacbb52bae48f321a0
        "Extend WT901C initial pose capture to 10 seconds"
tracking: origin/main (up to date)
working tree: clean
uncommitted changes: none
untracked files: none
active process: none known
```

### Branches (local + remote)

```
* main
  remotes/origin/HEAD -> origin/main
  remotes/origin/aaaaa
  remotes/origin/codex/p1-simulation-data-collection-prep
  remotes/origin/codex/p2-two-link-experiment-06
  remotes/origin/codex/repo-audit-cleanup
  remotes/origin/codex/so101-tuning-ux
  remotes/origin/feature/ik-season2-upgrade
  remotes/origin/feature/imu-unit-separation
  remotes/origin/feature/move-pre-imu-workflow-to-03
  remotes/origin/feature/pre-imu-forward-hemisphere-targets
  remotes/origin/main
```

### Latest relevant commits (most recent first, top of `main`)

```
3a0c757 Extend WT901C initial pose capture to 10 seconds
ce0377a Improve WT901C collection reliability
c5ebe1a Add WT901C data collection tools
a486651 Merge PR #10 (P2 two-link experiment, 06)
15d9a9f Merge PR #8 (P1 simulation data collection prep)
035eba8 / df85c18 Merge PR #7/#6 (move pre-IMU target workflow to 03)
a6846d4 Merge PR #5 (pre-imu forward hemisphere targets)
```

## Top-level numbered packages (existing)

```
01_control
02_imu
03_SO-101_control
02_FK_IK
01_Initial_Research_Archive
02_SO-101_rtos_motor_driver
06_simulation_and_data_collect_preparation
07_WT901C
```

`08_` was unused at task start → new package created at
`04_Data_Acquisition_Code/` (this folder).

## Survey findings so far (pre-audit)

- `AGENTS.md` / `README.md` (root) define global non-negotiable rules: Tracker
  is ground-truth only (never control feedback), IMU-only means three-IMU-
  per-link ShapeFK reconstruction (not accel double integration), no
  hardware-moving/writing commands without explicit user request, no bypass
  of SO-101 calibration/INIT/torque/fault guards, real experiment data must
  stay out of commits.
- `03_SO-101_control/main/platformio.ini` confirms the SO-101 firmware target
  is `esp32dev` (espressif32, Arduino framework) — "ESP32" in the master spec
  refers to this board.
- Two candidate locations for existing target/workspace-generation code were
  found; a background audit is determining which is authoritative for the
  real 5-DOF arm vs. which are reduced-DOF (1-link/2-link) teaching packages:
  `02_FK_IK/` (has `fkik/`,
  `workspace_points/`, `config/pre_imu_forward_hemisphere_workspace.json`,
  `docs/PRE_IMU_FORWARD_HEMISPHERE_PROTOCOL.md`) vs.
  `06_simulation_and_data_collect_preparation/p1_one_link_experiment/` and
  `p2_two_link_experiment/`.
- A second open question: `02_SO-101_rtos_motor_driver/` exists alongside
  `03_SO-101_control/main/`, and git history includes "Promote SO-101 RTOS
  main firmware" — root `README.md` (dated 2026-05-03) still claims
  `03_SO-101_control/main` is active. Being resolved by background audit
  before the ESP32/encoder collector is designed.
- `07_WT901C/` has three parallel collector folders (`WT901C_imu0`,
  `WT901C_encoder0`, `WT901C_all_data`) with near-duplicate
  `collect_4imu.py` / `robust_io.py` / `imu_pose_alignment.py` /
  `live_4imu_visualizer.py`. None of this code has been run against real
  WT901C-RS232 hardware (per `07_WT901C/README.md`). Full structured audit
  in progress → `docs/07_WT901C_CODE_AUDIT.md`.

## Restricted scope for this task (reaffirmed)

Not performed without explicit approval, per master spec section 1.5:
ESP32 firmware flash, real motor commands, torque enable, calibration
commands, hardware experiment mode, real Vive/IMU/ESP32 device
configuration changes, `git push`, PR creation, force push,
`git reset --hard`, `git clean`, history rewrite. None of these were run to
produce this snapshot.

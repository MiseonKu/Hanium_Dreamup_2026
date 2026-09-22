# SO-101 3DoF Flexible-Link 실험 준비 패키지

이 패키지는 SO-101 기반 3DoF flexible-link manipulator 실험 전에 필요한 target point, flex-aware FK/IK solution, workspace sampling, trajectory, trial schedule, 검증 리포트를 생성하기 위한 경량 Python 코드입니다.

학습 모델 자체는 포함하지 않습니다. 목적은 IMU time-series와 robot command를 수집하기 전에 입력 조건을 균일하게 설계하고, 이후 Vive Tracker ground truth와 결합할 수 있는 metadata CSV를 준비하는 것입니다.

## 로봇 구조와 단위

- 자유도: 3DoF
- q1: base yaw
- q2: shoulder pitch
- q3: elbow pitch
- C 링크: 사용하지 않음
- A link: 기본 preset 233 mm, config에서 변경 가능
- B link: 기본 preset 215 mm, config에서 변경 가능
- L3: 0 mm
- 기본 단위: position은 mm, joint angle은 radian
- base frame: x forward, y left, z up
- link local frame: x_l은 링크 길이 방향, y_l은 넓은 변 방향, z_l은 12 mm 두께 방향이며 주 처짐 방향

Vive Tracker 중심이 B-link endpoint와 다를 수 있으므로 `config/robot_config.yaml`의 `tracker_offset_mm: [x_l, z_l]`로 2D offset을 줄 수 있습니다.

## 폴더 구조

이 폴더는 FK/IK 계산과 실험용 workspace point 알고리즘을 분리하되, `scripts/`에서 하나의 파이프라인으로 연동되도록 구성했습니다.

```text
02_FK_IK/
  config/             # robot/material/flex/sampling/trajectory 설정
  data/samples/       # 작은 pre-IMU 예시 target/schedule CSV와 metadata
  docs/               # pre-IMU protocol 등 설계 문서
  fkik/               # rigid FK/IK, flex-aware FK/IK, Jacobian, config dataclass
  workspace_points/   # candidate 생성, binning, uniform sampling, balanced shuffle
  trajectories/       # minimum jerk trajectory 생성
  scripts/            # CSV/plot/report 생성 CLI
  tests/              # FK/IK, sampling, trajectory 검증
  outputs/            # 재생성 가능한 산출물, git에는 .gitkeep만 포함
```

## A/B 링크 길이 preset

링크 길이는 `robot.link_lengths_mm.A`, `robot.link_lengths_mm.B`에서 바꿉니다. 현재 preset은 SO-101 실험 준비값입니다.

```yaml
robot:
  link_length_preset: SO101_AB_current
  link_lengths_mm:
    A: 233.0
    B: 215.0
```

계산 모듈은 모두 config에서 읽은 A/B 길이를 사용합니다. 따라서 A/B 값을 바꾸면 다음 항목이 자동으로 새 길이에 맞춰 다시 계산됩니다.

- rigid FK/IK reach와 elbow-up/down solution
- flex-aware FK의 pseudo-rigid link segment 위치
- weak-axis bending proxy `M1_weak`, `M2_weak`
- tracker offset이 반영된 effective B-link IK 길이
- joint-space workspace candidates
- radial/height/posture binning
- target point selection
- trial schedule의 target/q_cmd metadata
- validation plot과 distribution report

`scripts/generate_target_points.py`는 `workspace_candidates.meta.json`을 확인합니다. 링크 길이, tracker offset, flex model, joint limits, candidate sampling 설정이 현재 config와 다르면 기존 `workspace_candidates.csv`를 재사용하지 않고 자동으로 후보를 재생성합니다.

기존 config 호환성을 위해 `link_lengths_mm.L1`, `link_lengths_mm.L2`도 읽을 수 있지만, 새 실험에서는 A/B 키를 권장합니다.

## TPU 95A 링크 preset

현재 material preset은 다음 조건을 기준으로 합니다.

- material: TPU 95A
- wall_count: 4
- top/bottom wall count: 4
- infill_percent: 15
- infill_type: honeycomb
- cross_section: 40 mm x 12 mm
- bending direction: 40 mm 변에 수직인 12 mm weak-axis 방향

기존 TPU 64D threshold는 사용하지 않습니다. bending class는 `bending_score`의 percentile rank로 low/mid/high를 나눕니다.

## Pre-IMU forward hemisphere workflow

IMU 입고 전 Vive Tracker label만으로 진행하는 사전 실험 workflow는 이 02 workspace 패키지 안에 별도 config/module/CLI로 추가했습니다. 기존 연구 harness나 full-workspace 알고리즘을 바꾸지 않고, `forward_trimmed_hemisphere` target generation만 독립적으로 제공합니다.

- config: `config/pre_imu_forward_hemisphere_workspace.json`
- module: `workspace_points/pre_imu_targets.py`
- target CLI: `scripts/generate_pre_imu_targets.py`
- schedule CLI: `scripts/generate_pre_imu_trial_schedule.py`
- protocol: `docs/PRE_IMU_FORWARD_HEMISPHERE_PROTOCOL.md`
- 변경 기록: `CHANGELOG.md`

Pre-IMU material/link preset은 다음 최신 물리 조건을 사용합니다.

- preset id: `TPU95A_40x12_wall4_topbottom4_infill15_honeycomb`
- link 1: 233 mm
- link 2: 215 mm
- cross section: 40 mm x 12 mm
- wall count: 4
- top/bottom wall count: 4
- infill: 15%
- infill pattern: honeycomb
- weak-axis installation: 12 mm thin-face bending

Stage는 두 개입니다.

- `P1_ONE_LINK_RIGID_TARGET_VIVE_LABEL`: link 1만 장착, Vive Tracker는 link 1 tip, target은 rigid FK 좌표, label은 Vive 측정값
- `P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL`: link 1/2 장착, Vive Tracker는 link 2 tip, target은 rigid FK 좌표, label은 Vive 측정값

좌표계는 robot base frame 기준으로 `+X_B`가 책상에서 멀어지는 forward 방향, `+Y_B`가 lateral, `+Z_B`가 upward입니다. Pilot mode는 workspace center 기준 azimuth `[-30, +30] deg`만 통과시키고, main mode는 `[-45, +45] deg`로 확장할 수 있습니다.

중요한 데이터 원칙은 기존 1안과 같습니다. Vive Tracker 측정값은 학습 입력이 아니라 label 및 검증용 ground truth입니다. Pre-IMU 단계에서는 IMU column을 요구하지 않으며, 사용 가능한 입력 후보는 `q_cmd`, encoder가 있으면 `q_encoder`, rigid FK target, trajectory metadata, workspace/bin metadata입니다.

## 1안 데이터 구조 원칙

이번 구현은 supervised learning 수집을 위한 1안 구조를 전제로 합니다.

- 학습 입력 X에 들어갈 수 있는 값: IMU time-series, q_cmd, trajectory feature, p_target, bending_score, d_flex, point_class
- Vive Tracker는 학습 입력 X에 넣지 않음
- Vive Tracker는 label Y 생성 및 검증용 ground truth로만 사용
- label 후보는 나중에 metadata와 raw log를 결합해 모두 만들 수 있음
  - `p_Vive_rel = p_Vive - p_Vive_init`
  - `delta_p_flex = p_Vive - p_flex_pred`
  - `delta_p_target = p_Vive - p_target`

권장 raw log directory convention:

```text
dataset/
  session_001/
    metadata.json
    imu_01_raw.csv
    imu_02_raw.csv
    imu_03_raw.csv
    vive_tracker_raw.csv
    robot_command.csv
    events.csv
    target_points.csv
    trial_schedule.csv
```

수집 timestamp는 `time.perf_counter_ns()` 또는 `time.monotonic_ns()` 같은 monotonic 기준을 사용합니다. absolute timestamp는 모델 입력에 넣지 않습니다.

## FK/IK-lite 알고리즘

`fkik/rigid_fk.py`는 nominal rigid FK를 제공합니다. 여기서 `LA`와 `LB`는 config에서 읽은 A/B 링크 길이입니다.

```text
rho = LA*cos(q2) + LB*cos(q2 + q3)
z   = LA*sin(q2) + LB*sin(q2 + q3)
x   = rho*cos(q1)
y   = rho*sin(q1)
```

`fkik/flex_fk.py`는 full FEM이 아니라 pseudo-rigid flex-aware FK입니다. 각 링크 처짐각은 다음 형태입니다.

```text
delta_1 = k1 * M1_weak
delta_2 = k2 * M2_weak
```

현재 `M_weak`는 gravity/posture 기반 proxy입니다. 함수가 `compute_weak_moment_proxy(q, config)`로 분리되어 있어 이후 실제 moment model 또는 calibration model로 교체할 수 있습니다.

`fkik/rigid_ik.py`는 elbow-up/elbow-down과 signed radial yaw branch를 모두 계산하고 다음 기준으로 best solution을 고릅니다.

- joint limit 내부 여부
- q_init과의 거리
- joint limit margin
- singularity score
- round-trip FK error

`fkik/flex_ik.py`는 rigid IK를 초기값으로 사용한 뒤 finite-difference Jacobian과 damped least-squares update를 적용합니다.

```text
dq = J.T @ inv(J @ J.T + lambda^2 I) @ e
e  = p_target - fk_flex(q).p_flex
```

joint limit 초과, 과도한 delta, 수렴 실패는 `IKResult.reason`에 남깁니다.

## Uniform bin sampling을 쓰는 이유

실험 point probability가 Gaussian처럼 중심에 몰리면 이후 관측된 위치 오차, 진동, settling time, IMU drift가 workspace 영역과 섞여 bias를 만들 수 있습니다.

이 패키지는 xyz 공간을 직접 무작위 샘플링하지 않습니다. 먼저 joint space에서 대량 후보를 만들고 FK/Flex metric을 계산한 뒤, 다음 bin label 조합이 가능한 한 균일해지도록 target을 선택합니다.

- radial_shell
- azimuth_bin
- height_bin
- bending_class
- joint_posture_class

같은 bin 안에서는 normalized feature space의 farthest point sampling을 사용합니다.

```text
s = [x, y, z, q1, q2, q3, d_flex, bending_score]
```

균일하게 만드는 것은 target 위치, radial shell, azimuth, height, joint posture, bending class, 반복 횟수입니다. 실제 위치 오차, settling time, vibration RMS, Vive 측정값, IMU 진동 크기는 결과 물리량이므로 균일화하지 않습니다.

## Balanced trial schedule

`workspace_points/balanced_shuffle.py`는 반복 trial 순서를 cycle 단위로 구성합니다.

- 한 cycle 안에 각 target이 한 번씩 등장
- cycle 내부에서 bin이 가능한 한 섞이도록 interleave
- `repeats_per_point`만큼 cycle 반복
- 같은 target의 반복 수행이 가까이 붙지 않음

이렇게 하면 TPU 피로, 서보 온도, IMU drift, Vive tracking condition 변화가 특정 workspace 영역과 강하게 결합되는 것을 줄일 수 있습니다.

## 실행 방법

패키지 폴더로 이동합니다.

```powershell
cd "C:\Users\PC\Documents\VS Code\Hanium_Dreamup_2026\02_FK_IK"
python -m pip install -r requirements.txt
```

workspace 후보, target point, trial schedule, plot/report를 생성합니다.

```powershell
python scripts/generate_workspace_candidates.py --config config/robot_config.yaml
python scripts/generate_target_points.py --config config/robot_config.yaml
python scripts/generate_trial_schedule.py --config config/robot_config.yaml
python scripts/plot_workspace.py --config config/robot_config.yaml
python scripts/validate_fk_ik.py --config config/robot_config.yaml
pytest tests/
```

Pre-IMU forward hemisphere pilot target과 schedule 예시를 생성합니다.

```powershell
python scripts/generate_pre_imu_targets.py `
  --config config/pre_imu_forward_hemisphere_workspace.json `
  --stage P1_ONE_LINK_RIGID_TARGET_VIVE_LABEL `
  --mode pilot `
  --count 30 `
  --seed 101 `
  --output data/samples/pre_imu_stage1_targets.csv `
  --metadata-output data/samples/pre_imu_stage1_targets_metadata.json

python scripts/generate_pre_imu_targets.py `
  --config config/pre_imu_forward_hemisphere_workspace.json `
  --stage P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL `
  --mode pilot `
  --count 40 `
  --seed 102 `
  --output data/samples/pre_imu_stage2_targets.csv `
  --metadata-output data/samples/pre_imu_stage2_targets_metadata.json

python scripts/generate_pre_imu_trial_schedule.py `
  --targets data/samples/pre_imu_stage2_targets.csv `
  --repeats 3 `
  --seed 202 `
  --output data/samples/pre_imu_stage2_schedule.csv `
  --metadata-output data/samples/pre_imu_stage2_schedule_metadata.json
```

생성되는 주요 산출물:

```text
outputs/
  workspace_candidates.csv
  workspace_candidates.meta.json
  target_points.csv
  trial_schedule.csv
  workspace_preview.png
  radial_shell_distribution.png
  bin_count_distribution.png
  bending_score_distribution.png
  q_distribution.png
  round_trip_error_report.csv
  point_distribution_report.csv
```

## 주요 설정

모든 실험 파라미터는 `config/robot_config.yaml`에서 조정합니다.

- A/B link length: `robot.link_lengths_mm.A`, `robot.link_lengths_mm.B`
- tracker offset: `robot.tracker_offset_mm`
- q_init: `robot.q_init_rad`
- material preset: `material`
- flex gain: `flex_model.k1`, `flex_model.k2`
- joint limits: `joint_limits_rad`
- candidate/target/repeat 수: `sampling`
- IK tolerance, damping, finite difference step: `ik`
- minimum jerk duration/dt: `trajectory`

현재 기본 `q_init`은 base yaw 0 rad, A link vertical, B link horizontal을 의미합니다.

```text
q_init = [0, pi/2, -pi/2]
```

## 구현 메모

- C 링크는 사용하지 않습니다.
- ROS2, Gazebo, Isaac Sim dependency는 없습니다.
- heavy dependency 없이 numpy, pandas, matplotlib, pyyaml, pytest만 사용합니다.
- 기본 `k1=k2=0`에서는 flex FK가 rigid FK와 동일합니다.
- bending class는 absolute threshold가 아니라 percentile rank 기반입니다.
- `target_points.csv`와 `trial_schedule.csv`에는 사용된 `link_length_preset`, `link_a_mm`, `link_b_mm`가 저장됩니다.
- output CSV는 pandas로 저장합니다.
- validation plot은 matplotlib로 저장합니다.
- hardware port, MCU pin, servo mapping은 이 패키지에서 변경하지 않습니다.

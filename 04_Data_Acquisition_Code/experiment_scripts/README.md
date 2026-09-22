# 실제 하드웨어 실험 스크립트

SO-101 3축(shoulder_pan / shoulder_lift / wrist_flex) + WT901C IMU 4개 +
Vive Tracker.

> **수집 완료 (2026-08-10).** `sessions/real_motion_demo/` segment 000~041.
> 융합 데이터는 **036~041** — IMU/엔코더/Vive 세 스트림 모두 유효하고 시간
> 정렬됨, Vive `tracking_valid` 약 99.6%. 정착(settled) 트라이얼 10,055개 중
> **9,951개 사용 가능**(50 타깃 × 약 200 반복). 수집 후 처리는
> `tools/prepare_dataset.py` → `build_trials.py` → `build_timeseries.py`.
> 숫자와 파일 구조는 `../docs/STATUS.md`의 "Collected dataset",
> 컬럼 정의는 `../docs/DATA_SCHEMA.md`의 "Derived dataset" 참고.
>
> 아래 절차와 주의사항은 **다음 수집 때도 그대로 유효**함. 특히 1번(트래커
> 확인)과 `INIT` 금지, raw 틱 사용은 여전히 필수.

3축인 건 모터가 모자라서가 아니라 의도된 실험 범위야 —
`../docs/THREE_AXIS_EXPERIMENT_SCOPE.md`. wrist_roll/gripper 자리에는 모터 대신
Vive Tracker가 달려 있어.

## 이 폴더만 떼어내면 실행 안 됨

상위 패키지 모듈을 import 함:

| import | 실제 파일 |
|---|---|
| `collectors.collect_imu_rs232.IMURS232Collector` | `../collectors/collect_imu_rs232.py` |
| `collectors.collect_esp32_state_poll.parse_state_text` | `../collectors/collect_esp32_state_poll.py` |
| `core.schemas`, `core.jsonl_writer` | `../core/` |
| `fkik.geometry`, `fkik.rigid_fk` | `../../02_FK_IK/fkik/` |

각 스크립트 상단이 절대경로로 `sys.path`를 잡으니, **배포 단위는
`04_Data_Acquisition_Code/` 폴더 전체**(플랜 생성까지 재현하려면
`02_FK_IK`도)야. 다른 PC에서는 그 경로를 고쳐야 해.

## 사전 조건

- ESP32(`03_SO-101_control/main`, `kJointCount=3`)가 COM20, 캘리브레이션이
  서보 EEPROM에 저장돼 있을 것
- WT901C IMU 4개: COM16(imu1_link_root) / COM17(imu2_link_mid) /
  COM18(imu0_base) / COM19(imu3_tip), 9600 baud
- SteamVR 실행 + 트래커 `LHR-A84263A3` + 베이스스테이션 2개
- `pip install openvr pyserial numpy matplotlib`

## 실행 순서

| # | 스크립트 | 언제 | 모터 움직임 |
|---|---|---|---|
| 1 | `probe_vive.py` | 매 실행 직전 | 없음 |
| 2 | `check_state.py` | 매 실행 직전 | 없음 |
| 3 | `capture_reference_pose.py` | 기준 자세가 바뀌었을 때만 | 없음(토크만 켬) |
| 4 | `generate_3d_points_v5.py` | 플랜을 새로 뽑을 때 | 없음 |
| 5 | `plot_points.py` | 선택 | 없음 |
| 6 | `collect_3d_sweep_50_v2.py` | 본 수집 | **있음** |
| — | `raise_fault_temp.py` | 재플래시 후 1회 | 없음 |
| — | `run_real_imu_capture2.py` | IMU만 벤치 확인 | 없음 |

**1번은 절대 건너뛰지 말 것.** segment 012는 트래커가 꺼진 채로 50포인트를 다
돌아서 Vive 데이터가 전량 무효였어. 30초 확인을 아끼려다 한 시간을 날림.
지금은 6번 스크립트가 트래커 없으면 **모터를 건드리기 전에** 중단해.

## 알아야 할 것들 (고치기 전에 읽을 것)

### `INIT`과 `D<axis><deg>`는 쓰지 않아 — 의도된 것

펌웨어의 `INIT`은 **목표 위치를 쓰기 전에 토크부터 켜서**(main.cpp:3278 vs
3309), 서보가 낡은 목표값으로 먼저 튀어. 실제로 shoulder_lift가 가동범위를
150° 가로질러 반대편 기계 한계까지 날아간 적 있어.

대신 `safe_engage()`가 이 순서를 씀:

```
CAL_AUTO LOAD → MOVE_RAW(현재위치) → TORQUE ON    ← 목표=현재라 안 튐
```

`MOVE_RAW`는 토크 레지스터를 안 건드리거든. 그래서 `INIT`을 안 부르고,
`g_is_initialized`가 false로 남고, 레거시 `D` 명령은 쓸 수 없어 — 그래서 모든
이동이 `MOVE_RAW`야. 이게 더 나은 이유도 있어: **한 명령에 한 축**이라 과열
대책(순차 명령)이 코드 규율이 아니라 프로토콜 수준에서 보장돼.

자세한 건 `../docs/FIRMWARE_MOTION_HAZARDS.md`.

### 각도는 raw 틱으로 다뤄, `logical_deg`로 다루지 마

펌웨어의 `logical_deg`는 **물리 각도가 아니라 라벨**이야. 캘리브레이션된 raw
범위를 임의 구간에 선형으로 늘려놓은 값이고, 축마다 배율이 달라 (pan 1.173 /
lift 1.300 / wrist 1.101 물리도 per 라벨도).

이걸 각도로 써서 FK에 넣으면 shoulder_lift에서 30% 스케일 오차가 나. 실제
Vive 데이터로 확인됨 — raw 틱 기반으로 바꾸니 강체 예측 오차가 **77.4 →
48.7mm**로 줄었어. `../docs/FK_CONVENTION_CORRECTION.md`.

ST3215는 4096카운트/360°니까 **1 물리도 = 11.378틱**.

### 기준 자세

`REFERENCE_RAW = (2055, 2093, 1970)` — `../config/reference_pose.yaml`에 실측
기록. raw로 박아둔 이유는 위와 같아.

`generate_3d_points_v5.py`는 이 자세 기준 **raw 증가 방향(= 왼쪽 = 실험 영역)**
으로만 샘플링해. 세 축 모두 모터 방향이 일치하게 배치돼 있음.

### 폐기된 생성기

`generate_3d_points_50_v3.py`, `generate_3d_points_v4.py`는 **실행하면 즉시
중단되고 이유를 출력해.** 둘 다 위의 `logical_deg` 오류가 있어. 지우지 않은 건
v3가 segment 012를 만든 플랜의 출처라 데이터 이력의 일부이기 때문.

### 나머지

- **fault 복구는 `FAULTRESET` → `safe_engage()`.** `FAULTRESET`만 보내면
  플래그만 지워지고 토크가 안 돌아와 — 예전에 34개 타깃을 멈춘 채로 전부
  "성공"으로 잘못 기록한 적 있음.
- **위치 고착 감지**가 따로 있어. fault 플래그를 못 믿어서, 직전 타깃과 최종
  위치가 2번 연속 같으면 강제 복구.
- **온도 자동 정지는 일부러 뺐음.** shoulder_lift의 온도 센서가 고장이야.
  `raw_temp`는 레지스터 카운트지 °C가 아니고(변환에 필요한 데이터시트가 repo에
  없음), 실제 안전장치는 별도 PC의 외부 센서 + 전원 차단이야.
- **IMU는 6번 스크립트가 같은 프로세스에서 띄워.** `run_real_imu_capture2.py`를
  동시에 돌리면 COM 포트를 두고 싸워. 별도 실행은 벤치 확인용이고 출력도 다른
  폴더(`sessions/imu_bench_check`)로 가.
- **`segment_id`는 자동 할당돼.** 손으로 번호 올리지 마.
- **Ctrl+C 해도 됨.** IMU 스레드와 writer를 정리하고, 잘린 세그먼트는
  `.partial`로 남겨서 완전한 것으로 오인되지 않게 해.
- ESP32에 새 연결을 열 때마다 보드가 리셋돼. 캘리브레이션은 EEPROM에 살아있지만
  런타임에 `CAL_AUTO LOAD`로 다시 올려야 하고, **서보의 실제 토크 상태는 리셋과
  무관해** — `STATE`가 `torque=false`인데 팔이 잡혀 있을 수 있어.

### FK 예측과 Vive 실측이 다른 건 정상

그 차이가 유연 링크의 휨이고, 이 실험이 재려는 값이야. 좌표계를 맞추거나 링크
길이를 튜닝해서 줄이려 하지 말 것 — 신호를 지우는 거야.

단, 잔차에는 **트래커 고정 오프셋**도 섞여 있어
(`robot_config.yaml`의 `tracker_offset_mm`이 `[0,0]` 미측정). 오프셋은 자세와
무관하게 일정하고 휨은 자세에 따라 변하니까 분리 가능해.

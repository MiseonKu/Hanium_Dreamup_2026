# 본 실험 전 확인 목록

2026-08-05 기준. demo 데이터 감사에서 **실제로 데이터를 못 쓰게 만든** 원인들만
모았음. 일반론 아님 — 아래 항목은 전부 한 번씩 실제로 터진 것들.

## A. 실행 전 (5분)

| # | 확인 | 왜 (실제로 터진 사례) |
|---|---|---|
| A1 | `python experiment_scripts/probe_vive.py` — 트래커가 `bPoseIsValid=1`로 잡히는지 | **segment 012 전량 폐기.** 트래커 배터리가 나간 채로 50포인트를 다 돌았고, Vive 유효율 0.0% (759개 전부 `tracking_valid=false`). 30초짜리 확인을 건너뛴 대가가 12분 런 하나. |
| A2 | `python experiment_scripts/check_state.py` — 3축 전부 `ping=true`, `mapping_ready=true` | `mapping_ready=false`면 `logical_deg=NA`가 찍히고 (firmware main.cpp:1366) 그 축의 각도 데이터가 통째로 null. 캘리브레이션이 EEPROM에 안 올라온 상태. |
| A3 | 외부 온도 센서 + 전원 차단 장치가 켜져 있는지 | 서보 온도 센서가 고장이라 **소프트웨어에는 열 감시가 없음** (일부러 뺌, `docs/ST3215_THERMAL_AND_SEQUENCING.md`). 외부 장치가 유일한 안전장치. |
| A4 | 4개 IMU가 다 살아있는지 — `run_real_imu_capture2.py` 60초 벤치 | 죽은 포트는 조용히 빈 파일만 남김. 끝나고 알면 늦음. |
| A5 | 디스크 여유 | segment 012 하나가 ~19MB. 50포인트 × IMU 4개면 금방 참. |

**A1과 A4는 메인 스크립트를 시작하기 직전에.** 몇 시간 전에 확인한 건 의미 없음
(트래커는 자동 절전으로 꺼짐).

## B. 실행 중

- 메인 스크립트가 **IMU를 같은 프로세스에서** 띄움. 별도로
  `run_real_imu_capture2.py`를 같이 돌리지 말 것 — COM 포트를 두고 싸움.
- `segment_id`를 손으로 정하지 말 것. 자동 할당됨.
- 콘솔에 `!!! position identical to previous target` 이 반복되면 팔이 멈춘 것.
  스크립트가 자동 복구하지만, 3~4회 연속이면 중단하고 기구부 확인.
- Ctrl+C 해도 됨. IMU 스레드와 writer를 정리하고, 잘린 세그먼트는 `.partial`로
  남겨서 완전한 것으로 오인되지 않게 함.

## C. 실행 후 — 데이터가 실제로 쓸 수 있는지 검증

파일이 생겼다 ≠ 라벨 데이터로 쓸 수 있다. segment 012는 파일 자체는 멀쩡했고
(759 encoder + 759 vive 레코드, 파싱 오류 0건) 그런데도 못 씀. 아래를 돌릴 것:

```bash
python -m pytest tests/test_demo_replay.py -q     # 정상/비정상 세그먼트 판별 로직
python tools/generate_audit_outputs.py            # sampling/packet/target-plan 감사 JSON
```

수동으로 볼 것:

1. **스트림 간 시간 겹침.** IMU 구간과 encoder/Vive 구간이 실제로 겹치는지.
   segment 012는 IMU가 encoder보다 **352.3초 먼저 끝나** 있었음 — 겹침 0.
   (지금은 한 프로세스라 구조적으로 안 생기지만, 확인은 할 것.)
2. **Vive 유효율.** `tracking_valid=true` 비율. 006/007은 1.0, 012는 0.0.
   0.5 밑이면 그 런은 라벨로 못 씀.
3. **target_id 커버리지.** 계획된 50개 target_id가 encoder 레코드에 전부
   나타나는지. 빠진 게 있으면 그 포인트에서 멈췄던 것.
4. **위치가 실제로 다른지.** 연속된 target들의 최종 `encoder_raw_tick`이 서로
   달라야 함. 예전 런에서 **34개 타깃이 같은 자리에 멈춘 채로 "성공"으로
   기록된 적 있음** — fault 플래그도 안 뜨고 레코드 수도 정상이었음.
   `FAULTRESET`만 보내고 토크를 다시 안 걸어준 게 원인.

   **정정 (2026-08-05):** 원문은 복구책으로 `INIT`을 적어놨는데, **`INIT`은
   쓰면 안 됨.** `INIT`은 goal position을 쓰기 *전에* 토크를 켜서, 서보가
   레지스터에 남아있던 옛 목표로 먼저 튀어나감 (`docs/FIRMWARE_MOTION_HAZARDS.md`
   #1 — shoulder_lift가 이 경로로 기계적 끝단까지 ~150° 끌려간 적 있음).
   복구는 `FAULTRESET` → `safe_engage()` (= 토크 꺼진 상태에서 현재 위치를
   `MOVE_RAW`로 써넣고 `TORQUE ON ALL`). 래치된 fault 직후는 stale goal이
   실제 위치에서 제일 멀 때라 `INIT`을 부르기 가장 나쁜 순간임.

   비교는 `q_encoder_rad`가 아니라 raw tick으로 할 것 — `logical_deg`(따라서
   `q_encoder_rad`)는 물리 각도가 아니라 축마다 스케일이 다른 라벨임
   (`docs/FK_CONVENTION_CORRECTION.md`).

## D. 아직 하드웨어로 확인 못 한 것

코드는 준비됐지만 실물로 검증 안 된 항목 — 첫 실런에서 같이 봐야 함:

| 항목 | 상태 |
|---|---|
| 순차 축 명령이 실제로 피크 전류를 낮추는지 | 논리적으로는 맞지만 **전류 실측 안 함**. `docs/ST3215_THERMAL_AND_SEQUENCING.md` |
| `AXIS_COMMAND_GAP_S = 0.6s`가 적정한지 | 배치 윈도우(40ms) 대비 여유만 보고 정한 값. 실측하면 줄일 수 있음 (50포인트 기준 ~90초 차이) |
| IMU 스케일 상수 (±16g / ±2000dps / ±180°) | 장치 레지스터로 확인한 적 없음 — `docs/07_WT901C_CODE_AUDIT.md`의 `UNVERIFIED_HARDWARE_SETTING` 그대로 |
| WT901C 출력 rate 설정 | 현재 packet type당 ~10Hz. UART는 46%만 쓰고 있어서 **링크가 아니라 센서 자체 설정이 병목** (`docs/SAMPLING_AND_UART_BUDGET.md`). 동역학 분석하려면 센서 rate 레지스터 + baud 둘 다 올려야 함 |
| quaternion(0x59) / time(0x50) 패킷 | 파서는 있는데 장치가 한 번도 안 보냄 — 출력 마스크 설정 필요 |
| ST3215 raw 레지스터 → 물리 단위 | 데이터시트가 repo에 없음. `raw_voltage/temp/current/load`는 카운트 그대로 저장 중 |

## E. 첫 실런 권장 형태

50포인트 풀런 말고 **5포인트 축소판 먼저**:

1. A1~A5 전부 확인
2. `target_plan_3d_random_n50_v3.json`에서 앞 5개만 남긴 사본으로 1회 실행
3. C의 4가지 검증을 그 5포인트에 대해 돌려서 전부 통과하는지 확인
4. 통과하면 그때 50포인트 풀런

이유: 위 실패 사례들은 전부 **런이 끝난 뒤에야** 발견됐음. 5포인트짜리로
파이프라인 전체가 온전한지 먼저 확인하면 12분이 아니라 1분 손해로 끝남.

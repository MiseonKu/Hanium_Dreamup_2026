# Data Acquisition Code

SO-101 실험에서 다음 데이터를 동시에 수집하기 위해 사용한 코드다.

- WT901C IMU 4개
- 모터 엔코더 값
- Vive Tracker 위치
- 측정 시간과 실험 상태

수집된 데이터는 원본 JSONL로 기록한 뒤 trial 단위 데이터셋으로 변환한다.

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `collectors/` | IMU, 엔코더 및 Vive 데이터 수집 코드 |
| `core/` | 데이터 형식, 저장, 시간 동기화 및 오류 처리 |
| `experiment_scripts/` | 실제 실험에 사용한 수집 스크립트 |
| `experiment/` | 실험 진행, 안전 확인 및 정지 상태 판정 코드 |
| `tools/` | 원본 검사와 trial 데이터셋 생성 도구 |
| `config/` | 센서와 실험 설정 예시 |
| `mocks/` | 하드웨어 없이 시험하기 위한 가상 센서 |
| `tests/` | 수집 및 저장 코드 테스트 |
| `docs/` | 데이터 형식, 실험 절차 및 안전 관련 문서 |
| `firmware/` | ESP32 통신 규격 설명 |

## 관련 폴더

- 수집 데이터 ZIP: `../03_Dataset_Collection/imu_added_experiment/`
- 데이터 정제 및 ML 학습: `../05_ML/`
- 로봇 제어 코드: `../../Inha-EEE-Capstone-26/04_SO-101_control/`

현재 이 폴더에는 수집 코드만 있으며 대용량 실제 데이터는 들어 있지 않다.
과거 수집 데이터와 파생 데이터는 `03_Dataset_Collection`의 ZIP 파일로
보관한다.

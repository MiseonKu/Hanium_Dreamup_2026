# Initial Research Archive

SO-101의 여러 IMU를 이용해 로봇 끝 위치를 추정하려던 초기 연구계획과
오프라인 프로토타입을 보관한 폴더다.

## 포함 내용

- `docs/`: 초기 연구계획과 실험 규칙
- `host/`: 오프라인 분석용 Python 코드
- `configs/`: 필터, 센서 융합 및 품질검사 설정
- `calibration/`: 센서와 좌표계 보정 양식
- `data/samples/`: 실제 측정값이 아닌 테스트용 합성 데이터
- `simulation/`: 시뮬레이션용 로봇 모델
- `tests/`: 계산 및 데이터 형식 검사 코드

## 현재 프로젝트와의 관계

이 폴더는 참고용 보관자료이며 현재 실험의 실행 폴더가 아니다.

- FK/IK 및 실험 목표점 설계: `../02_FK_IK/`
- 실험 데이터: `../03_Dataset_Collection/`
- 실제 데이터 수집 코드: `../04_Data_Acquisition_Code/`
- 데이터 정제 및 ML 학습: `../05_ML/`
- 위치 추정 데모: `../06_Position_Estimation/`

`data/raw`, `data/processed`, `data/analysis`에는 현재 실제 실험 데이터가
들어 있지 않다. 자세한 초기 설계가 필요할 때만 `docs/`를 참고한다.

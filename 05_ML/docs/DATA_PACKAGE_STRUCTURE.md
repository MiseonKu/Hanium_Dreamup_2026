# 데이터와 릴리스 패키지 구조

상태: 2026-08-18 전수 감사 기준. 이 문서는 원천 archive에서 준비 ZIP과
endpoint 회귀 결과 ZIP까지의 경로, 행 수, 주요 열, join key, 안전한 소비
순서를 설명한다. public schema의 규범 문서는
[DATA_CONTRACT.md](DATA_CONTRACT.md)와
[ENDPOINT_REGRESSION_CONTRACT.md](ENDPOINT_REGRESSION_CONTRACT.md)다.

## 전체 흐름

```text
08 수집/파생 evidence
  -> 06/imu_added_experiment/derived_dataset_archive.zip
       -> trials.parquet 한 member만 읽음
          -> 09 label-blind 정제와 split-before-augmentation
             -> ml_preparation_v1.zip
                -> nested group CV와 OOF 평가
                   -> endpoint_regression_experiment_v1.zip
```

| 계층 | 계약 또는 역할 | SHA-256 | 크기 |
|---|---|---|---:|
| source archive | 08 파생 결과의 LFS archive | `ceadf164a15a3ce302f8b2ffdde450df0f6d3a91c3b0c258a28acb5de9b92d64` | 541,320,066 B |
| preparation ZIP | `alphaf.ml_preparation.v1` | `94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930` | 104,446,557 B |
| experiment ZIP | `alphaf.endpoint_regression.experiment.v1` | `5eadbd2971c015d1bfd46d09ab19b3b6fa3447fe8bdc46079a54114f8c14d9bc` | 77,567,241 B |

두 ZIP은 서로 덮어쓰는 버전이 아니다. experiment ZIP은 고정된 preparation
ZIP을 읽어 만든 별도 결과 계약이다.

## 원천 archive 경계

V1 준비 파이프라인이 읽는 원천은 한 member뿐이다.

| 항목 | 값 |
|---|---|
| archive 경로 | `06_simulation_and_data_collect_preparation/imu_added_experiment/derived_dataset_archive.zip` |
| member | `derived/parquet/real_motion_demo/trials.parquet` |
| member SHA-256 | `1b0a821a1822f237adadb1c978ea73da02b0e4bb7f099ce085a63931e9792b54` |
| source table | 10,055행 x 83열 |
| member bytes | 3,575,342 |

archive/member hash, 정확한 Arrow field/type/order, 행 수 중 하나라도 다르면
중단한다. Git LFS pointer text도 데이터로 읽지 않는다. 08의
`timeseries.parquet`과 07의 연속 캡처는 V1 입력이 아니다.

## preparation ZIP 구조

ZIP root는 `ml_preparation_v1/`이고 총 35개 member다. `manifest.json`이
자신을 제외한 34개 member의 SHA-256을 기록한다.

```text
ml_preparation_v1/
├── README.md
├── manifest.json
├── config.resolved.json
├── references.json
├── canonical/
│   ├── canonical_real.parquet
│   ├── column_schema.csv
│   ├── feature_sets.json
│   └── source_schema.json
├── splits/
│   └── split_assignments.parquet
├── augmentation/
│   ├── provenance.parquet
│   ├── segment_loso/fold_{36..40}/
│   │   ├── local_interp_v2.parquet
│   │   └── cmixup_offline_v1.parquet
│   └── target_group/fold_{0..4}/
│       ├── local_interp_v2.parquet
│       └── cmixup_offline_v1.parquet
└── analysis/
    ├── audit.json
    ├── report.md
    ├── segment_summary.csv
    ├── segment_target_summary.csv
    └── target_summary.csv
```

### 핵심 table 크기

| member | 행 x 열 | 역할 |
|---|---:|---|
| `canonical/canonical_real.parquet` | 7,707 x 128 | 보존한 실측 trial과 eligibility/QC |
| `splits/split_assignments.parquet` | 231,060 x 9 | 준비 계약의 outer/inner role assignment |
| `augmentation/provenance.parquet` | 123,232 x 20 | 모든 synthetic parent와 제약 감사 정보 |
| 각 synthetic-only Parquet | fold별 가변 x 94 | 한 scheme/fold/method의 합성 feature/label |

source 10,055행에서 구조적으로 완전한 segment 36-40 실측 7,707행을
canonical로 보존한다. 그중 7,702행만 학습 적격이며, 실패한 segment 40의
5행도 삭제하지 않고 exclusion reason과 함께 남긴다. warning이 없는
strict-QC subset은 7,670행이다.

### split assignment를 크게 보이는 이유

`split_assignments.parquet`의 231,060행은 서로 다른 실측 trial 수가 아니다.
7,702개의 학습 적격 trial에 30개 context의 role을 부여한 long-form
assignment다.

| scheme/context | 수 |
|---|---:|
| segment outer refit | 5 |
| segment inner validation | 20 |
| target-group outer refit | 5 |
| 전체 context | 30 |
| assignment | `7,702 x 30 = 231,060` |

preparation ZIP에는 outer-refit synthetic만 materialize한다. segment inner
synthetic은 검증된 real rows와 split context에서 cache CLI로 재현하며,
experiment 단계의 target inner context도 해당 real-train에서 새로 생성한다.

### feature와 label

| view | 열 수 | 내용 |
|---|---:|---|
| `encoder_only_v1` | 3 | reference-relative encoder delta radian |
| `four_imu_endpoint_v1` | 40 | 4 IMU x Euler sin/cos 6 + accel 3 + gyro RMS 1 |
| `fusion_v1` | 43 | encoder 3 + four-IMU endpoint 40 |

label은 정렬하지 않은 raw `vive_world`의 XYZ millimetre다. Tracker/Vive,
planned 좌표, ID, timestamp, QC, provenance는 primary feature allow-list에
들어갈 수 없다. planned와 미보정 magnetometer 열이 synthetic file에
존재하는 이유는 각각 parent support 감사와 shortcut diagnostic을 위한
것이지 primary 모델 입력 허용을 뜻하지 않는다.

### synthetic 행의 두 표현

각 synthetic은 두 위치에서 보인다.

- scheme/fold/method Parquet: 실제 합성 feature와 label 94열.
- `augmentation/provenance.parquet`: parent와 생성 조건을 모은 20열 index.

두 파일은 `synthetic_id`로 연결한다. method별 61,616행, 전체 123,232행이다.
이는 `10 outer folds x 2 methods`에 각 outer real-train 수만큼 생성한 결과다.

preparation provenance의 `sample_weight=1.0`과 학습 objective weight를
혼동하면 안 된다. experiment recipe는 real과 추가 행을 각각 `0.5`로
가중해 두 집단의 mass를 1:1로 맞춘다.

## experiment ZIP 구조

ZIP root는 `endpoint_regression_experiment_v1/`이고 총 32개 member다.

```text
endpoint_regression_experiment_v1/
├── manifest.json
├── release_identity.json
├── config.resolved.json
├── environment.json
├── MODEL_CARD.md
├── analysis/
│   ├── eligible_real_audit.parquet
│   └── leakage_audit.json
├── splits/
│   ├── assignment_identity.json
│   └── evaluation_assignments.parquet
├── matrix/
│   ├── cells.json
│   └── feature_allowlists.json
├── augmentation/
│   ├── context_provenance.parquet
│   └── row_provenance.parquet
├── selection/
│   ├── inner_selection_trace.parquet
│   ├── final_selection_trace.parquet
│   └── outer_preprocessing.json
├── predictions/
│   ├── oof_seed.parquet
│   └── oof_ensemble.parquet
├── metrics/
│   ├── cell_metrics.parquet
│   ├── fold_metrics.parquet
│   └── aggregate_metrics.parquet
├── statistics/
│   ├── augmentation_gates.parquet
│   └── paired_comparisons.parquet
├── sensitivity/
│   ├── strict_oof_metrics.parquet
│   ├── strict_model.joblib
│   └── strict_refit_metadata.json
├── final/
│   ├── model.joblib
│   ├── model_metadata.json
│   ├── inference_fixture.parquet
│   └── inference_fixture_predictions.parquet
└── reports/
    ├── report.md
    └── segment_loso_macro_rmse.png
```

### 핵심 table 크기

| member | 행 x 열 | 역할 |
|---|---:|---|
| `analysis/eligible_real_audit.parquet` | 7,702 x 68 | 실험이 소비한 적격 실측 projection |
| `splits/evaluation_assignments.parquet` | 385,100 x 9 | segment와 target nested context 전체 |
| `augmentation/context_provenance.parquet` | 100 x 16 | context/recipe별 생성 설정과 hash |
| `augmentation/row_provenance.parquet` | 492,928 x 27 | inner 재생성 및 outer refit의 synthetic parent 감사 |
| `selection/inner_selection_trace.parquet` | 13,520 x 37 | hyperparameter 선택 evidence |
| `predictions/oof_seed.parquet` | 662,372 x 23 | seed별 outer-test 예측 |
| `predictions/oof_ensemble.parquet` | 477,524 x 23 | cell별 ensemble outer-test 예측 |
| `metrics/cell_metrics.parquet` | 15,500 x 29 | segment/target cell-balanced 지표 |
| `metrics/fold_metrics.parquet` | 310 x 32 | outer fold 지표 |
| `metrics/aggregate_metrics.parquet` | 62 x 37 | scheme/cell aggregate 지표 |
| `selection/final_selection_trace.parquet` | 145 x 11 | final candidate/config aggregate |
| `sensitivity/strict_oof_metrics.parquet` | 62 x 25 | strict-QC OOF sensitivity |
| `statistics/augmentation_gates.parquet` | 8 x 15 | 증강 채택 gate |
| `statistics/paired_comparisons.parquet` | 24 x 19 | deterministic multiway bootstrap 비교 |
| `final/inference_fixture.parquet` | 32 x 44 | 직렬화 재현성용 고정 입력 |
| `final/inference_fixture_predictions.parquet` | 32 x 4 | 고정 예상 예측 |

evaluation assignment 385,100행도 long form이다. preparation의 30개에
target inner 20개가 추가되어 `7,702 x 50 = 385,100`이다. OOF 행 수 역시
실측 trial 수가 늘어난 것이 아니라 여러 사전 고정 experiment cell과 seed가
각자의 outer-test 예측을 기록했기 때문에 크다. 같은 evaluation cell 안에서는
각 outer-test trial이 ensemble OOF 예측을 정확히 한 번 가져야 한다.

## 기본 키와 join 지도

| 대상 | 기본 키 또는 연결 키 |
|---|---|
| canonical real | `trial_uid` |
| split assignment | `trial_uid` + context columns |
| synthetic | `synthetic_id` |
| synthetic -> canonical parent | `parent_a_trial_uid`, `parent_b_trial_uid` -> `trial_uid` |
| context | `scheme/split_scheme`, `outer_fold`, `inner_fold`, `split_context` |
| experiment augmentation context | `(scheme, outer_fold, inner_fold, recipe)` |
| ensemble OOF prediction | `scheme`, `cell_id`, `trial_uid` |
| seed-level OOF prediction | 위 key + `prediction_kind`, `seed_index`, `model_seed` |
| fold/aggregate metric | `scheme`, `cell_id`, fold level columns |

`inner_fold=-1`은 outer-refit sentinel이며 문자열 context와 함께 해석한다. 서로
다른 scheme이나 cell의 OOF를 `trial_uid`만으로 바로 join하면 many-to-many가
되므로 금지한다. synthetic에는 실제 segment/target/timestamp/QC 상태를
fabricate하지 않으며 parent provenance를 통해서만 real row와 연결한다.

```text
canonical_real.trial_uid
  ├── split_assignments.trial_uid
  ├── provenance.parent_a_trial_uid
  └── provenance.parent_b_trial_uid

synthetic_file.synthetic_id
  └── provenance.synthetic_id

evaluation cell + outer fold + trial_uid
  └── OOF prediction and fold metric evidence
```

## 안전한 소비 순서

1. Git LFS 파일이 pointer text가 아니라 실제 ZIP인지 확인한다.
2. 외부 ZIP SHA-256을 고정값과 비교한다.
3. ZIP root와 `manifest.json` contract name을 확인한다.
4. 모든 manifest member의 size와 SHA-256을 검사한다.
5. Arrow field/type/order와 고정 행 수를 검사한다.
6. canonical에서 `train_eligible=true`인 real row만 train pool로 삼는다.
7. 먼저 split context와 role을 고른 뒤 그 context의 train row만 사용한다.
8. synthetic을 쓰는 recipe에서만 같은 context의 synthetic을 추가한다.
9. validation/test에는 real row만 남았는지 검사한다.
10. 학습 입력은 `feature_allowlists.json`의 정확한 열과 순서를 사용한다.

권장 검증 명령은 다음과 같다.

```powershell
cd "09_SO101_IMU_VIVE_LEARNING"
python -m imu_vive_learning.cli validate `
  --run-dir artifacts/static_endpoint_v1
python -m imu_vive_learning.cli validate-experiment `
  --run-dir artifacts/endpoint_regression_v1
```

패키지 validator는 ZIP 경로, CRC, member 집합, 크기와 member SHA-256을
검사한다. run validator는 여기에 split, schema, leakage, parent, metric,
inference 의미 검사를 더한다. endpoint run validator는 50개 split context
x 2개 method인 100개 context/recipe 조합의 production generator를 measured
real-train에서 재연해 492,928행의
seed, parent/lambda draw와 replacement schedule까지 exact 비교한다. 이
전수 검사는 CPU-bound라 수 분이 걸릴 수 있다.

패키지 소비자는 ZIP을 임의로 풀어 일부 파일만 신뢰하지 말고 public
validator 또는 동일한 fail-closed 검사를 사용해야 한다. 이번 감사에서는
두 ZIP의 CRC·중복·manifest 누락·크기·hash 불일치가 모두 0건이었고 공식
package validator도 통과했다.

## 메타데이터와 직렬화 주의점

- immutable preparation V1의 `column_schema.csv`는
  `*_gyro_rms_rad_s` unit을 `rad`로 적었지만 열 이름, QC threshold,
  실제 계약 단위는 `rad/s`다. V1 파일을 소급 수정하지 않고 이 차이를
  기록한다.
- planned XYZ는 mm이지만 package metadata에 frame이 비어 있다. 이를
  robot-base/TCP frame으로 추정하면 안 된다.
- preparation local 행의 `joint_parent_distance_deg`는 joint gate를 쓰지
  않아 `NaN`이다. experiment provenance는 실제 encoder parent 차이를
  재계산하고 generator 값을 별도로 보존한다.
- experiment `row_provenance.parquet`은 synthetic feature/label payload가
  아니라 재생성 provenance다. payload는 고정된 preparation release,
  config와 generator로 복원한다.
- 고정 fixture prediction은 모델 직렬화 재현성 확인용 4열 payload다.
  일반 `predict-endpoint` 출력의 frame/unit/model-hash metadata 계약과
  같은 파일 형식으로 해석하지 않는다.
- `model.joblib`은 checksum을 검증한 trusted release에서만 load한다.
  임의의 외부 ZIP에 대한 안전한 parser가 아니다.
- manifest의 `code_revision`은 결과를 동결한 실험 commit이다. 이후 PR의
  squash merge commit과 동일한 식별자로 해석하지 않는다.
- endpoint manifest는 preparation처럼 별도 Parquet schema fingerprint
  registry를 갖지 않는다. 각 member 전체 hash와 run validator의 실제
  schema/semantic 검사를 함께 사용한다.

## 결과 해석 경계

- preparation ZIP의 123,232 synthetic row는 실험 후보이며 최종 채택을
  뜻하지 않는다.
- 고정 gate를 통과한 증강 비교는 0/8이다.
- 최종 `model.joblib`은 real 7,702행, synthetic 0행으로 fit되었다.
- 최종 model SHA-256은
  `14ddf90e009f112e85600f9346281dd91b125da10f37ca4b211c42c03ae45401`이다.
- 출력은 static 3초 summary에서 추정한 raw `vive_world` XYZ mm다.
- robot base/TCP 좌표, 정식 ShapeFK, 저장소 정의의 `IMU-only`, 온라인
  aggregator, 제어 모델이 아니다.
- 같은 수집 session 내부의 탐색적 V1이므로 독립 실측 session 검증이
  필요하다.

증강 규칙의 논문 근거, 프로젝트 휴리스틱, 실제 parent 거리와 gate 결과는
[AUGMENTATION_EVIDENCE.md](AUGMENTATION_EVIDENCE.md), 전체 평가 해석은
[ENDPOINT_REGRESSION_RESULTS_V1.md](ENDPOINT_REGRESSION_RESULTS_V1.md)를
참조한다.

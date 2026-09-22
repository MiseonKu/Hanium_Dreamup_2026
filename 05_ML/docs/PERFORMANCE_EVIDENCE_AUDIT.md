# 성능 근거 감사 기록

상태: 2026-08-29 기준, 기존 결과를 변경하지 않는 사후 증거 감사 문서.

## 결론

현재 저장소가 직접 뒷받침하는 가장 강한 결론은 다음과 같다.

- 7,702개의 실측 정적 trial로 구성된 **동일 수집 corpus 내부**에서,
  encoder 3개와 네 WT901C의 3초 요약 feature 40개를 입력으로 사용하는
  MLP가 raw `vive_world` XYZ를 예측하는 신호를 보였다.
- frozen endpoint V1의 primary segment-LOSO 결과는 macro RMSE3D/P95
  `14.2628 / 21.0190 mm`다.
- robust-v2의 선택은 후보당 20개 segment-inner context에서 이루어졌고,
  group-balanced ERM이 `J=16.036852832 mm`로 선택되었다. 이미 관측한 outer
  fold를 다시 사용한 개발 진단은 macro RMSE3D/P95
  `14.2357 / 20.7053 mm`다.
- 위 수치는 독립 수집 session, robot-base/TCP 좌표, 동적 추정, 제어 성능을
  증명하지 않는다. robust-v2 outer 결과는 명시적으로
  `reused_holdout_development_diagnostic`이다.

증강에 대한 정확한 결론도 구분해야 한다.

- 성능 비교는 **증강 8종**이 아니라 C-Mixup과 local interpolation의
  **2개 recipe x 4개 모델 = 8개 비교**다.
- primary RMSE point estimate는 8개 중 4개가 개선되고 4개가 악화됐다.
- 그러나 고정 engineering 채택 gate를 통과한 비교는 `0 / 8`이었다.
- 따라서 “증강이 모두 성능을 악화시켰다”가 아니라, “V1의 고정 채택
  기준을 만족할 만큼 크고 안정적인 개선 근거가 없었다”가 맞다.
- robust-v2는 이 결정을 승계해 real-only로 학습했으며, robust-v2와
  C-Mixup/local interpolation의 조합을 직접 ablation하지 않았다.

전체 수치 표는 [ALL_DATA_COMPARISONS.md](ALL_DATA_COMPARISONS.md)에 있다.

## 증거 계보와 인증값

| 단계 | 추적된 근거 | SHA-256 또는 상태 |
| --- | --- | --- |
| 원천 archive | `06_simulation_and_data_collect_preparation/imu_added_experiment/derived_dataset_archive.zip` | `ceadf164a15a3ce302f8b2ffdde450df0f6d3a91c3b0c258a28acb5de9b92d64` |
| 고정 입력 member | `derived/parquet/real_motion_demo/trials.parquet` | `1b0a821a1822f237adadb1c978ea73da02b0e4bb7f099ce085a63931e9792b54` |
| 준비 release | `06_simulation_and_data_collect_preparation/imu_added_experiment/ml_preparation_v1.zip` | `94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930` |
| endpoint V1 release | `06_simulation_and_data_collect_preparation/imu_added_experiment/endpoint_regression_experiment_v1.zip` | `5eadbd2971c015d1bfd46d09ab19b3b6fa3447fe8bdc46079a54114f8c14d9bc` |
| V1 aggregate table | ZIP member `metrics/aggregate_metrics.parquet` | `48d50c10df8358ae31f7a4566f0e2c1abc52b3454477fcaffd9c55d1de80e285` |
| V1 paired table | ZIP member `statistics/paired_comparisons.parquet` | `4b5afee911c897fbeaeaef4afef53363f03e003670e701a72f7063a047be1407` |
| V1 augmentation gate | ZIP member `statistics/augmentation_gates.parquet` | `5a4e9f59913b8b9aa7517034db3e804e6f5dabeba5681f75cb4fcc28747098d7` |
| robust-v2 canonical config identity | loader가 canonicalize한 resolved config | `ed28b16cde0be391b1c893003a58bd26c4de57df144e6e1cdf64c7a1dca8339a` |
| robust-v2 YAML file bytes | `configs/endpoint_regression_gpu_robust_v2.yaml` | `b05910252eb8fee987af174e39f61f8724c3967da33ee2821978eb246167b0d2` |
| robust-v2 run manifest | 결과 문서에 기록된 ignored run identity | `b4c9c6216f5f23425ba4e2e7777cc93a64f84deab52abf1c94b1ef81c5232ade` |
| robust-v2 semantic validation | 결과 문서에 기록된 ignored validation | `8c65cd5efad98698ba588498510a8420350ed10d7abf4feefd926f1614e3b4b7` |
| robust-v2 model fingerprint | 공개 inference contract identity | `50a1b01c8a1de42e2ae7248ff37bcaf61a93234a6493b049e8454b1d3cc719a1` |

endpoint V1의 원본 비교 Parquet과 OOF prediction은 Git LFS ZIP에 들어 있다.
GPU-v1과 robust-v2의 대규모 run directory는 저장소 정책상 ignored이며,
추적된 결과 문서가 핵심 표와 인증 hash를 기록한다. 이 차이 때문에
robust-v2는 V1 release와 동일한 수준의 GitHub 내 원시-artifact 공개라고
표현하면 안 된다.

GPU-v1은 clean pre-launch source binding이 아니라 uninterrupted run 이후
복구·검증한 `post_run_uninterrupted_capture`다. 반면 robust-v2는 clean
training-source commit과 모든 task의 source fingerprint를 기록했다. 이
provenance 차이도 세 lineage 수치를 같은 증거 등급으로 표현하지 않는
이유다.

## 데이터와 평가 설계

| 항목 | 값 | 해석 |
| --- | ---: | --- |
| 감사한 source row | 10,055 | 전체 trial-builder 출력 |
| canonical real | 7,707 | segment 36--40의 구조적으로 완전한 실측 trial |
| train eligible | 7,702 | label-blind acquisition 기준을 통과한 학습 가능 trial |
| strict-QC sensitivity | 7,670 | warning이 없는 사후 민감도 subset |
| 준비 synthetic | 123,232 | 2 scheme x 2 method의 outer-train 전용 생성 행 |
| V1 primary outer group | 5 | segment 36--40 전체 holdout |
| V1 secondary outer group | 5 | deterministic target group 전체 holdout |
| robust-v2 inner context/후보 | 20 | ordered `(outer, validation)` segment pair |
| robust-v2 seeded fit/후보 | 60 | 20 context x 3 paired seed |

V1은 nested grouped evaluation을 사용한다. active context의 실측 train
row만 scaler, target transform, hyperparameter selection, synthetic parent
search, C-Mixup bandwidth에 사용할 수 있다. validation/test row는 항상
실측이며 synthetic parent가 될 수 없다.

primary metric은 held-out group마다 계산한

```text
RMSE3D = sqrt(mean(dx^2 + dy^2 + dz^2))
```

를 다섯 group에 같은 가중치로 평균한 macro RMSE3D다. pooled metric은
7,702개 OOF row를 같은 가중치로 계산하므로 group 크기가 다른 경우 macro와
같지 않다. P95도 macro와 pooled를 섞어 비교하면 안 된다.

## 성능 수치의 증거 수준

| 결과 | 선택에 사용한 evidence | RMSE3D / P95 (mm) | 증거 수준 |
| --- | --- | ---: | --- |
| frozen V1 scikit-learn MLP, segment | nested outer segment-LOSO macro | `14.2628 / 21.0190` | 동일 corpus exploratory OOF |
| frozen V1 scikit-learn MLP, target | nested outer target-group macro | `9.3298 / 15.7989` | secondary 동일 corpus OOF |
| GPU-v1 PyTorch MLP, segment | nested outer segment-LOSO macro | `14.9272 / 21.2712` | 동일 corpus exploratory OOF |
| GPU-v1 PyTorch MLP, target | nested outer target-group macro | `9.5700 / 17.2743` | secondary 동일 corpus OOF |
| robust-v2 balanced ERM | 20 inner context의 `J` | `J=16.036852832` | post-v1 development selection |
| robust-v2 balanced ERM, segment | 이미 관측한 outer fold macro | `14.2357 / 20.7053` | 재사용 개발 진단, 독립 test 아님 |

frozen V1, GPU-v1, robust-v2의 MLP 구현과 selection objective가 완전히
같지는 않다. lineage 표는 공통 corpus, feature, outer group, metric에 맞춘
descriptive comparison일 뿐, selection protocol이나 estimator가 일치하는
단일 변수 인과 실험이 아니다. robust-v2의 promotion은 outer 수치가 아니라
20개 inner context에서 계산한 `J`로 결정됐다.

## 증강 판단 근거

gate 구현은 각 모델의 같은 `fusion + real_only` cell을 reference로 삼는다.

```text
required_reduction = max(0.10 * reference_macro_RMSE3D, 3.0 mm)
rmse_pass = reference_macro_RMSE3D - candidate_macro_RMSE3D
            >= required_reduction
p95_pass = candidate_macro_P95 <= 1.05 * reference_macro_P95
passed = rmse_pass and p95_pass
```

8개 reference RMSE가 모두 30 mm 미만이어서 실제 요구 개선량은 전부
`3.0 mm`였다. 최대 point improvement는 XGBoost + C-Mixup의
`0.502818 mm`; MLP + C-Mixup은 `0.416820 mm`, 즉 약 `2.92%` 개선이었다.
두 값 모두 3 mm에 못 미쳤다. 모든 후보가 P95 제한은 지켰다.

MLP + C-Mixup의 primary paired crossed-factor bootstrap은
candidate-minus-reference RMSE delta `-0.321016 mm`, 95% interval
`[-1.101455, +0.474354] mm`였다. 0을 포함하므로 이 corpus에서도 개선
방향이 안정적이라고 단정할 수 없다. gate 표의 단순 macro 차이
`-0.416820 mm`와 paired 표의 observed delta `-0.321016 mm`는 각각
equal-group macro와 7,702-row pooled OOF RMSE 차이다. 뒤 값의 interval을
crossed-factor bootstrap으로 구했다. 집계 방식이 달라 생기는 차이이며
서로 모순이 아니다.

gate와 결과 파일은 Git에서 같은 최초 commit `0acabbb`에 추가됐다.
저장소는 이를 frozen/pre-fixed rule로 기록하지만, Git 이력만으로 결과 생성
전에 외부에 사전등록됐다는 시점을 독립 입증할 수는 없다. 대외 설명에서는
“고정 engineering gate”가 가장 보수적이고 검증 가능한 표현이다.

## 재현성과 semantic validation

- endpoint V1은 10,000-replicate deterministic crossed
  segment/target bootstrap을 기록하며 superiority p-value를 보고하지 않는다.
- endpoint V1 release는 310 outer task, 477,524 ensemble OOF prediction,
  662,372 seed-level OOF prediction과 leakage audit을 포함한다.
- 동일 code/config/release로 수행한 두 complete V1 run의 핵심 evidence
  member hash가 일치했다.
- robust-v2 validator는 120 inner result, 5 reused outer result, 3 final fit과
  17개 semantic gate를 재계산해 `status=ok`를 기록했다.
- robust-v2 후보당 20개 validation context는 최소 10회를 넘지만, 같은
  corpus에서 파생된 상관된 context다. 20개의 독립 수집 session이 아니다.

## 허용되는 주장과 금지되는 주장

허용되는 표현:

- “현재 단일 corpus의 grouped OOF에서 fusion MLP가 static raw-Vive endpoint
  signal을 보였다.”
- “frozen V1에서 2개 증강 recipe x 4개 모델의 8개 gate가 모두 탈락했다.”
- “robust-v2는 20개 inner validation context에서 balanced ERM을 선택했고,
  재사용 outer 개발 진단에서 macro RMSE3D 14.2357 mm를 기록했다.”

근거가 없는 표현:

- “augmentation은 항상 성능을 악화시킨다.”
- “robust-v2에서 augmentation도 시험했고 효과가 없었다.”
- “14 mm는 독립 session, robot-base/TCP 또는 실제 제어 오차다.”
- “모델은 ShapeFK, 정식 `IMU-only`, 동적 추정기 또는 control-ready 모델이다.”
- “final all-data refit의 training/resubstitution error가 generalization
  성능이다.”

추가 실험 수집이 불가능한 현재 조건에서는 새 독립-confirmatory 주장을 만들
수 없다. 가능한 다음 단계는 기존 corpus 안에서 새로운 개발 가설을 명시한
nested ablation을 수행하는 것이며, 그 결과도 독립 검증이 아니라 development
evidence로만 보고해야 한다.

## 관련 근거

- [전체 데이터·모델 비교표](ALL_DATA_COMPARISONS.md)
- [수집 데이터 감사](COLLECTED_DATA_AUDIT.md)
- [데이터 계약](DATA_CONTRACT.md)
- [증강 근거](AUGMENTATION_EVIDENCE.md)
- [endpoint V1 계약](ENDPOINT_REGRESSION_CONTRACT.md)
- [endpoint V1 결과](ENDPOINT_REGRESSION_RESULTS_V1.md)
- [GPU-v1 결과](GPU_ENDPOINT_REGRESSION_RESULTS_V1.md)
- [robust-v2 계약](GPU_ENDPOINT_REGRESSION_ROBUST_V2_CONTRACT.md)
- [robust-v2 결과](GPU_ENDPOINT_REGRESSION_ROBUST_V2_RESULTS.md)
- [고정 endpoint V1 ZIP](../../06_simulation_and_data_collect_preparation/imu_added_experiment/endpoint_regression_experiment_v1.zip)
- [V1 gate 설정](../configs/endpoint_regression_v1.yaml)
- [gate 구현](../src/imu_vive_learning/experiment_models.py)
- [gate 재계산 validator](../src/imu_vive_learning/experiment_validation.py)

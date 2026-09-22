# 데이터 증강 근거와 감사 결과

상태: `alphaf.ml_preparation.v1`과
`alphaf.endpoint_regression.experiment.v1`에 적용된 V1 증강 설계의 근거 및
사후 감사 기록.

이 문서는 논문이 뒷받침하는 일반 원리, 결과 이전에 고정했다고 프로젝트가
기록한 휴리스틱, 생성 후 데이터에서 관측한 값, 학습 결과에 따른 최종
결정을 분리한다. Git 이력만으로 외부 사전등록 시점을 독립 입증하지는
않는다. 논문 인용은 방법군을 정당화하지만, `100 mm`, `20 deg`, 최근접
`5`개, `1:1` 같은 값이 보편적으로 최적이라는 뜻은 아니다.

## 결론부터

| 구분 | 감사 결과 |
|---|---:|
| 구조적으로 보존한 실측 trial | 7,707 |
| 학습 적격 실측 trial | 7,702 |
| strict-QC 실측 trial | 7,670 |
| 준비 ZIP에 materialize한 synthetic row | 123,232 |
| 증강 성능 gate 통과 | 0 / 8 |
| 최종 전체 데이터 모델의 실측 / synthetic | 7,702 / 0 |

123,232개 synthetic row는 fold 안에서 증강 가설을 검증하기 위해 준비한
자료다. 성능이 입증되어 최종 모델에 자동 채택된 자료가 아니다. 고정된
segment-LOSO gate를 통과한 비교가 없었으므로 최종 배포 모델은
`mlp__fusion__real_only`이고 synthetic row를 사용하지 않는다.

## 근거의 네 층위

| 층위 | 의미 | 이 프로젝트에서의 예 |
|---|---|---|
| 연구 문헌 | 일반적인 방법과 위험을 뒷받침 | split 전 oversampling의 누출 위험, 회귀에서 label-local pairing |
| 고정 계약 | 결과 이전 고정으로 프로젝트가 기록한 구현 | 100 mm, 20 deg, `Beta(2,2)`, base seed `20260818` |
| 사후 관측 | 생성된 V1 자료를 감사한 기술 통계 | parent 거리, sigma 범위, coverage replacement 수 |
| 평가 결정 | 잠긴 outer-test 결과로 내린 결론 | 8개 증강 비교 모두 gate 실패, final은 real-only |

사후 관측값은 이미 고정된 반경, sigma 계산법, gate를 소급 조정하는 근거로
사용하지 않았다.

## 데이터 경계와 누출 방지

증강의 입력은 다음 해시로 고정된 준비 계약의 학습 적격 실측행뿐이다.

| 항목 | 고정값 |
|---|---|
| source archive SHA-256 | `ceadf164a15a3ce302f8b2ffdde450df0f6d3a91c3b0c258a28acb5de9b92d64` |
| source member SHA-256 | `1b0a821a1822f237adadb1c978ea73da02b0e4bb7f099ce085a63931e9792b54` |
| preparation ZIP SHA-256 | `94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930` |
| base seed | `20260818` |

Kapoor와 Narayanan의 데이터 누출 분류는 전체 데이터에 전처리나
over/under-sampling을 먼저 적용하거나, 서로 의존하는 행을 독립 행처럼
분할하면 평가가 과도하게 낙관적일 수 있음을 설명한다. Varma와 Simon,
Cawley와 Talbot은 model selection을 포함한 평가에 nested CV가 필요한
이유를 다룬다. Roberts 등은 공간·시간·계층 의존성이 있는 자료에서
의존성 구조에 맞는 block/group 평가를 권고한다.

이에 따라 V1은 다음 순서를 강제한다.

1. `trial_uid`와 segment/target group assignment를 먼저 고정한다.
2. 각 active context의 real-train ID만 parent pool에 넣는다.
3. C-Mixup sigma, scaler, target transform, hyperparameter selection을 모두
   해당 real-train에서만 계산한다.
4. inner context synthetic은 그 context에서 다시 생성한다. 준비 ZIP에
   materialize된 synthetic은 outer refit에만 사용한다.
5. validation과 test에는 원본 실측행만 둔다.
6. parent ID와 split context를 기록해 test ID가 parent나 fit trace에
   등장하면 validator가 실패하게 한다.

행 무작위 분할이나 split 전 증강은 V1의 허용 경로가 아니다.

## 공통 혼합 규칙

`mix_lambda`는 parent A의 가중치다. 선형 채널 `v`와 raw Vive label `y`는
같은 lambda로 다음처럼 만든다.

```text
v_syn = lambda * v_a + (1 - lambda) * v_b
y_syn = lambda * y_a + (1 - lambda) * y_b
```

원래 mixup 연구가 제시한 핵심은 입력과 label의 convex combination이며,
C-Mixup은 회귀에서 label이 가까운 parent를 더 높은 확률로 고르는 방향을
추가한다. V1은 이 원리를 static trial summary에 맞춰 적용했다.

| 필드 종류 | 합성 방법 |
|---|---|
| encoder delta, acceleration, magnetometer, gyro RMS | 선형 보간 |
| planned XYZ | parent 제약 감사 및 diagnostic 보존용 선형 보간 |
| raw `vive_world` XYZ mm label | 같은 lambda로 선형 보간 |
| Euler scalar | `[-pi,pi)`의 최단 호로 보간 |
| Euler sin/cos feature | 보간한 각도에서 다시 생성 |
| ID, timestamp, sample count, QC flag, fold role | 보간하거나 실제 측정값처럼 만들지 않음 |

Euler 보간은 다음 계약을 사용한다.

```text
delta = wrap_to_pi(angle_b - angle_a)
angle_syn = wrap_to_pi(angle_a + (1 - lambda) * delta)
```

따라서 `179 deg`와 `-179 deg` 사이의 경로는 0도가 아니라 180도 부근을
지난다. WT901C Euler 축 convention과 quaternion 변환이 실측으로 검증되지
않았으므로 임의 3D 회전 증강이나 Euler-to-quaternion 재해석은 하지 않았다.

## `local_interp_v2`

이 방법은 이 저장소의 국소 manifold 가설을 시험하는 결정론적 baseline이다.
별도 논문 알고리즘을 그대로 재현했다는 주장이 아니다.

1. parent는 같은 `(segment_id, set_id, repeat_index)`에 있어야 한다.
2. planned 거리 `<= 100 mm`인 후보 중 거리·ID 순서로 최근접 5개를 만든다.
3. stable anchor ID에서 유도한 seed로 parent B를 고른다.
4. lambda는 `{0.25, 0.5, 0.75}`에서 결정론적으로 고른다.
5. self-pair와 active real-train 밖의 parent를 금지한다.

이 가설은 planned 공간의 국소성이 센서와 raw Vive endpoint의 유효한
보간 구간을 제공할 수 있는지를 묻는다. 아래 사후 관측처럼 planned
국소성이 label 국소성을 보장하지 않았으며, 이 차이는 성능 gate 실패와
함께 그대로 공개한다.

## `cmixup_offline_v1`

이 방법은 Yao 등의 C-Mixup에서 회귀 label-local pairing 원리를 가져온
오프라인 프로젝트 적응판이다. 논문의 공식 구현과 동일한 benchmark
재현이라고 부르지 않는다.

1. 후보 pool은 active context의 학습 적격 real-train뿐이다.
2. planned parent 거리 `<= 100 mm`, encoder-derived 최대 관절 차
   `<= 20 deg`를 동시에 만족해야 한다.
3. self-pair는 금지한다.
4. raw Vive label 거리 `d_ij`에 대해 후보 가중치를 다음처럼 계산한다.

```text
w_ij = exp(-(d_ij ** 2) / (2 * sigma ** 2))
```

5. sigma는 해당 real-train에서 각 anchor의 최근접 admissible label 거리를
   구한 뒤 그 중앙값으로 정한다. 수치 floor도 context 안에서 기록한다.
6. `lambda ~ Beta(2,2)`이며 feature와 label에 같은 lambda를 쓴다.
7. seed는 base seed와 stable context text/hash로 결정론적으로 유도한다.
   preparation outer materialization은 base seed에서 row seed를 직접 만들고,
   experiment inner/final 재생성은 먼저 contract/context seed를 만든 뒤
   anchor/replica별 row seed를 다시 파생하는 2단계 구조다. Python의
   process-randomized `hash()`는 사용하지 않는다.

planned/joint gate와 train-fold sigma 규칙은 데이터의 물리적 support를
보수적으로 제한하기 위한 이 프로젝트의 추가 조건이다. C-Mixup 논문이
100 mm, 20도, 중앙값 sigma를 보편 상수로 제안한 것은 아니다.

## 프로젝트가 결과 이전 고정으로 기록한 휴리스틱

| 값 | 쓰임 | 근거의 성격 |
|---|---|---|
| planned 반경 `100 mm` | 두 증강의 support 제한 | 프로젝트 고정 휴리스틱 |
| 최대 관절 차 `20 deg` | C-Mixup의 encoder-space 제한 | 프로젝트 고정 휴리스틱 |
| 최근접 `5`개 | local 후보 집합 | 프로젝트 고정 휴리스틱 |
| local lambda `{.25,.5,.75}` | endpoint에 치우치지 않은 고정 혼합 강도 | 프로젝트 고정 휴리스틱 |
| C-Mixup `Beta(2,2)` | 중간 혼합에 질량을 둔 convex mix | 방법군에 부합하는 프로젝트 선택 |
| sigma = 최근접 admissible label 거리 중앙값 | context별 label scale | 프로젝트 적응 규칙 |
| synthetic:real `1:1` | 비교 가능한 고정 augmentation exposure | 실험 설계 선택 |
| base seed `20260818` | 재현 가능한 parent/lambda 선택 | 동결된 저장소 설정 |

이 값들은 결과를 본 뒤 성능이 좋아지는 방향으로 튜닝하지 않았다.

## coverage replacement

어떤 anchor는 동결된 제약 안에 합법적인 partner가 없다. 이때 거리가 먼
partner를 허용하면 방법 자체가 바뀌고, 행을 줄이면 fold별 1:1 계약이
깨진다. V1은 다음 결정론적 schedule을 쓴다.

1. 합법적 partner가 있는 anchor를 한 번씩 생성한다.
2. partner가 없는 원래 anchor 수를 센다.
3. 합법적 anchor에서 그 수만큼 복원추출해 빈 slot을 채운다.
4. `replacement_for_missing_anchor=true`와 증가한
   `anchor_replica_index`를 기록한다.

replacement는 partner가 없던 원래 anchor를 사칭하지 않는다. 기존 group,
거리, real-train 제약도 완화하지 않는다. fold 전체에 합법적 pair가 없으면
생성에 실패한다.

## 생성 데이터 사후 감사

다음 값은 패키지 생성 후 provenance 123,232행을 전수 집계한 관측치다.
휴리스틱을 소급 정당화하거나 바꾸기 위한 값이 아니다.

| 방법 | 행 | planned 거리 median / P95 / max (mm) | label 거리 median / P95 / max (mm) |
|---|---:|---:|---:|
| `local_interp_v2` | 61,616 | 67.652 / 95.933 / 99.608 | 79.574 / 153.879 / 239.647 |
| `cmixup_offline_v1` | 61,616 | 0.000 / 2.417 / 98.333 | 3.746 / 10.715 / 102.010 |

C-Mixup의 최대 관절 차는 median `0.439 deg`, P95 `2.812 deg`, max
`19.951 deg`였고, fold별 sigma 범위는 `2.456-2.713 mm`였다. 모든 방법에서
self-pair는 0건이었다.

| scheme | fold | local missing/replacement | C-Mixup missing/replacement |
|---|---:|---:|---:|
| segment-LOSO | 36 | 27 | 0 |
| segment-LOSO | 37 | 28 | 0 |
| segment-LOSO | 38 | 23 | 0 |
| segment-LOSO | 39 | 24 | 0 |
| segment-LOSO | 40 | 10 | 0 |
| target-group | 0 | 177 | 0 |
| target-group | 1 | 29 | 0 |
| target-group | 2 | 37 | 0 |
| target-group | 3 | 204 | 0 |
| target-group | 4 | 187 | 0 |
| **합계** |  | **746** | **0** |

local replacement 비율은 segment 쪽 `112 / 30,808 = 0.364%`, target 쪽
`634 / 30,808 = 2.058%`다. replacement 행도 원래 제약을 통과한 parent만
사용한다.

## 학습 평가와 최종 채택

증강 통과 조건은 같은 모델의 fusion real-only 대비 segment-LOSO macro
`RMSE_3D`를 `max(10%, 3 mm)` 이상 줄이고, macro P95를 5% 넘게 악화시키지
않는 것이다. 아래 `RMSE 감소`는 양수일수록 개선이다.

여덟 행은 증강 8종이 아니라 2개 recipe와 4개 model family의 교차다.
RMSE point estimate는 4개가 개선되고 4개가 악화됐다.

| 모델 | 증강 | baseline / candidate RMSE3D (mm) | RMSE 감소 (mm) | P95 조건 | 최종 gate |
|---|---|---:|---:|---|---|
| ExtraTrees | C-Mixup | 17.002 / 17.031 | -0.029 | 통과 | 실패 |
| ExtraTrees | local | 17.002 / 17.240 | -0.239 | 통과 | 실패 |
| MLP | C-Mixup | 14.263 / 13.846 | 0.417 | 통과 | 실패 |
| MLP | local | 14.263 / 14.860 | -0.598 | 통과 | 실패 |
| Ridge | C-Mixup | 26.286 / 26.277 | 0.009 | 통과 | 실패 |
| Ridge | local | 26.286 / 26.591 | -0.305 | 통과 | 실패 |
| XGBoost | C-Mixup | 19.262 / 18.760 | 0.503 | 통과 | 실패 |
| XGBoost | local | 19.262 / 19.140 | 0.123 | 통과 | 실패 |

P95 제한은 모두 지켰지만 요구된 3 mm RMSE 개선을 만족한 비교가 없다.
따라서 증강 gate 통과는 0/8이고, 최종 full-data model은 실측 7,702행만
사용한다. 이는 “증강 코드가 유효하지 않다”는 보편 결론이 아니라, 이
session·feature·고정 설정에서 최종 모델 채택 근거가 부족하다는 결론이다.

## feature와 weight 해석

- primary `fusion_v1`에는 encoder delta 3개와 4-IMU endpoint feature
  40개만 들어간다.
- raw Vive 값은 label 전용이다.
- planned 좌표는 parent support gate와 shortcut diagnostic에만 쓰며
  primary feature가 아니다.
- 미보정 magnetometer는 diagnostic에만 쓰며 최종 후보가 될 수 없다.
- 준비 ZIP synthetic provenance의 `sample_weight=1.0`은 materialized row의
  기본값이다. 학습 recipe에서는 real과 추가 행을 각각 `0.5`로 다시
  가중해 두 집단의 objective mass를 1:1로 유지한다.
- validation/test에는 synthetic row가 0개다.

## 검증 가능한 provenance

각 synthetic row는 최소한 다음 감사 정보를 가진다.

```text
synthetic_id, augmentation_method, split_scheme, outer_fold, inner_fold,
split_context, anchor_trial_uid, anchor_replica_index,
replacement_for_missing_anchor, parent_a_trial_uid, parent_b_trial_uid,
mix_lambda, planned_parent_distance_mm, label_parent_distance_mm,
joint_parent_distance_deg, cmixup_sigma_mm, derived_seed,
source_hash, config_hash, sample_weight
```

`joint_parent_distance_deg` 열은 C-Mixup의 joint gate에 적용된다.
preparation의 local 행은 joint gate를 쓰지 않으므로 이 값이 `NaN`이며,
experiment provenance는 실제 encoder parent 차이를 별도 열로 재계산해
감사하고 generator 원값도 보존한다.

release validator는 parent가 active train context 안에 있는지, self-pair,
중복 ID, non-finite, convex bound 위반, 물리 제약 위반, synthetic의
validation/test 혼입, replacement provenance 불일치, checksum drift를
전수 검사한다. 준비 ZIP validator는 canonical 7,707, eligible 7,702,
assignment 231,060, augmentation file 20개, synthetic 123,232와 manifest
member hash를 재검증한다.

endpoint run validator는 50개 split context x 2개 method인 100개
context/recipe 조합의 measured real-train을 production generator에 다시
넣어 492,928행의 `synthetic_id`, derived seed,
parent A/B, lambda, replica/replacement schedule, 거리, sigma, source/config
hash와 weight를 exact 비교한다. 따라서 거리 제약만 우연히 만족하는 다른
parent나 lambda로 provenance를 바꾸어도 통과하지 못한다. future final
augmentation 경로에도 같은 replay를 적용해 local top-5와 C-Mixup의 seeded
draw를 확인한다. 이 전수 replay는 CPU-bound이며 감사 환경의 완료 run에서
약 351초가 걸렸으므로 검증 명령은 수 분이 걸릴 수 있다.

## 적용하지 않은 증강

무작위 3D 회전, 축 재배치, time warp, crop, raw-sample jitter는 V1에
적용하지 않았다. 현재 계약은 3초 static summary이고 `timeseries.parquet`을
입력으로 허용하지 않으며, WT901C Euler 축 convention, full-scale,
tracker-to-TCP 외부 calibration이 검증되지 않았다. 따라서 이 결정은
현재 데이터 계약에 대한 안전 경계이며, 시계열 증강 일반론을 부정하는
주장이 아니다.

## 연구 출처

- Zhang et al., “mixup: Beyond Empirical Risk Minimization,” ICLR 2018:
  [arXiv](https://arxiv.org/abs/1710.09412),
  [OpenReview](https://openreview.net/forum?id=r1Ddp1-Rb).
- Yao et al., “C-Mixup: Improving Generalization in Regression,” NeurIPS
  2022: [논문과 supplement](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html),
  [공식 구현](https://github.com/huaxiuyao/C-Mixup).
- Kapoor and Narayanan, “Leakage and the reproducibility crisis in
  machine-learning-based science,” *Patterns* 2023:
  [DOI](https://doi.org/10.1016/j.patter.2023.100804),
  [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/).
- Varma and Simon, “Bias in error estimation when using cross-validation for
  model selection,” *BMC Bioinformatics* 2006:
  [DOI](https://doi.org/10.1186/1471-2105-7-91).
- Cawley and Talbot, “On Over-fitting in Model Selection and Subsequent
  Selection Bias in Performance Evaluation,” *JMLR* 2010:
  [full text](https://www.jmlr.org/papers/v11/cawley10a.html).
- Roberts et al., “Cross-validation strategies for data with temporal,
  spatial, hierarchical, or phylogenetic structure,” *Ecography* 2017:
  [DOI](https://doi.org/10.1111/ecog.02881).

정확한 필드와 split 계약은 [DATA_CONTRACT.md](DATA_CONTRACT.md), 패키지
경로와 join 규칙은 [DATA_PACKAGE_STRUCTURE.md](DATA_PACKAGE_STRUCTURE.md),
최종 모델 결과는
[ENDPOINT_REGRESSION_RESULTS_V1.md](ENDPOINT_REGRESSION_RESULTS_V1.md)를
따른다.

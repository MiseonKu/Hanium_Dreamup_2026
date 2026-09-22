# 전체 데이터·모델 비교표

상태: 2026-08-29 기준 성능 evidence snapshot.

이 문서의 “전체”는 현재 endpoint-regression 의사결정에 사용되거나 그 결정을
진단하는 다음 비교 집합을 뜻한다.

1. endpoint V1 `aggregate_metrics.parquet`의 62개 aggregate cell 전체;
2. 그중 primary augmentation gate 8개 전체;
3. `paired_comparisons.parquet`의 24개 비교 전체;
4. GPU-v1의 10개 outer fold에서 MLP/XGBoost 비교;
5. robust-v2의 6개 후보와 selected candidate의 5개 reused outer fold.

477,524개 ensemble OOF row와 662,372개 seed-level OOF row를 Markdown에
복제하지는 않는다. 그 행들은 추적된
[endpoint_regression_experiment_v1.zip](../../06_simulation_and_data_collect_preparation/imu_added_experiment/endpoint_regression_experiment_v1.zip)
안에 있고, 여기서는 모든 aggregate
비교와 의사결정 수치를 브라우저에서 읽을 수 있게 공개한다. 해시와 해석
경계는 [PERFORMANCE_EVIDENCE_AUDIT.md](PERFORMANCE_EVIDENCE_AUDIT.md)를
따른다.

이 범위에서 제외한 것은 성능 비교가 아닌 row-level prediction 복제,
all-data training/resubstitution error, legacy split-before-augmentation
archive, 아직 완료 결과가 아닌 outage/OpenAXES 개발 contract, 그리고
deployment runtime/safety 검증이다. 제외 항목을 모델 generalization 근거로
섞지 않기 위한 경계다.

## 1. corpus 비교

| Segment | Canonical real | Train eligible | 제외 |
| ---: | ---: | ---: | ---: |
| 36 | 1,672 | 1,672 | 0 |
| 37 | 1,597 | 1,597 | 0 |
| 38 | 1,707 | 1,707 | 0 |
| 39 | 1,659 | 1,659 | 0 |
| 40 | 1,072 | 1,067 | 5 |
| **합계** | **7,707** | **7,702** | **5** |

strict-QC sensitivity는 7,670행이다. 준비 package에는 123,232개 synthetic
row가 있지만 frozen V1 final, GPU-v1 final, robust-v2 final은 모두 7,702개
실측 row와 0개 synthetic row로 학습됐다.

## 2. endpoint V1 aggregate 62개 cell

아래 네 metric은 모두 같은 OOF prediction에서 계산된다. `Macro`는 다섯
held-out group에 같은 가중치를 주고, `Pooled`는 OOF row에 같은 가중치를
준다. 단위는 모두 mm다.

### 2.1 segment-LOSO real-only 19개

| Model | Feature | 용도 | Macro RMSE | Macro P95 | Pooled RMSE | Pooled P95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Dummy | none | baseline | 229.572952 | 351.256487 | 229.670260 | 352.262764 |
| ExtraTrees | encoder | candidate | 27.563107 | 45.528439 | 28.593049 | 47.819076 |
| ExtraTrees | four_imu | candidate | 17.051021 | 25.727092 | 17.487490 | 28.904775 |
| ExtraTrees | fusion | candidate | 17.001614 | 25.556983 | 17.442064 | 29.381527 |
| ExtraTrees | fusion_mag | diagnostic | 17.078086 | 25.589023 | 17.556012 | 29.297407 |
| ExtraTrees | fusion_planned | diagnostic | 17.126432 | 25.625817 | 17.627479 | 29.849069 |
| ExtraTrees | planned_only | diagnostic | 30.997059 | 51.852136 | 31.794636 | 51.747984 |
| MLP | encoder | candidate | 29.022608 | 47.875949 | 29.923083 | 48.066467 |
| MLP | four_imu | candidate | 14.964485 | 21.727141 | 15.153546 | 22.992283 |
| MLP | fusion | candidate | **14.262783** | **21.018984** | **14.466592** | **22.135416** |
| Ridge | encoder | candidate | 70.502787 | 115.849948 | 70.751378 | 118.747555 |
| Ridge | four_imu | candidate | 27.033268 | 46.063504 | 26.960102 | 47.300448 |
| Ridge | fusion | candidate | 26.285685 | 44.540184 | 26.386242 | 46.030164 |
| Ridge | fusion_mag | diagnostic | 18.328218 | 29.513671 | 18.342263 | 30.883479 |
| Ridge | fusion_planned | diagnostic | 25.146423 | 42.336003 | 25.301983 | 43.941217 |
| Ridge | planned_only | diagnostic | 81.865980 | 144.791805 | 81.985557 | 145.026938 |
| XGBoost | encoder | candidate | 28.439523 | 47.551589 | 29.634445 | 50.595495 |
| XGBoost | four_imu | candidate | 19.773740 | 29.789374 | 20.458483 | 32.451234 |
| XGBoost | fusion | candidate | 19.262477 | 28.461241 | 19.942370 | 32.866132 |

`fusion_mag`, `fusion_planned`, `planned_only`은 shortcut diagnostic이며 final
selection 대상이 아니다.

### 2.2 target-group real-only 19개

| Model | Feature | 용도 | Macro RMSE | Macro P95 | Pooled RMSE | Pooled P95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Dummy | none | baseline | 232.415979 | 354.137364 | 232.720569 | 358.648513 |
| ExtraTrees | encoder | candidate | 53.688375 | 93.607916 | 54.191392 | 95.480064 |
| ExtraTrees | four_imu | candidate | 39.263193 | 70.329922 | 39.581651 | 71.614781 |
| ExtraTrees | fusion | candidate | 38.927463 | 69.785550 | 39.228450 | 72.805799 |
| ExtraTrees | fusion_mag | diagnostic | 38.285806 | 70.077101 | 38.612851 | 72.811338 |
| ExtraTrees | fusion_planned | diagnostic | 38.161571 | 66.776171 | 38.340275 | 67.709898 |
| ExtraTrees | planned_only | diagnostic | 55.969462 | 98.335476 | 56.180983 | 97.497431 |
| MLP | encoder | candidate | 29.225439 | 47.504610 | 29.236668 | 48.222465 |
| MLP | four_imu | candidate | 10.859036 | 18.441996 | 11.367868 | 20.137707 |
| MLP | fusion | candidate | **9.329850** | **15.798926** | **9.623064** | **16.925551** |
| Ridge | encoder | candidate | 71.249629 | 120.407528 | 71.305321 | 122.237198 |
| Ridge | four_imu | candidate | 26.092834 | 47.717798 | 26.279985 | 48.107384 |
| Ridge | fusion | candidate | 26.736263 | 48.682113 | 27.091598 | 49.008996 |
| Ridge | fusion_mag | diagnostic | 18.060369 | 30.984221 | 18.358447 | 31.854716 |
| Ridge | fusion_planned | diagnostic | 25.038924 | 43.097237 | 25.417912 | 43.165855 |
| Ridge | planned_only | diagnostic | 86.115926 | 158.094477 | 87.078675 | 147.783116 |
| XGBoost | encoder | candidate | 71.654892 | 126.907934 | 72.078842 | 136.997111 |
| XGBoost | four_imu | candidate | 23.842007 | 40.897355 | 23.908004 | 43.138637 |
| XGBoost | fusion | candidate | 26.030664 | 44.500531 | 26.224783 | 48.442386 |

### 2.3 fusion augmentation/replay 24개

| Scheme | Model | Recipe | Macro RMSE | Macro P95 | Pooled RMSE | Pooled P95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| segment | ExtraTrees | C-Mixup | 17.030657 | 25.535524 | 17.481657 | 29.297221 |
| segment | ExtraTrees | local | 17.240415 | 25.984626 | 17.652646 | 29.346081 |
| segment | ExtraTrees | replay | 17.026099 | 25.599189 | 17.466482 | 29.059167 |
| segment | MLP | C-Mixup | **13.845963** | **20.050818** | **14.145576** | **22.034740** |
| segment | MLP | local | 14.860498 | 21.883961 | 15.095511 | 22.766342 |
| segment | MLP | replay | 14.820040 | 21.255784 | 15.160298 | 23.310068 |
| segment | Ridge | C-Mixup | 26.277100 | 44.503698 | 26.362524 | 46.019953 |
| segment | Ridge | local | 26.590553 | 44.681494 | 26.652514 | 46.423698 |
| segment | Ridge | replay | 26.285685 | 44.540184 | 26.386242 | 46.030164 |
| segment | XGBoost | C-Mixup | 18.759659 | 28.233985 | 19.446250 | 32.284091 |
| segment | XGBoost | local | 19.139597 | 28.478408 | 19.803910 | 32.315273 |
| segment | XGBoost | replay | 19.293901 | 28.717251 | 19.996995 | 33.037148 |
| target | ExtraTrees | C-Mixup | 38.555191 | 69.145001 | 38.776945 | 69.778901 |
| target | ExtraTrees | local | 36.003463 | 65.876416 | 36.258060 | 67.911845 |
| target | ExtraTrees | replay | 38.321182 | 68.676169 | 38.588701 | 70.086014 |
| target | MLP | C-Mixup | 9.207834 | 15.934888 | 9.487061 | 15.879283 |
| target | MLP | local | 8.839605 | 14.704266 | 9.096558 | 14.973728 |
| target | MLP | replay | **8.733167** | **14.319478** | **9.037895** | 15.599469 |
| target | Ridge | C-Mixup | 26.534601 | 48.199118 | 26.908025 | 48.389563 |
| target | Ridge | local | 27.282395 | 49.146144 | 27.641027 | 49.653186 |
| target | Ridge | replay | 26.736263 | 48.682113 | 27.091598 | 49.008996 |
| target | XGBoost | C-Mixup | 25.819490 | 42.987256 | 25.894097 | 45.711971 |
| target | XGBoost | local | 18.904374 | 34.233026 | 19.075588 | 35.549243 |
| target | XGBoost | replay | 26.173850 | 45.269920 | 26.420971 | 49.129996 |

replay는 합성법 채택 후보가 아니라 row exposure/update-budget의 negative
control이다. target scheme은 secondary이므로 target에서 수치가 좋아도 primary
segment gate를 대체하지 않는다.

## 3. primary augmentation gate 8개

`RMSE 개선 = real-only macro RMSE - candidate macro RMSE`이므로 양수가
좋다. 모든 비교에서 요구 개선량은 3.0 mm였다.

| Model | Recipe | Real -> candidate RMSE | 개선 | Real -> candidate P95 | P95 한계 | 판정 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ExtraTrees | C-Mixup | 17.001614 -> 17.030657 | -0.029043 | 25.556983 -> 25.535524 | 26.834832 | RMSE fail, P95 pass |
| ExtraTrees | local | 17.001614 -> 17.240415 | -0.238801 | 25.556983 -> 25.984626 | 26.834832 | RMSE fail, P95 pass |
| MLP | C-Mixup | 14.262783 -> 13.845963 | +0.416820 | 21.018984 -> 20.050818 | 22.069933 | RMSE fail, P95 pass |
| MLP | local | 14.262783 -> 14.860498 | -0.597715 | 21.018984 -> 21.883961 | 22.069933 | RMSE fail, P95 pass |
| Ridge | C-Mixup | 26.285685 -> 26.277100 | +0.008586 | 44.540184 -> 44.503698 | 46.767193 | RMSE fail, P95 pass |
| Ridge | local | 26.285685 -> 26.590553 | -0.304868 | 44.540184 -> 44.681494 | 46.767193 | RMSE fail, P95 pass |
| XGBoost | C-Mixup | 19.262477 -> 18.759659 | +0.502818 | 28.461241 -> 28.233985 | 29.884303 | RMSE fail, P95 pass |
| XGBoost | local | 19.262477 -> 19.139597 | +0.122880 | 28.461241 -> 28.478408 | 29.884303 | RMSE fail, P95 pass |

요약은 RMSE 개선 4개, 악화 4개, P95 pass 8개, 최종 gate pass 0개다.

## 4. paired crossed-factor bootstrap 24개

delta는 모두 `candidate - real_only`다. 음수가 candidate에 유리하다. 각
observed delta는 7,702-row pooled OOF metric의 차이고, 각 대괄호는
deterministic 10,000-replicate crossed-factor 95% bootstrap interval이다.
따라서 이 표의 observed delta와 section 3의 equal-group macro gate 차이는
같은 값일 필요가 없다.

| Scheme | Model | Recipe | Delta RMSE [95% CI] | Delta P95 [95% CI] |
| --- | --- | --- | --- | --- |
| segment | ExtraTrees | C-Mixup | +0.039594 [-0.082659, +0.156988] | -0.084306 [-0.332495, +0.612124] |
| segment | ExtraTrees | local | +0.210582 [-0.040087, +0.466615] | -0.035446 [-0.605392, +1.084876] |
| segment | ExtraTrees | replay | +0.024419 [-0.042303, +0.094345] | -0.322360 [-0.471863, +0.384104] |
| segment | MLP | C-Mixup | -0.321016 [-1.101455, +0.474354] | -0.100676 [-2.106942, +1.333220] |
| segment | MLP | local | +0.628919 [-0.390480, +1.561187] | +0.630925 [-1.345685, +2.233882] |
| segment | MLP | replay | +0.693705 [-0.550784, +1.961879] | +1.174651 [-1.543903, +2.958195] |
| segment | Ridge | C-Mixup | -0.023718 [-0.217248, +0.223349] | -0.010211 [-0.578200, +0.671303] |
| segment | Ridge | local | +0.266272 [-0.183131, +0.832335] | +0.393535 [-0.684741, +1.769358] |
| segment | Ridge | replay | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| segment | XGBoost | C-Mixup | -0.496120 [-1.094101, -0.046568] | -0.582042 [-1.429884, +0.834613] |
| segment | XGBoost | local | -0.138461 [-0.609964, +0.381468] | -0.550860 [-1.676353, +0.885675] |
| segment | XGBoost | replay | +0.054625 [-0.133363, +0.240507] | +0.171015 [-0.303148, +0.820107] |
| target | ExtraTrees | C-Mixup | -0.451505 [-1.910842, +0.528579] | -3.026898 [-9.411892, +1.233638] |
| target | ExtraTrees | local | -2.970390 [-5.679400, -0.493255] | -4.893954 [-17.128274, +2.977716] |
| target | ExtraTrees | replay | -0.639750 [-2.087772, +0.375564] | -2.719785 [-8.899954, +1.529135] |
| target | MLP | C-Mixup | -0.136003 [-0.950383, +0.539135] | -1.046267 [-4.063008, +2.894718] |
| target | MLP | local | -0.526506 [-1.607346, +0.376734] | -1.951823 [-5.245602, +1.300419] |
| target | MLP | replay | -0.585169 [-1.298121, -0.010564] | -1.326082 [-4.403261, -0.150215] |
| target | Ridge | C-Mixup | -0.183573 [-0.349883, -0.001515] | -0.619434 [-1.067032, +0.268741] |
| target | Ridge | local | +0.549429 [-0.091463, +1.175026] | +0.644189 [-2.118711, +3.068683] |
| target | Ridge | replay | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |
| target | XGBoost | C-Mixup | -0.330686 [-2.288213, +1.202247] | -2.730416 [-9.009484, +2.834590] |
| target | XGBoost | local | -7.149195 [-9.997595, -4.560952] | -12.893143 [-20.186936, -4.147036] |
| target | XGBoost | replay | +0.196188 [-0.303184, +0.710669] | +0.687610 [-1.174515, +3.172269] |

일부 interval이 0을 포함하지 않아도 그것만으로 final 채택이 되지는 않는다.
primary/secondary 역할, 고정 magnitude gate, P95 조건, recipe의 deployment
eligibility를 모두 따라야 한다. p-value는 보고하지 않았다.

## 5. GPU-v1 outer fold 전체

### 5.1 primary inner configuration 4개

| Hidden layers | Weight decay | Inner RMSE | Inner P95 | Max context RMSE | Selected |
| --- | ---: | ---: | ---: | ---: | --- |
| 64, 64 | 0.001 | 14.3288 | 20.7665 | 26.8150 | No |
| 64, 64 | 0.0001 | **14.2315** | **20.7107** | 26.8900 | **Yes** |
| 128, 64 | 0.001 | 14.6255 | 21.1398 | 26.9943 | No |
| 128, 64 | 0.0001 | 14.5228 | 20.9746 | 26.9043 | No |

### 5.2 outer aggregate 4개

| Scheme | Model | Macro RMSE / P95 | Pooled RMSE / P95 |
| --- | --- | ---: | ---: |
| Segment | PyTorch MLP | **14.9272 / 21.2712** | **15.1201 / 22.7191** |
| Segment | CUDA XGBoost | 19.2915 / 29.0726 | 19.9712 / 33.2668 |
| Target | PyTorch MLP | **9.5700 / 17.2743** | **9.7892 / 17.1599** |
| Target | CUDA XGBoost | 25.4604 / 43.3229 | 25.6419 / 46.5854 |

### 5.3 segment-LOSO fold

| Segment | Rows | MLP RMSE / P95 | XGBoost RMSE / P95 |
| ---: | ---: | ---: | ---: |
| 36 | 1,672 | 18.8190 / 23.9244 | 26.1638 / 40.3961 |
| 37 | 1,597 | 11.8455 / 18.1580 | 17.6396 / 26.4550 |
| 38 | 1,707 | 15.5062 / 23.9668 | 19.3286 / 30.4933 |
| 39 | 1,659 | 12.8541 / 18.8339 | 18.1682 / 25.8157 |
| 40 | 1,067 | 15.6111 / 21.4728 | 15.1573 / 22.2030 |
| **Macro** | **7,702** | **14.9272 / 21.2712** | **19.2915 / 29.0726** |

### 5.4 target-group fold

| Target group | Rows | MLP RMSE / P95 | XGBoost RMSE / P95 |
| ---: | ---: | ---: | ---: |
| 0 | 1,538 | 10.4486 / 24.0304 | 24.7243 / 39.1831 |
| 1 | 1,554 | 7.7332 / 15.9616 | 29.6710 / 54.2749 |
| 2 | 1,553 | 10.8913 / 20.7860 | 23.9065 / 35.8428 |
| 3 | 1,555 | 12.0059 / 12.9929 | 20.7473 / 33.9324 |
| 4 | 1,502 | 6.7711 / 12.6008 | 28.2531 / 53.3816 |
| **Macro** | **7,702** | **9.5700 / 17.2743** | **25.4604 / 43.3229** |

## 6. robust-v2 후보 6개 전체

후보마다 20개 inner validation context와 60개 seeded fit이 있다.
`J=(20-context mean RMSE + worst-validation-segment mean RMSE)/2`이며 낮을수록
좋다.

| Rank | Candidate | J | Mean RMSE | Worst-segment mean | Mean P95 | Mean seed SD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Balanced ERM | **16.036853** | **14.105688** | 17.968018 | **20.617648** | 0.783980 |
| 2 | V-REx beta=10 | 16.125245 | 14.346355 | **17.904136** | 21.143995 | **0.766520** |
| 3 | V-REx beta=100 | 16.216858 | 14.399620 | 18.034095 | 21.090268 | 0.851090 |
| 4 | Group DRO eta=0.01 | 16.474181 | 14.384197 | 18.564164 | 20.933789 | 0.776454 |
| 5 | Group DRO eta=0.1 | 16.491267 | 14.329163 | 18.653370 | 20.626768 | 0.772889 |
| 6 | Balanced ERM + SAM rho=0.05 | 18.420185 | 16.225889 | 20.614481 | 24.275614 | 0.852415 |

### 후보별 validation-segment mean 전체

| Candidate | S36 | S37 | S38 | S39 | S40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Balanced ERM | 17.968018 | 12.751152 | 14.283410 | 12.276677 | 13.249181 |
| Group DRO eta=0.01 | 18.564164 | 12.802273 | 14.492852 | 12.711963 | 13.349735 |
| Group DRO eta=0.1 | 18.653370 | 12.481212 | 14.447107 | 12.414282 | 13.649842 |
| V-REx beta=10 | **17.904136** | 12.901906 | 14.611729 | 12.852020 | 13.461982 |
| V-REx beta=100 | 18.034095 | 12.562074 | 14.567312 | 13.041413 | 13.793208 |
| Balanced ERM + SAM rho=0.05 | 20.614481 | 15.768460 | 16.373036 | 13.898120 | 14.475347 |

### selected balanced ERM의 reused outer 진단

| Segment | Rows | RMSE | P95 | Seed RMSE SD |
| ---: | ---: | ---: | ---: | ---: |
| 36 | 1,672 | 18.7172 | 23.7787 | 1.1446 |
| 37 | 1,597 | 11.0879 | 18.1110 | 0.6802 |
| 38 | 1,707 | 15.0828 | 23.6045 | 1.4609 |
| 39 | 1,659 | 12.0623 | 18.1069 | 0.4527 |
| 40 | 1,067 | 14.2283 | 19.9257 | 0.2561 |
| **Macro** | **7,702** | **14.2357** | **20.7053** | -- |
| **Pooled** | **7,702** | **14.5379** | **21.9799** | -- |

이 outer fold는 GPU-v1 이후 재사용됐으므로 selection이나 독립 test로
사용하지 않았다.

## 7. 공통 경계 기반 descriptive lineage 비교

| Study / model | Segment macro RMSE / P95 | Segment pooled RMSE / P95 | Target macro RMSE / P95 | 비고 |
| --- | ---: | ---: | ---: | --- |
| Frozen V1 scikit MLP real-only | 14.2628 / 21.0190 | 14.4666 / 22.1354 | 9.3298 / 15.7989 | nested OOF |
| GPU-v1 PyTorch MLP real-only | 14.9272 / 21.2712 | 15.1201 / 22.7191 | 9.5700 / 17.2743 | nested OOF |
| Robust-v2 balanced ERM | 14.2357 / 20.7053 | 14.5379 / 21.9799 | -- | reused outer development diagnostic |

세 행은 corpus, 43-feature view, outer segment group, metric을 공유하지만
estimator와 selection protocol은 다르다. 특히 robust-v2 outer는 이미 관측한
fold의 사후 개발 진단이므로 동일한 nested outer OOF 증거 등급이 아니다.

robust-v2는 frozen V1 대비 segment macro RMSE/P95가 point estimate상
`0.0271 / 0.3136 mm` 낮다. GPU-v1 대비로는 `0.6915 / 0.5658 mm` 낮다.
하지만 robust-v2의 outer fold가 이미 관측된 뒤이므로 이 차이에
confirmatory 의미를 부여하면 안 된다.

| Segment delta | Macro RMSE | Macro P95 | Pooled RMSE | Pooled P95 |
| --- | ---: | ---: | ---: | ---: |
| GPU-v1 - frozen V1 | +0.6644 | +0.2522 | +0.6535 | +0.5836 |
| robust-v2 - GPU-v1 | -0.6915 | -0.5658 | -0.5823 | -0.7392 |
| robust-v2 - frozen V1 | -0.0271 | -0.3136 | +0.0713 | -0.1555 |

즉 robust-v2는 GPU-v1보다 낮지만 frozen V1과는 사실상 비슷하다. frozen
V1 대비 macro RMSE는 0.0271 mm 낮고 pooled RMSE는 0.0713 mm 높다.

### segment별 MLP lineage

| Segment | Frozen V1 RMSE / P95 | GPU-v1 RMSE / P95 | Robust-v2 RMSE / P95 |
| ---: | ---: | ---: | ---: |
| 36 | 17.6490 / 22.8116 | 18.8190 / 23.9244 | 18.7172 / 23.7787 |
| 37 | 11.8942 / 19.1453 | 11.8455 / 18.1580 | 11.0879 / 18.1110 |
| 38 | 15.7967 / 24.8669 | 15.5062 / 23.9668 | 15.0828 / 23.6045 |
| 39 | 10.8039 / 16.4729 | 12.8541 / 18.8339 | 12.0623 / 18.1069 |
| 40 | 15.1701 / 21.7982 | 15.6111 / 21.4728 | 14.2283 / 19.9257 |

## 8. strict-QC sensitivity

| Scheme | Full pooled RMSE / P95 | Strict 7,670-row RMSE / P95 |
| --- | ---: | ---: |
| Segment selected V1 | 14.4666 / 22.1354 | 14.3467 / 22.0077 |
| Target selected V1 | 9.6231 / 16.9256 | 9.4324 / 16.6693 |

32개 warning row를 제외해도 변화가 작았다. 이 표는 이미 학습된 OOF
prediction의 subset sensitivity이며, strict-only all-data refit의 training
error를 generalization score로 바꾸지 않는다.

## 9. 원본 표 위치

| 비교 | 원본 |
| --- | --- |
| V1 62 aggregate cell | endpoint V1 ZIP `metrics/aggregate_metrics.parquet` |
| V1 310 fold metric | endpoint V1 ZIP `metrics/fold_metrics.parquet` |
| V1 15,500 cell-balanced metric | endpoint V1 ZIP `metrics/cell_metrics.parquet` |
| V1 24 paired comparison | endpoint V1 ZIP `statistics/paired_comparisons.parquet` |
| V1 8 augmentation gate | endpoint V1 ZIP `statistics/augmentation_gates.parquet` |
| V1 strict sensitivity 62 cell | endpoint V1 ZIP `sensitivity/strict_oof_metrics.parquet` |
| GPU-v1 fold/aggregate | `GPU_ENDPOINT_REGRESSION_RESULTS_V1.md` |
| robust-v2 후보/fold/검증 | `GPU_ENDPOINT_REGRESSION_ROBUST_V2_RESULTS.md` |

이 문서는 결과를 다시 선택하거나 gate를 바꾸지 않는다. 기존 evidence를
GitHub에서 한 번에 검토할 수 있게 정리한 감사 snapshot이다.

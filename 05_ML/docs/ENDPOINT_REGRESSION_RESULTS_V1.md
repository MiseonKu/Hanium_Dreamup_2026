# Endpoint Regression Experiment v1 Results

## Status and interpretation boundary

This is the locked result record for
`alphaf.endpoint_regression.experiment.v1`. The experiment is **exploratory**:
implementation-feasibility smoke work and this complete nested evaluation used
the same acquisition corpus. Nested group cross-validation limits selection
leakage within that corpus, but it does not replace an independently acquired
session.

The response is raw Vive Tracker XYZ in the `vive_world` frame, in millimetres,
from one static three-second trial-summary row. The 43-column fusion view
combines three encoder deltas with four WT901C endpoint-summary feature blocks.
It is a four-IMU endpoint-regression/fusion proxy. It is **not** robot-base or
TCP position, formal ShapeFK, the repository's `IMU-only` condition, a dynamic
or online estimator, a realtime aggregator, or a control model. Vive/Tracker
measurements remain labels and evaluation evidence only; they are not input
features or controller feedback.

## Locked corpus and evaluation

- Parent contract: `alphaf.ml_preparation.v1`
- Source trials audited by the parent contract: `10,055`
- Structurally complete canonical real trials: `7,707`
- Training-eligible measured trials: `7,702`
- Strict-QC measured subset: `7,670`
- Evaluation assignments: `385,100` rows across `50` nested contexts
- Frozen experiment matrix: `31` cells and `310` outer tasks
- Ensemble OOF predictions: `477,524` rows
- Seed-level OOF predictions: `662,372` rows
- Inner-selection trace: `13,520` rows
- Augmentation row provenance: `492,928` rows

The primary scheme holds out each collection segment 36--40 in turn. The
secondary scheme holds out each of five deterministic unseen-target groups.
Each outer training set has its own four grouped inner-validation contexts.
Scalers, target transforms, hyperparameter selection, augmentation parent
search, and C-Mixup bandwidth estimation are fit only on measured real rows in
the active training context. Validation and test roles always contain original
measured rows.

The primary score is
`RMSE_3D = sqrt(mean(dx^2 + dy^2 + dz^2))`, macro-averaged with equal weight
over the five outer groups. The implementation also verifies
`RMSE_3D = sqrt(3) * rmse_xyz_mm`; the maximum observed numerical discrepancy
was `5.68e-14` mm.

The packaged leakage audit found:

- zero duplicate, missing, wrong-ID, or wrong-truth ensemble OOF rows;
- zero test IDs in fitted preprocessors or selection-trace fields;
- zero synthetic validation or test rows;
- zero forbidden primary-feature fields; and
- zero overlap between forbidden IDs and preprocessing fit IDs.

## Main OOF results

The table reports the real-only fusion comparison used to orient the model
families. Dummy is the train-fold target mean. Values are group-macro results,
not the final all-data model's training error.

| Model | Segment-LOSO RMSE_3D (mm) | Segment P95 (mm) | Unseen-target RMSE_3D (mm) | Target P95 (mm) |
| --- | ---: | ---: | ---: | ---: |
| Dummy | 229.5730 | 351.2565 | 232.4160 | 354.1374 |
| Ridge, fusion | 26.2857 | 44.5402 | 26.7363 | 48.6821 |
| ExtraTrees, fusion | 17.0016 | 25.5570 | 38.9275 | 69.7856 |
| XGBoost, fusion | 19.2625 | 28.4612 | 26.0307 | 44.5005 |
| MLP, fusion | **14.2628** | **21.0190** | **9.3298** | **15.7989** |

For the selected MLP real-only cell, pooled segment-LOSO RMSE/P95 were
`14.4666 / 22.1354` mm and pooled unseen-target RMSE/P95 were
`9.6231 / 16.9256` mm. Segment-level RMSE values were `17.6490`, `11.8942`,
`15.7967`, `10.8039`, and `15.1701` mm for segments 36--40, respectively.
The spread is material and is part of the evidence; the macro average must not
be read as uniform behavior across segments.

These results support carrying the MLP fusion candidate into a new-session
validation study. They do not establish general superiority of neural networks,
cross-session accuracy, robot-frame accuracy, or control suitability.

## Augmentation and replay results

The frozen engineering gate required the same model's fusion augmentation to
reduce segment-LOSO macro RMSE by at least `max(10%, 3 mm)` while keeping macro
P95 within 105% of real-only. All eight augmentation comparisons satisfied the
P95 bound, but none satisfied the required RMSE reduction; therefore `0 / 8`
passed.

These are two synthetic recipes crossed with four model families, not eight
distinct augmentation methods. Four RMSE point estimates improved and four
worsened. The `0 / 8` result means that no comparison cleared the fixed
engineering adoption threshold; it does not mean that every augmented point
estimate was worse.

| Model | Recipe | Real-only RMSE (mm) | Candidate RMSE (mm) | Reduction (mm) | Candidate P95 (mm) | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Ridge | C-Mixup | 26.2857 | 26.2771 | 0.0086 | 44.5037 | Fail |
| Ridge | local interpolation | 26.2857 | 26.5906 | -0.3049 | 44.6815 | Fail |
| ExtraTrees | C-Mixup | 17.0016 | 17.0307 | -0.0290 | 25.5355 | Fail |
| ExtraTrees | local interpolation | 17.0016 | 17.2404 | -0.2388 | 25.9846 | Fail |
| XGBoost | C-Mixup | 19.2625 | 18.7597 | 0.5028 | 28.2340 | Fail |
| XGBoost | local interpolation | 19.2625 | 19.1396 | 0.1229 | 28.4784 | Fail |
| MLP | C-Mixup | 14.2628 | 13.8460 | 0.4168 | 20.0508 | Fail |
| MLP | local interpolation | 14.2628 | 14.8605 | -0.5977 | 21.8840 | Fail |

For the MLP fusion model, the four recipes had the following macro RMSE_3D:

| Scheme | Real-only | Replay | Local interpolation | C-Mixup |
| --- | ---: | ---: | ---: | ---: |
| Segment LOSO | 14.2628 | 14.8200 | 14.8605 | 13.8460 |
| Unseen target | 9.3298 | 8.7332 | 8.8396 | 9.2078 |

`replay` is a weighted exposure/update-budget negative control and is not a
deployment candidate. `local_interp_v2` remains the project's planned-manifold
hypothesis; it is not presented as C-Mixup. `cmixup_offline_v1` is a train-local
adaptation of label-local pairing. The two synthetic recipes were evaluated
separately and were not pooled.

## Paired crossed-factor bootstrap

The table below gives deterministic 10,000-replicate crossed
segment/target-bootstrap comparisons for the selected MLP fusion family. Each
delta is `candidate - real_only`, so a negative value favors the candidate.
The observed delta is the pooled 7,702-row OOF metric difference, whereas the
adoption gate above uses an equal-group macro difference. Intervals are 95%
bootstrap intervals. No p-values are reported.

| Scheme | Candidate | Delta RMSE_3D (mm) [95% CI] | Delta P95 (mm) [95% CI] |
| --- | --- | --- | --- |
| Segment LOSO | C-Mixup | -0.3210 [-1.1015, 0.4744] | -0.1007 [-2.1069, 1.3332] |
| Segment LOSO | Local interpolation | 0.6289 [-0.3905, 1.5612] | 0.6309 [-1.3457, 2.2339] |
| Segment LOSO | Replay | 0.6937 [-0.5508, 1.9619] | 1.1747 [-1.5439, 2.9582] |
| Unseen target | C-Mixup | -0.1360 [-0.9504, 0.5391] | -1.0463 [-4.0630, 2.8947] |
| Unseen target | Local interpolation | -0.5265 [-1.6073, 0.3767] | -1.9518 [-5.2456, 1.3004] |
| Unseen target | Replay | -0.5852 [-1.2981, -0.0106] | -1.3261 [-4.4033, -0.1502] |

The replay interval in the secondary scheme does not make replay deployable
and is not a confirmatory claim. It is a secondary negative-control observation
from the same corpus. Most synthetic-effect intervals span zero, the five
segment effects are heterogeneous, and no augmentation reached the fixed
engineering magnitude gate.

The later robust-v2 run inherited this real-only decision and did not run a
new robust-objective-plus-augmentation ablation. The v1 gate therefore does
not establish that C-Mixup or local interpolation would necessarily harm that
later objective.

## Strict-QC sensitivity

Applying strict QC only to predictions from the already locked OOF models
retained `7,670 / 7,702` rows.

| Scheme | Full pooled RMSE / P95 (mm) | Strict-subset RMSE / P95 (mm) |
| --- | ---: | ---: |
| Segment LOSO, selected cell | 14.4666 / 22.1354 | 14.3467 / 22.0077 |
| Unseen target, selected cell | 9.6231 / 16.9256 | 9.4324 / 16.6693 |

The small change indicates that the main OOF result is not driven by the 32
rows removed by strict QC. A separate strict-only full-data refit produced
RMSE/P95 `2.2064 / 3.8497` mm on its own fitting rows. That number is
**resubstitution-only** and must never be compared with independent or OOF
error.

## Shortcut diagnostics

Planned-coordinate and uncalibrated-magnetometer views were diagnostic-only
and were excluded from final selection. The clearest warning was Ridge:
adding magnetometer summaries changed fusion macro RMSE from `26.2857` to
`18.3282` mm for segment LOSO and from `26.7363` to `18.0604` mm for the
unseen-target scheme. Without an independent session and magnetic calibration,
this apparent gain can encode environment, schedule, segment, or session
identity. It is evidence of shortcut capacity, not a deployable sensor gain.

Planned-only Ridge errors were `81.8660` mm (segment) and `86.1159` mm
(target); planned-coordinate views likewise remain excluded because they can
reveal command context rather than measure the endpoint. No planned, target-ID,
magnetometer, Vive, Tracker, timestamp, QC, or provenance column is accepted by
the final inference allow-list.

## Final full-data model

- Selected cell: `mlp__fusion__real_only`
- Hidden layers / alpha: `(64, 64)` / `0.001`
- Features: `43`, exact ordered allow-list
- Ensemble members: `3`
- Segment-inner aggregate selection RMSE_3D: `15.8496` mm
- Final fit: `7,702` real rows and `0` synthetic rows
- Selection rule:
  `no_augmentation_passed_gate_fallback_to_best_primary_real_only`
- Final model SHA-256:
  `14ddf90e009f112e85600f9346281dd91b125da10f37ca4b211c42c03ae45401`
- Model contract fingerprint:
  `2c3445fb5e48713642413d1b4f011e83cff66bc1817cd920cb67123edb1116a5`
- Fixed-fixture prediction SHA-256:
  `30af3afa176b02d6f0139ca24b9dbb69cfe255c160b09de42808dd1d7a12ea3d`
- Output: `pred_vive_world_{x,y,z}_mm`, frame `vive_world`, unit `mm`
- Independent test score for this all-data refit: **not available**

The serialized joblib object is code-bearing Python data. It must be loaded
only from the trusted project release after ZIP and member-checksum validation;
it is not safe to load a package from an untrusted source.

## Release and reproducibility identities

| Identity | SHA-256 or value |
| --- | --- |
| Parent `ml_preparation_v1.zip` | `94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930` |
| Parent preparation config | `bf258a9be480b4f69aa14f6e121949042771f1104a69cdabc014adf6d95f7503` |
| Parent `trials.parquet` member | `1b0a821a1822f237adadb1c978ea73da02b0e4bb7f099ce085a63931e9792b54` |
| Experiment config | `1ae48cdd2508d0bf9daf3b0092598b49e3aab1951ce5054d3c5b3e6f85d4f5c3` |
| Frozen code revision | `36c9a1400fa533e359b0a9110e25cc3e9bd95713` |
| Experiment manifest | `f82b1194d273731f4c3b726c4cff2e32d0ffe95b3ba215ce22bd14b749f48d26` |
| Experiment release ZIP | `5eadbd2971c015d1bfd46d09ab19b3b6fa3447fe8bdc46079a54114f8c14d9bc` |
| Release ZIP size | `77,567,241` bytes (`32` members) |

Two complete runs under the same frozen code, lock, config, and parent release
produced identical report-layer and core-evidence hashes. Selected member
identities are:

| Member | SHA-256 |
| --- | --- |
| `predictions/oof_seed.parquet` | `82f154a25289a2a130cd995c6eeb7247ddf3687dbba57e2bd562a852db2abd7c` |
| `predictions/oof_ensemble.parquet` | `53f605002171da2918a26ba65f28eefbe90d675a6c3b1425ca64e02e9ad69613` |
| `metrics/fold_metrics.parquet` | `604b40e6e1acc168861dc2adcc2b541e0e7ea7c7d7f4e8dd10043e07e2ba636f` |
| `metrics/aggregate_metrics.parquet` | `48d50c10df8358ae31f7a4566f0e2c1abc52b3454477fcaffd9c55d1de80e285` |
| `selection/inner_selection_trace.parquet` | `19665feb7b97eeddf8abc1834ccd2ed95bd0ace9cff7b1ebf2ac63712faf0db2` |
| `statistics/paired_comparisons.parquet` | `4b5afee911c897fbeaeaef4afef53363f03e003670e701a72f7063a047be1407` |
| `statistics/augmentation_gates.parquet` | `5a4e9f59913b8b9aa7517034db3e804e6f5dabeba5681f75cb4fcc28747098d7` |
| `sensitivity/strict_oof_metrics.parquet` | `1c3ef3be32e0232ec46aed33cee5d140da20c3460237e51216b83a000cb22fa5` |
| `reports/report.md` | `c023d6e6021df58cffce1086353d1b85d0f027afcb5259445c5041a74eb36404` |
| `MODEL_CARD.md` | `5f3a03d99854f421461340b8f018acafe45f781db2a19570c34f11668ae52036` |

The locked environment used CPython `3.12.13`, scikit-learn `1.9.0`, XGBoost
`3.3.0`, NumPy `2.3.5`, SciPy `1.17.1`, pandas `2.3.3`, and PyArrow `21.0.0`.
Estimator-internal thread counts remained one and the declared runner context
parallelism remained four.

### Runtime scheduling transparency

After the implementation was frozen, higher host utilization was requested.
An ignored runtime helper at `artifacts/endpoint_outer_accelerator.py` invoked
the frozen `_run_outer_cell` entrypoint for disjoint, not-yet-computed target
outer cells. It changed task scheduling only: release, config, split, context,
seed, estimator, model-selection, and member-validation logic were unchanged.
Every helper result was accepted through the same task/config/member-hash
checks; there were no duplicate task identities, partial files, or recorded
worker errors.

The helper source SHA-256 was
`f3776c7b2758069c60f3666a2739d17a368d12bae48b61618e23f12c58910a2c`.
The `93` accelerated cells from target outer groups 2--4 had identical task and
member hashes in both complete runs, and all `310 / 310` outer tasks completed.
Observed CPU utilization reached approximately 94--100%. Memory was kept within
safe system headroom rather than artificially filled; unused RAM is not a
scientific workload target. The helper is not tracked source and is not a
release member.

## Research basis

The estimator comparison is motivated by the strong tabular baselines and
careful neural comparisons of [Grinsztajn et al., NeurIPS
2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html)
and [Gorishniy et al., NeurIPS
2021](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html).
ExtraTrees follows [Geurts, Ernst, and
Wehenkel](https://doi.org/10.1007/s10994-006-6226-1), XGBoost follows [Chen and
Guestrin](https://doi.org/10.1145/2939672.2939785), and the shared XYZ MLP is a
related-output hypothesis informed by [Caruana's multitask
learning](https://doi.org/10.1023/A:1007379606734). These papers motivate the
matrix; they do not predetermine its winner.

The label-local recipe adapts [C-Mixup, NeurIPS
2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html)
and was cross-checked against its [official
implementation](https://github.com/huaxiuyao/C-Mixup). The project-specific
`local_interp_v2` hypothesis is not attributed to that paper.

Nested selection and leakage barriers follow the concerns documented by
[Varma and Simon](https://doi.org/10.1186/1471-2105-7-91), [Cawley and
Talbot](https://www.jmlr.org/papers/v11/cawley10a.html), [Roberts et
al.](https://doi.org/10.1111/ecog.02881), and [Kapoor and
Narayanan](https://doi.org/10.1016/j.patter.2023.100804). The paired
crossed-factor bootstrap is based on [Owen and
Eckles](https://doi.org/10.1214/12-AOAS547). The decision to expose fold effects
and avoid p-value superiority claims is also informed by [Bengio and
Grandvalet](https://www.jmlr.org/papers/v5/grandvalet04a.html).

## Required next evidence

Before making a confirmatory or deployment claim, collect and lock at least one
independent measured session, preserve session-level holdout, independently
estimate Tracker-to-base/TCP calibration if robot-frame output is required,
define an online window aggregator for dynamic use, and evaluate latency,
drift, failure handling, and closed-loop safety separately. None of those
requirements is satisfied by this v1 package.

No hardware movement, serial access, torque command, calibration command,
firmware upload, Tracker SDK operation, or new data collection was performed to
produce or document these offline results.

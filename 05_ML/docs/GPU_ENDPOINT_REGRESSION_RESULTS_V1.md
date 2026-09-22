# GPU Endpoint Regression Experiment v1 Results

## Status and objective

`alphaf.endpoint_regression.gpu_experiment.v1` completed as an exploratory GPU
successor to the frozen endpoint-regression v1 study. Its objective is
Vive-free static endpoint estimation: one three-second trial-summary row of
three encoder deltas and four IMUs is mapped to raw endpoint XYZ in the
`vive_world` frame, in millimetres.

The inference allow-list is the 43-column `fusion_v1` view. Vive/Tracker,
planned-position, target identity, segment identity, timestamp, QC, and
provenance fields are not model inputs. All training examples are measured
real rows. Synthetic augmentation is excluded because the frozen v1 study
passed none of its eight augmentation engineering gates.

This result is not robot-base or TCP position, dynamic state estimation,
formal ShapeFK, or control authority. The current corpus contains no
independently acquired test session, and one inner MLP context was observed as
a hardware/convergence pilot before this GPU contract was frozen. The result
therefore remains exploratory.

## Contract, corpus, and evaluation

| Item | Value |
| --- | --- |
| Parent preparation contract | `alphaf.ml_preparation.v1` |
| Parent preparation SHA-256 | `94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930` |
| GPU experiment config SHA-256 | `9bda3280b91351bd00f67faece111454c9807d277463683d5a3c409747dc4402` |
| Eligible measured rows | 7,702 |
| Inner tasks | 480 |
| Outer tasks | 20 |
| Final seed fits | 3 |
| Primary scheme | Five segment-LOSO folds, segments 36--40 |
| Secondary scheme | Five held-out target groups, groups 0--4 |
| Runtime | Four NVIDIA CMP 170HX GPUs |
| Software | CPython 3.12.3, PyTorch 2.12.1+cu132, CUDA 13.2, XGBoost 3.3.0 |

Every outer fold used each non-test group once as inner validation. Feature
and target preprocessing was fitted only on measured real training rows.
Validation and test roles contain measured rows only. The outer labels were
accessed only after the corresponding inner selection was fixed, and outer
scores were not used to choose the final all-data configuration.

The primary reported aggregate is the equal-group macro mean. The pooled
metric instead gives every one of the 7,702 OOF rows equal weight. In both
cases,
`RMSE_3D = sqrt(mean(dx^2 + dy^2 + dz^2))`; macro P95 is the equal mean of the
five group-specific P95 values.

## Inner selection

The final all-data family was fixed to PyTorch MLP before outer evaluation.
The following trace is the equal-group mean over all 20 primary segment-inner
validation contexts. Values are rounded to four decimals; the selected score
stored in the model manifest is exactly `14.231536211519972` mm.

| Hidden layers | Weight decay | Config ID | Inner RMSE3D (mm) | Inner P95 (mm) | Maximum context RMSE3D (mm) | Selected |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 64, 64 | 0.001 | `e846a23ca5ccc512` | 14.3288 | 20.7665 | 26.8150 | No |
| 64, 64 | 0.0001 | `3b70dbff5237f6c3` | **14.2315** | **20.7107** | 26.8900 | **Yes** |
| 128, 64 | 0.001 | `f85a516ac56d1dab` | 14.6255 | 21.1398 | 26.9943 | No |
| 128, 64 | 0.0001 | `3cc13d755fffc793` | 14.5228 | 20.9746 | 26.9043 | No |

The target-group inner result was secondary and did not select the final
configuration. Its best MLP setting was `(128, 64)` with weight decay
`0.0001`, at `10.7523` mm. The primary-selected `(64, 64)` setting scored
`10.9785` mm there, showing that the architecture preference is weaker than
the group effect.

The primary-selected seed epoch budgets, obtained from the inner best-epoch
rule, are 291, 341, and 309. The selection record explicitly states
`outer_test_used=false`.

## Outer OOF results

The PyTorch entries are three-seed prediction ensembles. XGBoost is the
predeclared single-seed CUDA reference family. Fold-specific configurations
were selected only from that fold's four inner validation groups.

| Scheme | Model | Macro RMSE3D (mm) | Macro P95 (mm) | Pooled RMSE3D (mm) | Pooled P95 (mm) |
| --- | --- | ---: | ---: | ---: | ---: |
| Segment LOSO | PyTorch MLP | **14.9272** | **21.2712** | **15.1201** | **22.7191** |
| Segment LOSO | CUDA XGBoost | 19.2915 | 29.0726 | 19.9712 | 33.2668 |
| Target group | PyTorch MLP | **9.5700** | **17.2743** | **9.7892** | **17.1599** |
| Target group | CUDA XGBoost | 25.4604 | 43.3229 | 25.6419 | 46.5854 |

Relative to CUDA XGBoost, the MLP lowers macro RMSE/P95 by
`4.3643 / 7.8014` mm for segment LOSO and `15.8904 / 26.0486` mm for
target-group holdout. This supports retaining a smooth shared-XYZ MLP as the
candidate family; it does not establish cross-session accuracy.

### Segment-LOSO groups

| Segment | Rows | MLP configuration | MLP RMSE3D / P95 (mm) | XGBoost configuration | XGBoost RMSE3D / P95 (mm) |
| ---: | ---: | --- | ---: | --- | ---: |
| 36 | 1,672 | 64-64, wd `0.0001` | 18.8190 / 23.9244 | depth 8, child 1, lambda 1 | 26.1638 / 40.3961 |
| 37 | 1,597 | 64-64, wd `0.0001` | 11.8455 / 18.1580 | depth 8, child 5, lambda 1 | 17.6396 / 26.4550 |
| 38 | 1,707 | 128-64, wd `0.0001` | 15.5062 / 23.9668 | depth 8, child 5, lambda 1 | 19.3286 / 30.4933 |
| 39 | 1,659 | 64-64, wd `0.001` | 12.8541 / 18.8339 | depth 8, child 5, lambda 1 | 18.1682 / 25.8157 |
| 40 | 1,067 | 64-64, wd `0.0001` | 15.6111 / 21.4728 | depth 8, child 5, lambda 1 | 15.1573 / 22.2030 |

Segment 36 is the hardest MLP group. Segment 40 is the only segment where the
XGBoost point RMSE is lower, by `0.4539` mm, while its P95 remains `0.7302` mm
higher.

### Target groups

| Target group | Rows | MLP configuration | MLP RMSE3D / P95 (mm) | XGBoost configuration | XGBoost RMSE3D / P95 (mm) |
| ---: | ---: | --- | ---: | --- | ---: |
| 0 | 1,538 | 64-64, wd `0.0001` | 10.4486 / 24.0304 | depth 4, child 1, lambda 10 | 24.7243 / 39.1831 |
| 1 | 1,554 | 128-64, wd `0.001` | 7.7332 / 15.9616 | depth 4, child 1, lambda 1 | 29.6710 / 54.2749 |
| 2 | 1,553 | 128-64, wd `0.001` | 10.8913 / 20.7860 | depth 4, child 1, lambda 10 | 23.9065 / 35.8428 |
| 3 | 1,555 | 128-64, wd `0.0001` | 12.0059 / 12.9929 | depth 4, child 1, lambda 1 | 20.7473 / 33.9324 |
| 4 | 1,502 | 128-64, wd `0.001` | 6.7711 / 12.6008 | depth 4, child 1, lambda 10 | 28.2531 / 53.3816 |

Target group 3 has the highest MLP RMSE, while group 0 has the highest MLP
P95. The mismatch between those rankings is evidence that central error and
tail behavior must both remain visible.

## Comparison with frozen endpoint-regression v1

The only comparisons below use the same parent corpus, grouped outer schemes,
43-column fusion view, measured-real recipe, label frame/unit, and aggregate
definitions. They are protocol-matched comparisons, not bit-identical model
comparisons: frozen v1 used scikit-learn MLP and CPU XGBoost implementations,
whereas this study uses PyTorch and CUDA XGBoost. Neither study has an
independent acquisition session.

| Scheme / family | Metric | GPU v1 (mm) | Frozen v1 (mm) | GPU minus frozen (mm) |
| --- | --- | ---: | ---: | ---: |
| Segment / MLP | Macro RMSE3D | 14.9272 | 14.2628 | +0.6644 |
| Segment / MLP | Macro P95 | 21.2712 | 21.0190 | +0.2522 |
| Segment / MLP | Pooled RMSE3D | 15.1201 | 14.4666 | +0.6535 |
| Segment / MLP | Pooled P95 | 22.7191 | 22.1354 | +0.5836 |
| Target / MLP | Macro RMSE3D | 9.5700 | 9.3298 | +0.2402 |
| Target / MLP | Macro P95 | 17.2743 | 15.7989 | +1.4754 |
| Target / MLP | Pooled RMSE3D | 9.7892 | 9.6231 | +0.1661 |
| Target / MLP | Pooled P95 | 17.1599 | 16.9256 | +0.2343 |
| Segment / XGBoost | Macro RMSE3D | 19.2915 | 19.2625 | +0.0290 |
| Segment / XGBoost | Macro P95 | 29.0726 | 28.4612 | +0.6113 |
| Target / XGBoost | Macro RMSE3D | 25.4604 | 26.0307 | -0.5702 |
| Target / XGBoost | Macro P95 | 43.3229 | 44.5005 | -1.1776 |

The PyTorch inner winner looked better than frozen v1's matched segment-inner
winner (`14.2315` versus `15.8496` mm, a `1.6181` mm or `10.21%` reduction),
but that gain did not transfer to the outer OOF comparison. The GPU MLP is
slightly worse than frozen v1 on every matched outer aggregate above. This is
the central reason not to select a model from inner improvements alone.

CUDA XGBoost provides a useful pipeline check. Across its eight matched
segment-inner configurations, the mean absolute GPU-versus-frozen RMSE
difference is `0.0396` mm and the maximum is `0.0773` mm. Its outer segment
macro RMSE is also within `0.0290` mm of frozen v1. The target result changes
more, but only by `0.5702` mm in macro RMSE.

## Seed, convergence, and domain-shift findings

For outer evaluation, the equal-group mean of individual MLP seed RMSEs is
`15.6196` mm for segment LOSO and `10.8625` mm for target groups. Ensembling
reduces those values by `0.6925` and `1.2925` mm, respectively. Mean within-fold
seed RMSE standard deviation is `0.5332` mm for segment LOSO and `0.8659` mm
for target groups; the maxima are `0.9167` and `1.3878` mm. The ensemble is
therefore materially useful, especially for the target scheme, but it does not
remove group shift.

The final-selected MLP's 60 primary inner seed fits have mean selected-checkpoint
train RMSE `3.7932` mm and validation RMSE `15.1955` mm, a mean gap of
`11.4024` mm. Their ensemble inner RMSE is `14.2315` mm. At epoch 400, 22 of
60 seed fits were more than 10% worse than their best validation checkpoint,
and 37 of 60 had gone at least 50 epochs without a new best. All recorded
numeric histories remained finite. Checkpoint selection prevented the late
epoch regression from becoming the selected model, but the gap is direct
evidence of overfitting pressure.

Collection-segment shift dominates held-out-target shift. MLP macro RMSE is
`14.9272` mm for segment LOSO versus `9.5700` mm for target groups. The pooled
axis RMSEs are X/Y/Z `9.5315 / 8.7190 / 7.8580` mm for segment LOSO and
`7.3498 / 2.9082 / 5.7750` mm for target groups; the Y-axis difference,
`5.8107` mm, is the largest.

The aggregate bias partly hides opposing segment offsets. MLP fold biases for
segments 36 and 40 are X `-13.7842` and `+10.4324` mm; segment 38 has Z bias
`-9.2722` mm, while segment 39 has Y bias `+7.6999` mm. The hardest primary
inner context, validation segment 36 with outer segment 37 also excluded,
reached RMSE/P95 `26.8900 / 39.5414` mm. These repeated group-dependent offsets
point to acquisition/frame/calibration shift rather than a seed-only failure.

## Final all-data model

The final artifact contract is `alphaf.endpoint_regression.gpu_model.v1`.
It is a mean ensemble of three `(64, 64)` PyTorch MLPs with weight decay
`0.0001`, trained on all 7,702 measured real rows and zero synthetic rows.
The response is `vive_world` XYZ in millimetres.

| Seed index | Model seed | Epochs | Training RMSE3D (mm) | Checkpoint SHA-256 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 79,529,159 | 291 | 3.2513 | `a99542eae271512b87b0cbc32431507e652b85c7a20ae5d3f55579128848e602` |
| 1 | 3,549,711,029 | 341 | 2.8410 | `dbb370a8d9561a466d0416dd0690180b9c0f4e17fc0ca2d3f0d65096f6bc3d92` |
| 2 | 2,651,867,816 | 309 | 2.9752 | `872883a574d2ebb79cec494b489eac6ca6c2b81a31fc27535b5e6ee6d4e3c5e7` |

The training errors are resubstitution-only and must not be compared with OOF
or independent-test error.

| Artifact | SHA-256 |
| --- | --- |
| Final model manifest | `dc4379954d58eab6487b4c2dc0664bb3be79fb9fe95a65314f1f3827a869db5a` |
| Final preprocessing | `1b5da02ae428f7016e77e15220ad1be81bd8585a5df98d8388dedf3dde005096` |
| Inner selection | `bb414d2187180f6e8bdee660852e43032866afd90d6b630ed1fcd90619c0ef0b` |
| Evaluation summary | `1f1887525e61829cf3bc3d7b0eb3539f6dd870cbe620c6007537679b432f864d` |
| Run summary | `349cd3b080904c5bd6f2ee497357921462ad4dc7a0b8bf5eeb32427706d76375` |
| Model fingerprint | `240eebcca0c818867e171c623d797e650b874f53a72233848ac5e77cc9fb1e07` |
| Source fingerprint | `727a8510be492b5ce356fb00c19ff099bb85e5494f1b3351148d93ce67f37d25` |
| Exact source snapshot | `6678bace8d4f0a94d1f3a70ebfd91e53f22bfd42e8f2bc873668e5f136e063c3` |
| Source provenance | `44458d5c4f78ddf080feb6197d190fdf0569351c1dc5021f17370caf4f41fd24` |
| Experiment manifest file | `7287a2445fbbe8d31df3973c7819530ea5b3a4959039f41360ba4f32b2d922cb` |

There is no independent score for the all-data refit.

## Audit and provenance limitations

The repaired experiment manifest records 1,193 members, including the source
provenance ledger and snapshots. The strengthened validator reports
`status=ok` at `validation_level=semantic_recomputation`. It verifies all 480
inner results, 20 outer results, and three final fits; audits 51 split
contexts; recomputes inner selection, prediction metrics, and all four OOF
tables; and runs a CPU inference smoke test against the final three-checkpoint
ensemble. It also recomputes every final seed's training RMSE from its
checkpoint and the verified parent release; the maximum difference from the
recorded values is `3.604495271147812e-07` mm. The validator additionally
verifies the parent release, experiment config, source binding, file hashes,
checkpoint tensors, and the final model fingerprint.

Two fields remain hash-bound attestations rather than independently
recomputed values: an inner best epoch when it falls between recorded history
cadences, and the outer per-seed prediction metrics. Their containing task,
result, history, and manifest identities are checked, but this distinction
must remain explicit.

The run began from Git revision
`1b492b6a1f832579368f52a7f4eee0fedd61a5b3` with a dirty working tree and
untracked GPU-contract sources. The run was uninterrupted and did not use
resume. Source SHA-256 values were captured after the run and before the
recorded core-runner edit boundary. The authoritative
`training_source_snapshot_exact.tar` is 235,520 bytes and contains all 17
historical transitive source members with 17 of 17 SHA-256 matches. It is
deterministically rebuildable to the same archive hash. The three members
added by the transitive-source audit are `requirements-eval.lock`,
`cleaning.py`, and `splitting.py`. Its source fingerprint is
`727a8510be492b5ce356fb00c19ff099bb85e5494f1b3351148d93ce67f37d25`.

The older `training_source_snapshot.tar`, SHA-256
`04e5402d028b6a9bc2954c67c4759699f2cffebd387964931505c3402c1f442d`,
matched 12 of 17 ledger members, mismatched `cli.py` and `experiment_io.py`,
and omitted the three transitive members above. It is retained only as a
superseded audit artifact and is not the authoritative source snapshot.

One provenance limitation remains: the source fingerprint was reconstructed
and bound after the uninterrupted run rather than embedded in every task
before launch. The exact `cli.py` and `experiment_io.py` runtime bytes were
recovered from the Git index lineage and accepted only after matching the
already-recorded per-file SHA-256 ledger. The strengthened validator reports
this explicitly as `source_binding=post_run_uninterrupted_capture`; it should
not be misrepresented as pre-launch clean-commit binding.

`gpu_inference.py` is intentionally absent from the historical 17-member
training source closure because it was created after the run and participated
in neither historical training nor the original validator. The strengthened
validator now imports it for CPU inference smoke testing, so future
pre-launch source closures must include it.

## Claim boundary and next evidence

The defensible conclusion is that encoder-plus-four-IMU fusion contains a
strong within-corpus signal for static raw-Vive endpoint regression and that
the MLP is substantially better than the CUDA tree reference. The GPU MLP did
not improve on frozen v1's nested outer OOF result, and neither experiment
establishes independent-session, robot-frame, online, or control performance.

Before passive deployment, the project still needs independently collected
sessions with session-level holdout, independently estimated
Tracker-to-robot-base and endpoint/TCP calibration, online windowing and
dynamic-motion validation, uncertainty calibration, OOD detection, and an
abstention policy. Any control use is a separate safety-qualified study.

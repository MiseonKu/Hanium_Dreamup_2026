# Robust-v2 GPU endpoint-regression results

## Outcome and claim boundary

`alphaf.endpoint_regression.gpu_experiment.robust_v2` completed successfully on
2026-08-24. The selected candidate is group-balanced ERM with a `(64, 64)`
PyTorch MLP, weight decay `0.001`, and a three-seed mean ensemble. Its frozen
inner robust score is `J=16.036852831949602 mm`, below the historical GPU-v1
promotion threshold of `16.3801 mm` by `0.343247168050397 mm` (`2.0955%`). The
recorded action is therefore `promote_robust_v2`.

The ultimate model objective is Vive-free static endpoint estimation: map one
three-second trial-summary row of three encoder deltas and four WT901C IMU
summaries to raw endpoint XYZ in the `vive_world` frame, in millimetres. The
inference input is exactly the 43-column `fusion_v1` allow-list. Vive/Tracker,
planned position, target identity, segment identity, timestamp, QC, and
provenance fields are not inputs.

This result is development evidence, not an independent confirmation. It is
not robot-base/TCP position, dynamic state estimation, formal ShapeFK or the
repository's three-IMU `IMU-only` condition, online sensor aggregation,
uncertainty calibration, OOD detection, control authority, or a safety case.

## Frozen run identity

| Item | Value |
| --- | --- |
| Training source commit | `7400d870f1cbfa742e8665bb9a7b83dbb8a536e1` |
| Git state at launch | Clean |
| Robust-v2 config SHA-256 | `ed28b16cde0be391b1c893003a58bd26c4de57df144e6e1cdf64c7a1dca8339a` |
| Training-source fingerprint | `f8a81fdb714c0c23295ed3e52d949a2f6fd31ad4b82d23a2e70de85ac5274df7` |
| Preparation release SHA-256 | `94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930` |
| Eligible measured rows | 7,702 |
| Synthetic rows used | 0 |
| Candidate objectives | 6 |
| Inner contexts per candidate | 20 |
| Inner tasks / seeded fits | 120 / 360 |
| Reused outer diagnostic tasks | 5 |
| Final all-data seed fits | 3 |
| Runtime | Four NVIDIA CMP 170HX GPUs |
| Software | CPython 3.12.3, PyTorch 2.12.1+cu132, CUDA 13.2, XGBoost 3.3.0 |

The parent preparation release still validates structurally: 7,707 canonical
real rows, 7,702 train-eligible rows, 20 augmentation Parquet files, and
123,232 synthetic rows. However, the frozen endpoint-regression v1 study
passed zero of eight augmentation engineering gates. Robust v2 therefore used
only measured real rows. Structural generation validity is not evidence that
synthetic rows improve a model.

Those eight rows were two recipes crossed with four model families. Four V1
RMSE point estimates improved and four worsened; all eight missed the fixed
minimum reduction. Robust v2 did not directly compare augmentation, so its
real-only result must not be reported as evidence that augmentation always
hurts this objective.

## Research-grounded improvements evaluated

The comparison kept the architecture, data, optimizer budget, and paired
initializations fixed, changing only the training objective. This isolates the
effect of the proposed generalization improvement.

- Group-balanced mini-batches give every present collection segment equal
  mass rather than letting its row count determine the objective.
- Group DRO with `eta=0.01` and `eta=0.1` adapts mass toward high-loss
  segments, following the worst-group motivation in
  [Sagawa et al., ICLR 2020](https://openreview.net/forum?id=ryxGuJrFvS).
- V-REx with `beta=10` and `beta=100` penalizes variance between segment
  risks, after 40 pure-ERM warm-up epochs, following
  [Krueger et al., ICML 2021](https://proceedings.mlr.press/v139/krueger21a.html).
- Non-adaptive SAM with `rho=0.05` tests whether flatter local loss geometry
  improves held-out-segment behavior, following
  [Foret et al., ICLR 2021](https://openreview.net/forum?id=6Tm1mposlrM).

All methods completed without non-finite histories or objective-mass errors.
The literature motivates hypotheses; it does not guarantee improvement on
this corpus. The frozen validation rule alone selected the result.

## Validation design and selection rule

For each candidate, every ordered pair of distinct outer and inner segments
from 36--40 defines one validation context. The remaining three segments are
real-only training data, producing `5 * 4 = 20` contexts per candidate. Each
context fits three paired seeds and ensembles their predictions.

For candidate `c`, with context RMSE `r[c,o,v]`:

```text
A[c] = mean over all 20 r[c,o,v]
W[c] = max over validation segment v of mean over its four contexts
J[c] = (A[c] + W[c]) / 2
```

Lower `J` wins. Ties use lower mean P95, lower mean seed dispersion, then the
predeclared candidate order. The reused outer folds were not available to the
selector and cannot trigger reselection.

These 20 contexts exceed the requested minimum of ten validation runs, but
they are correlated nested contexts from the existing corpus. They are not 20
independently restarted acquisition sessions.

## Candidate results

Every candidate has exactly 20 contexts and 60 seeded validation fits. All
candidates have segment 36 as their worst validation segment.

| Rank | Candidate | `J` (mm) | Mean RMSE3D (mm) | Worst-segment mean RMSE3D (mm) | Mean P95 (mm) | Mean seed SD (mm) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | **Balanced ERM** | **16.0369** | **14.1057** | 17.9680 | **20.6176** | 0.7840 |
| 2 | V-REx, `beta=10` | 16.1252 | 14.3464 | **17.9041** | 21.1440 | **0.7665** |
| 3 | V-REx, `beta=100` | 16.2169 | 14.3996 | 18.0341 | 21.0903 | 0.8511 |
| 4 | Group DRO, `eta=0.01` | 16.4742 | 14.3842 | 18.5642 | 20.9338 | 0.7765 |
| 5 | Group DRO, `eta=0.1` | 16.4913 | 14.3292 | 18.6534 | 20.6268 | 0.7729 |
| 6 | Balanced ERM + SAM, `rho=0.05` | 18.4202 | 16.2259 | 20.6145 | 24.2756 | 0.8524 |

V-REx `beta=10` lowered the segment-36 mean by `0.0639 mm` relative to ERM,
but raised the overall 20-context mean by `0.2407 mm`, leaving its `J`
`0.0884 mm` worse. In paired context comparisons against ERM, Group DRO
`eta=0.01`, Group DRO `eta=0.1`, V-REx `beta=10`, V-REx `beta=100`, and SAM
won respectively 5, 6, 6, 8, and 2 of 20 contexts. Each alternative objective
was worse than ERM on paired mean RMSE. The selected robust-v2 recipe is
therefore segment-balanced ERM; the more complex robust losses added no mean
benefit within this frozen candidate grid. An unbalanced-ERM control was not
part of the grid, so this run does not isolate a causal effect of balancing
itself.

### Selected-candidate segment means

| Validation segment | Contexts | Mean inner RMSE3D (mm) |
| ---: | ---: | ---: |
| 36 | 4 | 17.9680 |
| 37 | 4 | 12.7512 |
| 38 | 4 | 14.2834 |
| 39 | 4 | 12.2767 |
| 40 | 4 | 13.2492 |

The hardest interaction was holding out segment 36 while segment 37 was also
absent from training. ERM reached `25.9362 mm` RMSE there. SAM reached
`31.3984 mm`; none of the robust objectives fixed this domain gap. The common
large train-to-validation gaps and segment-dependent reversals point to
collection/calibration shift rather than numeric training failure.

## Reused outer development diagnostic

The outer result is explicitly designated
`reused_holdout_development_diagnostic`. GPU-v1 already exposed these folds,
so the table is useful for regression diagnosis but is not an independent
test and did not select the candidate.

| Segment | Rows | RMSE3D (mm) | P95 (mm) | Seed RMSE SD (mm) |
| ---: | ---: | ---: | ---: | ---: |
| 36 | 1,672 | 18.7172 | 23.7787 | 1.1446 |
| 37 | 1,597 | 11.0879 | 18.1110 | 0.6802 |
| 38 | 1,707 | 15.0828 | 23.6045 | 1.4609 |
| 39 | 1,659 | 12.0623 | 18.1069 | 0.4527 |
| 40 | 1,067 | 14.2283 | 19.9257 | 0.2561 |

| Aggregate | RMSE3D (mm) | P95 (mm) |
| --- | ---: | ---: |
| Equal-segment macro | 14.2357 | 20.7053 |
| Pooled 7,702-row OOF | 14.5379 | 21.9799 |

For context only, GPU-v1's corresponding segment macro RMSE/P95 was
`14.9272 / 21.2712 mm`, while its pooled values were
`15.1201 / 22.7191 mm`. Using the exact artifacts, robust v2 is lower by
`0.6915 / 0.5658 mm` macro and `0.5823 / 0.7392 mm` pooled. Because the folds
were reused after v1, these improvements cannot be promoted to confirmatory
claims.

## Final all-data model

The promoted artifact contract is
`alphaf.endpoint_regression.gpu_model.robust_v2`. It is a mean ensemble of
three `(64, 64)` MLPs trained with segment-balanced ERM on all 7,702 measured
rows. The epoch budgets are medians of the corresponding inner best epochs.

| Seed | Model seed | Epochs | Training RMSE3D (mm) | Checkpoint SHA-256 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 79,529,159 | 317 | 3.0357 | `60ce175372911d40065765cd74a25cc1b729a167c098e9b98948cf26940444aa` |
| 1 | 3,549,711,029 | 338 | 6.1660 | `94e25d4a4f885f6b84d21a31e3d6e11c2fc454469c562f1555a06950622eac9a` |
| 2 | 2,651,867,816 | 360 | 3.0236 | `e7e3d9dfcc690f7dce3c479a60b001484cee8a90c07a1b946dcb72603ac123d8` |

The seed-1 resubstitution error is visibly higher, but finite and contract
compliant. Excluding it after seeing the result would be post-selection, so
the frozen three-seed ensemble is retained. A post-run read-only diagnostic
gave the complete ensemble a 7,702-row resubstitution RMSE/P95 of
`2.9306 / 5.1688 mm`; these training-data values are not generalization
scores.

The public inference entry point loaded the promoted run without a diagnostic
override and predicted an ordered eight-row fixture on CPU. All XYZ values
were finite, row IDs were preserved, and frame/unit/fingerprint matched
`vive_world`, `mm`, and
`50a1b01c8a1de42e2ae7248ff37bcaf61a93234a6493b049e8454b1d3cc719a1`.

## Semantic validation and artifact identity

The standalone validator reports `status=ok` at
`validation_level=semantic_recomputation`. It validates 120 inner results,
five outer results, and three final fits; recomputes selection without outer
data; verifies all 7,702 OOF rows, final checkpoint hashes, group-objective
mass, release/config/source binding, and promotion inference policy; and
passes all 17 semantic gates.

An independent read-only audit additionally opened the 360 prediction,
history, and group-history files referenced by the 120 inner results. It found
zero missing files, duplicate tasks, row-count or SHA mismatches, and
non-finite values. Recomputed selection values matched the runtime record with
maximum absolute difference `0.0`; recomputed prediction RMSE differed by at
most `3.55e-15 mm`.

Post-run verification also completed successfully:

- full offline test suite: `156 passed`;
- preparation validator: `status=ok`, 7,707 canonical rows, 7,702 eligible
  rows, 20 augmentation files, and 123,232 synthetic rows;
- GPU-v1 semantic validator: `status=ok`, 480 inner, 20 outer, and three final
  results;
- robust-v2 semantic validator: `status=ok`, all 17 gates;
- promoted-model CPU inference smoke: eight ordered finite predictions; and
- root repository agent doctor: `OK`, 17 required files and nine workspaces.

| Artifact | SHA-256 |
| --- | --- |
| Run manifest | `b4c9c6216f5f23425ba4e2e7777cc93a64f84deab52abf1c94b1ef81c5232ade` |
| Run summary | `03917fec6d1573ca03fe06947c182e4fa24db6fd0e65d74abe756c1409fbebee` |
| Semantic validation | `8c65cd5efad98698ba588498510a8420350ed10d7abf4feefd926f1614e3b4b7` |
| Inner selection | `f43d12173936d46051bb8afdf828709882684b116bf0e506e5a382b7d5fbb3ed` |
| Outer evaluation summary | `625d62c72922f807cb845300d329d5e10451c0a437944213ff6d136ae9a12da7` |
| Final model manifest | `6e233735c1e2bdca2a8767e88d9cb241c67730f76f5cda9c334d5f24a24cd6e0` |
| Training recipe | `396c89d5c041dfb4d842b706f28723a0584b00a2ca95f175c62a4e71a5604019` |
| Model fingerprint | `50a1b01c8a1de42e2ae7248ff37bcaf61a93234a6493b049e8454b1d3cc719a1` |

Generated runs remain under ignored `artifacts/` as required by repository
policy; this result document records their authenticated identities. No live
serial, motor, torque, calibration, firmware, or Tracker commands were run.

## Required confirmation and next improvement

The model and decisions must now remain frozen. A confirmatory study requires
at least ten newly acquired, independently restarted sessions that are not
used for tuning or reselection. It should preserve per-session calibration and
environment metadata, report session-level RMSE/P95 and failure tails, and
focus first on segment-36-like acquisition shifts. Only after that evidence
exists should uncertainty/OOD rejection thresholds or any robot-base/TCP
calibration and rollout policy be designed under a separate safety contract.

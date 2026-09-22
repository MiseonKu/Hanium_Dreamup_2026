# Endpoint Regression Methods and Evidence

## Interpretation boundary

This document fixes the rationale before the full result is interpreted. The
study is exploratory because its implementation-feasibility smoke work and
the final comparison use the same collected session. Nested group CV protects
model selection within that session; it cannot manufacture an independent
session. Any candidate must later be tested against independently collected
hardware sessions and an independently estimated Tracker-to-robot transform
before claims about deployment are considered.

The response is raw Vive world XYZ in millimetres from static three-second
trial summaries. The four WT901C summaries form a four-IMU endpoint-regression
proxy. They are not formal ShapeFK, the repository's `IMU-only` condition,
robot-base/TCP output, an online aggregator, or feedback control.

## Why compare these estimators

[Grinsztajn et al. (NeurIPS
2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html)
show that tree ensembles remain strong references on medium-sized tabular
problems. [Gorishniy et al. (NeurIPS
2021)](https://proceedings.neurips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html)
argue for well-tuned tabular deep-learning baselines rather than assuming
either trees or neural networks win universally. Those results motivate a
comparison, not an expected ranking, on this roughly 7.7K-row trainable table.

- Ridge supplies a regularized linear sensor baseline.
- ExtraTrees follows the randomized-tree construction of [Geurts, Ernst, and
  Wehenkel](https://doi.org/10.1007/s10994-006-6226-1).
- Gradient boosting follows the system and regularized objective described in
  the [XGBoost paper](https://doi.org/10.1145/2939672.2939785). Three separate
  boosters avoid assuming experimental multi-output behavior.
- The MLP shares a hidden representation and emits XYZ jointly. This is a
  related-output inductive-bias hypothesis informed by [Caruana's multitask
  learning](https://doi.org/10.1023/A:1007379606734), not evidence that shared
  output must be superior here.

The frozen, modest grids limit researcher degrees of freedom. Scaling is
restricted to Ridge and MLP. Target centring uses the real-train XYZ mean and
one common scale across all target elements, so preprocessing cannot silently
reweight coordinate axes.

## Nested grouped evaluation

[Varma and Simon](https://doi.org/10.1186/1471-2105-7-91) and [Cawley and
Talbot](https://www.jmlr.org/papers/v11/cawley10a.html) explain why model
selection assessed on the same validation evidence can produce optimistic
performance estimates. Every outer fold therefore owns a nested group
selection loop. [Roberts et al.](https://doi.org/10.1111/ecog.02881) motivate
CV structures that reflect grouped dependence rather than random-row splits.

The primary segment holdout asks whether a model transfers to an unobserved
collection segment. The secondary target-group holdout asks whether it
interpolates to targets excluded as groups. Both preserve repeated trials and
their group dependence. Scalers, target transforms, synthetic parents,
C-Mixup bandwidth, and hyperparameter choices use real training IDs only.
Validation/test rows stay real and cannot be augmentation parents.

These barriers also follow the broader leakage audit of [Kapoor and
Narayanan](https://doi.org/10.1016/j.patter.2023.100804): an apparently
innocent preprocessing or resampling step is part of training and must occur
after the evaluation split.

## Augmentation hypotheses and negative control

`replay` repeats each training row once while halving both copies' weights. It
is a negative control that separates exposure/update effects from synthetic
interpolation. Row-visit budgets are matched for the MLP.

`cmixup_offline_v1` adapts the regression-local pairing proposal in
[C-Mixup](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html).
Its label-distance Gaussian bandwidth and candidates are computed only from
the active real-train context, while the existing physical planned/joint gates
remain enforced. This study must cite it as an adaptation, not as an exact
reproduction of every benchmark detail.

`local_interp_v2` is a project-specific planned-manifold hypothesis: it pairs
nearby samples within the frozen segment/set/repeat and physical gates. It is
not attributed to the C-Mixup paper. The methods are evaluated separately and
are never pooled into a single augmentation condition.

The gate is deliberately an engineering threshold rather than a significance
test: macro 3D RMSE must improve by at least `max(10%, 3 mm)` and macro P95
must not degrade by more than 5%. A passing gate means only that the recipe is
eligible to compete with the deployment-safe real-only candidates for the
exploratory final refit; it does not force augmentation selection.

## Metrics and uncertainty

The primary `RMSE_3D` retains the project methodology's Euclidean residual
definition. Equal weighting of five held-out groups prevents a large group
from dominating model selection. Pooled and cell-balanced summaries are both
reported because repeats, targets, and segments are crossed rather than
independent rows.

Paired uncertainty uses a deterministic 10,000-replicate crossed
segment/target multiway bootstrap based on [Owen and
Eckles](https://doi.org/10.1214/12-AOAS547). The report exposes all fold
effects and a 95% interval. It does not report a superiority p-value. [Bengio
and Grandvalet](https://www.jmlr.org/papers/v5/grandvalet04a.html) show why a
simple unbiased variance estimator for K-fold CV is unavailable; with only
five segment groups, a precise-looking test would overstate the evidence.

## Shortcut and sensitivity analyses

Planned-only and fusion-plus-planned models measure command-context shortcut
capacity. Fusion-plus-magnetometer checks whether an uncalibrated,
session-specific magnetic signature makes prediction easier. Better
performance in either condition is a warning about within-session shortcuts,
not a deployable sensor improvement, so these views are excluded from final
selection.

Strict QC first subsets predictions from already locked OOF models. A separate
strict-only final-candidate refit is reported as sensitivity evidence. Neither
path changes the primary selection or retroactively removes difficult trials.

## Claims permitted by V1

The report may state within-session, nested-group exploratory errors and
paired effects for the frozen matrix. It may identify a candidate for future
validation. It may not claim augmentation or one estimator is generally
superior; that raw Vive predictions equal robot-base/TCP accuracy; that the
four-IMU view is ShapeFK or `IMU-only`; or that the packaged model is safe or
valid for real-time control.

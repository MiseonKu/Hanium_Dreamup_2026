# Robust-v2 GPU endpoint-regression contract

## Objective and status

The objective remains prediction of the static robot-arm endpoint without a
Vive Tracker at inference time.  The model receives the three encoder deltas
and the four-IMU `fusion_v1` view and predicts raw `vive_world` XYZ in
millimetres.

Robust v2 is a post-v1 development experiment, not a new independent test.
All five v1 outer groups have already been observed.  This disclosure is
material and frozen as `post_v1_outer_observed: true` and
`independent_test_available: false`.

The experiment is bound to both ancestors:

- GPU v1 canonical config SHA-256:
  `9bda3280b91351bd00f67faece111454c9807d277463683d5a3c409747dc4402`
- Preparation release SHA-256:
  `94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930`
- Robust-v2 canonical config SHA-256:
  `ed28b16cde0be391b1c893003a58bd26c4de57df144e6e1cdf64c7a1dca8339a`

## Frozen data and model boundary

Only measured real rows may be used.  Augmentation remains excluded because
endpoint-regression v1 passed 0 of 8 frozen augmentation gates.  The input is
exactly the 43-column `fusion_v1` allow-list; its canonical ordered-list digest
is `1756b0aafb8ba4c9a740683d0a0d12236d8ee6e0065cfbe9fc6fb9d656345b6f`.
Vive/Tracker measurements, planned positions, target and segment identities,
timestamps, QC, and provenance fields are forbidden inputs.

Post-run documentation clarification (2026-08-29): this is an inherited policy
decision, not a robust-v2 augmentation ablation.  The v1 point estimates
improved in four of eight comparisons but none cleared the fixed adoption
magnitude.  This contract therefore makes no claim that augmentation would
necessarily worsen the robust-v2 objective.

Every candidate uses the same MLP and training budget: hidden layers
`[64, 64]`, weight decay `0.001`, learning rate `0.001`, batch size `128`, 400
epochs, and three deterministic seeds.  Preprocessing is fitted only on the
real training rows.  Candidate comparisons use paired initializations: within
one split context, the same seed index resolves to the same model seed for all
six candidates.  Keeping architecture, data, budget, and initializations fixed
makes this a comparison of robust objectives rather than a larger-model search.

`batch_size=128` is the maximum requested batch-row budget.  Exact equal
segment counts use `floor(128/G)` rows per segment, so the effective inner,
outer, and final batch sizes are respectively 126 (`G=3`), 128 (`G=4`), and
125 (`G=5`).  The discarded remainder is never reassigned asymmetrically.

## Six candidates and research basis

The candidate order is also the final tie order:

1. group-balanced ERM;
2. Group DRO with `eta=0.01`;
3. Group DRO with `eta=0.1`;
4. V-REx with `beta=10`, activated at epoch 41 after 40 pure-ERM epochs;
5. V-REx with `beta=100`, activated at epoch 41 after 40 pure-ERM epochs; and
6. group-balanced ERM with non-adaptive SAM, `rho=0.05`.

For each optimization step, a group risk is the mean loss for one training
segment.  Balanced ERM gives present segments equal weight.  Group DRO adapts
weights toward high-loss segments; its float64 exponentiated update uses a
`1e-12` per-group numerical floor followed by renormalization to prevent exact
zero-weight underflow over long runs.  V-REx adds a penalty proportional to the
variance of segment risks after warm-up.  SAM evaluates the balanced-ERM
gradient at a radius-`rho` parameter perturbation before its optimizer update.

These choices are hypotheses, not guarantees of improvement:

- [Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts*,
  ICLR 2020](https://openreview.net/forum?id=ryxGuJrFvS) motivates optimizing
  worst-group performance and emphasizes regularization for worst-group
  generalization.
- [Krueger et al., *Out-of-Distribution Generalization via Risk Extrapolation*,
  ICML 2021](https://proceedings.mlr.press/v139/krueger21a.html) introduces
  V-REx as a variance penalty over environment risks.
- [Foret et al., *Sharpness-Aware Minimization for Efficiently Improving
  Generalization*, ICLR 2021](https://openreview.net/forum?id=6Tm1mposlrM)
  motivates minimizing loss together with local loss sharpness.

The papers use different tasks and do not establish performance on this robot
dataset.  Only the pre-registered validation rule below decides whether a
candidate is retained.

## Twenty validation contexts per candidate

The primary split is segment LOSO over groups 36 through 40.  For every
ordered pair `(outer=o, validation=v)` with `o != v`, the remaining three
segments are real-only training data.  Thus each candidate is evaluated in
`5 * 4 = 20` inner validation contexts, and every validation segment appears in
four contexts.  Six candidates therefore produce 120 context evaluations
(360 seeded fits), exceeding the requested minimum of ten validations without
pretending that correlated folds are independent sessions.

For candidate `c`, let `r[c,o,v]` be held-out validation RMSE3D in millimetres:

```text
A[c] = mean over all 20 r[c,o,v]
W[c] = max over validation segment v of mean over its four r[c,o,v]
J[c] = (A[c] + W[c]) / 2
```

The selected candidate minimizes `J`.  Exact ties are resolved by lower mean
P95 error, then lower seed dispersion, then the listed order above.  The
historical v1 score is frozen at `J=16.3801 mm`.  A robust-v2 candidate is
promoted only if its `J` is strictly smaller; otherwise the action is
`retain_v1`.  Values within `1e-12 mm` of the threshold are treated as equal
solely to prevent floating-point averaging artifacts from causing promotion.
When retained, a robust candidate checkpoint remains diagnostic evidence and
the normal inference entry point must reject it.

## Reused outer diagnostic and confirmation

Only after inner selection is frozen may the reused outer folds be evaluated.
Their exact designation is `reused_holdout_development_diagnostic`; they are
not an independent test and their scores cannot trigger candidate selection,
hyperparameter changes, or reselection.  Any later change creates a new
development contract.

Confirmatory evaluation requires the chosen model and all decisions to be
frozen first, followed by at least 10 independently restarted collection
sessions.  Those sessions may not be used for tuning or reselection.  This is
the first eligible route to a confirmatory claim; it still does not by itself
establish robot-base/TCP calibration, dynamic-state estimation, ShapeFK,
control authority, uncertainty calibration, OOD detection, or safe deployment.

# Methods And Research References

This document maps each research source to a local design decision. Citations
justify the method family; the exact repository contract remains the authority
for this dataset.

For the audited separation between published evidence, frozen project
heuristics, generated-parent distributions, and the completed 0/8 augmentation
gate result, see [AUGMENTATION_EVIDENCE.md](AUGMENTATION_EVIDENCE.md).

## Split Before Preprocessing Or Augmentation

Kapoor and Narayanan, “Leakage and the reproducibility crisis in
machine-learning-based science,” *Patterns* 4(9), 2023,
[DOI 10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804),
[open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/).

The paper's leakage taxonomy identifies using the full dataset for
preprocessing or over/under-sampling and ignoring dependencies between samples
as sources of overoptimistic evaluation. Local consequences are:

- immutable segment-LOSO and target-group assignments are made first;
- every parent search, sigma estimate, scaler, and future model-selection step
  is restricted to the applicable training fold;
- validation and test contain measured real trials only;
- parent IDs and fold context are stored so separation is machine-checkable.

Repeated targets and same-session segments create known dependencies, so a
row-random split is not an accepted evaluation protocol.

## Regression Mixup

Yao et al., “C-Mixup: Improving Generalization in Regression,” *NeurIPS 35*,
2022, [paper and supplemental material](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html),
[official implementation](https://github.com/huaxiuyao/C-Mixup).

The paper observes that uniformly pairing regression examples can yield
arbitrarily incorrect interpolated labels and instead biases pairing toward
similar labels. `cmixup_offline_v1` adopts that central idea but is explicitly
an offline, dataset-specific adaptation rather than a claim of exact paper
reproduction:

1. Its pool contains only eligible real rows in one outer training fold.
2. A candidate must be within 100 mm in planned position and 20 degrees in
   encoder-derived joint space; self-pairs are forbidden.
3. For labels `y_i,y_j` in raw Vive millimetres, candidate weight is
   `exp(-||y_i-y_j||^2/(2*sigma^2))`.
4. `sigma` is the median nearest admissible label distance computed inside
   that training fold only, with a recorded numerical floor.
5. `lambda ~ Beta(2,2)` and both features and label use the same lambda.
6. The synthetic fold total equals the outer-train real-row count, giving a
   1:1 mass and a fixed, reviewable release size.

The extra planned/joint constraints are conservative physical-support guards;
they are not part of a universal C-Mixup theorem. The release must therefore
compare no-augmentation, `local_interp_v2`, and `cmixup_offline_v1` during
later training rather than assume augmentation improves accuracy.

### Coverage replacement without constraint relaxation

Some real anchors have no legal partner under the frozen local-group or
planned/joint-distance rules. Assigning them a distant parent would change the
method after observing the data, while dropping their slots would break the
published fold-level 1:1 mass. The shared deterministic schedule therefore:

1. schedules every admissible anchor once with `anchor_replica_index=0`;
2. counts the original anchors that have no legal partner;
3. uses the SHA-derived `coverage_replacement` seed to sample that many
   admissible anchors uniformly with replacement;
4. marks each fill row `replacement_for_missing_anchor=true` and increments
   that anchor's `anchor_replica_index`.

The replacement anchor still chooses parent B under the original method and
the original fold's constraints. It does not impersonate or interpolate the
missing anchor. If a complete fold has no admissible pair, generation fails.
The manifest publishes missing and replacement counts so later training can
report the amount of coverage resampling alongside augmentation results.

## Deterministic Local Interpolation

`local_interp_v2` is a repository baseline rather than a separately claimed
published algorithm. It restricts two parents to the same
`(segment,set,repeat)` context, the five nearest admissible planned positions,
and 100 mm support radius. A seed derived from the stable anchor ID selects a
candidate and lambda from `0.25, 0.5, 0.75`; distance/ID ordering removes
row-order dependence. Anchors with no candidate use the shared coverage
replacement schedule above; their constraints are not widened.

`mix_lambda` is the weight of parent A, matching conventional mixup. For an
ordinary linear channel `v`, the synthetic value is
`lambda*v_a + (1-lambda)*v_b`. For an Euler scalar, it is:

```text
delta = wrap_to_pi(angle_b - angle_a)
angle = wrap_to_pi(angle_a + (1-lambda) * delta)
```

This ensures that `179 deg` and `-179 deg` interpolate near `180 deg`, not
through zero. Sine/cosine features are regenerated from the wrapped angle.
No IMU Euler-to-quaternion conversion is made because acquisition did not
verify the WT901C Euler convention. Captured Vive quaternion summaries remain
canonical audit metadata and are not copied into synthetic rows.

## Coordinate Alignment

Kabsch, “A solution for the best rotation to relate two sets of vectors,”
*Acta Crystallographica A* 32, 1976,
[DOI 10.1107/S0567739476001873](https://doi.org/10.1107/S0567739476001873),
[publisher issue](https://journals.iucr.org/a/issues/1976/05/00/).

Kabsch solves an optimal least-squares rotation between paired vector sets. It
does not make an alignment an independently measured tracker-to-robot
calibration. Fitting it on all labels before splitting lets test geometry
influence training coordinates. Therefore v1 keeps raw `vive_world` XYZ mm.
A later robot-base label contract must use independently measured extrinsics or
fit alignment only inside each training fold and report the distinction.

## IMU Shape Reconstruction Boundary

Hughes et al., “Sensing Soft Robot Shape Using IMUs: An Experimental
Investigation,” *International Symposium on Experimental Robotics*, 2021,
[DOI 10.1007/978-3-030-71151-1_48](https://doi.org/10.1007/978-3-030-71151-1_48),
[TU Delft repository record](https://repository.tudelft.nl/record/uuid%3Aeccca35c-d390-4878-8211-4731a9884dfb).

That work combines IMUs mounted along deformable segments, IMU quaternion
orientation, and a piecewise-constant-curvature model for shape
reconstruction. The current static dataset has four endpoint-rig sensor
summaries and no verified WT901C quaternion convention. It therefore supports
a supervised four-IMU endpoint proxy, not a claim that the formal 04 ShapeFK
`IMU-only` condition has been implemented or validated.

## Reproducibility Requirements

- The archive/member/config hashes, Arrow schema, counts, seed derivation, and
  every output member hash are release metadata.
- Base seed is `20260818`; fold seeds are derived from stable scheme/fold text,
  not Python's process-randomized `hash()`.
- Parquet row ordering is canonical before writing, and ZIP timestamps/member
  ordering are fixed for deterministic packaging.
- Analysis reports show exclusions and warning slices; they do not remove
  inconvenient labels or make a performance claim before model evaluation.
- The release validator recomputes the strict-QC sensitivity summary and
  verifies that every reported split, feature, parent-context, and synthetic
  leakage count is zero.

# `alphaf.ml_preparation.v1` Data Contract

Status: public, version-pinned offline preparation contract.

## Source Identity

| Item | Required value |
|---|---|
| Archive | `06_simulation_and_data_collect_preparation/imu_added_experiment/derived_dataset_archive.zip` |
| Archive SHA-256 | `ceadf164a15a3ce302f8b2ffdde450df0f6d3a91c3b0c258a28acb5de9b92d64` |
| Member | `derived/parquet/real_motion_demo/trials.parquet` |
| Member SHA-256 | `1b0a821a1822f237adadb1c978ea73da02b0e4bb7f099ce085a63931e9792b54` |
| Session | `real_motion_demo` |
| Source rows | `10,055` |

The reader rejects Git LFS pointer text, checksum drift, a missing or duplicate
member, an unexpected Arrow field/type/order, or a row-count mismatch. It does
not fall back to an untracked 08 file or another archive.

## Row Selection And Identity

The contract has two real-row states so that structural loss is auditable.

1. `canonical_real` selects segments 36-40, `keep=true`, and finite required
   Vive position, encoder, planned-position/grouping fields, and all four
   IMUs' Euler, acceleration, magnetometer, gyro-RMS, and sample-count fields.
   It contains exactly 7,707 rows.
2. `train_eligible` additionally requires:
   - `vive_tracking_valid_ratio == 1.0`;
   - `2.8 <= duration_s <= 3.2`;
   - `vive_n >= 170` and `encoder_n == 5`;
   - every IMU sample count `>= 28`;
   - encoder mean ticks inside inclusive ranges J0 `[1160,3083]`, J1
     `[1202,3186]`, and J2 `[1937,3968]`.

The result is exactly 7,702 eligible rows. The five rejected rows are retained
with `train_eligible=false` and a machine-readable
`train_exclusion_reasons`; they all have fewer than 28 `imu1_link_root`
samples. No label value, residual, target rarity, or quality-warning threshold
is a hard-filter criterion.

| Segment | Canonical real | Train eligible |
|---:|---:|---:|
| 36 | 1,672 | 1,672 |
| 37 | 1,597 | 1,597 |
| 38 | 1,707 | 1,707 |
| 39 | 1,659 | 1,659 |
| 40 | 1,072 | 1,067 |
| **Total** | **7,707** | **7,702** |

`trial_uid` is deterministic and content-independent of row ordering. Its
value is `trial_<hex>`, where `<hex>` is SHA-256 over the UTF-8 sequence
`<source-member-sha256>|real_motion_demo|<segment_id>|<t_start_ns>`. The
manifest records this recipe. Duplicate IDs are invalid.

## Coordinates, Units, And Quality Flags

- Labels are `vive_pos_raw_x_mm`, `vive_pos_raw_y_mm`, and
  `vive_pos_raw_z_mm`: the source `vive_world` XYZ metres multiplied by 1,000.
- V1 performs no all-data Kabsch/Procrustes fit and makes no claim that
  `vive_world` is the robot base or TCP frame.
- Encoder features are recomputed as
  `encoder_delta_rad_jk = (encoder_tick_jk - reference_jk) * 2*pi/4096`, with
  references `(2055, 2093, 1970)`. Source `q_encoder_rad_j*` and
  `logical_deg_j*` are provenance only and never primary features.
- Euler inputs remain the captured radian channels. Because the WT901C Euler
  axis convention is unverified, v1 does not invent IMU quaternions.
- `qc_warning_flags` records any IMU gyro RMS above `0.02 rad/s`, any Vive
  positional standard deviation above `0.001 m`, or any encoder tick standard
  deviation above `1.0`. These flags do not alter `train_eligible`.

## Feature Views

`feature_sets.json` is an allow-list. Labels, Vive quaternion, Tracker quality,
time, IDs, fold assignments, and synthetic provenance cannot appear in a
feature view.

- `encoder_only_v1` (3): `encoder_delta_rad_j0..j2`.
- `four_imu_endpoint_v1` (40): for each of four IMUs, three Euler sine and
  three Euler cosine values, three acceleration values, and one gyro-RMS.
- `fusion_v1` (43): union of `encoder_only_v1` and
  `four_imu_endpoint_v1`.
- `command_context_ablation_v1` (46): `fusion_v1` plus planned XYZ. It is
  explicitly an ablation and not a primary deployable view.
- `mag_ablation_v1` (55): `fusion_v1` plus twelve uncalibrated magnetometer
  channels. It is explicitly an ablation.

The name `four_imu_endpoint_v1` is deliberate. These four sensors do not meet
the repo's formal three-IMU-per-link ShapeFK `IMU-only` definition.

## Split Contract

Split assignment precedes every statistic-dependent transform and every
synthetic-parent search.

- `segment_loso`: five outer folds, with one entire segment in `{36..40}` as
  test. Inner validation leaves one of the remaining segments out. No segment
  is divided across roles within a fold.
- `target_group`: five outer folds where target fold is
  `int(target_id[1:]) % 5`. Each fold owns exactly ten of `T000..T049`; no
  target is divided across train and test.

`split_assignments.parquet` stores one row per eligible real trial and split
context, with `scheme`, `outer_fold`, `inner_fold`, `split_context`,
`trial_uid`, `segment_id`, `target_id`, `target_group`, and `role`. Outer-refit
contexts use `inner_fold=-1`; nested segment contexts explicitly mark train,
validation, and test. Scaling, imputation, model selection, or future
label-frame calibration must be fitted on the applicable training rows only.

## Synthetic Contract

Each outer scheme/fold contains one synthetic-only Parquet per method. For
each method, row count equals that fold's eligible real-train count; original
and synthetic sample mass are therefore 1:1. Across five folds, each real row
is train data four times, yielding 30,808 rows per scheme/method and 123,232
rows across two schemes and two methods.

The 1:1 requirement applies to the **fold total**, not to a guarantee that
every original row can serve as an anchor. The generator first keeps each
anchor that has at least one legal partner. For every anchor with no legal
partner, it fills one missing coverage slot by uniformly and deterministically
sampling, with replacement, from anchors that do have legal partners. The
method's group, planned-distance, joint-distance, real-only, and fold
constraints remain unchanged. A fold with no admissible pair fails rather
than relaxing a constraint.

- `local_interp_v2` restricts parents to the same
  `(segment_id,set_id,repeat_index)`, the nearest five admissible planned
  positions within 100 mm, and deterministic lambda values in
  `{0.25,0.5,0.75}`.
- `cmixup_offline_v1` uses only outer-train eligible rows, an admissible
  planned distance of at most 100 mm, encoder-derived joint distance at most
  20 degrees, a Gaussian probability from raw Vive-label distance, train-only
  sigma equal to the median nearest admissible label distance, and
  `lambda ~ Beta(2,2)`. The base seed is `20260818` plus a deterministic
  scheme/fold derivation.

Linear quantities and the raw Vive XYZ label use convex linear interpolation.
Euler angles use the shortest wrapped arc in `[-pi,pi)` before sine/cosine
regeneration. IDs, timestamps, sample counts, fold roles, and QC flags are not
interpolated.

Every synthetic row records `synthetic_id`, `augmentation_method`, split
scheme, outer/inner fold and `split_context`, `anchor_trial_uid`, `anchor_replica_index`,
`replacement_for_missing_anchor`, `parent_a_trial_uid`,
`parent_b_trial_uid`, `mix_lambda`, parent planned/label/joint distances,
`cmixup_sigma_mm`, `derived_seed`, source/config hashes, and sample weight.
Replica index `0` denotes the anchor's primary scheduled row; a positive value
identifies another use of that admissible anchor. A row with
`replacement_for_missing_anchor=true` fills fold-level coverage for an
original anchor that had no legal pair, without fabricating that missing
anchor's identity. The release manifest reports `missing_anchor_count` and
`replacement_count` for every scheme/fold/method. Synthetic rows have no
fabricated real segment, target, timestamp, or measurement-quality claim.

Validation fails for a parent outside the corresponding outer train set,
self-pairing, duplicate synthetic IDs, non-finite values, bound/convexity
violations, forbidden feature columns, row-count/mass drift, synthetic
validation/test data, inconsistent replica/replacement provenance, or
nondeterministic content hashes.

The packaged analysis includes a warning-free strict-QC sensitivity cohort
without changing eligibility: 7,670 of the 7,702 eligible rows have no warning
flag. It reports full-versus-strict label distributions, within-`(segment,
target)` repeatability, and between-segment target-centroid drift. It also
records count-valued checks for split duplication, prohibited primary
Tracker/planned features, parent-context leakage, self-pairs, synthetic ID
collisions, non-refit materialization, and synthetic rows entering real split
assignments. Every leakage-check count must be zero.

## Release Layout

For the audited member inventory, table shapes, join keys, weight semantics,
and the relationship to the endpoint experiment package, see
[DATA_PACKAGE_STRUCTURE.md](DATA_PACKAGE_STRUCTURE.md).

```text
ml_preparation_v1/
  README.md
  manifest.json
  config.resolved.json
  references.json
  canonical/canonical_real.parquet
  canonical/column_schema.csv
  canonical/feature_sets.json
  canonical/source_schema.json
  splits/split_assignments.parquet
  augmentation/<scheme>/<fold>/<method>.parquet
  augmentation/provenance.parquet
  analysis/audit.json
  analysis/report.md
  analysis/*.csv
```

`manifest.json` contains the contract/config/source identities, exact row
counts, schema fingerprints, seed derivations, and member SHA-256 values. The
release ZIP is tracked through Git LFS; routine generated runs stay ignored.

The release contains only materialized outer-refit augmentation. A requested
segment-LOSO inner-training context can be reproduced outside the release with
the cache CLI:

```powershell
python -m imu_vive_learning.cli cache-inner `
  --run-dir artifacts/static_endpoint_v1 `
  --outer-fold 36 `
  --inner-fold 37 `
  --output artifacts/inner_cache/segment_loso_outer36_inner37
```

The cache command consumes the validated run's real rows and split contract;
its synthetic output is training-only, ignored, and excluded from the release
manifest and ZIP.

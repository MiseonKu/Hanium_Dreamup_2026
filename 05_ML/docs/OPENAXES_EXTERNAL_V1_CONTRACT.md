# OpenAXES External V1 Preparation Contract

## Status and purpose

`openaxes.external.preparation.v1` is an isolated, preparation-only contract
for a secondary objective-portability benchmark. It does not add samples to
the SO101/Vive dataset, does not reuse SO101 model weights, and is not an
independent confirmation of the SO101 result. The OpenAXES robot, sensors,
feature schema, target frame, and target unit differ from the local system.

Preparation performs no model fitting, hyperparameter selection, challenge
access, GPU work, or hardware operation. The source notebook and publisher
programs are never executed. The source ZIP and prepared artifacts remain
under ignored `artifacts/` paths and must not be committed.

## Authenticated source

The only accepted source is the official OpenAXES Example Robot Dataset:

- DOI: `10.25835/psqqs9w0`
- expected filename: `openaxes-example-robot-dataset.zip`
- exact size: `423,808,715` bytes
- SHA-256:
  `78f31d1967842e693c2efb6ee0530e162cd509e4fa786745a485d502f2aa6da7`
- outer ZIP: exactly 97 entries and `505,766,974` uncompressed payload bytes
- nested `measure_raw-2022-09-15.tar.bz2`: exactly 1,226 entries and
  `908,374,762` payload bytes

The expected filename is enforced, not informational. The reader opens it once
as a regular `O_NOFOLLOW` file descriptor. Size, SHA-256, `ZipFile`, CRC scans,
and every consumed member are bound to that same handle. Pre/post `fstat`
receipts must keep device, inode, size, mtime-ns, and ctime-ns unchanged, and a
post-read path lookup must still resolve to that inode. In-place mutation,
removal, or atomic path replacement fails closed.

The reader also verifies entry counts, payload sizes, duplicate names,
encryption, links, and unsafe TAR member types. Absolute paths, `..`, `.`, empty
components, backslashes, colons, Windows drive forms such as `C:/`, non-NFC or
trailing-dot/space components, and case-folded canonical aliases are rejected.
It streams members without unpacking the archive. A changed source, damaged
member, schema drift, or unsafe path fails closed.

The data portal declares CC-BY-SA-3.0 while the ZIP's `LICENSE.txt` assigns
CC-BY-SA-4.0 to the measurement tree. The manifest records this conflict,
requires DOI attribution, and sets redistribution to false. Resolve the
license discrepancy before publishing any derived data. This contract is not
legal advice.

## Consumed and rejected members

The preparation consumes only:

- 32 raw IMU CSVs: four sensors in each of eight scenarios;
- eight raw robot-controller CSVs for TCP XYZ truth;
- 32 publisher-generated `*-aligned.csv` files from the nested TAR, using
  their identical `TIME` column only.

It does not consume publisher reconstructed TCP files, plots, MAT files,
videos, generated angles, or notebook results. The unavailable `dsense_vis`
generator therefore is not part of this benchmark.

The eight scenario clusters are:

- `circle-100`, `circle-150`, `circle-200`, `circle-250`;
- `triangle-100`, `triangle-150`, `triangle-200`, `triangle-250`.

They came from one installation and one short recording session. The three
trajectory repetitions inside a scenario are correlated repeats, not new
folds or independent experiments.

## Feature and target boundary

Each IMU contributes exactly nine continuous channels:

- ADXL355 acceleration XYZ;
- BMI160 acceleration XYZ;
- gyroscope XYZ.

The model feature view is exactly four sensors times nine channels, or 36
columns. The raw quaternion columns are required to be zero in every source
row and are excluded.

The target is robot-controller `tcp_x`, `tcp_y`, `tcp_z`, renamed to
`tcp_x_m`, `tcp_y_m`, `tcp_z_m`. It is a UR5 robot-base TCP position in metres.
It is not Vive-world millimetres and must not be pooled with SO101 labels.

The prepared parquet retains `row_uid`, `scenario_id`, `sample_index`, and
`aligned_time_us` only as non-feature metadata. The model feature allow-list
does not include those fields. The following are also forbidden as inputs:

- absolute or relative `TIME`, `INDEX`, MAC, IMU identity, shape, or speed;
- raw quaternion placeholders;
- robot orientation, `q0..q5`, TCP velocities, TCP angular velocities, or
  accelerations;
- any target or publisher-reconstructed TCP value.

The forbidden-input list names both raw `tcp_x/y/z` and metre-renamed
`tcp_x_m/y_m/z_m` targets, plus `normalized_tcp_*`, `normalized_tcp_*_m`, and
`tcp_*_normalized` aliases. Renaming or normalizing a target never turns it
into an input.

`scenario_id` is used only to enforce blocked folds. Training code must select
features from the manifest allow-list rather than infer them from dataframe
dtypes.

## Clock, bias, and interpolation rules

The four BLE streams begin 3.63--4.21 seconds apart. Joining by CSV row number
is forbidden. For every scenario and sensor, the preparation fits a
deterministic Huber-IRLS affine relation between the consecutive packet
`INDEX` and BLE `TIME`. This input-only fit suppresses packet-arrival jitter;
it does not inspect robot labels.

The first 20 gyro samples must have per-axis standard deviation no greater
than `0.01 rad/s`. Their per-axis mean is subtracted from that sensor's gyro
sequence. This rule is label-free and identical in all scenarios. It is
important for the anomalous circle-200 humerus stream, whose observed prefix
bias norm is about `0.469 rad/s`. Circle-200 robot truth begins during the
calibration movement, so the manifest marks its calibration truth incomplete.

The robust clock fit and quiet-prefix correction are frozen, per-recording
input calibrations. They are computed from each scenario's raw IMU inputs,
including a held-out scenario, but never from TCP targets, fold membership, or
model results. They must not be tuned or conditionally enabled by outer-fold
performance. This is distinct from fitting train-time statistics; protocols
that prohibit even fixed test-input sensor calibration must treat this
benchmark as incompatible rather than silently changing the preparation.
Because the affine clock fit sees the complete held-out recording, the
preparation is explicitly offline-transductive and acausal. It cannot support
causal, streaming, or online-deployment claims.

Robot `timestamp` uses a different clock domain. The nested publisher
alignment is an affine clock mapping derived by matching Madgwick-cal IMU
orientation to robot joint orientation. It is therefore explicitly
label-assisted. It is accepted only as frozen acquisition metadata. No model,
prediction, candidate, or outer fold may refit scale or offset.

The source mapping is checked independently for all four aligned robot segment
files. After intersecting the four raw IMU time spans:

- source robot rows: `39,793`;
- robot rows inside the four-IMU common span: `39,301`;
- unmatched rows: exactly `492`.

The 492 rows are circle-100 tail 195, circle-200 tail 60, and triangle-250
head 237. Any other accounting fails closed.

Features are linearly interpolated on robust IMU clocks and target XYZ on the
one frozen publisher robot clock. The output grid is exactly 100 Hz and no
extrapolation is allowed.

Preparation freezes, but does not execute, the downstream alignment
sensitivity recipe. Its exact signed target shifts are
`[-50, -20, -10, +10, +20, +50] ms`. A positive shift means target lookup at
`base_aligned_time_us + shift_ms * 1000`, so the sampled target is later than
the prediction time; the paired magnitudes are `±10`, `±20`, and `±50 ms`.
Rows without shifted-target support are dropped, never
extrapolated. Shifted and zero-shift results must pair on the same base
`row_uid`.

Every signed shift must be reported per outer fold over all eight primary LOSO
folds and both stress families, with family summaries. Required quantities are
`rmse3d_m`, `robust_J_m`, and the signed shift-minus-zero deltas for each.
RMSE is the root mean squared Euclidean TCP-XYZ error in metres. The frozen
outage score is `robust_J_m = (A + WS + WM) / 3`, using mean-context,
worst-segment-mean, and worst-mask-mean RMSE3D components. Reports include the
frozen comparator and objective separately, with paired method deltas signed
`objective - comparator`. Shift deltas are signed `shifted - zero` on the same
base UID pairs. Sensitivity outcomes may not select a model, alignment, or
hyperparameter and may not decide a claim gate.

### Read-only signed-shift truth API

Downstream code must not depend on private archive helpers. The stable API is:

```python
truth, receipt = reconstruct_openaxes_signed_shift_truth_v1(
    source_path, config, prepared_run_dir
)
```

It authenticates the prepared manifest/Parquet/COMPLETE envelope, then uses
one authenticated source FD/session to replay the canonical zero-shift base
frame and compare every prepared value exactly. It reconstructs all six shifts
without writing an artifact. The returned long table has exactly:

- `row_uid`, `scenario_id`, `base_aligned_time_us`;
- `shift_ms`, `shifted_target_time_us`;
- `shifted_tcp_x_m`, `shifted_tcp_y_m`, `shifted_tcp_z_m`.

Order is scenario, configured signed shift, then base time. There are no IMU
features, model outputs, metrics, selections, or gate decisions. Every
`(row_uid, shift_ms)` pair is unique and refers to a canonical base UID. Query
times outside source target support are dropped before interpolation.

The deterministic receipt binds source, manifest, Parquet and COMPLETE hashes,
the base-identity hash, exact schema/order, rows per shift, edge-drop counts,
and a binary canonical long-table hash. This API does not replace the semantic
validator: production model loading must first require
`validate_openaxes_external_v1(...)["status"] == "ok"` and a preparation stage
whose `downstream_model_results_blocked_by_uncommitted_contract` is exactly
false.

## Frozen active-trace rows

Calibration, approach, and return motions repeat across scenarios and could
inflate a blocked test. Only the three programmed active traces are prepared.
Their inclusive robot source-row boundaries are frozen in
`configs/openaxes_external_v1.yaml` before model results exist.

The annotation uses robot truth solely to define the evaluation phase. The
manifest discloses this label-side use and records the exact source rows and
geometry. Every selected circle row must remain on its frozen X plane and
approximately 0.2 m YZ circle. Every triangle row must remain on its frozen X
plane and three-edge YZ path. A source row leaving the tolerance fails closed;
the preparation never searches for a better boundary.

The authenticated smoke produced 22,482 active 100 Hz rows:

| Scenario | Prepared rows |
| --- | ---: |
| circle-100 | 3,577 |
| circle-150 | 2,403 |
| circle-200 | 1,812 |
| circle-250 | 1,457 |
| triangle-100 | 5,042 |
| triangle-150 | 3,417 |
| triangle-200 | 2,620 |
| triangle-250 | 2,154 |

These are temporally correlated rows from eight scenario clusters. Statistical
sample size must never be reported as 22,482 independent observations.

## Fold declarations

Preparation declares folds in the manifest but never assigns fold-conditioned
train-time statistics, fits fold-conditioned preprocessing, trains a model, or
reads results:

- eight primary leave-one-complete-scenario-out folds;
- four speed-pair stress folds, holding both shapes at one speed;
- two shape stress folds, holding all four speeds of one shape.

This is 14 overlapping blocked evaluations, not 14 independent replications.
Any future selection must use only scenario-blocked inner groups belonging to
the outer training set. Outer-test fitting, row-random splits, and splitting
the three loops across train/test are prohibited.

The benchmark can support only a secondary statement such as whether a frozen
outage objective improves over its comparator in the same direction on this
different dataset. It cannot establish SO101 deployment validity, independent
confirmation, cross-installation generalization, or robustness to observed
real sensor failures. All outage masks would be simulated.

## Immutable output and commands

The standalone CLI avoids modifying the shared workspace CLI:

```bash
cd 09_SO101_IMU_VIVE_LEARNING
env -u PYTHONPATH .venv312/bin/python -m imu_vive_learning.openaxes_external_v1_cli prepare \
  --source artifacts/openaxes_external_v1/source/openaxes-example-robot-dataset.zip \
  --config configs/openaxes_external_v1.yaml \
  --output artifacts/openaxes_external_v1/prepared_v1
```

Preparation refuses an existing output and atomically publishes exactly:

- `prepared.parquet`;
- `manifest.json`;
- `COMPLETE.json`.

`COMPLETE.json` binds source, config, manifest, and parquet hashes. Validation
reopens the authenticated source, recomputes all 22,482 rows in memory, checks
the exact file set and hashes, and compares every parquet value exactly:

The manifest additionally locks the YAML, all three OpenAXES Python sources,
the test, this contract, and `pyproject.toml` by path, byte count, and SHA-256.
It records Git HEAD and the complete porcelain dirty-path snapshot. Python,
NumPy, pandas, PyArrow, SciPy, PyArrow SIMD, NumPy/SciPy BLAS-LAPACK, platform,
and thread-environment receipts are sealed into the manifest; their hashes and
source hashes are repeated in `COMPLETE.json`.

Preparation may be smoke-tested while these new files are uncommitted, but
such an artifact is machine-marked
`downstream_model_results_blocked_by_uncommitted_contract=true`. Before any
model result is produced, the contract sources must be committed and the
artifact must be regenerated from that Git HEAD. Preparation itself never
authorizes or runs model fitting.

```bash
env -u PYTHONPATH .venv312/bin/python -m imu_vive_learning.openaxes_external_v1_cli validate \
  --source artifacts/openaxes_external_v1/source/openaxes-example-robot-dataset.zip \
  --config configs/openaxes_external_v1.yaml \
  --run-dir artifacts/openaxes_external_v1/prepared_v1
```

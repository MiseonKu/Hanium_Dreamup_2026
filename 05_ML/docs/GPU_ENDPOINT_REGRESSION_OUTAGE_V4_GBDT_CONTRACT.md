# Outage-v4 GBDT follow-up contract

## Status and narrow objective

`alphaf.endpoint_regression.gpu_experiment.outage_v4_gbdt` is a post-v3
development follow-up for the same static, same-session, forced-single-IMU-
outage task. It compares the already selected outage-v3 development winner
with one fixed XGBoost baseline and one fixed paired 0.5 blend. It does not
pretend that another collection session exists, and it does not modify or
finalize the v3 run.

The output remains raw static-summary `vive_world` XYZ in millimetres. It is
not robot-base/TCP calibrated, dynamic state estimation, formal ShapeFK, the
repository's three-IMU-per-link `IMU-only` condition, deployment evidence, or
control authority.

## Why this follow-up exists

Only 6,635 complete measured development rows are available. Broad tabular
benchmarks continue to find tree ensembles highly competitive on medium-sized
numeric tables, while modern neural tabular methods do not dominate every
dataset. That makes a strong GBDT comparator more informative than inventing
new synthetic collection or silently adding another tuned neural candidate.
The method-family motivation comes from the
[NeurIPS 2022 tabular benchmark](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html),
[RealMLP's NeurIPS 2024 comparison](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract-Conference.html),
and the broader
[TabArena benchmark](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html).
Those papers do not establish performance on this robot dataset, and this
local combination is not claimed as a novel method.

## Deferred incumbent freeze

The static authority is
`configs/endpoint_regression_gpu_outage_v4_gbdt.yaml`. It deliberately does
not guess the still-running v3 winner or result hashes. After v3 reaches
`development_complete_challenge_labels_sealed`, the standalone `prepare`
command must:

1. run the v3 semantic development validator and require 240 valid fits, 96
   valid v3 contexts, and an unopened label seal;
2. bind the actual v3 resolved-config semantic SHA-256, development-manifest
   file SHA-256, validator-payload SHA-256, selection file SHA-256, selected
   candidate ID, and the exact 240-task result/prediction tree SHA-256;
3. snapshot the incumbent's 48 paired seed predictions (four held-out
   segments by four outages by three seeds);
4. snapshot v3's durable 6,635-row development replay and its 3,295-row
   label-blind challenge identity.

The command only reads v3. It checks the forbidden v3 markers and exact v3
file manifest immediately before publishing `incumbent.freeze.json`, then
checks them again after publishing `prepared_manifest.json` and once more
after validating the prepared run, immediately before returning. Any mismatch
or post-publication validation failure claims `_state/prepare_invalid.json`
with an exclusive create; every v4 validator rejects that marker and it is
never overwritten. Throughout v4 the copied v3 evidence remains sealed and its
config, manifest, selection, and 240-task prediction-tree bindings are
rechecked. v4 never owns a label-access marker. The separate outage-v5
contract is the sole challenge-gate owner; neither v3 nor v4 may be invoked
with `--finalize-challenge`.

The v3 sealed-development semantic validator runs in a separate Python
process and v4 accepts only its finite JSON result. The challenge-capable v3
runner and v3 data module that exposes the real label loader are never imported
into the v4 process; v4 owns only local development-replay and label-free
identity structures. Before any output write, prepare also requires the
resolved v3 input and v4 output to be different sibling trees: equality,
either ancestor direction, and every symlink alias are rejected.

`prepare` also freezes the complete v4 executable closure before any GBDT
metric is produced. The closure covers the new v4 config/training/runner and
their tests plus every imported v3/config/data/training/preprocessing/metric
source and dependency lock listed by the runner. Every worker and validator
requires the current files to match that ordered hash tree. A separate
environment snapshot binds CPython and package versions, the CUDA-enabled
XGBoost 3.3.0 build metadata, the actual loaded `libxgboost` file size and
SHA-256, the fixed XGB-parameter digest, PyTorch/CUDA versions, and the exact
four visible GPU identities. Source or environment drift fails before a GBDT
metric can enter selection.

The parent snapshot records physical GPU IDs `0..3` together with each CUDA
UUID and PCI bus ID. Every isolated worker must see exactly one device as
`cuda:0`, and its visible UUID/PCI/name must match the requested physical GPU
record. Before fitting, the worker creates an exclusive task-execution receipt
that binds the task, phase, physical GPU, config hash, source fingerprint, and
environment fingerprint. Result JSON, bundle metadata, the exact serialized
booster configuration, and the bundle-save audit are checked as one identity;
self-consistent but differently configured boosters are not accepted.
Worker startup revalidates the frozen Python, platform, NumPy, pandas,
PyArrow, PyTorch/CUDA, XGBoost build, resolved `libxgboost` path/size/hash, and
fixed-parameter digest. Only isolated CUDA visibility differs from the parent
snapshot, and that difference is authenticated by UUID/PCI mapping. Persisted
counts and GPU IDs are exact JSON integers; booleans, floats, and numeric
strings are rejected.

## Data roles and input

- Development is the same 6,635 complete, measured, train-eligible rows from
  segments 36–39 used by outage-v3.
- Challenge is the same frozen 3,295 otherwise eligible rows from segments 40
  and 41 with IMU1 forced unavailable.
- Segment 41 is the primary previously unused, but still same-session,
  diagnostic. Segment 40 is secondary and historically observed.
- Synthetic rows, pilot segments 33–35, challenge rows in fitting, and any
  coordinate-dependent row selection are forbidden.

The model input is the same 43-column `fusion_v1` sensor allow-list plus four
raw availability bits. Continuous features are standardized from complete,
unmasked train rows only. An unavailable sensor's ten standardized values are
then set to exact zero; availability stays binary. Targets are centered by
train XYZ mean and divided by one train-only isotropic scale. Vive coordinates
are labels only.

## Fixed mask-replicated XGBoost model

Each complete training UID is materialized in exactly five states:

| State | Objective probability mass |
|---|---:|
| full | 0.25 |
| drop IMU0 | 0.1875 |
| drop IMU1 | 0.1875 |
| drop IMU2 | 0.1875 |
| drop IMU3 | 0.1875 |

For `N` train rows, `K` active train segments, and `n_s` rows in a row's
segment, that replica receives weight
`N / (K * n_s) * q_mask`. Summing the five mask probabilities, each original
UID therefore contributes `N / (K * n_s)`, not one. Summing over the `n_s`
UIDs gives every active segment objective mass `N / K`; summing the `K`
segments leaves total effective mass exactly `N`.

One bundle contains three scalar XGBoost 3.3.0 boosters, one for each scaled
XYZ axis. The exact recipe is: `gbtree`, `reg:squarederror`, `hist`, 1,000
rounds, eta 0.03, depth 6, minimum child weight 5, row subsampling 1, column
subsampling 0.8, L2 5, L1 0, gamma 0, 256 bins, depthwise growth, one parallel
tree, one thread, fixed SHA-256-derived per-axis seeds, and no early stopping.
Each axis base score is the exact train-only weighted target mean.

The training module exposes development LOSO bundles only. A missing
held-out segment, an all-development/final scope, and every final-refit bundle
constructor are outside v4. Production load, resultless recovery, and full
development validation require the serialized training device to be exactly
`cuda:0`; CPU bundles are unit-test-only and are rejected as production
evidence. The development validator also invokes the exact 12-bundle CUDA coverage
audit.

A forced validation outage is an evaluation transform, not a fit argument.
Therefore v4 fits exactly `4 held-out segments × 3 paired seeds = 12` unique
bundles (36 scalar boosters), then evaluates each bundle under four outages to
produce 48 seed records. It does not perform 48 redundant fits.

## Three candidates and exact v3 score

Candidate order is also the final tie order:

1. `v3_incumbent`: the immutable v3 development winner;
2. `xgboost_mask_replicated`: the fixed GBDT bundle above; and
3. `equal_blend_incumbent_xgboost`: a 0.5/0.5 arithmetic mean in raw XYZ
   millimetres, pairing the same seed index before context ensembling and seed
   dispersion.

Each candidate has 16 contexts: four held-out development segments by four
forced single-IMU outages. Let `r[c,s,m]` be the RMSE3D of the three-seed
context ensemble:

```text
A[c]  = mean over all 16 r[c,s,m]
WS[c] = max over segment s of mean over four outages r[c,s,m]
WM[c] = max over outage m of mean over four segments r[c,s,m]
J[c]  = (A[c] + WS[c] + WM[c]) / 3
```

Selection minimizes `J`. Exact ties use mean context P95, then mean population
standard deviation of seed-level context RMSE, then listed order. Missing
contexts fail before scoring; a non-finite context gives that candidate
positive infinity. Challenge values can never select or reselect a candidate.

## Development-only execution and v5 handoff

v4 has two state-changing commands: prepare the frozen incumbent, then
run/resume the development comparison. A read-only `validate` command accepts
only the prepared and development-complete states.

```bash
python -m imu_vive_learning.gpu_outage_gbdt_v4_experiment prepare \
  --config configs/endpoint_regression_gpu_outage_v4_gbdt.yaml \
  --v3-run artifacts/gpu_endpoint_outage_v3 \
  --output artifacts/gpu_endpoint_outage_v4_gbdt

python -m imu_vive_learning.gpu_outage_gbdt_v4_experiment run \
  --config configs/endpoint_regression_gpu_outage_v4_gbdt.yaml \
  --output artifacts/gpu_endpoint_outage_v4_gbdt \
  --gpus 0 1 2 3 --resume
```

Every invocation stops at
`development_complete_challenge_labels_sealed`. v4 has no final refit,
challenge prediction, challenge metric, source override, label-loader import,
or label-access marker. The CLI does not define `--finalize-challenge`; a
legacy command containing that flag fails in argument parsing. The Python
runner also rejects `finalize_challenge=True` before inspecting or creating
any final/challenge artifact. A run containing `final/`, a completed manifest,
a label marker, evaluation, or report is rejected as outside the v4 contract.

The separate outage-v5 contract is the sole owner of any selected-component
refit, challenge prediction freeze, one-time source-label gate, and challenge
report. v5 may consume authenticated v4 development selection as an immutable
input, but it must create its own contract, source closure, run identity, and
gate evidence. Neither v3 nor v4 may be finalized.

Resume is crash-safe and closed-world. The only accepted development tree is
12 declared bundle directories, four prediction members per bundle (48 seed
evaluation records), and exactly 16 context files under each of the three
candidates. Every file reference has the exact closed schema
`path/bytes/sha256`, plus a caller-declared exact integer `rows` or `tasks`
only where required. Context replay binds the full ordered trial, segment,
target, XYZ truth, row count, and ensemble prediction—not predictions alone.
If all four UBJ files, canonical metadata, preprocessing, and all
expected predictions are complete but only `result.json` was lost, resume
authenticates and replays
that evidence and reconstructs the result without fitting again. Any partial,
extra, invalid, or semantically non-replaying attempt is atomically moved to a
numbered `recovery/` quarantine with a member hash tree; it is never deleted,
and a clean task directory is then refit. A missing aggregate GPU ledger is
reconstructed only from the four durable per-worker readiness ledgers.

Each quarantine move is preceded by a durable intent journal. Resume
deterministically completes an interrupted pre-move or post-move/pre-audit
state, authenticates the audit, fsyncs the directories, and only then removes
the intent. Worker stop sentinels are queued only after every remaining task
result has been durably reloaded and authenticated, so a fast worker's normal
exit cannot be mistaken for failure while another GPU finishes a slow task.
Late `provenance/worker_ready/development/`, `recovery/`, `development/`, and
root `development_manifest.json` evidence is excluded from the immutable
prepared snapshot; later development/recovery validators authenticate the
applicable trees independently.

## Development result and limits

v4 reports only 48 development context records and the frozen selected
candidate. It materializes no segment-40/41 error metric and has zero permitted
source-label access passes. Any challenge result belongs to v5 and must not be
described as a v4 result.

Three seeds quantify optimization/model randomness within a context; they are
not three independent experiments. The four development segments and later
challenge segments all belong to one acquisition session. Consequently v4
cannot establish independent-session generalization, normal four-IMU
performance, calibrated uncertainty, OOD safety, deployment readiness, or
control safety.

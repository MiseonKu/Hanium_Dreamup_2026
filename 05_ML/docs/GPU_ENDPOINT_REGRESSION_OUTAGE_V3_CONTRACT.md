# Prospective outage-v3 GPU endpoint-regression contract

## Status and objective

`alphaf.endpoint_regression.gpu_experiment.outage_v3` is a prospective model-
selection contract for one missing WT901C summary at inference time. It asks
whether explicit availability, training-time sensor masking, a stronger
tabular ensemble, a four-sensor chain inductive bias, and masked-feature
reconstruction improve static endpoint regression under a forced single-IMU
outage.

The output remains raw static `vive_world` XYZ in millimetres. This experiment
does not modify or reinterpret GPU v1 or robust v2. It is not formal ShapeFK,
the repository's three-IMU-per-link `IMU-only` condition, robot-base/TCP
calibration, dynamic estimation, deployment evidence, or a control model.

The corpus already exists and prior v1/v2 results are known. “Prospective”
therefore means that this v3 candidate set, score, development population,
challenge population, and label-access sequence are frozen before the v3
challenge labels are opened. It does not mean that the data came from a new
independent collection session.

## Immutable source and role boundary

The only accepted source is:

```text
../06_simulation_and_data_collect_preparation/imu_added_experiment/
  derived_dataset_archive.zip
    derived/parquet/real_motion_demo/trials.parquet
```

| Item | Frozen value |
|---|---|
| Archive SHA-256 | `ceadf164a15a3ce302f8b2ffdde450df0f6d3a91c3b0c258a28acb5de9b92d64` |
| Member SHA-256 | `1b0a821a1822f237adadb1c978ea73da02b0e4bb7f099ce085a63931e9792b54` |
| Source rows | `10,055` |
| Session | `real_motion_demo` |

Git LFS pointer text, checksum drift, member drift, schema drift, or row-count
drift fails closed. The archive is read-only.

Development rows are read from the independently authenticated immutable
`ml_preparation_v1.zip` package with SHA-256
`94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930`;
the raw derived archive is used only for the later outage challenge identity
and final label access.

Rows have two non-overlapping roles:

- Development is exactly the 6,635 existing v1 canonical, train-eligible,
  complete measured rows from segments 36–39. Every development row has all
  four IMU summaries. Its labels may be used for all v3 model and epoch
  decisions.
- Challenge uses source segments 40 and 41. Its population is selected with
  the v1 label-blind acquisition rules while exempting only
  `imu1_link_root` feature completeness and sample count. The other three
  IMUs, encoder, duration, tracking-validity metadata, and sample-count/range
  requirements remain enforced. The selected ordered `trial_uid` list, count,
  and digest are frozen without reading coordinate labels. The frozen result
  is 3,295 rows (segment 40: 1,672; segment 41: 1,623), with compact canonical
  JSON string-array SHA-256
  `fa24e77ba5ce5f333bc4bfc5d7b75dd567d098651493ec89cda8dfbe9f12c0a1`.
- Pilot segments 33–35, synthetic rows, legacy augmentation archives, and any
  row outside those roles are prohibited.

Challenge selection never filters on a Vive coordinate value, residual, or
model error. If a selected label is non-finite when the seal is opened, the
whole challenge fails instead of deleting that row after seeing labels.

Segment 40 contributed a complete subset to preparation v1 and the earlier
GPU experiments, so it is not research-process-unseen. Segment 41 was absent
from the legacy augmentation population, `alphaf.ml_preparation.v1`, GPU v1,
and robust v2 training. Both remain later segments from the same
`real_motion_demo` session; segment 41 is not an independent-session test.

## Challenge label seal

The runner must enforce the following order:

1. Validate the immutable archive identity and build challenge identity and
   eligibility using a column projection that excludes Vive coordinate
   labels and all `imu1_link_root` values.
2. Freeze the ordered challenge `trial_uid` list, count, and hash.
3. Complete all 16-context development fits, candidate selection, scaler and
   epoch decisions, and final refit using segments 36–39 only.
4. Run the frozen selected model on the label-blind challenge inputs with
   `imu1_link_root` forced unavailable, then durably hash the model,
   preprocessing state, challenge identities, and predictions.
5. Perform one source-label access pass for exactly those identities, convert
   source `vive_world` metres to millimetres, and compute the predeclared
   challenge report.

Only the selected frozen candidate is evaluated. Candidate-specific challenge
metrics, tuning, threshold changes, blending changes, or reselection are
forbidden. Any change informed by the opened challenge creates a new
development contract and makes this challenge ordinary development evidence.
The validator must consume durable run evidence and must not silently reopen
the source labels as a second evaluation pass.

## Input and outage representation

The base sensor view is the ordered 43-column `fusion_v1` allow-list: three
encoder deltas and ten features for each of four IMUs (six Euler sine/cosine,
three acceleration, and one gyro-RMS value). Four raw binary availability
features are appended in physical sensor order:

```text
imu0_base_available
imu1_link_root_available
imu2_link_mid_available
imu3_tip_available
```

The final model input has 47 values. `fusion_v1` is signal-family-major, so
code must use an explicit index map when regrouping it into four 10-feature
sensor nodes; a blind sensor-major reshape is invalid.

For every active development train context, the 43 continuous features are
standardised with means and population standard deviations fitted only on
the complete, unmasked training rows. A constant feature receives scale one.
Availability bits remain unscaled 0/1. After standardisation, all ten values
belonging to an unavailable sensor are set to exact zero and the corresponding
availability bit is zero. Zeroing raw values before standardisation is
forbidden because zero would then encode a context-dependent non-mean value.
No validation or challenge statistic may enter either feature or target
preprocessing.

Training draws one categorical state independently for every row and epoch:

| State | Probability |
|---|---:|
| All four IMUs available | 0.25 |
| Drop IMU0 | 0.1875 |
| Drop IMU1 | 0.1875 |
| Drop IMU2 | 0.1875 |
| Drop IMU3 | 0.1875 |

At most one IMU is absent. A deterministic key over contract seed, held-out
segment, forced validation outage, paired seed index, epoch, and `trial_uid`
generates the mask. It excludes candidate identity so all trainable candidates
receive the same paired mask schedule. Development validation forces one
specified outage on every held-out row. Challenge always uses availability
`[1, 0, 1, 1]`; even a recorded segment-40 IMU1 value must not be loaded or
used.

## Six candidates fixed before development scoring

The ordered list is also the final tie order:

1. `masked_mlp`: a two-layer `[64, 64]` ReLU MLP over the 47 inputs;
2. `residual_mlp`: a width-192, depth-4 pre-normalised gated residual MLP;
3. `tabm_batchensemble`: a width-128, depth-3, eight-member TabM-style
   BatchEnsemble MLP with learned rank-one member factors;
4. `chain_endpoint_only`: a width-96, four-round chain model whose four IMU
   nodes exchange bidirectional messages over three encoder-conditioned
   neighbour edges and optimise endpoint loss only;
5. `chain_masked_reconstruction`: the same chain plus an auxiliary head that
   reconstructs the deliberately hidden sensor's ten unmasked standardised
   training features, with fixed loss
   `L = L_endpoint + 0.1 * L_reconstruction`; and
6. `equal_blend_tabm_chain_reconstruction`: the fixed 0.5/0.5 arithmetic mean
   of candidates 3 and 5 after conversion to physical XYZ millimetres. It has
   no learned blending parameter or separate fit.

The TabM candidate is explicitly a local **TabM-style BatchEnsemble**
implementation, not a claim of reproducing every detail of the reference
implementation. Parameter-efficient ensembling is motivated by
[TabM](https://arxiv.org/abs/2410.24210) and the shared-weight/rank-one member
construction by [BatchEnsemble](https://arxiv.org/abs/2002.06715). The chain
candidate adapts the general learned message/aggregation framework of
[message-passing neural networks](https://proceedings.mlr.press/v70/gilmer17a.html)
to the fixed physical sensor order. Reconstruction under deliberate input
corruption is a task-specific supervised auxiliary objective, motivated only
at the method-family level by
[denoising autoencoders](https://icml.cc/2008/papers/592.pdf). None of those
papers establishes performance on this robot dataset.

All trainable candidates use AdamW, learning rate `0.001`, weight decay
`0.001`, batch size `128`, at most `400` epochs, deterministic float32
training, and the same three paired seed values used directly as model seeds.
Every epoch uses deterministic segment-balanced batches: each batch assigns
`floor(128 / G)` rows to each of the `G` active training segments, cycles
smaller segments to the largest segment's epoch length, and averages endpoint
loss equally over the active segments. Each context checkpoint is the epoch
with minimum forced-outage validation RMSE3D; an exact tie keeps the earliest
epoch. The reconstruction
loss is computed only on artificially masked training features for which the
unmasked target is known. A truly absent challenge IMU has no reconstruction
target and one must not be invented.

## Sixteen development contexts and selection score

For every held-out development segment `s` in `{36,37,38,39}` and every
forced validation outage `m` in `{IMU0,IMU1,IMU2,IMU3}`, training uses the
other three complete development segments. This gives `4 × 4 = 16` contexts
per scored candidate. Three paired seeds give 240 actual fits for the five
trainable candidates; the sixth candidate is calculated from frozen
constituent predictions. Context ensembles and the fixed blend operate after
inverse transformation in raw Vive-world XYZ millimetres. For the blend's
seed-dispersion tie-break, constituent predictions with the same paired-seed
index are blended before calculating each seed-level RMSE.

Let `r[c,s,m]` be RMSE3D from the three-seed ensemble prediction for candidate
`c` in one context:

```text
A[c]  = mean over all 16 r[c,s,m]
WS[c] = max over s of mean over the four outages r[c,s,m]
WM[c] = max over m of mean over the four segments r[c,s,m]
J[c]  = (A[c] + WS[c] + WM[c]) / 3
```

Every context and group receives equal weight. The selected candidate has the
lowest `J`. Exact ties are resolved by lower mean context P95, then lower mean
population standard deviation of three seed-level context RMSE values, then
the listed order. A non-finite context makes that candidate's score infinite.
A missing context means that the frozen 96-context grid is incomplete, so the
run fails closed before candidate scoring rather than silently reweighting or
selecting from a partial grid. Challenge values cannot enter this calculation.

Execution is deliberately two-phase. A fresh run must stop after writing and
validating the sealed development selection. Final refit and the one-time
challenge-label projection require a separate invocation with both `--resume`
and `--finalize-challenge` against that unchanged development manifest. This
pause is the human review gate for the complete development evidence; a fresh
invocation cannot train and unseal the challenge in one command.

After selection, each final seed is refit on all development rows from
segments 36–39. Its epoch budget is the rounded median of that seed's best
epochs across the 16 development contexts, calculated separately for each
candidate and paired seed and rounded `.5` upward. If the fixed blend wins, each
constituent is refit using its own frozen epoch rule and the weights remain
exactly equal. The final-refit mask key reuses the training key recipe with
the literal held-out token `final_refit_all_development` and forced-outage
sensor field `imu1_link_root`; the latter is only a deterministic random-key
field because the categorical training distribution still contains the same
five fixed mask states. The exact final epoch state is frozen, with no
training-metric checkpoint selection.

## One-time challenge report

The challenge is forced IMU1 outage on the frozen otherwise-eligible rows from
segments 40 and 41. It reports per-segment and equal-segment macro RMSE3D/P95,
pooled RMSE3D/P95, per-axis bias/MAE/RMSE, and seed dispersion. There is no
challenge acceptance threshold and no comparison may change the model; it is
a one-time retrospective later-segment sensor-outage diagnostic.

Segment 41 is the primary previously unused diagnostic because it did not
enter the legacy analysis, augmentation preparation, GPU-v1, or robust-v2.
Segment 40 is explicitly secondary and previously observed. Consequently the
two-segment pooled and macro values are descriptive; they must not dilute or
replace the separately reported segment-41 result.

The one-pass seal is specifically the current v3 raw-source coordinate
projection, not a claim that segment 40 has never been read. Authentication of
the historical preparation release transiently deserialises its canonical
table, which contains the already-observed complete subset from segment 40.
The development API returns only the fixed 6,635 rows from segments 36–39, so
no segment-40 row or label can enter a v3 scaler, mask schedule, candidate
score, epoch choice, or final refit. Freshness applies only to segment 41.

This challenge cannot support an independent-session claim because all rows
come from one acquisition session. It cannot support a normal four-IMU claim
because IMU1 is intentionally absent. It also cannot establish calibration,
uncertainty coverage, OOD safety, runtime reliability, passive deployment, or
control authority.

## Frozen configuration and change rule

The machine-readable authority is
`configs/endpoint_regression_gpu_outage_v3.yaml`. It freezes source identity,
roles, eligibility, masking probabilities, 47-column layout, six candidates,
architectures, losses, optimiser budget, paired seeds, 16-context score,
tie-breaks, final-refit rule, challenge seal, and claim boundary.

This contract is a method hypothesis, not a novelty result. A claim of
academic novelty requires completed prospective results, ablations that show
which component matters, comparison with relevant prior work, and a claim
limited to the same-session static sensor-outage setting. No challenge label
or challenge metric was inspected to write this contract.

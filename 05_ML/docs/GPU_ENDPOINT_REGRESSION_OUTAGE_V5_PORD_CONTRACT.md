# GPU endpoint regression outage v5: PORD development contract

Status: pre-registered, same-session development-only contract. This document
describes `alphaf.endpoint_regression.gpu_experiment.outage_v5_pord`; the
machine-enforced source of truth is
`configs/endpoint_regression_gpu_outage_v5_pord.yaml` and its pinned semantic
SHA-256.

## Purpose and claim boundary

V5 studies static endpoint regression when exactly one of four WT901C feature
blocks is unavailable. The output remains raw Vive Tracker XYZ in
`vive_world`, in millimetres, from one measured three-second trial summary.
Vive is a supervised label and evaluation signal only. It is never an input
feature.

The development objective is called only the **locally proposed Paired
Outage-Risk Decomposition (PORD)**. The name does not imply a first-method
claim, a causal interpretation, a proof of robustness, or independent-session
validation. The fixed 12-atom upper-tail term is a controlled finite-atom
surrogate; it is not presented as a standard CVaR theorem. V5 is not
robot-base/TCP calibrated, ShapeFK, the repository's formal `IMU-only`
condition, a dynamic state estimator, deployment evidence, or control
authority.

The original group-mean expression
`D = R_missing - stop_gradient(R_full)` is retained only as the
`group_mean_pord` ablation. It is not pairing-specific: because
`mean(L_missing - L_full) = mean(L_missing) - mean(L_full)`, it does not use
same-UID covariance, and its segment `R_full` term cancels exactly under
two-way centering. The primary `paired_pord` instead forms the row-level atom
`A[s,m] = mean_i relu(L_missing[i,m] - stop_gradient(L_full[i]))`. ReLU before
the segment mean makes the atom depend on the joint pairing distribution.

That dependence is not an oracle-relative minimax-regret claim. PORD has no
Wasserstein ambiguity set, oracle comparator, or regret theorem. A frozen
wrong-pair control circularly deranges full losses within each segment after
sorting trial UIDs, preserving the full-loss marginal without self-pairs. A
no-stop-gradient control has the same forward value as primary PORD and changes
only the full-reference gradient. Wrong-pair evidence is bounded to this one
pre-registered sorted-UID, shift-one circular map; it makes no general claim
over all possible derangements. These controls permit a bounded empirical
pairing comparison; they do not authorize a causal, first, or universal
robustness claim.

No additional collection is possible for this study. Development segments
36--39 and challenge segments 40--41 came from the same collection session.
The only untouched diagnostic is the one-time sealed challenge evaluation;
even that is a same-session diagnostic, not an independent-session result.

## Frozen parents and data roles

Preparation semantically validates and freezes both parent runs before any v5
metric may be produced:

- the completed v3 run, with its challenge labels still sealed; and
- the completed, development-only v4 run, also with its challenge labels
  still sealed.

Both parents' challenge-access markers must be absent. Their manifests are
authenticated before and after snapshotting, copied development replay is
validated, and the label-blind challenge identity is frozen. V4's frozen
development winner is the incumbent. V5 is the protocol owner of the
challenge gate only under a frozen single-controller assumption: no v3 or v4
finalizer may run concurrently. There is no machine-wide or cross-process
mutex that makes this true automatically. V5 checks both parent access
markers before and after its label load and fails closed if either appears,
but those checks do not replace the no-concurrent-finalizer operating rule.

Parent artifact schemas remain version-specific. A sealed v3 preprocessing
reference has exactly `path`, `bytes`, and `sha256`. Its checkpoint reference
has exactly those three fields plus a positive exact-integer `tensor_count`;
this enriched reference is accepted only by the v3-parent adapter, never by
the generic v5 artifact verifier. The adapter authenticates the NPZ bytes,
loads it without pickle through the v3 checkpoint loader, and requires the
declared count to equal the actual set of non-empty tensor names with numeric,
finite values. Missing, wrong, or extra receipt fields and tampered archives
fail closed.

The exact measured-row roles are:

| Role | Segments | Rows | Interpretation |
| --- | --- | ---: | --- |
| development | 36, 37, 38, 39 | 6,635 | four leave-one-segment-out folds |
| challenge | 40, 41 | 3,295 | forced `imu1_link_root` outage |
| challenge primary | 41 | 1,623 | previously unused, same-session diagnostic |
| challenge secondary | 40 | 1,672 | previously observed, same-session diagnostic |

No synthetic row, newly collected trial, planned coordinate, target ID,
segment ID, timestamp, QC field, provenance field, Vive field, or Tracker
field may enter a model input.

## Input and view contract

The base measured view is the exact v3-compatible 43-column `fusion_v1`
ordering: three encoder deltas followed by four ten-column IMU blocks. A raw
four-bit availability vector is appended, giving exactly 47 model inputs.
Segment identity is used only for grouped splitting and aggregation and is
never a model input.

The feature scaler is fit on complete, unreplicated training rows inside the
active fold. A missing IMU block is then exactly zero in standardized space.
The target transform uses a train-only XYZ mean and one common isotropic scale;
the training API repeats that scalar three times and rejects unequal values.
Predictions are inverse transformed back to raw `vive_world` millimetres.

Every training UID is materialized after the split in exactly five views:

| View | Availability | Mask-law mass |
| --- | --- | ---: |
| full | `[1,1,1,1]` | 0.25 |
| drop IMU0 | `[0,1,1,1]` | 0.1875 |
| drop IMU1 | `[1,0,1,1]` | 0.1875 |
| drop IMU2 | `[1,1,0,1]` | 0.1875 |
| drop IMU3 | `[1,1,1,0]` | 0.1875 |

Group risks are empirical means over the complete active training fold at
each update. A minibatch estimate cannot substitute for these means.

The `constant_availability_erm` negative control uses the same continuous
five-view masks, including exact zeroing of the requested IMU block, but fixes
all four availability bits to `[1,1,1,1]` in training, checkpoint scoring, and
outer evaluation. It separates explicit availability bits from the implicit
zero-block signal; it is an interpretation-only comparison, not a challenge
eligibility gate.

## Controlled method matrix

The primary backbone is the v3-style `masked_mlp_64x64_relu`. It owns this
closed-world twelve-row table:

1. `full_gradient_only_mask_selected_control`: only the full view contributes
   training gradient, while the unchanged train-only outage criterion still
   selects its checkpoint;
2. `masked_erm`;
3. `constant_availability_erm`;
4. `group_mean_pord`, the non-pair-specific legacy ablation;
5. `paired_tail`;
6. `paired_interaction`;
7. `paired_pord`, the primary row-paired objective;
8. `paired_wrong_pair`;
9. `paired_no_stopgrad`;
10. GroupDRO;
11. V-REx; and
12. architecture-controlled MoFe.

The first row is deliberately named a **full-gradient-only,
mask-selected control**. It is not a complete no-mask pipeline because its
checkpoint is still chosen by the common four-outage train-only criterion.
This holds checkpoint exposure constant while isolating whether masked views
contribute gradient. A natural full-only checkpoint pipeline, 43D no-bit
models, unseen-outage tests, and label-shift studies are possible future
diagnostics, not part of this core grid.

The weaker `chain_endpoint_only` backbone is a separately reported transfer
table containing `masked_erm` and `paired_pord` only. It uses width 96, depth 4, expansion 2,
SiLU, LayerNorm, and zero dropout. Neither backbone has a reconstruction loss.
The closed-world validator rejects any other backbone/method allocation.

Within a fold, paired seed, and backbone, methods share the same initialization
hash, scaler, base physical five-view masks, AdamW optimizer, learning rate
`1e-3`, weight decay `1e-3`, 400 one-update epochs, and update count. One model
per fold and seed predicts full input and all four outages. After each update,
checkpoint scoring uses only the three training segments. Outer features,
labels, and UIDs are absent from the trainer API. The worker process does,
however, hold the authenticated full development snapshot, including held-out
labels. Thus the separation is enforced by the trainer API, executor code
order, and durable phase receipts; it is not a process capability sandbox.
The task execution receipt states this limitation explicitly. The earliest
strict minimum of

```text
0.5 * (mean four training-outage RMSE_3D
       + maximum four training-outage RMSE_3D)
```

fixes the checkpoint hash. Only then does a separate evaluator read and score
the outer held-out segment exactly once. The same fixed checkpoint is used for
full input and all four outages; there is no per-sensor checkpoint selection.
The history records `checkpoint_selection_scope=training_segments_only` and
zero outer evaluations, while the post-checkpoint receipt records
`outer_eval_count=1`.

The exact grid is `14 candidates x 4 folds x 3 paired seeds = 168 fits`.
It yields 192 primary outage contexts, 32 chain-transfer outage contexts, 224
outage contexts in total, 56 full-sensor contexts, and 672 fold/seed/outage
row-positive-excess summaries. Additional challenge
predictions are selected-only artifacts and do not enlarge this scientific
fit grid.

## PORD objective

For row `i`, training segment `s`, and missing sensor `m`, primary PORD uses

```text
d[i,s,m] = L_missing[i,m] - stop_gradient(L_full[i])
A[s,m] = mean over rows i in s of relu(d[i,s,m])
L_paired_PORD = L_mask + top3_symmetric(A over 12 atoms)
                + mean(abs(two_way_center(A)))
```

The ReLU is applied per row **before** the segment mean. The fixed top-3 of 12
nonnegative atoms shares the cutoff subgradient symmetrically across ties.
ReLU and absolute value use zero subgradient at zero. The interaction is the
two-way-centered nonadditivity of the paired positive atoms.

`group_mean_pord` instead uses `D=R_missing-stop_gradient(R_full)`,
`relu(D)`, and `two_way_center(D)`. Its audit validates that the latter is
exactly missing-risk nonadditivity because the full term cancels. The
wrong-pair map and hash are fixed before optimization. `paired_no_stopgrad`
must reproduce the primary forward value exactly while its full-row gradient
differs. Receipts store row excesses, atom values, pairing mode, map hash,
active tail cells and tie weights, interaction, group gradients, row-loss
gradients, and parameter-gradient hashes.

PORD is an objective-and-evaluation-protocol contribution for this fixed
availability-mask endpoint regressor. It does not introduce modality-specific
experts, a learned sensor router, a new fusion architecture, or joint
complete/incomplete Transformer fusion.

Controlled alternatives use the frozen constants: GroupDRO `eta=0.01` with
prior atom weights `1/12` for full and `1/16` for outage atoms; V-REx
`beta=1` with population-weighted group variance; and MoFe `lambda=0.1` with
the per-row hinge `relu(loss_full - loss_missing)`. Training evidence records
the exact group risks, row-paired atoms, interaction matrix, active tail atoms
and tie weights, objective terms, and gradient audits.

## Development aggregation and gates

Three paired-seed predictions are arithmetically ensembled within each held-out
segment and outage. The exact inherited v3 score is

```text
J = (mean RMSE over 16 segment/outage contexts
     + worst held-out-segment mean over four outages
     + worst sensor mean over four held-out segments) / 3.
```

Every candidate also stores row-paired
`max(outage 3D error - full 3D error, 0)` mean, P95, maximum, and positive
fraction. Missing, duplicated, or non-finite contexts fail closed before
scoring.

The masked-MLP `paired_pord` row and its attribution controls must pass every
pre-registered claim gate:

- masked ERM `J` is strictly lower than the full-gradient-only mask-selected
  control, attributing the observed difference specifically to masked-view
  gradient under the common outage checkpoint policy;
- `J` is strictly lower than masked-MLP ERM;
- worst-sensor mean is strictly lower than masked-MLP ERM;
- `J` is separately strictly lower than `group_mean_pord`, paired-tail-only,
  paired-interaction-only, paired-no-stop-gradient, the frozen wrong-pair
  control, GroupDRO, V-REx, and MoFe. The three added ablation gates are
  required evidence for the combined objective and stop-gradient choice;
- held-out-segment mean improves in at least three of four segments;
- paired-seed `J` improves in at least two of three paired seeds;
- its full-sensor score is no worse than the frozen v3 same-backbone baseline
  by more than `1e-9` mm; and
- its full-sensor score is no worse than same-v5 masked-MLP ERM by any positive
  amount (`0` mm tolerance).

The full score is

```text
F = 0.5 * (mean held-out full RMSE_3D + maximum held-out full RMSE_3D).
```

The constant-availability result is always reported but cannot make or break
challenge eligibility. The frozen v3 baseline is reconstructed fold by fold by replaying each
same-backbone forced-outage-index-1 checkpoint under the full mask and
ensembling its three seeds. The baseline is backbone-specific: masked PORD is
compared with frozen v3 masked MLP, while chain transfer is compared with
frozen v3 chain. Group-mean, paired-tail, paired-interaction, wrong-pair,
no-stop-gradient, constant-availability, full-gradient-only, and chain-transfer
results are honest secondary reports, never substitutes for a failed primary
gate. Chain `paired_pord` versus chain masked ERM is not a primary eligibility
gate. If transfer does not improve, the allowed interpretation is that the
primary evidence is specific to the masked-MLP architecture, not a
backbone-general robustness result.

V5 never uses an outer held-out label to choose a checkpoint: the trainer fixes
its checkpoint from training-segment scores, and the outer evaluator runs once.
This removes within-fit checkpoint-selection reuse of the v5 outer score.
Nevertheless, the four folds remain strictly **development evidence**, not a
confirmatory significance test or an independent-session estimate: the method
family, parents, and engineering gates arise from the same-session development
program, and there are only four acquisition segments. The three paired seeds
are correlated controlled repetitions sharing folds and data, not three
independent experiments.

The frozen v3 same-backbone full baseline is retained as a numerical
development/model-selection reference. Its earlier protocol did not implement
the v5 train-only-checkpoint/outer-once separation, so the full gate cannot be
reinterpreted as an unbiased head-to-head generalization estimate. It is a
pre-registered engineering gate only.

## Novel development selection

No challenge artifact may be finalized unless all claim gates pass. After
they pass, only two novel candidates exist:

- masked-MLP `paired_pord`; and
- the fixed raw-mm blend `0.5 * frozen v4 incumbent + 0.5 * PORD`.

Each is eligible only if its development `J` is no worse than the frozen v4
incumbent. Eligible rows are ordered by `J`, then mean P95, then population
standard deviation of seed RMSE, then listed order. If neither is eligible,
the exact terminal stage is `novelty_gate_failed_challenge_sealed`; the
challenge remains sealed and a future v6 may be designed. Challenge values
can never select, threshold, blend-weight tune, or reselect a candidate.

## Execution, provenance, and recovery

Production execution requires exactly the authenticated CUDA devices from the
prepared environment snapshot. Each device binding contains the uncoerced GPU
ID, UUID, PCI bus identity, and worker-visible CUDA mapping. CPU executors are
test-only dependencies and cannot satisfy the production path.

The controller uses only the multiprocessing `spawn` start method. It creates
one persistent process for each logical GPU slot, and each process consumes
its fixed FIFO serially. Before starting a child, the controller sets
`CUDA_VISIBLE_DEVICES` to the exact NVIDIA selector `GPU-<bare UUID>` derived
from that slot's frozen bare lowercase UUID. Identity receipts retain the bare
UUID; selector and identity are distinct contract fields, and an already
prefixed or malformed frozen identity is rejected. The child must observe the
canonical selector before Torch is first imported, must use only `cuda:0`, and
must re-read the visible device's bare UUID, PCI identity, and name and compare
them to the frozen physical row. A worker identity returned by a model executor
is never trusted. Results are restored to the pre-registered scientific task
order by controller assignment index, independently of heterogeneous worker
duration.

`workers_per_gpu` may be 1, 2, or 3 and changes scheduling only. The 168
scientific task identities remain unique. Every task binds its config, source
closure, environment, GPU, assignment, split IDs, scaler, views, initialization,
checkpoint, complete history, predictions, and result hash. Worker ledgers
live under a runtime-auth tree separate from the immutable prepared snapshot.
The global completion sentinel may be written only after all 168 task bundles
and all worker ledgers authenticate, including when workers finish at
heterogeneous times.

Prepare rejects source and output roots unless they are pairwise disjoint in
both directions. Source closure covers every imported implementation, test,
configuration, documentation file, and lock file. Current source must match
the prepared snapshot before metrics are accepted.

Resume follows a non-destructive state machine:

- a complete authenticated task bundle whose result JSON was lost is
  reconstructed from its completion evidence without rereading source data or
  refitting;
- task phases are exactly `execution.started -> training.completed ->
  outer_prediction.completed -> outer_eval.started -> outer_eval.completed ->
  bundle.completed -> result`, with an authenticated hash link at every edge;
- once `training.completed` exists, that scientific fit is never repeated;
  recovery may continue from its authenticated checkpoint to the feature-only
  prediction. It restores the authenticated frozen scaler without touching
  the label array, durably creates `outer_eval.started` with O_EXCL semantics,
  and only then materializes the held-out truth for the sole outer evaluation;
- once `outer_eval.completed` exists, recovery never reads the held-out source
  truth and never calls the evaluator again. It reconstructs the output only
  from the authenticated evaluation receipt and prediction artifact, checks
  the complete phase chain, and invokes the pure outer-evaluation verifier;
- `execution.started` without authenticated training completion is
  ambiguous and fails closed, and `outer_eval.started` without authenticated
  completion also fails closed rather than performing a second evaluation;
- quarantine uses durable intent, move, and audit records, and reconciliation
  closes crashes between any two move stages, but quarantine never grants
  authority for a replacement scientific fit; and
- missing or non-finite scientific evidence fails closed.

The Arrow schema is also closed: identity strings are exact Arrow `string`,
identity/index columns are exact `int64`, and features, truth, predictions,
and metrics are exact `float64`. Narrowed `int32`, `int8`, or `float32`
artifacts are rejected rather than coerced. Every field must retain the
writer's exact `nullable=true` flag and have no field-level metadata. Schema
metadata must contain only the canonical pandas metadata emitted by
`atomic_write_parquet`: empty index declarations; exact ordered column names
and field names; exact pandas/numpy type descriptors; null per-column
metadata; and creator versions equal to the source/environment-locked PyArrow
and pandas runtime. The canonical JSON bytes themselves are compared, so
extra keys, reordered or reformatted bytes, altered versions, and re-signed
nullable/schema-metadata drift all fail closed.

Production and synthetic testing are separate contracts. Custom validators,
snapshotters, task executors, component predictors, or label loaders require
an explicit private test-only mode. Such runs and every derived artifact are
marked `synthetic_test_only`; production validators, finalization, and parent
handoff reject them. Durable worker ledgers live under `runtime_auth/`, not
inside the immutable prepared snapshot. Standalone development and final
validators are pure and never create a missing ledger, aggregate, sentinel,
or root manifest. Only an explicit execution/resume route may perform
authenticated recovery writes.

## One-time challenge lifecycle

After development selection, the final predictor is the frozen 12-checkpoint
four-fold-by-three-seed CV ensemble; there is no all-development refit. PORD
averages the twelve checkpoints under the fixed IMU1 outage. The incumbent
replays its frozen v3/v4 component bundle under the same outage. A selected
blend averages the two raw-mm predictions with fixed weight 0.5.

Before any challenge label can be read, all component predictions, the
selected feature-only prediction, exact recipe weights, full UID/segment/target
identity, configuration, development manifest, selection, and semantic replay
receipt are written and validated. The O_EXCL freeze marker is the final
pre-label artifact. Finalization must be a separate explicit
`--resume --finalize-challenge` operation. It atomically creates the one
access marker with `O_CREAT|O_EXCL`. Before that marker, it validates and
preloads the selected prediction into memory; the marker hash-chain binds the
development manifest, selection, freeze, recipe, prediction and ordered
identity. After the label loader returns, every on-disk hash is revalidated,
but metrics use only the preloaded prediction. Labels are read exactly once,
segment 41 is primary, and segment 40 is secondary. No post-label threshold or
reselection is permitted.

The default production label loader also has no mutable non-label authority
after the access marker. Before `O_CREAT|O_EXCL`, v5 authenticates
`parent.freeze`, the challenge identity metadata and Parquet, copies and hashes
the frozen v3 resolved configuration into the v5 run, imports the final-only
loader code, and retains the validated source/configuration objects in memory.
An exact local label-source-authority descriptor and all of those references
are included in the access-marker hash chain and are rechecked after the label
read. Synthetic callback authority is a separately marked test-only contract
and cannot name or impersonate a production v3 source.

A completed challenge result is recovered only after authenticating the exact
started-to-completed link, rerunning freeze semantics, checking all 3,295
UID/segment/target rows and finite truth/prediction values, replaying overall,
segment-40 and segment-41 metrics, and matching the frozen prediction hash.
The explicit resume/finalize recovery route can reconstruct a missing or
corrupt root manifest from this completed evidence without reopening the label
source. The standalone final validator is non-mutating and rejects a present
manifest with extra fields or type drift; it also rejects missing runtime
evidence without repairing it, leaving the complete filesystem tree
unchanged. A started marker
without complete authenticated evidence fails closed; it is never retried or
overwritten. The one-time challenge remains an untouched same-session
diagnostic and cannot repair the development selection bias described above.

Standalone evidence checks are available without opening source archives or
challenge labels:

```bash
python -m imu_vive_learning.gpu_outage_pord_v5_experiment \
  --output artifacts/outage_v5 --validate-only
python -m imu_vive_learning.gpu_outage_pord_v5_experiment \
  --output artifacts/outage_v5 --validate-final-only
```

## Research context

The following primary sources motivate method families and evaluation
barriers; the frozen local configuration, not the papers, defines this exact
implementation:

- [GroupDRO, *Distributionally Robust Neural Networks for Group Shifts*](https://openreview.net/forum?id=ryxGuJrFvS)
- [V-REx, *Out-of-Distribution Generalization via Risk Extrapolation*](https://proceedings.mlr.press/v139/krueger21a.html)
- [Minimax regret optimization for robust machine learning](https://proceedings.mlr.press/v178/agarwal22b.html)
- [WDRO-MRO, an ICLR 2026 under-review regret-based multimodal proposal](https://openreview.net/pdf?id=Omf1As3tZt)
- [SimMLM / MoFe, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_SimMLM_A_Simple_Framework_for_Multi-modal_Learning_with_Missing_Modality_ICCV_2025_paper.html)
- [Masked training for sensor failure, CoRL proceedings published 2025](https://proceedings.mlr.press/v270/skand25a.html)
- [MoME modality experts and routing for sensor failure, CVPR 2025](https://arxiv.org/abs/2503.19776)
- [Joint complete/incomplete missing-modality Transformer optimization, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.pdf)
- [Blocked cross-validation for dependent data](https://doi.org/10.1111/ecog.02881)
- [Varma and Simon on separating tuning from error estimation](https://pubmed.ncbi.nlm.nih.gov/16504092/)
- [Missing-modality robustness and full-modality performance, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Maheshwari_Missing_Modality_Robustness_in_Semi-Supervised_Multi-Modal_Semantic_Segmentation_WACV_2024_paper.html)

These references do not establish that PORD is optimal or that development
gains will transfer outside this acquisition session.

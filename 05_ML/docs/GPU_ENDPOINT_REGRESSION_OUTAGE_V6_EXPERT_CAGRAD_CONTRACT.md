# GPU endpoint regression outage-v6 expert/CAGrad contract

## Status, trigger, and adaptive boundary

This document preregisters one conditional v6 development experiment. At this
revision, no v6 implementation, fit, result, artifact, or challenge-label
access exists. The v5 validator-amendment receipt below is now generated and
exactly bound, but that receipt alone authorizes no v6 implementation or run.
Execution remains fail-closed until the complete v6 code/test/source closure,
exact environment, and ten-process amendment-validation summary authenticate.

The authenticated v5 terminal stage is
`claim_gates_failed_challenge_sealed`. Trigger authentication must distinguish
two fields in two artifacts:

- `development/selection.json` has `claim_gates_passed=false`,
  `eligible_novel_candidates=[]`, `selected_candidate_id=null`, and
  `challenge_finalize_allowed=false`;
- `development/claim_gates.json` has `all_claim_gates_passed=false`.

The v5 contract had predeclared a possible v6 only for the different
`novelty_gate_failed_challenge_sealed` stage. The present claim-gate-failure
branch is therefore a newly authorized, post-v5 adaptive branch, not the branch
predeclared by v5. The absence of further collection does not make reuse of the
same development session independent. Every v6 result is same-session,
development-adaptive evidence only, not independent confirmation, unbiased
nested-CV evidence, or a significance test.

V5's correct-pair-versus-wrong-pair failure is negative evidence, so v6 contains
no PORD, pairing, tail, interaction, or regret term. The balanced budget was
also selected after all 12 v5 masked-ERM fold/seed fits chose update 400, their
allowed boundary. Those observed facts motivated this protocol and must remain
part of its limitation statement.

V6 can run exactly once after all prerequisites authenticate. A v5 success
would permanently suppress v6. Any missing, ambiguous, non-finite, or
hash-inconsistent trigger fails closed. Once a v6 terminal manifest exists,
there is no retry, retuning, renamed clone, semantic retry, or same-session
development-driven v7. A failed gate leaves the challenge sealed at
`v6_claim_failed_challenge_sealed`.

## Required v5 validator-only amendment

The original v5 trainer and run tree remain immutable. The final-replay
ndarray-to-torch validator defect may be corrected only by a separately
identified validator amendment that consumes existing v5 task checkpoints and
prediction artifacts. It must not train, change a checkpoint, invoke an outer
evaluator, recompute selection from outer labels, or access challenge labels.
Disabling current-source verification, including a
`verify_current_source=False` bypass, is forbidden.

The amendment must retain and verify the original trainer identity rather than
replace it:

- original training-source fingerprint:
  `fe6756ed599cc113dfe0dabed3cddb96be037a5fb7a0d9aefac5d8e7052d27a3`;
- original source snapshot SHA-256:
  `9d2af34cfed0b8422a8524a1ad53cf7fd2098faf3334421a24b7abb717c74e7f`;
- v5 development-manifest SHA-256:
  `c9eba0763735a5d1be763904a59eee885393c51fdf44ead3502da5ff62e78930`;
- v5 selection SHA-256:
  `f2daae3a879a1d1217ce8d1f9c450e81cdb4c0441822acd27bb85b19c46ab156`;
- v5 claim-gates SHA-256:
  `2966edcbb3529bae4df244ee6cc086efa292aab8db29bf549c4b442742bd4c5c`;
- v5 scores SHA-256:
  `b994022dd76017ed6ff25927ebc94b0e53de3366bfe0b8b847c31d55472b2b09`;
- v5 context-metrics SHA-256:
  `a7b85982595838c32546205ea2349f8bc3f0fb266e20f46c95064c10b66db217`.

The completed validator-only receipt has exact contract
`alphaf.endpoint_regression.gpu_outage_pord.validator_amendment.v1`. Its source
file is
`artifacts/gpu_endpoint_outage_v5_pord.validator_amendment_v1/VALIDATION_AMENDMENT.completed.json`,
exactly 4,324 bytes with raw file SHA-256
`dddcc746c0b1a88d349e7d4bdba79ac8cafac7af04df8eb45fbf31aa5c14522c`
and internal receipt SHA-256
`b08c8360cfb1307e2e582b55793bd691958e5929d7188d646d08fd6261b19d2d`.
The amendment source fingerprint is
`49dfbf5240fb00114f04792151f4b2638e1e85c6ec046c7a6163d28429f371f5`
at exact commit `fa472369ef93693f4089b64fbc527f6d2de2866e`.

The receipt records 168 existing-checkpoint replays, 3,000 checkpoint tensor
arrays, and 840 view replays, with zero training updates, outer-evaluator
calls, and challenge-label reads. It binds 3,082 v5-tree files totalling
69,853,604,943 bytes under tree-manifest SHA-256
`31b56cbbe3bb3e9c6aaa3bbcabdaadeeb9a72c92c710bfd13312c6c134e96cba`
and validation-result SHA-256
`e09d8c41e4b8db837872a0139004d22846bd30373cad592c186b8f2c45c8255f`.
The v5 tree was byte-identical before and after validation and its challenge
remained sealed with access count zero.

This is the first v6 preregistration commit, so every receipt and source value
is exact here; no later YAML parent-binding mutation is permitted. The receipt
is copied without alteration to `parents/v5_validator_amendment_v1.json`
inside the v6 run. The completed v6 manifest binds that path, raw file SHA,
internal receipt SHA, exact receipt contract, amendment fingerprint and commit,
while retaining the original v5 training fingerprint as a separate authority.
The fields are deliberately distinct: `parents.v5_validator_amendment.sha256`
is SHA-256 of the exact raw receipt-file bytes
(`dddcc746c0b1a88d349e7d4bdba79ac8cafac7af04df8eb45fbf31aa5c14522c`),
whereas `parents.v5_validator_amendment.receipt_internal_sha256` is the
authenticated internal receipt value
(`b08c8360cfb1307e2e582b55793bd691958e5929d7188d646d08fd6261b19d2d`).
Neither may substitute for the other.
Missing, null, placeholder, or mismatched values fail before task creation.

V6 trigger authentication is hash-only: it verifies the exact receipt, its
internal hash, and a newly computed current v5 path/bytes/SHA tree hash against
the receipt. It performs zero model/checkpoint semantic replays. In particular,
it must not call the existing amendment `validate-receipt` route, because that
route deliberately repeats 168 checkpoint and 840 view replays.

Separately, before v6 prepare, ten fresh processes must complete that read-only
amendment validation route and yield the same receipt, validation-result, and
v5-tree hashes. Across those ten validation-only passes there are 1,680
checkpoint replays, 30,000 checkpoint arrays, and 8,400 view replays, with zero
scientific fits, optimizer updates, outer-evaluator calls, selection mutation,
or challenge-label reads. Their closed summary receipt is hash-bound in the v6
parent freeze. A missing, nonzero-exit, nonidentical, or tampered pass blocks
v6. These validation-only counts never enter v6 scientific compute accounting.

All ten passes completed before this first v6 preregistration commit. Their
closed `TEN_VALIDATIONS.completed.json` is exactly 2,338 bytes with raw SHA-256
`8f1abbbaf28519216f5cc0be61ebac1f1fba410cd9f533405723cd175a4737e7`.
It records count ten, exact receipt equality, internal receipt SHA
`b08c8360cfb1307e2e582b55793bd691958e5929d7188d646d08fd6261b19d2d`,
and ten 4,324-byte stdout objects, each with SHA
`dddcc746c0b1a88d349e7d4bdba79ac8cafac7af04df8eb45fbf31aa5c14522c`.
Every status file is the exact bytes `0\n` with SHA
`9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa`,
and every stderr is empty. The 1,149-byte supervisor log has SHA
`c55da6bcafe6f8bcaf5f47d1d75ad6fd501fb47ae57afc8260b3e6d773217e95`;
the 4,858-byte supervisor script has SHA
`2620611150ea44658004872a88cf2eb48b2ae724cb559a93c2b4221902376fbf`.
Prepare copies and statically authenticates this evidence without another
semantic replay. These prerequisite passes are distinct from the ten new v6
terminal validations required after training.

## Frozen data boundary

V6 uses the same authenticated release and performs no new collection. Its
development snapshot contains 6,635 measured rows from segments 36-39, with
segment counts 1,672, 1,597, 1,707, and 1,659. Row UID
identity and all development feature and target bytes must equal the v5
snapshot. Its challenge identity contains 3,295 rows from segments 40-41 with
forced IMU1 outage. Before one-time challenge access, only challenge UID,
feature, and availability bytes may be bound; challenge target columns must be
absent. Development target bytes must never be conflated with the sealed
challenge-target boundary.

Every v3, v4, and v5 challenge-label marker must be absent. The source closure
must include the v3, v4, and original v5 sources, the separately identified v5
validator amendment, all v6 imports, configs, docs, tests, and environment
locks.

The model input is exactly 47 columns: three encoder deltas, 40 IMU values, and
four availability bits. Stored order is the exact v5 `FUSION_COLUMNS` order
followed by availability. The expert forward uses only a fixed permutation to
gather each sensor's six Euler sin/cos, three accelerometer, and one gyro-RMS
value; it creates no new feature. Segment is never a model input. `true_x_mm`,
`true_y_mm`, and `true_z_mm` are Vive-supervised labels and evaluation signals
only.

The primary split leaves out one segment. The report-only secondary split is
the already defined `int(target_id[1:]) % 5` target group over exact ASCII IDs
`T000` through `T049`; target ID and target group are split metadata, never
model inputs. Its group row counts are 1,325, 1,321, 1,328, 1,335, and 1,326.
The segment-by-group rows are, in group order 0-4:

```text
segment 36: 333 331 336 339 333
segment 37: 320 318 320 319 320
segment 38: 341 341 341 343 341
segment 39: 331 331 331 334 332
```

Feature scaling and the XYZ mean plus one isotropic target scale are fitted on
the exact training fold: three segments for primary LOSO, or all rows outside
the held target group across all four segments for secondary evaluation. The
stored three-element target scale must contain exactly equal entries. Every UID
has the vectorized training/checkpoint view
order `full, drop0, drop1, drop2, drop3`; a missing sensor block is zero after
standardization and its availability bit is zero. All five views share the same
UID and target.

## Exact environment lock

The executable environment is closed by `requirements-gpu.lock`, exactly 77
bytes with SHA-256
`fc602cce3bbc854cd4b538f43bae55dac2cdddced831b83c1ed613c116a37ab6`,
and `requirements-eval.lock`, exactly 830 bytes with SHA-256
`b72b9455129011ac88829c87cb2b0012215214ead5622838b9117a19267232aa`.
Required versions are CPython 3.12.3, torch 2.12.1+cu132, torch CUDA runtime
13.2, cuDNN integer version 92000, NumPy 2.3.5, SciPy 1.17.1, pandas 2.3.3,
PyArrow 21.0.0, and PyYAML 6.0.3.

Before torch is imported in a fit child, `PYTHONHASHSEED=0`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and OMP, MKL, OpenBLAS, and NumExpr thread
counts are all one. AMP/autocast and GradScaler are absent; CUDA-matmul TF32 and
cuDNN TF32 are false; float32 matmul precision is `highest`;
`torch.use_deterministic_algorithms(true)` is active; cuDNN benchmark is false
and deterministic mode is true. Any version or flag mismatch fails before a
task attempt.

The four physical devices are fixed as follows:

- 0: `GPU-462585f2-b4cc-3864-e27c-cbaae4ab6c9a`,
  `00000000:01:00.0`, NVIDIA CMP 170HX;
- 1: `GPU-517dfd2b-2ade-0b3f-70d7-0ea2a78f868b`,
  `00000000:2D:00.0`, NVIDIA CMP 170HX;
- 2: `GPU-b3db5781-61ae-33bb-63a0-64b14dd9ae6b`,
  `00000000:41:00.0`, NVIDIA CMP 170HX;
- 3: `GPU-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb`,
  `00000000:61:00.0`, NVIDIA CMP 170HX.

Each spawn child sees one canonical UUID selector as `cuda:0` and revalidates
the visible bare UUID, PCI ID, and name against this physical row.

## Closed 93-fit grid

The candidates, in fixed order, are:

1. `dense_7427__balanced_masked_erm` (A);
2. `availability_experts_7427__balanced_masked_erm` (B);
3. `availability_experts_7427__balanced_masked_erm_cagrad` (C, primary).
4. `availability_experts_7427__balanced_masked_erm_gmd_author_style` (D,
   report-only near-method control).

The primary segment-LOSO grid runs all four candidates over four folds and
paired seeds 20260824, 20260825, and 20260826: `4*4*3=48` fits. The secondary
unseen-target diagnostic runs A/B/C, but never D, over five held target groups
and the same seeds: `3*5*3=45` fits. The closed total is 93. Every fit runs 400
epochs. D and the entire secondary grid are report-only and cannot gate,
select, alter the challenge recipe, or authorize another same-session run.
There is no candidate, fold, seed, width, depth, loss, checkpoint, solver,
early-stop, or result-adaptive pruning choice.

Candidate A is the exact v5 dense factory, `47 -> 64 -> 64 -> 3`, with ReLU
after each of the first two linear layers and a linear three-value output. Its
parameter calculation is:

```text
weights = 47*64 + 64*64 + 64*3 = 7,296
biases  = 64 + 64 + 3             =   131
total                                  7,427
```

Candidates B, C, and D share one exact availability-gated architecture:

- four independent `10 -> 16 -> 16` experts, each with ReLU after both linear
  layers: `10*16 + 16 + 16*16 + 16 = 448` each, or 1,792 total;
- one `3 -> 16 -> 16` encoder expert with ReLU after both linear layers:
  `3*16 + 16 + 16*16 + 16 = 336`;
- PyTorch LayerNorm-16 after the second encoder ReLU, with
  `elementwise_affine=true`, `bias=false`, and `eps=1e-5`, giving 16 learned
  scale parameters;
- deterministic sensor gating `a_m * E_m(x_m)`;
- concatenation of encoder 16, four sensor embeddings totalling 64, and four
  raw availability bits, for width 84;
- an `84 -> 60 -> 3` fusion head with ReLU only after its hidden linear and a
  linear output: `84*60 + 60 + 60*3 + 3 = 5,283`.

The expert total is exactly `1,792 + 336 + 16 + 5,283 = 7,427`. There is no
learned router, reconstruction objective, dropout, imputation, or auxiliary
head. Dense and expert parameter counts match exactly. Their matrix MACs,
7,296 versus 7,188 per row excluding LayerNorm scaling, are reported and are
not claimed equal.

Within a scheme/fold and paired-seed index, its candidates share the scaler, UID
order, targets, physical views, optimizer settings, and exact balanced batch
indices. Primary B/C/D have identical initial-state hashes; secondary B/C do
too. Primary A uses the frozen v5 dense factory, model seed, and initial-state
hash, but its longer balanced
schedule intentionally produces a new checkpoint; no final v6 checkpoint or
prediction is required to equal v5. A target-group fold has no frozen-v5
target-fold scaler, batch, or initial-state counterpart, so no such equality is
claimed. Every candidate uses AdamW, learning rate
`1e-3`, weight decay `1e-3`, float32 parameters, and deterministic algorithms.

Initialization occurs on CPU in float32. Immediately before each factory,
`torch.manual_seed(model_seed)` is reset to the exact paired-seed value; CUDA
RNG is not used. A uses the frozen v5 sequential registration order. The expert
registers, in order, encoder Linear 3-to-16, encoder Linear 16-to-16,
scale-only LayerNorm-16, sensor experts 0 through 3 each with Linear 10-to-16
then 16-to-16, and fusion Linear 84-to-60 then 60-to-3. Within each Linear,
weight precedes bias. Every Linear uses the unmodified PyTorch default reset:
Kaiming uniform with `a=sqrt(5)`, followed by bias uniform within
`+-1/sqrt(fan_in)`. LayerNorm gamma is exactly one and has no bias. Custom
initialization is forbidden.

For a scheme/fold/seed, one ordered B expert state is constructed and frozen
before any B optimizer step. C, and primary D, receive byte-exact clones rather
than independent factory draws. Their names, shapes, dtypes, values, and state
hashes must match. Primary A's initial state must match the frozen v5 dense
state for that segment fold/seed.

AdamW consumes parameters in exact registration order with `lr=1e-3`,
`betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=1e-3`, `amsgrad=false`,
`maximize=false`, `foreach=None`, `capturable=false`,
`differentiable=false`, and `fused=None`. Optimizer state starts empty. Every
update begins with `zero_grad(set_to_none=true)` and has exactly one step. There
is no scheduler or gradient clipping.

## Exact balanced schedule and budget

Each primary fit has 400 epochs and a requested UID batch size of 128. Three active
training segments receive `floor(128/3)=42` UIDs each, so every optimizer update
uses 126 UIDs. Each UID contributes its same-target full view and four outage
views, producing one view-major `[5,126,47] -> [630,47]` vectorized forward.

For held-out segment `h`, zero-based paired-seed index `q`, and one-based epoch
`e`, the epoch seed is exactly:

```text
derive_context_seed(
    20260824,
    "outage_v6_segment_balanced_epoch",
    str(h),
    q,
    e,
)
```

Candidate ID and model state are excluded. One
`numpy.random.default_rng(epoch_seed)` is created and used as follows:

1. Normalize segment IDs to decimal strings and sort unique groups ascending.
   In authenticated training-row order, obtain each group's int64 indices with
   `np.flatnonzero(segment_ids == group)`.
2. In ascending group order, call that one generator's `permutation` exactly
   once for each index array. Let `L` be the largest unpermuted group length,
   `B=ceil(L/42)`, and `Q=42*B`.
3. For each group, tile its one permuted order `ceil(Q/group_length)` times and
   retain the first `Q` indices.
4. For batch index `b=0..B-1`, concatenate slices `[42b:42b+42]` in ascending
   group order. Using the same generator state after every group permutation,
   permute the 126 concatenated positions exactly once. Emit int64 indices.

No candidate or model operation advances this generator. This completely fixes
cycling, the largest-segment tail, within-batch order, and the definition of
epoch-one batch zero.

Holding out segment 36, 37, 38, or 39 gives 41, 41, 40, or 41 updates per epoch
and 16,400, 16,400, 16,000, or 16,400 updates per fit. Across three seeds, each
candidate therefore receives:

```text
3 * (16,400 + 16,400 + 16,000 + 16,400) = 195,600 updates.
```

Each of the four primary candidates receives 195,600 updates, for 782,400
primary updates. Candidate identity cannot change any batch index.

The secondary schedule balances both segment and remaining target group. With
held target group `h`, its exact 16 cells are the lexicographic Cartesian
product of segments 36-39 and groups `0..4 \ {h}`. Each update draws eight UIDs
from every cell: exactly 32 per segment, 32 per remaining target group, 128 UIDs
and 640 five-view rows. Its epoch seed is

```text
derive_context_seed(
    20260824,
    "outage_v6_target_group_cell_balanced_epoch",
    str(h),
    q,
    e,
)
```

One NumPy generator permutes each cell once in lexicographic order. If `L` is
the largest cell, `B=ceil(L/8)` and `Q=8B`; each one-time permutation is tiled
to `Q`. Batch `b` concatenates `[8b:8b+8]` from all 16 cells, then the same RNG
permutes the 128 positions once. The largest cells by held group are 343, 343,
343, 341, and 343, so every fold has 43 updates/epoch and 17,200 updates/fit.
This is not a 400-update shortcut. Each secondary candidate receives
`5*3*17,200=258,000` updates; A/B/C receive 774,000. The full experiment has
`782,400+774,000=1,556,400` optimizer updates.

Primary view risk equally averages its three training-segment means. Secondary
view risk equally averages its 16 segment-by-group cell means. Both then use
the same fixed full/outage weights. The secondary scaler and checkpoint see no
row from the held target group across any segment.

## V6 preflight, separate from scientific fits

After the validator-amendment receipt is frozen and bound, v6 consumes its v5
checkpoint/prediction replay result; v6 does not perform those 168 replays
again. V6 then has one distinct deterministic preflight over 12 primary dense
fold/seed cases and 15 secondary dense fold/seed cases.

For each case, the test input is A's `epoch=1, batch_index=0` from its exact
scheme schedule above. Two fresh spawned processes independently reconstruct
the scaler, five views, UID/target mapping, dense initial state, batch, and
direct objective. Primary cases also authenticate their frozen-v5 lineage;
secondary cases assert only their two fresh replays and seeded factory because
v5 has no target-fold counterpart. Each process performs one forward and one backward, with no
optimizer construction or step and no state update. The two processes must
have exact byte equality for the int64 batch indices, float32 objective scalar,
and every named pre-step parameter gradient in registration order. Missing,
non-finite, or unequal data fails closed.

The first balanced batch is a new v6 schedule object; no v5 batch comparison is
made. Primary preflight contributes 24 calls, 120 view-equivalents and 15,120
row-views. Secondary contributes 30, 150 and 19,200. The exact total is 54
vectorized calls, 270 view-equivalents, 34,320 row-views, and 54 gradient
traversals, with zero optimizer updates and zero scientific fits. It reads no
outer or challenge features or labels.

## ERM, CAGrad, and report-only GMD-style directions

Within each 126-UID primary or 128-UID secondary update, each row loss is mean
squared error over the three isotropically scaled XYZ coordinates. The
scheme-specific equal-mass view risks are defined above.
Candidates A and B use:

```text
g_direct = 0.25 * grad(R_full)
           + 0.1875 * sum_m grad(R_drop_m).
```

Candidate C applies CAGrad only to the four outage risks. Let:

```text
g_m   = grad(R_drop_m)
g_bar = (g_0 + g_1 + g_2 + g_3) / 4.
```

Each full/outage gradient is first materialized as finite float32 tensors for
every parameter. A missing parameter gradient fails closed. Tensors are made
contiguous, flattened, and concatenated in frozen registration order. Each
complete detached float32 vector is transferred once from CUDA to CPU and cast
once to float64. Gram construction, SLSQP, and composite arithmetic are CPU
float64; all 16 dot products form the 4-by-4 Gram matrix. With `c=0.4`, a
four-weight simplex vector minimizes the following objective for
`g_w = sum_m w_m*g_m`:

```text
dot(g_w, g_bar) + c * norm(g_bar) * norm(g_w).
```

The call is exactly `scipy.optimize.minimize(..., method="SLSQP")` with
`jac=None`, initial weights `[0.25,0.25,0.25,0.25]`, four `[0,1]` bounds, one
equality function `sum(w)-1` with no supplied Jacobian, `tol=None`, no callback,
and options `maxiter=100`, `ftol=1e-12`, `disp=false`. Acceptance requires
`success is True`, finite scalar objective, finite float64 `x` of shape four,
every bound violation at most `1e-10`, and `abs(sum(x)-1)<=1e-10`. The returned
weights are never clipped or renormalized. Failure has no fallback.

The float64 outage direction and final composite gradient are:

```text
d_out = g_bar
        + c * norm(g_bar) * g_w / (norm(g_w) + 1e-12)
g_C   = 0.25 * grad(R_full) + 0.75 * d_out.
```

The full gradient is also cast once to float64 for that final formula. Only
after the complete flat composite is formed is it cast once to float32, sliced
in frozen parameter order, assigned to `.grad`, and followed by the one AdamW
step. The `1e-12` epsilon appears only in the final `norm(g_w)` denominator;
there is no Gram/objective epsilon or extra rescaling. Exact-zero `g_bar` still
runs the same solver, and zero-`g_bar`/zero-`g_w` cases are recorded without a
shortcut. The views are correlated interventions on the same rows and the
full-risk term lies outside CAGrad, so no CAGrad convergence theorem is
transferred to this composite protocol.

Primary D is a report-only near-method control frozen against the official
GMD author repository at commit
`90e482f9f98a4e967f09e46c95c207ed7eb6fd30`. Its `optim/gmd.py` blob is
`31dc1a1fbdb3d22af54eb5cacefad7287984340a` with raw SHA-256
`85dc0ee47d54f3a63b47819d9aec448277646f6c7e3ae76d6c32fed80ed0b4a0`.
The repository is a frozen design reference, not a runtime dependency: v6 may
not clone, import, or execute it during prepare or training. The local
projection implementation and tests belong to the authenticated source
closure.
The author code's training entry point does not seed Python `random`; v6 makes
that order reproducible with one dedicated `random.Random` per fit, seeded by

```text
derive_context_seed(
    20260824,
    "outage_v6_gmd_author_style_python_rng",
    "segment_loso",
    str(held_out_segment),
    paired_seed_index,
)
```

That RNG persists through all epochs and no other operation may consume it.
For every update, the four original float32 CUDA outage gradients are retained
unchanged and a deep copy supplies mutable `g_i` in drop0-to-drop3 order. For
each `g_i`, exactly as in the frozen code, a newly allocated `[0,1,2,3]` is
shuffled and discarded, then the same original-gradient list is shuffled
in-place cumulatively. Thus every update makes four discarded no-op shuffles
and four cumulative gradient-list shuffles, interleaved per `g_i`: eight RNG
calls. It then traverses fixed indices 0-3 and performs exactly 16 tests

```text
dot = torch.dot(current_modified_g_i, original_g_j)
if dot < 0:
    g_i -= dot / ||original_g_j||^2 * original_g_j
```

The denominator is computed only for an actual conflict and must be finite and
positive. Shared coordinates are those whose `grad is not None` mask is true
for all four tasks and are averaged; nonshared coordinates are summed. There
is one sweep, no repeat or solver. D finally applies
`0.25*g_full + 0.75*g_GMD` in float32 and takes one AdamW step. Per update,
the 16 dot-test opportunities are fixed while the actual projection and norm
counts are outcome-dependent and recorded.

This is explicitly a `fixed author-code GMD-style four-outage report-only
baseline`, not a faithful paper or training reproduction: the author full-enum
uses 15 nonempty modality subsets, whereas D uses four single-outage risks; v6
adds a deterministic Python RNG, a lightweight regression expert, and the
fixed full-risk composite. The author's TensorBoard coefficient-vector writer
side effect is omitted because it never enters the merged gradient; the actual
conflicts, orders, and gradient commitments are persisted instead. No first or
novel projection claim is allowed.

## Checkpoint, outer-label, and selection boundary

At each epoch end, one train-only checkpoint pass evaluates the entire training
fold. Its full and four outage views are flattened into one `[5N,47]` forward.
The scaled float32 prediction is cast once to float64, multiplied by the exact
isotropic target scale, and shifted by the exact target mean. For each outage,
the raw-mm checkpoint RMSE is the empirical-row-pooled value

```text
RMSE_m = sqrt(mean over all training-fold rows of
              sum_xyz((truth_mm - prediction_mm_m)^2))
C = 0.5 * (mean(RMSE_0, RMSE_1, RMSE_2, RMSE_3)
           + max(RMSE_0, RMSE_1, RMSE_2, RMSE_3)).
```

This checkpoint reduction is float64 and deliberately uses the empirical row
pool, not an equal mean of split-group RMSEs. Exact strict `<`, without a
tolerance, selects the earliest minimum among 400 scores.

The trainer API and fit-child payload contain only training-fold features,
labels, segment/split metadata, UIDs, and train-only transforms; outer arrays and the full
development target snapshot are capability-isolated from the child. Each task
has the exact durable chain

```text
execution.started -> training.completed -> outer_prediction.completed
-> outer_eval.started -> outer_eval.completed -> bundle.completed -> result
```

Every phase receipt contains task, attempt, and a
`previous_phase_sha256` field. `execution.started` is the genesis receipt and
its field is the exact JSON value `null`, not an empty string or a hash. In
each of the six later phases, the field is the lowercase SHA-256 of the exact
raw bytes of the immediately preceding phase receipt; hashing canonicalized or
reserialized JSON is forbidden. After training, a
feature-only process predicts the held-out features and UIDs and records that
no outer label was accessed. A/B/D primary and every secondary fit persist the
five standard patterns. Each primary C fit instead persists all 16 patterns in
this exact order:

```text
1111,
0111, 1011, 1101, 1110,
0011, 0101, 0110, 1001, 1010, 1100,
0001, 0010, 0100, 1000,
0000
```

They are flattened pattern-major into one `[16N,47]` call before any target
access. The first five retain their preregistered full/single-outage gate
roles. The eleven extra multi/all-IMU outage cells are report-only. `0000` is
named `encoder_only_all_imu_missing_stress`: all IMU blocks and bits are zero,
while only the three encoder features remain. It is outside the author GMD
full-enum and cannot retrain, reselect a checkpoint, gate, select, or alter the
challenge. `outer_eval.started` is then created durably with
`O_CREAT|O_EXCL` before any held-out target read or materialization. Only then
may the controlled evaluator read truth exactly once. A started marker without
authenticated completion fails closed with no retry. Recovery from authenticated
completion reconstructs only from persisted prediction/evaluation receipts;
it invokes neither label loader nor evaluator and never refits. Standalone
validation is non-mutating and preserves the complete tree hash.

Thus outer held-out labels are evaluated exactly once per fit, giving 93 outer
calls. Only primary A/B/C's original five-pattern values may populate fixed
gates. D, all secondary results, and C's eleven extra patterns cannot change
training, checkpoint choice, candidates, thresholds, selection, challenge, or
another run. The final comparison remains adaptive development evidence, not
an unbiased estimate.

Only C may become challenge-eligible. There is no blend or final refit. If and
only if every development gate passes, the challenge predictor is the
raw-millimetre arithmetic mean of C's 12 frozen **primary segment-LOSO**
fold/seed IMU1-outage checkpoints. No target-group C or primary D checkpoint
may enter. Its feature-only UID predictions, recipe, component hashes, and
selection are frozen before one `O_CREAT|O_EXCL` label-access marker. Before
that marker, the controller preloads into memory the terminal development
manifest, gates, selection, challenge freeze, recipe and component hashes,
selected prediction, full UID identity, and immutable label-source authority;
the marker hash-chain binds all of them. After the marker the label loader may
read labels only, not mutable config, recipe, prediction, or identity files.
All preloaded disk references are rehashed after the loader returns, while
metrics use only the preloaded prediction bytes. Mutation fails closed.

Challenge labels can be read at most once and can never tune, threshold,
select, reselect, or alter CAGrad. Started-only recovery is permanently failed;
authenticated completed recovery and final validation perform zero label reads,
reselection, or evaluator calls and use only persisted authenticated payloads.
V3-v5 markers are checked before and after, under the explicit assumption of
one controller and no concurrent parent finalizer.

## Exact compute accounting

Scientific-fit accounting is separate from v6 preflight and from the earlier
validator-amendment receipt:

| Scientific-fit operation | Vectorized calls | View-equivalents | Row-view presentations |
|---|---:|---:|---:|
| Training updates | 1,556,400 | 7,782,000 | 988,272,000 |
| 93 fits x 400 checkpoint passes | 37,200 | 186,000 | 955,440,000 |
| One outer pass per fit | 93 | 597 | 915,630 |
| **Scientific-fit total** | **1,593,693** | **7,968,597** | **1,944,627,630** |

Primary A/B use 391,200 direct gradient traversals, C uses 978,000, and D uses
978,000. Secondary A/B use 516,000 and C uses 1,290,000. The scientific total
is 4,153,200. C makes 453,600 SLSQP calls and materializes 7,257,600 fixed
4-by-4 Gram scalar entries; outcome-dependent SLSQP objective evaluations are
not mislabeled as fixed dot-call counts. D makes 195,600 projection sweeps,
782,400 `g_i` loops, 3,129,600 fixed dot tests/projection opportunities, and
1,564,800 Python shuffles: 782,400 discarded and 782,400 cumulative. Its
actual projections and norm reductions remain outcome-dependent.

Combining the separate v6 preflight gives 1,593,747 calls, 7,968,867
view-equivalents, 1,944,661,950 row-views, and 4,153,254 gradient traversals,
still with 1,556,400 optimizer updates. The prior amendment's 168
checkpoint and 840 view replays remain separately receipt-bound and are not
rerun by the v6 trigger or silently included in v6 scientific-fit totals. The
ten prerequisite validation processes separately perform 1,680 checkpoint and
8,400 view replays; those validation-only counts are also outside this table
and are bound by their parent-freeze summary.

Each primary update uses `[630,47]`; each secondary update uses `[640,47]`;
each checkpoint pass uses one `[5N,47]`; and each outer fit uses one flattened
five- or sixteen-pattern call. Serial view loops, dummy batches, repeated
inference for utilization, and artificial load are forbidden. Four
authenticated GPUs have exactly three persistent spawn workers each. For
primary ordinal `q=3*fold_index+seed_index`, every A/B/C/D task goes to slot
`q`. Secondary ordinals 0-11 put one A/B/C task in the same numbered slot.
Secondary ordinals 12-14 use this exact extra mapping:

```text
A: 12->0, 13->8, 14->6
B: 12->4, 13->1, 14->7
C: 12->3, 13->5, 14->2
```

GPU is `slot % 4`, local worker is `floor(slot/4)`, and FIFO block order is
primary A/B/C/D, secondary A/B/C, then the assigned extra. Slots 0-8 have eight
fits and 9-11 have seven. GPUs receive 24, 23, 23, and 23 fits, with exact
update/training-call loads 401,600, 386,000, 384,400, and 384,400 and exact
gradient-equivalent loads 998,400, 1,054,800, 1,050,000, and 1,050,000.
Parameter equality does not imply equal wall time or FLOPs.

## Compact, bounded evidence

Each fit's epoch history is a Zstandard-compressed Parquet table with exactly
400 rows. Static config, scaler, UID, view, and parameter metadata are stored
once. Each epoch row records its update range and count, five mean risks,
mask-weighted loss, checkpoint score and selection flag, learning rate, state
hash, and a linked per-update gradient commitment. A/B commit, for every update,
to the composite named-parameter gradient. C commits to the full, four outage,
and composite named-parameter gradients. C also records solver counts,
residual, simplex-weight and pairwise-cosine summaries, degeneracy count, and a
linked commitment over the Gram matrix, simplex solution, solver status,
residual, and degeneracy fields.

D additionally records exact sweep, `g_i` loop, dot-test, actual-projection,
and eight-shuffle counts; Python RNG start/end hashes; and linked commitments
over the four cumulative original-gradient orders, conflict bits, original and
modified gradients, merged outage gradient, and final composite. Actual
projection counts are recorded, never assumed to equal the fixed 16 tests.

Update ranges are global, zero-based, start-inclusive, and end-exclusive.
Numeric summaries accumulate in float64. Risk and loss fields are arithmetic
means of their per-update scalar values; solver-call, solver-success, and
degeneracy fields are integer sums; simplex residual is the epoch maximum. At
each update, simplex L2 is the Euclidean norm of the four float64 weights. The
pairwise statistic is the minimum over the six outage-task gradient pairs of
`dot(g_i,g_j)/(norm(g_i)*norm(g_j)+1e-12)`. Each of those two per-update
statistics is reduced to epoch arithmetic mean, minimum, and maximum. The state
hash covers the model immediately after the epoch's final optimizer update.

These SHA-256 chains are tamper-evident commitments. They do not preserve or
permit reconstruction of raw gradients, Gram matrices, simplex weights, or
loss arrays, so no claim of full update-level reconstruction or equivalence is
allowed. Raw arrays, repeated metadata, and per-epoch checkpoint states are
forbidden. There are exactly 19,200 primary plus 18,000 secondary epoch rows,
37,200 total. Compressed history is capped at 1 MiB per fit and 97,517,568
bytes across all 93 fits; oversize evidence fails before completion. Exactly
93 selected checkpoints are stored once each.

## All-must-pass within-protocol contrasts and eligibility gates

All aggregation occurs in raw millimetres. Primary gate aggregation applies
only to A/B/C's first five patterns. A scaled float32 checkpoint output
is converted once by the exact float64 inverse target transform. For each
candidate, fold `s`, view `m`, and UID `i`, the three predictions are averaged
elementwise in fixed seed order 20260824, 20260825, 20260826 before any metric:

```text
p_bar[s,m,i] = (p[s,m,i,0] + p[s,m,i,1] + p[s,m,i,2]) / 3
r[s,m] = sqrt(mean_i sum_xyz((p_bar[s,m,i] - y[s,i])^2))
segment_mean[s] = mean_m r[s,m]
sensor_mean[m] = mean_s r[s,m]
J = (mean_{s,m} r[s,m] + max_s segment_mean[s]
     + max_m sensor_mean[m]) / 3
F = (mean_s r[s,full] + max_s r[s,full]) / 2.
```

Fold order is 36, 37, 38, 39 and outage order is 0, 1, 2, 3. D receives the
same J/F report but it is nonselecting. Secondary A/B/C use target-group order
0-4 and the analogous report-only equations over 20 outage cells: group mean
over four outages, sensor mean over five groups, and
`(mean20 + max5_group + max4_sensor)/3`; secondary full F uses five folds.

Primary C also reports all 64 ensemble fold-pattern RMSEs. The 44 cells from
the eleven added patterns are summarized by pattern (four-fold mean and max),
by missing-IMU count with pattern counts `1,4,6,4,1` (cell mean and max), and
as the four-fold encoder-only `0000` stress vector/mean/max. None is a gate.

Metrics are not averaged across seeds. For a primary paired-seed gate, the exact same `J` equation is
applied to that one seed's 16 predictions with no seed ensemble. Reductions are
finite float64. Every strict-lower gate means
`candidate + 1e-12 mm < baseline`; non-strict gates receive no hidden numeric
tolerance beyond an explicitly written tolerance such as the v3 full-view
`1e-9 mm` allowance.

The frozen v5 masked-ERM reference is exactly:

- `J=35.100751440384215 mm`;
- worst-sensor mean `38.48444497592299 mm`;
- full-view `F=34.60932697305651 mm`;
- scores SHA-256
  `b994022dd76017ed6ff25927ebc94b0e53de3366bfe0b8b847c31d55472b2b09`;
- context-metrics SHA-256
  `a7b85982595838c32546205ea2349f8bc3f0fb266e20f46c95064c10b66db217`;
- claim-gates SHA-256
  `2966edcbb3529bae4df244ee6cc086efa292aab8db29bf549c4b442742bd4c5c`.

The schedule/budget within-protocol contrast requires A to have J and
worst-sensor mean strictly below frozen v5, segment-mean improvement on at least
three of four folds, paired-seed J improvement on at least two of three seeds,
and `F(A) <= F(v5)`. The architecture within-protocol contrast applies the same
pattern to B versus A. The CAGrad within-protocol contrast applies it to C
versus B. Every strict comparison must exceed the fixed `1e-12 mm` numeric
guard. These are conditional contrasts under this fixed adaptive protocol, not
causal component isolation or independent evidence.

D, every target-group result, and C's eleven extra availability patterns are
forbidden as gate, ranking, selection, threshold, challenge, or retry inputs.
Their asymmetric report-only scope is fixed before outcomes; the absence of D
from target-group fits cannot be used to favor or disfavor C.

Capacity and overfit gates require exactly 7,427 trainable parameters for every
candidate, no capacity search, full training/outer-gap reporting, and:

```text
F(C) <= 16.477807892856227 + 1e-9 mm.
```

That v3 reference is bound by development-manifest SHA-256
`b930653581c39f9e577bff49a0ab964dde5019b0068ee57bd565de9bc67e8989`
and selection SHA-256
`58df0b7c61c20b40dbcdff7abe9b7359d6c19824ef80201dd38e26a792dff68e`.
V3 trained a separate model for each forced outage and selected epochs with
outer labels, so it is only a conservative full-performance eligibility hurdle,
not a like-for-like contrast.

Challenge eligibility also requires:

```text
J(C) <= 17.338039755106674 mm
worst_sensor_mean(C) < 19.91731611906303 mm.
```

Those v4-incumbent values are bound by development-manifest SHA-256
`13897721c53f8ba09985302bc5a3e913f3667d6aae8d24e95be8ecc1712ebac8`
and selection SHA-256
`b42c4a53b796d53b8ef849187d68323a4d4e2cfa80d03ae124f7a87843206ba3`.
Every gate is all-must-pass; a missing or non-finite value fails closed.

## Ten independent v6 terminal validations

After the authenticated v6 terminal manifest, ten fresh Python interpreters
must run the v6 read-only terminal validator. This is a new post-training
requirement, not the already completed ten-pass v5 amendment prerequisite.
Every process must exit zero, emit a byte-identical receipt, and leave the v6
tree hash identical before and after. The closed summary lives outside the
immutable run at
`artifacts/gpu_endpoint_outage_v6_expert_cagrad.validator_v1/TEN_VALIDATIONS.completed.json`;
its result-derived hash may be recorded after the run without changing config,
source, or run-tree bytes.

Each process authenticates 93 result bundles, 93 durable phase chains, 37,200
history rows, 93 selected-checkpoint file hashes, and 93 outer-prediction
bundles. Across ten processes those counts are 930, 930, 372,000, 930, and 930.
It performs zero scientific fits, updates, forwards, view-equivalents,
row-views, gradients, CAGrad solves, GMD sweeps, outer label-loader/evaluator
calls, challenge-label reads, or selection mutations. If challenge evaluation
completed, validation consumes only persisted authenticated payloads. A
missing, nonzero, nonidentical, mutating, or tampered pass leaves v6
unaccepted and fails closed without retry.

## Implementation acceptance boundary

Future implementation tests must reject duplicate YAML keys and exact-type
drift; the underscore amendment contract ID; any receipt, source, commit,
current-v5-tree, environment, GPU identity, initialization, optimizer, CAGrad,
GMD,
checkpoint, aggregation, phase-lineage, or accounting mismatch. Static trigger
authentication must have a checkpoint-predictor spy count of zero. Fresh
process tests must prove primary A's v5 initial-state match, primary B/C/D's
byte-exact clone, and secondary B/C's byte-exact clone.
Tiny-vector CAGrad tests must cover parameter order, dtype casts, exact SLSQP
arguments, bounds/sum acceptance, zero `g_bar`, zero `g_w`, and every no-fallback
failure.

GMD kernel tests bind the author commit/blob, dedicated fit seed, four
discarded plus four cumulative shuffles per update, mutable `g_i` versus
unchanged original `g_j`, `grad is not None` shared mask, shared mean,
nonshared sum, one sweep, 16 dot tests, outcome-dependent projection count,
and `0.25*full+0.75*GMD`. Target-group fixtures bind the exact ID regex/modulo,
cell table, 16-cell equal balance, 43 updates/epoch, train-only scaler, and
held-target capability boundary.

An unequal-segment checkpoint fixture must distinguish empirical row pooling
from equal-segment weighting. An outer-target spy must remain zero until the
durable started marker; a crash after that marker must fail closed on resume;
and authenticated completed recovery must succeed while both loader and
evaluator are patched to fail if called. Trainer signatures and serialized fit
payloads must contain no outer arrays or full-development target authority.
Primary C must persist all 16 feature-only predictions before its single outer
target access, and the encoder-only stress cell must be evaluated without
retraining. Tests must prove D, target-group values, and the extra eleven
patterns cannot reach gates or challenge construction. The exact 12-slot map
must cover each of 93 tasks once.
Challenge mutation during the loader must fail, and completed recovery must
read labels zero times. Raw-mm ensemble/metric order, single-seed gates, every
compute count, and a standalone-validator before/after tree hash are exact
acceptance tests. No scientific fit may start after any failed test.

## Literature and prior-art boundary

This expansion was frozen before any v6 outcome after checking primary
literature and author code. Sensor dropout is prior art
([CoRL 2017](https://proceedings.mlr.press/v78/liu17a.html)); CAGrad is prior
art ([NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html));
and gradient-guided missing-modality decoupling is prior art
([AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29474),
[frozen author code](https://github.com/HaoWang420/Gradient-guided-Modality-Decoupling/tree/90e482f9f98a4e967f09e46c95c207ed7eb6fd30)).
The D control, unseen-target diagnostic, and 16-pattern stress report were
added to narrow interpretation, not to manufacture a first-method claim.

## Claim boundary

Availability gating, CAGrad, sensor dropout, multimodal experts, and GMD-style
projection are known components and are not individually
claimed novel. At most, a passing result may describe the fixed,
parameter-matched combination and its outage-evaluation protocol as the scoped
object studied here. It may report only the three preregistered within-protocol
contrasts above; D and secondary/pattern extensions remain diagnostics. D is
an author-code-style protocol adaptation, not a paper or training reproduction.
The all-IMU-missing encoder-only cell is a stress condition, not evidence for a
supported training distribution. It cannot claim a first method, causal mechanism, proof,
theorem-backed result, independent experiment, learned modality router,
missing-data imputation, actual-hardware-outage training, or rehabilitation of
PORD or pairing.

Every report must distinguish the two scopes. Training and the primary gates
use synthetic single-IMU interventions on measured complete streams. The
eleven additional outer patterns are synthetic multi- or all-IMU availability
stress interventions; in particular, `0000` is the encoder-only,
all-IMU-missing stress cell. None of those eleven is a training distribution or
gate. Every report must also state that the evidence is an adaptive,
same-session development follow-up with no new collection.

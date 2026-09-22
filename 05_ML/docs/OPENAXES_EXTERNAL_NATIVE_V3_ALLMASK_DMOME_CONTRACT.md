# OpenAXES native v3 all-mask DMoME preregistration

This document preregisters one outcome-blind, development-only OpenAXES
experiment. The normative machine-readable contract is
`configs/openaxes_external_native_v3_allmask_dmome.yaml`. If prose and YAML
differ, execution must stop; prose never silently overrides the YAML.

The design was frozen after model-v1 had finished its ten fresh validations
and locked eleventh validation, but before the preregistration author opened
any model-v1 scientific value or checker disclosure, and before native-v2 had
an implementation, fit, prediction, or result. Native-v2 must finish before
v3 is implemented, but its outcome cannot alter, suppress, clone, tune, or
select this experiment. The historical Git receipt for the original two-file
preregistration is
`receipts/openaxes_external_native_v3_preregistration_source_audit.json`.
It reads both original preregistered files from Git objects, binds each path,
byte count, raw SHA-256, and blob OID, and binds the exact two-file source-set
digest. Its audited commit, receipt commit, erratum commit, later erratum
receipt commit, erratum-2 commit, and erratum-2 receipt commit must all be
ancestors of the implementation commit. None of these receipts contains a
scientific outcome.

## Outcome-blind erratum 1

This document and the YAML include erratum 1, which closes execution choices
that the original preregistration left underspecified. The erratum changes no
candidate, architecture, view, hyperparameter, fold, seed, metric, gate
direction, or claim boundary. Its author read no v1, v2, or v3 scientific
value, comparator payload, or result while making these decisions, and no v3
implementation, fit, prediction, result, or metric existed.

The original Git preregistration commit is
`7da0bb4e4bb10193465cf03d3e082275157aaa43`; its receipt commit is
`a6e28ac6df8bc864c81104f4e3be958804d87f6c`. The original receipt remains
byte-for-byte preserved at
`receipts/openaxes_external_native_v3_preregistration_source_audit.json`, raw
SHA-256
`cf12fc6e36a14b936ba978cbcefdd4e4db4766aae0a1bdca83ed5006297e988d`.
Implementation authority instead requires the later Git-object receipt
`receipts/openaxes_external_native_v3_preregistration_erratum1_source_audit.json`.
That receipt must bind the exact post-erratum YAML and document blobs and both
historical commits before any implementation or scientific action.

## Outcome-neutral erratum 2

Erratum 2 repairs authority and source closure only. It was prepared after a
fail-closed native-v2 signed-report exception was identified by static source
and target-support auditing, but before any native-v2 or v3 scientific outcome
was disclosed to the erratum author and before any v3 implementation, fit,
prediction, result, or metric existed. It changes no candidate, architecture,
view, hyperparameter, fold, seed, metric, gate direction, fit count, execution
count, or claim boundary.

The last complete erratum-1 authority commit is
`d54e9f08f81273d1f5a1931d00309b771853b1c6`. Its receipt remains immutable at
`receipts/openaxes_external_native_v3_preregistration_erratum1_source_audit.json`,
raw SHA-256
`88f4522c4cc4a28c8b2dd40fbe21a523dc2684bc3a43a08c7e8cdbbdb2e0e889`.
The outcome-neutral preissue record for this erratum is
`receipts/openaxes_external_native_v3_preregistration_erratum2_preissue.json`,
raw SHA-256
`fc050f08c32643ec2c126322ba0a9400b34eb241c57aeac21092383293ba8a31`.
After these amended YAML, document, and
preissue bytes are committed, a later Git-object receipt at
`receipts/openaxes_external_native_v3_preregistration_erratum2_source_audit.json`
must bind all three exact blobs. The preissue record alone cannot authorize
implementation.

V3 implementation has a compound native-v2 parent gate with exact `all_of`
semantics. First, `openaxes-native-v2-sealed-003` must have an authenticated
`development_manifest.json`, `COMPLETE.json`, and successful full validator.
Second, a separate terminal with contract
`openaxes.external.native_sensor_experts.v2.compound_recovery.terminal.v1`
must bind that exact RUN003 identity and both terminal raw references. The v3
parent receipt must bind both terminal references and their cross-equalities;
neither terminal can substitute for the other. The external terminal may
perform no fit, optimizer update, checkpoint selection, model forward, or task
evaluator call, and neither terminal's scientific values may enter the v3
parent gate. Missing, incomplete, extra, replaced, or hash-drifted authority
fails before v3 implementation.

Signed truth must come from the inherited public function
`reconstruct_openaxes_signed_shift_truth_v1` under contract
`openaxes.external.signed_shift_truth.v1`. Its exact six-file source/dependency
closure is pinned by path, byte count, raw SHA-256, and Git blob OID in YAML;
the canonical source-set SHA-256 is
`3be65b169226b8d088315311238b973c7f831cc4bf9a12116835cdba3a0a905c`.
Those inherited files are read-only. Reimplementing the reconstruction or
looking up `t + shift` in the prepared active-grid table is forbidden. The
public return and receipt require an independent v3 semantic validator before
metric use.

## Research question and claim boundary

The experiment asks whether a parameter-matched dynamic mixture of four
sensor experts is more robust than parameter-matched dense and static-expert
models when any nonempty subset of the four measured OpenAXES IMUs is
available. It separately tests whether independent-expert warm-up,
adjacent-mask More-vs-Fewer calibration, and conflict-averse cardinality
gradients improve that fixed dynamic model.

The intended contribution is an empirical protocol and a fixed combination
of known components. It is not an algorithmic-first, theorem, causal,
independent-session, cross-installation, physical-fault, online-control, or
deployment claim. All missing-sensor cases are synthetic masks over one
complete public UR5 recording. Failure of a gate is reportable negative
evidence and does not authorize retuning.

## Immutable data lineage

The sole input archive is `OpenAXES_Example_Robot_Dataset`, raw SHA-256
`78f31d1967842e693c2efb6ee0530e162cd509e4fa786745a485d502f2aa6da7`,
licensed CC BY-SA 3.0. The prepared table has 22,482 rows, 36 continuous
features, eight scenarios, and fourteen frozen outer folds. Its Parquet raw
SHA-256 is
`98747e51fd95506b91facfeb273be23c454d07da563deea13a6f2192ded48703`.
Table bytes, row order, folds, targets, and source-truth reconstruction must be
identical to the authenticated v1/v2 lineage. New collection, relabelling,
resampling, or a hidden/challenge-label route is forbidden.

Each input row contains nine continuous channels for each sensor in this
semantic order: `humerus`, `radius`, `carpus`, `digitus`. Four availability
bits are appended, giving 40 model inputs. The target is the three-coordinate
UR5 base-frame TCP position in metres. Metadata and target-derived values are
never inputs. Continuous scalers and the isotropic target transform are fit
using training-fold rows only. A missing sensor's standardized continuous
values are exactly zero; its availability bit is zero and is never
standardized.

Feature means and population standard deviations use `ddof=0` and the frozen
float64 reduction tree over authenticated training-row order. A nonfinite or
zero feature scale fails the fit. The target mean is the three-vector mean of
raw training XYZ. Its one isotropic scale is
`sqrt(mean_training_rows,XYZ((target_xyz - target_mean_xyz)^2))`, also with
`ddof=0`. Per-axis target scales are forbidden; a nonfinite or nonpositive
scale fails the fit. Standardized model tensors are cast once to float32.

## Complete nonempty availability lattice

Every row is evaluated under all 15 nonempty four-sensor masks. The exact
ordering is the YAML ordering: full view; four three-sensor views; six
two-sensor views; four one-sensor views. Every cardinality receives total
weight 1/4, and masks within a cardinality receive equal weight. Consequently
the per-mask weights are 1/4 for the full mask, 1/16 for each three- or
one-sensor mask, and 1/24 for each two-sensor mask.

The adjacent mask lattice contains exactly 28 directed rich-to-poor edges. An
edge removes exactly one sensor and may not end at the empty set: 12 edges
connect cardinality one to two, 12 connect two to three, and four connect
three to four. The same row UID and target are used on both ends of every
edge.

## Parameter-matched architectures

Every candidate has exactly 6,979 trainable scalar parameters.

The dense baseline is `40 -> 64 -> 64 -> 3` with ReLU hidden activations. Its
parameter arithmetic is
`40*64+64 + 64*64+64 + 64*3+3 = 6,979`.

The static-expert baseline has four `9 -> 20 -> 24` sensor encoders. Each
encoder has 704 parameters. The four 24-vectors are multiplied by their
availability bits, concatenated with the four bits, and passed through a
`100 -> 40 -> 3` fusion network. The four encoders contribute 2,816
parameters and fusion contributes 4,163, totaling 6,979.

The dynamic model is an output-space dynamic mixture of modality experts
(DMoME). Each of four sensor experts is `9 -> 27 -> 42 -> 3` and has 1,575
parameters. A `40 -> 15 -> 4` router has 679 parameters. Thus
`4*1,575 + 679 = 6,979`. All four experts are evaluated in frozen sensor
order. Unavailable router logits are negative infinity before softmax; the
remaining weights are nonnegative and sum to one. The final scaled XYZ
prediction is the weighted sum of expert XYZ outputs, with unavailable expert
weight exactly zero. Learned imputation, reconstruction heads, dropout,
normalization, and data-dependent sparse skipping are absent.

Parameter equality is a control for nominal capacity, not a claim of equal
compute, optimization difficulty, or statistical capacity.

Every fit is a fresh process. Python `random`, NumPy, torch CPU, and visible
CUDA device RNGs are set to the exact fit seed. Models are constructed on CPU
inside a torch RNG fork with an empty device list, using the torch-version-
bound default `Linear.reset_parameters` initialization: Kaiming uniform with
`a=sqrt(5)` and its fan-in uniform bias. Dense layers are declared in forward
order. Static experts are declared in sensor order, followed by fusion.
Dynamic experts are declared in sensor order, followed by the router. Their
within-module linear layers are declared in forward order.

For a given fold and seed, all four DMoME candidates must have identical
initial state-tensor hashes and initial optimizer-state hashes. Candidate or
process scheduling may not consume model-construction RNG state. The hash is
SHA-256 over sorted state-dict names and, for each entry without delimiters,
UTF-8 name, NumPy dtype, canonical JSON shape, and contiguous C-order bytes.
The run sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before torch import, enables
deterministic algorithms, disables cuDNN benchmarking and TF32, and uses
float32 model/training tensors without mixed precision.

## Six fixed candidates

Candidates execute in this exact order:

1. dense 6,979-parameter all-mask cardinality-balanced ERM;
2. static sensor experts 6,979-parameter all-mask cardinality-balanced ERM;
3. one-stage DMoME all-mask cardinality-balanced ERM;
4. two-stage DMoME all-mask cardinality-balanced ERM;
5. two-stage DMoME with More-vs-Fewer loss;
6. two-stage DMoME with More-vs-Fewer loss and CAGrad.

The sixth candidate is the sole fixed terminal model. There is no best-of
selection, blending, post-outcome replacement, or semantic clone.

Every fit performs exactly 400 AdamW updates. The first three candidates use
400 cooperative updates. Each two-stage DMoME performs 100 independent-expert
updates followed by 300 cooperative updates. Router parameters remain at
their initial state during stage one. The total update count is matched, but
the optimization path intentionally differs.

There is exactly one AdamW instance per fit, constructed once over all model
parameters in named-registration order. Before every update it uses
`zero_grad(set_to_none=True)`. During stage one the router has
`requires_grad=False`, is not used by the stage-one loss, receives no state,
and remains byte-identical to initialization. The four direct expert XYZ
outputs, each using only its sensor's nine channels, each form an equal-
scenario risk; the fixed mean of those four risks supplies one forward, one
backward traversal, and one AdamW step per update.

At the stage transition the same optimizer continues. Expert step counters,
first moments, and second moments are not reset; the router is enabled and
creates its AdamW state on the first cooperative update. The three two-stage
candidates must have identical model and optimizer hashes immediately after
update 100. Reconstructing the optimizer or changing its parameter groups is
forbidden.

## Losses and deterministic reductions

The row task loss is mean squared error over the three scaled target
coordinates. A scenario risk is the fixed-tree mean across rows in one
training scenario. A mask risk is the equal mean of training-scenario risks.
A cardinality risk is the equal mean of mask risks at that cardinality. The
base risk is

`R_base = (R_1 + R_2 + R_3 + R_4) / 4`.

Stage-one expert risk is the ordered equal mean of the humerus, radius,
carpus, and digitus direct-expert risks. Each expert risk first takes the
authenticated-row mean within each training scenario and then the equal mean
of scenarios in manifest order.

For the last two candidates, the cooperative objective adds the fixed
More-vs-Fewer term

`L_MoFe = mean_edges(mean_scenarios(mean_rows(relu(L_rich - L_poor))))`

over the exact 28 adjacent edges, with coefficient 0.1. This direction
penalizes cases where adding a measured sensor makes the paired row loss
worse. Its applicability to regression is literature-supported; whether it
helps this dataset is purely empirical.

Adjacent edges are enumerated with rich masks in the frozen 15-view order,
then qualifying one-sensor-smaller poor masks in that same order. Scenarios
follow each fold's manifest order and rows preserve authenticated Parquet
order. Thus edge, scenario, and row reductions are not interchangeable for
bitwise purposes.

For the terminal candidate only, each of the 300 cooperative updates treats
the four cardinality risks as four ordered tasks. Each task includes the same
`0.1 * L_MoFe` term. Let the four flattened parameter gradients be `g_k` and
`g_mean = sum_k(g_k)/4`. CAGrad chooses simplex weights `w` that minimize

`dot(g_w, g_mean) + 0.4 * norm(g_mean) * norm(g_w)`,

where `g_w = sum_k(w_k*g_k)`. Each complete task gradient is materialized in
frozen named-parameter order as float32, transferred once to CPU, and cast
once to float64. All sixteen Gram entries, the SLSQP solve, and composite
arithmetic are CPU float64.

The exact call uses `scipy.optimize.minimize` with method `SLSQP`, initial
weights `(0.25,0.25,0.25,0.25)`, four `[0,1]` bounds, equality
`sum(w)-1=0`, no Jacobians, `tol=None`, no callback, and options
`maxiter=100`, `ftol=1e-12`, `disp=false`. Acceptance requires literal solver
success, finite objective and shape-four weights, bound residual at most
`1e-10`, and simplex residual at most `1e-10`. Returned weights are not clipped
or renormalized. The final direction is

`g_mean + 0.4*norm(g_mean)*g_w/(norm(g_w)+1e-12)`.

The epsilon occurs only in that final denominator. Zero-norm cases still run
and record the same solver path. A missing task gradient, failed/nonfinite
solve, or residual violation fails the fit with no fallback or second solve.
The complete float64 direction is cast once to float32 before assignment and
the AdamW step. The four autograd traversals occur in cardinality order 1, 2,
3, 4; the first three retain the graph and the fourth releases it. Composite
assembly performs no additional backward traversal. No theorem or convergence
guarantee is transferred to this composite objective.

All metric and loss reductions use float64 and the frozen adjacent-pair binary
tree with odd-tail carry in the explicit YAML orders. Scaled predictions and
targets are cast to float64 before XYZ squared-error and loss reduction;
autograd returns float32 parameter gradients. RMSE3D is
`sqrt(mean_rows(sum_XYZ(error_m^2)))`, not mean Euclidean distance. Full-fold
updates are used; minibatching, shuffling,
mixed precision, early stopping, gradient clipping, and outer-driven restart
are forbidden. AdamW is fixed at learning rate 0.001, betas `(0.9,0.999)`,
epsilon `1e-8`, and weight decay 0.001, with non-fused/non-foreach execution.

After every optimizer update, including the 100 stage-one updates, one no-grad
eval-mode checkpoint forward computes all fifteen views in frozen order. For
each missing mask, its component is the equal mean over training scenarios of
that scenario's raw-metre RMSE3D. The score is half the mean plus half the
maximum of those fourteen components. The full-view output is computed and
recorded as a diagnostic but is absent from the score. Train mode is restored
before the next optimizer-update forward.

Stage-one checkpoints are ineligible, so the first eligible two-stage state is
global update 101. A nonfinite score fails the fit. A detached deep copy of
the complete model state replaces the current checkpoint only on a strictly
smaller finite score; equality retains the earlier update. After update 400,
the selected model is restored and irreversibly locked before any outer
feature load. Each fit then performs exactly one outer feature load, all-view
forward, label read, and evaluator call; outer information cannot change the
model or cause a rerun.

## Folds, seeds, metrics, and gates

There are three paired optimization seeds: 1701, 2603, and 3907. The primary
family has eight leave-one-scenario-out folds. Four speed-pair-out and two
shape-out folds are report-only stress tests. These overlapping folds and
paired seeds are not independent experiments.

For the exact same fold, view, and row UID, raw-metre prediction vectors are
cast to float64 and averaged coordinate-wise in seed order 1701, 2603, 3907
before every primary metric. Missing or unequal UID support fails closed.
Across the eight primary scenarios and fourteen missing masks:

- `Amean` is the mean RMSE3D;
- `WS` is the worst scenario mean;
- `WM` is the worst mask mean;
- `J = (Amean + WS + WM) / 3`;
- `F` is half the mean plus half the maximum of full-view scenario RMSE3D;
- `CR_adjacent` is the equal scenario/edge fraction for which rich-view
  Euclidean error is strictly greater than poor-view error;
- `PE_adjacent_m` is the corresponding mean positive rich-minus-poor error in
  metres.

Each context RMSE3D is computed only after the seed ensemble. A scenario fold
score is the mean of its fourteen missing-mask RMSE3Ds. A per-seed `J` uses
the same 8-by-14 `Amean`, `WS`, `WM`, and `J` formulas from only that seed,
without ensembling. Adjacent CR and PE first compute the ensemble's raw-metre
row Euclidean error, then reduce rows, scenarios, and frozen edges. Strict CR
compares rich error greater than poor; PE takes `relu(rich_error-poor_error)`.
Cardinality-specific means equally weight scenarios and masks within that
cardinality.

Router entropy, per-sensor router weight, and the six frozen signed timing
shifts are report-only as fixed below. They cannot select a checkpoint or
candidate.

Five component contrasts are fixed: static versus dense, dynamic versus
static, two-stage versus one-stage, MoFe versus two-stage ERM, and CAGrad
versus MoFe. Common passage requires strictly lower `J` and `WM`, full-view
`F` no worse than `1e-12`, strict fold-score wins in at least 6/8 primary
folds, and strict seed-level `J` wins for at least 2/3 seeds. MoFe additionally
requires strictly lower `CR_adjacent` and `PE_adjacent_m`; CAGrad requires
`CR_adjacent <= comparator + 1e-12` as a dimensionless fraction and
`PE_adjacent_m <= comparator + 1e-12 m`. Strictly lower conditions have no
numeric guard: any finite binary64 `<` counts, while equality does not.
Full-view nonharm is `F <= comparator F + 1e-12 m`.

The primary terminal claim is harder: the fixed terminal candidate must pass
every declared robust, monotonicity, full-view, fold, and seed condition
against both refit v3-law baselines, dense and static experts. Equality never
counts as a strict win. Native-v2 is report-only here because it used a
different five-view mask law.

## Fixed report-only router and timing analyses

The four DMoME candidates persist their four router weights from the same
single outer forward that produces predictions; router reporting adds no model
forward. Dense and static candidates have router value `not_applicable`.
Atomic weights are keyed by candidate, fold, seed, view, row UID, and sensor;
unavailable weights are exactly zero. Entropy is the unnormalized natural-log
quantity `-sum_available(w*log(w))`, with `0*log(0)=0`. It is computed per seed
before any summary; entropy of seed-mean weights is forbidden. Reports include
per-fold/seed/view row-mean entropy and available-sensor conditional mean
weights, cardinality-specific primary summaries, and the equal mean of the
four cardinality summaries. All summaries follow the frozen seed, scenario,
mask, sensor, and row orders. These values are interpretive diagnostics only.

Signed target shifts are `[-50,-20,-10,10,20,50] ms`. They reuse persisted
zero-shift seed-ensemble predictions and the canonical
`openaxes.external.signed_shift_truth.v1` target-only reconstruction. For base
time `t`, the target lookup is `t + shift_ms*1000 us`; thus a positive shift
scores the fixed prediction against a later target. Outer features and
predictions never move. There is no refit, checkpoint, model forward, or task
evaluator call. Reports use identical UID support for shift-minus-zero deltas.
Extrapolation is forbidden and the frozen preparation expects zero dropped
edges; any nonzero count fails only the signed report closed and cannot alter
the primary gates. The exact report world has 504 candidate/fold/shift
ensemble cells and 7,560 candidate/fold/shift/view metric cells.

The canonical public reconstruction is called exactly once per signed report.
Missing, incomplete, invalid, or exceptional reconstruction has the same
report-only consequence as a nonzero dropped-edge count: the signed report
fails closed, while primary-gate inputs, decisions, and claims remain
byte-independent of both the signed report and the external recovery terminal.

## Exact execution accounting

Six candidates, fourteen folds, and three seeds produce exactly 252 fits:
144 primary and 108 report-only stress fits. Exact declared totals are:

- 100,800 optimizer updates;
- 201,852 model forward calls;
- 2,889,180 view-equivalents;
- 138,600 parameter-gradient traversals;
- 12,600 CAGrad solves;
- 252 outer feature loads, label reads, and evaluator calls.

The 201,852 forwards are exactly 100,800 training forwards, 100,800
post-update checkpoint forwards, and 252 outer forwards. Training views split
into 756,000 for the first three all-view candidates, 50,400 for the three
two-stage four-expert warm-ups, and 567,000 for their cooperative all-view
updates. Checkpointing contributes 1,512,000 all-view equivalents and outer
prediction 3,780. The terminal candidate contributes 4,200 stage-one and
50,400 cooperative gradient traversals; the first five candidates contribute
84,000. Signed shifts and router reports add zero scientific forward,
gradient, update, fit, or evaluator call.

The view-equivalent identity is

`3*42*400*15 + 3*42*(100*4 + 300*15) + 6*42*400*15 + 252*15`

`= 756,000 + 617,400 + 1,512,000 + 3,780 = 2,889,180`.

The gradient-traversal identity is

`5*42*400 + 42*(100 + 300*4) = 84,000 + 54,600 = 138,600`.

Twelve persistent FIFO slots map by context ordinal modulo 12. Slot-to-GPU is
`slot mod 4`; local slot is `floor(slot/4)`. This assigns 11, 11, 10, and 10
contexts, or 66, 66, 60, and 60 fits, to the four authenticated GPUs. Runtime
rebalancing, GPU substitution, and artificial dummy load are forbidden. The
80% utilization target is an operational target only when useful computation
is bound by GPU work.

Fold ordinal follows the exact preparation-manifest order; seed ordinal
follows 1701, 2603, 3907; candidate ordinal follows the six-candidate order.
`context_ordinal = fold_ordinal*3 + seed_ordinal` and
`task_ordinal = context_ordinal*6 + candidate_ordinal`, yielding exactly
`task_000` through `task_251`. A slot processes context ordinal ascending and
candidate ordinal ascending. Slots 0--5 each receive four contexts and 24
fits; slots 6--11 each receive three contexts and 18 fits. The complete task
index and immutable slot rosters must exist before the first attempt.

## Validation, publication, and implementation authority

Task index, assignment, phase lineage, source closure, environment, actual
child GPU, and previous-phase raw hashes must be receipt-bound. The source and
result trees become immutable after the first task attempt. Missing, extra,
partial, symlinked, special, nonfinite, or hash-drifted state fails closed.

The terminal run requires ten fresh-process full validations, a locked
eleventh publication validation, and two independent metric recomputations
whose canonical overlap agrees exactly. No scientific disclosure is valid
before those checks pass.

Implementation may add only the exact future files listed in YAML after the
compound RUN003-plus-external-recovery parent gate, historical preregistration,
both erratum histories, erratum-2 preissue record, and final erratum-2
Git-object receipt authenticate. Earlier receipts and the preissue record are
retained but are insufficient individually. The inherited preparation public
API/config/dependency files must equal their six pinned Git blobs and remain
read-only. Existing native-v1, native-v2, SO101-v3/v6, data, fold, or evaluator
sources cannot be changed for this experiment. Any implementation deviation
requires a public fail-closed record; it cannot be repaired by
outcome-dependent tuning or retry.

## Limitations

This is evidence from one installation and one public recording containing
eight clustered scenarios. Synthetic masks do not establish robustness to a
physical outage, drift, corruption, clock error, or a different sensor
installation. Three seeds are optimization repeats. No new SO101 experiment,
independent session, cross-robot transfer, or closed-loop test is available.

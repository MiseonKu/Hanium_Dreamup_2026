# OpenAXES External Model V1 Development Contract

## Status and claim boundary

`openaxes.external.model_benchmark.v1` is a preregistered, development-only
benchmark of whether an outage-oriented objective can be ported to a different
public robot dataset. It is not an independent confirmation of the SO101/Vive
result. It does not pool SO101 rows, reuse SO101 weights, open SO101 challenge
labels, or establish cross-installation or deployment validity. OpenAXES uses a
UR5, different IMUs, a different feature schema, a robot-base TCP target in
metres, and one short public recording session.

The source contains measured complete four-IMU streams. Every one-IMU failure
condition in this benchmark is a synthetic intervention made by zero-masking
one measured stream after train-only standardization and clearing its
availability bit. No actual sensor failure was observed. “Robust to observed
outages,” “first,” “causal,” “provable,” “regret,” and statistical-significance
claims are outside this contract.

The local name is **Paired Outage-Degradation Decomposition**. Its paired term
uses a same-UID full/missing loss difference; it has no oracle comparator,
Wasserstein ambiguity set, causal identification, or regret theorem. Its
fixed-atom top-quarter term is an upper-tail surrogate, not an invocation of a
standard CVaR theorem.

No external-data metric was inspected when this contract, candidate set,
folds, and selection rule were written. The production preparation artifact
is the committed `f36ec9f` parent and records
`downstream_model_results_blocked_by_uncommitted_contract=false`. The parent
preparation Git head and the later model-contract Git head are deliberately
separate. Model execution proves, with explicit `git cat-file` commit checks
and `git show <revision>:<path>` bytes, that every parent source-closure blob is
unchanged at both heads; it separately binds the working tree. It then replays
the preparation's public validator in the parent's recorded Git/status
context. A model-only later commit is therefore allowed without pretending it
was the preparation commit.

## Authenticated preparation input

The only input contract is `openaxes.external.preparation.v1`. Before a model
run, its public `validate_openaxes_external_v1` API must return the exact closed
`status=ok` receipt after replaying the official ZIP, manifest, and parquet.
The model then freezes the following three preparation files and official
source/config authority by absolute path, size, and SHA-256:

- `prepared.parquet`;
- `manifest.json`;
- `COMPLETE.json`.

Production requires
`downstream_model_results_blocked_by_uncommitted_contract=false` in both the
validator receipt and manifest provenance. The production preparation API
accepts only source/config/preparation-run paths; it internally invokes the
public parent validator during fresh preparation and again immediately before
the first fit. A directly supplied or forged prepared-data object, a missing
path, and a synthetic fixture are rejected. A synthetic test fixture is
marked `synthetic_test_only=true` and is rejected by every production executor
and completed-result validator. The in-memory dataclass checker is deliberately
private and structural-only; it is never described or accepted as production
authority.

For this version, the existing ignored official source is
`artifacts/openaxes_external_v1/source/openaxes-example-robot-dataset.zip` and
the prepared authority directory is
`artifacts/openaxes_external_v1/prepared_v1_f36ec9f`. Those names do not bypass
the path/size/SHA-256 and public-validator checks above.

The preparation manifest is authoritative for feature order and folds. The
model never infers inputs from numeric dataframe columns. It consumes exactly
36 allow-listed continuous features: four sensors (`humerus`, `radius`,
`carpus`, `digitus`) times nine acceleration/gyroscope channels. It appends
four availability bits for an exact 40-dimensional input. `row_uid`,
`scenario_id`, `sample_index`, and `aligned_time_us` are metadata only. Shape,
speed, time, sensor identity, robot state, and target aliases are not model
inputs.

The three targets `tcp_x_m`, `tcp_y_m`, and `tcp_z_m` are label/evaluation
only. They are UR5 robot-base TCP coordinates in metres, not Vive-world
coordinates and not SO101 ShapeFK truth.

## Sealed-001 pre-fit deviation log

The first production launch, run ID `openaxes-v1-sealed-001`, was attempted at
model-contract Git head `ce09392a287f00447c5451471eae854d6aa726e6`. The
preserved terminal supervisor evidence is
`openaxes-v1-runs/openaxes-v1-sealed-001/LATEST_TERMINAL.json`, with SHA-256
`e032c62b899efd7e90647748f6c087b05d271ae1e312fc8ec64f390cc1bc20e3`.
It records a fail-closed pre-fit schema rejection. The public preparation had
completed and passed its public validator, but canonical `sort_keys=True` JSON
serialization reordered the `sensor_groups` mapping keys. The model validator
incorrectly treated that JSON mapping insertion order as scientific sensor
order. This was a key-order-only implementation defect: the exact four sensor
keys, each ordered nine-column list, and the flattened feature order were
otherwise unchanged.

The failure occurred before model preparation or execution: GPU scientific
fits `0`, outer predictions `0`, outer evaluations `0`, task result directories
`0`, execution sessions `0`, and model `PREPARED.json` markers `0`. No model
metric was produced or inspected. The model config and its frozen semantic
hash
`ce6a828e6d92118e29b7c67ec166cc3a555709e78bb41ef0838a79e982e80a3e`,
eight candidate methods, 14 folds, three seeds, selection rule, and directional
gates remain unchanged. The erratum makes mapping insertion order non-semantic
while still checking the exact key set and flattening group contents only in
`OPENAXES_MODEL_SENSOR_GROUPS` order. The failed run is retained as evidence
and is not resumable as scientific work; acceptance requires a new fresh run
from the corrected committed source.

## Sealed-002 pre-fit CUDA UUID representation deviation log

The next fresh production launch, run ID `openaxes-v1-sealed-002`, used the
source/evidence Git head `26795e6c34a7ac11995875134d8ff81758758622`.
Its preserved `RUN_SPEC.json` has SHA-256
`e86c303be806cc2e076676a4d680f8b20ef28adbf03dfea24b0df28a7d7c8e16`.
The supervisor's later `cleanup_blocked_nonterminal` current-state receipt has
SHA-256
`063644f72ad6d83eb3890d5ea0693ef3d42b0cb8f25831b2743fde87938b25e0`;
the supervisor and detached-launch logs have SHA-256
`4cad8de028568efd84c48b334b1285d38c8759318c751607c9455214007e33b8`
and
`056bc9f64d882e16cf1b139fd383ecc5224927c1e5e55a6d88f6923218a71d48`,
respectively. There is no terminal acceptance receipt for this run.

Preparation and model preparation completed, but execution stopped at the
worker identity check: model `PREPARED.json` markers `1`, execution sessions
`1`, `launch.started` receipts `1`, worker-ready receipts `0`, task results
`0`, scientific fits `0`, outer predictions `0`, and outer evaluations `0`.
The exact surviving model evidence consists of six files: the prepared
manifest, `PREPARED.json`, task index, pre-fit validation receipt, execution
session, and `launch.started` receipt. The worker-ready, phase-ledger, attempt-
ledger, attempt-roster, result, and quarantine directories contain zero files.
All marker processes were later observed absent (`0`), while the supervisor
remained fail-closed as `cleanup_blocked_nonterminal`. No model metric was
produced or inspected.

The frozen physical identity from `nvidia-smi` was
`GPU-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb`. In the sealed PyTorch `2.12.1+cu132`
runtime, `str(torch.cuda.get_device_properties(0).uuid)` represented the same
128-bit UUID without the `GPU-` prefix. The child incorrectly compared those
two valid representations as raw strings and stopped before writing its ready
receipt. This erratum accepts only the exact hexadecimal UUID grammar with an
optional case-insensitive `GPU-` prefix, restores that prefix and lowercases
hexadecimal digits solely for identity comparison, and rejects malformed or
different UUIDs. The exact frozen `CUDA_VISIBLE_DEVICES` value and the
`nvidia-smi` physical-index, PCI-bus-ID, name, and unique-live-UUID checks
remain mandatory.

The model config and its semantic hash
`ce6a828e6d92118e29b7c67ec166cc3a555709e78bb41ef0838a79e982e80a3e`,
eight candidate methods, 14 folds, three seeds, selection rule, and directional
gates remain unchanged. Sealed-002 is retained as outcome-free evidence and is
not resumable as scientific work; acceptance requires a fresh sealed-003 run
from the corrected committed source.

## Train-only transforms and five views

For each outer fold, the 36 continuous columns receive population mean/std
standardization fitted only on declared training scenarios. A zero or
non-finite scale fails closed. Targets use a three-vector training mean and one
scalar isotropic population scale fitted only on the same training rows;
per-axis target scales are prohibited.

Every training UID has exactly these five views, in this order:

1. full, availability `[1,1,1,1]`;
2. humerus masked, availability `[0,1,1,1]`;
3. radius masked, availability `[1,0,1,1]`;
4. carpus masked, availability `[1,1,0,1]`;
5. digitus masked, availability `[1,1,1,0]`.

Masking occurs after standardization, so the nine continuous values of the
missing sensor become zero. The empirical mask law is `0.25` full and
`0.1875` for each forced outage. Full and missing rows keep the same UID and
target. `scenario_id` is used only for blocked partitions and training-group
risk atoms; it never enters the network.

## Frozen backbone, optimization, and candidates

All methods use the same `40 -> 64 -> 64 -> 3` ReLU MLP, paired CPU
initialization, train-only transform, five-view tensor, AdamW (`lr=0.001`,
`weight_decay=0.001`), and 400 full-training-partition updates. There is no
reconstruction head. The eight ordered methods are:

1. `full_gradient_only_mask_selected_control`: gradients use the full view
   only, while the common training-outage checkpoint rule is retained. This is
   a gradient ablation, not a claim of a wholly no-mask pipeline.
2. `masked_erm`: the five-view mask-law mean.
3. `group_mean_pord`: ReLU after each scenario×outage mean difference.
4. `paired_pord`: ReLU on each same-UID missing-minus-stop-gradient-full loss
   before the scenario mean.
5. `paired_wrong_pair`: the same row-level objective after replacing full loss
   with a frozen circular derangement of sorted UIDs within each training
   scenario. This preserves each group’s full-loss marginal while breaking
   true pairing.
6. `groupdro`: scenario×five-view atoms, `eta=0.01`, with mask-law×uniform-group
   prior.
7. `vrex`: mask-law×uniform-group mean plus population weighted atom variance,
   `beta=1.0`.
8. `mofe`: mask-law mean plus `0.1 * mean ReLU(full row loss - missing row
   loss)`.

For the three PORD rows, the positive scenario×outage matrix is scored by its
fixed top quarter (`ceil(atom_count/4)`) with a symmetric cutoff-tie
subgradient plus the mean absolute two-way double-centered interaction. ReLU
and absolute-value zero subgradients are zero. The group-mean row is an
explicit weaker ablation because averaging before ReLU does not retain the
row-level pairing signal.

The low-level fit API cannot receive outer features or labels. This is an
application-level API boundary, not an OS capability-separation claim: the
spawned task process later opens the sealed parquet through distinct loaders.
After every update the fit API scores only training-scenario outage
predictions with

`C = 0.5 * (mean(four outage RMSE3D) + max(four outage RMSE3D))`.

The earliest strict minimum is frozen. Only then does a separate feature-only
loader create all five outer predictions, after which the task evaluator reads
outer truth and computes task metrics exactly once. A completed task
evaluation is recovered from its persisted truth/prediction/metric receipt and
the task evaluator is never called again. Read-only artifact validation does
independently recompute those persisted metrics; this semantic validation is
not a second task-level evaluator call or source-truth read.

## Fold grid and selection

The preparation declares 14 overlapping blocked folds:

- eight `scenario_loso` folds, the only primary selection evidence;
- four `speed_pair_out` folds, report-only stress;
- two `shape_out` folds, report-only stress.

With eight methods and three paired seeds (`1701`, `2603`, `3907`), the grid is
exactly `8 × 14 × 3 = 336` authenticated scientific fits: 192 primary and 144
stress (`192 primary fits` and `144 stress fits`). One fit predicts full plus
all four outages. At fit level this yields
1,344 outage contexts and 336 full contexts. Three raw prediction tables—not
three scalar scores—are averaged for every method×fold ensemble.

For each method, primary selection uses exactly 32 ensemble outage contexts
(eight held-out scenarios×four masks). Let `A` be their mean RMSE3D, `WS` the
worst held-out-scenario mean, and `WM` the worst-mask mean. The frozen score is

`robust_J_m = (A + WS + WM) / 3`.

The lower score wins, with exact equality resolved by the frozen method order.
Full-view performance is diagnostic only. Neither speed/shape stress nor any
alignment sensitivity value can select a method, tune a hyperparameter, or
decide a directional gate. These overlapping folds are development/model-
selection evidence, not independent replications or an unbiased significance
test. The three paired seeds are not independent experiments.

The separately frozen LOSO directional gate supports the PORD direction only
when `paired_pord` has strictly lower primary `robust_J_m` than each of
`masked_erm`, `group_mean_pord`, and `paired_wrong_pair`, and strictly wins the
mean four-outage context RMSE in at least six of eight held-out scenarios
against `masked_erm`. Equality always fails. All methods must carry the same
exact primary context-identity hash. Stress folds and alignment sensitivity
cannot enter this gate.

## Signed alignment sensitivity

The zero-shift parquet does not contain shifted targets. After all task fits
are durably authenticated and the frozen zero-shift selection has been
computed from them, the model calls only the preparation’s public read-only
API. The zero-shift and sensitivity objects are then persisted together in the
development manifest:

```python
reconstruct_openaxes_signed_shift_truth_v1(source_path, prep_config, prep_run)
```

The API re-authenticates the official source, exactly replays base UID/time
identity, and returns target-only long rows for signed shifts
`[-50,-20,-10,+10,+20,+50] ms`. Lookup time is
`base_aligned_time_us + shift_ms*1000`; unsupported edge rows are dropped and
never extrapolated. Shift and zero metrics pair on the same base UIDs.
Although dropping is the general no-extrapolation policy, this fixed official
source has support for every preregistered shift: every one of the exact
8-scenario × 6-shift dropped-edge counts must be the integer zero. Any nonzero,
missing, or extra count fails before sensitivity aggregation.

Only frozen `masked_erm` and `paired_pord` predictions are reported, for every
primary and stress fold and by family. Required values are RMSE3D,
`robust_J_m`, and signed shift-minus-zero deltas. The raw shifted truth is not
persisted or redistributed by the model artifact; its closed receipt and
metrics are stored. Private preparation helpers are forbidden.

Completed validation checks the exact `2 methods × 14 folds × 6 shifts`
method/fold/family/shift order, the complete (not subset) persisted-ensemble
base-UID identity, context count, delta arithmetic, zero-shift metrics against
the persisted authenticated ensembles, and all family summaries. Production COMPLETE validation always replays the
signed source: the public reconstructed truth and authenticated zero-shift
ensembles rebuild the entire canonical sensitivity object, not merely its
reconstruction receipt. A persisted-only cryptographic check is never reported
as signed-source semantic replay.

## GPU process and durable-state contract

Production requires four frozen physical GPU rows, each bound by exact index,
UUID, PCI bus ID, and name. The source-closed config and prepared manifest
freeze `workers_per_gpu=2`; the wider `1..3` range describes scheduler
capability only and cannot select a benchmark run. Before spawning, the
controller writes one immutable execution session that binds that value, all
GPU rows, every slot-to-GPU mapping, every slot FIFO, the task-index hash, and
the controller/source-closure hashes. A
resume must reuse this exact session and cannot remap a task or change worker
count. There is one narrow pre-launch crash recovery: when that authenticated
session exists but launch, ready, roster, attempt, task-phase, result, and
quarantine evidence are all exactly empty, `resume=True` first freezes a
session-bound `runtime/prelaunch_session_recovery.json` receipt and then creates
the deterministic ordinal-zero launch. A crash after that receipt but before
the launch may repeat the same operation. The exact root member set is
`prepared_manifest.json`, `PREPARED.json`, `tasks`, `runtime`, `results`, and
`quarantine`. Before the receipt, runtime contains exactly `ledgers`,
`execution_launches`, `worker_ready`, `attempt_ledger`, `attempt_roster`,
`pre_fit_validation.json`, and `execution_session.json`; afterward the receipt
is the sole additional runtime member. The seven execution/ready/attempt/
phase/result/quarantine directories named in the YAML must all be empty and
symlink-free.

Within the declared crash, corrupt-prefix, quarantine, and ordinary
partial-deletion threat model in which some local attempt authority remains,
the receipt permits only authenticated deterministic ordinal-zero launch
creation. It is not an attempt authority and does not authorize scientific
refitting, deletion-as-freshness, remapping, or a later launch without its
predecessor. Any partial or ambiguous evidence still fails closed. The receipt
records only that no local attempt evidence was observed at exclusive creation;
it does not prove that execution never happened if every local attempt authority
was maliciously erased. No external append-only log or signature authority is
added by this contract, so that total-erasure history is unprovable here. A
normal first launch uses `resume=False`, while
every later launch uses `resume=True`. The pre-fit public-validation receipt is
canonically recomputed from the prepared task grid and synthetic/production
marker. Its immutable
`runtime/pre_fit_validation.json` additionally binds the prepared-manifest
reference, parent public receipt, separate parent/model Git heads and ancestry
closure proof, model source set, task index, exact GPU inventory, and
336/192/144 counts. Editing and resealing the receipt or session cannot
substitute it. The controller uses the spawn context and creates one persistent
process per logical GPU slot. Each slot consumes its fixed FIFO serially, so a
GPU never exceeds the requested worker count and, while the local execution
authorities remain available, the controller cannot schedule a scientific task
twice.

The prepared runtime receipt has a closed schema for the exact Python version,
implementation, executable and cache tag; OS release and machine; NumPy,
pandas, PyArrow, PyTorch, and PyYAML versions; controller PyTorch-import state;
and the four thread-control environment variables. Validation rederives that
receipt from the current process. Python, platform, package, and thread fields
must remain exact throughout the run. The controller PyTorch-import state must
match only at the pre-spawn boundary: production preparation records `false`
and task launch independently rejects a pre-imported controller. Post-spawn and
final validation deliberately ignore only the current import-state bit because
strict checkpoint verification normally imports PyTorch in that same controller
process. The recorded bit and the complete recorded runtime receipt remain
immutable. `PREPARED.json`, the pre-fit receipt, and the execution session each
bind its canonical hash.

The child rejects a pre-imported PyTorch module, sets
`CUDA_VISIBLE_DEVICES=<frozen UUID>`, then imports PyTorch. It must see exactly
one `cuda:0` and revalidate its UUID, PCI mapping, name, and physical index.
The PyTorch property UUID is compared to the frozen `nvidia-smi` UUID only
after the strict representation normalization frozen in the sealed-002
erratum; UUID identity itself is unchanged.
Worker identity comes from the process transport, never a returned task field.
Controller results are reordered by canonical task ordinal, so heterogeneous
durations cannot change scientific ordering.

After the exact GPU check, each child durably writes a launch/session/slot
ready receipt containing physical index, UUID, PCI bus ID, name, assignment
hashes, controller hash, and child process identity. The controller verifies
each ready PID against the exact spawned process and name, requires unique PIDs,
persists one all-slot observation receipt, and only then releases the child
barrier. Every new task attempt references both ready authorities and first
writes a monotonic roster, followed by the ordinary attempt ledger. Every
launch field is checked against the immutable session, the complete canonical
launch hash and controller-observation hash are bound into the task execution
binding. The launch ID is itself the deterministic hash of the immutable
session ID, launch ordinal, exact launch mode, and exact predecessor reference,
so renaming a launch directory cannot be legitimized by resealing descendants.
The sole task
attempt ID is likewise a deterministic hash of the task and its first exact
execution binding. Every phase
stores and revalidates that exact binding. Resume
phases therefore preserve the exact session, launch, slot, task assignment,
controller/source hashes, and child-ready lineage. Phase launch ordinals cannot
move backward. The monotonic roster and attempt ledger are outside movable task
state and survive quarantine; deletion of only the ordinary ledger cannot
restore a fresh-fit state.

Each task has this hash-linked phase prefix:

1. `execution.started`;
2. `training.completed`;
3. `outer_prediction.completed`;
4. `outer_eval.started`;
5. `outer_eval.completed`;
6. `bundle.completed`;
7. `result.completed`.

An `execution.started`-only crash is an ambiguous possible fit and fails closed
rather than refitting. Once `training.completed` exists, prediction recovery
strict-loads the checkpoint and transform without fitting. An
`outer_eval.started`-only crash also fails closed. A durable
`outer_eval.completed` phase is verified and bundled without truth reread or
evaluator recall. Resultless recovery uses authenticated phase payloads.
Corrupt state can be moved—not deleted—through an intent/move/audit quarantine
event. Under the crash, corrupt-prefix, quarantine, and ordinary partial-deletion
threat model, every locally observed attempt state leaves persistent
roster/ledger evidence and quarantine cannot make the task look fresh or
authorize a refit. These authorities are nevertheless stored on a mutable local
filesystem. Simultaneous malicious deletion of the roster, attempt ledger,
phase ledger, result directory, quarantine events, and every other local attempt
authority is indistinguishable from a never-started task without an external
append-only log or signing authority; this contract does not claim resistance to
that total-erasure attack.

## Artifact lifecycle and validation

Preparation writes `prepared_manifest.json`, `PREPARED.json`, the 336-task
index, empty runtime ledger/results directories, source/config/environment/GPU
receipts, and no results. The frozen config and prepared manifest bind the exact
pre-launch recovery path, both allowed runtime member sets, every required-empty
directory, launch mode/ordinal, allowed action, forbidden actions, and the
non-total-erasure limitation. The last development marker is `COMPLETE.json`.
Every checkpoint, transform, history, prediction, persisted outer-evaluation
row, bundle, result, and phase is referenced by exact path, byte count, and
SHA-256. Every internal reference also requires one canonical relative POSIX
spelling: absolute paths, backslashes, `.`/`..`, empty components, and repeated
slashes fail closed. Prepared validation checks every quarantine event as a
closed, symlink-free intent/move/audit lineage; completed validation requires
the quarantine directory to be empty. Completed validation semantically rebuilds three-seed ensembles,
context scores, exact method-context identity, the LOSO directional gate,
paired-control equality, and selection from persisted task evidence. COMPLETE
binds the zero-shift object, directional gate, and alignment-sensitivity
object by separate hashes. Its mandatory signed-source validation replays the
public reconstruction API and canonically recomputes the full sensitivity
report before COMPLETE can be accepted. The immutable
`runtime/full_signed_source_replay.json` binds official source/preparation
references, the reconstruction-receipt hash, exact scenario/shift support
identity hashes, the signed-report hash, and validator/source-set hashes;
COMPLETE references that receipt by path, bytes, and SHA-256. Acceptance also
requires exactly 336 attempt rosters, attempt ledgers, task ledgers, results,
and authenticated task bindings. The highest launch ordinal must itself have an
authenticated terminal completion; an older completed launch cannot mask a
newer started-only launch. Missing or extra task evidence, noncanonical artifact
references, extra/symlink task-index or completed-result members, and malformed
launch/ready member sets fail closed.

There is intentionally no challenge/final-test route in this contract.

The focused tests do not read production data or use CUDA. They do include one
explicit two-update **synthetic CPU fit** to exercise the fit/predict/evaluate
API boundary; it is never accepted as benchmark evidence:

```bash
env -u PYTHONPATH .venv312/bin/python -m pytest \
  tests/test_openaxes_external_model_v1_config.py \
  tests/test_openaxes_external_model_v1_training.py \
  tests/test_openaxes_external_model_v1_experiment.py -q

env -u PYTHONPATH .venv312/bin/python -m py_compile \
  src/imu_vive_learning/openaxes_external_model_v1_config.py \
  src/imu_vive_learning/openaxes_external_model_v1_training.py \
  src/imu_vive_learning/openaxes_external_model_v1_experiment.py
```

The read-only doctor can also be invoked directly after loading the canonical
config; it hashes the nine-file model source closure, including the direct
runtime dependency `src/imu_vive_learning/io_utils.py`, and checks that the
public signed-shift API exists without reading data, importing CUDA, or
computing a metric:

```bash
env -u PYTHONPATH .venv312/bin/python -c \
  'from imu_vive_learning.openaxes_external_model_v1_config import load_openaxes_external_model_config as load; from imu_vive_learning.openaxes_external_model_v1_experiment import doctor_openaxes_external_model_v1 as doctor; print(doctor(load("configs/openaxes_external_model_v1.yaml")))'
```

## Preparation limitations inherited unchanged

Publisher clock alignment was derived with robot-side information and is
label-assisted acquisition metadata. Robust per-recording IMU clocks inspect
each complete held-out recording. The preparation is therefore offline,
transductive, and acausal. It cannot justify causal, streaming, online-control,
or deployment claims. Shape/speed traces came from one installation and one
short session. The portal/member license discrepancy must be resolved before
redistributing derived public-data artifacts.

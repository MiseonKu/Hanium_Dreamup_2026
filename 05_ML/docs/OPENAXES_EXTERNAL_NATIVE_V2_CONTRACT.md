# OpenAXES native sensor-expert v2 preregistration

This document freezes the native v2 experiment before any OpenAXES model-v1
result or metric is opened. The normative machine-readable contract is
`configs/openaxes_external_native_v2.yaml`. No v1 result value is reproduced
here. The only frozen predecessor facts are its contract identity, canonical
configuration and source-set digests, terminal-stage name, and comparator method ID.
The hardening-complete predecessor config digest is
`ce6a828e6d92118e29b7c67ec166cc3a555709e78bb41ef0838a79e982e80a3e`;
its exact nine-file source-set digest is
`dd6418badc644f7d7576c836d17074ee2914b4ae0cb555b26315eae8669b8488`.

Native v2 is preregistered but not implemented. Neither native model/training
implementation nor a fit is authorized until model-v1 has reached its declared
terminal stage, a
separate read-only auditor has issued the terminal receipt described below,
and a second audit has verified a literal implementation and its exact source
closure. The sole pre-outcome exception is the separately closed parent-receipt
auditor source and its authority erratum described below. Once v1 outcomes become visible, this
preregistration is immutable.

## Authority erratum 1

The original eight-file audit remains immutable history: audited source commit
`49d9cb64406d6f5c9457e210623c45324a0880ac`, evidence commit
`ce09392a287f00447c5451471eae854d6aa726e6`, and receipt
`receipts/openaxes_external_native_v2_parent_receipt_source_audit.json` are
preserved, not rewritten or deleted. They are retired and cannot authorize a
successful terminal lineage. The only active source-audit path is
`receipts/openaxes_external_native_v2_parent_receipt_source_audit_erratum1.json`.
Both issuance and public validation resolve that exact new Git path and reject
the old path, old schema, old audited commit, and old evidence commit.

This amendment follows the failed supervisor run `openaxes-v1-sealed-001`.
Its preserved `openaxes-v1-runs/openaxes-v1-sealed-001/LATEST_TERMINAL.json`
has raw SHA-256
`e032c62b899efd7e90647748f6c087b05d271ae1e312fc8ec64f390cc1bc20e3`.
Public preparation and validation had completed, but the successful model
lineage had zero model `PREPARED` markers, execution sessions, scientific fits,
predictions, evaluations, and results. No v1 result or metric existed or was
used. The public preparation manifest, Parquet, `COMPLETE.json`, and validator
stdout hashes are retained in the YAML deviation record.

The failure was an outcome-free JSON key-order consumer bug. The producer's
semantic sensor order was `humerus,radius,carpus,digitus`; canonical JSON with
sorted keys persisted the object in `carpus,digitus,humerus,radius` key order.
The mapping content and feature lists were correct, with canonical mapping
SHA-256 `30deb94cc249ec6c2d17ac141b5450a30e4c526c4f96378a8a57b0c94443762a`.
The failed consumer required mapping iteration order and flattened
`values()` in that order. The corrected consumer requires the exact key set
and indexes each group in the frozen semantic order.

The correction does not change model-v1 or native-v2 science. The exact v1
configuration remains 12,099 bytes, raw SHA-256
`848a19cc270401f6c37d391b399832996324792382a4a926f93b1584d7186c51`,
and canonical semantic SHA-256
`ce6a828e6d92118e29b7c67ec166cc3a555709e78bb41ef0838a79e982e80a3e`.
Tests compare that old Git blob with the new values and require unchanged
methods, hyperparameters, fold families/counts, seeds, and directional gates.
Only the consumer/doc/test byte identities change, producing the new nine-file
source-set digest above.

The terminal fields `committed_before_v1_prepare` and
`source_audited_before_v1_prepare` now mean before the preparation belonging to
the terminal-bound successful `openaxes-v1-sealed-002` lineage. The earlier
sealed-001 public preparation does not satisfy or falsify those assertions;
it belongs only to the preserved failed deviation. The terminal envelope binds
the erratum ID, active receipt path, this timing scope, and successful run ID.
That ID is also enforced structurally: model-v1 must resolve exactly below
`<project>/artifacts/openaxes_external_model_v1/run_openaxes-v1-sealed-002`
and preparation below
`<project>/artifacts/openaxes_external_v1/prepared_openaxes-v1-sealed-002`.
Both issuance and public validation reject a different basename, suffix, or
project-relative ancestry. The separately declared supervisor directory is
provenance only and is never consumed as scientific terminal authority.

That sealed-002 successful-lineage designation was the frozen intent of
erratum 1. It is now historical and cannot authorize issuance after the
outcome-free failure below.

## Authority erratum 2

The original and erratum-1 source audits remain immutable Git history. The
erratum-1 source/evidence commits are
`6f5b1226cff400b708f0c9d167973ce505c48938` and
`26795e6c34a7ac11995875134d8ff81758758622`; its receipt is
`receipts/openaxes_external_native_v2_parent_receipt_source_audit_erratum1.json`.
Neither that authority nor the original authority may authorize the successful
lineage. The sole active path is now
`receipts/openaxes_external_native_v2_parent_receipt_source_audit_erratum2.json`,
under contract
`openaxes.external.native_sensor_experts.v2.parent_receipt_source_audit.erratum2.v1`.

Run `openaxes-v1-sealed-002` completed a valid 22,482-row, 36-feature,
14-fold public preparation and created the model `PREPARED` marker, immutable
336-task index, pre-fit validation, execution session, and ordinal-zero launch.
It then failed during worker identity bootstrap, before any worker-ready
receipt or task attempt. Consequently there are exactly zero task ledgers,
attempt ledgers/rosters, result directories, quarantined members, scientific
fits, predictions, evaluations, and results. No model result or metric artifact
existed or was inspected. The YAML binds the raw hashes of all preparation,
model-authority, supervisor, and launch evidence.

The error was representational, not hardware drift. With sealed PyTorch
`2.12.1+cu132`, `torch.cuda.get_device_properties(0).uuid` has type `_CUuuid`
and string value `5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb`; the authenticated
NVIDIA UUID is the same canonical hex value prefixed by `GPU-`. Availability,
device count, name, PCI bus ID, and the `nvidia-smi` mapping were unchanged.
The corrected consumer parses only the canonical UUID forms and reconstructs
the literal `GPU-` prefix before comparing to the frozen identity.

The supervisor wrote a preserved `cleanup_blocked_nonterminal` record with raw
SHA-256
`063644f72ad6d83eb3890d5ea0693ef3d42b0cb8f25831b2743fde87938b25e0`.
Its refused PID was only Python's `multiprocessing.resource_tracker`; it
naturally exited without a manual signal, after which telemetry and `/proc`
showed zero run markers. The supervisor's `preparation_present=false` progress
field was also a report-only wrong-marker probe: the valid public
`COMPLETE.json` and its validation evidence remain authoritative. Because the
source/controller identity changes, sealed-002 cannot be resumed. The next
lineage must freshly prepare and run as `openaxes-v1-sealed-003`.

Erratum 2 changes no scientific choice. The model-v1 YAML remains exactly
12,099 bytes with raw SHA-256
`848a19cc270401f6c37d391b399832996324792382a4a926f93b1584d7186c51`
and semantic SHA-256
`ce6a828e6d92118e29b7c67ec166cc3a555709e78bb41ef0838a79e982e80a3e`.
Methods, hyperparameters, 14 folds, three seeds, the 336-fit grid, selection,
and directional gates are byte/semantic-equal to the erratum-1 authority.

## Git and parent-receipt boundary

These amended YAML and Markdown blobs must be committed before the
terminal-bound successful v1
`execution.started`. Their commit ID is intentionally absent here because
embedding it would be circular. The terminal auditor instead freezes the
commit, repository object format, both blob OIDs, and canonical
`{path,bytes,sha256}` references. For each path it obtains the bytes with
`git cat-file blob <v2-commit>:<path>` without text decoding or newline
conversion, verifies the object type and OID, and hashes those exact bytes.
It then requires
`git merge-base --is-ancestor <v2-commit> <v1-model-contract-head>` to exit
zero, where the descendant is v1's immutable `model_contract_git_head`.
Missing objects, shallow ancestry, byte drift, or a nonzero exit fails closed.

A separate parent-receipt auditor is implemented and preregistered without seeing
v1 outcomes. Its exact eight-file closure is the two files here, a dedicated
parent-receipt contract document, `requirements-eval.lock`, the package
`src/imu_vive_learning/__init__.py`, `io_utils.py`, one auditor module, and one
auditor test. Including `__init__.py` is mandatory because a normal
`imu_vive_learning.<module>` import executes it. That source must be implemented,
committed, and source-audited before the terminal-bound successful sealed-003
preparation. The active erratum source-audit receipt
is then committed as a separate Git blob; both its evidence commit and the
audited source commit must be ancestors of v1's prepared
`model_contract_git_head`. This proves the ordering without changing the v1
manifest schema or circularly embedding a commit. The auditor itself runs only
after validated v1 completion and is read-only: it cannot train, predict,
reconstruct source truth, invoke the evaluator, use CUDA, access a challenge,
or modify v1.
Before successful sealed-003 preparation, its synthetic tests must close the eight-file digest,
Git blob/ancestry checks, strict JSON/reference/payload schemas, masked-only
filter, and forbidden import/call surface; skipped or xfailed tests do not pass.
The auditor does not accept an external test receipt. After Git/worktree
closure verification it directly launches `<project>/.venv312/bin/python` with
`-P -S -m pytest -c /dev/null --noconftest -p no:cacheprovider -q` followed by
the single exact parent-auditor test path, from the canonical project root,
with `PYTHONPATH=src:<exact-.venv312-site-packages>`, `PYTHONHASHSEED=0`, plugin
autoload disabled, and
`PYTEST_ADDOPTS` absent. It is launched with the exact sanitized environment
frozen in YAML, including no user site or bytecode writes, no visible CUDA,
UTC/C.UTF-8, system/global Git config disabled, and an exact minimal `PATH`.
Therefore neither `pyproject.toml`, an ancestor
`conftest.py`, a shell alias, `.pth`/`sitecustomize`, nor ambient pytest plugins
can silently change the audited test semantics. The source-audit CLI process
itself must also be launched via `env -i`, the project venv, and `-P -S`; it
fails closed on bootstrap flag, environment, executable, module-path, or
`sys.path` drift. The receipt binds launch and real executable identities, raw
`pyvenv.cfg` and its exact five fields, the base `sys.prefix` observed under
`-S`, exact launch-venv site-packages, every version pin, each pin's dist-info
`METADATA`/`RECORD` and RECORD-owned installed-file closure before and after
tests. Each exact three-field RECORD row is checked against its installed raw
file digest and size before/after tests; raw paths, the narrow RECORD/generated
CPython-pyc empty-integrity exception, `Distribution.files`, and pinned
METADATA Name/Version are fail-closed. The receipt also binds the full visible distribution inventory, argv, working directory,
environment, exactly 12 passed with no other outcomes, and raw
stdout/stderr/report hashes. Its one exact pytest summary line permits either
`12 passed in <decimal>s` or the same line with pytest's optional
` (H:MM:SS)` wall-clock suffix; hours are canonical nonnegative decimal and
minutes/seconds are two digits in 00--59, while any other suffix, duplicate
summary, warning, or nonzero failed/skipped/xfailed outcome is rejected. The
producer and persisted validator share that parser and require raw-derived
counts to equal the stored outcome fields. The lock has no wheel/sdist artifact hashes, and
extra visible distributions are inventoried rather than rejected, so this is
not a closed-world venv claim.

The auditor first copies and verifies the active erratum2 source-audit receipt, then
creates three O_EXCL files in a fresh `parents` directory: that exact copy, a
terminal envelope, and a comparator bundle. All three parent files use UTF-8 canonical JSON with
sorted keys, comma/colon separators, finite values, and one final LF. Every file
reference has exactly `path`, `bytes`, and `sha256`; paths are canonical
relative POSIX paths. The terminal envelope persists canonical absolute
`workspace_root`, `project_root`, `v1_run_root`, `preparation_run_root`, and
`v2_run_root` values derived from mandatory CLI arguments. V1 references resolve
only below the v1 run root, preparation references only below the preparation
run root, and parent-envelope/bundle references only below the v2 run root;
Git-blob references use `git cat-file` at the bound commit instead of a
filesystem base. Issuance holds every root as a component-wise directory FD
opened from `/` with required `O_DIRECTORY|O_NOFOLLOW` until final parent-entry
revalidation. Directory identity is device/inode/mode only, so unrelated
sibling churn is permitted; each regular-file read separately retains the full
device/inode/mode/size/mtime/ctime stability check. Resolution uses component-wise no-follow opens and rejects an
absolute path, escape, alternate base, dot component, repeated separator,
backslash, symlink component, non-regular final object, or pre/post-read inode
or ctime drift. This is not a sandbox against a privileged OS or mount-namespace
attacker. Each payload hash covers every exact top-level key except the payload-hash
key itself.

The terminal envelope copies the source-audit receipt and enforces exact
identity, not merely a true-looking assertion. Specifically,
`auditor_source_audit.git_commit` equals the copied receipt's
`audited_git_commit`, `auditor_source_audit.source_set_sha256` equals its
`source_set_sha256`, and `git_lineage.v2_preregistration_git_commit` equals that
same audited source commit. Its exact eight-file map must equal the Git blobs at
that commit. Git runs with `GIT_NO_REPLACE_OBJECTS=1`, rejects replace refs and
legacy grafts, and accepts only regular `100644`/`100755` `ls-tree` entries.
The copied raw bytes must equal the blob at
`auditor_source_audit.source_audit_git_commit`, with the recorded blob OID.
Separate recorded `git merge-base --is-ancestor` commands for the audited-source
commit and the evidence commit must both exit zero against
`git_lineage.v1_model_contract_git_head`; absent or shallow objects fail closed.

The auditor validates all 336 task result-to-bundle chains, attempt receipts,
and seven-phase raw-reference lineages, and it deeply links training and metric
receipts for exactly the 24 primary `masked_erm` tasks. The bundle can contain
only v1 `masked_erm`: three seed initial-state hashes,
32 primary seed-ensemble outage contexts, eight primary full contexts, 96
per-seed outage contexts, their exact identities, and the fixed gate inputs
recomputed from those closed rows. It contains exactly 24 primary training
receipt references, 24 primary outer-metric receipt references, and the
development-manifest reference, but no
prediction table, evaluation-row table, source-truth reference, other method,
ranking, selection, stress outcome, or signed
outcome. The seed-ensemble contexts are sealed development evidence and are not
claimed to be reproducible by averaging individual metric JSON because v1
ensembled predictions before RMSE. The public validator reopens persisted
terminal/audit/bundle files and recomputes Git, v1, preparation, phase, and
reference authority; caller-supplied dictionaries are not authority. A separate preexecution lineage validator checks the envelope and
bundle; v2 trainers read no parent file, and v2 aggregation/gate code receives
only that verified comparator bundle. Direct v1 run-root access is prohibited.

## Scope and scientific question

The experiment asks three narrow, ordered, within-protocol questions on the
same prepared OpenAXES folds. Candidate A tests a scenario-balanced dense ERM
against the still-unseen, receipt-bound v1 `masked_erm`. Candidate B contrasts
a parameter-matched physical-sensor expert layout with A under the same direct
ERM. Candidate C contrasts CAGrad with B while holding the expert architecture
fixed. These are predictive contrasts, not causal attribution.

All ingredients are known components. The preregistration makes no novelty,
first, causal, proof, theorem, or general-superiority claim. The source is one
public recording session partitioned into eight scenario clusters. Its IMU
streams are measured complete streams and the outages are synthetic masks.
There is no observed physical outage, independent-session validation, SO101
confirmation, cross-installation validation, deployment validation, or
closed-loop control evidence. No new or additional collection is authorized;
v2 can only reuse the receipt-bound prepared table. The three seeds are paired
optimization repeats, not independent experiments; overlapping outer folds
are not independent replications.

## Input, transform, and views

The input is exactly 40-dimensional. The first 36 entries are four physical
groups in order `humerus`, `radius`, `carpus`, `digitus`, each containing these
nine entries in order: ADXL355 acceleration XYZ, BMI160 acceleration XYZ, and
gyroscope XYZ. The final four entries are availability bits in the same group
order. Exact prepared names are formed by joining the group, an underscore,
and the nine channel names frozen in the YAML. Metadata and targets never enter
the network.

For every outer fold, each continuous column is standardized with a population
mean and population standard deviation fitted on all and only the declared
training rows. Availability bits are not standardized. Targets use the
training-only XYZ mean and one isotropic population scale. Non-finite or zero
feature scales and a non-finite or non-positive target scale fail closed.

Every UID produces the ordered views `full`, `outage_humerus`,
`outage_radius`, `outage_carpus`, and `outage_digitus`. Masking happens after
standardization: the missing group's nine entries become zero and its
availability becomes zero. The UID and target remain unchanged. The mask law
is 0.25 for full and 0.1875 for each of the four outages.

## Exactly parameter-matched models

The dense network used by A is `40 -> 64 -> 64 -> 3`, with ReLU after each
hidden layer and a linear output. It has

`40*64 + 64 + 64*64 + 64 + 64*3 + 3 = 6979`

trainable parameters.

B and C use four experts. Each receives only its physical group's nine
continuous values and applies `9 -> 20 -> 24` with ReLU after both layers. One
expert has

`9*20 + 20 + 20*24 + 24 = 704`

parameters, so the four experts have 2,816. Each 24-vector is then multiplied
by its availability scalar. The four gated vectors are concatenated in sensor
order, followed by the four raw availability bits, producing 100 entries. A
`100 -> 40 -> 3` ReLU/linear fusion has

`100*40 + 40 + 40*3 + 3 = 4163`

parameters. Thus the expert model also has `2816 + 4163 = 6979`. There is no
normalization layer, dropout, router, learned imputation, reconstruction head,
or auxiliary head. Parameter equality, rather than equal matrix-operation
count, is the matching criterion; the dense and expert matrix MAC counts are
6,848 and 6,760 per example respectively.

The three fixed candidates are:

1. A: `dense_6979__scenario_balanced_masked_erm`;
2. B: `sensor_experts_6979__scenario_balanced_masked_erm`;
3. C: `sensor_experts_6979__scenario_balanced_cagrad`.

There is no best-of selection, cross-candidate blending, or post-outcome
replacement. C is the fixed terminal v2 candidate whether the gates pass or
fail.

## Reproducibility controls

Every candidate/fold/seed fit runs in a fresh process. Its exact seed is set
for Python, NumPy, PyTorch CPU, and CUDA. Linear layers are built on CPU in the
declared module order inside `torch.random.fork_rng(devices=[])`, after setting
the fit seed, with PyTorch `Linear.reset_parameters` (Kaiming uniform
with `a=sqrt(5)` and the corresponding fan-in uniform bias), then transferred
to CUDA. A must use the same dense factory and seed rule as the v1 comparator,
and its initial hash must equal the receipt-bound v1 `masked_erm` initial hash
for each seed.
B and C must have byte-identical initial state-tensor hashes and equal empty
optimizer states for every fold/seed pair.

State hashing is byte-for-byte compatible with v1. Iterate every `state_dict`
entry in Python-sorted name order, convert its tensor by
`detach().cpu().contiguous().numpy()`, and stream into SHA-256: UTF-8 name,
ASCII `str(array.dtype)`, ASCII compact JSON shape, then C-order payload bytes.
There are no delimiters or length prefixes between any fields or tensors.
Names, dtypes, and shapes are therefore bound independently as schema evidence
as well as through the compatibility hash.

Deterministic algorithms, deterministic cuDNN, disabled cuDNN benchmarking,
disabled TF32, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before importing PyTorch
are mandatory. Training uses float32 with no mixed precision; checkpoint,
metric, Gram-matrix, and gate accumulation uses float64. The environment,
packages, GPU identity, row order, transforms, initial states, and source set
are receipt-bound. There is no shuffle or batch RNG.

The authoritative UID order is the receipt-bound prepared Parquet's physical
row-group and row order. A fold is derived only by an order-preserving scenario
membership mask; UID sorting, index sorting, reset-driven reordering, or
post-mask sorting is forbidden. Every training and outer partition records
separate ordered-UID, ordered-scenario, and ordered UID/scenario-pair hashes.
Views are view-major, then use that partition order.

All float64 reductions use one fixed tree: add adjacent ordered pairs, carry an
odd tail unchanged, and repeat until one scalar remains. Dot products first
multiply in fixed order; means divide the fixed sum once; norms and RMSE use
the frozen square/root order. Scenario order comes from the fold manifest,
rows retain Parquet order, coordinates are XYZ, seeds are 1701/2603/3907,
folds retain manifest order, and outages retain sensor order. BLAS-dependent,
parallel, or implicit unordered float64 reductions are forbidden.

The environment is closed by `requirements-gpu.lock` SHA-256
`fc602cce3bbc854cd4b538f43bae55dac2cdddced831b83c1ed613c116a37ab6`
and `requirements-eval.lock` SHA-256
`b72b9455129011ac88829c87cb2b0012215214ead5622838b9117a19267232aa`.
Required versions are Python 3.12.3, torch 2.12.1+cu132 with CUDA runtime 13.2
and cuDNN 92000, NumPy 2.3.5, SciPy 1.17.1, PyArrow 21.0.0, pandas 2.3.3,
and PyYAML 6.0.3. The current driver reports 610.43.03 and CUDA UMD 13.3;
that does not replace the required torch CUDA runtime 13.2.

The four frozen physical devices are:

- index 0: `GPU-462585f2-b4cc-3864-e27c-cbaae4ab6c9a`,
  `00000000:01:00.0`, NVIDIA CMP 170HX;
- index 1: `GPU-517dfd2b-2ade-0b3f-70d7-0ea2a78f868b`,
  `00000000:2D:00.0`, NVIDIA CMP 170HX;
- index 2: `GPU-b3db5781-61ae-33bb-63a0-64b14dd9ae6b`,
  `00000000:41:00.0`, NVIDIA CMP 170HX;
- index 3: `GPU-5ebdc8be-b2f3-b6e2-4da1-af79ef91f7cb`,
  `00000000:61:00.0`, NVIDIA CMP 170HX.

Three persistent torch-free slot controllers bind to each UUID, giving 12
logical slots. Each controller serially spawns one fresh fit child, waits for
it to exit, and only then advances; each fit child sees only `cuda:0`. A
context is one fold/seed and its ordered A/B/C triplet. The 42 contexts use
`context_ordinal = fold_ordinal*3 + seed_ordinal` and
`slot = context_ordinal mod 12`. Slots 0--5 receive four triplets each and
slots 6--11 receive three. Slot-to-device mapping is interleaved rather than
contiguous: `physical_index = slot mod 4` and
`local_slot = floor(slot/4)`. Thus devices 0--3 receive respectively
11/11/10/10 contexts and 33/33/30/30 fits, while every context triplet remains
FIFO on one immutable slot. Remapping, runtime rebalancing, or substitution is
forbidden. Timing, utilization, memory, and temperature are report-only and
cannot change any scientific action.

## Full-fold risk and 400-update policy

For scenario `s`, view `v`, and row `i`, the row loss is the mean squared error
over the three scaled target coordinates. First average row losses within each
training scenario, then average those scenario means equally. This defines
`R_v`; different scenario row counts do not change their weights. The direct
ERM used by A and B is

`0.25 R_full + 0.1875 (R_humerus + R_radius + R_carpus + R_digitus)`.

Every update sees every training UID in all five views. The tensor is formed in
fixed view-major and authenticated UID order, a single vectorized forward is
performed, and exactly one AdamW optimizer step follows. AdamW is frozen at
`lr=0.001`, `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0.001`, without
AMSGrad, fused/foreach operation, clipping, or mixed precision. Every fit has
exactly 400 updates. Early stopping, pruning, minibatching, view cycling, and
outer-driven restart are prohibited. This is the same 400-full-training-fold
update budget as v1.

For C, let the four ordered outage-risk gradients be `g_1,...,g_4`,
`u=(1/4,...,1/4)`, `g_mean=sum(g_m)/4`, and `G_ij=g_i dot g_j`. Gradients are
float32 and the symmetrized Gram matrix and solver arithmetic are float64.
With `c=0.4`, one SLSQP solve per update starts at `u` and minimizes

`w^T G u + c sqrt(u^T G u) sqrt(w^T G w)`

subject to `0 <= w_m <= 1` and `sum(w)=1`. The YAML freezes the analytic
Jacobian, zero-norm subgradient, tolerances, iteration limit, and acceptance
residuals. There is one start and no clipping, renormalization, second solve,
or fallback. Failure stops the fit. With `g_w=sum(w_m g_m)` and
`epsilon=1e-12`, the applied gradient is

`0.25 g_full + 0.75 [g_mean + c ||g_mean|| g_w/(||g_w||+epsilon)]`.

The five captured task gradients are float32. They are cast once to float64
for Gram, solver, and direction arithmetic; the final direction is cast once
back to the corresponding float32 parameter gradients before AdamW.

A and B use one gradient traversal per update. C uses five, in fixed order:
full followed by the four outages. Assigning the assembled C gradient does not
perform an additional backward traversal.

## Training-only checkpoint and outer boundary

After every optimizer step 1 through 400, a separate no-gradient forward runs
all five training views. Its full-view prediction is a recorded diagnostic and
only the four outage views enter checkpoint selection. For each scenario/view pair, RMSE3D is
the square root of the mean rowwise sum of raw-metre XYZ squared errors. Each
outage component is the equal mean of its training-scenario RMSE3Ds. The
checkpoint score is

`0.5 * [mean(four outage components) + max(four outage components)]`.

The earliest strict finite minimum is selected; update zero is ineligible and
equality never replaces an earlier checkpoint. The score cannot see any outer
feature or label.

After the checkpoint is irreversibly locked, a feature-only loader makes one
vectorized forward for all five outer views and persists all predictions. Only
then may the task evaluator read the outer labels once and evaluate all five
views in one call. Outer information cannot cause a refit, recheckpoint,
candidate change, or rerun. Recovery of a completed task uses its persisted
receipts and cannot invoke the task evaluator again.

The task universe is serialized before any attempt. For
`context_ordinal = fold_ordinal*3 + seed_ordinal` and candidate ordinal 0--2,
`task_ordinal = context_ordinal*3 + candidate_ordinal`; the only legal path ID
is `task_000` through `task_125`. Every TaskSpec has the exact closed fields for
that ordinal, fold, family, seed, and candidate and a semantic hash over every
field except the hash itself. `task_index.json` contains exactly the 126 ordered
TaskSpec raw references and assignment rows. `runtime/slot_assignments.json`
contains exactly 12 ordered slot rows. Their independent assignment hashes are
bound by all later session, launch, attempt, and phase receipts.

Each task then has exactly this O_EXCL, hash-linked phase chain:

1. `execution.started`;
2. `training.completed`;
3. `outer_prediction.completed`;
4. `outer_eval.started`;
5. `outer_eval.completed`;
6. `bundle.completed`;
7. `result.completed`.

Every phase binds one task spec, one attempt ID, an immutable
`attempt_origin_binding`, the actual per-phase `executor_binding`, and the
immediately previous phase's exact `{path,bytes,sha256}` reference. The origin
freezes the first launch, slot/GPU, session, source, environment, parent,
pre-fit, and assignment identities across all seven phases. The executor
instead records the launch, controller-ready receipt, child-ready receipt,
child UUID/PID, and action class that actually wrote that phase; a recovery
launch cannot impersonate the origin launch. All invariant fields cross-match
the origin and session, while the executor launch ordinal may only increase.

The YAML gives each of the seven phase payloads an exact key set, exact artifact
paths, and exact action counts. For example, `training.completed` must record
400 updates, 400 training forwards, 400 checkpoint forwards, and the frozen
candidate-specific traversal/solver counts; `outer_prediction.completed`
records one loader, one forward, five view-equivalents, and zero label/evaluator
calls; the evaluation phases record the one permitted read/call; bundle/result
record zero replay. Each receipt contains a hash of its closed payload and a
second semantic hash of every top-level field except that second hash. The next
phase additionally binds the entire prior canonical file's raw byte reference.
Present phases must be a contiguous prefix. All phase files are exclusive,
fsynced regular files and cannot be overwritten or resealed. One monotonic
roster and one persistent attempt ledger are written before the controller
spawns the fresh fit child; their existence permits no second attempt.

The attempt roster/ledger is the earliest no-refit authority; it is durable
before the fit child is spawned, and `execution.started` confirms that the
authenticated ready child crossed the fit barrier. Preflight creates one
immutable prediction permit for each of the 126 tasks. Immediately before the sole outer
forward, the worker promotes that permit to a persistent claim by creating an
absent hard-link destination, fsyncing it, consuming the permit, and fsyncing
the permit directory. Only that uninterrupted process receives a nonserializable
single-use forward capability after claim-present and permit-absent are durable.
On recovery, claim-only without `outer_prediction.completed`,
neither file, or both files all fail rather than forwarding again. Thus deleting
a consumed claim cannot restore an unused permit. Labels use the same scheme:
preflight creates 126 label permits, and the uninterrupted evaluator promotes
one to a persistent claim before writing and fsyncing `outer_eval.started` with
O_EXCL. Only then is a single-use task/attempt/prediction/claim-bound label
capability minted. A started-only evaluation fails rather than rereading labels or
recalling the evaluator. Durable training may be loaded only for the one
unclaimed prediction; durable predictions may be evaluated once; durable
evaluation may be bundled without source-truth access; and durable bundles may
produce only their result metadata. No recovery path refits, reforwards, or
rereads a task label.

`execution_session.json` has one closed schema that binds the canonical roots,
configuration, 15-file future source closure, v1 terminal receipt, comparator
bundle, preparation manifest/table, environment, four-GPU inventory, task
index, slot assignment, and controller source. After both sets of 126 permits
exist, `pre_fit_validation.json` recomputes those identities and binds the two
complete permit-set hashes. A launch has an O_EXCL `launch.started.json`, exactly
12 torch-free `worker_ready/<launch_id>/slot_XX.json` controller receipts, and
an O_EXCL `launch.ready.json` binding their ordered raw references. No child may
start before that ready set validates. Launch zero has no recovery snapshot;
every later contiguous launch has a closed 126-row state snapshot obtained by a
no-follow scan of every attempt, child, phase, result, claim, permit, and
quarantine authority. It may perform only the snapshot's fixed next action. A
task with any attempt authority can never refit; a genuinely untouched task may
make its sole fresh fit on a later launch.

Every child writes a closed ready receipt with its deterministic UUID, live PID
and process-start identity, parent-controller identity, task/attempt, exact
launch/slot, and all invariant hashes. There must be exactly 126 receipts with
role `fresh_fit_full_task`, one per completed task; recovery children are
allowed only when referenced by the exact state-machine action. Before a fresh
fit or recovered prediction, the child first inherits
`CUDA_VISIBLE_DEVICES=<assigned UUID>`, then imports PyTorch and proves
`torch.cuda.is_available()`, device count one, and current device zero. It
queries current visible ordinal zero through `cudaDeviceGetUuid`,
`cudaDeviceGetPCIBusId`, and `torch.cuda.get_device_name`; UUID, PCI bus ID, and
name must equal the authenticated slot GPU. Environment echo or copied inventory
is not evidence. Every phase must reference its actual child receipt, and every
launch/controller/child directory is a closed, no-symlink join: an orphan,
empty, malformed, extra, cross-task, or resealed receipt fails closed.

The source tests must prove that trainers/checkpoint selectors cannot accept
outer inputs, the feature loader has no target field, prediction opens no
label, and the label loader rejects missing, wrong, or spent capabilities.
Opener spies must observe zero label access before the durable evaluation
barrier and exactly one label read/evaluator call afterward. Crash tests cover
started training, claimed prediction, started evaluation, completed
evaluation, completed bundle, and completed result. AST closure rejects direct
truth-loader imports in training or prediction. All tests use synthetic data;
missing, skipped, xfailed, or flaky capability tests block execution.
This is an application-level API/capability boundary, not an OS-level claim
that the task process cannot physically open the prepared Parquet.

`COMPLETE.json` is legal only for the closed 126-task world: exactly 126 task
specs, rosters, attempt ledgers, fresh-fit child receipts, prediction claims,
label claims, seven-phase ledgers, result directories, and semantically
validated `result.completed` chains; every additional recovery child must be
closed-world referenced. The development manifest binds the ordered raw
references of all 126 final phases. Completion also requires exactly the
declared parent/runtime/report members, zero unused prediction or label permits,
and an empty quarantine. Unknown,
extra, missing, duplicate, symlink, special, incomplete, ambiguous, or
quarantined state blocks completion. Quarantine is non-deleting
intent/move/audit evidence and cannot restore freshness. As with v1, this
local-filesystem guarantee does not claim to prove prior execution after
simultaneous total erasure of every attempt authority.

## Fixed grid and exact accounting

The preparation manifest supplies 14 outer folds: eight `scenario_loso`
primary folds, four `speed_pair_out` stress folds, and two `shape_out` stress
folds. The paired seeds are 1701, 2603, and 3907. Therefore the scientific grid
is `3 candidates * 14 folds * 3 seeds = 126 fits`: 72 primary and 54 stress.
Each candidate has 42 fits and 16,800 updates; the grid has 50,400 updates.

Training uses 50,400 vectorized model forwards (252,000 view-equivalents).
Post-update checkpointing uses another 50,400 (252,000 view-equivalents).
Outer prediction uses 126 calls (630 view-equivalents). The exact total is
100,926 model forwards and 504,630 view-equivalents. Gradient traversals are
16,800 for A, 16,800 for B, and 84,000 for C, totaling 117,600. C performs
16,800 SLSQP solves. There are exactly 126 zero-shift outer feature loads, 126
zero-shift outer-label reads, and 126 task-evaluator calls.

Configuration parsing, hash checks, parameter arithmetic, and zero-update
source preflight are not scientific fits. Preflight may not take an optimizer
step, forward prepared data, traverse a prepared-data gradient, or read any
outer feature or label. Consuming the v1 terminal receipt adds zero v1
fits, updates, forwards, label reads, or evaluator calls. Signed-shift analysis
also adds no fit or model forward. It makes one canonical public signed-truth
reconstruction and no task-evaluator call.

## Aggregation and all-must-pass primary gates

For each candidate/fold/view, average the three raw-metre prediction vectors
before computing a metric. On the eight LOSO folds, let the 32 outage-context
RMSE3Ds define:

- `Amean`: their arithmetic mean;
- `WS`: the maximum of the eight per-scenario means over four outages;
- `WM`: the maximum of the four per-mask means over eight scenarios;
- `J = (Amean + WS + WM)/3`.

Let `F` be half the mean plus half the maximum of the eight full-view RMSE3Ds.
For a seed-level comparison, compute the same `J` from that seed alone. A fold
score is its mean RMSE3D over the four outages. All arithmetic is float64 and
lower is better.

The ordered contrasts are A against the receipt-bound frozen v1 `masked_erm`,
B against A, and C against B. Each contrast must satisfy all five conditions:

1. `J` is lower by more than `1e-12 m`;
2. `WM` is lower by more than `1e-12 m`;
3. the challenger strictly wins at least six of eight fold scores;
4. the challenger strictly wins seed-level `J` for at least two of three
   seeds;
5. challenger `F <= comparator F + 1e-12 m`.

All three contrasts must pass for the primary claim to pass. Equality is not a
strict win. A missing, non-finite, or context-mismatched input fails closed.
The gates are evaluated exactly once after all fixed predictions exist. A
partial pass may be listed descriptively but cannot be called success.

The six stress folds and signed shifts `[-50,-20,-10,10,20,50] ms` are
report-only. Signed shifts reuse persisted zero-shift seed-ensemble predictions
and the public preparation reconstruction API; they cannot refit, forward,
select a checkpoint, tune, gate, or change the claim. The fixed count is 252
candidate/fold/shift ensemble cells and 1,260 view-metric cells. UID-support
hashes and dropped-edge counts are mandatory. The frozen preparation requires
zero dropped edges; a nonzero count fails the signed report closed but cannot
change the primary gate.

## Terminal lineage and source closure

Before native model implementation, a separate read-only auditor must issue
`openaxes.external.model_benchmark.v1.terminal_for_native_v2.v1`. It may bind
only existing validated v1 artifacts and may not refit v1, reread source truth,
or invoke its task evaluator. The receipt must bind the exact v1 terminal
stage, complete and development manifests, source set, 336-task index,
preparation manifest/table/folds, frozen `masked_erm` identity, and the exact
seed/fold/view prediction and metric inputs needed by the A-versus-v1 gates. It
must also bind the raw hashes of these two preregistration files and attest that
v1 outcomes did not alter them. No v1 performance number belongs in this
preregistration.

The parent auditor raw-hashes every recorded per-task checkpoint, transform,
history, training receipt, prediction parquet, evaluation-row parquet, outer
metric, bundle, and result reference, but it never decodes the binary files and
parses outer-metric JSON only for the 24 masked-ERM primary tasks. It validates
the 48-item scenario/shift support seal and the execution session, launch,
worker/controller, canonical attempt, and resume binding lineages. It remains a
JSON-only receipt auditor and is weaker than the official v1 validator: it does
not reproduce v1 metrics or perform the full signed-source semantic replay.
Ten successful official v1 validator runs are a separate operational
precondition. The three parent files are local SHA-256 consistency seals, not
digital signatures or proof against a hostile actor able to rewrite all local
evidence. A partial three-file write is invalid and requires a fresh v2 run;
success is returned only after all three persisted files are reopened and the
public authoritative validator recomputes them.

The receipt's raw SHA-256, its audit SHA-256, and its source-set SHA-256 are
written into the v2 execution manifest only after the receipt actually exists.
Empty, null, `TBD`, `UNKNOWN`, guessed, or placeholder values fail closed. A
missing or mismatched receipt prohibits native model implementation and execution.

The YAML lists the exact 15-file future v2 execution closure: these two files,
the dedicated parent-receipt document/module/test, both lockfiles, the executed
package `__init__.py`, shared `io_utils.py`, three other v2 modules, and three
other v2 tests. The declared future files do not yet exist; their absence is
expected at preregistration time but blocks a run. After literal implementation,
a separate audit must hash every exact path, reject undeclared executable source,
bind this canonical configuration, bind the receipt-bound preparation public-API
source set and the v1 receipt, check both 6,979-parameter counts, and freeze the
source-set receipt. The full source tests use the same config-independent
boundary as the parent audit: exact eval-Python bytes and version,
`-P -m pytest -c /dev/null --noconftest -p no:cacheprovider -q` plus the four exact
test paths, canonical project cwd, `PYTHONPATH=src`, no plugin autoload, and no
`PYTEST_ADDOPTS`, under the same exact sanitized environment. Any later config,
source, environment, parent-artifact, or
receipt drift prohibits fitting and outer access.

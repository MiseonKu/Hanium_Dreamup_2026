# OpenAXES native-v2 parent-receipt auditor contract

This document defines the sole implementation permitted before a successful
OpenAXES model-v1 lineage is prepared: the read-only parent-receipt auditor declared by
`configs/openaxes_external_native_v2.yaml`. The machine-readable
preregistration remains authoritative. This implementation does not authorize
native-v2 model code, fitting, prediction, evaluation, or outcome access before
the validated v1 terminal stage.

Authority erratum 1 followed the outcome-free `openaxes-v1-sealed-001`
canonical-JSON key-order failure. Its preserved supervisor terminal receipt is
`openaxes-v1-runs/openaxes-v1-sealed-001/LATEST_TERMINAL.json`, raw SHA-256
`e032c62b899efd7e90647748f6c087b05d271ae1e312fc8ec64f390cc1bc20e3`.
That run reached public preparation/validation but had zero model `PREPARED`
markers, execution sessions, scientific fits, predictions, evaluations, and
results.

Authority erratum 2 follows the independent outcome-free
`openaxes-v1-sealed-002` PyTorch CUDA-UUID representation failure. Sealed-002
had a valid public preparation and model `PREPARED`/task-index/pre-fit/session/
launch authority, but it failed before any worker-ready receipt or task
attempt. It produced zero task ledgers, attempt ledgers/rosters, result
directories, fits, predictions, evaluations, and results; no model result or
metric artifact existed or was inspected. Both failed lineages and both prior
source-audit authorities remain immutable history. The successful lineage is
now exclusively `openaxes-v1-sealed-003`.

## Frozen source and timing boundary

The auditor source closure contains exactly these eight project-relative paths:

1. `configs/openaxes_external_native_v2.yaml`
2. `docs/OPENAXES_EXTERNAL_NATIVE_V2_CONTRACT.md`
3. `docs/OPENAXES_EXTERNAL_NATIVE_V2_PARENT_RECEIPT_CONTRACT.md`
4. `requirements-eval.lock`
5. `src/imu_vive_learning/__init__.py`
6. `src/imu_vive_learning/io_utils.py`
7. `src/imu_vive_learning/openaxes_external_native_v2_parent_receipt.py`
8. `tests/test_openaxes_external_native_v2_parent_receipt.py`

All eight paths, including both preregistration blobs, must exist as ordinary
Git blobs in one audited source commit. The source-audit receipt maps every
exact path to a closed `{path,bytes,sha256}` reference and defines
`source_set_sha256` as SHA-256 of canonical JSON for that path-keyed map. The
active erratum source audit is committed later at
`receipts/openaxes_external_native_v2_parent_receipt_source_audit_erratum2.json`.
The audited source commit must be an ancestor of that evidence commit. Both
commits must be ancestors of v1's immutable `model_contract_git_head`.

The original authority is source commit
`49d9cb64406d6f5c9457e210623c45324a0880ac`, evidence commit
`ce09392a287f00447c5451471eae854d6aa726e6`, and path
`receipts/openaxes_external_native_v2_parent_receipt_source_audit.json`.
Erratum 1 is source commit
`6f5b1226cff400b708f0c9d167973ce505c48938`, evidence commit
`26795e6c34a7ac11995875134d8ff81758758622`, and path
`receipts/openaxes_external_native_v2_parent_receipt_source_audit_erratum1.json`.
All six identities remain immutable. The source-audit CLI can write only the
erratum-2 path; issuance and public validation reject either historical path,
schema, source commit, or evidence commit.

The closed source-audit receipt adds one exact `erratum` object. It binds the
active path; both historical authority tuples; sealed-001 and sealed-002 raw
evidence; successful sealed-003 roots; the UUID observation; exact sealed-002
stage counts; scoped absence of model result/metric artifacts; and unchanged
science. Its contract is
`openaxes.external.native_sensor_experts.v2.parent_receipt_source_audit.erratum2.v1`.

Sealed-001's failure was a canonical-JSON key-order consumer error, not a data
or scientific error. The producer formed sensor groups in
`humerus,radius,carpus,digitus` order; sorted canonical JSON persisted object
keys as `carpus,digitus,humerus,radius`. The mapping content and feature lists
were correct and have canonical SHA-256
`30deb94cc249ec6c2d17ac141b5450a30e4c526c4f96378a8a57b0c94443762a`.
The corrected consumer checks the exact key set and flattens by the frozen
semantic sensor order rather than mapping insertion order.

The v1 configuration remains byte-identical at 12,099 bytes, raw SHA-256
`848a19cc270401f6c37d391b399832996324792382a4a926f93b1584d7186c51`,
and canonical semantic SHA-256
`ce6a828e6d92118e29b7c67ec166cc3a555709e78bb41ef0838a79e982e80a3e`.
One of the existing 12 tests reads the pre-erratum Git blob and requires the
new methods, hyperparameters, folds, seeds, and gates to match it exactly. The
current nine-file v1 source-set digest is
`dd6418badc644f7d7576c836d17074ee2914b4ae0cb555b26315eae8669b8488`.
Only the two outcome-free consumers and their contract/tests changed across
the two errata.

Sealed-002's failed consumer compared the direct string form of
`torch.cuda.get_device_properties(0).uuid` to the frozen NVIDIA UUID. Under
sealed PyTorch `2.12.1+cu132`, the `_CUuuid` string omits the literal `GPU-`
prefix while retaining exactly the same canonical hex value. GPU availability,
count, name, PCI bus ID, and `nvidia-smi` mapping were unchanged. The correction
accepts only canonical UUID forms, reconstructs `GPU-`, then performs the
frozen equality check. The preserved cleanup state is
`cleanup_blocked_nonterminal`; the refused resource tracker naturally exited,
without a manual signal, and subsequent telemetry and `/proc` scans showed
zero run markers. The reported `preparation_present=false` was a report-only
wrong-marker probe and does not override the valid `COMPLETE.json` preparation.
Because the source authority changes, sealed-002 resume is forbidden.

Source-audit creation accepts no v1 run path and reads no v1 result, metric, or
artifact. It does not accept an external test receipt. Only after verifying the
Git and worktree eight-file closure, it directly launches the current
`<project>/.venv312/bin/python` and captures exit status and exact UTF-8
stdout/stderr bytes. The receipt binds the launch path, real executable path and
hash, Python version, the base `sys.prefix` observed under `-S`, exact launch-venv site-packages, project
working directory, argv, sanitized environment, audited commit/source-set, and
exactly 12 passed with zero failed/skipped/xfailed. It records independent byte
counts and SHA-256 values for stdout, stderr, and their canonical process
report. The single accepted pytest summary line is either the short
`12 passed in <decimal>s` form or that exact form followed by pytest's
` (H:MM:SS)` wall-clock suffix, where `H` is a canonical nonnegative decimal
and `MM`/`SS` are two digits from 00 through 59. Arbitrary suffixes, malformed
wall-clock fields, duplicate summaries, and warnings remain fail-closed. The
producer and persisted-receipt validator use this same closed parser, derive
passed/failed/skipped/xfailed counts from the raw report, and require those
derived counts to equal the stored 12/0/0/0 fields. The
source-audit payload hash covers every top-level field except
`payload_sha256` itself.

The exact mandatory test launch is:

```text
<project>/.venv312/bin/python -P -S -m pytest -c /dev/null --noconftest \
  -p no:cacheprovider -q \
  tests/test_openaxes_external_native_v2_parent_receipt.py
```

The production source-audit CLI process itself, and the test child it launches,
must both run through `env -i`, the project `.venv312/bin/python`, `-P -S`, and
the exact closed YAML environment. The CLI fails closed by checking
`sys.flags`, `os.environ`, `sys.executable`, `sys.path`, and the auditor
`__file__` before launching tests. `PYTHONPATH` is exactly
`src:<absolute-.venv312-site-packages>`. `-S` prevents `site`, `.pth`, and
`sitecustomize` processing; direct files/symlinks and undeclared import
directories below `src` are rejected. Every version in `requirements-eval.lock`
is checked through `importlib.metadata`. The receipt binds raw `pyvenv.cfg`
bytes and its five exact fields; for each of the 27 pins it binds the sole
dist-info `METADATA` and `RECORD` plus the count, byte total, and canonical
manifest SHA-256 of every actually present RECORD-owned file. Every RECORD row
has exactly three fields; its raw POSIX path is checked before resolution, its
`sha256=<unpadded-urlsafe-base64>` digest and canonical decimal size are
compared with the no-follow raw file read, and `Distribution.files` must equal
the RECORD path set. Empty digest/size pairs are accepted only for that
distribution's RECORD itself or a generated CPython 3.12 `__pycache__` file.
The bound METADATA has exactly one normalized pinned Name and exact Version.
The installed file count, byte total, and manifest hash are runtime-computed
receipt values, not preregistered machine-specific constants. That installed
file closure is recomputed immediately before and after tests and must be
identical. The complete visible distribution name/version inventory is also
bound. Extra visible distributions are inventoried, not rejected, so this is
not a closed-world venv claim; explicit `PYTHONPATH`, `-S`, the source import
boundary, and disabled plugin autoload constrain the executed import surface.
The lock is version-pinned, not wheel/sdist-artifact-hash-pinned; its Windows
resolution comment is not represented as a Linux hash lock. Plugin autoload,
user site, bytecode writes, visible CUDA,
system/global Git configuration, replace objects, ambient `PYTEST_ADDOPTS`,
ancestor `conftest.py`, and project pytest configuration cannot affect this
launch.

## Allowed implementation surface

The module imports only the Python standard library. Its two subprocess call
sites are the fixed local `/usr/bin/git` wrapper and the exact source-test
runner above. Git argv is checked against a command-and-operand allowlist for
`rev-parse`, `ls-tree`, `cat-file`, `hash-object`, `for-each-ref`, and
`merge-base --is-ancestor`; it never invokes a remote operation. Static AST
checks reject aliases, dynamic import/eval/exec, extra subprocess APIs,
`os.system`/popen/spawn/exec/fork, and training/prediction/truth/evaluator/CUDA/
network calls. This is defense in depth over a Git-authenticated reviewed blob,
not a proof against a malicious program rewriting its own checker and not an
OS sandbox.

The module does not import or call model-v1 preparation, training, prediction,
truth reconstruction, or evaluation code. It does not import PyTorch, CUDA,
NumPy, pandas, PyArrow, an HTTP client, or a socket API. The post-v1 issuer
parses existing JSON receipts, hashes the preparation table as opaque bytes,
and never decodes that table or rereads source truth. It writes no v1 or
preparation file. The only production writes are the three O_EXCL files below
a new v2 `parents` directory.

## Canonical roots and references

The issuer requires canonical absolute `workspace_root`, `v1_run_root`,
`preparation_run_root`, and `v2_run_root` arguments. `workspace_root` must equal
`git rev-parse --show-toplevel`; `project_root` is exactly its
`09_SO101_IMU_VIVE_LEARNING` child. Every root must exist, equal its real path,
and contain no symlink component. The three run roots must be pairwise distinct
and may not contain one another.

The successful-lineage roots are not free caller labels. `v1_run_root` must be
exactly `<project>/artifacts/openaxes_external_model_v1/run_openaxes-v1-sealed-003`
and `preparation_run_root` exactly
`<project>/artifacts/openaxes_external_v1/prepared_openaxes-v1-sealed-003`.
Their basenames must retain the common `openaxes-v1-sealed-003` suffix.
Issuance and public validation both recompute these equalities. The declared
supervisor path `openaxes-v1-runs/openaxes-v1-sealed-003` is provenance only;
the parent auditor does not consume it or present it as scientific authority.

Filesystem references contain exactly `path`, `bytes`, and `sha256`. A path is
a nonempty relative POSIX path with no absolute prefix, backslash, empty,
`.`/`..`, or repeated-separator component. Resolution never uses the current
directory, a search path, or an alternate base. Each component is opened below
its declared root with `O_NOFOLLOW`; intermediate components must be
directories and the final component a regular file. Before issuance, every
root is opened componentwise from `/` with required `O_DIRECTORY|O_NOFOLLOW`,
and all handles remain live through final revalidation. Directory-component
identity uses device, inode, and mode, so unrelated sibling creation/removal in
a shared ancestor does not create a false failure. Each regular-file read
separately requires device, inode, mode, size, nanosecond mtime, and nanosecond
ctime to remain identical and rechecks the final parent entry. Root handles and
their parent directory entries are
revalidated after output fsync against the original pre-issuance
device/inode/mode baseline; creating `parents` never refreshes that baseline,
so a concurrent chmod also fails closed. This protects against ordinary path-component
replacement; a privileged kernel, mount-namespace, or compromised-OS attacker
is outside this contract.

V1 references resolve only below `v1_run_root`. The preparation manifest,
parquet table, and completion marker resolve at exact paths below
`preparation_run_root`; the preparation config resolves at the exact project
config path; and the receipt-declared canonical source ZIP is opened and raw
hashed at its absolute authority path. Copied parent
files resolve only below `v2_run_root`. Preregistration and source-audit blobs
are resolved from their bound commits with `git cat-file`, never by joining a
filesystem base.

## Git object proof

The repository object format is exactly SHA-1. `GIT_NO_REPLACE_OBJECTS=1` is
mandatory, and any `refs/replace/*` or legacy `.git/info/grafts` entry fails
closed. For every bound Git blob, the auditor:

1. resolves `<commit>:<repository-relative-path>` to an object ID;
2. requires the exact `git ls-tree` entry to have mode `100644` or `100755`
   (never `120000`) and `git cat-file -t` to report `blob`;
3. reads exact bytes with `git cat-file blob`;
4. requires `git hash-object --stdin` of those exact bytes to equal the object
   ID; and
5. records their project-relative `{path,bytes,sha256}` reference.

The native-v2 YAML and Markdown blob commit equals the eight-file audited source
commit. The copied source-audit bytes must exactly equal the evidence-commit
blob and its recorded OID. Missing objects, shallow history, drift, a wrong
type, or any nonzero ancestry result fails closed.

The terminal `auditor_source_audit` object additionally binds the exact active
receipt path, erratum ID, successful run ID `openaxes-v1-sealed-003`, and timing
scope. Its historical boolean names `committed_before_v1_prepare` and
`source_audited_before_v1_prepare` mean before the preparation of that
terminal-bound successful lineage. They do not claim erratum 2 preceded either
preserved failed preparation in sealed-001 or sealed-002.

## Read-only v1 terminal binding

Issuance is allowed only after pre-existing v1 `prepared_manifest.json`,
`PREPARED.json`, all 336 sealed result references,
`development_manifest.json`, `COMPLETE.json`, and the immutable signed-source
replay receipt agree on the frozen v1 contract.
The issuer checks the exact config and nine-file source-set identities, exact
336-task deterministic index, production markers, terminal stage, 336 fits and
outer evaluations, complete/development hashes, immutable marker, absence of a
challenge route/access, and the receipt-bound preparation manifest/table.

For every task, the auditor checks task/spec identity, the canonical attempt ID
derived from the first execution binding, attempt roster/ledger, the seven
exact phase filenames and previous-phase raw references, and the sealed
`result.json -> bundle.json` chain. It raw-hashes every recorded checkpoint,
transform, history, training receipt, prediction parquet, evaluation-row
parquet, outer metric, bundle, and result reference for all 336 tasks; it never
deserializes the binary files. Only for the exact 24 `masked_erm`
scenario-LOSO primary tasks does it parse the linked outer-metric JSON. The full signed-source
replay has a closed schema and binds the prepared/development references,
the exact `{source,config,manifest,parquet,complete}` projection of the
preparation snapshot (excluding its synthetic marker), parent validation/authority, signed report, frozen v1
validator source, source set, semantic flags, and payload hash. Its
`scenario_shift_support_identity` is an exact 48-item list ordered by the eight
lexical scenarios and, within each, shifts `[-50,-20,-10,10,20,50]`; every
item has exactly `scenario_id`, `shift_ms`, positive `base_uid_rows`, and
lowercase `base_uid_sha256`, and the canonical list hash is rechecked.

Every production execution binding has the closed four-field GPU identity
`{physical_index,uuid,pci_bus_id,name}` and exact raw references to the
launch-started, worker-ready, and controller-ready observation receipts. Those
paths are derived from the lowercase-hex launch ID and zero-padded transport
slot; the controller receipt's `controller_observations_sha256` must equal the
binding. The replay `receipt_kind` is exactly
`immutable_full_signed_source_semantic_replay`.

The execution session has its exact 17-field semantic/self-hash authority,
four unique GPUs, eight two-per-GPU slots, and 42 tasks per slot. Launch-start,
worker-ready, controller-observation, and latest launch-completion receipts are
closed and raw-bound. If `runtime/prelaunch_session_recovery.json` exists, its
raw reference, exact closed official receipt schema, session/pre-fit references,
frozen recovery-contract hash, semantic payload, and self-hash are recomputed;
every launch must carry that exact raw reference and semantic hash. Launches
also carry the exact authenticated session reference, every worker-ready
receipt carries the exact current launch reference, and controller references
remain exact. Each task binding must name the authenticated session slot whose
42-task roster actually contains that task ID. Phase zero equals the attempt's first binding. Later
resume phases must preserve the nine stable session/slot/GPU fields while the
six launch-specific fields may move to a higher launch ordinal; ordinals never
decrease, and two phases at the same ordinal must have identical full bindings.
An authenticated launch that crashed before controller observations may retain
a nonempty partial `worker_ready/launch_*` directory. Every present member must
be one unique in-range `slot_NNNN.json` receipt with a unique positive process
ID and exact session/launch/slot authority; orphan, empty, unknown, linked,
special, or extra members fail. A launch with controller observations still
requires the exact eight worker-ready slots and their controller equalities.

This is a receipt check, not a replay. No checkpoint is loaded, no prediction
table or evaluation-row table is decoded, no metric function is called, and no
source truth is reconstructed. Those opaque files are opened read-only solely
to recompute recorded byte counts and SHA-256 values. This parent JSON-only
audit is deliberately weaker than the official v1 validator: it does not
reproduce v1 metrics or perform the full signed-source semantic replay. Ten
successful official v1 validator runs are a separate operational precondition
before issuance.

The issuer then creates a fresh `parents` directory and writes, in order:

1. `parent_receipt_auditor_source_audit.json` — an exact byte copy of the
   committed source-audit blob;
2. `v1_masked_erm_comparator_bundle.json` — the filtered comparator evidence;
3. `v1_terminal_for_native_v2.json` — the terminal lineage envelope.

Each file uses UTF-8 canonical JSON with sorted keys, comma/colon separators,
finite values, and exactly one final LF. Each is opened as a regular new file
with `O_CREAT|O_EXCL|O_NOFOLLOW`, fsynced, and followed by parent-directory and
v2-run-directory fsync. Existing `parents` evidence fails closed and is never
overwritten. The public authoritative validator reopens a `parents` directory
containing exactly these three regular nonsymlink files and no extra file or
subdirectory before it reads any parent receipt. If one new file write fails before fsync completion, only that
newly-created partial filename is unlinked through its pinned parent FD;
pre-existing evidence is never removed. There is no transactional all-or-none
guarantee across all three files: a failure after an earlier file was fsynced
leaves an invalid partial `parents` directory, and issuance must restart with a
fresh v2 run directory. Before returning success, the issuer reopens all three
persisted files and invokes the public authoritative validator.

## Masked-ERM-only comparator bundle

The bundle contract is
`openaxes.external.native_sensor_experts.v2.masked_erm_parent_bundle.v1`.
Its method is exactly `masked_erm`, seed order is `[1701,2603,3907]`, and view
order is `full`, then outages of humerus, radius, carpus, and digitus. Primary
fold order is the eight `scenario_loso` declarations in the preparation
manifest.

The bundle contains exactly:

- one initial-state SHA-256 per seed, identical across that seed's eight
  primary folds;
- 32 seed-ensemble outage contexts in primary-fold then outage-view order;
- eight seed-ensemble full contexts in primary-fold order;
- 96 per-seed outage contexts in seed, primary-fold, outage-view order;
- one exact 32-row context-identity seal;
- 24 masked-ERM primary training-receipt references;
- 24 masked-ERM primary outer-metric JSON references; and
- the v1 development-manifest reference.

It contains no other method identifier or metric, selection/ranking/stress/
signed outcome, persisted prediction table, evaluation-row table, source-truth
reference, or challenge reference. Training receipts provide only the frozen
initial-state identity and training seal. Per-task RMSE vectors come only from
already-persisted `outer_evaluation.json`; seed-ensemble contexts come only
from the already-sealed v1 development manifest. V1 formed those ensemble
scores from raw predictions before RMSE, so the individual metric JSON vectors
are not claimed to reproduce that ensemble statistic. The auditor does not
recompute either from prediction or truth rows.

All v2 gate inputs are recomputed from the closed bundle context rows. Float64
sums use the preregistered adjacent-pair binary tree with odd-tail carry, means
divide once by the exact count, folds and outages retain their frozen order,
and maximum ties retain the first ordered value. The bundle records `A`, `WS`,
`WM`, robust `J`, eight fold outage scores, three per-seed `J` values, and full
`F = 0.5*mean(full)+0.5*max(full)`. Its payload SHA-256 covers every exact
top-level key except `payload_sha256`.

## Commands

After the closure is committed, source-audit evidence is created once. The
command itself directly runs and records the exact sanitized tests:

```text
env -i <exact-environment-fields> \
<project>/.venv312/bin/python -P -S -m imu_vive_learning.openaxes_external_native_v2_parent_receipt \
  source-audit \
  --workspace-root <canonical-git-root> \
  --audited-git-commit <eight-file-source-commit> \
  --output <project>/receipts/openaxes_external_native_v2_parent_receipt_source_audit_erratum2.json
```

That receipt must then be committed before terminal-bound successful sealed-003
preparation. Only after validated v1 completion may the separate read-only issuance command run:

```text
env -i <exact-environment-fields> \
<project>/.venv312/bin/python -P -S -m imu_vive_learning.openaxes_external_native_v2_parent_receipt \
  issue \
  --workspace-root <canonical-git-root> \
  --v1-run-dir <canonical-v1-run-root> \
  --preparation-run-dir <canonical-preparation-run-root> \
  --v2-run-dir <canonical-v2-run-root> \
  --source-audit-evidence-git-commit <receipt-evidence-commit>
```

Neither production CLI nor either public builder/issuer API accepts a timestamp
override. `created_at_utc` is report-only and is always obtained from the
private `_now_utc` clock at the public boundary. Explicitly selected synthetic
tests monkeypatch that clock and substitute `_run_exact_source_tests`,
`_verify_v1_sources_at_model_head`, and
`_verify_live_source_audit_environment` so synthetic repositories can avoid
recursive test launches and unavailable real Git/V1 authorities. The CLI test
under that fixture proves parser, fixed path, exclusive creation, and receipt
shape only; it is not called an operational production bootstrap run. The real
`env -i` production source-audit command above, including its direct exact-8
test child and live dependency-closure checks, remains a separate mandatory
pre-successful-v1 operation and none of those production guards exposes a bypass. If the
`receipts` directory is absent on a fresh checkout, the source-audit command
creates exactly that directory through a pinned project dirfd with
`mkdir`/`O_NOFOLLOW`; no broader path creation is allowed.

## Failure and claim boundary

Every schema is closed. Duplicate JSON keys, non-finite values, missing/extra
fields, noncanonical parent JSON, unknown or placeholder identities, unsafe
paths, symlinks, special files, raw-hash drift, unexpected methods, wrong
counts/orders, derived-value drift, source/evidence mismatch, or Git ancestry
failure aborts without fallback.

The exported terminal validator is authoritative rather than a structural
dictionary checker. It accepts caller-pinned workspace, v1, preparation, v2,
and evidence-commit roots, reopens the persisted copied audit, bundle, and
terminal, and recomputes Git blobs/OIDs/dual ancestry, nine frozen v1 source
blobs, preparation identity, all result/phase references, and the comparator.
The structural validator used before output is private; caller-supplied bundle
or audit objects cannot establish public validity.

The three parent files provide locally persisted, SHA-256-sealed consistency
under the pinned-root and Git assumptions. They are not digital signatures and
do not prove authenticity against an external hostile forger able to rewrite
all local evidence. The receipt proves only fixed pre-outcome ordering and
faithful read-only binding of the declared v1 comparator. It does not prove scientific novelty,
independence, causality, observed physical outages, SO101 confirmation,
deployment readiness, or resistance to simultaneous total erasure of all local
evidence.

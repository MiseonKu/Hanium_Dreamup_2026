# OpenAXES native V3 parent recovery-validation erratum3

Status: outcome-neutral preregistration overlay, created before V3
implementation, fit, prediction, checkpoint selection, result, metric, or gate
evaluation and without reading a RUN003 scientific payload.

This document is additive. It does not replace or mutate the V3 all-mask
DMoME contract or either prior erratum. The exact V3 candidates, views,
parameters, training schedule, folds, seeds, metrics, gates, execution counts,
and claims remain those audited through commit
`f38a0213ed53f8698a9e12d23cc496f27084423d`.

## Defect and fixed authority decision

The erratum2 compound parent gate requires a validated RUN003 `COMPLETE.json`
and a separately validated external recovery terminal. That is necessary but
does not by itself prove that a recovery-aware validator exercised all three
validation layers at the same live boundary:

1. the embedded semantic validator for the exact seven recovered files;
2. the frozen full native-V2 RUN003 validator; and
3. the recovery authority validator for source audit, stage seal, external
   compound terminal, raw references, poststate, and action accounting.

The authority module's `validate-terminal` CLI is a necessary raw-authority
check but is insufficient on its own because it is not the recovery module's
full validation path. V3 parent authority therefore requires ten successful
fresh-process invocations of:

`python -m imu_vive_learning.openaxes_external_native_v2_recovery validate`

The implementation must use the exact argument order registered in
`configs/openaxes_external_native_v3_erratum3_recovery_validation.yaml`.
Every root and input file is passed through an inherited `/proc/self/fd/<n>`
descriptor opened without following symlinks. No invocation may share a
Python interpreter, validator object, or process with another invocation.
Ordinals 0 through 9 run strictly sequentially, and each later process starts
only after the previous process exits.

## Exact per-process evidence

The parent authority writes exactly these canonical JSON records in ordinal
order:

- `parents/native_v2_compound_recovery_validations/validation_00.json`
- `parents/native_v2_compound_recovery_validations/validation_01.json`
- `parents/native_v2_compound_recovery_validations/validation_02.json`
- `parents/native_v2_compound_recovery_validations/validation_03.json`
- `parents/native_v2_compound_recovery_validations/validation_04.json`
- `parents/native_v2_compound_recovery_validations/validation_05.json`
- `parents/native_v2_compound_recovery_validations/validation_06.json`
- `parents/native_v2_compound_recovery_validations/validation_07.json`
- `parents/native_v2_compound_recovery_validations/validation_08.json`
- `parents/native_v2_compound_recovery_validations/validation_09.json`

Each record has contract
`openaxes.external.native_sensor_experts.v3.parent_recovery_validation_record.v1`
and the exact key set frozen in the overlay configuration. It binds its
ordinal, invocation UUIDv4, PID and `/proc/<pid>/stat` start ticks; the
recovery implementation and source-evidence commits; and raw
`{path,bytes,sha256}` references to all of:

- the recovery source-audit receipt;
- external `STAGE_SEAL.json`;
- external `COMPOUND_TERMINAL.json`;
- RUN003 `COMPLETE.json`; and
- RUN003 `development_manifest.json`.

The retained input descriptors are hashed before and after the child and must
have identical descriptor identity, bytes, and SHA-256. The same five raw
references and the same recovery evidence commit are required in every
record. The source-audit receipt must name the recovery implementation commit,
the evidence commit must contain the exact receipt blob, and the
implementation commit must be a strict ancestor of the evidence commit.

Success means exit code zero with empty stdout and stderr. The record's exact
semantic assertion map states that the recovery module CLI passed; embedded
seven-file semantic validation passed; the frozen full RUN003 validator
passed; the nested recovery authority validator passed; live source-audit Git
object validation passed; stage and publication references are identical;
the prestate outside the seven authorized outputs is unchanged; signed-report
failure remains isolated from the primary gate; and no performance value was
disclosed. A parent controller may set these booleans only from the exact
validator exit and authenticated raw boundary, never by decoding a report.

PID/start-tick pairs and invocation UUIDs must each be distinct across all ten
records. Missing, extra, duplicate, reordered, noncanonical, failed, or
signaled invocations fail the V3 parent gate.

## Ordered receipt-set authority

For each canonical record, the parent forms a raw reference with exact keys
`path`, `bytes`, and `sha256`. It orders the ten references by validation
ordinal and computes:

`SHA256(canonical_json([reference_00, ..., reference_09]))`

where canonical JSON is UTF-8, keys are recursively sorted, separators are
`,` and `:`, non-finite numbers are forbidden, and there is no final newline
inside the hash. The parent receipt embeds both the ordered reference array
and `recovery_validation_receipt_set_sha256`. It also repeats the five common
raw references and recovery evidence commit, which must equal the values in
every record.

## Publication and failure boundary

The V3 parent gate is an `all_of_no_substitution` conjunction of the erratum2
compound pair and this ten-record recovery-aware validation set. RUN003
`COMPLETE.json` alone, the external terminal alone, either combined without
these ten records, or ten authority-CLI-only checks are all insufficient.

Any path, count, order, hash, raw identity, source ancestry, process identity,
semantic assertion, or cross-reference mismatch fails before V3
implementation. The gate may inspect receipt authority and raw references but
must not decode a prediction, metric, comparator, gate value, or scientific
report payload.

This erratum caused zero preparation API calls, RUN003 reads, fits, optimizer
updates, checkpoint selections, model forwards, predictions, evaluator calls,
or scientific outcome disclosures. It changes only parent-validation
authority.

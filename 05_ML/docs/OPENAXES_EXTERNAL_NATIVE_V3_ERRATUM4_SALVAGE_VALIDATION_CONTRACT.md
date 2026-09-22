# OpenAXES native V3 salvage-aware parent authority erratum4

Status: outcome-neutral preregistration overlay, issued before salvage
implementation or outcome and without calling the preparation API or reading
RUN003.

This overlay preserves all V3 scientific choices audited through erratum3.
It changes only the parent-authority consequence of the native-V2 recovery
stage that failed before its seal.

## Pinned V2 salvage authority

The preissued V2 authority is commit
`6a953c9996cc7ff446a1a6a3aa7e7c29ccc7044d`. Its exact Git objects are:

- `docs/OPENAXES_EXTERNAL_NATIVE_V2_RUN003_RECOVERY_ERRATUM2.md`, 4,390
  bytes, SHA-256
  `d053b8bed1aa7e427d06688b4978ad9bf6c48e105dd6b6da29a16bda682ae906`,
  blob `94da89b25763e91ee16659b5f03a8ad0528c1fd7`;
- `receipts/openaxes_external_native_v2_compound_recovery_salvage_preissue.json`,
  5,149 bytes, SHA-256
  `0b15daf0d47878d57a798a34bdc5df09092b4523dc1bd3070cf0653fe7a4f1b5`,
  canonical payload SHA-256
  `02ecc9a881ad6cef04ad4cd905ba7a6c1bcdaa205591167163d5881f225bd5a0`,
  blob `e6ad11ee6c558cf37d3b4e44490f9e28e15a4633`.

V3 consumes only these authority bytes and their later audited lineage. It
does not open the failed stage or any scientific field during preregistration.

## Dual lineage

The final V3 parent receipt must bind both independent histories:

1. the original recovery implementation commit
   `c77e8bb30a5cb1040194b3b0ade35cbd78aa3977`, its source-audit evidence
   commit `9cf2c36a47eb203ea07840b0410eec96c639fb43`, its exact source-audit raw
   reference, corrected standard stage seal, and standard compound terminal;
2. the future salvage implementation commit, its strict-descendant
   source-audit evidence commit, the exact V2 erratum2 and salvage-preissue
   Git blobs above, corrected-stage authority, and a separate terminal with
   contract
   `openaxes.external.native_sensor_experts.v2.compound_recovery.salvage.terminal.v1`.

The source audits must cross-bind these histories. The standard terminal and
salvage terminal are jointly required and neither may substitute for the
other. Authority-envelope validation alone is insufficient.

## Ten fresh-process composite validations

After successful no-recompute salvage, V3 parent issuance runs exactly ten
strictly sequential fresh Python processes. Each process invokes the
salvage-aware full validator exactly once and, within the same authenticated
validation boundary, invokes the original recovery module's full `validate`
path exactly once. Both must return success; an authority-only command cannot
satisfy either requirement.

Every process receives all roots and files through retained no-follow
`/proc/self/fd/<n>` descriptors. Descriptor identity and raw hashes must be
equal before and after the process. A process succeeds only with exit code
zero, empty stdout and stderr, and the exact semantic assertions registered
in the overlay configuration.

Each canonical validation record binds:

- both implementation and source-evidence commits;
- both source-audit raw references;
- the V2 salvage erratum2 and preissue raw references;
- the corrected standard stage seal;
- the standard compound terminal and separate salvage terminal;
- RUN003 `COMPLETE.json` and `development_manifest.json`;
- successful original full recovery validation and salvage-aware validation;
- exact seven-file, frozen full RUN003, original authority, salvage authority,
  failed/corrected-stage, dual-lineage, no-recomputation, and non-disclosure
  assertions.

PID/start-tick pairs and invocation UUIDv4 values are unique across ordinals
0 through 9. Processes are not reused or concurrent.

The parent creates exact `{path,bytes,sha256}` references for the ten records
in ordinal order and computes
`SHA256(canonical_json([reference_00,...,reference_09]))`. Canonical JSON is
UTF-8 with recursively sorted keys, compact separators, finite values only,
and no final newline inside the hash. Missing, extra, duplicate, reordered,
noncanonical, failed, or cross-inconsistent evidence fails before V3
implementation.

## Scientific boundary

Erratum4 authorizes no preparation API call, RUN003 read, reconstruction,
aggregate, gate replay, fit, update, checkpoint selection, model forward,
prediction, evaluator call, or scientific payload disclosure. It changes no
model, data, view, parameter, split, seed, training rule, metric, threshold,
contrast, execution count, or claim.

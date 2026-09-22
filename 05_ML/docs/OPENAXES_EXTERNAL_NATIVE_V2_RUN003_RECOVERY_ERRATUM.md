# OpenAXES native V2 RUN003 compound finalization recovery erratum

Status: active outcome-neutral authority, preissued before any RUN003 model
prediction, evaluation value, aggregate, gate value, or performance metric was
decoded or disclosed.

This document supersedes the unused RUN004 execution authorization committed
as `75f470b`. No RUN004 root, session, attempt, fit, prediction, label read,
evaluation, or result was created. The earlier document remains immutable
history of the rejected alternative.

## Decision

Fresh RUN004 is prohibited. Although attempt enforcement is stored below each
run root, a new root would scientifically repeat the same 126 TaskSpecs after
all 126 RUN003 fits completed. That is a larger deviation from
`maximum_attempts_per_task: 1` and `retry_refit_or_reseed_allowed: false` than
deterministically finalizing the already fixed RUN003 results.

Naively running patched code against RUN003 is also prohibited. RUN003's
scientific files are bound to its original 15-file source set and controller.
A new writer must not be misattributed to that source set. The only authorized
path is a compound terminal which separately binds the original scientific
execution authority and a new finalize-only recovery authority.

## Frozen failed prestate

The exact failed prestate and non-disclosure facts are those sealed in
`OPENAXES_EXTERNAL_NATIVE_V2_RUN004_ERRATUM.md`: 2,799 regular files,
318,851,615 bytes, canonical tree-record SHA-256
`e42472225abdaaa081da7a33dfdbe6d3543a607b579654a8ef7172aeb944b08b`,
126 complete task chains, an empty reports directory, and absent
`development_manifest.json` and `COMPLETE.json`.

The original science authority remains:

- run ID `openaxes-native-v2-sealed-003`;
- execution source-set SHA-256
  `3cc3133cd925fc3ebeacd53a4f746090a99d8b52d92ce2157d1d0b36576beb99`;
- controller source raw SHA-256
  `83a49e943494061bf0797d4acebe55a303aeb80e397db00f3a31d925d4ec9e59`;
- canonical configuration SHA-256
  `832a7618f0b7ce94c2813d4b5659153bb88d878d1e67f6ccba3f8f1e3555d59a`;
- 126 persisted `result.completed` receipts and their referenced artifacts.

No performance value has been read. Raw byte hashing, path counting, source
inspection, and the preparation-only signed-support diagnostic are not model
outcome disclosure.

## Recovery authority and allowed actions

The recovery implementation and tests must be committed and source-audited
before execution. It may run exactly once and may perform only these actions:

1. authenticate the frozen prestate, original source set, session, parents,
   task index, 126 attempt/phase chains, 126 result bundles, empty report set,
   and absent terminal files without decoding a performance value during
   preissue/audit;
2. replay zero-shift aggregation from the already persisted authenticated
   evaluation rows, with no source-truth read, fit, checkpoint, model forward,
   outer label read, or task-evaluator call;
3. invoke the receipt-bound canonical public preparation reconstruction API
   exactly once, validate its source/preparation/config identity and signed
   truth receipt, and join shifted targets to fixed seed-ensemble predictions
   by exact `(row_uid, shift_ms)` identity;
4. stage the exact five reports, development manifest, and COMPLETE bytes in a
   new external O_EXCL recovery state, validate all staged bytes, then append
   only those seven previously absent standard files to RUN003 with O_EXCL;
5. issue an external compound terminal receipt binding the original science
   source set, recovery source set, prestate, staged/output byte references,
   action accounting, and post-write semantic validation.

If canonical signed support contains any dropped row, the signed appendix must
be explicit `incomplete`, contain support/drop counts and hashes, and contain
no partial signed metric. This report-only state cannot change the primary
gate or block the compound terminal.

The recovery must account honestly for:

- original failed in-memory gate computation: 1;
- deterministic recovery gate replay: 1;
- persisted authoritative gate decision: 1;
- original noncanonical prepared-table signed read: 1;
- recovery canonical public signed reconstruction: 1;
- recovery fits, optimizer updates, checkpoint selections, model forwards,
  outer feature loads, outer label reads, and evaluator calls: all 0.

An ambiguous failure before the staged seven-file set is sealed fails closed.
After the stage is sealed, a crash may only copy the already authenticated
remaining staged bytes; it may not recompute any aggregate or signed truth.
Existing RUN003 files must remain byte-identical.

RUN003 `COMPLETE.json` alone is not publication authority. All downstream
consumers, including the native-V3 parent gate, must validate the pair of the
RUN003 terminal and the external compound recovery receipt. The recovery-aware
validator must semantically verify the primary reports, signed report branch,
cross-references, action accounting, source attribution, pre/post tree
boundary, and exact seven-file addition. The legacy payload-hash-only report
check is insufficient.

This erratum changes no model, data split, seed, training rule, checkpoint,
metric, contrast, gate threshold, or claim. It authorizes only transparent
finalization of results already fixed before the implementation defect was
observed.

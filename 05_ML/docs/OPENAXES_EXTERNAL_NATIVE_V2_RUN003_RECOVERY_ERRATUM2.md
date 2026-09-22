# OpenAXES native V2 RUN003 unsealed-stage salvage erratum

Status: outcome-neutral preissue, 2026-08-26T01:11:47Z.

This append-only erratum was written after the first authorized compound
recovery invocation failed closed and before any RUN003 performance value was
read or disclosed.  The prior recovery erratum, implementation, source audit,
and failed external stage remain immutable evidence.

## Observed boundary

The invocation authenticated the unchanged RUN003 failed prestate, replayed
the already persisted gates once, invoked the canonical public signed-truth
reconstruction once, and wrote the ordered seven candidate terminal files to
the fresh private external stage.  It then failed before `STAGE_SEAL.json`,
before any RUN003 write, and before `COMPOUND_TERMINAL.json` with the exact
exception `development summary payload hash changed`.

The exact seven raw references and both the failed-stage and corrected-stage
reference-set hashes are frozen in
`receipts/openaxes_external_native_v2_compound_recovery_salvage_preissue.json`.
RUN003 still has canonical prestate tree SHA-256
`e42472225abdaaa081da7a33dfdbe6d3543a607b579654a8ef7172aeb944b08b`,
2,799 regular files, 318,851,615 bytes, an empty `reports/`, and absent
`development_manifest.json` and `COMPLETE.json`.

## Exact implementation defect

The producer inserted a mutable four-report reference mapping into the
payload-hashed development summary and then mutated that same mapping by
adding `development_summary.json`.  Consequently the staged summary has
exactly one surplus self-reference while retaining the payload hash computed
before that insertion.  Removing only that self-reference yields canonical
newline bytes of length 2,113 and SHA-256
`86e97674b466f8c229f5b93e1f8c7fafb07c9e378850c484ff6fcd8707b69d14`.
The removed self-reference already names exactly those corrected bytes.  No
scientific value was needed to diagnose this aliasing defect.

## Authorized no-recompute salvage

The failed stage is permanently read-only evidence.  A separately committed
and source-audited salvage implementation may run once and only as follows:

1. authenticate the exact prior recovery authorities, unchanged RUN003
   prestate, exact failed-stage closed roster, raw references, absence of a
   seal, and empty original compound authority directory;
2. take two no-follow identity and raw-hash projections of the failed stage,
   before and after construction, and require byte-for-byte equality;
3. validate the four scientific reports without changing their bytes; require
   the development summary to have exactly the single aliasing defect above;
4. construct the corrected summary by deleting only its exact self-reference;
   preserve the other six staged files byte-for-byte after semantic and
   cross-reference validation;
5. write the corrected ordered seven files with `O_EXCL` into a new private,
   empty stage, run the full seven-file recovery validator, and only then
   issue the standard stage seal required by the original recovery publisher;
6. issue a separate external salvage receipt binding the failed stage, the
   corrected stage and seal, the prior recovery authority, the new salvage
   source authority, action accounting, and both stage projections;
7. publication may use the already audited original recovery publisher only
   from that corrected sealed stage.  The resulting original compound
   terminal and the separate salvage receipt are jointly required.

The salvage performs zero gate replays, canonical reconstructions, aggregate
calls, scientific payload recomputations, fits, optimizer updates, checkpoint
selections, model forwards, outer feature or label loads, and evaluator calls.
It may not delete, overwrite, rename, or modify any failed-stage or existing
RUN003 file.  A crash after corrected-stage sealing may resume by copying only
the same sealed bytes.  Any extra, missing, replaced, hash-drifted, symlinked,
special, hard-linked, or identity-changing member fails closed.

Downstream publication and native V3 must validate the original RUN003
terminal, the original compound terminal, the salvage terminal, and ten fresh
process successes of the salvage-aware full validator.  Authority-envelope
validation alone remains insufficient.  This erratum changes no model, data,
split, seed, training rule, checkpoint, metric, threshold, contrast, or claim.

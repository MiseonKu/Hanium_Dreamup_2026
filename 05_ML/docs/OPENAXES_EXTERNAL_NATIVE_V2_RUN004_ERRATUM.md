# OpenAXES native V2 RUN004 outcome-neutral execution erratum

Status: preissued before any RUN003 model prediction, evaluation value,
aggregate, gate value, or performance metric was decoded or disclosed.

Preissue time: `2026-08-26T00:07:42Z` (`2026-08-26T09:07:42+09:00`).

This erratum authorizes one fresh execution lineage named
`openaxes-native-v2-sealed-004`. It does not amend the frozen scientific
projection in `configs/openaxes_external_native_v2.yaml`: the three candidates,
6,979-parameter budgets, 14 folds, three seeds, 400 updates, training and
checkpoint rules, zero-shift aggregation, ordered primary contrasts, gate
thresholds, stress folds, and signed shifts remain exact.

## Preserved failed lineage

`openaxes-native-v2-sealed-003` completed all 126 fresh fits and all 882 durable
task phases, then stopped before writing any report, development manifest, or
completion marker. The exception was
`OpenAxesExternalNativeV2Error: signed-shift reconstruction dropped an edge`.
The run root is preserved and must not be resumed, completed, copied into the
new run, used for initialization, or used for any model or hyperparameter
choice.

At preissue the failed run contained 2,799 regular files and 318,851,615 bytes.
The SHA-256 of the canonical newline records
`project-relative-path<TAB>bytes<TAB>raw-file-sha256`, ordered by raw path, was
`e42472225abdaaa081da7a33dfdbe6d3543a607b579654a8ef7172aeb944b08b`.
Its exact top-level members were `PREPARED.json`, `parents`,
`prepared_manifest.json`, `quarantine`, `reports`, `results`, `runtime`,
`task_index.json`, and `tasks`. `reports` was empty;
`development_manifest.json` and `COMPLETE.json` were absent. There were exactly
126 `result.completed.json` phase receipts.

Bound evidence:

- failed run source-set raw SHA-256:
  `330e2a4643186242f74a3b5ebee627a93fc7c4383601b202a273ee66f42eeb6c`;
- failed run execution-session raw SHA-256:
  `9500a36414e1408e998502b320015b9eba07140a0919302a7673274e845bdea7`;
- failed run task-index raw SHA-256:
  `544a0ba786c9708047d0689124597a5d94a76d790fd544bc82fd71b80fe5975b`;
- controller stderr raw SHA-256:
  `e5eca4e44398db0603680bfaeb59a349f9bfd5d6bab6149a77da0e56e70cb682`;
- controller stdout was empty, SHA-256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- frozen YAML raw SHA-256:
  `c0115d95b9f6971d23b8b70d07b1dce55d0613ebf56eb963fe30175cbbb2f794`;
- frozen contract raw SHA-256:
  `591d1cf4409f80945c8b2916d9e220c57292efabd399e91f83c451d08f8016c6`;
- failed controller source raw SHA-256:
  `83a49e943494061bf0797d4acebe55a303aeb80e397db00f3a31d925d4ec9e59`;
- failed source/evidence Git head:
  `0c1bc2db1d1f7cefe7c7487aa1467812c8f249b2`.

No RUN003 performance report existed. No task prediction parquet, evaluation
row parquet, metric JSON, result value, zero-shift aggregate, primary gate
value, or stress value was opened during diagnosis. Raw hashes and closed-world
member counts are not performance disclosure.

## Outcome-neutral failure diagnosis

The failed implementation did not call the receipt-bound public preparation
reconstruction API. It instead looked up `base_aligned_time_us + shift` in the
prepared 100 Hz active grid. That private grid necessarily lacks boundary keys
and the first absence aborted the whole finalizer, even though the frozen
contract says a nonzero drop may fail only the report-only signed appendix and
must not change the primary gate.

An independent, model-artifact-free diagnostic invoked the canonical public
preparation reconstruction once before this preissue. It read only the frozen
source/preparation authority and disclosed no prediction or performance value.
It reproduced 22,482 base rows, 134,892 signed rows, exactly 22,482 rows for
each of the six shifts, exact base replay, and zero dropped rows in all 48
scenario/shift cells. This diagnostic invocation is not the RUN004 signed
report invocation and cannot be reused as a RUN004 artifact.

## Exact permitted implementation delta

Before RUN004 starts, a new implementation commit and a later append-only
execution-source-audit receipt commit must exist. The runtime delta is limited
to:

1. change only the run identity from sealed-003 to sealed-004;
2. bind and verify the preparation manifest's exact public-API source set;
3. call `reconstruct_openaxes_signed_shift_truth_v1` exactly once after all
   zero-shift predictions are fixed;
4. validate its receipt, schema, unique `(row_uid, shift_ms)` support, base
   identity, shift-time identity, dropped counts, and UID support before metric
   use;
5. join shifted targets to persisted seed-ensemble predictions by exact
   `row_uid` and shift without refit or model forward;
6. if a future authenticated reconstruction has any dropped row, write a
   discriminated incomplete signed report with no partial signed metrics while
   still writing the unchanged zero-shift primary terminal;
7. semantically validate both complete and incomplete signed-report branches;
8. add synthetic tests for the public-API boundary, zero-drop complete branch,
   nonzero-drop incomplete branch, tampering, and primary-gate independence.

No RUN003 artifact may be read by RUN004. RUN004 must freshly create its own
root, source set, execution session, attempt ledgers, 126 fits, predictions,
label claims, evaluations, reports, development manifest, and completion
marker. Each RUN004 task still has exactly one attempt. Artificial GPU load is
forbidden; utilization remains compute-bound.

This erratum changes operational correctness and provenance only. It provides
no permission to tune after outcome, rerun based on performance, or strengthen
the scientific claims and limitations frozen in the original contract.

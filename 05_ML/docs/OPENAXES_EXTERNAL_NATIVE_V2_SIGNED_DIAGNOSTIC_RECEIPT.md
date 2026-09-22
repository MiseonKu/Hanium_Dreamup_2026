# OpenAXES native V2 signed-support diagnostic receipt

Status: outcome-neutral diagnostic disclosure issued before any RUN003 model
prediction, evaluation value, aggregate, gate value, or performance metric was
decoded or disclosed.

This receipt corrects the historical statement in
`OPENAXES_EXTERNAL_NATIVE_V2_RUN004_ERRATUM.md` that the independent canonical
support diagnostic was invoked once. The independent auditor actually invoked
`reconstruct_openaxes_signed_shift_truth_v1` exactly twice before reporting the
final raw hashes: once to inspect row/drop counts and once to capture canonical
receipt/output hashes. No further diagnostic invocation is authorized.

Both calls used the same `.venv312` environment, `PYTHONPATH=src`,
`load_openaxes_config(configs/openaxes_external_v1.yaml)`, the authenticated
OpenAXES source ZIP, and
`prepared_openaxes-v1-sealed-003`. Neither call opened any V1/V2 model
prediction, evaluation, metric, aggregate, gate, or result payload. The public
reconstruction receipt attested `features_included=false`,
`model_fit_performed=false`, `metrics_computed=false`,
`selection_performed=false`, `claim_gate_decided=false`, and
`artifact_written=false`.

The second deterministic call produced:

- canonical reconstruction-receipt SHA-256:
  `a96db9fb32d8fc3465b609c643b3ca5e6e42f13732fe777ebddcfa37647fdffb`;
- signed long-table SHA-256:
  `ed3e884be3838acc58d2fbc5347da4d16ca7122f31bcf1b5eec48db145142ce0`;
- base-identity SHA-256:
  `93093657de2f07b6bb7151d8e3da1e363a3bc111fa3de012a1a4a825da01f82c`;
- total signed rows: 134,892;
- rows per each of six shifts: 22,482;
- exact scenario/shift dropped rows: zero in all 48 cells.

These two diagnostic calls are external preimplementation diagnostics. They
cannot be reused as a recovery artifact and do not reduce or increase the
compound recovery's separately authorized exact one canonical reconstruction
call. The compound terminal must report `external_diagnostic_calls=2` and
`authoritative_recovery_calls=1` as distinct fields.

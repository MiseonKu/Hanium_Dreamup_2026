# PORD-v5 completed-run validator amendment v1

## Scope

The completed PORD-v5 development run was trained and aggregated under the
original 48-file source fingerprint
`fe6756ed599cc113dfe0dabed3cddb96be037a5fb7a0d9aefac5d8e7052d27a3`.
The original experiment and training module SHA-256 values are respectively
`09541553ee538947d4d15942f47a28d6e13f0a13cf5dceba0d0725214aabd6af`
and
`bd64a7e3bdb38e37593058347970e452e980e7de64fff65e80e6eff5122ec5f5`.
Those files and the frozen source fingerprint remain the training authority;
this amendment neither replaces nor relabels them.

The post-run validator reconstructed every `checkpoint.npz` member as a
`numpy.ndarray`. The original feature-only prediction helper correctly accepts
the in-memory tensor state produced during training, but passes the reconstructed
arrays directly to `torch.nn.Module.load_state_dict`, which rejects them before
semantic replay. The checkpoint bytes are not defective.

Amendment v1 is a separate three-file validation source closure. During one
completed-run validation call it replaces only the experiment module's imported
`predict_pord_checkpoint_outer` reference with a wrapper that:

1. builds the frozen backbone;
2. requires exact tensor name order/count, shape, NumPy dtype, C-contiguity,
   finiteness, and the already-recorded selected-state hash;
3. converts without casting or reshaping to CPU tensors;
4. invokes the frozen prediction helper; and
5. restores the original function in `finally`.

The same scope installs fail-sentinels around training, AdamW construction,
outer evaluation, development execution, challenge prediction, and challenge
finalization. Any such call aborts validation. The original completed-run
validator is invoked only with `verify_current_source=True`; the amendment has
no false-valued source bypass and no resume route.

## Evidence boundary

Validation hashes the complete v5 run tree as ordered path/bytes/SHA-256 rows
before and after semantic replay and requires exact equality. The original
validator then replays 168 checkpoints and performs its frozen
`numpy.array_equal` comparison for all five views, giving 840 bitwise view-array
comparisons. No fit, optimizer update, outer evaluator, or challenge-label
access occurs.

The optional immutable receipt is written with `O_CREAT|O_EXCL` into the exact
sibling directory `<v5-run-name>.validator_amendment_v1`. It is never placed
inside or used to rewrite the v5 run. Its distinct fields bind the old training
fingerprint, the amendment-source fingerprint, original module hashes, complete
v5 tree hash, the development-manifest SHA-256, replay counts, zero
forbidden-call audit, scoped-patch restoration, and original validator result.

## Commands

From `09_SO101_IMU_VIVE_LEARNING`, after confirming the challenge marker is
absent, create the one-time sibling receipt with:

```bash
PYTHONPATH=src .venv312/bin/python -m \
  imu_vive_learning.gpu_outage_pord_v5_validator_amendment \
  --run-dir artifacts/gpu_endpoint_outage_v5_pord \
  --evidence-dir artifacts/gpu_endpoint_outage_v5_pord.validator_amendment_v1 \
  --create-receipt
```

Subsequent validation is read-only:

```bash
PYTHONPATH=src .venv312/bin/python -m \
  imu_vive_learning.gpu_outage_pord_v5_validator_amendment \
  --run-dir artifacts/gpu_endpoint_outage_v5_pord \
  --evidence-dir artifacts/gpu_endpoint_outage_v5_pord.validator_amendment_v1 \
  --validate-receipt
```

There is intentionally no `--resume`, training, finalization, or challenge
option. Because the run tree is large, the two full path/bytes/SHA passes are
expected to be I/O intensive; that cost is validation rather than scientific
computation.

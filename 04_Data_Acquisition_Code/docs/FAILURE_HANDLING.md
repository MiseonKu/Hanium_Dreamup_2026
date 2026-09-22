# Failure Handling -- Reference Index

Tight index of failure modes this package handles: what module implements
the handling, and which test exercises it. Not a design document -- read
the cited module's own docstring for the "why."

## Disconnect / reconnect

- **IMU (`collectors/collect_imu_rs232.py`)**: `IMURS232Collector._reader_loop`
  catches any I/O exception from the connection, closes it, publishes a
  `sensor_disconnected` event, waits `reconnect_wait_s` (default 1.0s), and
  reopens via the same `connection_factory` -- fixed 1s retry, not
  exponential backoff.
- **Encoder, `STREAM` path (`collectors/collect_esp32_encoder.py`)**:
  `Esp32EncoderCollector._reconnect_with_backoff` retries with doubling
  backoff (default `retry_backoff_s=1.0`, capped at `max_backoff_s=10.0`)
  when a `reconnect_fn` is given; without one, waits once and retries the
  existing connection. **Note (2026-08-05): this collector targets
  `02_SO-101_rtos_motor_driver`, which is not the firmware on this rig.**
- **Encoder, `STATE` path (`collectors/collect_esp32_state_poll.py`)** —
  the one this rig actually uses (`docs/ENCODER_COMMAND_COVERAGE.md`): it
  has **no reconnect logic of its own.** It is host-paced (`poll_once()`),
  so reconnect is the caller's responsibility. Two firmware behaviours make
  that non-trivial and are documented in `docs/FIRMWARE_MOTION_HAZARDS.md`:
  reopening the serial port reboots the ESP32 (so `CAL_AUTO LOAD` must be
  re-sent or `logical_deg` reads `NA`), and the reboot does **not** release
  servo torque (so `STATE`'s `torque=false` cannot be trusted as "the arm
  is loose"). Recovery from a latched fault must use `safe_engage()`, never
  `INIT`.
- Vive tracker (`collectors/collect_vive_tracker.py`) has no reconnect logic
  of its own -- `source.next_sample()` raising is caught and retried after
  `retry_wait_s` (default 0.5s); reconnect behavior for a real OpenVR
  source would live inside that source, behind the `VivePoseSource`
  interface. **Note (2026-08-05):** a real source (`OpenVRTrackerSource`)
  now exists and does not implement reconnect either. This is not a
  hypothetical gap: in segment 012 the tracker's battery died and the run
  produced 759 correctly-invalid records — the no-forward-fill policy held,
  but nothing noticed or recovered, and the capture was a write-off. The
  mitigation today is a pre-run check, not code
  (`docs/PRE_RUN_CHECKLIST.md` A1).
- **Test:** `tests/test_disconnect_reconnect.py` (both IMU and encoder,
  via `mocks/fault_injector.py:with_disconnect`).

## Checksum resync

- **`collectors/collect_imu_rs232.py`: `extract_frames`**. On an 8-bit-sum
  checksum mismatch, discards exactly one byte and re-searches for the next
  `0x55` header rather than dropping the whole buffer. Frames recovered
  this way are flagged `quality_flags: ["resynced"]` on the record.
  Reimplemented from `07_WT901C/WT901C_imu0/collect_4imu.py` (see
  `docs/07_WT901C_CODE_AUDIT.md`), which had zero tests for this logic.
- **Test:** `tests/test_wt901_parser.py` (clean frame, corrupted-checksum
  resync, and a frame split across two `read()` chunks) and
  `collect_imu_rs232.py`'s own `_self_check()`.

## Queue overflow / backpressure states

- **`core/bounded_queue.py`: `BoundedQueue`**. Five depth states by
  percent-full: `NORMAL` (0-60%), `WARN` (60-80%), `DEGRADED` (80-95%),
  `CRITICAL` (95-100%), `OVERFLOW` (100%). `put_nowait_tracked()` never
  raises and never silently drops -- on overflow it returns `False` and
  records `first_dropped_seq`/`last_dropped_seq`/`drop_count`.
- **Escalation policy: `core/health_supervisor.py`: `HealthSupervisor`**.
  Folds every registered stream's queue state into one health record per
  sample tick and exposes `should_suppress_live_output()` (DEGRADED+),
  `should_block_new_targets()` (CRITICAL+), `should_request_safe_stop()`
  (OVERFLOW) for a caller to act on.
  `experiment/session_controller.py`'s `run_for()` polls
  `should_request_safe_stop()` and triggers a clean-as-possible shutdown if
  it trips.
- Pull-style collectors (encoder, vive_tracker) don't own a `BoundedQueue`
  themselves -- `experiment/session_controller.py`'s
  `_WriterQueueDepthAdapter` sources the same depth-state classification
  from the `JsonlWriter`'s own backlog (`pending_count()`) instead, so they
  still feed the health supervisor a real signal.
- **Test:** `tests/test_queue_overflow.py` (state transitions at each
  boundary) and `core/bounded_queue.py`'s own `_self_check()`.

## Interrupt-safe shutdown

- **`core/interrupt_handler.py`**: wires `SIGINT`/`SIGTERM` (+`SIGBREAK` on
  Windows) to set a `shutdown_requested` flag instead of raising/exiting
  immediately -- flag-setting only, per master-spec section 16 step 1.
- **`experiment/session_controller.py`: `SessionController.shutdown()`**
  implements the rest of the 12-step sequence: set flag -> request safe
  stop on every collector -> join pull-style reader threads and close their
  writers -> stop self-managed IMU collectors (drains their own queue +
  writer) -> stop the health writer -> stop the event logger -> write the
  final manifest (`session_complete`/`interrupted` flags). Idempotent -- a
  second call is a no-op. `clean = not interrupted`: an interrupted
  shutdown leaves every stream's segment file as `.partial` rather than
  atomically renaming it (see partial-file recovery below).
- **Test:** `tests/test_interrupt_shutdown.py` (signal handler sets the
  flag) and `tests/test_session_controller_smoke.py` (end-to-end: a short
  mock session produces a complete manifest + raw files).

## Partial-file recovery

- **`core/jsonl_writer.py`: `JsonlWriter`**. Always writes to
  `<stream>_segment{NNN}.jsonl.partial` first; only a clean session end
  (`close(clean=True)`) atomically `os.replace()`s it to the final
  `<stream>_segment{NNN}.jsonl` name. An interrupted close
  (`close(clean=False)` / `close_interrupted()`) leaves the `.partial` file
  in place, flushed and fsynced up to whatever made it into the write
  queue. Refuses to reopen/append into an existing `.partial` path
  (`open(..., "x", ...)`) -- a crashed segment is never resumed, only
  superseded by a new, higher `segment_id`.
- **`core/recovery.py`: `recover_partial_jsonl`**. Reads a `.partial` file
  line by line; the first line that fails to parse (and everything after
  it, since a crash mid-write only ever corrupts the tail) is separated out
  as one dropped fragment. Writes `<name>.recovered.jsonl` (valid records
  only) and `<name>.recovery_report.json` (dropped-fragment detail +
  `host_seq` gap list) -- never modifies the original `.partial` file.
- **CLI:** `tools/recover_partial_jsonl.py <path>.jsonl.partial`.
- **Test:** `tests/test_jsonl_recovery.py` (valid records recovered,
  truncated tail flagged, deliberate `host_seq` gap reported) and
  `tests/test_jsonl_writer.py` (writer-side flush/fsync/drain/rename
  behavior).

## Frozen-sensor guard

Two independent checks, deliberately kept both (they look at different
data -- see `experiment/adaptive_settling_detector.py`'s docstring):

- **Collector-side, per-frame-type (`collectors/collect_imu_rs232.py`)**:
  for accel/gyro frames only, tracks a run length of byte-identical
  consecutive frames per frame type; at `stale_run_length` (default 20)
  repeats, flags `quality_flags: ["stale_suspect"]` on the record. Marked
  with a `ponytail:` comment as a naive exact-byte-equality heuristic --
  upgrade to a tolerance-based comparison if it ever produces a false
  positive in practice.
- **Detector-side, window-based (`experiment/adaptive_settling_detector.py`:
  `AdaptiveSettlingDetector._frozen_suspect`)**: trusts an upstream
  `stale_suspect` flag directly if present; otherwise checks the last
  `frozen_min_repeats` (default 5) samples in its own rolling window for an
  exact repeat (by `raw_hex`, or the parsed physical tuple as a fallback).
  Feeds into the `not_frozen` guard in `evaluate()` -- a sensor that looks
  stable only because it's frozen can never be declared
  `TIP_IMU_STABLE_CONFIRMED`.
- **Test:** `tests/test_stale_sensor.py` (collector-side flagging) and
  `tests/test_adaptive_settling.py` (detector-side guard, including that a
  guard failure resets the candidate-duration timer to 0).

## Other guarded failure modes (brief)

- **Duplicate / reordered packets**: `mocks/fault_injector.py`'s
  `with_duplicate_packets`/`with_reordered_packets` exist for testing;
  `collect_imu_rs232.py` flags an exact-repeat duplicate on any frame type
  via the record's `duplicate` field (see `docs/DATA_SCHEMA.md`).
- **Vive tracking loss**: never forward-filled or skipped -- see
  `docs/RAW_DATA_POLICY.md` rule 7 and `tests/test_vive_no_forward_fill.py`.
- **Hardware motion double-arm gate**: `experiment/safety_gate.py` --
  see `docs/EXPERIMENT_RUNBOOK.md` section 6. Its own `_self_check()` plus
  `experiment/session_controller.py`'s gate-check path are the only
  coverage; no dedicated `tests/test_safety_gate.py` file exists (see
  `docs/TASKS.md`).

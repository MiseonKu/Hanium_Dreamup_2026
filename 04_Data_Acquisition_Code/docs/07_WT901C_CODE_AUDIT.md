# 07_WT901C Code Audit

**Scope commit:** `3a0c757e913454ade5007cbacbb52bae48f321a0`
("Extend WT901C initial pose capture to 10 seconds")
**Audit date:** 2026-08-04
**Method:** Full read of every file under `07_WT901C/`, byte-level diff of the
three duplicated modules, grep for hardware-validation evidence. Read-only —
no files under `07_WT901C/` were modified to produce this audit.

> **Amendment (2026-08-05) — what real data has since settled.**
> This audit's findings about `07_WT901C/`'s own code stand as written and
> are not rewritten here. But it was produced before any hardware ran, and
> two of its blanket claims have since been overtaken by real captures in
> `04_Data_Acquisition_Code/sessions/real_motion_demo/`:
>
> - **"No hardware-in-the-loop evidence anywhere in the repo"** (summary
>   table, last row) is no longer true of the repo. 95k+ real WT901C frames
>   across 4 physically separate sensors, 100% checksum-OK, zero parse
>   errors. It remains true that `07_WT901C/`'s own code has no such
>   evidence attached to it.
> - **The frame-layout / byte-order `UNVERIFIED_HARDWARE_SETTING` flags**
>   (`s16()` little-endian low-byte-first; `0x55` as both header and frame
>   type) are **confirmed correct** by that data, as reimplemented in
>   `08`'s `collectors/collect_imu_rs232.py`.
>
> Still genuinely unverified, and not weakened by any of the above:
>
> - **The full-scale-range constants** (±16g / ±2000dps / ±180°). Never read
>   back from the device's range registers; captured data cannot settle
>   them, since a wrongly-scaled stream is self-consistent.
> - **The quaternion/Euler axis convention.** The device does not emit
>   quaternion (`0x59`) packets at all under its current output mask.
>
> One decode error this audit did *not* catch, found only in real data: the
> angle (`0x53`) packet's 4th word was being decoded as `temperature_c` the
> same way accel/gyro's is. It is not a temperature — it reads a constant
> 19582 counts (195.82 "°C") bit-identically across all 4 sensors and all
> 95,856 angle records, i.e. a version number. Fixed in `08` as
> `angle_packet_word4_raw`; `07_WT901C/`'s copy still has it.
> See `08/docs/IMU_DATA_COVERAGE.md`.

## Duplicate-file diff results

`diff -q` across `WT901C_imu0/`, `WT901C_encoder0/`, `WT901C_all_data/`:

| File | Result |
|---|---|
| `collect_4imu.py` | Byte-identical in all three folders |
| `robust_io.py` | Byte-identical in all three folders |
| `imu_pose_alignment.py` | Byte-identical in all three folders |
| `live_4imu_visualizer.py` | Diverged — intentional per-mode trigger logic, not drift |

The most recent commit touched all three `live_4imu_visualizer.py` copies
with the same symmetric edit (hardcoded delays replaced by named constants
`INITIAL_POSE_SETTLE_SECONDS=3.0` / `INITIAL_POSE_CAPTURE_SECONDS=10.0`).
That is not the only divergence — the three visualizers have always differed
in their continuous-recording *start-trigger* logic, by design:

- `WT901C_imu0`: gates recording start on IMU 3's gyro magnitude exceeding
  then staying below `IMU3_STOP_GYRO_THRESHOLD_DPS=3.0` for
  `IMU3_STOP_STABLE_SECONDS=0.5`s (`update_imu3_stop_gate`).
- `WT901C_encoder0`: adds an `EncoderFileReader` thread +
  `JsonFileEncoderSource`; gates recording start on two consecutive live
  encoder samples whose element-wise delta is exactly `0.0`
  (`handle_encoder_sample`).
- `WT901C_all_data`: no sensor-based gate at all — starts recording a fixed
  3 seconds after the button press, with the record label literally saying
  "temporary wait state before encoder integration" (엔코더 연동 전 임시
  대기 상태).

`imu0` vs `all_data`: 124 diff lines. `encoder0` vs `all_data`: 145 diff
lines. Structural, not accidental.

## Hardware verification status (repo-wide)

`README.md` does not contain a literal "never verified on real hardware"
sentence, but every other signal corroborates that conclusion:

- `encoder_source.py` docstring: *"Temporary encoder adapter. Replace
  `read_new_sample` for real hardware later."*
- README: `encoder_source.py` is *"an adapter placeholder until the real
  motor feedback transport is connected."*
- `WT901C_imu0/사용법.txt` labels the IMU3 stop thresholds *"현재 임시 정지
  기준"* ("current temporary/provisional stop criteria").
- `tests/test_robust_io.py` is the only test file in the tree and it tests
  `AsyncJsonArrayWriter` only (pure in-memory JSON writer). Zero tests exist
  for `extract_frames`, `decode_frame`, `s16`, `detect_baud`, or the
  checksum/resync logic — the actual serial protocol parser has no
  automated coverage.
- No committed sample captures, calibration logs, "tested against sensor
  S/N ..." comments, or hardware-in-the-loop harness anywhere.
- `DEFAULT_PORTS = ("COM22", "COM21", "COM20", "COM18")` looks tied to one
  specific developer machine; inconsistent with a `COM14` reference
  elsewhere in `collect_4imu.py`, suggesting these values were never
  stabilized against a fixed rig.

**Blanket flag: `HARDWARE_NOT_TESTED` applies to every file below.** This is
in addition to, not a substitute for, the per-file/per-function verdicts.

## File: `collect_4imu.py` (identical across all three folders)

**Purpose:** standalone CLI batch tool. Auto-detects baud on N given COM
ports, hard-synchronizes a common start instant across per-port threads,
captures for a fixed `--duration` seconds, writes a CSV of every decoded
packet plus a `summary.json`. Also the shared decode/parse library imported
by `live_4imu_visualizer.py` (`decode_frame`, `detect_baud`,
`extract_frames`).

### `extract_frames(buffer: bytearray) -> list[bytes]`
- Packet header: `0x55` (`buffer.index(0x55)`).
- Frame length: fixed 11 bytes (1 header + 1 type + 8 payload + 1 checksum).
- Checksum: `sum(candidate[:10]) & 0xFF == candidate[10]` — 8-bit sum mod 256.
- Partial-frame handling: present and correct — loop breaks
  (`if len(buffer) < 11: break`), leaving incomplete trailing bytes for the
  next call.
- Resynchronization: present — on checksum mismatch, discards exactly 1
  byte and re-searches for the next `0x55`. Clears the buffer only when no
  `0x55` exists in it at all.
- Mutates the caller's `bytearray` in place.

### `s16(low, high) -> int`
`struct.unpack("<h", bytes((low, high)))[0]` — little-endian signed 16-bit
reconstruction. Self-consistent; the assumption that the device sends
low-byte-first per axis is a protocol-layout assumption never verified
against real captures. **UNVERIFIED_HARDWARE_SETTING.**

### `decode_frame(frame: bytes) -> dict`
- Frame type byte = `frame[1]`, dispatched via `FRAME_NAMES` (`0x50`=time,
  `0x51`=acceleration, `0x52`=angular_velocity, `0x53`=angle,
  `0x54`=magnetic_field, `0x55`=port_status, `0x56`=pressure, `0x57`=gps,
  `0x58`=velocity, `0x59`=quaternion). `0x55` doubles as both the frame
  header byte and a valid frame type code — consistent with the known
  WitMotion WT901 layout pattern, but never checked against a real
  WT901C-RS232 unit here. **UNVERIFIED_HARDWARE_SETTING.**
- Scaling formulas (raw count → SI):
  - Accel: `raw/32768.0 * 16.0` → g (assumes ±16g full scale)
  - Gyro: `raw/32768.0 * 2000.0` → °/s (assumes ±2000 dps full scale)
  - Angle: `raw/32768.0 * 180.0` → degrees (assumes ±180° range)
  - Quaternion: `raw/32768.0` (no multiplier), w,x,y,z in packet order
  - Magnetic field: left as raw counts, explicitly labeled `mag_x_raw`/etc.
    (honest — no scaling silently applied)
  - Temperature: `raw/100.0` → °C
  - All full-scale-range constants are hardcoded assumptions never
    confirmed against a range/config register readback.
    **UNVERIFIED_HARDWARE_SETTING** — a differently-configured real unit
    would silently produce wrong physical units.
- Device's own embedded time (`0x50` frame) is decoded to a human-readable
  string but only used as an auxiliary column — never used to timestamp or
  correct a sample. Timestamp source is PC-arrival-only.

### `detect_baud(port_name, candidates) -> (baud, valid_count, error)`
Tries `[9600, 115200, 57600, 38400, 19200, 4800]`, `timeout=0.08`, requires
≥2 valid checksummed frames within 1.2s to accept.

### `collect_port(...)` (one thread per port)
- Serial config: `serial.Serial(port_name, baud, timeout=0.05)`, reads 1
  byte at a time in a spin loop.
- Hard start sync: busy-waits, discarding bytes before a shared
  `common_start_ns` instant, then resets the input buffer — good design so
  all 4 threads' elapsed times are comparable. **Not present** in
  `live_4imu_visualizer.py`'s `SensorReader`.
- Timestamp source: PC-arrival only (`time.perf_counter_ns()` at frame
  extraction).
- **Disconnect/reconnect: absent.** Single `serial.Serial(...)` open; any
  exception ends the thread with no retry — that port stops collecting for
  the rest of the run. Contradicts the README's "automatic serial reconnect"
  claim, which applies only to `live_4imu_visualizer.py`, not this CLI tool.
- Raw-byte preservation: yes (`raw_hex` per record).
- `records` list is lock-protected; `stats[port_name]` is written unlocked
  but each thread only writes its own key (safe under CPython's GIL, if
  undocumented as such).

**REUSE VERDICT:**
- `extract_frames`, `s16`, `decode_frame`, `FRAME_NAMES`, `detect_baud` —
  **REUSABLE_AFTER_FIX**. Header validation, checksum validation,
  partial-frame handling, resync, raw-byte preservation, and explicit unit
  scaling are all structurally present, but zero unit tests exist and the
  full-scale-range assumptions are unverified. Add synthetic-byte-stream
  tests (valid frame, corrupted-checksum requiring resync, split-across-reads
  partial frame) and confirm range assumptions against the actual configured
  units before trusting output units.
- `collect_port` / `main` (CLI batch tool) — **DO_NOT_USE** as a continuous
  collector: no disconnect/reconnect handling at all. Fine only as a short
  fixed-duration bench-test utility.

## File: `robust_io.py` (identical across all three folders)

Infrastructure utilities, no serial/protocol code.

### `AsyncJsonArrayWriter`
Streams a growing `"samples": [...]` JSON array from a dedicated writer
thread fed by a `queue.Queue`. 1.0s default flush interval. Producers only
call `.put()` (thread-safe); the writer thread is the sole file owner.
Draining before close is correct — `_STOP` sentinel is FIFO-ordered behind
queued items, `close()` joins the thread, `os.fsync()` is called before
final close. Tested with 4 concurrent producer threads × 100 items each,
plus an empty-capture case — real multithreaded coverage.

**REUSE VERDICT: REUSABLE.** Meets every applicable rejection criterion
(thread-safe hand-off, drains-before-close, fsync, real concurrent test
coverage). No serial/checksum concerns apply.

### `WindowsCollectionGuard`
Best-effort `SetThreadExecutionState`/`SetPriorityClass` (Above Normal) via
`ctypes.windll.kernel32`, restored on `stop()`. No-ops cleanly on
non-Windows. Not unit tested but trivial/low-risk (failure mode is "sleep
isn't prevented," not data corruption).

**REUSE VERDICT: REUSABLE_AFTER_FIX** (add a test; functionally fine as-is).

## File: `imu_pose_alignment.py` (identical across all three folders)

Post-processing math for "Initial pose registration" — resamples ~13s of
already-decoded 4-sensor records onto a common timeline via linear
interpolation, unwraps angle channels, averages the last 0.2s for a
reference pose. Not serial/protocol code.

- `unwrap_degrees`: correct sequential ±360° unwrap.
- `interpolate`: standard `bisect`-based linear interpolation, clamped
  extrapolation at both ends.
- `create_initial_pose_reference`: requires ≥2 samples per role/kind and a
  ≥0.2s common overlapping window across all 4 sensors, else raises. UTC
  anchor computed as the median of per-record offsets (robust to a single
  outlier record).
- Timestamp source: explicitly `"PC packet arrival time"`, consistent with
  the rest of the codebase.

**REUSE VERDICT: REUSABLE_AFTER_FIX.** Math is sound with no
serial/protocol concerns, but zero unit tests exist for the pure functions
(`unwrap_degrees`/`interpolate` are trivially testable and currently
untested), it's tightly coupled to the upstream record schema rather than a
documented/generic interface, and it has never processed a real 4-sensor
hardware capture end-to-end.

## File: `live_4imu_visualizer.py` (diverges across folders)

Tkinter+matplotlib GUI. Connects to 4 COM ports, live-plots roll/pitch/yaw,
and on a mode-specific trigger starts a 7-hour continuous RAW JSON capture
via `AsyncJsonArrayWriter`.

### Shared logic
- Quaternion/Euler helpers (`euler_to_quaternion`,
  `quaternion_conjugate/multiply/to_euler/normalize/rotate`): standard
  ZYX-intrinsic half-angle product formula, scalar-first `(w,x,y,z)`.
  Conventional/defensible but never stated or tested, and never checked
  against a real sensor's actual axis convention.
  **UNVERIFIED_HARDWARE_SETTING.** Used only for the live-plot display
  offset — not used in the persisted RAW JSON, so a convention error here
  is display-only, not data-corrupting. No unit tests (no round-trip
  `euler→quat→euler` check, no gimbal-lock check).
- `SensorReader(threading.Thread)` — the best-designed reader in the repo:
  - Serial config: `serial.Serial(port_name, baud, timeout=0.05)`.
  - Reads whatever's currently buffered (`serial.read(max(1, in_waiting))`),
    more efficient than `collect_4imu.py`'s byte-at-a-time read.
  - Partial-frame/resync/checksum: delegates to `extract_frames`/
    `decode_frame` — same properties as `collect_4imu.py`.
  - Timestamp source: PC-arrival only, same shared-anchor scheme, but
    **without** `collect_4imu.py`'s hard-synchronized-start busy-wait —
    each reader thread timestamps relative to a shared anchor established
    once at `connect()` time, no explicit boundary-alignment step.
  - Disconnect/reconnect: **present and correct** — outer loop closes the
    handle in `finally`, emits a status event, waits 1.0s, retries.
    `connection_epoch` increments on every successful (re)open and is
    stamped into every record, matching the README's "reconnect epochs for
    later gap auditing" claim.
  - Raw-byte preservation: yes (`raw_hex` per record).
  - Thread safety: reader thread never touches Tkinter state directly, only
    pushes decoded dicts onto a `queue.Queue` drained by the Tk main thread
    via `.after()` — correct producer/consumer separation. One minor gap:
    `self.serial` is written by the reader thread (on reconnect) and
    read+closed by the main thread without a lock; pyserial's `close()` is
    generally idempotent, so this is a narrow, low-severity race rather
    than a functional bug.
- `LiveVisualizer(tk.Tk)` — the GUI shell. Tightly coupled to
  Tkinter/matplotlib; not portable wholesale to a headless collector.

### Per-folder trigger-logic divergence
See the diff summary above — `update_imu3_stop_gate` (imu0),
`handle_encoder_sample` + `JsonFileEncoderSource` (encoder0), fixed 3s
delay (all_data). All three are self-described as provisional in their own
UI strings or usage docs.

**REUSE VERDICT:**
- `SensorReader` — **REUSABLE_AFTER_FIX**. Strongest reconnect/resync/
  epoch-tracking implementation in the repo, best starting point for a new
  collector's reader thread — but needs decoupling from the Tkinter-tuple
  queue protocol, a fix for the unlocked `self.serial` cross-thread access,
  unit tests (mocked/fake serial port), and hardware validation.
- Quaternion/Euler helper functions — **REUSABLE_AFTER_FIX**: standard
  formulas, untested, unverified axis-convention assumption.
- `LiveVisualizer` GUI class — **REFERENCE_ONLY**.
- Per-mode trigger implementations — **REFERENCE_ONLY / DO_NOT_USE as tuned
  values**: thresholds are self-described as provisional, never validated
  against real motion or a real encoder.

## File: `WT901C_encoder0/encoder_source.py` (unique to this folder)

`JsonFileEncoderSource` polls a JSON file (`encoder_input.json`) for new
samples via `(mtime_ns, size)` signature comparison, validates a non-empty
list of finite floats, raises `EncoderInputError` otherwise. Explicitly
self-documented as a stub: *"Temporary encoder adapter. Replace
`read_new_sample` for real hardware later."* No serial/checksum concerns
apply (it's a file-poller, not a device driver) — but there is no hardware
behind it at all yet. No unit tests.

**REUSE VERDICT: DO_NOT_USE** as an encoder driver — it isn't one, and per
the repo's own README it is slated for wholesale replacement once real
motor-feedback telemetry is connected (which, per the target-generator/
firmware audit, is `02_SO-101_rtos_motor_driver`'s `STREAM`/`ENCODER`
protocol — see DECISIONS.md).

## File: `tests/test_robust_io.py`

`unittest`-based coverage for `AsyncJsonArrayWriter` only: 4 concurrent
producer threads × 100 puts each drain into valid JSON with all 400 samples
present, plus an empty-capture case. Imports from `WT901C_all_data/
robust_io.py` (all three copies are byte-identical, so this covers all
three).

**REUSE VERDICT: REUSABLE.** Good pattern to carry forward for the writer
utility. Coverage gap to flag loudly: no tests exist anywhere in the repo
for the serial protocol parser (`extract_frames`, `decode_frame`, `s16`,
checksum, resync), `imu_pose_alignment.py`, or `SensorReader`'s reconnect
logic — the highest-risk, least-verified code has zero automated coverage.

## File: `README.md`

Accurately describes the three modes and the interruption-protection
feature set — every claim was independently verified against the code
above and holds true for the `live_4imu_visualizer.py` continuous-capture
path specifically, not for `collect_4imu.py`'s standalone CLI tool (which
lacks reconnect). Correctly states these safeguards don't survive power
loss/forced kill/reboot/storage failure, and correctly flags
`encoder_source.py` as a placeholder.

**REUSE VERDICT: N/A (documentation).** Worth carrying forward the
interruption-protection description into this package's docs, adjusted to
note it applies to the reconnecting reader path only.

## Summary verdict table

| File / unit | Verdict | Primary blocker(s) |
|---|---|---|
| `collect_4imu.py`: `extract_frames`, `s16`, `decode_frame`, `detect_baud` | REUSABLE_AFTER_FIX | No tests; unverified full-scale-range assumptions (±16g/±2000dps/±180°); unverified frame-layout convention |
| `collect_4imu.py`: `collect_port`/`main` (CLI batch tool) | DO_NOT_USE (as continuous collector) | No disconnect/reconnect handling at all |
| `robust_io.py`: `AsyncJsonArrayWriter` | REUSABLE | — (meets all applicable criteria, real concurrent test coverage) |
| `robust_io.py`: `WindowsCollectionGuard` | REUSABLE_AFTER_FIX | No tests (low risk) |
| `imu_pose_alignment.py` (all functions) | REUSABLE_AFTER_FIX | No tests; never run against a real multi-sensor capture |
| `live_4imu_visualizer.py`: `SensorReader` | REUSABLE_AFTER_FIX | No tests; unlocked `self.serial` cross-thread access; Tkinter-coupled queue protocol needs decoupling |
| `live_4imu_visualizer.py`: quaternion/Euler helpers | REUSABLE_AFTER_FIX | No tests; unverified axis-convention assumption |
| `live_4imu_visualizer.py`: `LiveVisualizer` GUI class | REFERENCE_ONLY | Tkinter/matplotlib-coupled, not portable to a headless collector |
| Per-mode trigger logic (IMU3 stop-gate / encoder Δ=0 / fixed delay) | REFERENCE_ONLY | Self-described as provisional; never validated against real motion/encoder |
| `encoder_source.py` / `JsonFileEncoderSource` | DO_NOT_USE | Explicit stub, not a real hardware driver |
| `tests/test_robust_io.py` | REUSABLE | Good pattern; but flags the coverage gap on everything else |
| **All files, blanket** | **HARDWARE_NOT_TESTED** | No hardware-in-the-loop evidence attached to `07_WT901C/`'s own code. ("...anywhere in the repo" as originally written is stale as of 2026-08-05 — see the amendment at the top of this file.) |

## Reuse decision for the new package

Functions carried into `04_Data_Acquisition_Code/` must be
reimplemented with the original file path + this commit SHA cited in a
comment (per master-spec section 4), not imported directly across package
boundaries, since `07_WT901C` code is itself only REUSABLE_AFTER_FIX at
best. See `DECISIONS.md` for exactly what gets ported vs. rewritten.

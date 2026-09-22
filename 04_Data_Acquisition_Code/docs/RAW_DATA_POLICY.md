# Raw Data Policy

This package acquires raw data only. It does not filter, resample,
coordinate-correct, or latency-correct anything at acquisition time. This is
existing project policy, not a preference invented for this task -- see
`DECISIONS.md` sections 5 and 6 for the audit evidence behind each rule
below.

## The rules

### 1. No filtering

No smoothing, no debouncing, no outlier rejection on sensor values. A noisy
or momentarily-wrong sample is written exactly as received, flagged if a
quality signal exists (`quality_flags`, `checksum_ok`, `valid`), never
silently cleaned up. `experiment/adaptive_settling_detector.py`'s metrics
(gyro RMS, jerk RMS, etc.) are a read-only *evaluation* of stability for
gating measurement-window acceptance -- they never modify or drop a raw
record. Per that module's own docstring: "Raw data collection is never
gated by this state machine... even if the stable detector fails, all raw
data stays."

### 2. No resampling

Every collector writes one record per sensor-native sample/frame/line, at
whatever rate the sensor actually produced it. There is no upsampling,
downsampling, or interpolation onto a common clock anywhere in this
package. `tools/audit_sampling.py` *measures* the actual rate/jitter/gaps
after the fact (see `docs/SAMPLING_AND_ALIASING.md`) -- it does not change
the data.

### 3. No coordinate correction

Vive poses are written in the tracker's own reported frame
(`frame_id: "vive_world"`); IMU orientation is written in the sensor's own
convention. No frame alignment, no axis remapping, no sign-convention fix is
applied at acquisition time, even where a known issue exists (e.g. the
WT901C quaternion axis convention is explicitly flagged
`UNVERIFIED_HARDWARE_SETTING` in `docs/07_WT901C_CODE_AUDIT.md` -- it is
carried through unmodified, not silently "fixed").

### 4. No latency correction

`core/clock_sync.py`'s `ClockSyncEstimator` computes a PC<->ESP32 clock
offset/drift estimate and RTT statistics, but per its own docstring:
"NEVER applies any correction to raw data records -- `report()` returns a
plain dict, nothing here mutates a `SyncSample` or reaches into any
sensor/encoder record." The reserved `t_sync_ns` field on every record
(`core/schemas.py`'s `base_record`) exists for a *future* downstream/
analysis-time step to fill in; no code in this package writes anything into
it. `t_pc_rx_ns` (raw PC-monotonic receive time) is the only authoritative
sample timestamp this package ever produces.

### 5. Raw is immutable

Once a JSONL line is written and flushed, nothing in this package rewrites
it. `core/recovery.py`'s partial-file recovery is the one place that
touches an existing raw file at all, and even there: "This module never
touches the original `.partial` file -- it only reads it and writes NEW
files next to it" (`<name>.recovered.jsonl`, `<name>.recovery_report.json`).
A crashed session's `.partial` file is left exactly as it was; recovery is
opt-in and additive, never destructive.

### 6. Raw and processed never mix in one file

Every stream is its own file (`<stream>_segment{NNN}.jsonl`), never merged
into a combined training-ready CSV -- this matches existing, pre-task
project policy (`실험절차.md:445-470`, cited in `DECISIONS.md` #6). This
package produces no processed/derived output at all: no clean-dataset CSV,
no ML-ready feature file, no coordinate-transformed pose. Building a bridge
from this raw format to any processed format is explicitly out of scope
(`DECISIONS.md` #5) and left for future post-processing work.

### 7. A lost/invalid sample is still a sample

The clearest instance of "raw, not curated": when Vive tracking is lost, or
an IMU frame fails checksum validation, or the encoder line doesn't parse,
the record is still written -- with `valid=False` and a `parse_status`
explaining why, and null data fields -- never skipped, and never
forward-filled from the last-good value. `core.schemas.vive_record`'s own
docstring states this as a hard rule, cross-referenced to `DECISIONS.md` #6
and master-spec sections 1.3/17.2 (Vive is ground-truth-only, never a
control-feedback signal, and tracking loss must never be papered over with
a stale value). `tests/test_vive_no_forward_fill.py` and the "frozen sensor"
guard machinery (`docs/FAILURE_HANDLING.md`) exist specifically to catch a
violation of this rule.

## What "raw" does NOT mean here

Being unfiltered does not mean uninterpreted. Collectors still do real,
necessary parsing work: WT901C frame decoding (checksum validation, raw
int16 -> physical units via a *documented, config-adjustable* scale
constant), encoder line regex parsing, degrees-to-radians conversion. These
are lossless, reversible unit/structure conversions applied uniformly to
every sample, not selective correction of some samples based on their
content.

**Caveat added 2026-08-05 — "lossless" is not the same as "meaningful".**
The encoder path's degrees-to-radians conversion operates on the firmware's
`logical_deg`, which is an arbitrary label interval mapped linearly onto
each axis's calibrated-active raw span — **not a physical joint angle**, and
distorted by a different factor per axis. `q_encoder_rad` is therefore a
faithfully-converted label, not a measurement of geometry. This is exactly
why raw-first matters here: `encoder_raw_tick` (and `raw_line`) preserve the
ST3215 counts the label was derived from, so the mistake was recoverable
after the fact rather than baked into the archive. Use raw ticks for
anything geometric. `docs/FK_CONVENTION_CORRECTION.md`,
`config/joint_limits.reference.yaml`. The distinction the rules above protect is: nothing here decides
"this sample looks wrong, I'll drop/smooth/replace it" -- that class of
judgment call is explicitly deferred to downstream analysis, operating on
the full, unmodified raw record.

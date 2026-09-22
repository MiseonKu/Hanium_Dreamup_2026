# Sampling / UART Budget Audit (real demo IMU captures)

Dated 2026-08-05. Extends `docs/SAMPLING_AND_ALIASING.md` (which already
covers `tools/audit_sampling.py`'s core per-stream jitter/gap metrics and
the "why measured_rate_hz looks like 4x configured_output_rate_hz" gotcha)
with the UART byte-budget question from the task: *is the WT901C link
naturally rate-limiting the ~10Hz-per-packet-type figure, or is there
headroom?* Numbers below are computed directly from
`sessions/real_motion_demo/imu*_segment{006,007,012}.jsonl` (real hardware,
read-only) via `tools/audit_sampling.py`'s new `compute_packet_type_rates`
and `uart_budget_report` functions
(`python tools/audit_sampling.py <path> --uart-baud 9600`), and archived
machine-readably at `outputs/sampling_audit.json` /
`outputs/packet_integrity.json` (`tools/generate_audit_outputs.py`).

## Confirmed inputs

- **Baud rate: 9600.** Confirmed directly (not assumed) from
  `experiment_scripts/run_real_imu_capture2.py` (`baud=9600` passed to every
  `IMURS232Collector`) and `experiment_scripts/README.md` ("WT901C IMU 4개가
  COM16~19에 9600 baud로 연결").
- **Frame size: 11 bytes, fixed.** `FRAME_SIZE_BYTES` in
  `tools/audit_sampling.py`; independently confirmed by
  `outputs/packet_integrity.json` -- every real IMU record across all three
  segments has `frame_length == 11`, zero exceptions, zero parse errors,
  100% `checksum_ok`.
- **Packet types actually observed: 4** -- `acceleration`, `angular_velocity`,
  `angle`, `magnetic_field`. Not the `accel/gyro/angle/quaternion` cycle
  `config/sampling.example.yaml`'s comment describes -- the real captures
  never emit a `quaternion` or `time` frame at all. Worth a follow-up
  correction to that comment, but out of this task's scope (parser/collector
  behavior is a concurrent workstream's, not this audit's).

## Measured rate

| Segment | Records (imu0_base) | Span (s) | Combined rate (Hz) | Per-packet-type rate (Hz) |
|---|---|---|---|---|
| 006 | 6040 | 149.87 | 40.30 | 10.08 |
| 007 | 14500 | 359.88 | 40.29 | 10.07 |
| 012 | 29510 | 732.53 | 40.29 | 10.07 |

All 4 IMUs, all 3 segments land within a tight band of the same figure
(`outputs/sampling_audit.json` has the per-sensor breakdown) -- this is not
segment- or sensor-specific noise, it's a stable, reproducible rate. This
confirms the user's own stated "~10Hz-per-packet-type" figure against real
data, rather than trusting it as given.

## The arithmetic: is 9600 baud the bottleneck?

8N1 framing (1 start + 8 data + 1 stop bit, no parity) = 10 bits/byte.

```
raw byte capacity = 9600 baud / 10 bits-per-byte = 960 bytes/sec
```

Required bytes/sec at the **measured** combined rate (4 packet types,
11 bytes/frame, ~40.3 Hz combined):

```
required = 40.3 frames/sec * 11 bytes/frame ≈ 443 bytes/sec
utilization = 443 / 960 ≈ 46%
```

**Answer: no, 9600 baud is not saturated.** At the measured rate the link
uses under half its raw byte capacity. The theoretical *ceiling* the raw
byte budget alone would allow is:

```
max combined rate = 960 bytes/sec / 11 bytes/frame ≈ 87.3 frames/sec
max per-packet-type rate (4 types) = 87.3 / 4 ≈ 21.8 Hz
```

The measured ~10.1Hz-per-type is well under this ~21.8Hz byte-budget
ceiling -- so the UART link itself is *not* what's capping the WT901C at
~10Hz. That cap comes from the sensor's own internal output-rate
configuration (the WT901C's `RRATE` register / firmware default for a
9600-baud link), not from serial bandwidth exhaustion. Raising the baud
rate alone, without also reconfiguring the sensor's internal output rate,
would not raise the measured rate above ~10Hz per type.

## What *would* it take to run faster?

`config/sampling.example.yaml` documents `output_rate_hz: 100.0` as a
target (matching what `mocks/mock_imu.py` accepts). At 100Hz full-sample
rate, all 4 packet types every sample:

```
required = 100 Hz * 4 packet types * 11 bytes/frame = 4400 bytes/sec
```

9600 baud (960 bytes/sec) falls ~4.6x short of that. To carry 4400 bytes/sec
with the same ~54% headroom margin observed above, the link needs roughly:

```
target baud ≈ 4400 bytes/sec * 10 bits/byte / 0.5 (target utilization) ≈ 88000 baud
```

**115200 baud** (960*12=11520 bytes/sec raw capacity, ~2.6x the 4400 B/s
required, standard rate, already used elsewhere in this repo for the
ESP32 encoder link -- `collectors/collect_esp32_encoder.py` defaults to it)
comfortably clears this with headroom to spare, and is the recommended
baud for a `dynamic_high_rate` profile if one is ever added -- **but the
WT901C's own output-rate register must also be reconfigured to actually
emit at 100Hz**, since (per the section above) baud rate alone did not
cause the ~10Hz ceiling this task measured. This is a documented
recommendation only; no runtime config-switching mechanism was built for
it (see `config/sampling.example.yaml`'s `imu_streams.output_rate_profiles`
comment block for the concrete numbers next to the existing 9600-baud
`raw_all_fields` defaults).

## Source data

- `outputs/sampling_audit.json` -- per (segment, sensor) `packet_type_rate_hz`,
  `combined_rate_hz`, and full `uart_budget_report()` output.
- `outputs/packet_integrity.json` -- per (segment, sensor) record count,
  `frame_length` distribution (always `{11: count}`), `checksum_ok` pass
  rate (1.0 everywhere in the real captures), parse-error count (0
  everywhere).

Regenerate both with `python tools/generate_audit_outputs.py`.

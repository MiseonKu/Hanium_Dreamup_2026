# Sampling and Aliasing

## What `tools/audit_sampling.py` measures

`compute_sampling_stats(t_pc_rx_ns, configured_rate_hz=None)` is a pure,
stdlib-only function (importable independently of the CLI -- see
`tests/test_sampling_audit.py`) over a stream's sequence of `t_pc_rx_ns`
values (the authoritative PC-monotonic receive timestamp on every record,
see `docs/DATA_SCHEMA.md`). It reports:

| Metric | Meaning |
|---|---|
| `sample_count` | Number of timestamps given. |
| `span_s` | Time from first to last timestamp (sorted defensively -- `out_of_order` records exist by design, see `core/schemas.py`). |
| `configured_rate_hz` | Echoed back from the argument (or auto-detected -- see below). Not computed from the data. |
| `measured_rate_hz` | `(sample_count - 1) / span_s` -- the actual observed arrival rate. |
| `median_period_ms` | Median of consecutive-sample time deltas. |
| `jitter_p95_ms` | 95th percentile of `\|period - median_period\|` -- how far a typical period strays from the median. |
| `max_gap_ms` / `max_gap_index` | The single largest consecutive-sample gap, and where it occurred. |
| `packet_loss_ratio` | Only computed when `configured_rate_hz` is known: how many fewer samples arrived than a gap-free stream at that rate would have produced over the observed span. `max(0.0, (expected_n - n) / expected_n)`. |

CLI usage:

```powershell
python tools/audit_sampling.py <path-to-jsonl> [--configured-rate-hz HZ]
```

If `--configured-rate-hz` is omitted, the tool reads the first record's
`sensor_config.configured_output_rate_hz` field itself (IMU records only --
see `_auto_configured_rate_hz`). For non-IMU streams, pass
`--configured-rate-hz` explicitly or `packet_loss_ratio` stays `None`.

## The IMU "4x rate" gotcha

**Read the amendment below this section first if you are looking at real
data — on real hardware the measured rate is ~0.4x, not 4x, and both facts
have the same cause.**

Running `audit_sampling.py` against an `imuN_*` stream file will show
`measured_rate_hz` roughly **4x** `configured_output_rate_hz`. This is not a
bug or a real 4x oversampling -- it's a direct consequence of the schema:
`collectors/collect_imu_rs232.py` writes **one JSONL record per decoded WT901C
frame** (accel, gyro, angle, or quaternion, cycling in that order), and a
full logical "sample" is 4 frames. `configured_output_rate_hz` describes the
full-sample rate (100 or 200 Hz); the per-record arrival rate this tool
measures is therefore close to 4x that. See `docs/DATA_SCHEMA.md`'s IMU
section and `mocks/mock_imu.py`'s `next_packet_bytes` docstring. If you want
a per-full-sample rate instead, filter to one `packet_type` first before
running this tool.

### Amendment (2026-08-05): what real hardware actually does

The "4x" arithmetic above is structurally right (one record per frame, not
per full sample) but its illustration was mock-derived, and two of its
details do not survive contact with the real units:

- **`configured_output_rate_hz` (100.0) is a label, not the device's rate.**
  Nothing in this package configures the WT901C's internal output-rate
  register. The real units emit **~10.1 Hz per packet type, ~40.3 Hz
  combined** — so `measured_rate_hz` comes out *below*
  `configured_output_rate_hz`, not 4x above it, and `packet_loss_ratio`
  against the 100 Hz label is meaningless (it will read ~0.9 with nothing
  actually lost). Measured over the real captures in
  `docs/SAMPLING_AND_UART_BUDGET.md`: 443 B/s of a 960 B/s raw capacity,
  46% UART utilization — the link is not the bottleneck, the sensor's own
  rate setting is.
- **Filtering to `"quaternion"` records will give you nothing.** The device
  never emits `0x59` quaternion (or `0x50` time, or `0x55` port_status)
  packets with its current output mask. The four types actually observed
  are acceleration, angular_velocity, angle, magnetic_field. Filter to one
  of those. `docs/IMU_DATA_COVERAGE.md`.

## Raw-first: no downsampling policy

This tool is diagnostic-only. It never modifies the data it reads, and
nothing in this package ever resamples, decimates, or filters a raw stream
to hit a target rate -- see `docs/RAW_DATA_POLICY.md` rule 2. A high
`packet_loss_ratio` or `jitter_p95_ms` is a signal to investigate (serial
buffer sizing, CPU contention, a marginal USB cable), not something this
tool or any collector "fixes" by smoothing the timeline.

## Aliasing consideration -- not exercised in this task

The master spec raises a 100Hz-vs-200Hz pilot-comparison idea: running the
same motion at both WT901C output rates and comparing what each captures,
as a way to reason about whether 100Hz risks aliasing real arm-tip motion
frequencies. **This comparison still has not been executed.**

**Amendment (2026-08-05):** the original reason given ("no real hardware was
available") is stale — real WT901C data and a real Vive-tracked motion
reference both exist now (`sessions/real_motion_demo/`). The blocker is
different and more concrete: **the sensors are not running at 100 Hz or
200 Hz.** They emit ~10 Hz per packet type, and neither rate is reachable
without writing the device's own output-rate register (and raising baud with
it) — see the amendment above and `docs/SAMPLING_AND_UART_BUDGET.md`.
Nothing in this repo writes that register.

So the prerequisite chain is: reconfigure the device → verify the new rate
with `tools/audit_sampling.py` → *then* run the same motion at both rates
and compare measured-rate/jitter against the actual motion-frequency
content. Until step one happens, any aliasing conclusion drawn from this
data is about a ~10 Hz stream, not a 100 Hz one.

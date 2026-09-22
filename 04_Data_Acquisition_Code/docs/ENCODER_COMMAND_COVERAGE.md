# Encoder / command data coverage (`03_SO-101_control/main` STATE protocol)

Written 2026-08-05. Companion to `docs/IMU_DATA_COVERAGE.md` — same
question, other stream: what does the device actually report, what did we
actually record, and what was silently thrown away.

## Which collector talks to what

| Collector | Firmware | Protocol | Used on this rig? |
|---|---|---|---|
| `collectors/collect_esp32_state_poll.py` **(new)** | `03_SO-101_control/main` | host polls `STATE`, firmware replies with one detail line per axis | **Yes — this is what is on the board** |
| `collectors/collect_esp32_encoder.py` | `02_SO-101_rtos_motor_driver/main` | firmware pushes `STREAM t_ms=… seq=…` lines | No — never flashed here (see `DECISIONS.md` #3/#7) |

Both are kept. The `STREAM` one is correct for the protocol it targets; it
just isn't the protocol this rig speaks.

## What the firmware reports vs. what was recorded

`printJointDetailLine()` (`03_SO-101_control/main/src/main.cpp:1400`) emits
**25 fields per axis per poll**. The ad hoc sweep script that produced the
real demo data parsed **6** of them:

| Field | In firmware output | Old ad hoc script | New collector |
|---|---|---|---|
| `raw_pos` | yes | ✅ `encoder_raw_tick` | ✅ |
| `logical_deg` | yes | ✅ `q_encoder_rad` | ✅ (`NA` → `None`) |
| `torque` | yes | ✅ `motor_status` | ✅ |
| `fault` | yes | ✅ `fault` | ✅ |
| `raw_temp` | yes | ✅ (used only for a temp trip, not stored per-axis) | ✅ `raw_telemetry` |
| `axis` / `alias` / `id` | yes | ✅ (routing only) | ✅ |
| `raw_speed` | yes | ❌ dropped | ✅ |
| `raw_load` | yes | ❌ dropped | ✅ |
| `raw_voltage` | yes | ❌ dropped | ✅ |
| `raw_current` | yes | ❌ dropped | ✅ |
| `mapping_ready` | yes | ❌ dropped | ✅ |
| `raw_cal_min` / `raw_cal_max` | yes | ❌ dropped | ✅ |
| `raw_active_min` / `raw_active_max` | yes | ❌ dropped | ✅ |
| `speed` / `acc` / `range_percent` / `direction` | yes | ❌ dropped | ✅ |
| `last_update_ms` | yes | ❌ dropped | ✅ |
| `read_failures` | yes | ❌ dropped | ✅ |
| `overload_samples` | yes | ❌ dropped | ✅ |
| `fault_latched` / `reason` / `latched_ms` | yes (separate line) | ❌ substring check only | ✅ `fault_detail` |
| estimated velocity | — (derived) | ❌ | ✅ finite difference |
| command↔response linkage | — (derived) | ❌ | ✅ `q_cmd_rad` + `command_record` |

## The dropped fields are recoverable from existing data

Because every record kept its `raw_line`, the old sessions can be
backfilled without re-running the hardware. Verified by replaying the real
files through the new parser:

```
seg006: 310/310 records fully parsed, all 3 axes, 38 fault lines
seg007: 542/542 records fully parsed, all 3 axes, 356 fault lines
seg012: 759/759 records fully parsed, all 3 axes, 6 fault lines
```

All 25 firmware fields recovered in every segment. This is the raw-first
policy (`docs/RAW_DATA_POLICY.md`) paying for itself — a parser gap cost
nothing permanent.

Note segment 007's 356 fault-latched records out of 542: the old script
only did a `"fault_latched=true" in text` substring check and never stored
`reason`, so *why* those faults fired was not recorded at the time. It is
in the raw lines and can now be extracted.

## Units — read before doing anything numeric

`raw_voltage`, `raw_temp`, `raw_current`, `raw_load` are **servo register
counts, not volts / °C / amps / newton-metres.** The new collector
deliberately does not scale them, and stores them under
`data.raw_telemetry`, not in `encoder_record`'s `temperature_c` /
`current_or_load` slots as if they were physical units. Converting them
needs the ST3215 datasheet, which is not in this repo — see
`docs/ST3215_THERMAL_AND_SEQUENCING.md`'s UNVERIFIED rows.

Separately: the `raw_temp` reading on the affected servo is **confirmed
broken** by the project owner. Do not build any control logic on it. The
real thermal safety net is external hardware.

Only `logical_deg` is converted (degrees → radians), because the firmware
itself already did the raw→physical mapping using the calibration in servo
EEPROM.

## Command / response linkage

Section 6 of the improvement request asked for `COMMAND_QUEUED` /
`COMMAND_SENT` / `COMMAND_ACK` with real timestamps. What the firmware
actually makes possible:

- `COMMAND_SENT` — `t_command_pc_ns`, taken immediately before the host
  writes `D<axis><deg>`.
- `COMMAND_ACK` — real, not assumed: the firmware echoes
  `[LEGACY] queued axis=<name> logical=<deg>` (main.cpp:3189) for each
  accepted command. `send_axis_command()` only sets `command_ack=True` if
  the echo names *this* axis. A command that was silently refused (offline
  axis, `mapping_ready=false`, fault latched) produces
  `parse_status="no_ack"` and a `command_not_acked` quality flag instead of
  a false success.
- `t_ack_pc_ns` + `ack_latency_ms` are recorded per command.
- `COMMAND_QUEUED` as a distinct host-side state is **not** modelled — the
  host does not queue, it writes straight to the port, so a separate
  queued-vs-sent timestamp would be two names for the same instant. The
  *firmware* queues (that's what the echo reports), and that is captured as
  the ACK.

**No ESP32-side source timestamp or sequence number exists in this
protocol.** `last_update_ms` is the firmware's `millis()` at the last servo
read for that axis (main.cpp:1431) — useful for staleness, but it is not a
per-message timestamp and there is no `seq` field. So `t_esp_us` and
`source_seq` stay `null` rather than being synthesised from `host_seq`,
per the "never fabricate a source field" rule in section 8 of the request
and `core/schemas.py`'s missing-is-null convention. The `05` firmware's
`STREAM` protocol *does* carry `t_ms`/`seq`; adopting it would be a
firmware swap, not a collector change.

## Tests

`tests/test_collect_esp32_state_poll.py` — 21 tests covering field typing,
the `NA` → `None` rule (a 0.0 there would read as a real commanded-zero
pose), negative `raw_speed`, fault-summary parsing, tolerance to an added
firmware field, missing-axis handling, unparsable-response flagging,
raw-telemetry preservation, first-sample-velocity-is-None, one-axis-per-
command, and ACK-for-the-wrong-axis not counting.

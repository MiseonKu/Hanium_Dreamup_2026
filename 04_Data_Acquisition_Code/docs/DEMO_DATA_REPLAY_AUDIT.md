# Demo data replay audit

2026-08-05. What the already-collected real data in
`sessions/real_motion_demo/` actually contains, verified by replaying it
rather than by trusting the run logs. Every number below came from reading
the real files; none of the raw data was modified.

Machine-readable companions: `outputs/sampling_audit.json`,
`outputs/packet_integrity.json`, `outputs/target_plan_audit.json`
(regenerate with `python tools/generate_audit_outputs.py`).
Executable form: `tests/test_demo_replay.py`.

## Segment verdicts

| Segment | Streams present | Cross-stream overlap | Vive valid ratio | Usable as fused label data? |
|---|---|---|---|---|
| 006 | 4 IMU + encoder + Vive | ✅ genuine overlap | **1.00** | ✅ pilot-grade |
| 007 | 4 IMU + encoder + Vive | ✅ genuine overlap | **1.00** | ✅ pilot-grade |
| 012 | 4 IMU + encoder + Vive | ❌ **none** | **0.00** | ❌ |

Segment 012 is the important one: it looks fine file-by-file. 759 encoder
records, 759 Vive records, zero parse errors, 50/50 planned targets all
present in the encoder stream, positions genuinely distinct (the run really
did move). It is still unusable, for two independent reasons:

1. **No temporal overlap.** The IMU capture window ends
   **352.3 seconds before** the encoder/Vive window begins. The IMU
   collector was a separate process with a fixed `DURATION_S`; the sweep was
   interrupted and restarted, and by the time the successful sweep ran, the
   IMU capture had already expired. There is no instant at which an IMU
   sample and an encoder sample coexist, so nothing can be fused.
2. **Vive tracking 0% valid.** All 759 Vive records have
   `tracking_valid=false` and null pose fields — the tracker went offline
   (battery / auto-sleep) before the run and nothing checked. Correctly
   recorded as invalid rather than forward-filled (the no-forward-fill rule
   held), but that means zero end-effector ground truth.

Neither failure raised an error at runtime. That is precisely why
`tests/test_demo_replay.py` asserts the **bad** case as bad, not just that
both segments parse.

## Packet integrity (all segments)

- Every IMU record: `frame_length == 11`, `checksum_ok == true`, zero parse
  errors, across all 4 sensors and all 3 segments.
- Every encoder record: `raw_line` present and fully reparsable — 310/310
  (seg006), 542/542 (seg007), 759/759 (seg012), all 3 axes, all 25 firmware
  fields recovered. See `docs/ENCODER_COMMAND_COVERAGE.md`.
- JSONL line-level integrity: no truncated or malformed lines in any
  non-`.partial` file.

The raw-first policy (`docs/RAW_DATA_POLICY.md`) is what makes the parser
fixes below retroactive instead of requiring a re-run.

## What replay revealed that the original run did not

| Finding | Evidence | Fixed by |
|---|---|---|
| Angle-packet word 4 was mislabeled `temperature_c` (constant 195.82 °C) | Raw word is exactly `19582` in **all 95,856** angle records, across 4 independently-wired sensors and 3 segments; accel/gyro temperature from the same dies at the same instants reads a normal, drifting 26–29 °C. WitMotion's published protocol: those bytes are `VL/VH` = version number. | Renamed `angle_packet_word4_raw`; `temperature_c` no longer set for angle frames |
| Magnetic packet raw counts were never decoded | `magnetic_field` raw bytes were preserved but `mag_raw`/`mag_uT` were empty in every record | `decode_frame` now decodes `mag_raw`; `mag_uT` stays null (no protocol-defined scale factor — not invented) |
| Magnetic word 4 is documented as temperature but reads exactly 0 | Exactly `0` in all **50,074** magnetic records | Kept as `mag_packet_word4_raw`, not asserted as temperature |
| 19 of 25 encoder fields were being discarded | Firmware emits 25 fields/axis (main.cpp:1400); the ad hoc script parsed 6 | New `collectors/collect_esp32_state_poll.py` captures all of them |
| Fault *reasons* were never stored | Segment 007 has **356 fault-latched records** out of 542; the old code did a `"fault_latched=true" in text` substring check and dropped `reason`/`latched_ms` | New collector stores `fault_detail`; recoverable retroactively from `raw_line` |
| Sensor rate, not the UART link, caps throughput | Measured 4 packet types × ~10.08 Hz = ~40.3 Hz combined = 443 B/s vs. 960 B/s capacity at 9600 baud 8N1 → **46% utilization** | Documented in `docs/SAMPLING_AND_UART_BUDGET.md`; needs a sensor-side output-rate register change, not just more baud |
| Device never emits quaternion (0x59) or time (0x50) frames | Zero occurrences in any captured session | Parser support exists; hardware output mask needs enabling |

## Target plan

`target_plan_3d_random_n50_v3.json` was confirmed — not assumed — to be the
plan that drove segment 012, by matching recorded `logical_deg` for T000
(`[48.3, 14.4, 92.6]`) against the plan's T000 (`[48.26, 15.19, 92.51]`).
The v1/v2 plans reuse the same `T000`–`T049` ids with different values, so
id matching alone would have been ambiguous.

Audit result: 50/50 points, 0 duplicates, 0 joint-limit violations
(margin 0.05 rad, same `within_joint_limits` call the generator used),
azimuth range **[-59.63°, +59.98°]** (inside the intended ±60°), workspace
extent x=[22.9, 298.3] y=[-256.6, 244.9] z=[31.5, 445.2] mm.

## Structural fixes so these failures can't recur silently

- IMU collectors now start and stop **inside** the sweep process, so the
  segment-012 non-overlap is structurally impossible rather than a timing
  discipline the operator has to maintain.
- `segment_id` is allocated by scanning the output directory (including
  `.partial` files) instead of being typed in.
- `tests/test_demo_replay.py` encodes each verdict above as an assertion, so
  a future parser or pipeline change that reintroduces one of these fails
  the suite.
- `docs/PRE_RUN_CHECKLIST.md` lists the pre-flight checks (tracker alive,
  mapping ready, IMUs streaming) that would have caught 012 in 30 seconds.

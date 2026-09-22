# WT901C-RS232 IMU Data Coverage

What the real WT901C-RS232 units in this project actually emit, what
`collectors/collect_imu_rs232.py`'s `decode_frame()` decodes, and the
evidence behind each call. Builds on `docs/07_WT901C_CODE_AUDIT.md`
(reimplementation-source audit) -- this doc is about the current package's
parser, checked against real captured data.

**Data used for the investigation below:**
`sessions/real_motion_demo/imu{0_base,1_link_root,2_link_mid,3_tip}_segment{006,007,012}.jsonl`
(12 files; segments 006/007 = known-good pilot capture, segment 012 = a
later real 50-point motion run), plus, for the packet-type coverage claims
in the last column, every `imu*_segment*.jsonl` file in that same session
directory (all 13 segments x 4 sensors). All read-only; no hardware was
touched to produce this document.

## Coverage matrix

| Packet type (hex) | Device emits it (from data) | Parser decodes it | Fields populated | Confidence / caveats |
|---|---|---|---|---|
| time (`0x50`) | **No** -- 0 occurrences in any `real_motion_demo` segment (all 13 segments x 4 sensors checked) | Yes (added this task) | `t_sensor` (device wall-clock string, top-level, never overwrites `t_pc_*_ns`) | Byte layout (`YY MM DD HH MN SS MSL MSH`) confirmed from WitMotion's own published serial protocol doc (see Sources). Year-as-`2000+YY` and the decode as a whole are **UNVERIFIED_HARDWARE_SETTING** -- never cross-checked against a real sample, because the device never emits this packet type in any captured session. |
| acceleration (`0x51`) | Yes -- 95845 records | Yes (pre-existing) | `accel_raw`, `accel_mps2`, `temperature_c` | `temperature_c` reads a plausible, time-varying 26-29C across sensors/segments -- used as the positive control for the angle-packet investigation below. Full-scale range (+-16g) still `UNVERIFIED_HARDWARE_SETTING` per the prior audit. |
| angular_velocity (`0x52`) | Yes -- 95848 records | Yes (pre-existing) | `gyro_raw`, `gyro_rads`, `temperature_c` | Same `temperature_c` values as the accel packet from the same die at the same instant (see below) -- internally consistent. Full-scale range (+-2000dps) still `UNVERIFIED_HARDWARE_SETTING`. |
| angle (`0x53`) | Yes -- 95856 records | Yes (pre-existing decode; word4 field **renamed** this task) | `euler_rad`, `angle_packet_word4_raw` (was wrongly `temperature_c`) | See "195.82C investigation" below. Angle full-scale range (+-180 deg) still `UNVERIFIED_HARDWARE_SETTING`; word4 confirmed NOT temperature (protocol doc + real data, high confidence). |
| magnetic_field (`0x54`) | Yes -- 95854 records | Yes (added this task) | `mag_raw`, `mag_packet_word4_raw`; `mag_uT` intentionally left `None` | `mag_raw` (mx,my,mz) varies plausibly (hundreds-to-thousands of counts, changes sample to sample) -- real magnetometer data. `mag_uT`: no protocol-documented raw-count -> uT scale factor exists (calibration-dependent); not invented. word4: see below, contradicts the general WitMotion doc. |
| port_status (`0x55`) | **No** -- 0 occurrences in any `real_motion_demo` segment | No (not attempted; only base fields) | `frame_type`/`frame_name`/`packet_type` only, via `raw_hex` | `0x55` is both the frame header AND this frame type's code -- a real, tested disambiguation case (`tests/test_wt901_parser.py:test_port_status_frame_header_and_type_byte_both_0x55_decode_correctly`), but the device apparently never emits this packet type in any session captured so far, so there's nothing to decode a payload meaning from. Not claiming false coverage. |
| pressure (`0x56`) | No -- 0 occurrences | No | base fields only | Out of scope for this task; never observed in data. |
| gps (`0x57`) | No -- 0 occurrences | No | base fields only | Out of scope for this task; never observed in data. |
| velocity (`0x58`) | No -- 0 occurrences | No | base fields only | Out of scope for this task; never observed in data. |
| quaternion (`0x59`) | **No** -- 0 occurrences in any `real_motion_demo` segment | Yes (pre-existing, tested) | `quaternion_xyzw` | Parser support exists and is unit-tested against synthetic frames (`tests/test_wt901_parser.py`). Real hardware output is **not enabled** in this project's current sessions -- the device's output mask apparently only streams accel/gyro/angle/magnetic. |

## The 195.82C investigation (angle packet word4)

**Symptom reported:** `decode_frame`'s `FRAME_ANGLE` (0x53) branch decoded
word 4 (bytes 8-9) the same way as accel/gyro's `temperature_c`
(`d / 100.0`), and in the real demo data this came out to a constant
195.82C -- physically implausible.

**Investigation (read-only, via `python`/`Grep` over the 12 files listed
above):**

1. Pulled every `packet_type == "angle"` record's `data.temperature_c` and
   recomputed the raw int16 word directly from `raw_hex` bytes 8-9. Result:
   **exactly `19582` (0x4C7E), with zero variation**, across:
   - all 4 independently-wired physical sensors (`imu0_base`, `imu1_link_root`,
     `imu2_link_mid`, `imu3_tip`),
   - all 3 segments checked (006, 007, 012),
   - all 95856 angle-packet records total (full-session scan, all 13
     segments).
2. Cross-checked the *same instants'* accel (`0x51`) and gyro (`0x52`)
   `temperature_c` from the same physical dies: these read a normal,
   slowly drifting **26.25-29.43C**, varying per sensor and per segment in
   a way consistent with real ambient/self-heating drift -- and matching
   each other exactly within each file (accel and gyro share the same
   temperature sensor on this chip, as expected).
3. A real temperature cannot be bit-identical across 4 physically separate,
   independently-wired sensors, let alone across different capture
   sessions on different days -- while a firmware/protocol version number
   would be. That alone was already strong evidence of a field-meaning
   issue, not 4 coincidentally-identical broken temperature sensors.
4. Confirmed via WitMotion's own published serial protocol documentation
   (`wit-motion.gitbook.io`, see Sources): the angle packet's byte layout is
   `RollL RollH PitchL PitchH YawL YawH VL VH SUM`, and bytes 8-9 (`VL`/`VH`)
   are explicitly documented as `Version=(VH<<8)|VL` -- a firmware/protocol
   version number, not temperature. `0x4C7E = 19582` is consistent with a
   version identifier.

**Conclusion:** angle packet word4 is a version number, not temperature.
`decode_frame`'s `FRAME_ANGLE` branch (`collectors/collect_imu_rs232.py`)
no longer sets `temperature_c` for this packet type; it sets
`angle_packet_word4_raw` (raw int16, unlabeled) instead.
`core/schemas.py:imu_record()` gained the corresponding field. Regression
test: `tests/test_wt901_parser.py:test_angle_frame_word4_is_raw_not_mislabeled_temperature`.
`temperature_c` for accel/gyro packets is untouched -- that word reads a
plausible, sensor-specific, time-varying value and was not part of this
investigation's finding.

## Magnetic packet word4: a related but distinct finding

WitMotion's published protocol documents the magnetic packet (`0x54`) byte
layout as `HxL HxH HyL HyH HzL HzH TL TH SUM` with `TL/TH` explicitly labeled
temperature (`Temperature=((TH<<8)|TL)/100`) -- the *same* convention as
accel/gyro, unlike the angle packet above.

However, empirically, in all **50074** magnetic-field records checked (the
same 12-file set), that word is **exactly 0 every single time** -- 0
nonzero occurrences -- while `mx`/`my`/`mz` in the same records vary
normally (real magnetometer counts) and accel/gyro `temperature_c` at the
same instants reads the normal ~26-29C range. A live temperature sensor
does not read exactly 0.00C for an entire multi-minute, multi-session
capture. So although the general WitMotion doc calls this word
"temperature," the real hardware in this project does not appear to
populate it that way (at minimum, not during these captures) -- possibly
firmware/output-mask specific to this WT901C-RS232 unit or configuration.

Given that direct conflict between documentation and real data,
`decode_frame`'s `FRAME_MAGNETIC` branch does **not** assert this word as
`temperature_c`; it's kept as `mag_packet_word4_raw`, honestly unlabeled,
pending real hardware confirmation. `mag_uT` is left `None` for a separate,
unrelated reason: WitMotion does not define a fixed raw-count -> uT scale
factor at all (calibration-dependent per unit); `mag_raw` is populated so
the counts aren't lost.

## Time packet (0x50) and port_status (0x55)

- **Time**: byte layout confirmed from WitMotion's own documentation
  (`YY MM DD HH MN SS MSL MSH`, not 4x int16 like every other frame type)
  and implemented (`t_sensor`, a new top-level field on `imu_record()` --
  purely supplementary, `t_pc_rx_ns` remains authoritative per
  `core/schemas.py`'s existing timestamp policy). The device never emits a
  time packet in any `real_motion_demo` segment, so this decode has only
  been checked against the protocol doc and a synthetic frame
  (`tests/test_wt901_parser.py:test_time_frame_decodes_calendar_bytes_not_int16_pairs`)
  -- **not** against a real sample. Flagged `UNVERIFIED_HARDWARE_SETTING`.
- **Port status**: `0x55` is simultaneously the frame header byte and this
  frame type's code. `extract_frames`/`decode_frame` were already
  structurally correct here (header search doesn't special-case byte 1),
  confirmed by a new synthetic test
  (`test_port_status_frame_header_and_type_byte_both_0x55_decode_correctly`)
  plus a companion test that a lone stray `0x55` byte (not a real frame)
  gets resynced past rather than mistaken for one
  (`test_garbage_0x55_byte_does_not_masquerade_as_port_status_frame`). The
  real device never emits a genuine port_status frame in any captured
  session -- 0 occurrences, so there is no real payload to decode a meaning
  from; not claimed as covered beyond frame-type labeling.

## Quaternion (0x59) hardware status

Parser support is complete and tested
(`tests/test_wt901_parser.py:test_quaternion_reordered_from_wire_wxyz_to_schema_xyzw`).
Real hardware status: **quaternion output is not enabled** on the physical
units used for `real_motion_demo` -- 0 occurrences of `packet_type ==
"quaternion"` across every segment of that session. The device's current
output-mask configuration streams acceleration, angular_velocity, angle,
and magnetic_field only.

## Duplicate / out-of-order / late packet handling (test coverage check)

Per the task-5 ask to check `tests/test_stale_sensor.py` against what
section 15 (`docs/DATA_SCHEMA.md`) wants:

- **Duplicate**: `mocks/fault_injector.py:with_duplicate_packets` was
  already tested at the mock/byte level (`tests/test_mocks.py`), but the
  collector's own `duplicate` field
  (`IMURS232Collector._handle_frame`'s exact-byte-repeat check) had no
  end-to-end regression test. Added:
  `tests/test_wt901_parser.py:DuplicatePacketTests.test_duplicate_wire_bytes_flag_records_as_duplicate_end_to_end`.
- **Out-of-order**: intentionally NOT given a detection test. Per
  `collect_imu_rs232.py`'s own `_handle_frame` docstring/comment, the
  WT901C wire format carries no in-frame sequence number, so `out_of_order`
  is unconditionally `False` by design -- there is no independent ordering
  signal this collector could use. The new duplicate test asserts this
  documented invariant instead of fabricating a scenario the collector
  structurally cannot detect.
- **Late**: `late_by_ms` is defined in the shared schema
  (`core/schemas.py:base_record`) but is never computed by
  `collect_imu_rs232.py` (always `None` for this stream) -- no deadline/
  target-rate concept exists in this collector to compute lateness against.
  Out of scope for this task; not implemented or tested here.

## Sources (protocol claims not otherwise verifiable from this repo)

- WitMotion WIT Standard Communication Protocol,
  `https://wit-motion.gitbook.io/witmotion-sdk/wit-standard-protocol/wit-standard-communication-protocol`
  -- time packet byte layout, angle packet `VL/VH` = Version, magnetic
  packet `TL/TH` = Temperature (per general doc; contradicted by real data
  for this specific field, see above).
- WT901C-RS232 manual (`cdn.robotshop.com` / `manuals.plus` mirrors) --
  same model as the hardware used in this project; direct PDF fetch was
  blocked (HTTP 403) during this investigation, so the general WIT
  Standard Protocol page above was used instead. Anything sourced only from
  general knowledge rather than a fetched page is called out inline as
  such.

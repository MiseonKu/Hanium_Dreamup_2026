# Data Schema Reference

Authoritative source: [`core/schemas.py`](../core/schemas.py). Every
collector, mock, and tool builds records through these functions instead of
hand-assembling dicts, so this file describes the actual, current record
shape -- not a spec document written independently of the code. If this file
and the code ever disagree, the code wins; re-generate this file from
`core/schemas.py` rather than editing around the discrepancy.

All records are one JSON object per line (JSONL), written by
[`core/jsonl_writer.py`](../core/jsonl_writer.py). See
[`RAW_DATA_POLICY.md`](RAW_DATA_POLICY.md) for the raw-first principles
these records follow, and
[`../firmware/esp32_so101_raw_bridge/serial_protocol.md`](../firmware/esp32_so101_raw_bridge/serial_protocol.md)
for the encoder stream's wire-to-schema field mapping specifically.

## Timestamp policy

- `t_pc_rx_ns` is the only authoritative sample time: `time.perf_counter_ns()`
  (`core.schemas.monotonic_ns()`), monotonic, never wall-clock.
- `utc_iso` (`core.schemas.utc_iso_now()`) is a human-readable wall-clock
  cross-check only -- ISO-8601 UTC with microsecond precision and a `Z`
  suffix. Never used for ordering or as the authoritative sample time.
- A missing/not-applicable value is always JSON `null`, never `0` or `""` --
  a present-but-zero value and an absent value must stay distinguishable
  downstream (enforced by `core/schemas.py`'s own self-check).

## Common fields (`base_record`, every stream)

Built by `base_record(...)`. Every other record type below starts from one
of these dicts and adds its own `data` (or equivalent) block on top.

| Field | Type | Meaning |
|---|---|---|
| `schema` | str | One of the `SCHEMA_*` constants below -- identifies record shape + version. |
| `session_id` | str | `session_YYYYMMDD_NNN`, matches the session directory name. |
| `segment_id` | int | Which output-file segment this record belongs to (see `core/jsonl_writer.py`; a crash rotates to a new segment rather than resuming). |
| `stream` | str | One of the `STREAM_*` constants below -- which logical stream this record is part of. |
| `sensor_id` | str \| null | Logical sensor id (e.g. `imu3_tip`, `esp32_encoder`) where applicable. |
| `source_seq` | int \| null | Sequence number assigned by the sensor/firmware itself, if the wire protocol has one. `null` when it doesn't (e.g. the WT901C wire format carries no in-frame sequence -- see `collectors/collect_imu_rs232.py`). |
| `host_seq` | int | Sequence number assigned by the PC-side collector, strictly increasing per stream instance. Always present. |
| `trial_id` | int \| null | Which experiment trial this record belongs to, when applicable. |
| `target_id` | str \| null | Which target-plan target this record belongs to, when applicable. |
| `planned_order_index` | int \| null | Position in the planned target-dispatch order, when applicable. |
| `workspace_mode` | str \| null | Which azimuth-sector target-generation mode was active (`pilot`/`main`/`front_120`), when applicable. Not to be confused with the session *execution* mode (`mock`/`passive`/`soak`/`experiment`/`inventory`), which `experiment/session_controller.py` writes to `manifest.json["execution_mode"]` -- a deliberately different field name, see `DECISIONS.md`. |
| `experiment_state` | str \| null | Current `experiment.experiment_state_machine.ExperimentState` value at record time, when applicable. |
| `t_source` | int \| null | Sensor/firmware-native timestamp, in whatever unit that source uses (e.g. `t_esp_us` for the encoder). Never used for ordering. |
| `t_pc_first_byte_ns` | int \| null | PC-monotonic time the first byte of this record's underlying frame/line began arriving, when a collector tracks it (see `collectors/collect_imu_rs232.py`'s bracketing-timestamp note). |
| `t_pc_frame_end_ns` | int \| null | PC-monotonic time frame extraction/validation finished for the batch this record came from. |
| `t_pc_rx_ns` | int | **Authoritative sample time.** PC-monotonic receive time (`time.perf_counter_ns()`). Always present. |
| `t_sync_ns` | int \| null | Reserved for a future clock-sync-corrected timestamp; `core/clock_sync.py` explicitly never writes this -- see `RAW_DATA_POLICY.md`. |
| `utc_iso` | str | Wall-clock cross-check only, ISO-8601 UTC, e.g. `2026-08-04T11:08:08.365612Z`. Always present, always fresh at record-build time. |
| `valid` | bool | Whether this record represents a successfully-parsed, trustworthy sample. |
| `parse_status` | str | e.g. `"ok"`, `"parse_error"`, `"tracking_lost"`, `"device_disconnected"`. |
| `checksum_ok` | bool \| null | Whether a wire-level checksum passed, when the protocol has one (IMU only; `null` for streams without a checksum). |
| `duplicate` | bool | Whether this record's underlying frame was byte-identical to the immediately preceding one. |
| `out_of_order` | bool | Whether this record was detected arriving out of order. Always `False` for IMU records today -- the WT901C wire format has no independent ordering signal (see `collectors/collect_imu_rs232.py`'s comment). |
| `late_by_ms` | float \| null | Reserved for a future lateness signal; not populated by any current collector. |
| `quality_flags` | list[str] | Never `null` -- defaults to `[]`. Known values in use today: `"resynced"` and `"stale_suspect"` (`collect_imu_rs232.py:470,493`); `"mapping_not_ready"` and `"fault_latched"` (`collect_esp32_state_poll.py:241,243`); `"command_not_acked"` (same file, `:345`); `"joint_limits_not_ready"` (`collect_esp32_encoder.py:229`, STREAM path only — not produced on this rig). |
| `raw_hex` | str \| null | Raw frame bytes, hex-encoded, for binary protocols (IMU). |
| `raw_line` | str \| null | Raw text line, for line-oriented protocols (encoder). |

### Schema / stream constants

| `SCHEMA_*` constant | value | `STREAM_*` constant | value |
|---|---|---|---|
| `SCHEMA_IMU` | `alphaf.imu.raw.v1` | `STREAM_IMU` | `imu` |
| `SCHEMA_ENCODER` | `alphaf.encoder.raw.v1` | `STREAM_ENCODER` | `encoder` |
| `SCHEMA_COMMAND` | `alphaf.command.raw.v1` | `STREAM_COMMAND` | `commands` |
| `SCHEMA_VIVE` | `alphaf.vive.raw.v1` | `STREAM_VIVE` | `vive_tracker` |
| `SCHEMA_EVENT` | `alphaf.event.raw.v1` | `STREAM_EVENT` | `events` |
| `SCHEMA_HEALTH` | `alphaf.health.raw.v1` | `STREAM_HEALTH` | `health` |
| `SCHEMA_STABILITY` | `alphaf.stability_metrics.raw.v1` | `STREAM_STABILITY` | `stability_metrics` |

## IMU (`imu_record`)

`schema=SCHEMA_IMU`, `stream=STREAM_IMU`. One record per decoded WT901C
frame (accel, gyro, angle, or quaternion) -- **not** one record per full
4-frame sample. See `SAMPLING_AND_ALIASING.md` for what this means for
measured-rate math. Written by `collectors/collect_imu_rs232.py`.

Top-level additions beyond `base_record`:

| Field | Type | Meaning |
|---|---|---|
| `packet_type` | str \| null | e.g. `"acceleration"`, `"angular_velocity"`, `"angle"`, `"magnetic_field"`, `"quaternion"`. Real hardware emits only the first four; see `docs/IMU_DATA_COVERAGE.md`. |
| `frame_length` | int \| null | Wire frame length in bytes (always 11 for WT901C, confirmed across 95k+ real frames). |
| `t_sensor` | str \| null | Device-reported wall-clock string, decoded from a WT901C time (`0x50`) packet only. Supplementary; never overwrites `t_pc_rx_ns`. `null` for every other packet type -- and in practice always `null`, because the device does not emit `0x50` in any captured session. |
| `connection_epoch` | int | Extra top-level key (not part of `base_record`/`imu_record`'s formal shape, stamped separately) -- increments on every reconnect, so a timestamp gap across a dropped/reopened connection is identifiable. |

`data` sub-object:

| Field | Type | Meaning |
|---|---|---|
| `accel_raw` | [int, int, int] \| null | Raw int16 accelerometer counts, present only on an acceleration frame. |
| `gyro_raw` | [int, int, int] \| null | Raw int16 gyro counts, present only on an angular_velocity frame. |
| `accel_mps2` | [float, float, float] \| null | Acceleration, m/s^2 (converted via a configurable full-scale-range constant -- see `UNVERIFIED_HARDWARE_SETTING` in `docs/07_WT901C_CODE_AUDIT.md`). |
| `gyro_rads` | [float, float, float] \| null | Angular velocity, rad/s. |
| `mag_raw` | [int, int, int] \| null | Raw int16 magnetometer counts, present only on a magnetic_field (`0x54`) frame. |
| `mag_uT` | [float, float, float] \| null | Magnetometer, microtesla. Deliberately always `null`: no protocol-documented raw-count → µT scale factor exists (it is calibration-dependent), so none was invented. |
| `mag_packet_word4_raw` | int \| null | The magnetic_field packet's undecoded 4th int16 word, kept raw rather than guessed at. |
| `euler_rad` | [float, float, float] \| null | Roll/pitch/yaw, radians, present only on an angle frame. |
| `quaternion_xyzw` | [float, float, float, float] \| null | Present only on a quaternion frame. Reordered from the wire's `(w,x,y,z)` to schema convention `(x,y,z,w)`. Parser tested against synthetic frames only — the real device never emits `0x59`. |
| `temperature_c` | float \| null | Sensor die temperature, Celsius. **Accel and gyro frames only.** Reads a plausible 26-29°C on real hardware. |
| `angle_packet_word4_raw` | int \| null | The angle (`0x53`) packet's 4th int16 word. **Was previously decoded as `temperature_c`, which was wrong** — it read a constant 195.82°C, bit-identical (19582 / 0x4C7E) across 4 physically separate sensors and 95,856 records, i.e. a version number, not a temperature. Renamed and kept raw. Full investigation: `docs/IMU_DATA_COVERAGE.md`. |

`sensor_config` sub-object:

| Field | Type | Meaning |
|---|---|---|
| `serial_port` | str \| null | e.g. `"COM5"`; `null` for a mock/test byte source. |
| `configured_baud` | int \| null | Serial baud rate the collector opened the port at. |
| `configured_output_rate_hz` | float \| null | Nominal full-sample rate (100 or 200 Hz for WT901C). |
| `measured_output_rate_hz` | float \| null | Reserved; not populated by the collector itself -- compute with `tools/audit_sampling.py` instead. |

## Encoder (`encoder_record`)

`schema=SCHEMA_ENCODER`, `stream=STREAM_ENCODER`, `sensor_id="esp32_encoder"`.

**Corrected 2026-08-05.** This section previously described only the
`STREAM`-protocol collector and said every list field held **5** values in
the order `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, gripper`.
Both were wrong for this rig:

- **Three axes, not five**, ordered
  `[shoulder_pan, shoulder_lift, wrist_flex]`. `elbow_flex`/`gripper` are
  not actuated here — `wrist_roll`/`gripper` positions carry a Vive Tracker
  mount. Firmware agrees: `robot_config.h:33` `kJointCount = 3`. See
  `docs/THREE_AXIS_EXPERIMENT_SCOPE.md`. Confirmed in real data: every
  list in `sessions/real_motion_demo/encoder_segment012.jsonl` has length 3.
- **Two collectors, two protocols.** The board on this rig runs
  `03_SO-101_control/main` and is read by
  `collectors/collect_esp32_state_poll.py` (host sends `STATE`, firmware
  replies with one detail line per axis).
  `collectors/collect_esp32_encoder.py` speaks the pushed `STREAM ...`
  protocol of `02_SO-101_rtos_motor_driver`, which is **not flashed here**.
  Both are kept; only the first produces real data.
  See `docs/ENCODER_COMMAND_COVERAGE.md`.

Top-level addition beyond `base_record`: `t_esp_us` (int \| null) -- ESP32
boot-relative timestamp in microseconds. Populated from the `STREAM` line's
`t_ms * 1000`; **always `null` on the `STATE` path** (the detail line has no
board-relative timestamp), so `t_source` is `null` in real records too.

`data` sub-object (per-collector notes where they differ):

| Field | Type | Meaning |
|---|---|---|
| `q_cmd_raw` | list[int] \| null | Reserved; not populated by either collector. |
| `q_cmd_rad` | list[float] \| null | Last commanded joint angle per axis, radians, 3 values. `null` on the `STREAM` path and until a command has been sent on the `STATE` path. |
| `encoder_raw_tick` | list[int] \| null | Raw ST3215 encoder ticks, 3 values, `[pan, lift, wrist]`. **This is the trustworthy angle source** — 4096 counts / 360°, so 1 physical degree = 11.378 ticks. |
| `q_encoder_rad` | list[float \| null] | The firmware's `logical_deg` converted to radians. **NOT a physical angle** — `logical_deg` is an arbitrary label interval mapped linearly onto each axis's calibrated-active raw span, so its scale is distorted per axis (see `config/joint_limits.reference.yaml` and `docs/FK_CONVENTION_CORRECTION.md`). Use `encoder_raw_tick` for anything geometric. Per-joint `null` when the axis reported `logical_deg=NA` (`STATE`, mapping not loaded) or the `-999.00` sentinel (`STREAM`). |
| `estimated_velocity_rad_s` | list[float \| null] \| null | Finite difference between consecutive polls on the `STATE` path (`null` for the first record); not populated on the `STREAM` path. Inherits `q_encoder_rad`'s label-scale distortion. |
| `motor_status` | list[str] \| null | `"enabled"`/`"disabled"` per joint (`STATE`: from each axis's `torque` field). |
| `temperature_c` | list[int \| null] \| null | **MISLEADING FIELD NAME — not Celsius on the `STATE` path.** `collect_esp32_state_poll.py:290` puts each axis's `raw_temp` *register count* here, unscaled (no ST3215 datasheet in the repo to convert it). The name comes from `core.schemas.encoder_record`'s generic signature. Not populated at all on the `STREAM` path. Note also that the onboard servo temperature sensor is **known broken** on this rig (`docs/STATUS.md`) — do not use this field as a thermal signal. |
| `current_or_load` | list[int \| null] \| null | Same shape of problem: each axis's `raw_load` register count (`:291`), not amps or Nm. |
| `fault` | list[str] \| null | `"fault"`/`"ok"` per joint. |
| `esp32_state` | str \| null | `"fault"` if a fault is latched, else `"ok"`, on the `STATE` path (`:293`). The `"calibrating"`/`"armed"`/`"idle"` values belong to the `STREAM` path. |
| `command_ack` | bool \| null | Reserved here; the `STATE` collector records acks on the separate `command` record instead. |
| `watchdog_state` | str \| null | Reserved; not populated. |
| `raw_telemetry` | list[dict \| null] \| null | **`STATE` path only** (added by `collect_esp32_state_poll.py:274`, not part of `core.schemas.encoder_record`'s formal shape). One dict per axis carrying every remaining firmware field verbatim: `raw_speed`, `raw_load`, `raw_voltage`, `raw_temp`, `raw_current`, `mapping_ready`, `raw_cal_min/max`, `raw_active_min/max`, `speed`, `acc`, `range_percent`, `direction`, `last_update_ms`, `read_failures`, `overload_samples`. Servo counts, unscaled. |
| `fault_detail` | dict \| null | **`STATE` path only** (`:275`) -- `fault_latched`/`reason`/`latched_ms` from the firmware's separate fault line. |

Note that the full raw firmware line is preserved on every record in
`raw_line`, which is what allowed all 25 firmware fields to be recovered
from sessions captured before this parser existed
(`docs/ENCODER_COMMAND_COVERAGE.md`).

## Vive Tracker (`vive_record`)

`schema=SCHEMA_VIVE`, `stream=STREAM_VIVE`. One record per poll, including
tracking-lost polls -- **never forward-filled and never skipped** (see
`RAW_DATA_POLICY.md`). Written by `collectors/collect_vive_tracker.py`.

Top-level addition beyond `base_record`: `frame_id` (str, always
`"vive_world"`).

`data` sub-object:

| Field | Type | Meaning |
|---|---|---|
| `tracker_serial` | str \| null | Device serial, also used as `sensor_id`. |
| `device_index` | int \| null | OpenVR-style device index. |
| `tracking_result` | str \| null | e.g. `"TrackingResult_Running_OK"`, `"TrackingResult_Running_OutOfRange"`, `"TrackingResult_Uninitialized"`. |
| `tracking_valid` | bool \| null | Whether the pose fields below are meaningful this sample. |
| `device_connected` | bool \| null | Whether the tracker is connected at all. |
| `tracking_origin` | str \| null | e.g. `"TrackingUniverseStanding"`. |
| `pose_matrix_3x4` | list[list[float]] (3x4) \| null | Raw rotation+translation matrix. `null` when tracking is invalid. |
| `position_m` | [float, float, float] \| null | Position, meters. `null` when tracking is invalid -- never the last-good position. |
| `quaternion_xyzw` | [float, float, float, float] \| null | Orientation. `null` when tracking is invalid. |
| `source_timestamp` | float \| null | Source-native timestamp (mock: simulated seconds since start). Never used for ordering. |

## Event (`event_record`)

`schema=SCHEMA_EVENT`, `stream=STREAM_EVENT`. Written by
`collectors/collect_events.py:EventCollector.log_event()`, either called
directly or via `EventBus` subscription (`wire_event_type`).

Top-level additions beyond `base_record`:

| Field | Type | Meaning |
|---|---|---|
| `event_type` | str | e.g. `"session_started"`, `"sensor_disconnected"`, `"health_state_changed"`, `"collector_unavailable"`, `"shutdown_initiated"`. Free-form, not an enum -- see call sites in `experiment/session_controller.py` for the current set in use. |
| `message` | str \| null | Human-readable summary. |
| `details` | dict | Never `null` -- defaults to `{}`. Free-form structured payload. |

## Health (`health_record`)

`schema=SCHEMA_HEALTH`, `stream=STREAM_HEALTH`. One record per
`HealthSupervisor.interval_s` (default 1.0s), persisted by
`experiment/session_controller.py`'s own writer loop (see
`core/health_supervisor.py`'s docstring: it computes/exposes signals but
does not persist them itself).

`data` sub-object (every `dict[str, X]` field is keyed by stream/collector
logical id, e.g. `"imu3_tip"`, `"encoder"`):

| Field | Type | Meaning |
|---|---|---|
| `queue_depth` | dict[str, int] | Current queue occupancy per stream. |
| `writer_lag_ms` | dict[str, float] | Reserved; not populated by the current supervisor. |
| `cpu_percent` | float \| null | Always `null` -- no `psutil` dependency in this repo (see `core/health_supervisor.py`'s docstring). |
| `memory_percent` | float \| null | Always `null`, same reason. |
| `disk_free_bytes` | int \| null | `shutil.disk_usage(session_dir).free`, or `null` on an `OSError`. |
| `packet_rate_hz` | dict[str, float] | Reserved; not populated (use `tools/audit_sampling.py` per-stream instead). |
| `packet_loss_ratio` | dict[str, float] | Reserved; same. |
| `checksum_failures` | dict[str, int] | Reserved; not populated (checksum failures show up as `quality_flags: ["resynced"]` on individual IMU records instead). |
| `reconnect_counts` | dict[str, int] | Reserved; not populated (see each collector's own `.stats()`/`reconnect_count` instead). |
| `sampling_jitter_ms` | dict[str, float] | Reserved; same as `packet_rate_hz`. |
| `late_count` | dict[str, int] | Reserved; not populated. |
| `out_of_order_count` | dict[str, int] | Reserved; not populated. |
| `duplicate_count` | dict[str, int] | Reserved; not populated (per-record `duplicate` flag exists on IMU records instead). |
| `queue_state` | dict[str, str] | One of `NORMAL`/`WARN`/`DEGRADED`/`CRITICAL`/`OVERFLOW` per stream -- the field actually driving the escalation policy (`should_suppress_live_output()` etc., see `docs/FAILURE_HANDLING.md`). |

## Stability metrics (`stability_record`)

`schema=SCHEMA_STABILITY`, `stream=STREAM_STABILITY`. Built by
`experiment.adaptive_settling_detector.stability_record_from_evaluation()`,
one per `AdaptiveSettlingDetector.evaluate()` call during
`ADAPTIVE_SETTLING_MONITOR`/`STABLE_CANDIDATE`/`TIP_IMU_STABLE_CONFIRMED`/
`MEASUREMENT_WINDOW` states. Only produced in `experiment` mode (see
`experiment/experiment_state_machine.py`) -- not written by `mock`/`soak`
sessions.

`data` sub-object:

| Field | Type | Meaning |
|---|---|---|
| `tip_gyro_rms` | float \| null | RMS of `||gyro||_2` over the rolling window. `null` until the window has enough samples (`ready=False`). |
| `dynamic_accel_rms` | float \| null | RMS of window-mean-subtracted accel magnitude (crude high-pass, removes gravity). |
| `orientation_spread_rad` | float \| null | L2 norm of per-axis Euler std-dev over the window. |
| `jerk_rms` | float \| null | RMS of finite-difference accel derivative magnitude. |
| `drift_slope` | float \| null | OLS slope of yaw vs. time over the window, rad/s. |
| `stable_candidate_duration_s` | float | How long the debounce timer has been continuously passing. |
| `stability_margin` | float | `min` over per-metric `(threshold - value) / threshold`; positive = comfortably inside every threshold, `<=0` = at least one metric/guard failing. |
| `settling_state` | str \| null | Current `ExperimentState` value at evaluation time. |
| `guard_flags` | dict[str, bool] | Never `null` -- defaults to `{}`. Keys: `encoder_settled`, `base_imu_stable`, `link_imu_no_vibration`, `tip_freshness_ok`, `clock_sync_valid`, `queue_ok`, `motor_fault_ok`, `not_frozen`. |

All numeric thresholds behind this record are reasonable-but-**UNVALIDATED**
defaults (`experiment.adaptive_settling_detector.SettlingConfig`) -- see
`docs/HARDWARE_TEST_PLAN.md` and `docs/TASKS.md`.

## Command (`command_record`)

`schema=SCHEMA_COMMAND`, `stream=STREAM_COMMAND`.

**Corrected 2026-08-05.** This section used to say "no collector in this
package currently writes this stream". That is now false:
`collectors/collect_esp32_state_poll.py:335-360` builds a `command_record`
with `command_type="legacy_axis_move"` for every axis command it sends, and
`quality_flags: ["command_not_acked"]` when the firmware's queue
acknowledgement doesn't come back.

**Two caveats before relying on it.** That path sends the legacy
`D<axis><deg>` wire command, which (a) requires `INIT` — commands are
silently dropped without it, `docs/FIRMWARE_MOTION_HAZARDS.md` #3 — and (b)
takes the distorted `logical_deg` label scale rather than raw ticks. The
current collection flow deliberately uses `MOVE_RAW` instead and never
calls `INIT`. Treat this record type as correct-for-the-protocol-it-targets,
not as the shape a `MOVE_RAW`-based writer should copy.

No session mode writes a `commands` stream file today; the record is
produced by direct calls to the collector.

Top-level addition beyond `base_record`: `command_type` (str).

`data` sub-object:

| Field | Type | Meaning |
|---|---|---|
| `q_cmd_rad` | list[float] \| null | Commanded joint angles, radians. |
| `payload` | dict | Never `null` -- defaults to `{}`. Free-form. |

## Derived dataset (`derived/parquet/`, added 2026-08-10)

Everything above describes the raw JSONL contract, which is the source of
truth. The three layers below are **post-acquisition, rebuildable, and
git-ignored** — they exist because 25 M raw records across four packet types
are not a shape analysis can use directly. If a derived file and the raw
JSONL disagree, the JSONL wins; rebuild rather than patch.

| Built by | Output | One row is |
|---|---|---|
| `tools/prepare_dataset.py` | `<session>/{imu,encoder,vive_tracker,events,health}.parquet` | one raw record, unchanged, columnar |
| `tools/build_trials.py` | `<session>/trials.parquet` (83 cols) | one settled trial |
| `tools/build_timeseries.py` | `<session>/timeseries.parquet` (68 cols) | one 50 Hz grid point inside one trial |

### `trials.parquet` — one row per settled measurement

A *trial* is a contiguous run of records whose `experiment_state` is
`MEASUREMENT_WINDOW`, cut on the Vive stream (the 60 Hz reference); encoder
and IMU samples join a trial by their `t_pc_rx_ns` falling inside its span.

Note the record-level `trial_id` field is **not** a trial index — it is the
set index (0..11), shared by every target in a set. Use `block` (the cut
index) or `target_id` + `repeat_index`.

Column groups: identity (`segment_id`, `block`, `set_id`, `target_id`,
`planned_order_index`, `repeat_index`), timing (`t_start_ns`, `t_end_ns`,
`utc_start`, `duration_s`), Vive (`vive_n`, `vive_tracking_valid_ratio`,
mean `vive_pos_{x,y,z}_m` with per-axis `_std_m`, `vive_quat_*`), plan
(`planned_p_{x,y,z}_mm`), encoder per joint j0/j1/j2 (`encoder_tick_*` +
`_std`, `q_encoder_rad_*`, `logical_deg_*`), and per IMU sensor
(`euler_*_rad`, `accel_*_mps2`, `mag_*_raw`, `gyro_rms_rad_s`, `_n`).

`keep` is the usability filter, and it is the column to filter on:
`vive_n >= 30` **and** `vive_tracking_valid_ratio == 1.0` **and** a non-null
Vive position **and** an attached encoder sample. 9,951 of 10,055 trials pass
on `real_motion_demo`.

Joint order is always `shoulder_pan, shoulder_lift, wrist_flex` (j0/j1/j2);
it is also stamped into the Parquet schema metadata.

### `timeseries.parquet` — the transient, not just the settled pose

Same trials, but the window starts at the beginning of the `SETTLING` run, so
the move, the ringing and the settled hold are all present, resampled onto a
uniform grid (`t_rel_s`, default 50 Hz) with every stream on the same clock.
`phase` distinguishes `SETTLING` from `MEASUREMENT_WINDOW` rows.

Why resample rather than pivot on packet type: the WT901C emits acceleration,
angular velocity, angle and quaternion as four *separate* frames, so a "40 Hz
IMU sample" does not exist as a row anywhere in the raw data. Snapping each
packet type onto a shared grid aligns them with each other, with the 60 Hz
Vive and with the ~2 Hz encoder in one step. Nothing is filled beyond
`tolerance` of a grid point — a gap stays a gap, per
[`RAW_DATA_POLICY.md`](RAW_DATA_POLICY.md).

### Clock caveat for both

`t_pc_rx_ns` is `perf_counter_ns()`, monotonic **per segment** — each segment
is a fresh process, so the counter restarts. Never compare or interval-match
`t_pc_rx_ns` across segments; both builders group by `segment_id` first.
`utc_time` / `utc_start` is the only cross-segment ordering key.

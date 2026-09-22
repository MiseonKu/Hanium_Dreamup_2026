# Status

Updated 2026-08-12. `PASS` means "observed working, against the described
scope" — the scope is stated per row, because it now varies: some rows are
verified against `mocks/*.py` only, some against **real hardware data**
already collected in `sessions/real_motion_demo/`, and some against real
hardware in the loop.

**Important correction to earlier versions of this file:** real hardware
*has* been connected and real data *has* been collected (4x WT901C IMU on
COM16–19 @ 9600 baud, ESP32 running `03_SO-101_control/main` on COM20,
Vive Tracker via SteamVR). Rows below that previously said "never
connected" were stale.

**Acquisition is complete as of 2026-08-10.** `sessions/real_motion_demo/`
holds segments 000–041; the final fused sweeps are 036–041. The dataset was
then prepared, collapsed to trials, and resampled — numbers in
"Collected dataset" below. Everything the 2026-08-05 version of this file
listed as "not achieved yet" about a fused run is now done; what remains
open is listed in `docs/TASKS.md` and is no longer blocking data collection.

## Test / verification status

| Area | Status | Scope and notes |
|---|---|---|
| Unit/integration test suite (`tests/`) | **PASS_WITH_NOTE** | 150 tests. Last full re-run 2026-08-12: **140 passed, 10 failed** — every failure was `tests/test_demo_replay.py` opening `sessions/real_motion_demo/*.jsonl` after `tools/prepare_dataset.py` gzipped those files in place. The test's opener is now `.gz`-aware (2026-08-12); re-run to confirm 150 rather than trusting this row. Nothing in the collectors or core was implicated. |
| Every module `__main__` self-check | **PASS** | All 15 run standalone (`python core/schemas.py` etc.). Several previously crashed on import with `ModuleNotFoundError: core` — the `sys.path` line their siblings had was missing, so the checks could never actually be run. |
| Sibling package `02_FK_IK` | **PASS** | 25/25, from that package's root. |
| Mock-mode session (`tools/run_mock_session.py`) | **PASS** | All 8 collector slots report `"started"`; manifest ends `session_complete: true`; `session_summary.json` now written alongside it. |
| Soak-mode session (`tools/run_soak_test.py`) | **PASS** | Same path as mock mode, short duration only; the full 30+ min default was not run end-to-end. |
| WT901C frame parser | **PASS** | Synthetic frames + `tests/test_wt901_parser.py`, **and** replayed against real captures. Now decodes magnetic (`0x54`) and time (`0x50`). The angle packet's (`0x53`) word4 was proven — from 95,856 real records across 4 physically separate sensors plus WitMotion's protocol doc — to be a version number, not temperature; renamed `angle_packet_word4_raw`. See `docs/IMU_DATA_COVERAGE.md`. |
| ESP32 `STATE` protocol parser (`collect_esp32_state_poll.py`) | **PASS (real data)** | 21 unit tests, **plus** replayed against every real encoder record: 310/310 (seg006), 542/542 (seg007), 759/759 (seg012) fully parsed, all 3 axes, all 25 firmware fields. See `docs/ENCODER_COMMAND_COVERAGE.md`. |
| ESP32 `STREAM` protocol parser (`collect_esp32_encoder.py`) | **PASS (mock only)** | Against `mocks/mock_esp32.py`. Targets `02_SO-101_rtos_motor_driver`, which is **not** the firmware on this rig — kept for the protocol it was written against. |
| Vive no-forward-fill policy | **PASS (real data)** | `tests/test_vive_no_forward_fill.py`, and confirmed in real segment 012, where a dead tracker produced 759 correctly-invalid records rather than 759 forward-filled fakes. |
| Real OpenVR source (`OpenVRTrackerSource`) | **PASS (stub only)** | `tests/test_openvr_tracker_source.py` against a fake stub `openvr` module. The equivalent call pattern *did* run against real SteamVR in the archived sweep script, but this class specifically has not. |
| Disconnect/reconnect handling | **PASS (mock only)** | `mocks/fault_injector.py:with_disconnect`, IMU + encoder paths. |
| Queue overflow / backpressure | **PASS** | `tests/test_queue_overflow.py`; all 5 `BoundedQueue` states. |
| Interrupt-safe shutdown | **PASS** | `tests/test_interrupt_shutdown.py` + session-controller smoke. A real start-barrier race (shutdown joining a not-yet-started thread → leaked writer handle) was found and fixed this round. |
| Partial-file (`.partial`) recovery | **PASS** | `tests/test_jsonl_recovery.py` + manual `tools/recover_partial_jsonl.py`. |
| Frozen/stale-sensor guard | **PASS (synthetic)** | Both layers; thresholds still unvalidated. |
| Adaptive settling detector | **PASS_WITH_WARNING** | Logic correct against synthetic streams; every numeric threshold self-documented `UNVALIDATED` against real sensor noise. Section-10 event strings now emitted via `experiment_state_machine.py`. |
| Demo-data replay audit | **PASS (real data)** | `tests/test_demo_replay.py` — asserts good segments good *and* bad segments bad. See `docs/DEMO_DATA_REPLAY_AUDIT.md`. |
| Sampling / UART budget audit | **PASS (real data)** | Measured ~10.08 Hz per packet type, ~40.3 Hz combined, 443 B/s vs 960 B/s capacity = **46% UART utilization** — the sensor's own rate setting is the bottleneck, not the link. `docs/SAMPLING_AND_UART_BUDGET.md`. |
| `front_120` / v3 target-plan audit | **PASS (real data)** | 50/50 points, 0 duplicates, 0 joint-limit violations, azimuth [-59.63°, +59.98°]. `docs/TARGET_PLAN_AUDIT_V3.md`. |
| Config examples (`config/*.yaml`) | **PASS** | All validated as parseable YAML. |
| Dataset pipeline (`tools/prepare_dataset.py`) | **PASS (real data)** | Ran end-to-end on all 6 session directories 2026-08-10: partial-file recovery, Parquet working copy, then gzip of 363 raw files, 49.0 GB → 3.3 GB (6.7%) in 162 s. Log: `prepare_dataset.log`. |
| Trial collapse (`tools/build_trials.py`) | **PASS (real data)** | 10,055 trial windows cut from `experiment_state == MEASUREMENT_WINDOW` runs on the Vive stream, 9,951 pass the `keep` filter (≥30 Vive samples, `tracking_valid` ratio == 1.0, encoder attached). |
| Uniform resample (`tools/build_timeseries.py`) | **PASS (real data)** | 6,456,460 rows @ 50 Hz across segments 023–041, 537.7 MB. Log: `build_timeseries.log`. Fills nothing outside `tolerance` of a grid point. |
| Session-reading tools vs. gzipped archives | **KNOWN GAP** | `tools/inspect_jsonl.py`, `tools/validate_raw_session.py`, `tools/audit_sampling.py` open plain `.jsonl` only. After `prepare_dataset.py`'s gzip phase, point them at a decompressed copy — or extend them the way `tests/test_demo_replay.py` was. `docs/TASKS.md` #14. |
| Config loader | **PARTIAL** | `SessionController` now reads `sensor_map` YAML for real-mode port resolution. Other `config/*.yaml` files are still templates nothing reads. |

## Real-hardware status

| Area | Status | Notes |
|---|---|---|
| Real WT901C IMU hardware | **VERIFIED** | 4 units captured across segments 000–041 (24.9 M records). 100% checksum-OK, zero parse errors, every frame 11 bytes. Scale constants (±16g/±2000dps/±180°) still `UNVERIFIED_HARDWARE_SETTING` — never read back from the device's range registers. |
| Real ESP32 (`03_SO-101_control/main`) | **VERIFIED** | 3-axis firmware, real calibration in EEPROM, real motion across 50-point sweeps. |
| Real Vive Tracker / OpenVR | **VERIFIED** | 006/007 100% valid; 012 was a total loss (tracker asleep, 0% valid). Fixed procedurally — `collect_3d_sweep_50_v2.py` now aborts before touching a motor if the tracker is absent, and `probe_vive.py` runs first every time. Segments 036–041 hold ~99.6% `tracking_valid`. |
| Sequential axis dispatch (thermal fix) | **NOT_TESTED on hardware** | Logic is correct against the firmware's documented batching behaviour, but **peak current was not measured**. `docs/ST3215_THERMAL_AND_SEQUENCING.md`. |
| Fully-fused IMU + encoder + Vive run | **ACHIEVED** | Segments 036–041 (2026-08-05 → 2026-08-10): all three streams valid and time-aligned in the same segments, ~1,600–1,700 trial windows each, ~99.6% Vive valid. Confirmed by `build_timeseries.py` producing 6.46 M fused 50 Hz rows and by `trials.parquet`'s 9,951 `keep` rows carrying non-null Vive, encoder and 4-IMU columns together. |
| `experiment` session mode (hardware motion via SessionController) | **NOT_TESTED** | Real motion has only ever been driven by `experiment_scripts/collect_3d_sweep_50_v2.py`, not through `SessionController`'s gated `experiment` mode. |
| ST3215 raw register → physical units | **BLOCKED** | No datasheet in repo. `raw_voltage/temp/current/load` stored as counts, unscaled. |
| Onboard servo temperature sensor | **KNOWN BROKEN** | Confirmed by the project owner. All software temperature-trip logic deliberately removed. External sensor + hardware power cutoff is the real safety net. |

## Collected dataset (2026-08-10)

All of it is git-ignored and rebuildable from `sessions/` — see
`docs/RAW_DATA_POLICY.md`. `derived/` is never a source of truth.

| Layer | Content |
|---|---|
| `sessions/real_motion_demo/` | Segments 000–041 from `experiment_scripts/collect_3d_sweep_50_v2.py`. Raw JSONL, gzipped in place: 363 files, 49.0 GB → 3.3 GB. Interrupted segments survive as `.jsonl.partial.gz` plus a `.recovered.jsonl.gz` and a recovery report — never silently completed. |
| `sessions/real_hw_check/`, `sessions/session_2026080*_00*/` | IMU bench check and mock sessions, kept for provenance. |
| `derived/parquet/<session>/{imu,encoder,vive_tracker,events,health}.parquet` | Per-sample working copy. `real_motion_demo`: IMU 24,880,024 rows / Vive 9,372,862 / encoder 339,965. |
| `derived/parquet/real_motion_demo/trials.parquet` | 10,055 settled trial windows, 9,951 with `keep == True`, 83 columns. 50 unique targets × ~200 repeats; median window 2.89 s. |
| `derived/parquet/real_motion_demo/timeseries.parquet` | 6,456,460 rows, uniform 50 Hz per trial, spanning `SETTLING` through `MEASUREMENT_WINDOW` so the transient (move → ringing → settle) is preserved. |

Per-segment trial counts, including the dropped ones, are in
`build_timeseries.log` and reproducible from `trials.parquet`. Segments 023–035
are short pilots (2–30 trials each); the bulk is 036–041.

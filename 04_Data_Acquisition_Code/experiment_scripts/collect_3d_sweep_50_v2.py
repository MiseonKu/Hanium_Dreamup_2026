"""Target sweep over the left-of-straight workspace: commands the 3 active
axes and records encoder + Vive + all 4 IMUs into one time-aligned segment.

(The `50_v2` in the filename is historical -- target count comes from
`MAX_TARGETS` and the plan file, currently 15 targets from
`target_plan_v5.json`.)

Motion goes through `MOVE_RAW` only, never `INIT` or the legacy `D<axis><deg>`
commands -- see `safe_engage()` and `docs/FIRMWARE_MOTION_HAZARDS.md` for why
`INIT` is unsafe on this firmware.

Two structural fixes over the run that produced segment 012:

- **IMUs run in this process.** They used to be a separate script with its
  own fixed `DURATION_S`; if the sweep was interrupted and restarted, the
  IMU capture had already expired by the time the real motion happened, and
  segment 012 ended up with zero temporal overlap between IMU and
  encoder/Vive. Here they start before the first move and stop after the
  last one, so overlap is structural rather than something you have to time
  by hand.
- **Segment id is allocated, not typed.** It scans the output dir and takes
  the next free number, so a leftover `.partial` from an aborted run can't
  collide with a fresh one.

FK-predicted position and Vive-measured position are expected to disagree.
That disagreement is the training signal; do not "fix" it by aligning frames.
"""
import glob
import json
import math
import os
import random
import re
import sys
import threading
import time

sys.path.insert(0, r"C:\Users\hsw10\Documents\한이음 실험\capstone-code\04_Data_Acquisition_Code")

import serial
import openvr

from collectors.collect_imu_rs232 import IMURS232Collector
from experiment.adaptive_settling_detector import (
    AdaptiveSettlingDetector,
    SettlingConfig,
    TipSample,
)
from core.schemas import (
    SCHEMA_ENCODER, STREAM_ENCODER, SCHEMA_VIVE, STREAM_VIVE,
    base_record, encoder_record, vive_record, monotonic_ns,
)
from core.jsonl_writer import JsonlWriter

# --- config ---------------------------------------------------------------
SESSION_ID = "session_real_motion_demo_001"
OUTPUT_DIR = r"C:\Users\hsw10\Documents\한이음 실험\capstone-code\04_Data_Acquisition_Code\sessions\real_motion_demo"
PLAN_PATH = OUTPUT_DIR + r"\target_plan_v6.json"

# Take only the first N targets of the plan. Safe to do because the plan is
# built by greedy farthest-point sampling: each point is the one furthest
# from everything already chosen, so ANY prefix is itself a well-spread
# subset of the workspace. Taking the first 15 is not "a corner of the
# hemisphere" -- it's a coarser sampling of the whole thing.
# Set to None for the full plan.
MAX_TARGETS = None

# Repeat the whole plan until the tracker battery dies. Each cycle re-orders
# the targets by interleaving across workspace bins (the approach in
# 06_.../sampling/balanced_shuffle.py), so consecutive trials never collapse
# into one region and, over many cycles, every region accumulates attempts at
# the same rate.
# One SET = every target in the plan, visited REPEATS_PER_POINT times
# (50 points x 3 = 150 trials). Sets repeat until the tracker battery dies.
REPEATS_PER_POINT = 3
REPEAT_UNTIL_TRACKER_DIES = True
MAX_SETS = 1000                    # backstop; the tracker will stop us first
TRACKER_DEAD_AFTER_S = 90.0        # continuous invalid tracking => battery gone

# Confirmed by experiment_scripts/probe_vive.py on 2026-08-05. Checked rather
# than blindly grabbing the first GenericTracker, so a second tracker or a
# controller enumerating first can't silently become the end-effector
# reference. Set to None to accept whatever is found first.
EXPECTED_TRACKER_SERIAL = "LHR-A84263A3"

ESP32_PORT = "COM20"
ESP32_BAUD = 115200
IMU_BAUD = 9600
# Physical wiring, per the project owner 2026-08-05: the ports run from the
# end effector DOWN to the base. This was reversed in every run up to and
# including segment 014 -- the stream labelled `imu3_tip` was really reading
# the base, which matters a lot now that settling is judged on the tip: the
# base barely moves, so it would have read "stable" instantly, always.
IMU_PORT_MAP = {
    # Verified 2026-08-06 by shaking only the end effector and watching which
    # port responded: COM12 barely moved (mean gyro 0.018 rad/s vs 0.41-0.48
    # for the outer sensors), which is what a base-mounted sensor looks like.
    # NOTE this is the OPPOSITE direction from the earlier wiring, where COM16
    # was the tip -- the numbers do not carry over across reconnects, so
    # re-run the shake test after any replug.
    "imu0_base": "COM12",       # base
    "imu1_link_root": "COM13",  # shoulder_lift area
    "imu2_link_mid": "COM14",   # wrist area
    "imu3_tip": "COM15",        # end effector -- drives the settling detector
}

MANDATORY_REST_S = 2.5    # idle polling after every point -- duty-cycle relief,
                          # see docs/ST3215_THERMAL_AND_SEQUENCING.md

# ST3215: 4096 encoder counts per 360 deg.
TICKS_PER_DEG = 4096.0 / 360.0

# --- adaptive settling on the tip IMU -------------------------------------
# The arm is measured when the END EFFECTOR is actually still, not after a
# fixed delay and not when the encoders report arrival. On a flexible link
# those are different things: the servo can be exactly on target while the
# link tip is still oscillating, which is precisely the deflection this
# experiment measures.
#
# Every number below is measured from imu3_tip in segment 014 (9350 real
# samples, 232s), not guessed. SettlingConfig's own defaults are documented
# as UNVALIDATED and two of them are outright unusable on this hardware:
#
#   tip_output_rate_hz = 100.0  -- the real WT901C emits ~10.07 Hz per packet
#       type, so the detector's min_samples (window_s * rate * 0.5 = 15)
#       could never be reached in a 0.3s window (only ~3 samples) and
#       `ready` would stay False forever.
#   tip_expected_period_s = 0.01 -- freshness allows 5 periods = 0.05s, but
#       the real inter-sample period is 0.099s, so `freshness_ok` would be
#       False on every single evaluation.
#
# Measured 0.3s-window gyro RMS distribution, which separates cleanly by 34x:
#     still   1-25 pct : 0.0005 - 0.0045 rad/s
#     moving 50-99 pct : 0.15   - 0.32   rad/s
# 0.02 rad/s (1.1 deg/s) sits in the empty middle -- ~4x above the noise
# floor, ~7x below the slowest real motion.
TIP_RATE_HZ = 10.07
SETTLING_CONFIG = SettlingConfig(
    tip_output_rate_hz=TIP_RATE_HZ,
    tip_expected_period_s=1.0 / TIP_RATE_HZ,
    window_s=1.0,                            # ~10 samples at the real rate
    tip_gyro_rms_threshold_rad_s=0.02,       # the discriminating signal
    # Accel/orientation/jerk separate far less cleanly than gyro (measured
    # dynamic-accel RMS: 25 pct 0.115, 50 pct 0.289 m/s^2 -- overlapping),
    # so they are kept deliberately loose: sanity guards against a wildly
    # wrong state, not the primary criterion. Tightening them without
    # measuring first would just make settling hang.
    dynamic_accel_rms_threshold_mps2=0.30,
    orientation_spread_threshold_rad=0.05,
    jerk_rms_threshold_mps3=40.0,
    drift_slope_threshold_rad_s=0.05,
    candidate_duration_s=0.5,
    confirmed_duration_s=1.0,                # ~10 consecutive good samples
)

# Reference ("1자") pose in raw ticks -- config/reference_pose.yaml, measured
# 2026-08-05 with the arm held under torque. Raw, not logical: the firmware's
# logical_deg is a distorted label scale (1.17 / 1.30 / 1.10 physical degrees
# per logical degree, see docs/FK_CONVENTION_CORRECTION.md), so raw ticks are
# the only lossless way to name a pose.
REFERENCE_RAW = (2055, 2093, 1970)

# Where the arm is parked when the run ends -- measured 2026-08-06 with torque
# OFF and confirmed stable (5 consecutive reads, 0 ticks of drift, 0 current).
# shoulder_lift and wrist_flex sit PAST their active limits here (1202..3186 and
# 1937..3968), resting on their mechanical stops, which is exactly why the pose
# holds itself with no torque. MOVE_RAW clamps to the active range, so the last
# ~70 ticks are covered by gravity once torque is released -- the pose is on the
# downhill side, so it settles the rest of the way rather than hanging.
REST_RAW = (2067, 1132, 4019)
AXIS_NAMES = ("shoulder_pan", "shoulder_lift", "wrist_flex")

# Stamped into every encoder and Vive record's `experiment_state` field, so
# downstream analysis can select the usable samples instead of guessing.
# Without this the settle result lived only in the console log and the data
# had no way to distinguish a settled pose from mid-flight motion -- which is
# exactly the distinction the whole tip-IMU settling machinery exists to make.
STATE_SETTLING = "SETTLING"            # commanded, waiting for the tip to go quiet
STATE_MEASURING = "MEASUREMENT_WINDOW"  # tip confirmed stable -- THIS is the label data
STATE_TIMEOUT = "SETTLE_TIMEOUT"        # never confirmed; recorded, but do not use as a settled label
STATE_REST = "REST"                     # between targets
MEASUREMENT_WINDOW_S = 2.0              # how long to hold and record once stable

experiment_state = STATE_REST           # mutated as the run progresses


def raw_to_q_rad(raw_ticks):
    """Raw encoder ticks -> FK model joint angles, using exactly the mapping
    `generate_3d_points_v5.py` used to build the plan, so a recorded pose and
    a planned pose are directly comparable.

    NOT `radians(logical_deg)`: logical_deg is a label the firmware stretches
    over each axis's calibrated span, not an angle. Feeding it to the FK was
    a real 30% scale error on shoulder_lift -- switching to tick-derived
    angles cut the rigid-model residual against real Vive data from 77.4mm to
    48.7mm on segment 006.
    """
    out = []
    for i, raw in enumerate(raw_ticks):
        if raw is None:
            out.append(None)
            continue
        rel = (raw - REFERENCE_RAW[i]) / TICKS_PER_DEG * math.pi / 180.0
        # Reference pose is declared to be q2 = pi/2, q3 = 0 (the pan/lift
        # anchors are unobservable from tip position and deliberately not
        # fitted -- see docs/FK_CONVENTION_CORRECTION.md).
        out.append(rel + math.pi / 2 if i == 1 else rel)
    return out


def allocate_segment_id(output_dir):
    """Next segment number not used by ANY stream in `output_dir`, counting
    `.partial` files too -- an aborted run's leftover `.partial` still owns
    its number (JsonlWriter refuses to reuse it, by design)."""
    used = set()
    for path in glob.glob(os.path.join(output_dir, "*_segment[0-9][0-9][0-9].jsonl*")):
        m = re.search(r"_segment(\d{3})\.jsonl", os.path.basename(path))
        if m:
            used.add(int(m.group(1)))
    return max(used) + 1 if used else 0


SEGMENT_ID = allocate_segment_id(OUTPUT_DIR)
print(f"=== allocated segment_id={SEGMENT_ID:03d} ===")

STATE_LINE_RE = re.compile(
    r"axis=(?P<axis>\S+) alias=(?P<alias>\S+) id=(?P<id>\d+) ping=(?P<ping>\w+) "
    r"torque=(?P<torque>\w+) raw_pos=(?P<raw_pos>-?\d+) raw_speed=(?P<raw_speed>-?\d+) "
    r"raw_load=(?P<raw_load>-?\d+) raw_voltage=(?P<raw_voltage>-?\d+) raw_temp=(?P<raw_temp>-?\d+) "
    r"raw_current=(?P<raw_current>-?\d+) logical_deg=(?P<logical_deg>NA|-?\d+\.?\d*) "
    r"mapping_ready=(?P<mapping_ready>\w+) .*? fault=(?P<fault>\w+)"
)

with open(PLAN_PATH, encoding="utf-8") as f:
    plan = json.load(f)
if MAX_TARGETS is not None:
    plan = plan[:MAX_TARGETS]
print(f"=== plan: {len(plan)} targets from {os.path.basename(PLAN_PATH)} ===")

# Freeze the plan next to this segment's data. `target_plan_v5.json` is
# rewritten every time the generator runs, so a record's `target_id` alone is
# NOT enough to recover which coordinate was commanded -- segments 021-024 all
# reference "T000" with different coordinates. Without this copy the pairing of
# "planned point" to "measured tracker position" is unrecoverable, which is the
# entire deliverable.
PLAN_SNAPSHOT = os.path.join(OUTPUT_DIR, f"target_plan_segment{SEGMENT_ID:03d}.json")
with open(PLAN_SNAPSHOT, "w", encoding="utf-8") as _f:
    json.dump(plan, _f, indent=2)
print(f"=== plan snapshot: {os.path.basename(PLAN_SNAPSHOT)} ===")

# Also stamp the commanded coordinate straight into each record, so a single
# file answers "planned here, measured there" without a join.
PLANNED_BY_ID = {t["target_id"]: t for t in plan}

vrsys = openvr.init(openvr.VRApplication_Other)
TRACKER_INDEX = None
TRACKER_SERIAL = None
for i in range(openvr.k_unMaxTrackedDeviceCount):
    if vrsys.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_GenericTracker:
        continue
    serial_str = vrsys.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
    if EXPECTED_TRACKER_SERIAL is not None and serial_str != EXPECTED_TRACKER_SERIAL:
        print(f"  skipping tracker {serial_str} (expected {EXPECTED_TRACKER_SERIAL})")
        continue
    TRACKER_INDEX, TRACKER_SERIAL = i, serial_str
    break
print(f"Vive tracker: device_index={TRACKER_INDEX} serial={TRACKER_SERIAL}")

# Fail before touching the motors, not 12 minutes into a run: a dead tracker
# is exactly how segment 012 produced 759 records of nothing.
if TRACKER_INDEX is None:
    openvr.shutdown()
    raise SystemExit(f"no tracker matching {EXPECTED_TRACKER_SERIAL!r} -- aborting before any motion")
_probe = vrsys.getDeviceToAbsoluteTrackingPose(
    openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
)[TRACKER_INDEX]
if not (_probe.bPoseIsValid and _probe.bDeviceIsConnected):
    openvr.shutdown()
    raise SystemExit(
        f"tracker {TRACKER_SERIAL} present but not tracking "
        f"(valid={_probe.bPoseIsValid} connected={_probe.bDeviceIsConnected}) -- aborting before any motion"
    )
print(f"  tracker pose OK (eTrackingResult={_probe.eTrackingResult})")

vive_writer = JsonlWriter(OUTPUT_DIR, STREAM_VIVE, SEGMENT_ID)
vive_host_seq = 0


def matrix_to_quat_xyzw(m):
    r = [[m[i][j] for j in range(3)] for i in range(3)]
    tr = r[0][0] + r[1][1] + r[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25*s; x=(r[2][1]-r[1][2])/s; y=(r[0][2]-r[2][0])/s; z=(r[1][0]-r[0][1])/s
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = math.sqrt(1.0+r[0][0]-r[1][1]-r[2][2]) * 2
        w=(r[2][1]-r[1][2])/s; x=0.25*s; y=(r[0][1]+r[1][0])/s; z=(r[0][2]+r[2][0])/s
    elif r[1][1] > r[2][2]:
        s = math.sqrt(1.0+r[1][1]-r[0][0]-r[2][2]) * 2
        w=(r[0][2]-r[2][0])/s; x=(r[0][1]+r[1][0])/s; y=0.25*s; z=(r[1][2]+r[2][1])/s
    else:
        s = math.sqrt(1.0+r[2][2]-r[0][0]-r[1][1]) * 2
        w=(r[1][0]-r[0][1])/s; x=(r[0][2]+r[2][0])/s; y=(r[1][2]+r[2][1])/s; z=0.25*s
    return [x, y, z, w]


def read_vive_raw():
    if TRACKER_INDEX is None:
        return None, None, None, False
    poses = vrsys.getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)
    p = poses[TRACKER_INDEX]
    if not (p.bPoseIsValid and p.bDeviceIsConnected):
        return None, None, None, False
    m = p.mDeviceToAbsoluteTracking
    matrix = [[m[r][c] for c in range(4)] for r in range(3)]
    position_m = [matrix[0][3], matrix[1][3], matrix[2][3]]
    return matrix, position_m, matrix_to_quat_xyzw(matrix), True


def capture_vive_sample(target_id=None, planned_order_index=None):
    global vive_host_seq
    t_pc_rx_ns = monotonic_ns()
    vive_host_seq += 1
    matrix, position_m, quat, ok = read_vive_raw()
    planned = PLANNED_BY_ID.get(target_id)
    base = base_record(
        schema=SCHEMA_VIVE, session_id=SESSION_ID, segment_id=SEGMENT_ID, stream=STREAM_VIVE,
        host_seq=vive_host_seq, t_pc_rx_ns=t_pc_rx_ns, valid=ok,
        parse_status="ok" if ok else "tracking_invalid",
        target_id=target_id, planned_order_index=planned_order_index,
        workspace_mode="left_of_straight_v5_rawtick_fps", experiment_state=experiment_state,
    )
    rec = vive_record(
        base, tracker_serial=TRACKER_SERIAL, device_index=TRACKER_INDEX,
        tracking_result="TrackingResult_Running_OK" if ok else "invalid",
        tracking_valid=ok, device_connected=TRACKER_INDEX is not None,
        tracking_origin="TrackingUniverseStanding",
        pose_matrix_3x4=matrix, position_m=position_m, quaternion_xyzw=quat,
    )
    # "planned here, measured there" in one record, no join required.
    rec["data"]["planned_p_xyz_mm"] = planned["p_xyz_mm"] if planned else None
    rec["data"]["planned_raw_ticks"] = planned["raw_ticks"] if planned else None
    rec["trial_id"] = trial_id
    vive_writer.put(rec)
    if ok:
        global vive_last_valid_ns
        vive_last_valid_ns = t_pc_rx_ns


# Vive runs on its own thread. It used to be sampled inside the encoder poll
# loop, which pinned it to that loop's pace -- a real run logged only 2.1
# Vive samples/sec. OpenVR itself has no such limit, and nothing about the
# tracker needs to wait for a serial round-trip.
VIVE_RATE_HZ = 60.0
_vive_stop = threading.Event()


def _vive_loop():
    period = 1.0 / VIVE_RATE_HZ
    nxt = time.perf_counter()
    while not _vive_stop.is_set():
        try:
            capture_vive_sample(current_target_id, current_order_index)
        except Exception:
            pass  # a tracker hiccup must not kill the run; records mark invalid
        nxt += period
        delay = nxt - time.perf_counter()
        if delay > 0:
            _vive_stop.wait(delay)
        else:
            nxt = time.perf_counter()


current_target_id = None
current_order_index = None
trial_id = 0                 # cycle index, stamped into every record
completed_targets = 0
vive_last_valid_ns = None    # for the battery watch


# IMUs start here, before the ESP32 connection (which resets the board and
# drops torque) and before any motion -- so their window strictly contains
# the whole sweep. They stop in the shutdown block at the bottom.
settling = AdaptiveSettlingDetector(SETTLING_CONFIG)


# The WT901C sends ONE quantity per frame (accel, gyro, angle, magnetic in
# round-robin, ~10 Hz each). `TipSample`/`from_imu_record` want accel AND gyro
# in the same sample, which no single frame ever carries -- feeding records
# straight through would silently drop 100% of them and the detector would
# never see any data at all (verified by replaying segment 014: 0 usable
# samples). So pair each gyro frame with the most recent accel/angle frame.
# Staleness is bounded by one round-robin cycle (~0.1s), which is far below
# the 1.0s metric window.
_tip_latest = {"accel": None, "euler": None}


_tip_stats = {"seen": 0, "fed": 0, "err": None}


def _stamp(record):
    """Label every IMU record with the run context, on the reader thread.

    IMURS232Collector builds the record and calls this hook BEFORE handing it
    to the writer queue, so mutating the dict here is what actually lands in
    the file. Without it the IMU streams carried no target_id / trial_id /
    experiment_state at all: 22,900 samples per sensor with no way to tell
    which belong to a settled measurement, while the encoder and Vive streams
    were fully labelled. Timestamps could be joined after the fact, but the
    point of the settling machinery is to mark the usable samples directly.
    """
    record["target_id"] = current_target_id
    record["planned_order_index"] = current_order_index
    record["trial_id"] = trial_id
    record["experiment_state"] = experiment_state
    record["workspace_mode"] = "binbalanced_v6"
    planned = PLANNED_BY_ID.get(current_target_id)
    if planned is not None:
        record["data"]["planned_p_xyz_mm"] = planned["p_xyz_mm"]
        record["data"]["planned_raw_ticks"] = planned["raw_ticks"]


def _feed_tip(record):
    """Tip IMU: label it, then feed the settling detector."""
    _stamp(record)
    _tip_stats["seen"] += 1
    d = record.get("data") or {}
    if d.get("accel_mps2") is not None:
        _tip_latest["accel"] = tuple(d["accel_mps2"])
    if d.get("euler_rad") is not None:
        _tip_latest["euler"] = tuple(d["euler_rad"])
    gyro = d.get("gyro_rads")
    if gyro is None or _tip_latest["accel"] is None:
        return
    try:
        settling.add_tip_sample(TipSample(
            t_pc_rx_ns=record["t_pc_rx_ns"],
            gyro_rads=tuple(gyro),
            accel_mps2=_tip_latest["accel"],
            euler_rad=_tip_latest["euler"],
            raw_hex=record.get("raw_hex"),
            checksum_ok=bool(record.get("checksum_ok", True)),
            stale_suspect="stale_suspect" in (record.get("quality_flags") or []),
        ))
        _tip_stats["fed"] += 1
    except Exception as exc:  # noqa: BLE001
        # IMURS232Collector swallows on_record exceptions (correctly -- an
        # observer must not kill capture) and its _publish is a no-op without
        # an event bus, so without this the failure would be invisible.
        if _tip_stats["err"] is None:
            _tip_stats["err"] = repr(exc)
        raise


imu_collectors = {}
for _sensor_id, _port in IMU_PORT_MAP.items():
    _c = IMURS232Collector(
        sensor_id=_sensor_id,
        session_id=SESSION_ID,
        segment_id=SEGMENT_ID,
        output_dir=OUTPUT_DIR,
        port=_port,
        baud=IMU_BAUD,
        # Every sensor gets the context stamp; only the tip additionally
        # drives the settling detector.
        on_record=_feed_tip if _sensor_id == "imu3_tip" else _stamp,
    )
    _c.start()
    imu_collectors[_sensor_id] = _c
    print(f"  IMU started: {_sensor_id} on {_port}")

# `.start()` only spawns the reader thread -- it returns just as happily when
# the port cannot be opened at all, and the collector then retries forever in
# the background while writing an empty file. That is exactly what happened
# with the tip IMU (COM16 vanished mid-session, twice): the run proceeded for
# 30 targets, every settle timed out at 25s because the detector never
# received a sample, and the only clue was a 0-byte JSONL discovered
# afterwards. Verify real frames are arriving BEFORE anything moves.
print("  verifying IMU streams ...")
time.sleep(2.5)
_dead = [sid for sid, c in imu_collectors.items() if c.frames_emitted == 0]
for _sid, _c in imu_collectors.items():
    print(f"    {_sid:15s} {IMU_PORT_MAP[_sid]}  frames={_c.frames_emitted:5d} "
          f"disconnects={_c.disconnect_count}")
if _dead:
    for _c in imu_collectors.values():
        _c.stop(clean=False)
    openvr.shutdown()
    raise SystemExit(
        f"no data from {_dead} -- aborting before any motion. "
        f"Check the USB cable/adapter for those sensors."
    )

ser = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=0.3)
time.sleep(1.8)
ser.reset_input_buffer()

enc_writer = JsonlWriter(OUTPUT_DIR, STREAM_ENCODER, SEGMENT_ID)
enc_host_seq = 0


def send_and_report(cmd, wait=0.8):
    print(f">>> {cmd}")
    ser.write((cmd + "\r\n").encode("ascii"))
    time.sleep(wait)
    resp = ser.read(4000)
    text = "".join(chr(c) if 32 <= c < 127 or c in (9, 10, 13) else "." for c in resp)
    print(text)
    return text


def poll_for(seconds, period_s=0.15, target_id=None, planned_order_index=None, until=None):
    """Poll STATE for `seconds`, writing an encoder + Vive record each cycle.

    Returns `(max_raw_temp, faulted, settled)`. `until`, if given, is called
    with the parsed axes dict each cycle; returning True stops early with
    `settled=True`. Without it the full duration always elapses and
    `settled` is False.
    """
    global enc_host_seq
    max_temp = 0
    deadline = time.time() + seconds
    while time.time() < deadline:
        ser.write(b"STATE\r\n")
        # Deliberately paced at ~2.1 Hz, NOT as fast as possible. A
        # non-blocking read reached ~17.6 Hz, but at that rate the servo's
        # `raw_speed` reads non-zero often enough (quantisation and micro-jitter
        # while merely holding position) that the `encoder_settled` guard never
        # passed -- 411 of 411 polls failed it while the tip IMU was already
        # quiet, so every target hit the settle timeout. At 2.1 Hz those moments
        # fall between samples. Raise this only together with a settle guard
        # that tolerates brief non-zero speed.
        time.sleep(period_s)
        resp = ser.read(4000)
        t_pc_rx_ns = monotonic_ns()
        text = resp.decode("ascii", errors="replace")
        axes = {m.group("axis"): m.groupdict() for m in STATE_LINE_RE.finditer(text)}
        if axes:
            enc_host_seq += 1
            raw_ticks, logical_deg, motor_status, fault_list = [], [], [], []
            for axis in AXIS_NAMES:
                a = axes.get(axis)
                if a is None:
                    raw_ticks.append(None); logical_deg.append(None)
                    motor_status.append("unknown"); fault_list.append("unknown")
                    continue
                deg = a["logical_deg"]
                logical_deg.append(None if deg == "NA" else float(deg))
                raw_ticks.append(int(a["raw_pos"]))
                motor_status.append("enabled" if a["torque"] == "true" else "disabled")
                fault_list.append("fault" if a["fault"] == "true" else "ok")
                max_temp = max(max_temp, int(a["raw_temp"]))
            base = base_record(
                schema=SCHEMA_ENCODER, session_id=SESSION_ID, segment_id=SEGMENT_ID,
                stream=STREAM_ENCODER, sensor_id="esp32_encoder", host_seq=enc_host_seq,
                t_pc_rx_ns=t_pc_rx_ns, valid=True, parse_status="ok",
                target_id=target_id, planned_order_index=planned_order_index,
                workspace_mode="left_of_straight_v5_rawtick_fps", raw_line=text.strip(), experiment_state=experiment_state,
                # Self-identifying: in segments 000-012 `q_encoder_rad` held
                # radians(logical_deg), which is NOT a physical angle. From
                # here it is tick-derived and comparable to the plan's q_rad.
                # Without this flag the two conventions would be
                # indistinguishable under the same field name.
                quality_flags=["q_from_raw_ticks"],
            )
            rec = encoder_record(
                base, encoder_raw_tick=raw_ticks, q_encoder_rad=raw_to_q_rad(raw_ticks),
                motor_status=motor_status, fault=fault_list,
            )
            # Keep the firmware's own label values too -- lossless record of
            # what the device reported, even though they are not angles.
            rec["data"]["firmware_logical_deg"] = logical_deg
            rec["trial_id"] = trial_id
            _pl = PLANNED_BY_ID.get(target_id)
            rec["data"]["planned_p_xyz_mm"] = _pl["p_xyz_mm"] if _pl else None
            rec["data"]["planned_raw_ticks"] = _pl["raw_ticks"] if _pl else None
            enc_writer.put(rec)
            if "fault_latched=true" in text:
                print("  !!! UNEXPECTED FAULT LATCH:", text.strip().splitlines()[-1])
                return max_temp, True, False
            if until is not None and until(axes):
                return max_temp, False, True
    return max_temp, False, False


SETTLE_TOL_TICKS = 40          # ~3.5 physical degrees. Only a guard against
                               # "still travelling" -- the real settle criterion
                               # is tip-IMU stability, and a higher servo speed
                               # overshoots past a tighter band (seen at 27 ticks).
SETTLE_TIMEOUT_S = 25.0
POS_STABLE_TOL_TICKS = 3       # encoder quantisation / dither, ~0.26 deg
POS_STABLE_SAMPLES = 3         # consecutive polls the position must hold within tol


def wait_until_settled(target_raw, target_id=None, planned_order_index=None):
    """Wait until the END EFFECTOR is actually still, then measure.

    The criterion is the tip IMU (`imu3_tip`), not a fixed delay and not
    encoder arrival:

    - A fixed delay is the wrong instrument. Segment 014 used 2.0s + 1.5s
      windows and 17 of its 20 poses were still in motion when recorded as
      done (command error up to 912 ticks = 80 physical degrees), because the
      servo speeds had been lowered to 120/100/90 for thermal reasons without
      extending the windows. Time-to-settle depends on travel distance and
      speed; both change.
    - Encoder arrival is also not enough. On a flexible link the servo can sit
      exactly on target while the tip is still swinging -- that oscillation is
      the very thing being measured, so measuring during it corrupts the
      label.

    So: `AdaptiveSettlingDetector` on the tip IMU decides, with encoder
    arrival kept only as a guard (an arm still travelling must never be
    declared settled just because the tip happens to be momentarily quiet).

    Returns `(max_raw_temp, faulted, settled)`. `settled=False` means the
    timeout expired -- reported, never silently accepted.
    """
    settling.reset()
    last = {"ev": None, "n": 0, "enc_fail": 0}
    hist = []          # recent raw_pos triples, for the "position constant" test

    def stable(axes):
        # The guard asks "is the arm still travelling?", NOT "did it land
        # exactly on target". Those are different questions and only the first
        # one belongs here: the measurement is taken wherever the arm actually
        # came to rest, and that pose is recorded, so a servo parking a few
        # degrees short is not an error -- it is just the pose. Checking
        # against the target instead rejected genuinely settled poses (seen at
        # 41 ticks / 3.6 deg) and burned the full 25s timeout for nothing.
        #
        # "Stopped" is judged by the POSITION going constant, not by
        # `raw_speed == 0`. raw_speed is a coarse instantaneous reading that
        # flickers non-zero while the servo merely holds station -- polling
        # faster made that obvious (411 of 411 polls failed the speed test on a
        # visibly stationary arm). Requiring the encoder count itself to stop
        # changing is both what "settled" actually means and far less jittery.
        pos = []
        for axis in AXIS_NAMES:
            a = axes.get(axis)
            pos.append(None if a is None else int(a["raw_pos"]))
        hist.append(pos)
        if len(hist) > POS_STABLE_SAMPLES:
            hist.pop(0)
        encoder_ok = (
            len(hist) >= POS_STABLE_SAMPLES
            and all(v is not None for p in hist for v in p)
            and all(
                max(p[k] for p in hist) - min(p[k] for p in hist) <= POS_STABLE_TOL_TICKS
                for k in range(len(AXIS_NAMES))
            )
        )
        ev = settling.evaluate(monotonic_ns(), encoder_settled=encoder_ok)
        last["ev"] = ev
        last["n"] += 1
        if not encoder_ok:
            last["enc_fail"] += 1
        return ev.is_confirmed

    result = poll_for(
        SETTLE_TIMEOUT_S,
        target_id=target_id,
        planned_order_index=planned_order_index,
        until=stable,
    )
    if not result[2]:
        # Say WHY it timed out. A bare "did not settle" sends you guessing at
        # five thresholds and six guards; the numbers narrow it to one.
        ev = last["ev"]
        if ev is None:
            print(f"    settle diag: detector never evaluated ({last['n']} polls)")
        else:
            m = ev.metrics
            fmt = lambda v: "None" if v is None else f"{v:.4f}"
            tipc = imu_collectors.get("imu3_tip")
            print(f"    settle diag: polls={last['n']} encoder_fail={last['enc_fail']} "
                  f"ready={m.ready} good_for={ev.stable_candidate_duration_s:.2f}s "
                  f"| tip seen={_tip_stats['seen']} fed={_tip_stats['fed']} "
                  f"window={len(settling._window)} "
                  f"collector_err={getattr(tipc, 'on_record_errors', '?')} "
                  f"first_err={_tip_stats['err']}")
            print(f"      gyro_rms={fmt(m.tip_gyro_rms)}/{SETTLING_CONFIG.tip_gyro_rms_threshold_rad_s} "
                  f"accel_rms={fmt(m.dynamic_accel_rms)}/{SETTLING_CONFIG.dynamic_accel_rms_threshold_mps2} "
                  f"orient={fmt(m.orientation_spread_rad)}/{SETTLING_CONFIG.orientation_spread_threshold_rad} "
                  f"jerk={fmt(m.jerk_rms)}/{SETTLING_CONFIG.jerk_rms_threshold_mps3} "
                  f"drift={fmt(m.drift_slope)}/{SETTLING_CONFIG.drift_slope_threshold_rad_s}")
            print(f"      guards={ {k: v for k, v in ev.guard_flags.items() if not v} or 'all ok'}")
    return result


# Temperature-trip/cooldown logic removed: the onboard temp sensor on the
# affected servo is confirmed broken (not a real reading), and a separate
# external sensor + hardware power cutoff on another machine is the real
# thermal safety net now. Fault-latch handling (which may still fire on
# the bad reading) stays -- recover_full() below actually restores control
# rather than just clearing the flag.


def safe_engage():
    """Bring the arm under torque WITHOUT letting it lunge, and without INIT.

    `INIT` enables torque on every axis (main.cpp:3278) BEFORE writing any
    goal position (main.cpp:3309). An ST3215 drives to whatever goal it still
    holds the instant torque comes on, so INIT makes the arm jump to a stale
    target first. That is how a previous session ended up slamming
    shoulder_lift 150 deg across its travel to the far mechanical end.

    `MOVE_RAW` (main.cpp:1847) writes the goal register via WritePosEx and
    never touches the torque-enable register, so setting goal = current
    position first turns the subsequent TORQUE ON into a pure hold.

    This means INIT is never called, so `g_is_initialized` stays false and the
    legacy D-commands (which require it, main.cpp:3167) are unavailable --
    hence every move in this script goes through MOVE_RAW instead. That is
    also strictly better here: the plan is generated in raw ticks, so this
    skips a lossy round-trip through the firmware's distorted `logical_deg`
    label scale.
    """
    send_and_report("CAL_AUTO LOAD", wait=2.0)
    pos = get_last_pos()
    print(f"  holding at current raw {pos}")
    for axis, raw in zip(AXIS_NAMES, pos):
        if raw is not None:
            send_and_report(f"MOVE_RAW {axis} {raw}", wait=0.5)
    send_and_report("TORQUE ON ALL", wait=1.5)
    after = get_last_pos()
    for axis, b, a in zip(AXIS_NAMES, pos, after):
        if b is not None and a is not None and abs(a - b) > 30:
            print(f"  !!! {axis} moved {a - b:+d} ticks on torque engage -- expected a pure hold")


def recover_full():
    """FAULTRESET only clears the fault flag -- it does not restore control
    (confirmed the hard way: an early run froze for 34 straight targets after
    a FAULTRESET-only recovery, positions byte-identical despite different
    commands, with no fault flag and no record-rate anomaly to catch it).

    Recovery re-engages torque, but via safe_engage() rather than INIT -- a
    fault mid-run is exactly when a stale goal register is most likely to be
    far from where the arm actually is, which is the worst possible moment
    for INIT's torque-before-goal ordering.
    """
    send_and_report("FAULTRESET")
    safe_engage()


def get_last_pos():
    """Read current raw_pos for all 3 axes via one STATE call (for
    stuck-detection -- compare against this after issuing a move)."""
    ser.write(b"STATE\r\n")
    time.sleep(0.3)
    resp = ser.read(4000)
    text = resp.decode("ascii", errors="replace")
    axes = {m.group("axis"): m.groupdict() for m in STATE_LINE_RE.finditer(text)}
    return tuple(int(axes[a]["raw_pos"]) if a in axes else None for a in AXIS_NAMES)


# --- sequential axis dispatch (ST3215 daisy-chain thermal protection) -----
# All 3 ST3215s share one power rail on this bus. The firmware's legacy
# D-command path (03_SO-101_control/main/src/main.cpp:3099 flushLegacyBatch)
# collects every axis queued inside kLegacyBatchWindowMs (40ms) into ONE
# RegWriteAction(), and flushes immediately once all axes are pending -- so
# three back-to-back D2/D3/D5 writes start all three servos at the same
# instant. That stacks three simultaneous stall-current draws on the shared
# rail, which is the fastest way to heat this chain. 03_SO-101_control/
# manuals/SO-101_Waveshare_ST3215_Final_Codex_Manual.md section 10.2 warns
# about exactly this ("다축 동시 이동 -> 순간 전류 폭증").
#
# Spacing each axis past the 40ms batch window makes the firmware flush one
# single-axis move per batch, so only one servo is accelerating at a time.
# See docs/ST3215_THERMAL_AND_SEQUENCING.md for the load/duty reasoning.
# MOVE_RAW is inherently one axis per command, so sequencing is structural
# here rather than something the gap has to enforce. The gap still matters:
# it keeps only one servo accelerating at a time on the shared rail, and
# doubles as settle time.
AXIS_COMMAND_GAP_S = 0.6


def send_targets_sequential(raw_ticks, target_id=None, planned_order_index=None):
    """Dispatch one axis at a time via MOVE_RAW, polling (and therefore
    logging encoder + Vive) during each gap so no data is lost while waiting.
    Returns `(max_temp, faulted)` -- same contract as poll_for, and bails out
    early on a fault latch instead of piling more commands onto a faulted bus.

    The firmware clamps each target to that axis's active raw range
    (main.cpp:1843), so an out-of-range value is refused rather than driven
    into a hard stop."""
    max_temp = 0
    for axis, raw in zip(AXIS_NAMES, raw_ticks):
        ser.write(f"MOVE_RAW {axis} {round(raw)}\r\n".encode("ascii"))
        temp, faulted, _ = poll_for(
            AXIS_COMMAND_GAP_S, target_id=target_id, planned_order_index=planned_order_index
        )
        max_temp = max(max_temp, temp)
        if faulted:
            return max_temp, True
    return max_temp, False


_vive_thread = threading.Thread(target=_vive_loop, name="vive", daemon=True)
_vive_thread.start()
print(f"=== Vive thread started at {VIVE_RATE_HZ:.0f} Hz ===")

print("=== engaging torque (no INIT) ===")
safe_engage()

print("=== baseline at reference pose (2s) ===")
poll_for(2)

aborted = False
interrupted = False
tracker_dead = False
stuck_streak = 0
prev_final_pos = get_last_pos()
retry_counts = {}
MAX_RETRIES_PER_TARGET = 4
cycle = 0
_rng = random.Random(20260806)


def interleaved_order(targets, rng):
    """Order one cycle by taking one target from each workspace bin in turn.

    Straight from 06_.../sampling/balanced_shuffle.py's
    `_interleaved_target_order`: consecutive trials then never collapse into a
    single region, so if the run is cut short partway through a cycle the
    attempts collected so far are still spread evenly instead of covering only
    whichever corner happened to come first.
    """
    groups = {}
    for k, tt in enumerate(targets):
        groups.setdefault(tt.get("bin_id", "-"), []).append(k)
    bins = list(groups)
    rng.shuffle(bins)
    for v in groups.values():
        rng.shuffle(v)
    order, layer = [], 0
    while True:
        active = [b for b in bins if layer < len(groups[b])]
        if not active:
            break
        rng.shuffle(active)
        order.extend(groups[b][layer] for b in active)
        layer += 1
    return order


try:
  while cycle < MAX_SETS and not aborted and not tracker_dead:
    # One set = REPEATS_PER_POINT independently interleaved passes. Building the
    # order this way -- rather than repeating each point back-to-back -- puts
    # ~50 other moves between a point's attempts, so the repeats sample
    # independent approaches to that pose instead of the servo simply holding
    # still and re-reading the same settled position three times.
    order = []
    for _rep in range(REPEATS_PER_POINT):
        order.extend(interleaved_order(plan, _rng))
    print(f"\n########## SET {cycle} : {len(plan)} points x {REPEATS_PER_POINT} "
          f"= {len(order)} trials, bin-interleaved ##########")
    trial_id = cycle
    i = 0
    while i < len(order):
        if aborted or tracker_dead:
            break
        t = plan[order[i]]
        tid = t["target_id"]
        retry_counts[tid] = retry_counts.get(tid, 0)
        if retry_counts[tid] > MAX_RETRIES_PER_TARGET:
            print(f"  !!! {tid} failed to recover after {MAX_RETRIES_PER_TARGET} retries -- aborting for safety")
            aborted = True
            break
        idx = t["planned_order_index"]
        d = t["raw_ticks"]
        rel = t.get("deg_from_reference", [None] * 3)
        current_target_id, current_order_index = tid, idx
        print(f"=== {tid}: raw={d}  (from reference: "
              f"pan{rel[0]:+.1f} lift{rel[1]:+.1f} wrist{rel[2]:+.1f} deg) ===")
        experiment_state = STATE_SETTLING
        temp0, faulted0 = send_targets_sequential(d, target_id=tid, planned_order_index=idx)
        if faulted0:
            print("  fault latched during sequential dispatch -- recovering")
            recover_full()
            retry_counts[tid] += 1
            continue  # retry the SAME target (do not advance i)

        temp1, faulted1, settled = wait_until_settled(d, target_id=tid, planned_order_index=idx)
        if faulted1:
            print("  fault latched (likely bad temp sensor reading) -- recovering")
            recover_full()
            retry_counts[tid] += 1
            continue  # retry the SAME target (do not advance i)

        # The measurement window: the only samples marked as a settled label.
        # Everything before this is SETTLING (the arm was still moving or the
        # tip was still ringing); if the tip never went quiet the window is
        # marked SETTLE_TIMEOUT instead, so the data says so rather than
        # looking identical to a good measurement.
        if settled:
            experiment_state = STATE_MEASURING
        else:
            experiment_state = STATE_TIMEOUT
            reached = get_last_pos()
            err = [None if (r is None or c is None) else r - round(c) for r, c in zip(reached, d)]
            print(f"  !!! {tid} tip never went quiet within {SETTLE_TIMEOUT_S}s -- "
                  f"reached={reached} err_ticks={err} (marked {STATE_TIMEOUT})")
        temp2, faulted2, _ = poll_for(MEASUREMENT_WINDOW_S, target_id=tid, planned_order_index=idx)
        if faulted2:
            print("  fault latched (likely bad temp sensor reading) -- recovering")
            recover_full()
            retry_counts[tid] += 1
            continue  # retry the SAME target

        final_pos = get_last_pos()
        if final_pos == prev_final_pos:
            stuck_streak += 1
            print(f"  !!! position identical to previous target ({final_pos}) -- stuck_streak={stuck_streak}")
            if stuck_streak >= 2:
                print("  !!! looks frozen (no fault flag raised) -- forcing full recovery")
                recover_full()
                stuck_streak = 0
                retry_counts[tid] += 1
                continue  # retry the SAME target
        else:
            stuck_streak = 0
        prev_final_pos = final_pos

        # raw_temp is a servo register count, NOT degrees C -- no datasheet in
        # the repo to convert it, and the sensor on shoulder_lift is a known
        # bad unit. Printed only as a relative "is anything climbing" hint.
        print(f"  {tid} {'SETTLED' if settled else 'TIMEOUT '} pos={final_pos}, "
              f"max raw_temp={max(temp0, temp1, temp2)}, rest {MANDATORY_REST_S}s")
        experiment_state = STATE_REST
        poll_for(MANDATORY_REST_S)
        current_target_id, current_order_index = None, None
        completed_targets += 1
        i += 1

        # Battery watch. The tracker does not announce that it is dying -- it
        # just stops returning valid poses, and segment 012 showed how easy it
        # is to keep sweeping for an hour afterwards recording nothing usable.
        if vive_last_valid_ns is not None:
            gone_s = (monotonic_ns() - vive_last_valid_ns) / 1e9
            if gone_s > TRACKER_DEAD_AFTER_S:
                print(f"\n!!! tracker has been invalid for {gone_s:.0f}s "
                      f"-- treating the battery as dead and stopping cleanly")
                tracker_dead = True

    cycle += 1
    if not REPEAT_UNTIL_TRACKER_DIES:
        break

except KeyboardInterrupt:
    interrupted = True
    print("\n!!! interrupted -- parking the arm before shutdown")

# Park, then release. Every exit path lands here -- normal finish, dead
# tracker, abort, or Ctrl+C -- so the arm is never left holding a pose under
# torque with nobody watching. Torque-off is the LAST step because the rest
# pose leans on its mechanical stops and only stays put once the servos let go.
try:
    experiment_state = STATE_REST
    current_target_id, current_order_index = None, None
    print("=== parking at rest pose ===")
    send_targets_sequential(REST_RAW)
    # Wait for the arm to actually ARRIVE, not a fixed 3s. wrist_flex has the
    # longest travel to the rest pose and a fixed wait left it stranded
    # part-way; travel time depends on distance and the current speed setting,
    # so wait on the position going constant instead.
    park_hist = []

    def _parked(axes):
        pos = [None if axes.get(a) is None else int(axes[a]["raw_pos"]) for a in AXIS_NAMES]
        park_hist.append(pos)
        if len(park_hist) > POS_STABLE_SAMPLES:
            park_hist.pop(0)
        return (
            len(park_hist) >= POS_STABLE_SAMPLES
            and all(v is not None for p in park_hist for v in p)
            and all(max(p[k] for p in park_hist) - min(p[k] for p in park_hist) <= POS_STABLE_TOL_TICKS
                    for k in range(len(AXIS_NAMES)))
        )

    _, _, park_ok = poll_for(20.0, until=_parked)
    print(f"  parked at {get_last_pos()} (commanded {REST_RAW}, clamped to the active range)"
          + ("" if park_ok else "  !!! did not come to rest within 20s"))
    send_and_report("TORQUE OFF ALL", wait=1.5)
    time.sleep(1.5)
    print(f"  settled at {get_last_pos()} with torque released")
except Exception as exc:  # noqa: BLE001
    print(f"  !!! parking failed ({exc}) -- releasing torque anyway")
    try:
        send_and_report("TORQUE OFF ALL", wait=1.5)
    except Exception:
        pass

# Shutdown always runs, so an interrupt can't leave 4 IMU reader threads and
# 6 open writers behind. clean=False on an interrupt leaves the segment's
# `.partial` files in place on purpose: a truncated segment must not be
# mistaken for a complete one, and allocate_segment_id() skips past it next run.
clean = not interrupted
_vive_stop.set()
_vive_thread.join(timeout=5.0)
for _sensor_id, _c in imu_collectors.items():
    _c.stop(clean=clean)
    print(f"  IMU stopped: {_sensor_id}: {_c.stats()}")
enc_writer.close(clean=clean)
vive_writer.close(clean=clean)
ser.close()
openvr.shutdown()
summary = {
    "session_id": SESSION_ID,
    "segment_id": SEGMENT_ID,
    "plan_file": os.path.basename(PLAN_PATH),
    "plan_snapshot": os.path.basename(PLAN_SNAPSHOT),
    "points_per_set": len(plan),
    "repeats_per_point": REPEATS_PER_POINT,
    "trials_per_set": len(plan) * REPEATS_PER_POINT,
    "sets_completed": cycle,
    "trials_completed": completed_targets,
    "stopped_because": ("tracker_battery_dead" if tracker_dead else
                        "interrupted" if interrupted else
                        "aborted" if aborted else "max_sets"),
    "encoder_records": enc_host_seq,
    "vive_records": vive_host_seq,
    "imu": {sid: c.stats() for sid, c in imu_collectors.items()},
    "rates_hz": {"encoder_poll": round(1.0 / 0.45, 2), "vive": VIVE_RATE_HZ, "imu_per_sensor": 40},
    "reference_raw": list(REFERENCE_RAW),
    "settling": {
        "criterion": "tip IMU (imu3_tip / COM16) adaptive settling",
        "tip_gyro_rms_threshold_rad_s": SETTLING_CONFIG.tip_gyro_rms_threshold_rad_s,
        "confirmed_duration_s": SETTLING_CONFIG.confirmed_duration_s,
        "measurement_window_s": MEASUREMENT_WINDOW_S,
        "timeout_s": SETTLE_TIMEOUT_S,
    },
    "experiment_state_values": [STATE_SETTLING, STATE_MEASURING, STATE_TIMEOUT, STATE_REST],
    "note": "Use experiment_state == MEASUREMENT_WINDOW for settled labels. "
            "Each record carries planned_p_xyz_mm (rigid FK prediction, UNCALIBRATED model) "
            "and the Vive stream carries the measured position_m.",
}
SUMMARY_PATH = os.path.join(OUTPUT_DIR, f"session_summary_segment{SEGMENT_ID:03d}.json")
with open(SUMMARY_PATH, "w", encoding="utf-8") as _f:
    json.dump(summary, _f, indent=2, ensure_ascii=False)
print(f"=== summary -> {os.path.basename(SUMMARY_PATH)} ===")
print(
    f"=== done (aborted={aborted}, interrupted={interrupted}, tracker_dead={tracker_dead}): "
    f"{cycle} sets, {completed_targets} trials, "
    f"{enc_host_seq} encoder / {vive_host_seq} vive records (segment {SEGMENT_ID:03d}) ==="
)

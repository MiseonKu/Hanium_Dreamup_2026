"""Capture the reference ("1자") pose that every run starts from.

Records, for one static pose held under torque:
  - the 3 joint angles the firmware reports (logical_deg + raw ticks)
  - all 4 IMUs' gravity-referenced Euler angles, averaged over a window
  - the Vive tracker pose, if SteamVR is up

Why the IMUs matter here: the FK model's q1/q2 offsets are unobservable from
tip position alone (the shoulder sits at the model's origin, so a constant
q2 offset just rotates the whole arm about that origin -- a global rigid
rotation, which any registration absorbs). What is NOT recoverable that way
is where the links actually point relative to GRAVITY, and that is what
decides how the flexible link sags and how much floor clearance a planned
pose really has. The IMUs measure it directly.

Writes `config/reference_pose.yaml`. Nothing reads that file automatically
yet -- it is the measured record the plan generator's HOME_DEG and the
sag/clearance analysis are checked against.

Run with the arm already in the reference pose and torque ON.
"""
import json
import math
import os
import statistics
import sys
import time

REPO = r"C:\Users\hsw10\Documents\한이음 실험\capstone-code"
PKG = os.path.join(REPO, "04_Data_Acquisition_Code")
sys.path.insert(0, PKG)

import serial  # noqa: E402

from collectors.collect_esp32_state_poll import parse_state_text  # noqa: E402
from collectors.collect_imu_rs232 import decode_frame, extract_frames  # noqa: E402

ESP32_PORT, ESP32_BAUD = "COM20", 115200
IMU_BAUD = 9600
IMU_PORTS = {
    "imu0_base": "COM18",
    "imu1_link_root": "COM16",
    "imu2_link_mid": "COM17",
    "imu3_tip": "COM19",
}
CAPTURE_S = 8.0
OUT_PATH = os.path.join(PKG, "config", "reference_pose.yaml")


def _send(ser, cmd, wait):
    ser.write((cmd + "\r\n").encode("ascii"))
    time.sleep(wait)
    return ser.read(8000).decode("ascii", errors="replace")


def capture_joints():
    """Hold the hand-set pose under torque WITHOUT letting the arm snap.

    `applyTorque` (main.cpp:1559) only calls `EnableTorque` -- it never writes
    the current position to the servo's goal register first. An ST3215 drives
    to whatever goal it still holds the moment torque comes on, and after a
    hand-repositioning that goal is stale (here: the previous run's home
    return). Enabling torque directly would therefore yank the arm back to
    the old goal and destroy the pose being captured.

    `MOVE_RAW` (main.cpp:1847) writes the goal register via `WritePosEx`
    without touching the torque-enable register, so writing goal = current
    position while torque is still OFF makes the subsequent TORQUE ON a
    no-op hold.
    """
    ser = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=0.3)
    time.sleep(1.8)
    ser.reset_input_buffer()
    # Opening the port resets the board, clearing the runtime mapping (though
    # not the servos' actual torque state).
    _send(ser, "CAL_AUTO LOAD", 2.0)

    before = parse_state_text(_send(ser, "STATE", 1.0))["axes"]
    print("  hand-set pose (torque off):")
    for name in ("shoulder_pan", "shoulder_lift", "wrist_flex"):
        a = before.get(name)
        if a:
            print(f"    {name:15s} raw_pos={a['raw_pos']}  logical_deg={a['logical_deg']}")

    for name in ("shoulder_pan", "shoulder_lift", "wrist_flex"):
        a = before.get(name)
        if a is None or a.get("raw_pos") is None:
            continue
        out = _send(ser, f"MOVE_RAW {name} {a['raw_pos']}", 0.6)
        if "write failed" in out:
            print(f"    !! MOVE_RAW failed for {name}: {out.strip()}")

    _send(ser, "TORQUE ON ALL", 1.5)
    after = parse_state_text(_send(ser, "STATE", 1.0))["axes"]

    for name in ("shoulder_pan", "shoulder_lift", "wrist_flex"):
        b, a = before.get(name), after.get(name)
        if not b or not a or b.get("raw_pos") is None or a.get("raw_pos") is None:
            continue
        drift = a["raw_pos"] - b["raw_pos"]
        # A few ticks is normal (the servo takes up backlash / holds against
        # gravity). A large jump means the goal was clamped to the active
        # range and the arm really did move -- say so rather than silently
        # capturing a different pose than the one that was set.
        if abs(drift) > 30:
            print(f"    !! {name} moved {drift:+d} ticks when torque engaged "
                  f"(likely clamped to active range) -- captured pose differs from the one you set")
    ser.close()
    return after


def capture_imus():
    conns = {}
    for name, port in IMU_PORTS.items():
        try:
            conns[name] = serial.Serial(port, IMU_BAUD, timeout=0.05)
        except Exception as exc:
            print(f"  {name}: OPEN FAILED {exc}")
    bufs = {n: bytearray() for n in conns}
    carry = {n: 0 for n in conns}
    euler = {n: [] for n in conns}

    time.sleep(0.5)
    for c in conns.values():
        c.reset_input_buffer()

    deadline = time.time() + CAPTURE_S
    while time.time() < deadline:
        for name, conn in conns.items():
            data = conn.read(max(1, getattr(conn, "in_waiting", 0) or 1))
            if not data:
                continue
            bufs[name].extend(data)
            frames, carry[name] = extract_frames(bufs[name], carry[name])
            for fb, _resync in frames:
                d = decode_frame(fb)
                if d.get("euler_rad") is not None:
                    euler[name].append(d["euler_rad"])
    for c in conns.values():
        c.close()

    out = {}
    for name, samples in euler.items():
        if not samples:
            out[name] = None
            continue
        out[name] = {
            "n_samples": len(samples),
            "roll_deg": round(math.degrees(statistics.fmean(s[0] for s in samples)), 2),
            "pitch_deg": round(math.degrees(statistics.fmean(s[1] for s in samples)), 2),
            "yaw_deg": round(math.degrees(statistics.fmean(s[2] for s in samples)), 2),
            "roll_sd_deg": round(math.degrees(statistics.pstdev([s[0] for s in samples])), 3),
            "pitch_sd_deg": round(math.degrees(statistics.pstdev([s[1] for s in samples])), 3),
            "yaw_sd_deg": round(math.degrees(statistics.pstdev([s[2] for s in samples])), 3),
        }
    return out


def capture_vive():
    try:
        import openvr
    except Exception:
        return None
    try:
        vr = openvr.init(openvr.VRApplication_Other)
    except Exception as exc:
        return {"error": str(exc)}
    try:
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if vr.getTrackedDeviceClass(i) != openvr.TrackedDeviceClass_GenericTracker:
                continue
            serial_str = vr.getStringTrackedDeviceProperty(i, openvr.Prop_SerialNumber_String)
            p = vr.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount
            )[i]
            m = p.mDeviceToAbsoluteTracking
            return {
                "tracker_serial": serial_str,
                "device_index": i,
                "tracking_valid": bool(p.bPoseIsValid and p.bDeviceIsConnected),
                "position_m": [round(m[r][3], 5) for r in range(3)],
            }
        return None
    finally:
        openvr.shutdown()


def main():
    print("=== joints (torque ON, holding the pose) ===")
    axes = capture_joints()
    for name in ("shoulder_pan", "shoulder_lift", "wrist_flex"):
        a = axes.get(name)
        if a is None:
            print(f"  {name}: MISSING")
            continue
        print(f"  {name:15s} logical_deg={a['logical_deg']}  raw_pos={a['raw_pos']}  torque={a['torque']}")

    print(f"\n=== IMUs (gravity-referenced Euler, {CAPTURE_S:.0f}s average) ===")
    imus = capture_imus()
    for name in IMU_PORTS:
        v = imus.get(name)
        if not v:
            print(f"  {name}: NO DATA")
            continue
        print(f"  {name:15s} roll={v['roll_deg']:+7.2f} pitch={v['pitch_deg']:+7.2f} "
              f"yaw={v['yaw_deg']:+8.2f}  (sd {v['roll_sd_deg']:.2f}/{v['pitch_sd_deg']:.2f}/"
              f"{v['yaw_sd_deg']:.2f}, n={v['n_samples']})")

    print("\n=== Vive ===")
    vive = capture_vive()
    print(f"  {vive}")

    doc = {
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Measured reference pose. Joint angles are the firmware's logical_deg; "
                "IMU angles are gravity-referenced and are what pins the arm's real "
                "orientation (the FK model's q1/q2 offsets cannot be recovered from tip "
                "position alone).",
        "joints": {
            n: {"logical_deg": axes[n]["logical_deg"], "raw_pos": axes[n]["raw_pos"]}
            for n in ("shoulder_pan", "shoulder_lift", "wrist_flex") if n in axes
        },
        "imu_euler_deg": imus,
        "vive": vive,
    }
    # Written as JSON, which is valid YAML -- avoids adding a yaml dependency
    # for one file (nothing in this package imports yaml today).
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()

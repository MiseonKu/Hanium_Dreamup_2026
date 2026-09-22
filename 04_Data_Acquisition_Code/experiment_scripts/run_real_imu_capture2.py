"""IMU-only capture, for bench-checking the 4 IMUs on their own.

NOT part of the sweep flow any more. `collect_3d_sweep_50_v2.py` now starts
and stops these same collectors inside its own process, so the two can't
drift out of time-alignment the way they did for segment 012 (fixed
DURATION_S here expired before the restarted sweep actually ran, leaving
zero overlap between IMU and encoder/Vive data). Running this alongside the
sweep would also fight it for the same COM ports.

Use it to answer "are all 4 IMUs alive and streaming?" before a real run.
"""
import sys
import time

sys.path.insert(0, r"C:\Users\hsw10\Documents\한이음 실험\capstone-code\04_Data_Acquisition_Code")

from collectors.collect_imu_rs232 import IMURS232Collector

SESSION_ID = "session_real_motion_demo_001"
# Separate directory on purpose: bench-check output must not land in
# sessions/real_motion_demo/, where collect_3d_sweep_50_v2.py's
# allocate_segment_id() scans -- a bench file there would push every future
# real run's segment number forward.
OUTPUT_DIR = r"C:\Users\hsw10\Documents\한이음 실험\capstone-code\04_Data_Acquisition_Code\sessions\imu_bench_check"
DURATION_S = 60
SEGMENT_ID = 0

# End effector -> base, per the project owner 2026-08-05. Segments 000-014
# were captured with this reversed, so their sensor_id labels are wrong (the
# raw data itself is fine).
PORT_MAP = {
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


def main():
    collectors = {}
    for sensor_id, port in PORT_MAP.items():
        c = IMURS232Collector(
            sensor_id=sensor_id,
            session_id=SESSION_ID,
            segment_id=SEGMENT_ID,
            output_dir=OUTPUT_DIR,
            port=port,
            baud=9600,
        )
        collectors[sensor_id] = c

    print(f"Starting {len(collectors)} real IMU collectors for {DURATION_S}s ...")
    for sensor_id, c in collectors.items():
        c.start()
        print(f"  started {sensor_id} on {PORT_MAP[sensor_id]}")

    time.sleep(DURATION_S)

    print("Stopping (clean)...")
    for sensor_id, c in collectors.items():
        c.stop(clean=True)
        stats = c.stats() if hasattr(c, "stats") else None
        print(f"  stopped {sensor_id}: {stats}")

    print(f"\nOutput dir: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""Resample every stream onto one uniform time grid per trial.

`build_trials.py` collapses a trial to a single settled row; this keeps the
transient. For each trial the window runs from the start of the SETTLING run
through the end of its MEASUREMENT_WINDOW -- the move, the ringing, and the
settled hold -- resampled to a fixed rate with all streams on the same clock.

Why resampling rather than a packet pivot: the WT901C emits acceleration,
angular_velocity, angle and quaternion as four separate frames, so a "40 Hz IMU
sample" does not exist as a row anywhere in the raw data. Snapping each packet
type onto a shared grid aligns them with each other, with the 60 Hz Vive and
with the ~2 Hz encoder in one step, and fills nothing that was not actually
measured within `tolerance` of the grid point.

Clock: `t_pc_rx_ns` is monotonic per segment, so grids are built per segment.

Output: `derived/parquet/<session>/timeseries.parquet`
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

PARQUET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "derived", "parquet")

SETTLED = "MEASUREMENT_WINDOW"
SETTLING = "SETTLING"
JOINTS = ["j0", "j1", "j2"]
IMU_SENSORS = ["imu0_base", "imu1_link_root", "imu2_link_mid", "imu3_tip"]

# packet_type -> (source columns, output suffixes). A given run emits only some
# of these -- the rigs are configured for acceleration/angular_velocity/angle
# plus EITHER magnetic_field or quaternion -- so the schema is narrowed to the
# types actually present in the session (see `present_packets`).
IMU_PACKETS = {
    "acceleration": (["accel_mps2_x", "accel_mps2_y", "accel_mps2_z"], ["accel_x", "accel_y", "accel_z"]),
    "angular_velocity": (["gyro_rads_x", "gyro_rads_y", "gyro_rads_z"], ["gyro_x", "gyro_y", "gyro_z"]),
    "angle": (["euler_rad_x", "euler_rad_y", "euler_rad_z"], ["euler_x", "euler_y", "euler_z"]),
    # mag_uT was never filled by the collector, so the raw counts are what exist.
    "magnetic_field": (["mag_raw_x", "mag_raw_y", "mag_raw_z"], ["mag_x_raw", "mag_y_raw", "mag_z_raw"]),
    "quaternion": (
        ["quaternion_xyzw_x", "quaternion_xyzw_y", "quaternion_xyzw_z", "quaternion_xyzw_w"],
        ["quat_x", "quat_y", "quat_z", "quat_w"],
    ),
}


def present_packets(imu_path: str) -> dict[str, tuple[list[str], list[str]]]:
    """The subset of IMU_PACKETS this session actually recorded."""
    if not os.path.exists(imu_path):
        return {}
    seen = ds.dataset(imu_path).to_table(columns=["packet_type"]).column("packet_type").unique().to_pylist()
    return {name: spec for name, spec in IMU_PACKETS.items() if name in seen}

# How far a grid point may reach for a real sample, per stream. Roughly 1.5x the
# nominal period -- wide enough to bridge one dropped frame, tight enough that a
# disconnected sensor leaves NaN instead of a stale value smeared across seconds.
TOLERANCE_NS = {"vive": 40_000_000, "imu": 60_000_000, "encoder": 1_000_000_000}

VIVE_SOURCE = ["position_m_x", "position_m_y", "position_m_z",
               "quaternion_xyzw_x", "quaternion_xyzw_y", "quaternion_xyzw_z", "quaternion_xyzw_w"]
VIVE_OUT = ["vive_pos_x_m", "vive_pos_y_m", "vive_pos_z_m",
            "vive_quat_x", "vive_quat_y", "vive_quat_z", "vive_quat_w"]
ENCODER_SOURCE = [f"encoder_raw_tick_{j}" for j in JOINTS] + [f"q_encoder_rad_{j}" for j in JOINTS]
ENCODER_OUT = [f"encoder_tick_{j}" for j in JOINTS] + [f"q_encoder_rad_{j}" for j in JOINTS]


def output_columns(packets: dict[str, tuple[list[str], list[str]]]) -> list[tuple[str, str]]:
    cols = [("segment_id", "i4"), ("block", "i4"), ("target_id", "str"), ("set_id", "f8"),
            ("t_rel_s", "f8"), ("t_pc_rx_ns", "i8"), ("phase", "str")]
    cols += [(name, "f8") for name in VIVE_OUT]
    cols += [(name, "f8") for name in ENCODER_OUT]
    for sensor in IMU_SENSORS:
        for _, suffixes in packets.values():
            cols += [(f"{sensor}_{suffix}", "f8") for suffix in suffixes]
    return cols


ARROW = {"i4": pa.int32(), "i8": pa.int64(), "f8": pa.float64(), "str": pa.string()}


def trial_windows(vive: pd.DataFrame) -> pd.DataFrame:
    """One row per trial: the SETTLING start (when there is one) through the end
    of the MEASUREMENT_WINDOW that follows it."""
    vive = vive.sort_values("host_seq", kind="stable").reset_index(drop=True)
    state = vive["experiment_state"].fillna("__none__")
    vive["block_id"] = (state != state.shift()).cumsum()

    blocks = vive.groupby("block_id").agg(
        state=("experiment_state", "first"),
        t_start=("t_pc_rx_ns", "min"),
        t_end=("t_pc_rx_ns", "max"),
        target_id=("target_id", "first"),
        set_id=("trial_id", "first"),
        n=("host_seq", "size"),
    ).reset_index()

    windows = []
    for position, row in enumerate(blocks.itertuples()):
        if row.state != SETTLED:
            continue
        start = row.t_start
        settle_start = row.t_start
        if position > 0:
            previous = blocks.iloc[position - 1]
            if previous["state"] == SETTLING:
                start = previous["t_start"]
        windows.append({
            "block": int(row.block_id),
            "t_start": int(start),
            "t_settled": int(settle_start),
            "t_end": int(row.t_end),
            "target_id": row.target_id,
            "set_id": row.set_id,
        })
    return pd.DataFrame(windows)


def build_grid(windows: pd.DataFrame, step_ns: int) -> pd.DataFrame:
    """Uniform samples inside each window, concatenated in time order."""
    frames = []
    for row in windows.itertuples():
        times = np.arange(row.t_start, row.t_end + 1, step_ns, dtype=np.int64)
        if len(times) == 0:
            continue
        frames.append(pd.DataFrame({
            "t_pc_rx_ns": times,
            "block": np.int32(row.block),
            "target_id": row.target_id,
            "set_id": row.set_id,
            "t_rel_s": (times - row.t_settled) / 1e9,
            "phase": np.where(times >= row.t_settled, SETTLED, SETTLING),
        }))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("t_pc_rx_ns", kind="stable").reset_index(drop=True)


def snap(grid: pd.DataFrame, source: pd.DataFrame, source_cols: list[str],
         out_cols: list[str], tolerance_ns: int) -> pd.DataFrame:
    """Attach the nearest real sample within tolerance to each grid point."""
    source = source[["t_pc_rx_ns"] + source_cols].dropna(subset=["t_pc_rx_ns"])
    source = source.sort_values("t_pc_rx_ns", kind="stable")
    if source.empty:
        for name in out_cols:
            grid[name] = np.nan
        return grid

    merged = pd.merge_asof(
        grid[["t_pc_rx_ns"]], source, on="t_pc_rx_ns",
        direction="nearest", tolerance=tolerance_ns,
    )
    for source_name, out_name in zip(source_cols, out_cols):
        grid[out_name] = merged[source_name].to_numpy()
    return grid


def build_session(session_dir: str, rate_hz: float) -> None:
    session = os.path.basename(session_dir.rstrip(os.sep))
    vive_path = os.path.join(session_dir, "vive_tracker.parquet")
    if not os.path.exists(vive_path):
        print(f"[timeseries] {session}: no vive_tracker.parquet, skipped", flush=True)
        return

    step_ns = int(round(1e9 / rate_hz))
    imu_path = os.path.join(session_dir, "imu.parquet")
    packets = present_packets(imu_path)
    columns = output_columns(packets)
    schema = pa.schema([pa.field(name, ARROW[code]) for name, code in columns], metadata={
        "session": session,
        "rate_hz": str(rate_hz),
        "imu_packet_types": ",".join(packets) or "none",
        "window": f"{SETTLING} start through {SETTLED} end, per trial",
        "t_rel_s": f"seconds relative to the start of {SETTLED} (negative during {SETTLING})",
        "joint_order": "shoulder_pan,shoulder_lift,wrist_flex",
    })

    segments = ds.dataset(vive_path).to_table(columns=["segment_id"]).column("segment_id").unique().to_pylist()
    out_path = os.path.join(session_dir, "timeseries.parquet")
    total_rows = 0

    with pq.ParquetWriter(out_path, schema, compression="zstd") as writer:
        for segment_id in sorted(segment for segment in segments if segment is not None):
            in_segment = ds.field("segment_id") == segment_id

            vive = ds.dataset(vive_path).to_table(
                columns=["segment_id", "host_seq", "experiment_state", "t_pc_rx_ns",
                         "target_id", "trial_id"] + VIVE_SOURCE,
                filter=in_segment).to_pandas()
            if vive.empty:
                continue

            windows = trial_windows(vive)
            if windows.empty:
                continue
            grid = build_grid(windows, step_ns)
            if grid.empty:
                continue
            grid["segment_id"] = np.int32(segment_id)

            grid = snap(grid, vive, VIVE_SOURCE, VIVE_OUT, TOLERANCE_NS["vive"])

            encoder_path = os.path.join(session_dir, "encoder.parquet")
            if os.path.exists(encoder_path):
                encoder = ds.dataset(encoder_path).to_table(
                    columns=["t_pc_rx_ns"] + ENCODER_SOURCE, filter=in_segment).to_pandas()
                grid = snap(grid, encoder, ENCODER_SOURCE, ENCODER_OUT, TOLERANCE_NS["encoder"])
            else:
                for name in ENCODER_OUT:
                    grid[name] = np.nan

            imu = pd.DataFrame()
            if packets:
                needed = sorted({column for source, _ in packets.values() for column in source})
                imu = ds.dataset(imu_path).to_table(
                    columns=["t_pc_rx_ns", "sensor_id", "packet_type"] + needed,
                    filter=in_segment).to_pandas()

            for sensor in IMU_SENSORS:
                subset = imu[imu["sensor_id"] == sensor] if not imu.empty else pd.DataFrame()
                for packet_type, (source_cols, suffixes) in packets.items():
                    out_cols = [f"{sensor}_{suffix}" for suffix in suffixes]
                    rows = subset[subset["packet_type"] == packet_type] if not subset.empty else pd.DataFrame()
                    if rows.empty:
                        for name in out_cols:
                            grid[name] = np.nan
                    else:
                        grid = snap(grid, rows, source_cols, out_cols, TOLERANCE_NS["imu"])

            table = pa.Table.from_arrays(
                [pa.array(grid[name].to_numpy() if code != "str" else grid[name],
                          type=ARROW[code], from_pandas=True) for name, code in columns],
                schema=schema)
            writer.write_table(table)
            total_rows += len(grid)
            print(f"[timeseries] {session} seg{segment_id:03d}: {len(windows):,} trials, {len(grid):,} rows", flush=True)

    if total_rows == 0:
        # Mock/pilot sessions never reach MEASUREMENT_WINDOW; don't leave an
        # empty parquet behind for them.
        os.remove(out_path)
        print(f"[timeseries] {session}: no trial windows, nothing written", flush=True)
        return

    print(f"[timeseries] {session}: {total_rows:,} rows @ {rate_hz:g} Hz -> {out_path} "
          f"({os.path.getsize(out_path) / 1e6:,.1f} MB)", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", action="append", help="session name (repeatable); default = all")
    parser.add_argument("--rate-hz", type=float, default=50.0,
                        help="grid rate; 50 Hz sits above the 40 Hz IMU packet rate without "
                             "inventing detail the sensors never resolved (default: 50)")
    args = parser.parse_args(argv)

    if args.session:
        session_dirs = [os.path.join(PARQUET_DIR, name) for name in args.session]
    else:
        session_dirs = sorted(p for p in glob.glob(os.path.join(PARQUET_DIR, "*")) if os.path.isdir(p))

    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            print(f"error: no such session: {session_dir}", file=sys.stderr)
            return 1
        build_session(session_dir, args.rate_hz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Collapse the per-sample Parquet working copy into one row per trial.

A *trial* is one settled measurement: a contiguous run of records whose
`experiment_state` is `MEASUREMENT_WINDOW`. The record-level `trial_id` column
is NOT that -- it is the set index (0..11), so a plan of 50 targets x 3 repeats
x 12 sets shares one `trial_id` across 150 measurements. Trials are therefore
cut from state transitions on the Vive stream (the 60 Hz reference), and the
encoder / IMU samples are attached to a trial by their host clock falling
inside that trial's `t_pc_rx_ns` span.

Clock note: `t_pc_rx_ns` is monotonic *per segment* (each segment is a fresh
process), so intervals are only ever matched within one segment. `utc_time` is
carried through for ordering across segments.

Output: `derived/parquet/<session>/trials.parquet`
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
JOINTS = ["j0", "j1", "j2"]
IMU_SENSORS = ["imu0_base", "imu1_link_root", "imu2_link_mid", "imu3_tip"]

# Minimum samples for a trial to be usable at all. The Vive runs at 60 Hz into a
# 2 s window, so a healthy trial holds ~120 samples; anything under 30 is a
# window that was cut short by a fault or a segment ending mid-trial.
MIN_VIVE_SAMPLES = 30

VIVE_COLUMNS = [
    "segment_id", "host_seq", "experiment_state", "t_pc_rx_ns", "utc_time",
    "trial_id", "target_id", "planned_order_index", "tracking_valid",
    "position_m_x", "position_m_y", "position_m_z",
    "quaternion_xyzw_x", "quaternion_xyzw_y", "quaternion_xyzw_z", "quaternion_xyzw_w",
    "planned_p_xyz_mm_x", "planned_p_xyz_mm_y", "planned_p_xyz_mm_z",
]
ENCODER_COLUMNS = (
    ["segment_id", "t_pc_rx_ns"]
    + [f"encoder_raw_tick_{j}" for j in JOINTS]
    + [f"q_encoder_rad_{j}" for j in JOINTS]
    + [f"firmware_logical_deg_{j}" for j in JOINTS]
)
IMU_COLUMNS = [
    "segment_id", "t_pc_rx_ns", "sensor_id", "packet_type",
    "euler_rad_x", "euler_rad_y", "euler_rad_z",
    "quaternion_xyzw_x", "quaternion_xyzw_y", "quaternion_xyzw_z", "quaternion_xyzw_w",
    "gyro_rads_x", "gyro_rads_y", "gyro_rads_z",
    "accel_mps2_x", "accel_mps2_y", "accel_mps2_z",
    "mag_raw_x", "mag_raw_y", "mag_raw_z",
]


def mean_quaternion(frame: pd.DataFrame, prefix: str = "quaternion_xyzw") -> np.ndarray:
    """Average a near-identical quaternion cluster.

    q and -q are the same rotation, so signs are flipped to agree with the first
    sample before averaging; the result is renormalised. This is the cheap
    small-angle average, which is what a settled pose is -- it is NOT valid for
    widely separated rotations.
    """
    values = frame[[f"{prefix}_{axis}" for axis in "xyzw"]].to_numpy(dtype=float)
    values = values[~np.isnan(values).any(axis=1)]
    if len(values) == 0:
        return np.full(4, np.nan)
    values = values * np.sign(values @ values[0])[:, None]
    mean = values.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else np.full(4, np.nan)


def cut_trials(vive: pd.DataFrame) -> pd.DataFrame:
    """One row per contiguous MEASUREMENT_WINDOW run, with its host-clock span."""
    vive = vive.sort_values(["segment_id", "host_seq"], kind="stable").reset_index(drop=True)
    state = vive["experiment_state"].fillna("__none__")
    # A new block starts wherever the state changes or the segment changes.
    boundary = (state != state.shift()) | (vive["segment_id"] != vive["segment_id"].shift())
    vive["block"] = boundary.cumsum()
    return vive[state == SETTLED]


def summarise_vive(settled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block, group in settled.groupby("block", sort=True):
        quaternion = mean_quaternion(group)
        rows.append({
            "segment_id": int(group["segment_id"].iloc[0]),
            "block": int(block),
            # named set_id, not set_index -- `df.set_index` is a DataFrame method,
            # so that column name would be unreachable by attribute access.
            "set_id": group["trial_id"].iloc[0],
            "target_id": group["target_id"].iloc[0],
            "planned_order_index": group["planned_order_index"].iloc[0],
            "t_start_ns": int(group["t_pc_rx_ns"].min()),
            "t_end_ns": int(group["t_pc_rx_ns"].max()),
            "utc_start": group["utc_time"].min(),
            "duration_s": (group["t_pc_rx_ns"].max() - group["t_pc_rx_ns"].min()) / 1e9,
            "vive_n": len(group),
            "vive_tracking_valid_ratio": float(group["tracking_valid"].fillna(False).mean()),
            "vive_pos_x_m": group["position_m_x"].mean(),
            "vive_pos_y_m": group["position_m_y"].mean(),
            "vive_pos_z_m": group["position_m_z"].mean(),
            "vive_pos_x_std_m": group["position_m_x"].std(),
            "vive_pos_y_std_m": group["position_m_y"].std(),
            "vive_pos_z_std_m": group["position_m_z"].std(),
            "vive_quat_x": quaternion[0],
            "vive_quat_y": quaternion[1],
            "vive_quat_z": quaternion[2],
            "vive_quat_w": quaternion[3],
            "planned_p_x_mm": group["planned_p_xyz_mm_x"].iloc[0],
            "planned_p_y_mm": group["planned_p_xyz_mm_y"].iloc[0],
            "planned_p_z_mm": group["planned_p_xyz_mm_z"].iloc[0],
        })
    return pd.DataFrame(rows)


def assign_blocks(frame: pd.DataFrame, trials: pd.DataFrame) -> pd.Series:
    """Map each sample to the trial whose host-clock span contains it.

    Trials never overlap and are sorted, so a single searchsorted on the start
    edges finds the candidate; the end edge then rejects samples that fell in
    the gap between two trials.
    """
    starts = trials["t_start_ns"].to_numpy()
    ends = trials["t_end_ns"].to_numpy()
    blocks = trials["block"].to_numpy()

    times = frame["t_pc_rx_ns"].to_numpy()
    index = np.searchsorted(starts, times, side="right") - 1
    inside = (index >= 0) & (times <= ends[np.clip(index, 0, len(ends) - 1)])
    result = np.where(inside, blocks[np.clip(index, 0, len(blocks) - 1)], -1)
    return pd.Series(result, index=frame.index)


def summarise_encoder(encoder: pd.DataFrame) -> pd.DataFrame:
    aggregations = {"encoder_n": ("t_pc_rx_ns", "size")}
    for joint in JOINTS:
        aggregations[f"encoder_tick_{joint}"] = (f"encoder_raw_tick_{joint}", "mean")
        aggregations[f"q_encoder_rad_{joint}"] = (f"q_encoder_rad_{joint}", "mean")
        aggregations[f"logical_deg_{joint}"] = (f"firmware_logical_deg_{joint}", "mean")
        # A joint that moved during the window means the trial was not settled.
        aggregations[f"encoder_tick_{joint}_std"] = (f"encoder_raw_tick_{joint}", "std")
    return encoder.groupby("block").agg(**aggregations)


def summarise_imu(imu: pd.DataFrame) -> pd.DataFrame:
    """Per sensor: settled orientation, plus gyro magnitude as a motion check.

    Which packet types exist depends on how the WT901C was configured for that
    run -- the rigs emit acceleration/angular_velocity/angle plus EITHER
    magnetic_field or quaternion, never both. Absent types are skipped rather
    than emitted as all-NaN columns.
    """
    per_sensor = []
    for sensor in IMU_SENSORS:
        subset = imu[imu["sensor_id"] == sensor]
        if subset.empty:
            continue

        parts = []

        angle = subset[subset["packet_type"] == "angle"]
        if not angle.empty:
            euler = angle.groupby("block")[["euler_rad_x", "euler_rad_y", "euler_rad_z"]].mean()
            euler.columns = [f"{sensor}_euler_{axis}_rad" for axis in "xyz"]
            parts.append(euler)

        quaternion_rows = subset[subset["packet_type"] == "quaternion"]
        if not quaternion_rows.empty:
            quaternions = quaternion_rows.groupby("block").apply(mean_quaternion, include_groups=False)
            parts.append(pd.DataFrame(
                quaternions.tolist(), index=quaternions.index,
                columns=[f"{sensor}_quat_{axis}" for axis in "xyzw"],
            ))

        # Mean specific force at rest is the gravity direction, so it is an
        # independent check on the angle packet's tilt.
        accel = subset[subset["packet_type"] == "acceleration"]
        if not accel.empty:
            means = accel.groupby("block")[["accel_mps2_x", "accel_mps2_y", "accel_mps2_z"]].mean()
            means.columns = [f"{sensor}_accel_{axis}_mps2" for axis in "xyz"]
            parts.append(means)

        # The collector never filled mag_uT -- no scale factor was applied at
        # capture time -- so the magnetometer is carried as raw counts.
        magnetic = subset[subset["packet_type"] == "magnetic_field"]
        if not magnetic.empty:
            means = magnetic.groupby("block")[["mag_raw_x", "mag_raw_y", "mag_raw_z"]].mean()
            means.columns = [f"{sensor}_mag_{axis}_raw" for axis in "xyz"]
            parts.append(means)

        gyro_rows = subset[subset["packet_type"] == "angular_velocity"].copy()
        gyro_rows["_mag"] = np.sqrt(
            gyro_rows["gyro_rads_x"] ** 2 + gyro_rows["gyro_rads_y"] ** 2 + gyro_rows["gyro_rads_z"] ** 2
        )
        if not gyro_rows.empty:
            parts.append(gyro_rows.groupby("block").agg(**{
                f"{sensor}_gyro_rms_rad_s": ("_mag", lambda s: float(np.sqrt((s ** 2).mean()))),
                f"{sensor}_n": ("_mag", "size"),
            }))

        if parts:
            per_sensor.append(pd.concat(parts, axis=1))

    if not per_sensor:
        return pd.DataFrame()
    return pd.concat(per_sensor, axis=1)


def build_session(session_dir: str) -> None:
    session = os.path.basename(session_dir.rstrip(os.sep))
    vive_path = os.path.join(session_dir, "vive_tracker.parquet")
    if not os.path.exists(vive_path):
        print(f"[trials] {session}: no vive_tracker.parquet, skipped", flush=True)
        return

    vive = ds.dataset(vive_path).to_table(columns=VIVE_COLUMNS).to_pandas()
    settled = cut_trials(vive)
    if settled.empty:
        print(f"[trials] {session}: no {SETTLED} samples, skipped", flush=True)
        return

    trials = summarise_vive(settled).sort_values("t_start_ns", kind="stable").reset_index(drop=True)

    # repeat_index distinguishes the N repeats of the same planned target within
    # one set -- the plan says repeats_per_point: 3, and nothing in the record
    # itself numbers them.
    trials["repeat_index"] = trials.groupby(
        ["segment_id", "set_id", "target_id"], dropna=False
    ).cumcount()

    for name, columns, summarise in (
        ("encoder", ENCODER_COLUMNS, summarise_encoder),
        ("imu", IMU_COLUMNS, summarise_imu),
    ):
        path = os.path.join(session_dir, f"{name}.parquet")
        if not os.path.exists(path):
            continue
        frame = ds.dataset(path).to_table(
            columns=columns, filter=ds.field("experiment_state") == SETTLED
        ).to_pandas()
        # Trials are cut per segment, and the host clock restarts each segment,
        # so interval matching has to happen inside a segment.
        assigned = []
        for segment_id, group in frame.groupby("segment_id"):
            segment_trials = trials[trials["segment_id"] == segment_id]
            if segment_trials.empty:
                continue
            group = group.assign(block=assign_blocks(group, segment_trials))
            assigned.append(group[group["block"] >= 0])
        if not assigned:
            continue
        summary = summarise(pd.concat(assigned))
        trials = trials.merge(summary, how="left", left_on="block", right_index=True)

    trials["keep"] = (
        (trials["vive_n"] >= MIN_VIVE_SAMPLES)
        & (trials["vive_tracking_valid_ratio"] == 1.0)
        & trials["vive_pos_x_m"].notna()
        & trials.get("encoder_n", pd.Series(np.nan, index=trials.index)).notna()
    )

    out_path = os.path.join(session_dir, "trials.parquet")
    table = pa.Table.from_pandas(trials, preserve_index=False)
    table = table.replace_schema_metadata({
        "session": session,
        "trial_definition": f"contiguous run of experiment_state == {SETTLED} on the vive stream",
        "joint_order": "shoulder_pan,shoulder_lift,wrist_flex",
        "min_vive_samples": str(MIN_VIVE_SAMPLES),
    })
    pq.write_table(table, out_path, compression="zstd")

    kept = int(trials["keep"].sum())
    print(f"[trials] {session}: {len(trials):,} trials ({kept:,} keep / {len(trials) - kept:,} drop) "
          f"-> {out_path} ({os.path.getsize(out_path) / 1e6:,.1f} MB)", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", action="append", help="session name (repeatable); default = all")
    args = parser.parse_args(argv)

    if args.session:
        session_dirs = [os.path.join(PARQUET_DIR, name) for name in args.session]
    else:
        session_dirs = sorted(p for p in glob.glob(os.path.join(PARQUET_DIR, "*")) if os.path.isdir(p))

    for session_dir in session_dirs:
        if not os.path.isdir(session_dir):
            print(f"error: no such session: {session_dir}", file=sys.stderr)
            return 1
        build_session(session_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

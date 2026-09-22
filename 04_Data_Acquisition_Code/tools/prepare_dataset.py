"""Turn raw acquisition sessions into an archive + an analysis-ready working copy.

Three phases, run in order (`--phase all` by default):

1. `recover` -- every `*.jsonl.partial` left behind by a crashed segment is run
   through `core.recovery.recover_partial_jsonl`, producing `*.recovered.jsonl`
   next to it. Originals are never modified.
2. `parquet` -- all JSONL for a session is flattened into one Parquet file per
   stream under `derived/parquet/<session>/<stream>.parquet`. The `data` dict is
   expanded into scalar columns, vectors are split per component, and the raw
   passthrough fields (`raw_hex`, `raw_line`, `pose_matrix_3x4`) are dropped --
   they live on in the gzip archive, and the working copy is for analysis.
3. `gzip` -- every raw `*.jsonl` / `*.jsonl.partial` is compressed in place to
   `.gz`. JSONL stays streamable while compressed, so the archive is still
   directly readable via `gzip.open(path, "rt")`.

Timestamps: `t_pc_rx_ns` is the monotonic host clock (use this to align streams
within one segment); `utc_time` is the wall clock, useful across segments.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import multiprocessing
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.recovery import recover_partial_jsonl

SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions")
PARQUET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "derived", "parquet")

BATCH_ROWS = 200_000

# The SO-101 arm exposes three driven joints; every per-joint vector below is in
# this order (confirmed against the encoder `raw_line` axis= fields).
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "wrist_flex"]
JOINT_SUFFIX = ["j0", "j1", "j2"]
XYZ = ["x", "y", "z"]
XYZW = ["x", "y", "z", "w"]

# Present on every record regardless of stream.
COMMON_SCALARS = {
    "segment_id": "i4",
    "sensor_id": "str",
    "host_seq": "i8",
    "trial_id": "i4",
    "target_id": "str",
    "planned_order_index": "i4",
    "experiment_state": "str",
    "t_pc_rx_ns": "i8",
    "valid": "bool",
    "parse_status": "str",
    "checksum_ok": "bool",
    "duplicate": "bool",
    "out_of_order": "bool",
    "late_by_ms": "f8",
}

# Per stream: extra top-level scalars, `data` scalars, and `data` vectors
# (field -> (component suffixes, dtype)).
STREAM_SPEC = {
    "imu": {
        "top": {
            "t_pc_first_byte_ns": "i8",
            "t_pc_frame_end_ns": "i8",
            "packet_type": "str",
            "frame_length": "i4",
            "connection_epoch": "i4",
        },
        "data": {
            "temperature_c": "f8",
            "mag_packet_word4_raw": "f8",
            "angle_packet_word4_raw": "f8",
        },
        "vec": {
            "accel_raw": (XYZ, "f8"),
            "gyro_raw": (XYZ, "f8"),
            "mag_raw": (XYZ, "f8"),
            "accel_mps2": (XYZ, "f8"),
            "gyro_rads": (XYZ, "f8"),
            "mag_uT": (XYZ, "f8"),
            "euler_rad": (XYZ, "f8"),
            "quaternion_xyzw": (XYZW, "f8"),
            "planned_p_xyz_mm": (XYZ, "f8"),
        },
    },
    "vive_tracker": {
        "top": {"frame_id": "str"},
        "data": {
            "tracking_result": "str",
            "tracking_valid": "bool",
            "device_connected": "bool",
            "device_index": "i4",
            "source_timestamp": "f8",
        },
        "vec": {
            "position_m": (XYZ, "f8"),
            "quaternion_xyzw": (XYZW, "f8"),
            "planned_p_xyz_mm": (XYZ, "f8"),
            "planned_raw_ticks": (JOINT_SUFFIX, "f8"),
        },
    },
    "encoder": {
        "top": {"t_esp_us": "i8"},
        "data": {
            "esp32_state": "str",
            "command_ack": "str",
            "watchdog_state": "str",
        },
        "vec": {
            "encoder_raw_tick": (JOINT_SUFFIX, "f8"),
            "q_encoder_rad": (JOINT_SUFFIX, "f8"),
            "q_cmd_raw": (JOINT_SUFFIX, "f8"),
            "q_cmd_rad": (JOINT_SUFFIX, "f8"),
            "estimated_velocity_rad_s": (JOINT_SUFFIX, "f8"),
            "firmware_logical_deg": (JOINT_SUFFIX, "f8"),
            "temperature_c": (JOINT_SUFFIX, "f8"),
            "current_or_load": (JOINT_SUFFIX, "f8"),
            "motor_status": (JOINT_SUFFIX, "str"),
            "fault": (JOINT_SUFFIX, "str"),
            "planned_p_xyz_mm": (XYZ, "f8"),
            "planned_raw_ticks": (JOINT_SUFFIX, "f8"),
        },
    },
    # events/health are low-rate diagnostics whose payloads are per-sensor dicts
    # with no fixed key set -- not worth a column spec, so they keep their
    # payload as one JSON string column.
    "events": {"top": {"event_type": "str", "message": "str"}, "data": {}, "vec": {}, "json": ["details"]},
    "health": {"top": {}, "data": {}, "vec": {}, "json": ["data"]},
}

ARROW = {
    "i4": pa.int32(),
    "i8": pa.int64(),
    "f8": pa.float64(),
    "bool": pa.bool_(),
    "str": pa.string(),
}


def stream_of(filename: str) -> str:
    """`imu3_tip_segment012.jsonl` -> `imu`, `vive_tracker_segment000.jsonl` -> `vive_tracker`."""
    base = os.path.basename(filename)
    for name in ("vive_tracker", "encoder", "events", "health"):
        if base.startswith(name):
            return name
    if base.startswith("imu"):
        return "imu"
    return base.split("_segment")[0]


def columns_for(stream: str) -> list[tuple[str, str]]:
    """Ordered (column_name, dtype_code) for a stream's flattened table."""
    spec = STREAM_SPEC[stream]
    cols = [("utc_time", "ts")]
    cols += list(COMMON_SCALARS.items())
    cols += [("quality_flags", "str")]
    cols += list(spec["top"].items())
    cols += list(spec["data"].items())
    for field, (suffixes, dtype) in spec["vec"].items():
        cols += [(f"{field}_{s}", dtype) for s in suffixes]
    cols += [(f"{field}_json", "str") for field in spec.get("json", [])]
    return cols


def schema_for(stream: str, metadata: dict[str, str]) -> pa.Schema:
    fields = [
        pa.field(name, pa.timestamp("ns", tz="UTC") if code == "ts" else ARROW[code])
        for name, code in columns_for(stream)
    ]
    return pa.schema(fields, metadata=metadata)


def flatten_batch(records: list[dict], stream: str) -> dict[str, list]:
    """One pass over a batch of parsed records -> column-major dict of lists."""
    spec = STREAM_SPEC[stream]
    out: dict[str, list] = {name: [] for name, _ in columns_for(stream)}

    for rec in records:
        out["utc_time"].append(rec.get("utc_iso"))
        for key in COMMON_SCALARS:
            value = rec.get(key)
            # target_id shows up as int or str depending on plan version; the
            # column is declared string so normalise here rather than let arrow
            # reject the batch.
            out[key].append(str(value) if key == "target_id" and value is not None else value)

        flags = rec.get("quality_flags") or []
        out["quality_flags"].append(";".join(flags) if flags else None)

        for key in spec["top"]:
            out[key].append(rec.get(key))

        data = rec.get("data") or {}
        for key in spec["data"]:
            out[key].append(data.get(key))

        for field, (suffixes, _) in spec["vec"].items():
            value = data.get(field)
            if not isinstance(value, list):
                value = []
            for index, suffix in enumerate(suffixes):
                out[f"{field}_{suffix}"].append(value[index] if index < len(value) else None)

        for field in spec.get("json", []):
            value = rec.get(field)
            out[f"{field}_json"].append(json.dumps(value, ensure_ascii=False) if value else None)

    return out


def to_table(columns: dict[str, list], schema: pa.Schema) -> pa.Table:
    columns = dict(columns)
    # Vectorised ISO-8601 parse; `format="ISO8601"` avoids pandas guessing per row.
    columns["utc_time"] = pd.to_datetime(pd.Series(columns["utc_time"], dtype="object"),
                                         format="ISO8601", utc=True)
    arrays = [pa.array(columns[field.name], type=field.type, from_pandas=True) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def jsonl_inputs(session_dir: str) -> dict[str, list[str]]:
    """stream -> segment files, in segment order. `.recovered.jsonl` is included;
    the `.partial` it came from is not, so records are never counted twice.
    Already-archived `.jsonl.gz` counts too, so the parquet phase works whether
    or not the gzip phase has run."""
    paths = glob.glob(os.path.join(session_dir, "**", "*.jsonl"), recursive=True)
    paths += glob.glob(os.path.join(session_dir, "**", "*.jsonl.gz"), recursive=True)
    by_stream: dict[str, list[str]] = {}
    for path in sorted(paths):
        by_stream.setdefault(stream_of(path), []).append(path)
    return by_stream


def phase_recover(session_dirs: list[str]) -> None:
    partials = []
    for session_dir in session_dirs:
        partials += sorted(glob.glob(os.path.join(session_dir, "**", "*.jsonl.partial"), recursive=True))
    print(f"[recover] {len(partials)} partial file(s)", flush=True)

    total_valid = total_dropped = 0
    for index, path in enumerate(partials, 1):
        recovered = os.path.splitext(os.path.splitext(path)[0])[0] + ".recovered.jsonl"
        if os.path.exists(recovered):
            print(f"[recover] {index}/{len(partials)} skip (already recovered) {os.path.basename(path)}", flush=True)
            continue
        result = recover_partial_jsonl(path)
        total_valid += result.valid_record_count
        total_dropped += len(result.dropped_lines)
        print(f"[recover] {index}/{len(partials)} {os.path.basename(path)}: "
              f"{result.valid_record_count} records, {len(result.dropped_lines)} fragment(s) dropped, "
              f"{len(result.seq_gaps)} seq gap(s)", flush=True)
    print(f"[recover] done: {total_valid} records recovered, {total_dropped} tail fragment(s) dropped", flush=True)


def phase_parquet(session_dirs: list[str]) -> None:
    for session_dir in session_dirs:
        session = os.path.basename(session_dir.rstrip(os.sep))
        out_dir = os.path.join(PARQUET_DIR, session)
        os.makedirs(out_dir, exist_ok=True)

        for stream, paths in jsonl_inputs(session_dir).items():
            if stream not in STREAM_SPEC:
                print(f"[parquet] {session}/{stream}: no spec, skipped", flush=True)
                continue

            out_path = os.path.join(out_dir, f"{stream}.parquet")
            metadata = {
                "session": session,
                "stream": stream,
                "joint_order": ",".join(JOINT_ORDER),
                "source_files": str(len(paths)),
                "note": "raw_hex/raw_line/pose_matrix_3x4 dropped; see the .jsonl.gz archive for those",
            }
            schema = schema_for(stream, metadata)
            started = time.time()
            rows = bad = 0

            with pq.ParquetWriter(out_path, schema, compression="zstd") as writer:
                batch: list[dict] = []
                for path in paths:
                    opener = gzip.open if path.endswith(".gz") else open
                    with opener(path, "rt", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                batch.append(json.loads(line))
                            except json.JSONDecodeError:
                                bad += 1
                                continue
                            if len(batch) >= BATCH_ROWS:
                                writer.write_table(to_table(flatten_batch(batch, stream), schema))
                                rows += len(batch)
                                batch = []
                if batch:
                    writer.write_table(to_table(flatten_batch(batch, stream), schema))
                    rows += len(batch)

            size_mb = os.path.getsize(out_path) / 1e6
            print(f"[parquet] {session}/{stream}: {rows:,} rows from {len(paths)} file(s) -> "
                  f"{size_mb:,.1f} MB in {time.time() - started:,.0f}s"
                  + (f" ({bad} unparseable line(s) skipped)" if bad else ""), flush=True)


def gzip_one(path: str) -> tuple[str, str | None]:
    """Compress one file to `<path>.gz`. The source is unlinked only after the
    compressed copy is fully written and closed, so an interrupted run leaves
    the original intact and simply re-does that file next time."""
    out_path = path + ".gz"
    try:
        with open(path, "rb") as src, gzip.open(out_path, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst, length=8 << 20)
        os.remove(path)
        return path, None
    except Exception as exc:  # noqa: BLE001 -- report and keep the rest going
        if os.path.exists(out_path):
            os.remove(out_path)
        return path, repr(exc)


def phase_gzip(session_dirs: list[str], jobs: int) -> None:
    targets = []
    for session_dir in session_dirs:
        for pattern in ("*.jsonl", "*.jsonl.partial"):
            targets += glob.glob(os.path.join(session_dir, "**", pattern), recursive=True)
    targets = sorted(path for path in targets if not os.path.exists(path + ".gz"))

    total_bytes = sum(os.path.getsize(path) for path in targets)
    print(f"[gzip] {len(targets)} file(s), {total_bytes / 1e9:,.1f} GB, {jobs} parallel job(s)", flush=True)
    if not targets:
        return

    started = time.time()
    done = 0
    with multiprocessing.Pool(jobs) as pool:
        for path, error in pool.imap_unordered(gzip_one, targets):
            done += 1
            if error:
                print(f"[gzip] FAILED {os.path.basename(path)}: {error}", flush=True)
            elif done % 25 == 0 or done == len(targets):
                print(f"[gzip] {done}/{len(targets)}", flush=True)

    after = sum(os.path.getsize(path + ".gz") for path in targets if os.path.exists(path + ".gz"))
    print(f"[gzip] done: {total_bytes / 1e9:,.1f} GB -> {after / 1e9:,.1f} GB "
          f"({after / total_bytes:.1%}) in {time.time() - started:,.0f}s", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", default="all", choices=["all", "recover", "parquet", "gzip"])
    parser.add_argument("--session", action="append", help="session dir name (repeatable); default = all")
    parser.add_argument("--jobs", type=int, default=4, help="parallel gzip processes")
    args = parser.parse_args(argv)

    if args.session:
        session_dirs = [os.path.join(SESSIONS_DIR, name) for name in args.session]
    else:
        session_dirs = sorted(path for path in glob.glob(os.path.join(SESSIONS_DIR, "*")) if os.path.isdir(path))
    missing = [path for path in session_dirs if not os.path.isdir(path)]
    if missing:
        print(f"error: no such session dir: {missing}", file=sys.stderr)
        return 1
    print(f"sessions: {[os.path.basename(p) for p in session_dirs]}", flush=True)

    if args.phase in ("all", "recover"):
        phase_recover(session_dirs)
    if args.phase in ("all", "parquet"):
        phase_parquet(session_dirs)
    if args.phase in ("all", "gzip"):
        phase_gzip(session_dirs, args.jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fail-closed loading of the versioned trial-level source archive."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re
from typing import Any, Mapping
from zipfile import BadZipFile, ZipFile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class SourceContractError(ValueError):
    """Raised when the immutable source does not match its declared contract."""


_LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _field(name: str, data_type: pa.DataType) -> pa.Field:
    return pa.field(name, data_type, nullable=True)


# Keep this explicit and ordered. A source producer changing a name, type, order,
# or time-zone is a public-contract change and must not pass unnoticed.
EXPECTED_TRIALS_SCHEMA = pa.schema(
    [
        _field("segment_id", pa.int64()),
        _field("block", pa.int64()),
        _field("set_id", pa.float64()),
        _field("target_id", pa.string()),
        _field("planned_order_index", pa.float64()),
        _field("t_start_ns", pa.int64()),
        _field("t_end_ns", pa.int64()),
        _field("utc_start", pa.timestamp("ns", tz="UTC")),
        _field("duration_s", pa.float64()),
        _field("vive_n", pa.int64()),
        _field("vive_tracking_valid_ratio", pa.float64()),
        _field("vive_pos_x_m", pa.float64()),
        _field("vive_pos_y_m", pa.float64()),
        _field("vive_pos_z_m", pa.float64()),
        _field("vive_pos_x_std_m", pa.float64()),
        _field("vive_pos_y_std_m", pa.float64()),
        _field("vive_pos_z_std_m", pa.float64()),
        _field("vive_quat_x", pa.float64()),
        _field("vive_quat_y", pa.float64()),
        _field("vive_quat_z", pa.float64()),
        _field("vive_quat_w", pa.float64()),
        _field("planned_p_x_mm", pa.float64()),
        _field("planned_p_y_mm", pa.float64()),
        _field("planned_p_z_mm", pa.float64()),
        _field("repeat_index", pa.int64()),
        _field("encoder_n", pa.float64()),
        _field("encoder_tick_j0", pa.float64()),
        _field("q_encoder_rad_j0", pa.float64()),
        _field("logical_deg_j0", pa.float64()),
        _field("encoder_tick_j0_std", pa.float64()),
        _field("encoder_tick_j1", pa.float64()),
        _field("q_encoder_rad_j1", pa.float64()),
        _field("logical_deg_j1", pa.float64()),
        _field("encoder_tick_j1_std", pa.float64()),
        _field("encoder_tick_j2", pa.float64()),
        _field("q_encoder_rad_j2", pa.float64()),
        _field("logical_deg_j2", pa.float64()),
        _field("encoder_tick_j2_std", pa.float64()),
        _field("imu0_base_euler_x_rad", pa.float64()),
        _field("imu0_base_euler_y_rad", pa.float64()),
        _field("imu0_base_euler_z_rad", pa.float64()),
        _field("imu0_base_accel_x_mps2", pa.float64()),
        _field("imu0_base_accel_y_mps2", pa.float64()),
        _field("imu0_base_accel_z_mps2", pa.float64()),
        _field("imu0_base_mag_x_raw", pa.float64()),
        _field("imu0_base_mag_y_raw", pa.float64()),
        _field("imu0_base_mag_z_raw", pa.float64()),
        _field("imu0_base_gyro_rms_rad_s", pa.float64()),
        _field("imu0_base_n", pa.float64()),
        _field("imu1_link_root_euler_x_rad", pa.float64()),
        _field("imu1_link_root_euler_y_rad", pa.float64()),
        _field("imu1_link_root_euler_z_rad", pa.float64()),
        _field("imu1_link_root_accel_x_mps2", pa.float64()),
        _field("imu1_link_root_accel_y_mps2", pa.float64()),
        _field("imu1_link_root_accel_z_mps2", pa.float64()),
        _field("imu1_link_root_mag_x_raw", pa.float64()),
        _field("imu1_link_root_mag_y_raw", pa.float64()),
        _field("imu1_link_root_mag_z_raw", pa.float64()),
        _field("imu1_link_root_gyro_rms_rad_s", pa.float64()),
        _field("imu1_link_root_n", pa.float64()),
        _field("imu2_link_mid_euler_x_rad", pa.float64()),
        _field("imu2_link_mid_euler_y_rad", pa.float64()),
        _field("imu2_link_mid_euler_z_rad", pa.float64()),
        _field("imu2_link_mid_accel_x_mps2", pa.float64()),
        _field("imu2_link_mid_accel_y_mps2", pa.float64()),
        _field("imu2_link_mid_accel_z_mps2", pa.float64()),
        _field("imu2_link_mid_mag_x_raw", pa.float64()),
        _field("imu2_link_mid_mag_y_raw", pa.float64()),
        _field("imu2_link_mid_mag_z_raw", pa.float64()),
        _field("imu2_link_mid_gyro_rms_rad_s", pa.float64()),
        _field("imu2_link_mid_n", pa.float64()),
        _field("imu3_tip_euler_x_rad", pa.float64()),
        _field("imu3_tip_euler_y_rad", pa.float64()),
        _field("imu3_tip_euler_z_rad", pa.float64()),
        _field("imu3_tip_accel_x_mps2", pa.float64()),
        _field("imu3_tip_accel_y_mps2", pa.float64()),
        _field("imu3_tip_accel_z_mps2", pa.float64()),
        _field("imu3_tip_mag_x_raw", pa.float64()),
        _field("imu3_tip_mag_y_raw", pa.float64()),
        _field("imu3_tip_mag_z_raw", pa.float64()),
        _field("imu3_tip_gyro_rms_rad_s", pa.float64()),
        _field("imu3_tip_n", pa.float64()),
        _field("keep", pa.bool_()),
    ]
)


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_sha256(value: Any, field_name: str) -> str:
    text = str(value)
    if not _SHA256_RE.fullmatch(text):
        raise SourceContractError(f"{field_name} must be a 64-character SHA-256 hex digest")
    return text.lower()


def _assert_materialized(path: Path) -> None:
    if not path.is_file():
        raise SourceContractError(f"source archive does not exist: {path}")
    with path.open("rb") as stream:
        header = stream.read(128)
    if header.startswith(_LFS_POINTER_HEADER):
        raise SourceContractError(
            f"source archive is an unmaterialized Git LFS pointer: {path}; run git lfs pull"
        )


def _validate_schema(actual: pa.Schema) -> None:
    actual = actual.remove_metadata()
    if actual.equals(EXPECTED_TRIALS_SCHEMA, check_metadata=False):
        return

    expected_fields = [(field.name, str(field.type)) for field in EXPECTED_TRIALS_SCHEMA]
    actual_fields = [(field.name, str(field.type)) for field in actual]
    raise SourceContractError(
        "trials.parquet Arrow schema mismatch; "
        f"expected {expected_fields!r}, received {actual_fields!r}"
    )


def load_trials(
    archive_path: str | Path,
    *,
    member: str,
    expected_archive_sha256: str,
    expected_member_sha256: str,
    expected_rows: int,
) -> pd.DataFrame:
    """Load the declared Parquet member only after all source checks pass."""

    path = Path(archive_path).resolve()
    _assert_materialized(path)

    declared_archive_hash = _declared_sha256(
        expected_archive_sha256, "expected_archive_sha256"
    )
    actual_archive_hash = sha256_file(path)
    if actual_archive_hash != declared_archive_hash:
        raise SourceContractError(
            "source archive SHA-256 mismatch: "
            f"expected {declared_archive_hash}, received {actual_archive_hash}"
        )

    try:
        with ZipFile(path) as archive:
            matches = [entry for entry in archive.infolist() if entry.filename == member]
            if len(matches) != 1:
                raise SourceContractError(
                    f"source archive must contain exactly one {member!r} member; found {len(matches)}"
                )
            raw_member = archive.read(matches[0])
    except BadZipFile as exc:
        raise SourceContractError(f"source archive is not a valid ZIP file: {path}") from exc

    if raw_member.startswith(_LFS_POINTER_HEADER):
        raise SourceContractError(f"archive member {member!r} is a Git LFS pointer")

    declared_member_hash = _declared_sha256(
        expected_member_sha256, "expected_member_sha256"
    )
    actual_member_hash = sha256(raw_member).hexdigest()
    if actual_member_hash != declared_member_hash:
        raise SourceContractError(
            "source member SHA-256 mismatch: "
            f"expected {declared_member_hash}, received {actual_member_hash}"
        )

    try:
        table = pq.read_table(pa.BufferReader(raw_member))
    except (pa.ArrowException, OSError) as exc:
        raise SourceContractError(f"source member {member!r} is not readable Parquet") from exc

    _validate_schema(table.schema)
    if table.num_rows != int(expected_rows):
        raise SourceContractError(
            f"source row-count mismatch: expected {int(expected_rows)}, received {table.num_rows}"
        )

    frame = table.to_pandas()
    frame.attrs.update(
        {
            "source_archive_path": str(path),
            "source_archive_sha256": actual_archive_hash,
            "source_member": member,
            "source_member_sha256": actual_member_hash,
        }
    )
    return frame


def load_trials_from_config(
    source_config: Mapping[str, Any], *, base_dir: str | Path = "."
) -> pd.DataFrame:
    """Resolve a source mapping relative to the workspace and load it."""

    required = {
        "archive",
        "member",
        "archive_sha256",
        "member_sha256",
        "expected_rows",
    }
    missing = sorted(required.difference(source_config))
    if missing:
        raise SourceContractError(f"source config is missing required fields: {missing}")

    archive_path = Path(str(source_config["archive"]))
    if not archive_path.is_absolute():
        archive_path = Path(base_dir) / archive_path
    return load_trials(
        archive_path,
        member=str(source_config["member"]),
        expected_archive_sha256=str(source_config["archive_sha256"]),
        expected_member_sha256=str(source_config["member_sha256"]),
        expected_rows=int(source_config["expected_rows"]),
    )

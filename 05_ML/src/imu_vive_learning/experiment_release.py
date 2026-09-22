"""Fail-closed reader for the immutable ML-preparation release.

The endpoint-regression experiment intentionally consumes the packaged
``alphaf.ml_preparation.v1`` artifact rather than reaching back into the 08
working directories.  This module validates the outer ZIP, its signed-by-hash
member index, Arrow schemas, and the preparation contract before exposing any
data frame to model code.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from zipfile import BadZipFile, ZipFile

import pandas as pd
import pyarrow.parquet as pq

from .features import LABEL_COLUMNS, validate_feature_views
from .io_utils import sha256_file
from .splitting import validate_split_assignments


PREPARATION_CONTRACT = "alphaf.ml_preparation.v1"
PREPARATION_PACKAGE_ROOT = "ml_preparation_v1"
PINNED_RELEASE_SHA256 = (
    "94deccb960b0554fb9a683875d36ef5726afc825d4b4ea244a1490f3353b1930"
)
PINNED_CANONICAL_ROWS = 7_707
PINNED_ELIGIBLE_ROWS = 7_702
PINNED_STRICT_QC_ROWS = 7_670
PINNED_PREPARATION_ASSIGNMENT_ROWS = 231_060
PINNED_MATERIALIZED_SYNTHETIC_ROWS = 123_232
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


class ReleaseContractError(ValueError):
    """Raised before unverified preparation data can reach the experiment."""


@dataclass(frozen=True)
class ReleaseData:
    """Verified real data and lazy access to verified outer-fold synthetic data."""

    path: Path
    archive_sha256: str
    manifest: Mapping[str, Any]
    preparation_config: Mapping[str, Any]
    canonical: pd.DataFrame
    assignments: pd.DataFrame
    feature_sets: Mapping[str, tuple[str, ...]]
    _member_hashes: Mapping[str, str]
    _outer_members: Mapping[tuple[str, int, str], str]

    @property
    def eligible_real(self) -> pd.DataFrame:
        """Return an independent, deterministic copy of the 7,702 eligible rows."""

        result = self.canonical.loc[self.canonical["train_eligible"].eq(True)].copy()
        return result.sort_values("trial_uid", kind="mergesort").reset_index(drop=True)

    @property
    def source_member_sha256(self) -> str:
        return str(self.preparation_config["source"]["member_sha256"])

    @property
    def preparation_config_sha256(self) -> str:
        return str(self.manifest["config_sha256"])

    @property
    def base_seed(self) -> int:
        return int(self.preparation_config["contract"]["base_seed"])

    def outer_synthetic_member(self, scheme: str, outer_fold: int, method: str) -> str:
        key = (str(scheme), int(outer_fold), str(method))
        try:
            return self._outer_members[key]
        except KeyError as exc:
            raise ReleaseContractError(
                "preparation release has no materialized outer synthetic member for "
                f"{key!r}"
            ) from exc

    def load_outer_synthetic(
        self, scheme: str, outer_fold: int, method: str
    ) -> pd.DataFrame:
        """Load one outer-refit member and recheck its member hash and context."""

        relative = self.outer_synthetic_member(scheme, outer_fold, method)
        member = f"{PREPARATION_PACKAGE_ROOT}/{relative}"
        payload = _read_member(self.path, member)
        expected_hash = self._member_hashes[relative]
        actual_hash = sha256(payload).hexdigest()
        if actual_hash != expected_hash:
            raise ReleaseContractError(
                f"release member changed after verification: {relative}; "
                f"expected {expected_hash}, received {actual_hash}"
            )
        frame = pd.read_parquet(BytesIO(payload), engine="pyarrow")
        _validate_outer_synthetic_frame(
            frame,
            scheme=str(scheme),
            outer_fold=int(outer_fold),
            method=str(method),
        )
        return frame


def verify_and_load_release(
    path: str | Path,
    *,
    expected_sha256: str = PINNED_RELEASE_SHA256,
    enforce_pinned_counts: bool = True,
) -> ReleaseData:
    """Authenticate and load ``alphaf.ml_preparation.v1``.

    ``expected_sha256`` is mandatory and must be a 64-character lowercase
    hexadecimal digest.  Tests can pass the digest of a deliberately tiny
    fixture; production callers use the pinned default.
    """

    release_path = Path(path).resolve()
    if not release_path.is_file():
        raise FileNotFoundError(f"ML-preparation release does not exist: {release_path}")
    with release_path.open("rb") as handle:
        prefix = handle.read(max(200, len(_LFS_HEADER)))
    if prefix.startswith(_LFS_HEADER):
        raise ReleaseContractError(
            "ML-preparation release is a Git LFS pointer; fetch the LFS object first"
        )
    if not _is_sha256(expected_sha256):
        raise ReleaseContractError("expected release SHA-256 must be 64 lowercase hex digits")
    actual_archive_hash = sha256_file(release_path)
    if actual_archive_hash != expected_sha256:
        raise ReleaseContractError(
            "ML-preparation release SHA-256 mismatch: "
            f"expected {expected_sha256}, received {actual_archive_hash}"
        )

    try:
        with ZipFile(release_path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            _validate_zip_names(names)
            corrupt = archive.testzip()
            if corrupt is not None:
                raise ReleaseContractError(f"ZIP CRC check failed for member: {corrupt}")
            manifest_member = f"{PREPARATION_PACKAGE_ROOT}/manifest.json"
            if manifest_member not in names:
                raise ReleaseContractError("release manifest.json is missing")
            manifest = _load_json(archive.read(manifest_member), "manifest.json")
            member_index = _validate_manifest(manifest, names, infos)
            for relative, expected in member_index.items():
                member_payload = archive.read(f"{PREPARATION_PACKAGE_ROOT}/{relative}")
                actual = sha256(member_payload).hexdigest()
                if actual != expected:
                    raise ReleaseContractError(
                        f"release member SHA-256 mismatch for {relative}: "
                        f"expected {expected}, received {actual}"
                    )

            config = _load_json(
                archive.read(f"{PREPARATION_PACKAGE_ROOT}/config.resolved.json"),
                "config.resolved.json",
            )
            feature_payload = _load_json(
                archive.read(f"{PREPARATION_PACKAGE_ROOT}/canonical/feature_sets.json"),
                "canonical/feature_sets.json",
            )
            canonical_bytes = archive.read(
                f"{PREPARATION_PACKAGE_ROOT}/canonical/canonical_real.parquet"
            )
            assignment_bytes = archive.read(
                f"{PREPARATION_PACKAGE_ROOT}/splits/split_assignments.parquet"
            )
            _validate_parquet_schema(
                canonical_bytes,
                manifest["schemas"]["canonical_real"],
                "canonical/canonical_real.parquet",
            )
            _validate_parquet_schema(
                assignment_bytes,
                manifest["schemas"]["split_assignments"],
                "splits/split_assignments.parquet",
            )
            synthetic_schema = manifest["schemas"]["synthetic_fold"]
            provenance_schema = manifest["schemas"]["augmentation_provenance"]
            provenance_bytes = archive.read(
                f"{PREPARATION_PACKAGE_ROOT}/augmentation/provenance.parquet"
            )
            _validate_parquet_schema(
                provenance_bytes,
                provenance_schema,
                "augmentation/provenance.parquet",
            )
            if _parquet_num_rows(provenance_bytes) != int(
                manifest["counts"]["synthetic_materialized"]
            ):
                raise ReleaseContractError(
                    "augmentation provenance row count disagrees with manifest counts"
                )
            outer_members = _outer_member_index(manifest)
            outer_rows = {
                str(entry["path"]): int(entry["rows"])
                for entry in manifest["augmentation"]
            }
            for relative in outer_members.values():
                synthetic_bytes = archive.read(
                    f"{PREPARATION_PACKAGE_ROOT}/{relative}"
                )
                _validate_parquet_schema(
                    synthetic_bytes,
                    synthetic_schema,
                    relative,
                )
                if _parquet_num_rows(synthetic_bytes) != outer_rows[relative]:
                    raise ReleaseContractError(
                        f"materialized synthetic row count mismatch for {relative}"
                    )
    except BadZipFile as exc:
        raise ReleaseContractError(f"invalid ML-preparation ZIP: {exc}") from exc

    _validate_preparation_config(config, manifest)
    canonical = pd.read_parquet(BytesIO(canonical_bytes), engine="pyarrow")
    assignments = pd.read_parquet(BytesIO(assignment_bytes), engine="pyarrow")
    feature_sets = _validate_feature_sets(feature_payload, canonical)
    _validate_loaded_frames(
        canonical,
        assignments,
        manifest,
        enforce_pinned_counts=enforce_pinned_counts,
    )
    return ReleaseData(
        path=release_path,
        archive_sha256=actual_archive_hash,
        manifest=manifest,
        preparation_config=config,
        canonical=canonical,
        assignments=assignments,
        feature_sets=feature_sets,
        _member_hashes=member_index,
        _outer_members=outer_members,
    )


def _read_member(path: Path, member: str) -> bytes:
    try:
        with ZipFile(path) as archive:
            return archive.read(member)
    except (BadZipFile, KeyError) as exc:
        raise ReleaseContractError(f"cannot read verified release member {member!r}") from exc


def _load_json(payload: bytes, description: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseContractError(f"invalid JSON in {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseContractError(f"{description} must contain a JSON object")
    return value


def _is_sha256(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_zip_names(names: list[str]) -> None:
    if len(names) != len(set(names)):
        raise ReleaseContractError("release ZIP contains duplicate member names")
    root_prefix = f"{PREPARATION_PACKAGE_ROOT}/"
    for name in names:
        path = PurePosixPath(name)
        if (
            not name.startswith(root_prefix)
            or "\\" in name
            or name.startswith("/")
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ReleaseContractError(f"unsafe or unexpected ZIP member path: {name!r}")


def _validate_manifest(
    manifest: Mapping[str, Any], names: list[str], infos: list[Any]
) -> dict[str, str]:
    if manifest.get("contract") != PREPARATION_CONTRACT:
        raise ReleaseContractError(
            f"release contract must be {PREPARATION_CONTRACT!r}, "
            f"received {manifest.get('contract')!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ReleaseContractError("release manifest files index is missing or empty")
    actual_sizes = {info.filename: int(info.file_size) for info in infos}
    index: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ReleaseContractError("manifest file entries must be objects")
        relative = entry.get("path")
        digest = entry.get("sha256")
        byte_count = entry.get("bytes")
        if not isinstance(relative, str) or not relative:
            raise ReleaseContractError("manifest file path must be a non-empty string")
        if relative in index:
            raise ReleaseContractError(f"duplicate manifest file path: {relative}")
        _validate_relative_member(relative)
        if not _is_sha256(digest):
            raise ReleaseContractError(f"invalid manifest SHA-256 for {relative}")
        member = f"{PREPARATION_PACKAGE_ROOT}/{relative}"
        if member not in actual_sizes:
            raise ReleaseContractError(f"manifest member is missing from ZIP: {relative}")
        if not isinstance(byte_count, int) or byte_count < 0:
            raise ReleaseContractError(f"invalid manifest byte count for {relative}")
        if actual_sizes[member] != byte_count:
            raise ReleaseContractError(
                f"manifest byte count mismatch for {relative}: expected {byte_count}, "
                f"received {actual_sizes[member]}"
            )
        index[relative] = digest
    expected_names = {
        f"{PREPARATION_PACKAGE_ROOT}/manifest.json",
        *(f"{PREPARATION_PACKAGE_ROOT}/{relative}" for relative in index),
    }
    if set(names) != expected_names:
        unexpected = sorted(set(names) - expected_names)
        missing = sorted(expected_names - set(names))
        raise ReleaseContractError(
            f"release ZIP membership differs from manifest; missing={missing}, "
            f"unexpected={unexpected}"
        )
    schemas = manifest.get("schemas")
    required_schemas = {
        "canonical_real",
        "split_assignments",
        "synthetic_fold",
        "augmentation_provenance",
    }
    if not isinstance(schemas, dict) or not required_schemas.issubset(schemas):
        raise ReleaseContractError(
            f"release manifest lacks required Arrow schemas: {sorted(required_schemas)}"
        )
    return index


def _validate_relative_member(relative: str) -> None:
    path = PurePosixPath(relative)
    if (
        relative.startswith("/")
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReleaseContractError(f"unsafe manifest member path: {relative!r}")


def _arrow_fields(payload: bytes) -> list[dict[str, Any]]:
    try:
        schema = pq.ParquetFile(BytesIO(payload)).schema_arrow
    except Exception as exc:  # pyarrow exposes several version-specific errors
        raise ReleaseContractError(f"member is not readable Parquet: {exc}") from exc
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def _parquet_num_rows(payload: bytes) -> int:
    try:
        return int(pq.ParquetFile(BytesIO(payload)).metadata.num_rows)
    except Exception as exc:  # pyarrow exposes several version-specific errors
        raise ReleaseContractError(f"member is not readable Parquet: {exc}") from exc


def _validate_parquet_schema(
    payload: bytes, expected: Mapping[str, Any], description: str
) -> None:
    if not isinstance(expected, dict) or not isinstance(expected.get("columns"), list):
        raise ReleaseContractError(f"missing schema declaration for {description}")
    fields = _arrow_fields(payload)
    if fields != expected["columns"]:
        raise ReleaseContractError(f"ordered Arrow schema mismatch for {description}")
    fingerprint = sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if fingerprint != expected.get("fingerprint_sha256"):
        raise ReleaseContractError(f"Arrow schema fingerprint mismatch for {description}")


def _outer_member_index(manifest: Mapping[str, Any]) -> dict[tuple[str, int, str], str]:
    entries = manifest.get("augmentation")
    if not isinstance(entries, list):
        raise ReleaseContractError("release augmentation index is missing")
    result: dict[tuple[str, int, str], str] = {}
    total_rows = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ReleaseContractError("augmentation index entries must be objects")
        try:
            key = (
                str(entry["scheme"]),
                int(entry["outer_fold"]),
                str(entry["method"]),
            )
            relative = str(entry["path"])
            rows = int(entry["rows"])
            real_rows = int(entry["real_train_rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ReleaseContractError("malformed augmentation index entry") from exc
        if key in result:
            raise ReleaseContractError(f"duplicate augmentation context: {key!r}")
        if key[0] not in {"segment_loso", "target_group"}:
            raise ReleaseContractError(f"unknown materialized split scheme: {key[0]!r}")
        if key[2] not in {"local_interp_v2", "cmixup_offline_v1"}:
            raise ReleaseContractError(f"unknown materialized augmentation: {key[2]!r}")
        if rows != real_rows or rows <= 0:
            raise ReleaseContractError(
                f"augmentation must be 1:1 with outer real train for {key!r}"
            )
        _validate_relative_member(relative)
        result[key] = relative
        total_rows += rows
    expected_keys = {
        (scheme, fold, method)
        for scheme, folds in (("segment_loso", range(36, 41)), ("target_group", range(5)))
        for fold in folds
        for method in ("local_interp_v2", "cmixup_offline_v1")
    }
    if set(result) != expected_keys:
        raise ReleaseContractError("materialized outer augmentation context set changed")
    if total_rows != int(manifest["counts"]["synthetic_materialized"]):
        raise ReleaseContractError("augmentation index row total disagrees with manifest counts")
    return result


def _validate_preparation_config(
    config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    try:
        contract = config["contract"]
        source = config["source"]
        splits = config["splits"]
        features = config["features"]
    except KeyError as exc:
        raise ReleaseContractError(f"preparation config section missing: {exc}") from exc
    if contract.get("name") != PREPARATION_CONTRACT:
        raise ReleaseContractError("embedded preparation config contract changed")
    if int(contract.get("base_seed", -1)) != 20_260_818:
        raise ReleaseContractError("embedded preparation base seed changed")
    if source.get("member") != "derived/parquet/real_motion_demo/trials.parquet":
        raise ReleaseContractError("embedded source member is not pinned trials.parquet")
    if not _is_sha256(source.get("member_sha256")):
        raise ReleaseContractError("embedded source member hash is invalid")
    if splits.get("segment_outer_values") != [36, 37, 38, 39, 40]:
        raise ReleaseContractError("embedded segment fold set changed")
    if int(splits.get("target_group_count", -1)) != 5:
        raise ReleaseContractError("embedded target group count changed")
    if features.get("label_columns") != list(LABEL_COLUMNS):
        raise ReleaseContractError("embedded raw Vive label columns changed")
    source_summary = manifest.get("source")
    if not isinstance(source_summary, dict):
        raise ReleaseContractError("manifest source summary is missing")
    if source_summary.get("member_sha256") != source.get("member_sha256"):
        raise ReleaseContractError("manifest/config source member hash disagreement")
    config_digest = sha256(
        json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if config_digest != manifest.get("config_sha256"):
        raise ReleaseContractError("embedded preparation config hash disagrees with manifest")


def _validate_feature_sets(
    payload: Mapping[str, Any], canonical: pd.DataFrame
) -> dict[str, tuple[str, ...]]:
    required = {
        "encoder_only_v1",
        "four_imu_endpoint_v1",
        "fusion_v1",
        "command_context_ablation_v1",
        "mag_ablation_v1",
    }
    if set(payload) != required:
        raise ReleaseContractError(
            f"feature set names changed: expected {sorted(required)}, received {sorted(payload)}"
        )
    result: dict[str, tuple[str, ...]] = {}
    for name, values in payload.items():
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
            or len(values) != len(set(values))
        ):
            raise ReleaseContractError(f"invalid feature allow-list for {name!r}")
        result[name] = tuple(values)
    try:
        validate_feature_views(canonical, result)
    except (TypeError, ValueError) as exc:
        raise ReleaseContractError(f"invalid embedded feature sets: {exc}") from exc
    return result


def _validate_loaded_frames(
    canonical: pd.DataFrame,
    assignments: pd.DataFrame,
    manifest: Mapping[str, Any],
    *,
    enforce_pinned_counts: bool,
) -> None:
    required_real = {
        "trial_uid",
        "segment_id",
        "target_id",
        "train_eligible",
        "strict_qc_eligible",
        "is_synthetic",
        *LABEL_COLUMNS,
    }
    missing = sorted(required_real - set(canonical.columns))
    if missing:
        raise ReleaseContractError(f"canonical data is missing required columns: {missing}")
    if canonical["trial_uid"].isna().any() or canonical["trial_uid"].duplicated().any():
        raise ReleaseContractError("canonical trial_uid values must be non-null and unique")
    if canonical["is_synthetic"].isna().any() or canonical["is_synthetic"].astype(bool).any():
        raise ReleaseContractError("canonical release must contain measured real rows only")
    eligible = canonical["train_eligible"].eq(True)
    counts = manifest.get("counts", {})
    observed = {
        "canonical_real": len(canonical),
        "train_eligible_real": int(eligible.sum()),
        "split_assignments": len(assignments),
    }
    for key, value in observed.items():
        if int(counts.get(key, -1)) != value:
            raise ReleaseContractError(
                f"manifest/data count mismatch for {key}: {counts.get(key)!r} != {value}"
            )
    if enforce_pinned_counts:
        pinned = {
            "canonical_real": PINNED_CANONICAL_ROWS,
            "train_eligible_real": PINNED_ELIGIBLE_ROWS,
            "split_assignments": PINNED_PREPARATION_ASSIGNMENT_ROWS,
            "synthetic_materialized": PINNED_MATERIALIZED_SYNTHETIC_ROWS,
        }
        for key, value in pinned.items():
            if int(counts.get(key, -1)) != value:
                raise ReleaseContractError(
                    f"pinned preparation count changed for {key}: expected {value}, "
                    f"received {counts.get(key)!r}"
                )
        strict_rows = int(canonical["strict_qc_eligible"].eq(True).sum())
        if strict_rows != PINNED_STRICT_QC_ROWS:
            raise ReleaseContractError(
                "pinned strict-QC cohort changed: expected "
                f"{PINNED_STRICT_QC_ROWS}, received {strict_rows}"
            )
    try:
        validate_split_assignments(assignments, expected_trial_count=int(eligible.sum()))
    except ValueError as exc:
        raise ReleaseContractError(f"invalid preparation split assignments: {exc}") from exc
    eligible_ids = set(canonical.loc[eligible, "trial_uid"].astype(str))
    assignment_ids = set(assignments["trial_uid"].astype(str))
    if eligible_ids != assignment_ids:
        raise ReleaseContractError("preparation assignments do not cover eligible real IDs exactly")


def _validate_outer_synthetic_frame(
    frame: pd.DataFrame, *, scheme: str, outer_fold: int, method: str
) -> None:
    required = {
        "synthetic_id",
        "augmentation_method",
        "split_scheme",
        "outer_fold",
        "inner_fold",
        "parent_a_trial_uid",
        "parent_b_trial_uid",
        "sample_weight",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ReleaseContractError(f"outer synthetic data is missing columns: {missing}")
    if len(frame) == 0 or frame["synthetic_id"].isna().any() or frame["synthetic_id"].duplicated().any():
        raise ReleaseContractError("outer synthetic IDs must be non-null, unique, and non-empty")
    checks = {
        "split_scheme": (frame["split_scheme"].astype(str) == scheme).all(),
        "outer_fold": (frame["outer_fold"].astype(int) == int(outer_fold)).all(),
        "inner_fold": (frame["inner_fold"].astype(int) == -1).all(),
        "augmentation_method": (frame["augmentation_method"].astype(str) == method).all(),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ReleaseContractError(f"outer synthetic context mismatch: {failed}")


__all__ = [
    "PINNED_RELEASE_SHA256",
    "PREPARATION_CONTRACT",
    "ReleaseContractError",
    "ReleaseData",
    "verify_and_load_release",
]

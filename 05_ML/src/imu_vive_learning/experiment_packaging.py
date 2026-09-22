"""Deterministic ZIP packaging for endpoint-regression experiment v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from .experiment_config import EXPERIMENT_CONTRACT


EXPERIMENT_PACKAGE_ROOT = "endpoint_regression_experiment_v1"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(name, ZIP_TIMESTAMP)
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def package_experiment_run(run_dir: str | Path, output_path: str | Path) -> Path:
    source = Path(run_dir).resolve()
    output = Path(output_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"experiment run directory does not exist: {source}")
    if output == source or source in output.parents:
        raise ValueError("experiment package output must be outside the run directory")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != EXPERIMENT_CONTRACT:
        raise ValueError("experiment manifest contract changed")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("experiment manifest files list is missing")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with ZipFile(temporary, "w", compression=ZIP_STORED, allowZip64=True) as archive:
            archive.writestr(
                _zip_info(f"{EXPERIMENT_PACKAGE_ROOT}/manifest.json"),
                manifest_path.read_bytes(),
            )
            for entry in files:
                relative = str(entry["path"])
                path = source / Path(relative)
                payload = path.read_bytes()
                if len(payload) != int(entry["bytes"]):
                    raise ValueError(f"manifest size mismatch before packaging: {relative}")
                if hashlib.sha256(payload).hexdigest() != str(entry["sha256"]):
                    raise ValueError(f"manifest SHA-256 mismatch before packaging: {relative}")
                archive.writestr(_zip_info(f"{EXPERIMENT_PACKAGE_ROOT}/{relative}"), payload)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def validate_experiment_package(path: str | Path) -> dict[str, Any]:
    archive_path = Path(path).resolve()
    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        raw_names = [info.orig_filename for info in infos]
        if raw_names != names:
            raise ValueError("experiment package contains a non-canonical raw ZIP member path")
        if len(names) != len(set(names)):
            raise ValueError("experiment package contains duplicate ZIP members")
        for name in names:
            member = PurePosixPath(name)
            if (
                "\\" in name
                or name.endswith("/")
                or member.is_absolute()
                or not member.parts
                or member.as_posix() != name
                or member.parts[0] != EXPERIMENT_PACKAGE_ROOT
                or any(part in {"", ".", ".."} for part in member.parts)
            ):
                raise ValueError(f"unsafe experiment package member path: {name!r}")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"experiment ZIP CRC failure: {bad}")
        manifest_name = f"{EXPERIMENT_PACKAGE_ROOT}/manifest.json"
        try:
            manifest = json.loads(archive.read(manifest_name).decode("utf-8"))
        except KeyError as exc:
            raise ValueError("experiment package manifest is missing") from exc
        if manifest.get("contract") != EXPERIMENT_CONTRACT:
            raise ValueError("unexpected experiment package contract")
        entries = manifest.get("files", [])
        if not isinstance(entries, list):
            raise ValueError("experiment package manifest files list is missing")
        entry_paths = [str(entry.get("path", "")) for entry in entries]
        canonical_entry_paths = [PurePosixPath(path).as_posix() for path in entry_paths]
        if (
            len(entry_paths) != len(set(entry_paths))
            or entry_paths != canonical_entry_paths
            or "manifest.json" in entry_paths
            or any(
                not path
                or "\\" in path
                or PurePosixPath(path).is_absolute()
                or any(part in {"", ".", ".."} for part in PurePosixPath(path).parts)
                for path in entry_paths
            )
        ):
            raise ValueError("experiment package manifest contains duplicate/reserved paths")
        expected_names = {manifest_name}
        for entry in entries:
            name = f"{EXPERIMENT_PACKAGE_ROOT}/{entry['path']}"
            expected_names.add(name)
            payload = archive.read(name)
            if len(payload) != int(entry["bytes"]):
                raise ValueError(f"packaged size mismatch: {name}")
            if hashlib.sha256(payload).hexdigest() != str(entry["sha256"]):
                raise ValueError(f"packaged SHA-256 mismatch: {name}")
        actual_names = {name for name in names if not name.endswith("/")}
        if actual_names != expected_names:
            raise ValueError("experiment package contains unmanifested or missing files")
    return {
        "status": "ok",
        "contract": manifest["contract"],
        "files": len(expected_names),
        "bytes": archive_path.stat().st_size,
        "manifest": manifest,
    }


def read_package_member(path: str | Path, relative: str) -> bytes:
    if relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError("unsafe experiment package member path")
    with ZipFile(Path(path).resolve()) as archive:
        return archive.read(f"{EXPERIMENT_PACKAGE_ROOT}/{relative}")


__all__ = [
    "EXPERIMENT_PACKAGE_ROOT",
    "package_experiment_run",
    "read_package_member",
    "validate_experiment_package",
]

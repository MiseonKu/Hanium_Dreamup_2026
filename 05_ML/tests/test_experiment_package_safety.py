from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest

from imu_vive_learning.experiment_config import EXPERIMENT_CONTRACT
from imu_vive_learning.experiment_packaging import (
    EXPERIMENT_PACKAGE_ROOT,
    validate_experiment_package,
)


MANIFEST_MEMBER = f"{EXPERIMENT_PACKAGE_ROOT}/manifest.json"
PAYLOAD = b"endpoint-package-safety-fixture-v1"


def _manifest(entries: list[dict[str, object]]) -> bytes:
    return (
        json.dumps(
            {
                "contract": EXPERIMENT_CONTRACT,
                "files": entries,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _entry(
    path: str = "payload.bin",
    *,
    payload: bytes = PAYLOAD,
    recorded_payload: bytes | None = None,
) -> dict[str, object]:
    recorded = payload if recorded_payload is None else recorded_payload
    return {
        "path": path,
        "bytes": len(recorded),
        "sha256": hashlib.sha256(recorded).hexdigest(),
    }


def _write_package(
    path: Path,
    *,
    entries: list[dict[str, object]],
    members: list[tuple[str, bytes]],
) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(path, "w", compression=ZIP_STORED) as archive:
            archive.writestr(MANIFEST_MEMBER, _manifest(entries))
            for name, payload in members:
                archive.writestr(name, payload)
    return path


def test_validate_rejects_duplicate_zip_members(tmp_path: Path) -> None:
    member = f"{EXPERIMENT_PACKAGE_ROOT}/payload.bin"
    package = _write_package(
        tmp_path / "duplicate-member.zip",
        entries=[_entry()],
        members=[(member, PAYLOAD), (member, PAYLOAD)],
    )

    with pytest.raises(ValueError, match="duplicate ZIP members"):
        validate_experiment_package(package)


@pytest.mark.parametrize(
    "unsafe_member",
    [
        f"{EXPERIMENT_PACKAGE_ROOT}/../escape.bin",
        f"{EXPERIMENT_PACKAGE_ROOT}\\payload.bin",
        f"wrong_root/payload.bin",
        f"/{EXPERIMENT_PACKAGE_ROOT}/payload.bin",
    ],
    ids=["traversal", "backslash", "wrong-root", "absolute"],
)
def test_validate_rejects_unsafe_zip_member_paths(
    tmp_path: Path, unsafe_member: str
) -> None:
    # ``zipfile`` normalizes ``os.sep`` to ``/`` while writing on Windows.
    # Patch both the local and central-directory names byte-for-byte so the
    # reader sees the adversarial backslash spelling that an external ZIP can
    # contain. The replacement is length-preserving and does not touch CRCs.
    written_member = unsafe_member.replace("\\", "/")
    package = _write_package(
        tmp_path / "unsafe-member.zip",
        entries=[],
        members=[(written_member, PAYLOAD)],
    )
    if written_member != unsafe_member:
        raw = package.read_bytes()
        assert raw.count(written_member.encode("utf-8")) == 2
        package.write_bytes(
            raw.replace(
                written_member.encode("utf-8"), unsafe_member.encode("utf-8")
            )
        )

    with pytest.raises(
        ValueError,
        match="unsafe experiment package member path|non-canonical raw ZIP member path",
    ):
        validate_experiment_package(package)


@pytest.mark.parametrize(
    "entries",
    [
        [_entry(), _entry()],
        [_entry("manifest.json", payload=_manifest([]))],
    ],
    ids=["duplicate", "reserved-manifest"],
)
def test_validate_rejects_duplicate_or_reserved_manifest_paths(
    tmp_path: Path, entries: list[dict[str, object]]
) -> None:
    package = _write_package(
        tmp_path / "bad-manifest-paths.zip",
        entries=entries,
        members=[],
    )

    with pytest.raises(ValueError, match="duplicate/reserved paths"):
        validate_experiment_package(package)


@pytest.mark.parametrize(
    ("entries", "members"),
    [
        (
            [_entry("./manifest.json", payload=_manifest([]))],
            [(f"{EXPERIMENT_PACKAGE_ROOT}/./manifest.json", _manifest([]))],
        ),
        (
            [_entry(), _entry("./payload.bin")],
            [
                (f"{EXPERIMENT_PACKAGE_ROOT}/payload.bin", PAYLOAD),
                (f"{EXPERIMENT_PACKAGE_ROOT}/./payload.bin", PAYLOAD),
            ],
        ),
    ],
    ids=["normalized-reserved-manifest", "normalized-duplicate"],
)
def test_validate_rejects_noncanonical_manifest_paths(
    tmp_path: Path,
    entries: list[dict[str, object]],
    members: list[tuple[str, bytes]],
) -> None:
    package = _write_package(
        tmp_path / "noncanonical-manifest-path.zip",
        entries=entries,
        members=members,
    )

    with pytest.raises(ValueError, match="duplicate/reserved|unsafe"):
        validate_experiment_package(package)


def test_validate_rejects_crc_drift(tmp_path: Path) -> None:
    member = f"{EXPERIMENT_PACKAGE_ROOT}/payload.bin"
    package = _write_package(
        tmp_path / "crc-drift.zip",
        entries=[_entry()],
        members=[(member, PAYLOAD)],
    )
    raw = bytearray(package.read_bytes())
    payload_offset = raw.find(PAYLOAD)
    assert payload_offset >= 0, "stored fixture payload must be present verbatim"
    raw[payload_offset] ^= 0x01
    package.write_bytes(raw)

    with pytest.raises(ValueError, match="ZIP CRC failure"):
        validate_experiment_package(package)


def test_validate_rejects_member_sha256_drift_with_valid_crc(tmp_path: Path) -> None:
    member = f"{EXPERIMENT_PACKAGE_ROOT}/payload.bin"
    changed_payload = b"endpoint-package-safety-fixture-v2"
    package = _write_package(
        tmp_path / "sha-drift.zip",
        entries=[_entry(recorded_payload=PAYLOAD)],
        members=[(member, changed_payload)],
    )

    with pytest.raises(ValueError, match="packaged SHA-256 mismatch"):
        validate_experiment_package(package)

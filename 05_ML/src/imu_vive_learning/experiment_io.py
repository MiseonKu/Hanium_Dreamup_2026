"""Atomic, deterministic I/O helpers for the experiment contract."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path, PurePosixPath
import sys
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy
import sklearn
import threadpoolctl

from .config import canonical_json_bytes
from .io_utils import relative_file_manifest, sha256_file


ENDPOINT_EVAL_PACKAGE_VERSIONS = {
    "numpy": "2.3.5",
    "pandas": "2.3.3",
    "pyarrow": "21.0.0",
    "scipy": "1.17.1",
    "scikit_learn": "1.9.0",
    "xgboost": "3.3.0",
    "matplotlib": "3.11.1",
    "joblib": "1.5.3",
    "threadpoolctl": "3.6.0",
}


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing JSON file: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {source}")
    return payload


def _temporary_for(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_for(output)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)
    return output


def atomic_write_text(path: str | Path, text: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_for(output)
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, output)
    return output


def atomic_write_parquet(path: str | Path, frame: pd.DataFrame) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_for(output)
    frame.to_parquet(
        temporary,
        engine="pyarrow",
        compression="zstd",
        index=False,
        version="2.6",
        data_page_version="1.0",
        use_dictionary=True,
        write_statistics=True,
    )
    os.replace(temporary, output)
    return output


def atomic_joblib_dump(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_for(output)
    joblib.dump(value, temporary, compress=3, protocol=5)
    os.replace(temporary, output)
    return output


def environment_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "matplotlib": matplotlib.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "byteorder": sys.byteorder,
    }
    try:
        import xgboost

        payload["xgboost"] = xgboost.__version__
    except ImportError:
        payload["xgboost"] = None
    return payload


def validate_endpoint_eval_environment(payload: dict[str, Any]) -> None:
    """Fail before fitting unless the exact CPython/evaluation lock is active."""

    if not str(payload.get("python", "")).startswith("3.12."):
        raise ValueError("endpoint-regression v1 requires Python 3.12.x")
    if payload.get("python_implementation") != "CPython":
        raise ValueError("endpoint-regression v1 requires CPython")
    for package, expected in ENDPOINT_EVAL_PACKAGE_VERSIONS.items():
        if payload.get(package) != expected:
            raise ValueError(
                f"frozen environment changed for {package}: "
                f"{payload.get(package)!r} != {expected!r}"
            )


def prepare_experiment_directory(
    run_dir: str | Path,
    *,
    identity: dict[str, Any],
    resume: bool,
) -> Path:
    """Create a run directory or verify the immutable identity of a resumed run."""

    run = Path(run_dir).resolve()
    state = run / "_state"
    identity_path = state / "run_identity.json"
    if run.exists() and any(run.iterdir()):
        if not resume:
            raise FileExistsError(f"output directory is not empty; pass --resume: {run}")
        actual = read_json_object(identity_path)
        if actual != identity:
            raise ValueError("resume refused: task/config/release identity changed")
    else:
        run.mkdir(parents=True, exist_ok=True)
        atomic_write_json(identity_path, identity)
    return run


def manifest_files(run_dir: str | Path) -> list[dict[str, Any]]:
    """Manifest public members while excluding mutable resume/cache state."""

    run = Path(run_dir)
    entries = relative_file_manifest(run, exclude={"manifest.json"})
    return [
        entry
        for entry in entries
        if not entry["path"].startswith("_state/")
        and not entry["path"].startswith("_cache/")
        and not _is_atomic_temporary_path(str(entry["path"]))
    ]


def _is_atomic_temporary_path(relative: str) -> bool:
    """Recognize only temporary names emitted by this module's atomic writers."""

    name = PurePosixPath(relative).name
    return name.startswith(".") and (
        name.endswith(".tmp") or name.endswith(".tmp.npz")
    )


def verify_file_manifest(run_dir: str | Path, expected: list[dict[str, Any]]) -> None:
    actual = manifest_files(run_dir)
    if actual != expected:
        raise ValueError("experiment manifest file list, size, or SHA-256 changed")


__all__ = [
    "atomic_joblib_dump",
    "atomic_write_json",
    "atomic_write_parquet",
    "atomic_write_text",
    "ENDPOINT_EVAL_PACKAGE_VERSIONS",
    "environment_payload",
    "manifest_files",
    "prepare_experiment_directory",
    "read_json_object",
    "sha256_file",
    "stable_hash",
    "validate_endpoint_eval_environment",
    "verify_file_manifest",
]

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        output,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
        data_page_version="1.0",
    )


def relative_file_manifest(root: str | Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    base = Path(root).resolve()
    if not base.is_dir():
        raise ValueError(f"manifest root is not a directory: {base}")
    excluded = exclude or set()
    entries: list[dict[str, Any]] = []
    paths = sorted(base.rglob("*"))
    symlinks = [path.relative_to(base).as_posix() for path in paths if path.is_symlink()]
    if symlinks:
        raise ValueError(f"manifest tree contains symbolic links: {symlinks[:5]}")
    for path in paths:
        if not path.is_file():
            continue
        if not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
            raise ValueError(f"manifest member is not a regular file: {path}")
        resolved = path.resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"manifest member escapes root: {path}") from exc
        relative = path.relative_to(base).as_posix()
        if relative in excluded:
            continue
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries

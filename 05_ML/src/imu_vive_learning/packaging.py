from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo


PACKAGE_ROOT = "ml_preparation_v1"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def package_run(run_dir: str | Path, output_path: str | Path) -> Path:
    source = Path(run_dir).resolve()
    output = Path(output_path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {source}")
    if output == source or source in output.parents:
        raise ValueError("package output must be outside the run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with ZipFile(temporary, "w", compression=ZIP_STORED, allowZip64=True) as archive:
            for path in sorted(item for item in source.rglob("*") if item.is_file()):
                relative = path.relative_to(source).as_posix()
                info = ZipInfo(f"{PACKAGE_ROOT}/{relative}", ZIP_TIMESTAMP)
                info.compress_type = ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output

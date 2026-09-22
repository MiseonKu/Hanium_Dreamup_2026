from __future__ import annotations

from pathlib import Path

import pytest

from imu_vive_learning.experiment_io import manifest_files, verify_file_manifest


def test_atomic_temporary_files_do_not_poison_manifest(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "nested").mkdir(parents=True)
    (run / "result.json").write_text("complete\n", encoding="utf-8")

    expected = manifest_files(run)
    assert [entry["path"] for entry in expected] == ["result.json"]

    (run / ".manifest.json.tmp").write_text("interrupted\n", encoding="utf-8")
    (run / "nested" / ".history.parquet.tmp").write_bytes(b"partial parquet")
    (run / "nested" / ".torch_mlp_seed0.npz.tmp.npz").write_bytes(
        b"partial npz"
    )

    assert manifest_files(run) == expected
    verify_file_manifest(run, expected)


def test_manifest_preserves_hidden_files_outside_atomic_temp_convention(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    (run / "nested").mkdir(parents=True)
    retained = {
        ".gitkeep",
        ".history.tmp.backup",
        "nested/.metadata",
        "nested/visible.tmp",
        "nested/visible.tmp.npz",
    }
    for relative in retained:
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    paths = {str(entry["path"]) for entry in manifest_files(run)}

    assert paths == retained


@pytest.mark.parametrize("link_kind", ["file", "directory"])
def test_manifest_rejects_symbolic_links(tmp_path: Path, link_kind: str) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    run = tmp_path / "run"
    run.mkdir()
    if link_kind == "file":
        target = outside / "result.json"
        target.write_text("external\n", encoding="utf-8")
        (run / "result.json").symlink_to(target)
    else:
        target = outside / "results"
        target.mkdir()
        (target / "result.json").write_text("external\n", encoding="utf-8")
        (run / "results").symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic links"):
        manifest_files(run)

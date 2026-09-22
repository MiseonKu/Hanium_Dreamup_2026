from __future__ import annotations

import json

from research_host.calibration_contracts import (
    REQUIRED_CALIBRATION_TYPES,
    validate_calibration_dir,
    validate_calibration_file,
)


def test_validate_calibration_file_catches_template_placeholder(tmp_path) -> None:
    path = tmp_path / "baseframe.json"
    path.write_text(
        json.dumps(
            {
                "calibration_type": "baseframe",
                "version": "v001",
                "date": "YYYY-MM-DD",
                "R_world_to_base": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "p_world_origin_in_base_mm": [0, 0, 0],
            }
        ),
        encoding="utf-8",
    )

    result = validate_calibration_file(path)

    assert not result.ok
    assert "date still contains template placeholder" in result.issues


def test_validate_calibration_dir_requires_all_types(tmp_path) -> None:
    results = validate_calibration_dir(tmp_path)

    missing = {result.calibration_type for result in results if result.issues}
    assert set(REQUIRED_CALIBRATION_TYPES).issubset(missing)

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from imu_vive_learning.config import load_config
from imu_vive_learning.pipeline import validate_frozen_config


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "static_endpoint_v1.yaml"


def test_checked_in_config_matches_frozen_v1_implementation() -> None:
    validate_frozen_config(load_config(CONFIG_PATH))


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("augmentation", "ratio", 2.0, "ratio=1.0"),
        ("features", "euler_unit", "deg", "euler_unit"),
        ("features", "include_planned_primary", True, "planned"),
        ("features", "include_magnetometer_primary", True, "magnetometer"),
    ],
)
def test_frozen_config_rejects_declared_behavior_drift(
    section: str, key: str, value: object, message: str
) -> None:
    config = deepcopy(load_config(CONFIG_PATH))
    config[section][key] = value

    with pytest.raises(ValueError, match=message):
        validate_frozen_config(config)

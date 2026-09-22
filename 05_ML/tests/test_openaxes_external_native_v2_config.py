from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest
import yaml

from imu_vive_learning.openaxes_external_native_v2_config import (
    OPENAXES_EXTERNAL_NATIVE_V2_CONFIG_SHA256,
    OPENAXES_NATIVE_V2_CANDIDATE_IDS,
    OPENAXES_NATIVE_V2_CONTINUOUS_FEATURES,
    OPENAXES_NATIVE_V2_FIT_COUNTS_BY_GPU,
    OPENAXES_NATIVE_V2_FIT_COUNTS_BY_SLOT,
    OPENAXES_NATIVE_V2_PHASES,
    OpenAxesExternalNativeV2ConfigError,
    _UniqueKeySafeLoader,
    _canonical_json,
    load_openaxes_external_native_v2_config,
    load_openaxes_external_native_v2_config_at,
    openaxes_external_native_v2_config_sha256,
    validate_openaxes_external_native_v2_config,
)


CONFIG = Path("configs/openaxes_external_native_v2.yaml")


def test_frozen_config_digest_and_literal_grid() -> None:
    config = load_openaxes_external_native_v2_config(CONFIG)
    assert openaxes_external_native_v2_config_sha256(config) == (
        OPENAXES_EXTERNAL_NATIVE_V2_CONFIG_SHA256
    )
    assert len(OPENAXES_NATIVE_V2_CONTINUOUS_FEATURES) == 36
    assert len(set(OPENAXES_NATIVE_V2_CONTINUOUS_FEATURES)) == 36
    assert len(OPENAXES_NATIVE_V2_CANDIDATE_IDS) == 3
    assert len(OPENAXES_NATIVE_V2_PHASES) == 7
    assert OPENAXES_NATIVE_V2_FIT_COUNTS_BY_SLOT == (
        12,
        12,
        12,
        12,
        12,
        12,
        9,
        9,
        9,
        9,
        9,
        9,
    )
    assert OPENAXES_NATIVE_V2_FIT_COUNTS_BY_GPU == (33, 33, 30, 30)
    assert config["folds_and_fits"]["total_scientific_fits"] == 3 * 14 * 3
    assert config["optimization"]["full_training_fold_updates_per_fit"] == 400


def test_production_fd_loader_uses_workspace_relative_openat() -> None:
    workspace_fd = os.open("..", os.O_RDONLY | os.O_DIRECTORY)
    try:
        config = load_openaxes_external_native_v2_config_at(workspace_fd)
    finally:
        os.close(workspace_fd)
    assert config["_config_relative_path"].endswith(
        "configs/openaxes_external_native_v2.yaml"
    )
    assert openaxes_external_native_v2_config_sha256(config) == (
        OPENAXES_EXTERNAL_NATIVE_V2_CONFIG_SHA256
    )


def test_duplicate_nonfinite_and_semantic_mutation_fail_closed() -> None:
    with pytest.raises(OpenAxesExternalNativeV2ConfigError, match="duplicate"):
        yaml.load("root:\n  item: 1\n  item: 2\n", Loader=_UniqueKeySafeLoader)
    with pytest.raises(OpenAxesExternalNativeV2ConfigError, match="finite"):
        _canonical_json({"bad": float("nan")})
    config = load_openaxes_external_native_v2_config(CONFIG)
    mutated = deepcopy(config)
    mutated["folds_and_fits"]["total_scientific_fits"] = 125
    with pytest.raises(OpenAxesExternalNativeV2ConfigError):
        validate_openaxes_external_native_v2_config(mutated)
    nonstring = deepcopy(config)
    nonstring[7] = "hidden-extra"
    with pytest.raises(OpenAxesExternalNativeV2ConfigError, match="string-keyed"):
        validate_openaxes_external_native_v2_config(nonstring)


def test_loader_private_keys_are_the_only_digest_exclusion() -> None:
    config = load_openaxes_external_native_v2_config(CONFIG)
    augmented = deepcopy(config)
    augmented["_synthetic_loader_note"] = {"fd": 7}
    assert openaxes_external_native_v2_config_sha256(augmented) == (
        OPENAXES_EXTERNAL_NATIVE_V2_CONFIG_SHA256
    )
    public_mutation = deepcopy(config)
    public_mutation["contract"]["description"] = json.dumps({"changed": True})
    assert openaxes_external_native_v2_config_sha256(public_mutation) != (
        OPENAXES_EXTERNAL_NATIVE_V2_CONFIG_SHA256
    )

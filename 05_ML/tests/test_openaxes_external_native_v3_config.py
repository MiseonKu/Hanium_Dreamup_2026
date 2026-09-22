from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import pytest
import yaml

from imu_vive_learning.openaxes_external_native_v3_config import (
    OPENAXES_EXTERNAL_NATIVE_V3_CONFIG_SHA256,
    OPENAXES_NATIVE_V3_CANDIDATE_IDS,
    OPENAXES_NATIVE_V3_CONTEXT_COUNT,
    OPENAXES_NATIVE_V3_FITS_PER_GPU,
    OPENAXES_NATIVE_V3_MASKS,
    OPENAXES_NATIVE_V3_SALVAGE_INPUT_REFERENCE_ORDER,
    OPENAXES_NATIVE_V3_SALVAGE_RECORD_KEYS,
    OPENAXES_NATIVE_V3_SEEDS,
    OPENAXES_NATIVE_V3_TASK_COUNT,
    OPENAXES_NATIVE_V3_VIEW_IDS,
    OpenAxesExternalNativeV3ConfigError,
    _UniqueKeySafeLoader,
    candidate_schedule,
    context_ordinal,
    gpu_for_context,
    load_openaxes_external_native_v3_config,
    load_openaxes_external_native_v3_config_at,
    openaxes_external_native_v3_config_sha256,
    slot_for_context,
    task_ordinal,
    validate_openaxes_external_native_v3_config,
)


def test_frozen_config_and_effective_erratum4_overlay() -> None:
    config = load_openaxes_external_native_v3_config()
    assert openaxes_external_native_v3_config_sha256(config) == OPENAXES_EXTERNAL_NATIVE_V3_CONFIG_SHA256
    assert tuple(config["_effective_salvage_record_keys"]) == OPENAXES_NATIVE_V3_SALVAGE_RECORD_KEYS
    assert tuple(config["_effective_salvage_input_reference_order"]) == OPENAXES_NATIVE_V3_SALVAGE_INPUT_REFERENCE_ORDER
    assert OPENAXES_NATIVE_V3_SALVAGE_INPUT_REFERENCE_ORDER[4] == "v2_salvage_accounting_supplement_reference"
    assert "v2_salvage_accounting_supplement_reference" in OPENAXES_NATIVE_V3_SALVAGE_RECORD_KEYS
    assert len(OPENAXES_NATIVE_V3_VIEW_IDS) == len(OPENAXES_NATIVE_V3_MASKS) == 15
    assert len(set(OPENAXES_NATIVE_V3_MASKS)) == 15
    assert OPENAXES_NATIVE_V3_SEEDS == (1701, 2603, 3907)
    assert len(OPENAXES_NATIVE_V3_CANDIDATE_IDS) == 6
    assert OPENAXES_NATIVE_V3_CONTEXT_COUNT == 42
    assert OPENAXES_NATIVE_V3_TASK_COUNT == 252
    assert OPENAXES_NATIVE_V3_FITS_PER_GPU == (66, 66, 60, 60)


def test_candidate_schedules_and_scheduler_arithmetic() -> None:
    config = load_openaxes_external_native_v3_config()
    one_stage = candidate_schedule(config, OPENAXES_NATIVE_V3_CANDIDATE_IDS[2])
    terminal = candidate_schedule(config, OPENAXES_NATIVE_V3_CANDIDATE_IDS[-1])
    assert (one_stage.independent_expert_updates, one_stage.cooperative_updates) == (0, 400)
    assert (terminal.independent_expert_updates, terminal.cooperative_updates) == (100, 300)
    assert [context_ordinal(fold, seed) for fold in range(14) for seed in range(3)] == list(range(42))
    assert [task_ordinal(fold, seed, candidate) for fold in range(14) for seed in range(3) for candidate in range(6)] == list(range(252))
    assert [slot_for_context(context) for context in range(42)].count(0) == 4
    assert [slot_for_context(context) for context in range(42)].count(11) == 3
    fits = [0, 0, 0, 0]
    for context in range(42):
        fits[gpu_for_context(context)] += 6
    assert tuple(fits) == OPENAXES_NATIVE_V3_FITS_PER_GPU


def test_workspace_fd_loader_and_mutation_fail_closed() -> None:
    workspace_fd = os.open("..", os.O_RDONLY | os.O_DIRECTORY)
    try:
        config = load_openaxes_external_native_v3_config_at(workspace_fd)
    finally:
        os.close(workspace_fd)
    validate_openaxes_external_native_v3_config(config)
    mutation = deepcopy(config)
    mutation["training"]["exact_optimizer_updates_per_fit"] = 399
    with pytest.raises(OpenAxesExternalNativeV3ConfigError):
        validate_openaxes_external_native_v3_config(mutation)
    mutation = deepcopy(config)
    mutation["availability_lattice"]["exact_order"][0]["availability"] = [0, 0, 0, 0]
    with pytest.raises(OpenAxesExternalNativeV3ConfigError):
        validate_openaxes_external_native_v3_config(mutation)


def test_duplicate_yaml_and_wrong_raw_authority_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(OpenAxesExternalNativeV3ConfigError, match="duplicate"):
        yaml.load("root:\n  key: 1\n  key: 2\n", Loader=_UniqueKeySafeLoader)
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text("contract: {}\n", encoding="utf-8")
    with pytest.raises(OpenAxesExternalNativeV3ConfigError, match="raw authority"):
        load_openaxes_external_native_v3_config(wrong)

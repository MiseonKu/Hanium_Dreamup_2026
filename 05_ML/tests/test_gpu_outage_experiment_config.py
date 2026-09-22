from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from imu_vive_learning.gpu_outage_experiment_config import (
    GPUOutageExperimentConfigError,
    OUTAGE_CANDIDATE_IDS,
    PINNED_OUTAGE_GPU_CONFIG_SHA256,
    load_outage_gpu_config,
    outage_candidate_grid,
    outage_gpu_config_sha256,
    resolve_outage_development_release_path,
    resolve_outage_source_path,
    validate_outage_gpu_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "endpoint_regression_gpu_outage_v3.yaml"


def test_frozen_outage_v3_config_loads_with_pinned_identity() -> None:
    config = load_outage_gpu_config(CONFIG_PATH)
    assert outage_gpu_config_sha256(config) == PINNED_OUTAGE_GPU_CONFIG_SHA256
    assert tuple(candidate["candidate_id"] for candidate in outage_candidate_grid(config)) == (
        OUTAGE_CANDIDATE_IDS
    )
    assert resolve_outage_development_release_path(config).name == "ml_preparation_v1.zip"
    assert resolve_outage_source_path(config).name == "derived_dataset_archive.zip"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda c: c["contract"].update({"confirmatory_claim_allowed": True}), "confirmatory"),
        (lambda c: c["row_roles"]["challenge"].update({"segments": [39, 40, 41]}), "segments"),
        (lambda c: c["masking"]["states"][0].update({"probability": 0.3}), "probabilities"),
        (lambda c: c["selection"].update({"trainable_seeded_fit_count": 10}), "counts"),
        (
            lambda c: c["selection"].update({"nonfinite_context_score": "fail_closed"}),
            "non-finite context",
        ),
        (
            lambda c: c["selection"].update({"missing_context_action": "positive_infinity"}),
            "missing context",
        ),
        (lambda c: c["challenge"].update({"maximum_source_label_access_passes": 2}), "access"),
    ],
)
def test_semantic_contract_drift_fails_closed(mutator, message: str) -> None:
    config = load_outage_gpu_config(CONFIG_PATH)
    changed = deepcopy(config)
    mutator(changed)
    with pytest.raises(GPUOutageExperimentConfigError, match=message):
        validate_outage_gpu_config(changed)


def test_unrecognized_material_change_is_caught_by_pinned_digest() -> None:
    config = load_outage_gpu_config(CONFIG_PATH)
    changed = deepcopy(config)
    changed["monitoring"]["history_interval_epochs"] = 11
    with pytest.raises(GPUOutageExperimentConfigError, match="pinned canonical"):
        validate_outage_gpu_config(changed)


def test_path_overrides_are_non_material(tmp_path: Path) -> None:
    development = tmp_path / "dev.zip"
    source = tmp_path / "source.zip"
    config = load_outage_gpu_config(
        CONFIG_PATH,
        development_release_override=development,
        source_override=source,
    )
    assert resolve_outage_development_release_path(config) == development.resolve()
    assert resolve_outage_source_path(config) == source.resolve()
    assert outage_gpu_config_sha256(config) == PINNED_OUTAGE_GPU_CONFIG_SHA256

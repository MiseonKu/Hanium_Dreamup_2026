from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from imu_vive_learning import gpu_outage_pord_v5_experiment as experiment
from imu_vive_learning import gpu_outage_pord_v5_training as training
from imu_vive_learning import gpu_outage_pord_v5_validator_amendment as amendment


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _state_arrays(backbone_id: str) -> tuple[Any, dict[str, np.ndarray], str]:
    import torch

    torch.manual_seed(7)
    model = training.build_pord_backbone(torch, backbone_id=backbone_id)
    arrays = {
        name: value.detach().cpu().numpy().copy(order="C")
        for name, value in model.state_dict().items()
    }
    return model, arrays, training.state_dict_sha256(arrays)


@pytest.mark.parametrize(
    ("backbone_id", "expected_count"),
    [("masked_mlp_64x64_relu", 6), ("chain_endpoint_only", 89)],
)
def test_strict_npz_adapter_loads_mlp_and_chain_without_casting(
    tmp_path: Path, backbone_id: str, expected_count: int
) -> None:
    import torch

    model, arrays, digest = _state_arrays(backbone_id)
    checkpoint = tmp_path / f"{backbone_id}.npz"
    np.savez_compressed(checkpoint, **arrays)
    with np.load(checkpoint, allow_pickle=False) as archive:
        reconstructed = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }

    assert len(reconstructed) == expected_count
    assert tuple(reconstructed) == tuple(model.state_dict())
    with pytest.raises((TypeError, RuntimeError)):
        model.load_state_dict(reconstructed, strict=True)

    converted = amendment.strict_pord_v5_ndarray_state_dict_to_tensors(
        model=model,
        state_dict=reconstructed,
        expected_state_sha256=digest,
    )
    assert len(converted) == expected_count
    assert all(type(value) is torch.Tensor for value in converted.values())
    assert training.state_dict_sha256(converted) == digest
    model.load_state_dict(converted, strict=True)


@pytest.mark.parametrize(
    "attack",
    [
        "extra",
        "missing",
        "reordered",
        "wrong_name",
        "wrong_shape",
        "wrong_dtype",
        "nonfinite",
        "non_array",
        "hash_tamper",
    ],
)
def test_strict_npz_adapter_rejects_malformed_or_tampered_state(
    attack: str,
) -> None:
    model, original, original_hash = _state_arrays("masked_mlp_64x64_relu")
    state = {name: value.copy() for name, value in original.items()}
    first = next(iter(state))
    if attack == "extra":
        state["unexpected.weight"] = np.zeros((1,), dtype=np.float32)
    elif attack == "missing":
        state.pop(first)
    elif attack == "reordered":
        state = dict(reversed(tuple(state.items())))
    elif attack == "wrong_name":
        value = state.pop(first)
        state = {"renamed.weight": value, **state}
    elif attack == "wrong_shape":
        state[first] = state[first].reshape(-1)
    elif attack == "wrong_dtype":
        state[first] = state[first].astype(np.float64)
    elif attack == "nonfinite":
        state[first].flat[0] = np.nan
    elif attack == "non_array":
        state[first] = state[first].tolist()  # type: ignore[assignment]
    elif attack == "hash_tamper":
        state[first].flat[0] += np.float32(1.0)
    else:  # pragma: no cover
        raise AssertionError(attack)

    expected_hash = original_hash
    if attack not in {"nonfinite", "non_array", "hash_tamper"}:
        try:
            expected_hash = training.state_dict_sha256(state)
        except training.PORDV5TrainingError:
            expected_hash = original_hash
    with pytest.raises(amendment.PORDV5ValidatorAmendmentError):
        amendment.strict_pord_v5_ndarray_state_dict_to_tensors(
            model=model,
            state_dict=state,
            expected_state_sha256=expected_hash,
        )


def test_original_48_file_authority_is_unchanged_and_separate() -> None:
    root = _workspace_root()
    original = experiment.pord_v5_source_payload(root)
    assert len(experiment.PORD_V5_SOURCE_FILES) == 48
    assert (
        original["fingerprint"]
        == amendment.PORD_V5_FROZEN_TRAINING_SOURCE_FINGERPRINT
        == "fe6756ed599cc113dfe0dabed3cddb96be037a5fb7a0d9aefac5d8e7052d27a3"
    )
    assert (
        experiment.sha256_file(
            root / "src/imu_vive_learning/gpu_outage_pord_v5_experiment.py"
        )
        == amendment.PORD_V5_FROZEN_EXPERIMENT_SHA256
    )
    assert (
        experiment.sha256_file(
            root / "src/imu_vive_learning/gpu_outage_pord_v5_training.py"
        )
        == amendment.PORD_V5_FROZEN_TRAINING_SHA256
    )
    added = amendment.pord_v5_validator_amendment_source_payload(root)
    assert len(added["files"]) == 3
    assert set(amendment.PORD_V5_VALIDATOR_AMENDMENT_SOURCE_FILES).isdisjoint(
        experiment.PORD_V5_SOURCE_FILES
    )
    assert added["fingerprint"] != original["fingerprint"]


def test_production_replay_audit_requires_168_checkpoints_and_840_views() -> None:
    audit = amendment._new_audit()
    audit["checkpoint_replay_count"] = 168
    audit["prediction_view_array_count"] = 840
    audit["checkpoint_tensor_array_count"] = 3000
    audit["backbone_checkpoint_counts"] = {
        "masked_mlp_64x64_relu": 144,
        "chain_endpoint_only": 24,
    }
    validated = amendment._validate_replay_audit(
        audit, expected_checkpoint_replays=168
    )
    assert validated["prediction_view_array_count"] == 840

    for key, value in (
        ("checkpoint_replay_count", 167),
        ("prediction_view_array_count", 839),
        ("checkpoint_tensor_array_count", 0),
    ):
        attacked = deepcopy(audit)
        attacked[key] = value
        with pytest.raises(amendment.PORDV5ValidatorAmendmentError):
            amendment._validate_replay_audit(
                attacked, expected_checkpoint_replays=168
            )


def _fake_predictor(**kwargs: Any) -> dict[str, Any]:
    import torch

    state = kwargs["training_result"]["state_dict"]
    assert state
    assert all(type(value) is torch.Tensor for value in state.values())
    return {"predictions_raw_mm": np.zeros((5, 2, 3), dtype=np.float64)}


def test_scoped_patch_adapts_arrays_and_restores_every_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    model, arrays, digest = _state_arrays("masked_mlp_64x64_relu")
    del model
    monkeypatch.setattr(amendment, "_FROZEN_PREDICTOR", _fake_predictor)
    monkeypatch.setattr(experiment, "predict_pord_checkpoint_outer", _fake_predictor)
    originals = {
        name: getattr(experiment, name)
        for name in amendment._FORBIDDEN_EXPERIMENT_GLOBALS
    }
    original_adamw = torch.optim.AdamW
    audit = amendment._new_audit()

    with amendment._scoped_completed_validation_patch(audit):
        output = experiment.predict_pord_checkpoint_outer(
            training_result={
                "backbone_id": "masked_mlp_64x64_relu",
                "state_dict": arrays,
                "selected_state_sha256": digest,
            }
        )
        assert output["predictions_raw_mm"].shape == (5, 2, 3)
        for name in amendment._FORBIDDEN_EXPERIMENT_GLOBALS:
            with pytest.raises(amendment.PORDV5ValidatorAmendmentError):
                getattr(experiment, name)()
        with pytest.raises(amendment.PORDV5ValidatorAmendmentError):
            torch.optim.AdamW([])

    assert experiment.predict_pord_checkpoint_outer is _fake_predictor
    assert all(getattr(experiment, name) is value for name, value in originals.items())
    assert torch.optim.AdamW is original_adamw
    assert audit["checkpoint_replay_count"] == 1
    assert audit["prediction_view_array_count"] == 5
    assert all(value == 1 for value in audit["forbidden_call_counts"].values())


def test_scoped_patch_setup_failure_preserves_original_cause_and_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    unexpected = lambda **kwargs: kwargs  # noqa: E731
    monkeypatch.setattr(experiment, "predict_pord_checkpoint_outer", unexpected)
    originals = {
        name: getattr(experiment, name)
        for name in amendment._FORBIDDEN_EXPERIMENT_GLOBALS
    }
    original_adamw = torch.optim.AdamW
    with pytest.raises(
        amendment.PORDV5ValidatorAmendmentError,
        match="original predictor identity changed before amendment",
    ):
        with amendment._scoped_completed_validation_patch(amendment._new_audit()):
            raise AssertionError("unreachable")
    assert experiment.predict_pord_checkpoint_outer is unexpected
    assert all(getattr(experiment, name) is value for name, value in originals.items())
    assert torch.optim.AdamW is original_adamw


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fake_completed_run(tmp_path: Path) -> Path:
    run = tmp_path / "v5_run"
    source = experiment.pord_v5_source_payload(_workspace_root())
    _write_json(run / "provenance/source_snapshot.json", source)
    _write_json(run / "development_manifest.json", {"complete": True})
    (run / "challenge").mkdir()
    (run / "payload.bin").write_bytes(b"immutable-v5-tree")
    return run


def _install_fake_completed_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, arrays, digest = _state_arrays("masked_mlp_64x64_relu")
    monkeypatch.setattr(amendment, "_FROZEN_PREDICTOR", _fake_predictor)
    monkeypatch.setattr(experiment, "predict_pord_checkpoint_outer", _fake_predictor)

    def fake_validator(
        run_dir: str | Path,
        *,
        verify_current_source: bool,
        allow_completed_challenge: bool,
        allow_synthetic_test_only: bool,
        verify_live_environment: bool,
    ) -> dict[str, Any]:
        del run_dir, verify_live_environment
        assert verify_current_source is True
        assert allow_completed_challenge is False
        assert allow_synthetic_test_only is True
        experiment.predict_pord_checkpoint_outer(
            training_result={
                "backbone_id": "masked_mlp_64x64_relu",
                "state_dict": {name: value.copy() for name, value in arrays.items()},
                "selected_state_sha256": digest,
            }
        )
        return {
            "status": "ok",
            "stage": "claim_gates_failed_challenge_sealed",
            "controlled_fit_count": 1,
            "challenge_labels_sealed": True,
            "challenge_label_access_passes": 0,
        }

    monkeypatch.setattr(experiment, "validate_pord_v5_development_run", fake_validator)


def test_pure_validation_and_sibling_receipt_do_not_mutate_v5_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _fake_completed_run(tmp_path)
    _install_fake_completed_validator(monkeypatch)
    before = amendment._exact_run_tree_manifest(run)
    receipt = amendment.validate_completed_pord_v5_with_amendment(
        run,
        workspace_root=_workspace_root(),
        verify_live_environment=False,
        _private_test_only=True,
        _private_expected_checkpoint_replays=1,
    )
    assert amendment._exact_run_tree_manifest(run) == before
    assert receipt["frozen_training_authority"]["training_source_fingerprint"] == (
        amendment.PORD_V5_FROZEN_TRAINING_SOURCE_FINGERPRINT
    )
    assert receipt["amendment_source"]["fingerprint"] != (
        receipt["frozen_training_authority"]["training_source_fingerprint"]
    )
    assert receipt["verify_current_source_false_call_count"] == 0
    assert receipt["scoped_patch_restored"] is True
    assert receipt["development_manifest_sha256"] == experiment.sha256_file(
        run / "development_manifest.json"
    )
    assert receipt[
        "training_optimizer_outer_evaluator_challenge_call_counts_all_zero"
    ] is True
    assert receipt["checkpoint_replay_audit"]["prediction_view_array_count"] == 5

    evidence = amendment.default_pord_v5_validator_amendment_evidence_dir(run)
    persisted = amendment.materialize_pord_v5_validator_amendment_receipt(
        run,
        evidence,
        workspace_root=_workspace_root(),
        verify_live_environment=False,
        _private_test_only=True,
        _private_expected_checkpoint_replays=1,
    )
    assert persisted["receipt_sha256"] == receipt["receipt_sha256"]
    assert evidence.parent == run.parent and evidence != run
    assert amendment._exact_run_tree_manifest(run) == before
    assert amendment.validate_pord_v5_validator_amendment_receipt(
        run,
        evidence,
        workspace_root=_workspace_root(),
        verify_live_environment=False,
        _private_test_only=True,
        _private_expected_checkpoint_replays=1,
    ) == persisted
    with pytest.raises(amendment.PORDV5ValidatorAmendmentError):
        amendment.materialize_pord_v5_validator_amendment_receipt(
            run,
            evidence,
            workspace_root=_workspace_root(),
            verify_live_environment=False,
            _private_test_only=True,
            _private_expected_checkpoint_replays=1,
        )

    (run / "payload.bin").write_bytes(b"attacker mutation")
    with pytest.raises(amendment.PORDV5ValidatorAmendmentError):
        amendment.validate_pord_v5_validator_amendment_receipt(
            run,
            evidence,
            workspace_root=_workspace_root(),
            verify_live_environment=False,
            _private_test_only=True,
            _private_expected_checkpoint_replays=1,
        )


def test_challenge_marker_and_source_fingerprint_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _fake_completed_run(tmp_path)
    _install_fake_completed_validator(monkeypatch)
    (run / "challenge/label_access_1.started.json").write_text("{}\n")
    with pytest.raises(amendment.PORDV5ValidatorAmendmentError):
        amendment.validate_completed_pord_v5_with_amendment(
            run,
            workspace_root=_workspace_root(),
            verify_live_environment=False,
            _private_test_only=True,
            _private_expected_checkpoint_replays=1,
        )

    (run / "challenge/label_access_1.started.json").unlink()
    source_path = run / "provenance/source_snapshot.json"
    source = json.loads(source_path.read_text())
    source["fingerprint"] = "0" * 64
    _write_json(source_path, source)
    with pytest.raises(amendment.PORDV5ValidatorAmendmentError):
        amendment.validate_completed_pord_v5_with_amendment(
            run,
            workspace_root=_workspace_root(),
            verify_live_environment=False,
            _private_test_only=True,
            _private_expected_checkpoint_replays=1,
        )


def test_cli_has_validation_actions_but_no_resume() -> None:
    parser = amendment._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--run-dir",
                "run",
                "--evidence-dir",
                "run.validator_amendment_v1",
                "--create-receipt",
                "--resume",
            ]
        )

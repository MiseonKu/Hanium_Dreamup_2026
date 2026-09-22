from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import pytest
from numpy.lib import format as numpy_format

from imu_vive_learning.cli import build_parser, main
from imu_vive_learning.experiment_io import stable_hash
from imu_vive_learning.features import FUSION_COLUMNS, LABEL_COLUMNS
from imu_vive_learning.gpu_inference import (
    GPUInferenceError,
    GPU_MODEL_CONTRACT,
    PREDICTION_COLUMNS,
    load_gpu_endpoint_model,
    predict_gpu_endpoint,
)


torch = pytest.importorskip("torch")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _state_arrays(
    hidden: tuple[int, int], output_bias: tuple[float, float, float]
) -> dict[str, np.ndarray]:
    widths = (len(FUSION_COLUMNS), *hidden, 3)
    state: dict[str, np.ndarray] = {}
    for layer_index in range(len(widths) - 1):
        module_index = 2 * layer_index
        state[f"{module_index}.weight"] = np.zeros(
            (widths[layer_index + 1], widths[layer_index]), dtype=np.float32
        )
        bias = np.zeros(widths[layer_index + 1], dtype=np.float32)
        if layer_index == len(widths) - 2:
            bias[:] = np.asarray(output_bias, dtype=np.float32)
        state[f"{module_index}.bias"] = bias
    return state


def _write_fixture(root: Path) -> Path:
    hidden = (64, 64)
    params = {"hidden_layer_sizes": list(hidden), "weight_decay": 0.001}
    preprocessing = {
        "feature_names": list(FUSION_COLUMNS),
        "feature_mean": [0.0] * len(FUSION_COLUMNS),
        "feature_scale": [1.0] * len(FUSION_COLUMNS),
        "target_mean": [10.0, 20.0, 30.0],
        "target_scale": 2.0,
        "leakage_audit": {
            "fit_scope": "real_train_only",
            "split_context": "all_eligible_real",
            "fit_row_count": 7702,
            "fit_trial_uid_sha256": "a" * 64,
            "forbidden_uid_count": 0,
            "forbidden_uid_overlap_count": 0,
            "feature_scaling_enabled": True,
            "target_centered_per_axis": True,
            "target_scale_kind": "single_isotropic_xyz_scale",
        },
    }
    preprocessing_path = root / "final" / "preprocessing.json"
    _write_json(preprocessing_path, preprocessing)

    checkpoint_entries = []
    for seed_index in range(3):
        checkpoint = (
            root / "final" / "checkpoints" / f"torch_mlp_seed{seed_index}.npz"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            checkpoint,
            **_state_arrays(
                hidden,
                (
                    float(seed_index),
                    float(seed_index + 1),
                    float(seed_index + 2),
                ),
            ),
        )
        checkpoint_entries.append(
            {
                "seed_index": seed_index,
                "model_seed": 1000 + seed_index,
                "trained_epochs": 20 + seed_index,
                "train_rmse_3d_mm": 2.0 + seed_index,
                "path": f"final/checkpoints/torch_mlp_seed{seed_index}.npz",
                "sha256": _sha256(checkpoint),
                "tensor_count": 6,
            }
        )

    manifest = {
        "contract": GPU_MODEL_CONTRACT,
        "model_family": "torch_mlp",
        "ensemble_kind": "mean_raw_vive_world_xyz",
        "ensemble_seed_count": 3,
        "feature_view": "fusion_v1",
        "feature_names": list(FUSION_COLUMNS),
        "label_columns": list(LABEL_COLUMNS),
        "label_frame": "vive_world",
        "label_unit": "mm",
        "selected_config_id": stable_hash(
            {"model": "torch_mlp", "params": params}
        )[:16],
        "selected_params": params,
        "selection_source": "20_segment_inner_validation_contexts",
        "selection_score_mm": 12.5,
        "independent_test_score_available": False,
        "preprocessing": {
            "path": "final/preprocessing.json",
            "sha256": _sha256(preprocessing_path),
        },
        "checkpoints": checkpoint_entries,
    }
    _write_json(root / "final" / "model_manifest.json", manifest)
    return root


def _read_manifest(root: Path) -> dict[str, Any]:
    return json.loads(
        (root / "final" / "model_manifest.json").read_text(encoding="utf-8")
    )


def _write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    _write_json(root / "final" / "model_manifest.json", manifest)


def _feature_frame(rows: int = 2, *, row_ids: bool = False) -> pd.DataFrame:
    matrix = np.arange(rows * len(FUSION_COLUMNS), dtype=np.float64).reshape(
        rows, len(FUSION_COLUMNS)
    )
    frame = pd.DataFrame(matrix, columns=FUSION_COLUMNS)
    if row_ids:
        frame.insert(0, "row_id", [f"row-{index}" for index in range(rows)])
    return frame


def test_load_and_predict_exact_raw_xyz_mean_on_cpu(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "run")
    ensemble = load_gpu_endpoint_model(root, device="cpu")

    prediction, row_ids = ensemble.predict_frame(_feature_frame(row_ids=True))

    assert str(ensemble.device) == "cpu"
    assert len(ensemble.models) == 3
    assert all(
        not parameter.requires_grad
        for model in ensemble.models
        for parameter in model.parameters()
    )
    assert prediction == pytest.approx(np.tile([12.0, 24.0, 36.0], (2, 1)))
    assert row_ids is not None
    assert row_ids.tolist() == ["row-0", "row-1"]
    assert len(ensemble.model_fingerprint) == 64


def test_parquet_helper_preserves_row_ids_and_output_contract(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "run")
    source = tmp_path / "features.parquet"
    output = tmp_path / "predictions.parquet"
    _feature_frame(row_ids=True).to_parquet(source, index=False)

    summary = predict_gpu_endpoint(root, source, output, device="cpu", batch_size=1)
    result = pd.read_parquet(output)

    assert summary["status"] == "ok"
    assert summary["contract"] == GPU_MODEL_CONTRACT
    assert summary["rows"] == 2
    assert result.columns.tolist() == [
        "row_id",
        *PREDICTION_COLUMNS,
        "model_contract_fingerprint",
        "output_frame",
        "output_unit",
    ]
    assert result.loc[:, list(PREDICTION_COLUMNS)].to_numpy() == pytest.approx(
        np.tile([12.0, 24.0, 36.0], (2, 1))
    )
    assert result["output_frame"].unique().tolist() == ["vive_world"]
    assert result["output_unit"].unique().tolist() == ["mm"]


def test_gpu_inference_cli_shape_and_success_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_fixture(tmp_path / "run")
    source = tmp_path / "features.parquet"
    output = tmp_path / "predictions.parquet"
    _feature_frame(row_ids=True).to_parquet(source, index=False)
    parsed = build_parser().parse_args(
        [
            "predict-gpu-endpoint",
            "--run-dir",
            str(root),
            "--input",
            str(source),
            "--output",
            str(output),
        ]
    )
    assert parsed.device == "cpu"
    assert parsed.batch_size == 4096

    exit_code = main(
        [
            "predict-gpu-endpoint",
            "--run-dir",
            str(root),
            "--input",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--batch-size",
            "1",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "status": "ok",
        "contract": GPU_MODEL_CONTRACT,
        "rows": 2,
        "output": str(output.resolve()),
        "model_contract_fingerprint": payload["model_contract_fingerprint"],
        "frame": "vive_world",
        "unit": "mm",
    }
    assert len(payload["model_contract_fingerprint"]) == 64
    assert output.is_file()


def test_gpu_inference_cli_error_is_stable_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _write_fixture(tmp_path / "run")
    source = tmp_path / "features.parquet"
    output = tmp_path / "predictions.parquet"
    _feature_frame().to_parquet(source, index=False)

    exit_code = main(
        [
            "predict-gpu-endpoint",
            "--run-dir",
            str(root),
            "--input",
            str(source),
            "--output",
            str(output),
            "--batch-size",
            "0",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.err)

    assert exit_code == 1
    assert captured.out == ""
    assert payload == {
        "status": "error",
        "command": "predict-gpu-endpoint",
        "error": "GPU inference batch_size must be a positive integer",
    }
    assert not output.exists()


def test_inference_frame_is_strict_about_order_types_and_finite_values(
    tmp_path: Path,
) -> None:
    ensemble = load_gpu_endpoint_model(_write_fixture(tmp_path / "run"))
    valid = _feature_frame()

    with pytest.raises(GPUInferenceError, match="schema/order mismatch"):
        ensemble.predict_frame(valid.loc[:, list(reversed(FUSION_COLUMNS))])

    extra = valid.copy()
    extra["planned_p_x_mm"] = 0.0
    with pytest.raises(GPUInferenceError, match="schema/order mismatch"):
        ensemble.predict_frame(extra)

    strings = valid.copy()
    strings[FUSION_COLUMNS[0]] = ["1", "2"]
    with pytest.raises(GPUInferenceError, match="real numeric"):
        ensemble.predict_frame(strings)

    booleans = valid.copy()
    booleans[FUSION_COLUMNS[0]] = [True, False]
    with pytest.raises(GPUInferenceError, match="real numeric"):
        ensemble.predict_frame(booleans)

    nonfinite = valid.copy()
    nonfinite.loc[0, FUSION_COLUMNS[0]] = np.inf
    with pytest.raises(GPUInferenceError, match="must be finite"):
        ensemble.predict_frame(nonfinite)

    duplicated = valid.copy()
    duplicated.columns = [FUSION_COLUMNS[0], *FUSION_COLUMNS[:-1]]
    with pytest.raises(GPUInferenceError, match="columns must be unique"):
        ensemble.predict_frame(duplicated)

    repeated_ids = valid.copy()
    repeated_ids.insert(0, "row_id", ["same", "same"])
    with pytest.raises(GPUInferenceError, match="non-missing and unique"):
        ensemble.predict_frame(repeated_ids)


def test_preprocessing_and_checkpoint_hash_drift_fail_closed(tmp_path: Path) -> None:
    preprocessing_root = _write_fixture(tmp_path / "preprocessing")
    preprocessing_path = preprocessing_root / "final" / "preprocessing.json"
    preprocessing_path.write_bytes(preprocessing_path.read_bytes() + b" \n")
    with pytest.raises(GPUInferenceError, match="preprocessing SHA-256 mismatch"):
        load_gpu_endpoint_model(preprocessing_root)

    checkpoint_root = _write_fixture(tmp_path / "checkpoint")
    checkpoint_path = (
        checkpoint_root / "final" / "checkpoints" / "torch_mlp_seed0.npz"
    )
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b"drift")
    with pytest.raises(GPUInferenceError, match="checkpoint 0 SHA-256 mismatch"):
        load_gpu_endpoint_model(checkpoint_root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("contract", "wrong.contract", "manifest 'contract' changed"),
        ("feature_names", list(reversed(FUSION_COLUMNS)), "feature allow-list/order"),
        ("ensemble_seed_count", 2, "ensemble_seed_count"),
        ("independent_test_score_available", True, "independent_test_score"),
        ("unexpected", 1, "field set changed"),
    ],
)
def test_manifest_contract_drift_fails_closed(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    root = _write_fixture(tmp_path / field)
    manifest = _read_manifest(root)
    manifest[field] = value
    _write_manifest(root, manifest)

    with pytest.raises(GPUInferenceError, match=message):
        load_gpu_endpoint_model(root)


def test_parameter_identity_seed_order_and_paths_are_verified(tmp_path: Path) -> None:
    config_root = _write_fixture(tmp_path / "config")
    config_manifest = _read_manifest(config_root)
    config_manifest["selected_config_id"] = "0" * 16
    _write_manifest(config_root, config_manifest)
    with pytest.raises(GPUInferenceError, match="disagrees with parameters"):
        load_gpu_endpoint_model(config_root)

    seed_root = _write_fixture(tmp_path / "seed")
    seed_manifest = _read_manifest(seed_root)
    seed_manifest["checkpoints"] = list(reversed(seed_manifest["checkpoints"]))
    _write_manifest(seed_root, seed_manifest)
    with pytest.raises(GPUInferenceError, match="seed order changed"):
        load_gpu_endpoint_model(seed_root)

    path_root = _write_fixture(tmp_path / "path")
    path_manifest = _read_manifest(path_root)
    path_manifest["checkpoints"][0]["path"] = "../torch_mlp_seed0.npz"
    _write_manifest(path_root, path_manifest)
    with pytest.raises(GPUInferenceError, match="member path changed"):
        load_gpu_endpoint_model(path_root)


def _replace_checkpoint(
    root: Path, seed_index: int, state: dict[str, np.ndarray]
) -> None:
    checkpoint = (
        root / "final" / "checkpoints" / f"torch_mlp_seed{seed_index}.npz"
    )
    np.savez_compressed(checkpoint, **state)
    manifest = _read_manifest(root)
    manifest["checkpoints"][seed_index]["sha256"] = _sha256(checkpoint)
    _write_manifest(root, manifest)


def test_checkpoint_shape_dtype_and_finite_tensors_are_verified(
    tmp_path: Path,
) -> None:
    shape_root = _write_fixture(tmp_path / "shape")
    wrong_shape = _state_arrays((64, 64), (0.0, 1.0, 2.0))
    wrong_shape["0.weight"] = np.zeros((63, 43), dtype=np.float32)
    _replace_checkpoint(shape_root, 0, wrong_shape)
    with pytest.raises(GPUInferenceError, match="header changed"):
        load_gpu_endpoint_model(shape_root)

    dtype_root = _write_fixture(tmp_path / "dtype")
    wrong_dtype = _state_arrays((64, 64), (0.0, 1.0, 2.0))
    wrong_dtype["0.weight"] = wrong_dtype["0.weight"].astype(np.float64)
    _replace_checkpoint(dtype_root, 0, wrong_dtype)
    with pytest.raises(GPUInferenceError, match="header changed"):
        load_gpu_endpoint_model(dtype_root)

    finite_root = _write_fixture(tmp_path / "finite")
    nonfinite = _state_arrays((64, 64), (0.0, 1.0, 2.0))
    nonfinite["0.weight"][0, 0] = np.nan
    _replace_checkpoint(finite_root, 0, nonfinite)
    with pytest.raises(GPUInferenceError, match="non-finite"):
        load_gpu_endpoint_model(finite_root)


def test_object_array_checkpoint_is_rejected_before_unpickling(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "object")
    state = _state_arrays((64, 64), (0.0, 1.0, 2.0))
    state["0.weight"] = np.asarray([[{"unsafe": True}]], dtype=object)
    _replace_checkpoint(root, 0, state)

    with pytest.raises(GPUInferenceError, match="header changed"):
        load_gpu_endpoint_model(root)


def test_malicious_large_npy_shape_is_rejected_before_numpy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_fixture(tmp_path / "large-header")
    checkpoint = root / "final" / "checkpoints" / "torch_mlp_seed0.npz"
    with ZipFile(checkpoint) as archive:
        members = [(info.filename, archive.read(info)) for info in archive.infolist()]
    header = BytesIO()
    numpy_format.write_array_header_1_0(
        header,
        {
            "descr": "<f4",
            "fortran_order": False,
            "shape": (2**40, len(FUSION_COLUMNS)),
        },
    )
    malicious_member = header.getvalue() + b"\x00\x00\x00\x00"
    with ZipFile(checkpoint, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in members:
            archive.writestr(
                name, malicious_member if name == "0.weight.npy" else payload
            )
    manifest = _read_manifest(root)
    manifest["checkpoints"][0]["sha256"] = _sha256(checkpoint)
    _write_manifest(root, manifest)
    original_load = np.load
    called = False

    def tracked_load(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", tracked_load)
    with pytest.raises(GPUInferenceError, match="header changed"):
        load_gpu_endpoint_model(root)
    assert called is False


def test_json_contract_values_require_exact_scalar_types(tmp_path: Path) -> None:
    manifest_root = _write_fixture(tmp_path / "manifest-type")
    manifest = _read_manifest(manifest_root)
    manifest["ensemble_seed_count"] = 3.0
    _write_manifest(manifest_root, manifest)
    with pytest.raises(GPUInferenceError, match="ensemble_seed_count"):
        load_gpu_endpoint_model(manifest_root)

    audit_root = _write_fixture(tmp_path / "audit-type")
    preprocessing_path = audit_root / "final" / "preprocessing.json"
    preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    preprocessing["leakage_audit"]["feature_scaling_enabled"] = 1
    _write_json(preprocessing_path, preprocessing)
    audit_manifest = _read_manifest(audit_root)
    audit_manifest["preprocessing"]["sha256"] = _sha256(preprocessing_path)
    _write_manifest(audit_root, audit_manifest)
    with pytest.raises(GPUInferenceError, match="leakage audit changed"):
        load_gpu_endpoint_model(audit_root)

    seed_root = _write_fixture(tmp_path / "seed-type")
    seed_manifest = _read_manifest(seed_root)
    seed_manifest["checkpoints"][0]["seed_index"] = 0.0
    _write_manifest(seed_root, seed_manifest)
    with pytest.raises(GPUInferenceError, match="seed order changed"):
        load_gpu_endpoint_model(seed_root)


def test_cuda_request_fails_before_device_use_when_cuda_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_fixture(tmp_path / "run")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(GPUInferenceError, match="CUDA is unavailable"):
        load_gpu_endpoint_model(root, device="cuda:0")


def test_manifest_input_path_is_supported_without_loading_pickle(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "run")
    manifest_path = root / "final" / "model_manifest.json"

    ensemble = load_gpu_endpoint_model(manifest_path)

    assert ensemble.predict(np.zeros((1, 43))) == pytest.approx(
        np.asarray([[12.0, 24.0, 36.0]])
    )


def test_unpromoted_robust_checkpoint_is_diagnostic_only(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "robust")
    _write_json(
        root / "config.resolved.json",
        {"contract": {"name": "alphaf.endpoint_regression.gpu_experiment.robust_v2"}},
    )
    recipe = {
        "contract": "alphaf.endpoint_regression.gpu_training.robust_v2",
        "promotion_eligible": False,
        "promotion_action": "retain_v1",
    }
    recipe_path = root / "final" / "training_recipe.json"
    _write_json(recipe_path, recipe)
    _write_json(
        root / "manifest.json",
        {
            "contract": "alphaf.endpoint_regression.gpu_experiment.robust_v2",
            "training_recipe": {
                "path": "final/training_recipe.json",
                "sha256": _sha256(recipe_path),
            },
        },
    )

    with pytest.raises(GPUInferenceError, match="retain_v1 forbids inference"):
        load_gpu_endpoint_model(root)
    diagnostic = load_gpu_endpoint_model(
        root, allow_unpromoted_diagnostic=True
    )
    assert diagnostic.predict(np.zeros((1, 43))).shape == (1, 3)


def test_copied_robust_model_cannot_bypass_promotion_gate(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path / "copied-robust")
    candidate = {"candidate_id": "balanced_erm", "objective": "balanced_erm"}
    recipe = {
        "contract": "alphaf.endpoint_regression.gpu_training.robust_v2",
        "candidate_id": "balanced_erm",
        "candidate": candidate,
        "promotion_eligible": False,
        "promotion_action": "retain_v1",
    }
    recipe_path = root / "final" / "training_recipe.json"
    _write_json(recipe_path, recipe)
    manifest = _read_manifest(root)
    manifest.update(
        {
            "contract": "alphaf.endpoint_regression.gpu_model.robust_v2",
            "candidate_id": "balanced_erm",
            "candidate": candidate,
            "promotion_eligible": False,
            "promotion_action": "retain_v1",
            "training_recipe": {
                "path": "final/training_recipe.json",
                "sha256": _sha256(recipe_path),
            },
        }
    )
    _write_manifest(root, manifest)

    with pytest.raises(GPUInferenceError, match="retain_v1 forbids inference"):
        load_gpu_endpoint_model(root)
    diagnostic = load_gpu_endpoint_model(
        root, allow_unpromoted_diagnostic=True
    )
    assert diagnostic.manifest["contract"].endswith("robust_v2")

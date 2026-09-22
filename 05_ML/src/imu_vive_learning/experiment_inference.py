"""Strict offline inference from a validated experiment package."""

from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd

from .experiment_config import EXPERIMENT_CONTRACT
from .experiment_ensemble import SerializableEndpointEnsemble
from .experiment_io import atomic_write_parquet, sha256_file
from .experiment_packaging import EXPERIMENT_PACKAGE_ROOT, validate_experiment_package
from .experiment_runner import PREDICTION_COLUMNS


class EndpointInferenceError(ValueError):
    """Raised when package or feature input does not match the final model."""


def predict_endpoint(
    model_package: str | Path,
    input_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    package = Path(model_package).resolve()
    source = Path(input_path).resolve()
    output = Path(output_path).resolve()
    package_validation = validate_experiment_package(package)
    manifest = package_validation["manifest"]
    if manifest.get("contract") != EXPERIMENT_CONTRACT:
        raise EndpointInferenceError("unexpected model-package contract")
    final = manifest.get("final_model")
    if not isinstance(final, dict):
        raise EndpointInferenceError("model package lacks final model metadata")
    member_name = f"{EXPERIMENT_PACKAGE_ROOT}/final/model.joblib"
    metadata_name = f"{EXPERIMENT_PACKAGE_ROOT}/final/model_metadata.json"
    with ZipFile(package) as archive:
        try:
            model_payload = archive.read(member_name)
            metadata = json.loads(archive.read(metadata_name).decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EndpointInferenceError("cannot read final model package members") from exc
    model_hash = hashlib.sha256(model_payload).hexdigest()
    if model_hash != metadata.get("model", {}).get("sha256"):
        raise EndpointInferenceError("final model member hash disagrees with metadata")
    with tempfile.TemporaryDirectory(prefix="alphaf_endpoint_inference_") as directory:
        model_path = Path(directory) / "model.joblib"
        model_path.write_bytes(model_payload)
        model = joblib.load(model_path)
    if not isinstance(model, SerializableEndpointEnsemble):
        raise EndpointInferenceError("final model wrapper type changed")
    if model.contract_fingerprint() != metadata.get("model", {}).get(
        "contract_fingerprint"
    ):
        raise EndpointInferenceError("final model contract fingerprint changed")
    try:
        frame = pd.read_parquet(source)
    except FileNotFoundError as exc:
        raise EndpointInferenceError(f"feature input does not exist: {source}") from exc
    prediction, row_ids = model.predict_frame(frame)
    if prediction.shape != (len(frame), 3) or not np.isfinite(prediction).all():
        raise EndpointInferenceError("model produced invalid finite XYZ output")
    result = pd.DataFrame(
        {
            PREDICTION_COLUMNS[0]: prediction[:, 0],
            PREDICTION_COLUMNS[1]: prediction[:, 1],
            PREDICTION_COLUMNS[2]: prediction[:, 2],
        }
    )
    if row_ids is not None:
        result.insert(0, "row_id", row_ids)
    package_hash = sha256_file(package)
    result["model_sha256"] = model_hash
    result["model_contract_fingerprint"] = model.contract_fingerprint()
    result["model_package_sha256"] = package_hash
    result["output_frame"] = "vive_world"
    result["output_unit"] = "mm"
    atomic_write_parquet(output, result)
    return {
        "status": "ok",
        "contract": EXPERIMENT_CONTRACT,
        "rows": int(len(result)),
        "output": str(output),
        "model_sha256": model_hash,
        "model_package_sha256": package_hash,
        "frame": "vive_world",
        "unit": "mm",
    }


__all__ = ["EndpointInferenceError", "predict_endpoint"]

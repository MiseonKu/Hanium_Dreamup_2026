"""Fast engineering refit that excludes imu2_link_mid from the feature contract."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "03_Dataset_Collection/imu_added_experiment/ml_preparation_v1.zip"
MEMBER = "ml_preparation_v1/canonical/canonical_real.parquet"
ARTIFACT = ROOT / "05_ML/artifacts/three_imu_fast_v1"
DEPLOY = ROOT / "06_Position_Estimation/data"
HOLDOUT_TARGETS = ("T002", "T032", "T034", "T015", "T044")
SEED = 20_260_923


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit(X: np.ndarray, y: np.ndarray) -> tuple[MLPRegressor, np.ndarray, np.ndarray, np.ndarray, float]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0.0] = 1.0
    ymean = y.mean(axis=0)
    yscale = float(np.sqrt(np.mean(np.square(y - ymean)))) or 1.0
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64), activation="relu", solver="adam",
        alpha=0.001, batch_size=128, learning_rate="constant",
        learning_rate_init=0.001, max_iter=400, shuffle=True,
        random_state=SEED, tol=0.0, early_stopping=False,
        n_iter_no_change=401,
    )
    model.fit((X - mean) / scale, (y - ymean) / yscale)
    return model, mean, scale, ymean, yscale


def predict(bundle, X: np.ndarray) -> np.ndarray:
    model, mean, scale, ymean, yscale = bundle
    return model.predict((X - mean) / scale) * yscale + ymean


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    delta = pred - y
    distance = np.linalg.norm(delta, axis=1)
    return {
        "rows": int(len(y)),
        "rmse_3d_mm": float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))),
        "mean_3d_error_mm": float(distance.mean()),
        "median_3d_error_mm": float(np.median(distance)),
        "p95_3d_error_mm": float(np.quantile(distance, 0.95)),
        "bias_x_mm": float(delta[:, 0].mean()),
        "bias_y_mm": float(delta[:, 1].mean()),
        "bias_z_mm": float(delta[:, 2].mean()),
    }


def main() -> None:
    with zipfile.ZipFile(SOURCE) as archive:
        frame = pd.read_parquet(io.BytesIO(archive.read(MEMBER)))
    frame = frame.loc[frame["train_eligible"].fillna(False)].copy()
    old_meta = json.loads((DEPLOY / "model.json").read_text(encoding="utf-8"))
    features = [name for name in old_meta["feature_names"] if not name.startswith("imu2_link_mid_")]
    if len(frame) != 7702 or len(features) != 33:
        raise RuntimeError(f"unexpected contract: rows={len(frame)}, features={len(features)}")
    X = frame.loc[:, features].to_numpy(dtype=np.float64)
    y = frame.loc[:, ["vive_pos_raw_x_mm", "vive_pos_raw_y_mm", "vive_pos_raw_z_mm"]].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise RuntimeError("non-finite training data")

    holdout = frame["target_id"].isin(HOLDOUT_TARGETS).to_numpy()
    validation_bundle = fit(X[~holdout], y[~holdout])
    validation_pred = predict(validation_bundle, X[holdout])
    report: dict[str, object] = {
        "status": "rapid_engineering_validation_not_independent_control_validation",
        "excluded_sensor": "imu2_link_mid",
        "feature_count": len(features),
        "fit_rows_without_five_targets": int((~holdout).sum()),
        "holdout_rows": int(holdout.sum()),
        "holdout_targets": list(HOLDOUT_TARGETS),
        "holdout_all": metrics(y[holdout], validation_pred),
        "holdout_by_target": {},
    }
    holdout_frame = frame.loc[holdout].reset_index(drop=True)
    for target in HOLDOUT_TARGETS:
        mask = holdout_frame["target_id"].eq(target).to_numpy()
        report["holdout_by_target"][target] = metrics(y[holdout][mask], validation_pred[mask])

    final_bundle = fit(X, y)
    model, mean, scale, ymean, yscale = final_bundle
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    DEPLOY.mkdir(parents=True, exist_ok=True)
    weights = {
        "feature_min": X.min(axis=0), "feature_max": X.max(axis=0),
        "mean_0": mean, "scale_0": scale, "ymean_0": ymean,
        "yscale_0": np.asarray(yscale),
    }
    for index, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        weights[f"w_0_{index}"] = coef
        weights[f"b_0_{index}"] = intercept
    artifact_weights = ARTIFACT / "weights.npz"
    np.savez_compressed(artifact_weights, **weights)
    deploy_weights = DEPLOY / "weights.npz"
    deploy_weights.write_bytes(artifact_weights.read_bytes())
    weights_hash = sha256(deploy_weights)
    metadata = {
        "model_id": "three_imu_fast_v1",
        "source_dataset_sha256": sha256(SOURCE),
        "weights_sha256": weights_hash,
        "excluded_sensor": "imu2_link_mid",
        "feature_names": features,
        "fit_rows": int(len(frame)),
        "ensemble_members": 1,
        "seed": SEED,
        "architecture": {"hidden_layer_sizes": [64, 64], "alpha": 0.001, "max_iter": 400},
        "output_frame": "training_vive_world",
        "output_unit": "mm",
        "rapid_validation": report,
        "independent_control_validation": False,
    }
    text = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    (ARTIFACT / "model.json").write_text(text, encoding="utf-8")
    (ARTIFACT / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (DEPLOY / "model.json").write_text(text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"final_iterations={model.n_iter_}")
    print(f"deployed_weights_sha256={weights_hash}")


if __name__ == "__main__":
    main()

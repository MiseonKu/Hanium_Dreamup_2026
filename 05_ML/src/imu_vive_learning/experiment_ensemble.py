"""Serializable deterministic ensemble wrapper for final endpoint inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from .experiment_models import SerializableEndpointRegressor


@dataclass
class SerializableEndpointEnsemble:
    members: tuple[SerializableEndpointRegressor, ...]
    model_name: str
    recipe: str
    feature_names: tuple[str, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("endpoint ensemble requires at least one member")
        expected = tuple(self.feature_names)
        for member in self.members:
            if tuple(member.feature_names) != expected:
                raise ValueError("endpoint ensemble members have different feature contracts")
            if member.model_name != self.model_name or member.recipe != self.recipe:
                raise ValueError("endpoint ensemble member identity changed")

    def predict(self, X: Any) -> np.ndarray:
        predictions = [member.predict(X) for member in self.members]
        result = np.mean(np.stack(predictions, axis=0), axis=0)
        if result.ndim != 2 or result.shape[1] != 3 or not np.isfinite(result).all():
            raise ValueError("endpoint ensemble produced invalid predictions")
        return result

    def validate_inference_frame(self, frame: Any) -> tuple[np.ndarray, np.ndarray | None]:
        return self.members[0].validate_inference_frame(frame)

    def predict_frame(self, frame: Any) -> tuple[np.ndarray, np.ndarray | None]:
        matrix, row_ids = self.validate_inference_frame(frame)
        return self.predict(matrix), row_ids

    def contract_fingerprint(self) -> str:
        payload = {
            "model_name": self.model_name,
            "recipe": self.recipe,
            "feature_names": list(self.feature_names),
            "member_fingerprints": [
                member.contract_fingerprint() for member in self.members
            ],
            "metadata": self.metadata,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["SerializableEndpointEnsemble"]

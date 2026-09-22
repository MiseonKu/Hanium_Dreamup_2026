"""Post-run validation amendment for frozen PORD-v5 checkpoints.

The completed v5 run was trained under the original 48-file source closure.
That closure is never replaced here.  This module supplies a narrowly scoped
validator adapter for the persisted NPZ boundary: arrays are checked against
the exact model state schema and converted to tensors only while the original
completed-run validator replays checkpoints.  No fit, optimizer, outer
evaluator, challenge prediction, or finalizer is authorized.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import threading
from typing import Any, Iterator

import numpy as np

from .experiment_io import read_json_object, sha256_file, stable_hash
from . import gpu_outage_pord_v5_experiment as _experiment
from . import gpu_outage_pord_v5_training as _training


PORD_V5_VALIDATOR_AMENDMENT_CONTRACT = (
    "alphaf.endpoint_regression.gpu_outage_pord.validator_amendment.v1"
)
PORD_V5_FROZEN_TRAINING_SOURCE_FINGERPRINT = (
    "fe6756ed599cc113dfe0dabed3cddb96be037a5fb7a0d9aefac5d8e7052d27a3"
)
PORD_V5_FROZEN_EXPERIMENT_SHA256 = (
    "09541553ee538947d4d15942f47a28d6e13f0a13cf5dceba0d0725214aabd6af"
)
PORD_V5_FROZEN_TRAINING_SHA256 = (
    "bd64a7e3bdb38e37593058347970e452e980e7de64fff65e80e6eff5122ec5f5"
)
PORD_V5_FROZEN_SOURCE_FILE_COUNT = 48
PORD_V5_EXPECTED_CHECKPOINT_REPLAYS = 168
PORD_V5_EXPECTED_PREDICTION_VIEW_ARRAYS = 840
PORD_V5_EXPECTED_BACKBONE_CHECKPOINTS = {
    "masked_mlp_64x64_relu": 144,
    "chain_endpoint_only": 24,
}
PORD_V5_VALIDATOR_AMENDMENT_SOURCE_FILES = (
    "docs/GPU_ENDPOINT_REGRESSION_OUTAGE_V5_VALIDATOR_AMENDMENT_V1.md",
    "src/imu_vive_learning/gpu_outage_pord_v5_validator_amendment.py",
    "tests/test_gpu_outage_pord_v5_validator_amendment.py",
)
PORD_V5_VALIDATOR_AMENDMENT_RECEIPT = "VALIDATION_AMENDMENT.completed.json"

_FROZEN_EXPERIMENT_RELATIVE = (
    "src/imu_vive_learning/gpu_outage_pord_v5_experiment.py"
)
_FROZEN_TRAINING_RELATIVE = (
    "src/imu_vive_learning/gpu_outage_pord_v5_training.py"
)
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PATCH_LOCK = threading.Lock()
_FROZEN_PREDICTOR = _experiment.predict_pord_checkpoint_outer
_FORBIDDEN_EXPERIMENT_GLOBALS = (
    "train_pord_fold_seed",
    "evaluate_pord_outer_predictions_once",
    "run_pord_v5_development",
    "run_pord_v5_experiment",
    "freeze_pord_v5_challenge_predictions",
    "finalize_pord_v5_challenge",
    "_predict_pord_challenge_component",
    "_predict_v3_incumbent_challenge",
)


class PORDV5ValidatorAmendmentError(RuntimeError):
    """Raised when the post-run validator amendment fails closed."""


def _workspace_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / _FROZEN_EXPERIMENT_RELATIVE).is_file():
        raise PORDV5ValidatorAmendmentError("cannot locate the v5 source workspace")
    return root


def _exact_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or _LOWER_SHA256.fullmatch(value) is None:
        raise PORDV5ValidatorAmendmentError(
            f"{name} must be an exact lowercase SHA-256"
        )
    return value


def _exact_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PORDV5ValidatorAmendmentError(f"{name} must be an exact integer")
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PORDV5ValidatorAmendmentError(
            "amendment receipt must be finite JSON"
        ) from exc


def pord_v5_validator_amendment_source_payload(
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Hash the separate three-file amendment closure."""

    root = _workspace_root() if workspace_root is None else Path(workspace_root).resolve()
    files: list[dict[str, Any]] = []
    for relative in PORD_V5_VALIDATOR_AMENDMENT_SOURCE_FILES:
        token = PurePosixPath(relative)
        if token.is_absolute() or ".." in token.parts:
            raise PORDV5ValidatorAmendmentError("unsafe amendment source path")
        path = (root / Path(*token.parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise PORDV5ValidatorAmendmentError(
                "amendment source escapes workspace"
            ) from exc
        if not path.is_file() or path.is_symlink():
            raise PORDV5ValidatorAmendmentError(
                f"amendment source is missing or symbolic: {relative}"
            )
        files.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return {"files": files, "fingerprint": stable_hash(files)}


def strict_pord_v5_ndarray_state_dict_to_tensors(
    *,
    model: Any,
    state_dict: Any,
    expected_state_sha256: Any,
) -> dict[str, Any]:
    """Convert one exact NPZ-reconstructed state to CPU tensors.

    Names, order, tensor count, shape, dtype, C-contiguity, finiteness, and the
    pre-recorded state hash must all match the freshly built backbone.  No
    numeric cast, reshape, missing-key fill, or extra-key tolerance exists.
    """

    import torch

    expected_hash = _exact_sha256(
        expected_state_sha256, name="selected checkpoint state hash"
    )
    if type(state_dict) is not dict or not state_dict:
        raise PORDV5ValidatorAmendmentError(
            "checkpoint state must be one exact non-empty dictionary"
        )
    expected = model.state_dict()
    expected_names = tuple(expected.keys())
    actual_names = tuple(state_dict.keys())
    if (
        len(actual_names) != len(expected_names)
        or actual_names != expected_names
        or any(type(name) is not str or not name for name in actual_names)
    ):
        raise PORDV5ValidatorAmendmentError(
            "checkpoint tensor name/order/count differs from the backbone"
        )
    try:
        actual_hash = _training.state_dict_sha256(state_dict)
    except Exception as exc:
        raise PORDV5ValidatorAmendmentError(
            "checkpoint state hash input is malformed"
        ) from exc
    if actual_hash != expected_hash:
        raise PORDV5ValidatorAmendmentError("checkpoint state hash changed")

    converted: dict[str, Any] = {}
    for name, expected_tensor in expected.items():
        value = state_dict[name]
        if type(value) is not np.ndarray:
            raise PORDV5ValidatorAmendmentError(
                f"checkpoint tensor {name!r} is not an exact numpy.ndarray"
            )
        try:
            expected_array = expected_tensor.detach().cpu().numpy()
        except (TypeError, RuntimeError) as exc:
            raise PORDV5ValidatorAmendmentError(
                f"backbone tensor {name!r} has an unsupported dtype/layout"
            ) from exc
        if tuple(value.shape) != tuple(expected_array.shape):
            raise PORDV5ValidatorAmendmentError(
                f"checkpoint tensor {name!r} shape changed"
            )
        if value.dtype != expected_array.dtype:
            raise PORDV5ValidatorAmendmentError(
                f"checkpoint tensor {name!r} dtype changed"
            )
        if not value.flags.c_contiguous:
            raise PORDV5ValidatorAmendmentError(
                f"checkpoint tensor {name!r} is not C-contiguous"
            )
        if value.dtype.hasobject or not np.issubdtype(value.dtype, np.number):
            raise PORDV5ValidatorAmendmentError(
                f"checkpoint tensor {name!r} is not numeric"
            )
        if not np.isfinite(value).all():
            raise PORDV5ValidatorAmendmentError(
                f"checkpoint tensor {name!r} is non-finite"
            )
        tensor = torch.from_numpy(np.array(value, copy=True, order="C"))
        if (
            tuple(tensor.shape) != tuple(expected_tensor.shape)
            or tensor.dtype != expected_tensor.dtype
            or tensor.device.type != "cpu"
            or tensor.layout != torch.strided
            or not bool(torch.isfinite(tensor).all().item())
        ):
            raise PORDV5ValidatorAmendmentError(
                f"checkpoint tensor {name!r} materialization changed semantics"
            )
        converted[name] = tensor
    if _training.state_dict_sha256(converted) != expected_hash:
        raise PORDV5ValidatorAmendmentError(
            "tensor materialization changed checkpoint bytes"
        )
    return converted


def _new_audit() -> dict[str, Any]:
    return {
        "checkpoint_replay_count": 0,
        "prediction_view_array_count": 0,
        "checkpoint_tensor_array_count": 0,
        "backbone_checkpoint_counts": {
            "masked_mlp_64x64_relu": 0,
            "chain_endpoint_only": 0,
        },
        "forbidden_call_counts": {
            **{name: 0 for name in _FORBIDDEN_EXPERIMENT_GLOBALS},
            "torch.optim.AdamW": 0,
        },
    }


def _validate_replay_audit(
    audit: Any, *, expected_checkpoint_replays: int
) -> dict[str, Any]:
    expected = _exact_int(
        expected_checkpoint_replays,
        name="expected checkpoint replay count",
        minimum=1,
    )
    if type(audit) is not dict or set(audit) != {
        "checkpoint_replay_count",
        "prediction_view_array_count",
        "checkpoint_tensor_array_count",
        "backbone_checkpoint_counts",
        "forbidden_call_counts",
    }:
        raise PORDV5ValidatorAmendmentError("validator replay audit schema changed")
    replay_count = _exact_int(
        audit["checkpoint_replay_count"], name="checkpoint replay count"
    )
    view_count = _exact_int(
        audit["prediction_view_array_count"], name="prediction view array count"
    )
    tensor_count = _exact_int(
        audit["checkpoint_tensor_array_count"], name="checkpoint tensor array count"
    )
    if replay_count != expected or view_count != expected * 5 or tensor_count <= 0:
        raise PORDV5ValidatorAmendmentError(
            "completed validator did not replay every checkpoint and five views"
        )
    backbones = audit["backbone_checkpoint_counts"]
    if type(backbones) is not dict or set(backbones) != set(
        PORD_V5_EXPECTED_BACKBONE_CHECKPOINTS
    ):
        raise PORDV5ValidatorAmendmentError("backbone replay audit schema changed")
    for value in backbones.values():
        _exact_int(value, name="backbone checkpoint count")
    if sum(backbones.values()) != expected:
        raise PORDV5ValidatorAmendmentError("backbone replay counts changed")
    if expected == PORD_V5_EXPECTED_CHECKPOINT_REPLAYS and backbones != (
        PORD_V5_EXPECTED_BACKBONE_CHECKPOINTS
    ):
        raise PORDV5ValidatorAmendmentError(
            "production MLP/chain checkpoint allocation changed"
        )
    forbidden = audit["forbidden_call_counts"]
    expected_forbidden = {
        *set(_FORBIDDEN_EXPERIMENT_GLOBALS),
        "torch.optim.AdamW",
    }
    if type(forbidden) is not dict or set(forbidden) != expected_forbidden:
        raise PORDV5ValidatorAmendmentError("forbidden call audit schema changed")
    if any(type(value) is not int or value != 0 for value in forbidden.values()):
        raise PORDV5ValidatorAmendmentError(
            "training/evaluation/optimizer/challenge code was invoked"
        )
    return deepcopy(audit)


@contextmanager
def _scoped_completed_validation_patch(audit: dict[str, Any]) -> Iterator[None]:
    """Patch one imported predictor and forbid every mutating execution path."""

    import torch

    if not _PATCH_LOCK.acquire(blocking=False):
        raise PORDV5ValidatorAmendmentError(
            "validator amendment cannot run concurrently in one process"
        )
    originals: dict[str, Any] = {}
    replacement_by_name: dict[str, Any] = {}
    original_adamw = torch.optim.AdamW
    original_predictor = _FROZEN_PREDICTOR
    adamw_patched = False
    try:
        if _experiment.predict_pord_checkpoint_outer is not original_predictor:
            raise PORDV5ValidatorAmendmentError(
                "original predictor identity changed before amendment"
            )

        def adapted_predictor(**kwargs: Any) -> Mapping[str, Any]:
            training_result = kwargs.get("training_result")
            if type(training_result) is not dict:
                raise PORDV5ValidatorAmendmentError(
                    "production training result must be an exact dictionary"
                )
            backbone_id = training_result.get("backbone_id")
            if backbone_id not in PORD_V5_EXPECTED_BACKBONE_CHECKPOINTS:
                raise PORDV5ValidatorAmendmentError(
                    "checkpoint backbone lies outside the frozen v5 grid"
                )
            model = _training.build_pord_backbone(torch, backbone_id=backbone_id)
            converted = strict_pord_v5_ndarray_state_dict_to_tensors(
                model=model,
                state_dict=training_result.get("state_dict"),
                expected_state_sha256=training_result.get("selected_state_sha256"),
            )
            adapted = dict(training_result)
            adapted["state_dict"] = converted
            call = dict(kwargs)
            call["training_result"] = adapted
            output = original_predictor(**call)
            if type(output) is not dict:
                raise PORDV5ValidatorAmendmentError(
                    "checkpoint predictor output schema changed"
                )
            predictions = output.get("predictions_raw_mm")
            if type(predictions) is not np.ndarray or predictions.ndim != 3:
                raise PORDV5ValidatorAmendmentError(
                    "checkpoint predictor did not return exact view arrays"
                )
            if predictions.shape[0] != 5 or predictions.shape[2] != 3:
                raise PORDV5ValidatorAmendmentError(
                    "checkpoint predictor five-view shape changed"
                )
            if not np.isfinite(predictions).all():
                raise PORDV5ValidatorAmendmentError(
                    "checkpoint predictor output is non-finite"
                )
            audit["checkpoint_replay_count"] += 1
            audit["prediction_view_array_count"] += 5
            audit["checkpoint_tensor_array_count"] += len(converted)
            audit["backbone_checkpoint_counts"][backbone_id] += 1
            return output

        _experiment.predict_pord_checkpoint_outer = adapted_predictor
        replacement_by_name["predict_pord_checkpoint_outer"] = adapted_predictor

        for name in _FORBIDDEN_EXPERIMENT_GLOBALS:
            original = getattr(_experiment, name)
            originals[name] = original

            def forbidden(*args: Any, _name: str = name, **kwargs: Any) -> Any:
                del args, kwargs
                audit["forbidden_call_counts"][_name] += 1
                raise PORDV5ValidatorAmendmentError(
                    f"validator amendment forbids {_name}"
                )

            setattr(_experiment, name, forbidden)
            replacement_by_name[name] = forbidden

        def forbidden_adamw(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            audit["forbidden_call_counts"]["torch.optim.AdamW"] += 1
            raise PORDV5ValidatorAmendmentError(
                "validator amendment forbids optimizer construction"
            )

        torch.optim.AdamW = forbidden_adamw
        adamw_patched = True
        yield
    finally:
        tampered = False
        try:
            if "predict_pord_checkpoint_outer" in replacement_by_name:
                tampered = (
                    _experiment.predict_pord_checkpoint_outer
                    is not replacement_by_name["predict_pord_checkpoint_outer"]
                )
                _experiment.predict_pord_checkpoint_outer = original_predictor
            for name, original in originals.items():
                if getattr(_experiment, name) is not replacement_by_name[name]:
                    tampered = True
                setattr(_experiment, name, original)
            if adamw_patched:
                if torch.optim.AdamW is not forbidden_adamw:
                    tampered = True
                torch.optim.AdamW = original_adamw
        finally:
            _PATCH_LOCK.release()
        if tampered:
            raise PORDV5ValidatorAmendmentError(
                "validator globals changed during the scoped amendment"
            )


def _exact_run_tree_manifest(run_dir: str | Path) -> list[dict[str, Any]]:
    run = Path(run_dir).resolve()
    if not run.is_dir() or run.is_symlink():
        raise PORDV5ValidatorAmendmentError("v5 run must be one real directory")
    rows: list[dict[str, Any]] = []
    for path in sorted(run.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise PORDV5ValidatorAmendmentError("v5 run tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PORDV5ValidatorAmendmentError(
                "v5 run tree contains a non-regular member"
            )
        relative = path.relative_to(run).as_posix()
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ino != after.st_ino
        ):
            raise PORDV5ValidatorAmendmentError(
                f"v5 run member changed while hashing: {relative}"
            )
        rows.append(
            {"path": relative, "bytes": int(after.st_size), "sha256": digest}
        )
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise PORDV5ValidatorAmendmentError("v5 run tree is empty or duplicated")
    return rows


def _validate_frozen_training_authority(
    run: Path, *, workspace_root: Path
) -> dict[str, Any]:
    source_path = run / "provenance" / "source_snapshot.json"
    source = read_json_object(source_path)
    if (
        set(source) != {"files", "fingerprint"}
        or type(source["files"]) is not list
        or len(source["files"]) != PORD_V5_FROZEN_SOURCE_FILE_COUNT
        or source.get("fingerprint")
        != PORD_V5_FROZEN_TRAINING_SOURCE_FINGERPRINT
        or stable_hash(source["files"]) != source["fingerprint"]
    ):
        raise PORDV5ValidatorAmendmentError(
            "frozen v5 training source authority changed"
        )
    current = _experiment.pord_v5_source_payload(workspace_root)
    if current != source:
        raise PORDV5ValidatorAmendmentError(
            "original 48-file source closure no longer matches prepare"
        )
    by_path = {
        row.get("path"): row
        for row in source["files"]
        if type(row) is dict and type(row.get("path")) is str
    }
    expected_hashes = {
        _FROZEN_EXPERIMENT_RELATIVE: PORD_V5_FROZEN_EXPERIMENT_SHA256,
        _FROZEN_TRAINING_RELATIVE: PORD_V5_FROZEN_TRAINING_SHA256,
    }
    for relative, digest in expected_hashes.items():
        row = by_path.get(relative)
        path = workspace_root / relative
        if (
            type(row) is not dict
            or row.get("sha256") != digest
            or sha256_file(path) != digest
        ):
            raise PORDV5ValidatorAmendmentError(
                f"frozen original module hash changed: {relative}"
            )
    return {
        "training_source_fingerprint": source["fingerprint"],
        "training_source_file_count": len(source["files"]),
        "source_snapshot_sha256": sha256_file(source_path),
        "frozen_experiment_sha256": PORD_V5_FROZEN_EXPERIMENT_SHA256,
        "frozen_training_sha256": PORD_V5_FROZEN_TRAINING_SHA256,
    }


def _challenge_must_be_sealed(run: Path) -> None:
    challenge = run / "challenge"
    if (challenge / "label_access_1.started.json").exists() or (
        challenge / "label_access_1.completed.json"
    ).exists():
        raise PORDV5ValidatorAmendmentError(
            "validator amendment refuses any challenge label-access marker"
        )


def _receipt_core(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "receipt_sha256"}


def validate_completed_pord_v5_with_amendment(
    run_dir: str | Path,
    *,
    workspace_root: str | Path | None = None,
    verify_live_environment: bool = True,
    _private_test_only: bool = False,
    _private_expected_checkpoint_replays: int | None = None,
) -> dict[str, Any]:
    """Purely validate a completed v5 run under the separate amendment.

    This function writes nothing.  The original validator is always invoked
    with ``verify_current_source=True``; no false-valued bypass exists.
    """

    run = Path(run_dir).resolve()
    root = _workspace_root() if workspace_root is None else Path(workspace_root).resolve()
    _challenge_must_be_sealed(run)
    if not (run / "development_manifest.json").is_file():
        raise PORDV5ValidatorAmendmentError(
            "amendment requires a completed development manifest"
        )
    frozen = _validate_frozen_training_authority(run, workspace_root=root)
    amendment_source = pord_v5_validator_amendment_source_payload(root)
    before = _exact_run_tree_manifest(run)
    audit = _new_audit()
    with _scoped_completed_validation_patch(audit):
        validation = _experiment.validate_pord_v5_development_run(
            run,
            verify_current_source=True,
            allow_completed_challenge=False,
            allow_synthetic_test_only=_private_test_only,
            verify_live_environment=verify_live_environment,
        )
    if _experiment.predict_pord_checkpoint_outer is not _FROZEN_PREDICTOR:
        raise PORDV5ValidatorAmendmentError(
            "original predictor was not restored after validation"
        )
    _challenge_must_be_sealed(run)
    after = _exact_run_tree_manifest(run)
    if before != after:
        raise PORDV5ValidatorAmendmentError(
            "v5 tree path/bytes/SHA changed during pure validation"
        )
    expected_replays = (
        PORD_V5_EXPECTED_CHECKPOINT_REPLAYS
        if _private_expected_checkpoint_replays is None
        else _private_expected_checkpoint_replays
    )
    if not _private_test_only and expected_replays != PORD_V5_EXPECTED_CHECKPOINT_REPLAYS:
        raise PORDV5ValidatorAmendmentError(
            "production checkpoint replay count cannot be overridden"
        )
    replay = _validate_replay_audit(
        audit, expected_checkpoint_replays=expected_replays
    )
    if (
        type(validation) is not dict
        or validation.get("status") != "ok"
        or validation.get("challenge_labels_sealed") is not True
        or validation.get("challenge_label_access_passes") != 0
        or (
            not _private_test_only
            and validation.get("controlled_fit_count")
            != PORD_V5_EXPECTED_CHECKPOINT_REPLAYS
        )
    ):
        raise PORDV5ValidatorAmendmentError(
            "original completed-run validator result changed"
        )
    tree_bytes = sum(_exact_int(row["bytes"], name="v5 tree bytes") for row in before)
    core: dict[str, Any] = {
        "contract": PORD_V5_VALIDATOR_AMENDMENT_CONTRACT,
        "status": "validated_completed_v5_with_scoped_ndarray_tensor_adapter",
        "authority_scope": "post_run_completed_development_validation_only",
        "run_origin": str(run),
        "frozen_training_authority": frozen,
        "amendment_source": amendment_source,
        "training_source_and_amendment_fingerprints_are_separate": True,
        "amendment_does_not_replace_or_relabel_training_source": True,
        "original_validator_verify_current_source_true": True,
        "verify_current_source_false_call_count": 0,
        "scoped_patch_target": (
            "imu_vive_learning.gpu_outage_pord_v5_experiment."
            "predict_pord_checkpoint_outer"
        ),
        "scoped_patch_restored": True,
        "v5_tree_file_count": len(before),
        "v5_tree_total_bytes": tree_bytes,
        "v5_tree_path_bytes_sha256_manifest_sha256": stable_hash(before),
        "v5_tree_before_after_exact": True,
        "development_manifest_sha256": sha256_file(
            run / "development_manifest.json"
        ),
        "checkpoint_replay_audit": replay,
        "prediction_view_bitwise_comparison_authority": {
            "original_experiment_sha256": PORD_V5_FROZEN_EXPERIMENT_SHA256,
            "comparison": "numpy.array_equal_persisted_vs_replayed",
            "checkpoint_count": replay["checkpoint_replay_count"],
            "view_array_count": replay["prediction_view_array_count"],
        },
        "training_optimizer_outer_evaluator_challenge_call_counts_all_zero": True,
        "challenge_labels_sealed": True,
        "challenge_label_access_passes": 0,
        "validation_result": deepcopy(validation),
        "validation_result_sha256": stable_hash(validation),
    }
    receipt = {**core, "receipt_sha256": stable_hash(core)}
    return receipt


def default_pord_v5_validator_amendment_evidence_dir(
    run_dir: str | Path,
) -> Path:
    run = Path(run_dir).resolve()
    return run.with_name(f"{run.name}.validator_amendment_v1")


def _validated_evidence_dir(run: Path, evidence_dir: str | Path) -> Path:
    evidence = Path(evidence_dir).resolve()
    expected = default_pord_v5_validator_amendment_evidence_dir(run)
    if evidence != expected or evidence.parent != run.parent or evidence == run:
        raise PORDV5ValidatorAmendmentError(
            "amendment evidence must use the exact sibling directory"
        )
    return evidence


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical_json_bytes(payload)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PORDV5ValidatorAmendmentError(
            "amendment receipt already exists; overwrite is forbidden"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize_pord_v5_validator_amendment_receipt(
    run_dir: str | Path,
    evidence_dir: str | Path,
    *,
    workspace_root: str | Path | None = None,
    verify_live_environment: bool = True,
    _private_test_only: bool = False,
    _private_expected_checkpoint_replays: int | None = None,
) -> dict[str, Any]:
    """Validate first, then create one immutable receipt outside the v5 tree."""

    run = Path(run_dir).resolve()
    evidence = _validated_evidence_dir(run, evidence_dir)
    if evidence.exists():
        raise PORDV5ValidatorAmendmentError(
            "amendment evidence directory already exists"
        )
    receipt = validate_completed_pord_v5_with_amendment(
        run,
        workspace_root=workspace_root,
        verify_live_environment=verify_live_environment,
        _private_test_only=_private_test_only,
        _private_expected_checkpoint_replays=_private_expected_checkpoint_replays,
    )
    evidence.mkdir(mode=0o700)
    try:
        _write_exclusive_json(evidence / PORD_V5_VALIDATOR_AMENDMENT_RECEIPT, receipt)
    except BaseException:
        # An empty sibling directory is an unambiguous failed evidence attempt.
        # It is deliberately not deleted automatically.
        raise
    return receipt


def validate_pord_v5_validator_amendment_receipt(
    run_dir: str | Path,
    evidence_dir: str | Path,
    *,
    workspace_root: str | Path | None = None,
    verify_live_environment: bool = True,
    _private_test_only: bool = False,
    _private_expected_checkpoint_replays: int | None = None,
) -> dict[str, Any]:
    """Purely revalidate the run and exact sibling receipt."""

    run = Path(run_dir).resolve()
    evidence = _validated_evidence_dir(run, evidence_dir)
    receipt_path = evidence / PORD_V5_VALIDATOR_AMENDMENT_RECEIPT
    if not evidence.is_dir() or {path.name for path in evidence.iterdir()} != {
        PORD_V5_VALIDATOR_AMENDMENT_RECEIPT
    }:
        raise PORDV5ValidatorAmendmentError(
            "amendment evidence directory is missing or not closed-world"
        )
    persisted = read_json_object(receipt_path)
    if (
        persisted.get("receipt_sha256")
        != stable_hash(_receipt_core(persisted))
        or persisted.get("contract") != PORD_V5_VALIDATOR_AMENDMENT_CONTRACT
    ):
        raise PORDV5ValidatorAmendmentError("amendment receipt hash/schema changed")
    replayed = validate_completed_pord_v5_with_amendment(
        run,
        workspace_root=workspace_root,
        verify_live_environment=verify_live_environment,
        _private_test_only=_private_test_only,
        _private_expected_checkpoint_replays=_private_expected_checkpoint_replays,
    )
    if persisted != replayed:
        raise PORDV5ValidatorAmendmentError(
            "persisted amendment receipt differs from pure semantic replay"
        )
    return persisted


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate completed PORD-v5 evidence with amendment v1"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--evidence-dir", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--create-receipt", action="store_true")
    action.add_argument("--validate-receipt", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.create_receipt:
        result = materialize_pord_v5_validator_amendment_receipt(
            args.run_dir, args.evidence_dir
        )
    else:
        result = validate_pord_v5_validator_amendment_receipt(
            args.run_dir, args.evidence_dir
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = [
    "PORDV5ValidatorAmendmentError",
    "PORD_V5_FROZEN_EXPERIMENT_SHA256",
    "PORD_V5_FROZEN_TRAINING_SHA256",
    "PORD_V5_FROZEN_TRAINING_SOURCE_FINGERPRINT",
    "PORD_V5_VALIDATOR_AMENDMENT_CONTRACT",
    "PORD_V5_VALIDATOR_AMENDMENT_SOURCE_FILES",
    "default_pord_v5_validator_amendment_evidence_dir",
    "main",
    "materialize_pord_v5_validator_amendment_receipt",
    "pord_v5_validator_amendment_source_payload",
    "strict_pord_v5_ndarray_state_dict_to_tensors",
    "validate_completed_pord_v5_with_amendment",
    "validate_pord_v5_validator_amendment_receipt",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

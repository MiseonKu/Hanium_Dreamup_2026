"""Execution controller for the preregistered OpenAXES native-V3 grid.

The controller deliberately keeps orchestration separate from model code.  It
materializes the complete 252-task roster before the first attempt, assigns a
fixed FIFO queue to each of twelve logical slots, and starts one fresh Python
process for every fit.  Three slots map to each of four physical GPUs.  A
restart may skip a task only after its immutable result validates; an attempt
without a valid result is never silently repeated.

The fit child calls ``openaxes_external_native_v3_training.run_experiment_fit``
exactly once.  That narrow adapter keeps the training implementation free to
own tensors, checkpoints, and data loading while this module owns process,
GPU, roster, and durable-state semantics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import uuid


RUN_ID = "openaxes-native-v3-allmask-dmome-001"
DEFAULT_RUN_DIR = "artifacts/openaxes_external_native_v3/run_openaxes-native-v3-001"
DEFAULT_PREPARED_ROOT = "artifacts/openaxes_external_v1/prepared_openaxes-v1-sealed-003"
EXPERIMENT_CONTRACT = "openaxes.external.native_sensor_experts.v3.execution.v1"
TASK_CONTRACT = "openaxes.external.native_sensor_experts.v3.task_spec.v1"
RESULT_CONTRACT = "openaxes.external.native_sensor_experts.v3.fit_result.v1"
EVALUATION_CONTRACT = "openaxes.external.native_sensor_experts.v3.evaluation.v1"

CANDIDATE_IDS = (
    "dense_6979__allmask_cardinality_balanced_erm",
    "static_sensor_experts_6979__allmask_cardinality_balanced_erm",
    "dmome_6979__one_stage_allmask_cardinality_balanced_erm",
    "dmome_6979__two_stage_allmask_cardinality_balanced_erm",
    "dmome_6979__two_stage_allmask_cardinality_balanced_mofe",
    "dmome_6979__two_stage_allmask_cardinality_balanced_mofe_cagrad",
)
SEEDS = (1701, 2603, 3907)
VIEW_IDS = (
    "full",
    "drop_humerus",
    "drop_radius",
    "drop_carpus",
    "drop_digitus",
    "only_humerus_radius",
    "only_humerus_carpus",
    "only_humerus_digitus",
    "only_radius_carpus",
    "only_radius_digitus",
    "only_carpus_digitus",
    "only_humerus",
    "only_radius",
    "only_carpus",
    "only_digitus",
)
FOLD_COUNT = 14
CONTEXT_COUNT = 42
TASK_COUNT = 252
LOGICAL_SLOT_COUNT = 12
GPU_COUNT = 4
FITS_PER_SLOT = (24, 24, 24, 24, 24, 24, 18, 18, 18, 18, 18, 18)
FITS_PER_GPU = (66, 66, 60, 60)
CONTEXTS_PER_SLOT = (4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3)
CONTEXTS_PER_GPU = (11, 11, 10, 10)
SENSOR_GROUPS = ("humerus", "radius", "carpus", "digitus")
CHANNELS = (
    "accel_adxl355_x_mps2",
    "accel_adxl355_y_mps2",
    "accel_adxl355_z_mps2",
    "accel_bmi160_x_mps2",
    "accel_bmi160_y_mps2",
    "accel_bmi160_z_mps2",
    "gyro_x_rad_s",
    "gyro_y_rad_s",
    "gyro_z_rad_s",
)
FEATURE_COLUMNS = tuple(
    f"{sensor}_{channel}" for sensor in SENSOR_GROUPS for channel in CHANNELS
)
TARGET_COLUMNS = ("tcp_x_m", "tcp_y_m", "tcp_z_m")
METADATA_COLUMNS = ("row_uid", "scenario_id", "sample_index", "aligned_time_us")
TASK_ID_PATTERN = re.compile(r"task_(?:00[0-9]|0[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[01])\Z")
HEX64_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class OpenAxesExternalNativeV3ExperimentError(RuntimeError):
    """Raised when execution state is incomplete, ambiguous, or off-grid."""


def _fail(message: str) -> None:
    raise OpenAxesExternalNativeV3ExperimentError(message)


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OpenAxesExternalNativeV3ExperimentError(
            "value is not finite canonical JSON"
        ) from exc


def _canonical_file(value: Any) -> bytes:
    return _canonical(value) + b"\n"


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["payload_sha256"] = _canonical_hash(result)
    return result


def _validate_payload(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        _fail(f"{name} is not a string-keyed mapping")
    result = dict(value)
    digest = result.pop("payload_sha256", None)
    if type(digest) is not str or HEX64_PATTERN.fullmatch(digest) is None:
        _fail(f"{name} payload hash is malformed")
    if _canonical_hash(result) != digest:
        _fail(f"{name} payload hash changed")
    result["payload_sha256"] = digest
    return result


def _read_json(
    path: Path, *, name: str, require_canonical: bool = True
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpenAxesExternalNativeV3ExperimentError(f"cannot read {name}") from exc
    if require_canonical and raw != _canonical_file(value):
        _fail(f"{name} is not canonical JSON with one final newline")
    # Inherited preparation manifests predate the V3 canonical-file format.
    # Parse and finite-check those bytes, while keeping every V3-owned receipt
    # and result on the strict canonical path above.
    if not require_canonical:
        _canonical(value)
    if not isinstance(value, Mapping):
        _fail(f"{name} is not a mapping")
    return dict(value)


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail("exclusive write made no progress")
            offset += written
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        if info.st_size != len(raw) or not os.path.isfile(path):
            _fail("exclusive output identity changed")
    except FileExistsError as exc:
        raise OpenAxesExternalNativeV3ExperimentError(
            f"immutable output already exists: {path}"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_json_exclusive(path: Path, value: Any) -> None:
    _write_exclusive(path, _canonical_file(value))


def _write_status(path: Path, value: Any) -> None:
    """Atomically replace the monitoring projection (never scientific data)."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
    )
    raw = _canonical_file(value)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                _fail("status write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _reference(path: Path, *, relative_to: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _process_start_ticks(pid: int) -> int:
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except OSError as exc:
        raise OpenAxesExternalNativeV3ExperimentError(
            "child exited before process identity was captured"
        ) from exc
    closing = raw.rfind(b")")
    fields = raw[closing + 2 :].split() if closing >= 0 else []
    if len(fields) < 20 or not fields[19].isdigit():
        _fail("child /proc start ticks are malformed")
    return int(fields[19])


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_ordinal: int
    context_ordinal: int
    fold_id: str
    fold_ordinal: int
    fold_family: str
    train_scenarios: tuple[str, ...]
    test_scenarios: tuple[str, ...]
    seed: int
    seed_ordinal: int
    candidate_id: str
    candidate_ordinal: int
    logical_slot: int
    physical_gpu: int
    local_slot: int
    task_spec_sha256: str

    def payload(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "contract": TASK_CONTRACT,
            "schema_version": 1,
            "run_id": RUN_ID,
            "task_id": self.task_id,
            "task_ordinal": self.task_ordinal,
            "context_ordinal": self.context_ordinal,
            "fold_id": self.fold_id,
            "fold_ordinal": self.fold_ordinal,
            "fold_family": self.fold_family,
            "train_scenarios": list(self.train_scenarios),
            "test_scenarios": list(self.test_scenarios),
            "seed": self.seed,
            "seed_ordinal": self.seed_ordinal,
            "candidate_id": self.candidate_id,
            "candidate_ordinal": self.candidate_ordinal,
            "logical_slot": self.logical_slot,
            "physical_gpu": self.physical_gpu,
            "local_slot": self.local_slot,
        }
        if include_hash:
            value["task_spec_sha256"] = self.task_spec_sha256
        return value


def _fold_items(preparation_manifest: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    folds = preparation_manifest.get("folds")
    if not isinstance(folds, Mapping) or type(folds.get("items")) is not list:
        _fail("preparation manifest has no fold item roster")
    items = folds["items"]
    if len(items) != FOLD_COUNT:
        _fail("preparation manifest must contain exactly fourteen folds")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    allowed_families = {"scenario_loso", "speed_pair_out", "shape_out"}
    for ordinal, raw in enumerate(items):
        if not isinstance(raw, Mapping):
            _fail("fold item is not a mapping")
        fold_id = raw.get("fold_id")
        family = raw.get("family")
        train = raw.get("train_scenarios")
        test = raw.get("test_scenarios")
        if (
            type(fold_id) is not str
            or not fold_id
            or fold_id in seen
            or family not in allowed_families
            or type(train) is not list
            or type(test) is not list
            or not train
            or not test
            or any(type(item) is not str for item in [*train, *test])
            or set(train) & set(test)
        ):
            _fail(f"fold {ordinal} is malformed")
        seen.add(fold_id)
        result.append(
            {
                "fold_id": fold_id,
                "family": family,
                "train_scenarios": tuple(train),
                "test_scenarios": tuple(test),
            }
        )
    families = [row["family"] for row in result]
    if families.count("scenario_loso") != 8 or families.count("speed_pair_out") != 4 or families.count("shape_out") != 2:
        _fail("fold family counts changed")
    return tuple(result)


def build_native_v3_task_grid(
    preparation_manifest: Mapping[str, Any],
) -> tuple[TaskSpec, ...]:
    """Return the frozen fold-major, seed-major, candidate-major task grid."""

    tasks: list[TaskSpec] = []
    for fold_ordinal, fold in enumerate(_fold_items(preparation_manifest)):
        for seed_ordinal, seed in enumerate(SEEDS):
            context_ordinal = fold_ordinal * len(SEEDS) + seed_ordinal
            logical_slot = context_ordinal % LOGICAL_SLOT_COUNT
            for candidate_ordinal, candidate_id in enumerate(CANDIDATE_IDS):
                task_ordinal = context_ordinal * len(CANDIDATE_IDS) + candidate_ordinal
                base = {
                    "contract": TASK_CONTRACT,
                    "schema_version": 1,
                    "run_id": RUN_ID,
                    "task_id": f"task_{task_ordinal:03d}",
                    "task_ordinal": task_ordinal,
                    "context_ordinal": context_ordinal,
                    "fold_id": fold["fold_id"],
                    "fold_ordinal": fold_ordinal,
                    "fold_family": fold["family"],
                    "train_scenarios": list(fold["train_scenarios"]),
                    "test_scenarios": list(fold["test_scenarios"]),
                    "seed": seed,
                    "seed_ordinal": seed_ordinal,
                    "candidate_id": candidate_id,
                    "candidate_ordinal": candidate_ordinal,
                    "logical_slot": logical_slot,
                    "physical_gpu": logical_slot % GPU_COUNT,
                    "local_slot": logical_slot // GPU_COUNT,
                }
                tasks.append(
                    TaskSpec(
                        task_id=base["task_id"],
                        task_ordinal=task_ordinal,
                        context_ordinal=context_ordinal,
                        fold_id=fold["fold_id"],
                        fold_ordinal=fold_ordinal,
                        fold_family=fold["family"],
                        train_scenarios=fold["train_scenarios"],
                        test_scenarios=fold["test_scenarios"],
                        seed=seed,
                        seed_ordinal=seed_ordinal,
                        candidate_id=candidate_id,
                        candidate_ordinal=candidate_ordinal,
                        logical_slot=logical_slot,
                        physical_gpu=logical_slot % GPU_COUNT,
                        local_slot=logical_slot // GPU_COUNT,
                        task_spec_sha256=_canonical_hash(base),
                    )
                )
    if len(tasks) != TASK_COUNT or [task.task_ordinal for task in tasks] != list(range(TASK_COUNT)):
        _fail("task grid is not the exact 252-task sequence")
    return tuple(tasks)


def verify_task_spec(value: Any, expected: TaskSpec) -> dict[str, Any]:
    if not isinstance(value, Mapping) or dict(value) != expected.payload():
        _fail(f"task spec changed: {expected.task_id}")
    if _canonical_hash(expected.payload(include_hash=False)) != expected.task_spec_sha256:
        _fail(f"task spec self hash changed: {expected.task_id}")
    return dict(value)


def build_slot_assignments(tasks: Sequence[TaskSpec]) -> tuple[dict[str, Any], ...]:
    if len(tasks) != TASK_COUNT:
        _fail("slot assignment requires exactly 252 tasks")
    rows: list[dict[str, Any]] = []
    for slot in range(LOGICAL_SLOT_COUNT):
        selected = [task for task in tasks if task.logical_slot == slot]
        selected.sort(key=lambda item: (item.context_ordinal, item.candidate_ordinal))
        contexts = sorted({task.context_ordinal for task in selected})
        rows.append(
            {
                "logical_slot": slot,
                "physical_gpu": slot % GPU_COUNT,
                "local_slot": slot // GPU_COUNT,
                "ordered_context_ordinals": contexts,
                "ordered_task_ids": [task.task_id for task in selected],
            }
        )
    slot_counts = tuple(len(row["ordered_task_ids"]) for row in rows)
    gpu_counts = tuple(
        sum(len(rows[slot]["ordered_task_ids"]) for slot in range(gpu, LOGICAL_SLOT_COUNT, GPU_COUNT))
        for gpu in range(GPU_COUNT)
    )
    context_slot_counts = tuple(len(row["ordered_context_ordinals"]) for row in rows)
    context_gpu_counts = tuple(
        sum(len(rows[slot]["ordered_context_ordinals"]) for slot in range(gpu, LOGICAL_SLOT_COUNT, GPU_COUNT))
        for gpu in range(GPU_COUNT)
    )
    if (
        slot_counts != FITS_PER_SLOT
        or gpu_counts != FITS_PER_GPU
        or context_slot_counts != CONTEXTS_PER_SLOT
        or context_gpu_counts != CONTEXTS_PER_GPU
    ):
        _fail("slot/GPU accounting changed")
    return tuple(rows)


def build_launch_waves(tasks: Sequence[TaskSpec]) -> tuple[tuple[TaskSpec, ...], ...]:
    """Interleave slot FIFOs; a wave contains at most three fits per GPU."""

    by_id = {task.task_id: task for task in tasks}
    slots = build_slot_assignments(tasks)
    waves: list[tuple[TaskSpec, ...]] = []
    for offset in range(max(FITS_PER_SLOT)):
        wave = tuple(
            by_id[row["ordered_task_ids"][offset]]
            for row in slots
            if offset < len(row["ordered_task_ids"])
        )
        if wave:
            per_gpu = tuple(sum(task.physical_gpu == gpu for task in wave) for gpu in range(GPU_COUNT))
            if any(count > 3 for count in per_gpu):
                _fail("launch wave oversubscribes a preregistered GPU slot count")
            waves.append(wave)
    if sum(len(wave) for wave in waves) != TASK_COUNT:
        _fail("launch waves do not cover the exact task roster")
    return tuple(waves)


def _task_from_payload(value: Mapping[str, Any]) -> TaskSpec:
    try:
        task = TaskSpec(
            task_id=value["task_id"],
            task_ordinal=value["task_ordinal"],
            context_ordinal=value["context_ordinal"],
            fold_id=value["fold_id"],
            fold_ordinal=value["fold_ordinal"],
            fold_family=value["fold_family"],
            train_scenarios=tuple(value["train_scenarios"]),
            test_scenarios=tuple(value["test_scenarios"]),
            seed=value["seed"],
            seed_ordinal=value["seed_ordinal"],
            candidate_id=value["candidate_id"],
            candidate_ordinal=value["candidate_ordinal"],
            logical_slot=value["logical_slot"],
            physical_gpu=value["physical_gpu"],
            local_slot=value["local_slot"],
            task_spec_sha256=value["task_spec_sha256"],
        )
    except (KeyError, TypeError) as exc:
        raise OpenAxesExternalNativeV3ExperimentError("task payload is malformed") from exc
    verify_task_spec(value, task)
    if (
        TASK_ID_PATTERN.fullmatch(task.task_id) is None
        or task.task_ordinal != task.context_ordinal * 6 + task.candidate_ordinal
        or task.context_ordinal != task.fold_ordinal * 3 + task.seed_ordinal
        or task.logical_slot != task.context_ordinal % 12
        or task.physical_gpu != task.logical_slot % 4
        or task.local_slot != task.logical_slot // 4
        or task.candidate_id != CANDIDATE_IDS[task.candidate_ordinal]
        or task.seed != SEEDS[task.seed_ordinal]
    ):
        _fail("task ordinal or placement semantics changed")
    return task


def prepare_run(run_dir: Path | str, preparation_manifest: Mapping[str, Any]) -> tuple[TaskSpec, ...]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    index_path = root / "task_index.json"
    if index_path.exists():
        return load_task_grid(root)
    existing = {path.name for path in root.iterdir()}
    if existing - {"parents"}:
        _fail("unindexed run directory contains non-parent execution state")
    if "parents" in existing and not (root / "parents").is_dir():
        _fail("preissued parent authority is not a directory")
    for relative in ("tasks", "results", "attempts", "runtime", "logs", "reports"):
        (root / relative).mkdir(mode=0o700)
    tasks = build_native_v3_task_grid(preparation_manifest)
    references: list[dict[str, Any]] = []
    for task in tasks:
        path = root / "tasks" / f"{task.task_id}.json"
        _write_json_exclusive(path, task.payload())
        references.append(_reference(path, relative_to=root))
    slots = list(build_slot_assignments(tasks))
    assignments = _payload(
        {
            "contract": "openaxes.external.native_sensor_experts.v3.slot_assignments.v1",
            "schema_version": 1,
            "run_id": RUN_ID,
            "logical_slot_count": LOGICAL_SLOT_COUNT,
            "physical_gpu_count": GPU_COUNT,
            "fits_per_slot": list(FITS_PER_SLOT),
            "fits_per_gpu": list(FITS_PER_GPU),
            "slots": slots,
            "slot_assignment_sha256": _canonical_hash(slots),
        }
    )
    _write_json_exclusive(root / "runtime" / "slot_assignments.json", assignments)
    manifest_reference = {
        "bytes": len(_canonical_file(preparation_manifest)),
        "sha256": hashlib.sha256(_canonical_file(preparation_manifest)).hexdigest(),
    }
    index = _payload(
        {
            "contract": "openaxes.external.native_sensor_experts.v3.task_index.v1",
            "schema_version": 1,
            "run_id": RUN_ID,
            "task_count": TASK_COUNT,
            "context_count": CONTEXT_COUNT,
            "candidate_order": list(CANDIDATE_IDS),
            "seed_order": list(SEEDS),
            "preparation_manifest_reference": manifest_reference,
            "task_spec_references": references,
            "task_assignment_sha256": _canonical_hash(
                [
                    {
                        "task_id": task.task_id,
                        "logical_slot": task.logical_slot,
                        "physical_gpu": task.physical_gpu,
                    }
                    for task in tasks
                ]
            ),
            "slot_assignments_reference": _reference(
                root / "runtime" / "slot_assignments.json", relative_to=root
            ),
        }
    )
    _write_json_exclusive(index_path, index)
    _write_status(root / "runtime" / "status.json", status_projection(root, tasks))
    return tasks


def load_task_grid(run_dir: Path | str) -> tuple[TaskSpec, ...]:
    root = Path(run_dir)
    index = _validate_payload(_read_json(root / "task_index.json", name="task index"), name="task index")
    if (
        index.get("contract") != "openaxes.external.native_sensor_experts.v3.task_index.v1"
        or index.get("run_id") != RUN_ID
        or index.get("task_count") != TASK_COUNT
        or index.get("context_count") != CONTEXT_COUNT
        or index.get("candidate_order") != list(CANDIDATE_IDS)
        or index.get("seed_order") != list(SEEDS)
    ):
        _fail("task index identity changed")
    references = index.get("task_spec_references")
    if type(references) is not list or len(references) != TASK_COUNT:
        _fail("task index reference roster changed")
    tasks: list[TaskSpec] = []
    for ordinal, reference in enumerate(references):
        expected_path = f"tasks/task_{ordinal:03d}.json"
        if not isinstance(reference, Mapping) or reference.get("path") != expected_path:
            _fail("task reference order changed")
        path = root / expected_path
        if _reference(path, relative_to=root) != dict(reference):
            _fail("task reference bytes changed")
        tasks.append(_task_from_payload(_read_json(path, name=expected_path)))
    slots = _validate_payload(
        _read_json(root / "runtime" / "slot_assignments.json", name="slot assignments"),
        name="slot assignments",
    )
    expected_slots = list(build_slot_assignments(tasks))
    if (
        slots.get("slots") != expected_slots
        or slots.get("fits_per_slot") != list(FITS_PER_SLOT)
        or slots.get("fits_per_gpu") != list(FITS_PER_GPU)
        or slots.get("slot_assignment_sha256") != _canonical_hash(expected_slots)
        or index.get("slot_assignments_reference")
        != _reference(root / "runtime" / "slot_assignments.json", relative_to=root)
    ):
        _fail("slot assignment authority changed")
    return tuple(tasks)


def task_state(run_dir: Path | str, task_id: str) -> str:
    root = Path(run_dir)
    attempt = (root / "attempts" / f"{task_id}.json").exists()
    result = (root / "results" / task_id / "result.json").exists()
    error = (root / "results" / task_id / "error.json").exists()
    if not attempt and not result and not error:
        return "pending"
    if attempt and result and not error:
        return "completed"
    if attempt and error and not result:
        return "failed"
    if attempt and not result and not error:
        return "interrupted"
    return "ambiguous"


def status_projection(run_dir: Path | str, tasks: Sequence[TaskSpec] | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    roster = tuple(tasks) if tasks is not None else load_task_grid(root)
    states = {task.task_id: task_state(root, task.task_id) for task in roster}
    counts = {state: list(states.values()).count(state) for state in ("pending", "completed", "failed", "interrupted", "ambiguous")}
    completed_by_gpu = [
        sum(states[task.task_id] == "completed" and task.physical_gpu == gpu for task in roster)
        for gpu in range(GPU_COUNT)
    ]
    return {
        "contract": "openaxes.external.native_sensor_experts.v3.status.v1",
        "schema_version": 1,
        "run_id": RUN_ID,
        "task_count": len(roster),
        "state_counts": counts,
        "completed_by_gpu": completed_by_gpu,
        "expected_fits_by_gpu": list(FITS_PER_GPU),
        "complete": counts["completed"] == TASK_COUNT,
        "valid_terminal_candidate": counts["failed"] == counts["interrupted"] == counts["ambiguous"] == 0,
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _json_safe_training_result(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_payload") and callable(value.to_payload):
        value = value.to_payload()
    elif is_dataclass(value):
        value = asdict(value)
    if not isinstance(value, Mapping):
        _fail("training result must be a mapping, dataclass, or to_payload object")
    result = dict(value)
    _canonical(result)
    return result


def _default_fit_runner(
    task: TaskSpec,
    *,
    prepared_root: str,
    output_dir: str,
    device: str,
) -> Mapping[str, Any]:
    from . import openaxes_external_native_v3_training as native_training

    import numpy as np
    import pyarrow.parquet as pq

    runner = getattr(native_training, "run_experiment_fit", None)
    if runner is None or not callable(runner):
        _fail("native-V3 training module does not expose run_experiment_fit")
    table_path = Path(prepared_root) / "prepared.parquet"
    columns = [*METADATA_COLUMNS, *FEATURE_COLUMNS, *TARGET_COLUMNS]
    try:
        table = pq.read_table(table_path, columns=columns, use_threads=False)
        frame = table.to_pandas()
    except (OSError, ValueError) as exc:
        raise OpenAxesExternalNativeV3ExperimentError(
            "cannot load the frozen prepared table"
        ) from exc
    if (
        list(frame.columns) != columns
        or frame["row_uid"].duplicated().any()
        or set(frame["scenario_id"]) != set(task.train_scenarios) | set(task.test_scenarios)
    ):
        _fail("prepared table identity or scenario roster changed")
    training_frame = frame[frame["scenario_id"].isin(task.train_scenarios)].copy()
    outer_frame = frame[frame["scenario_id"].isin(task.test_scenarios)].copy()
    if (
        len(training_frame) < 1
        or len(outer_frame) < 1
        or set(training_frame["scenario_id"]) != set(task.train_scenarios)
        or set(outer_frame["scenario_id"]) != set(task.test_scenarios)
    ):
        _fail("prepared train/outer partition changed")
    training_task = {
        "X_train": training_frame[list(FEATURE_COLUMNS)].to_numpy(
            dtype="float64", copy=True
        ),
        "y_train_m": training_frame[list(TARGET_COLUMNS)].to_numpy(
            dtype="float64", copy=True
        ),
        "training_scenarios": training_frame["scenario_id"].tolist(),
        "train_uids": training_frame["row_uid"].tolist(),
        "declared_scenario_order": list(task.train_scenarios),
        "candidate_id": task.candidate_id,
        "seed": task.seed,
    }
    result = dict(runner(
        training_task,
        prepared_root=prepared_root,
        output_dir=output_dir,
        device=device,
    ))
    checkpoint = result.get("checkpoint_reference")
    transform_row = result.get("transform")
    if (
        not isinstance(checkpoint, Mapping)
        or type(checkpoint.get("path")) is not str
        or not isinstance(transform_row, Mapping)
    ):
        _fail("training result lacks checkpoint or transform authority")
    checkpoint_path = Path(output_dir) / checkpoint["path"]
    checkpoint_raw = checkpoint_path.read_bytes()
    if (
        len(checkpoint_raw) != checkpoint.get("bytes")
        or hashlib.sha256(checkpoint_raw).hexdigest() != checkpoint.get("sha256")
    ):
        _fail("checkpoint reference changed before outer prediction")
    state = native_training.load_checkpoint_state(checkpoint_raw)
    transform_payload = {
        key: value for key, value in transform_row.items() if key != "sha256"
    }
    if hashlib.sha256(_canonical(transform_payload)).hexdigest() != transform_row.get("sha256"):
        _fail("train-only transform receipt changed")
    transform = native_training.TrainOnlyTransform(
        feature_mean=np.asarray(transform_row["feature_mean"], dtype="float64"),
        feature_scale=np.asarray(transform_row["feature_scale"], dtype="float64"),
        target_mean_m=np.asarray(transform_row["target_mean_m"], dtype="float64"),
        target_scale_m=float(transform_row["target_scale_m"]),
        train_rows=int(transform_row["train_rows"]),
        ordered_row_uid_sha256=transform_row["ordered_row_uid_sha256"],
        ordered_scenario_id_sha256=transform_row["ordered_scenario_id_sha256"],
        ordered_uid_scenario_pair_sha256=transform_row[
            "ordered_uid_scenario_pair_sha256"
        ],
    )
    feature_batch = native_training.FeatureOnlyBatch(
        continuous_features=outer_frame[list(FEATURE_COLUMNS)].to_numpy(
            dtype="float64", copy=True
        ),
        row_uids=tuple(outer_frame["row_uid"].tolist()),
        scenario_ids=tuple(outer_frame["scenario_id"].tolist()),
        sample_indices=tuple(int(item) for item in outer_frame["sample_index"]),
        aligned_time_us=tuple(int(item) for item in outer_frame["aligned_time_us"]),
    )
    predictions = native_training.predict_feature_only_batch_once(
        candidate_id=task.candidate_id,
        seed=task.seed,
        state_dict=state,
        transform=transform,
        feature_batch=feature_batch,
        device=device,
    )
    truth = outer_frame[list(TARGET_COLUMNS)].to_numpy(dtype="float64", copy=True)
    metrics = [
        {
            "view_id": view_id,
            "rows": len(outer_frame),
            "rmse3d_m": native_training.fixed_rmse3d_m(
                predictions[index], truth, name=f"outer {view_id}"
            ),
        }
        for index, view_id in enumerate(VIEW_IDS)
    ]
    prediction_path = Path(output_dir) / "outer_predictions.npy"
    with open(prediction_path, "xb") as handle:
        np.save(handle, np.asarray(predictions, dtype="float64"), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    result["outer_view_metrics"] = metrics
    result["view_metrics"] = metrics
    result["outer_predictions_reference"] = _reference(
        prediction_path, relative_to=Path(output_dir)
    )
    result["outer_feature_loads"] = 1
    result["outer_label_reads"] = 1
    result["outer_evaluator_calls"] = 1
    result["outer_model_forward_calls"] = 1
    return result


def run_internal_fit(
    run_dir: Path | str,
    task_id: str,
    *,
    prepared_root: str,
    gpu_index: int,
    gpu_selector: str,
    attempt_id: str,
    fit_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    task = _task_from_payload(
        _read_json(root / "tasks" / f"{task_id}.json", name="fit task")
    )
    if task.physical_gpu != gpu_index or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_selector:
        _fail("fit child GPU binding changed")
    result_path = root / "results" / task_id / "result.json"
    error_path = root / "results" / task_id / "error.json"
    if result_path.exists() or error_path.exists():
        _fail("fit child result authority already exists")
    (root / "results" / task_id).mkdir(parents=True, exist_ok=True, mode=0o700)
    runner = fit_runner or _default_fit_runner
    try:
        scientific = _json_safe_training_result(
            runner(
                task,
                prepared_root=prepared_root,
                output_dir=str(root / "results" / task_id),
                device="cuda:0",
            )
        )
        result = _payload(
            {
                "contract": RESULT_CONTRACT,
                "schema_version": 1,
                "run_id": RUN_ID,
                "task_id": task.task_id,
                "task_ordinal": task.task_ordinal,
                "task_spec_sha256": task.task_spec_sha256,
                "candidate_id": task.candidate_id,
                "fold_id": task.fold_id,
                "seed": task.seed,
                "logical_slot": task.logical_slot,
                "physical_gpu": task.physical_gpu,
                "gpu_selector": gpu_selector,
                "attempt_uuid_v4": attempt_id,
                "process_identity": {
                    "pid": os.getpid(),
                    "proc_start_ticks": _process_start_ticks(os.getpid()),
                },
                "optimizer_updates": 400,
                "training_result": scientific,
                "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        )
        _write_json_exclusive(result_path, result)
        return result
    except BaseException as exc:
        if not error_path.exists() and not result_path.exists():
            error = _payload(
                {
                    "contract": "openaxes.external.native_sensor_experts.v3.fit_error.v1",
                    "schema_version": 1,
                    "run_id": RUN_ID,
                    "task_id": task.task_id,
                    "task_spec_sha256": task.task_spec_sha256,
                    "attempt_uuid_v4": attempt_id,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "failed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            _write_json_exclusive(error_path, error)
        raise


def validate_task_result(run_dir: Path | str, task: TaskSpec) -> dict[str, Any]:
    root = Path(run_dir)
    value = _validate_payload(
        _read_json(root / "results" / task.task_id / "result.json", name="fit result"),
        name="fit result",
    )
    required = {
        "contract", "schema_version", "run_id", "task_id", "task_ordinal",
        "task_spec_sha256", "candidate_id", "fold_id", "seed", "logical_slot",
        "physical_gpu", "gpu_selector", "attempt_uuid_v4", "process_identity",
        "optimizer_updates", "training_result", "completed_at_utc", "payload_sha256",
    }
    if set(value) != required:
        _fail(f"fit result key set changed: {task.task_id}")
    if (
        value["contract"] != RESULT_CONTRACT
        or value["run_id"] != RUN_ID
        or value["task_id"] != task.task_id
        or value["task_ordinal"] != task.task_ordinal
        or value["task_spec_sha256"] != task.task_spec_sha256
        or value["candidate_id"] != task.candidate_id
        or value["fold_id"] != task.fold_id
        or value["seed"] != task.seed
        or value["logical_slot"] != task.logical_slot
        or value["physical_gpu"] != task.physical_gpu
        or value["optimizer_updates"] != 400
        or not isinstance(value["training_result"], Mapping)
    ):
        _fail(f"fit result semantics changed: {task.task_id}")
    attempt = _validate_payload(
        _read_json(root / "attempts" / f"{task.task_id}.json", name="fit attempt"),
        name="fit attempt",
    )
    if value["attempt_uuid_v4"] != attempt.get("attempt_uuid_v4"):
        _fail("result no longer binds its sole attempt")
    return value


def _attempt_payload(task: TaskSpec, gpu_selector: str) -> dict[str, Any]:
    return _payload(
        {
            "contract": "openaxes.external.native_sensor_experts.v3.fit_attempt.v1",
            "schema_version": 1,
            "run_id": RUN_ID,
            "task_id": task.task_id,
            "task_spec_sha256": task.task_spec_sha256,
            "attempt_uuid_v4": str(uuid.uuid4()),
            "scientific_fit_attempt": 1,
            "logical_slot": task.logical_slot,
            "physical_gpu": task.physical_gpu,
            "local_slot": task.local_slot,
            "gpu_selector": gpu_selector,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )


def _child_command(
    python_executable: str,
    run_dir: Path,
    prepared_root: str,
    task: TaskSpec,
    attempt: Mapping[str, Any],
) -> list[str]:
    return [
        python_executable,
        "-P", "-s", "-B", "-m",
        "imu_vive_learning.openaxes_external_native_v3_experiment",
        "internal-fit",
        "--run-dir", str(run_dir),
        "--prepared-root", prepared_root,
        "--task-id", task.task_id,
        "--gpu-index", str(task.physical_gpu),
        "--gpu-selector", str(attempt["gpu_selector"]),
        "--attempt-id", str(attempt["attempt_uuid_v4"]),
    ]


def run_scheduled_fits(
    run_dir: Path | str,
    *,
    prepared_root: str,
    python_executable: str = "/proc/self/exe",
    gpu_selectors: Sequence[str] = ("0", "1", "2", "3"),
    candidate_subset: Sequence[str] | None = None,
    context_subset: Sequence[int] | None = None,
    popen_factory: Callable[..., Any] = subprocess.Popen,
) -> dict[str, Any]:
    """Run never-attempted tasks, preserving the fixed slot FIFO and GPU map."""

    root = Path(run_dir)
    tasks = load_task_grid(root)
    if len(gpu_selectors) != GPU_COUNT or len(set(gpu_selectors)) != GPU_COUNT:
        _fail("exactly four distinct GPU selectors are required")
    selected_candidates = tuple(candidate_subset or CANDIDATE_IDS)
    selected_contexts = tuple(context_subset or range(CONTEXT_COUNT))
    if (
        not selected_candidates
        or len(set(selected_candidates)) != len(selected_candidates)
        or any(candidate not in CANDIDATE_IDS for candidate in selected_candidates)
        or not selected_contexts
        or len(set(selected_contexts)) != len(selected_contexts)
        or any(type(context) is not int or not 0 <= context < CONTEXT_COUNT for context in selected_contexts)
    ):
        _fail("candidate/context execution subset is invalid")
    selected = tuple(
        task
        for task in tasks
        if task.candidate_id in selected_candidates
        and task.context_ordinal in selected_contexts
    )
    initial = {task.task_id: task_state(root, task.task_id) for task in selected}
    unsafe = {task_id: state for task_id, state in initial.items() if state in {"failed", "interrupted", "ambiguous"}}
    if unsafe:
        _fail("resume found an attempted task without one valid immutable result")
    base_environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    selected_ids = set(initial)
    for wave_ordinal, full_wave in enumerate(build_launch_waves(tasks)):
        wave = tuple(task for task in full_wave if task.task_id in selected_ids)
        if not wave:
            continue
        running: list[tuple[TaskSpec, Mapping[str, Any], Any, Any, Any]] = []
        for task in wave:
            if initial[task.task_id] == "completed":
                validate_task_result(root, task)
                continue
            selector = str(gpu_selectors[task.physical_gpu])
            attempt = _attempt_payload(task, selector)
            _write_json_exclusive(root / "attempts" / f"{task.task_id}.json", attempt)
            stdout_file = open(root / "logs" / f"{task.task_id}.stdout.log", "xb")
            stderr_file = open(root / "logs" / f"{task.task_id}.stderr.log", "xb")
            environment = dict(base_environment)
            environment["CUDA_VISIBLE_DEVICES"] = selector
            command = _child_command(python_executable, root, prepared_root, task, attempt)
            try:
                process = popen_factory(
                    command,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    close_fds=True,
                    start_new_session=True,
                )
                process_row = _payload(
                    {
                        "contract": "openaxes.external.native_sensor_experts.v3.fit_process.v1",
                        "schema_version": 1,
                        "run_id": RUN_ID,
                        "task_id": task.task_id,
                        "wave_ordinal": wave_ordinal,
                        "attempt_uuid_v4": attempt["attempt_uuid_v4"],
                        "pid": process.pid,
                        "proc_start_ticks": _process_start_ticks(process.pid),
                        "command": command,
                        "gpu_selector": selector,
                    }
                )
                _write_json_exclusive(root / "runtime" / f"{task.task_id}.process.json", process_row)
            except BaseException:
                stdout_file.close()
                stderr_file.close()
                raise
            running.append((task, attempt, process, stdout_file, stderr_file))
        failures: list[str] = []
        for task, attempt, process, stdout_file, stderr_file in running:
            returncode = process.wait()
            stdout_file.close()
            stderr_file.close()
            exit_row = _payload(
                {
                    "contract": "openaxes.external.native_sensor_experts.v3.fit_process_exit.v1",
                    "schema_version": 1,
                    "run_id": RUN_ID,
                    "task_id": task.task_id,
                    "attempt_uuid_v4": attempt["attempt_uuid_v4"],
                    "returncode": returncode,
                }
            )
            _write_json_exclusive(root / "runtime" / f"{task.task_id}.exit.json", exit_row)
            if returncode != 0:
                failures.append(task.task_id)
            else:
                validate_task_result(root, task)
        _write_status(root / "runtime" / "status.json", status_projection(root, tasks))
        if failures:
            _fail(f"fresh fit children failed: {','.join(failures)}")
    return validate_run(
        root,
        require_complete=False,
        required_task_ids=tuple(task.task_id for task in selected),
    )


def validate_run(
    run_dir: Path | str,
    *,
    require_complete: bool = True,
    required_task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    tasks = load_task_grid(root)
    completed: list[dict[str, Any]] = []
    states = {task.task_id: task_state(root, task.task_id) for task in tasks}
    for task in tasks:
        if states[task.task_id] == "completed":
            completed.append(validate_task_result(root, task))
    bad = {task_id: state for task_id, state in states.items() if state in {"failed", "interrupted", "ambiguous"}}
    if bad:
        _fail("run contains failed, interrupted, or ambiguous tasks")
    required = set(required_task_ids or ())
    if required and (not required <= {task.task_id for task in tasks} or any(states[task_id] != "completed" for task_id in required)):
        _fail("required execution subset is not complete")
    if require_complete and len(completed) != TASK_COUNT:
        _fail("run is not complete")
    by_gpu = tuple(sum(result["physical_gpu"] == gpu for result in completed) for gpu in range(GPU_COUNT))
    if require_complete and by_gpu != FITS_PER_GPU:
        _fail("completed result GPU accounting changed")
    return {
        "contract": EXPERIMENT_CONTRACT,
        "schema_version": 1,
        "run_id": RUN_ID,
        "task_count": TASK_COUNT,
        "completed_task_count": len(completed),
        "required_task_count": TASK_COUNT if require_complete else len(required),
        "completed_fits_by_gpu": list(by_gpu),
        "valid": not bad and (not require_complete or len(completed) == TASK_COUNT),
    }


def _finite_number(value: Any, *, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        _fail(f"{name} is not finite numeric data")
    return float(value)


def evaluate_result_payloads(
    results: Sequence[Mapping[str, Any]], *, require_full_grid: bool = True
) -> dict[str, Any]:
    """Aggregate persisted 15-view fit metrics without selecting a candidate."""

    if require_full_grid and len(results) != TASK_COUNT:
        _fail("evaluation requires exactly 252 fit results")
    if not results:
        _fail("evaluation requires at least one fit result")
    observed_candidates = tuple(
        candidate
        for candidate in CANDIDATE_IDS
        if any(result.get("candidate_id") == candidate for result in results)
    )
    cells: dict[tuple[str, str], list[float]] = {
        (candidate, view): [] for candidate in observed_candidates for view in VIEW_IDS
    }
    seen_tasks: set[str] = set()
    for result in results:
        task_id = result.get("task_id")
        candidate = result.get("candidate_id")
        training = result.get("training_result")
        if type(task_id) is not str or task_id in seen_tasks or candidate not in CANDIDATE_IDS or not isinstance(training, Mapping):
            _fail("evaluation result identity changed")
        seen_tasks.add(task_id)
        rows = training.get("view_metrics")
        if type(rows) is not list or len(rows) != len(VIEW_IDS):
            _fail("fit result must expose exactly fifteen view metrics")
        observed_views: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping) or row.get("view_id") not in VIEW_IDS:
                _fail("view metric row is malformed")
            view = row["view_id"]
            observed_views.append(view)
            cells[(candidate, view)].append(
                _finite_number(row.get("rmse3d_m"), name="rmse3d_m")
            )
        if observed_views != list(VIEW_IDS):
            _fail("view metric order changed")
    summaries: list[dict[str, Any]] = []
    expected_per_candidate = CONTEXT_COUNT if require_full_grid else None
    for candidate in observed_candidates:
        view_rows = []
        for view in VIEW_IDS:
            values = cells[(candidate, view)]
            if not values or (expected_per_candidate is not None and len(values) != expected_per_candidate):
                _fail("candidate/view evaluation cell count changed")
            view_rows.append(
                {
                    "view_id": view,
                    "fit_count": len(values),
                    "mean_rmse3d_m": math.fsum(values) / len(values),
                }
            )
        summaries.append({"candidate_id": candidate, "views": view_rows})
    return _payload(
        {
            "contract": EVALUATION_CONTRACT,
            "schema_version": 1,
            "run_id": RUN_ID,
            "task_count": len(results),
            "candidate_order": list(observed_candidates),
            "view_order": list(VIEW_IDS),
            "candidate_summaries": summaries,
            "candidate_selected": False,
            "full_preregistered_grid_complete": require_full_grid,
            "deferred_preregistered_ablation_fit_count": TASK_COUNT - len(results),
        }
    )


def evaluate_run(
    run_dir: Path | str,
    *,
    candidate_subset: Sequence[str] | None = None,
    context_subset: Sequence[int] | None = None,
    require_full_grid: bool = True,
) -> dict[str, Any]:
    root = Path(run_dir)
    tasks = load_task_grid(root)
    selected_candidates = set(candidate_subset or CANDIDATE_IDS)
    selected_contexts = set(context_subset or range(CONTEXT_COUNT))
    selected = [
        task for task in tasks
        if task.candidate_id in selected_candidates and task.context_ordinal in selected_contexts
    ]
    results = [validate_task_result(root, task) for task in selected]
    evaluation = evaluate_result_payloads(results, require_full_grid=require_full_grid)
    completed_full_grid = sum(task_state(root, task.task_id) == "completed" for task in tasks)
    evaluation.pop("payload_sha256")
    evaluation["completed_full_grid_fit_count"] = completed_full_grid
    evaluation["deferred_preregistered_ablation_fit_count"] = TASK_COUNT - completed_full_grid
    evaluation = _payload(evaluation)
    subset_hash = _canonical_hash(
        {
            "candidate_ids": [item for item in CANDIDATE_IDS if item in selected_candidates],
            "context_ordinals": sorted(selected_contexts),
        }
    )
    report_name = (
        "evaluation.json"
        if require_full_grid
        else f"evaluation_partial_{subset_hash[:16]}.json"
    )
    _write_json_exclusive(root / "reports" / report_name, evaluation)
    return evaluation


def validate_config_api() -> None:
    """Cross-check the implementation constants against the config module."""

    from . import openaxes_external_native_v3_config as native_config

    checks = {
        "OPENAXES_NATIVE_V3_CANDIDATE_IDS": CANDIDATE_IDS,
        "OPENAXES_NATIVE_V3_SEEDS": SEEDS,
        "OPENAXES_NATIVE_V3_VIEW_IDS": VIEW_IDS,
        "OPENAXES_NATIVE_V3_FOLD_COUNT": FOLD_COUNT,
        "OPENAXES_NATIVE_V3_CONTEXT_COUNT": CONTEXT_COUNT,
        "OPENAXES_NATIVE_V3_TASK_COUNT": TASK_COUNT,
        "OPENAXES_NATIVE_V3_FITS_PER_GPU": FITS_PER_GPU,
    }
    for name, expected in checks.items():
        observed = getattr(native_config, name, None)
        if isinstance(expected, tuple):
            observed = tuple(observed) if observed is not None else None
        if observed != expected:
            _fail(f"config API changed: {name}")


def parse_context_selector(value: str) -> tuple[int, ...]:
    """Parse ``0,3-5`` into a sorted unique context ordinal tuple."""

    if type(value) is not str or not value:
        _fail("context selector is empty")
    selected: set[int] = set()
    for token in value.split(","):
        if re.fullmatch(r"[0-9]+", token):
            selected.add(int(token))
            continue
        match = re.fullmatch(r"([0-9]+)-([0-9]+)", token)
        if match is None:
            _fail("context selector is malformed")
        start, end = (int(item) for item in match.groups())
        if end < start:
            _fail("context selector range is reversed")
        selected.update(range(start, end + 1))
    result = tuple(sorted(selected))
    if not result or any(not 0 <= item < CONTEXT_COUNT for item in result):
        _fail("context selector is outside 0..41")
    return result


def parse_candidate_selector(value: str) -> tuple[str, ...]:
    if value == "all":
        return CANDIDATE_IDS
    requested = tuple(item for item in value.split(",") if item)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(item not in CANDIDATE_IDS for item in requested)
    ):
        _fail("candidate selector is outside the frozen six-candidate roster")
    return tuple(item for item in CANDIDATE_IDS if item in requested)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    prepare.add_argument("--prepared-root", default=DEFAULT_PREPARED_ROOT)
    prepare.add_argument("--manifest", default=None)
    run = sub.add_parser("run")
    run.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    run.add_argument("--prepared-root", default=DEFAULT_PREPARED_ROOT)
    run.add_argument("--python-executable", default="/proc/self/exe")
    run.add_argument("--gpu-selectors", default="0,1,2,3")
    run.add_argument("--candidates", default="all")
    run.add_argument("--contexts", default="0-41")
    internal = sub.add_parser("internal-fit")
    internal.add_argument("--run-dir", required=True)
    internal.add_argument("--prepared-root", required=True)
    internal.add_argument("--task-id", required=True)
    internal.add_argument("--gpu-index", required=True, type=int)
    internal.add_argument("--gpu-selector", required=True)
    internal.add_argument("--attempt-id", required=True)
    status = sub.add_parser("status")
    status.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    for name in ("validate", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
        command.add_argument("--candidates", default="all")
        command.add_argument("--contexts", default="0-41")
        command.add_argument("--allow-partial", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        manifest_path = Path(args.manifest) if args.manifest else Path(args.prepared_root) / "manifest.json"
        manifest = _read_json(
            manifest_path, name="preparation manifest", require_canonical=False
        )
        prepare_run(args.run_dir, manifest)
    elif args.command == "run":
        selectors = tuple(item for item in args.gpu_selectors.split(",") if item)
        if not (Path(args.run_dir) / "task_index.json").exists():
            manifest = _read_json(
                Path(args.prepared_root) / "manifest.json",
                name="preparation manifest",
                require_canonical=False,
            )
            prepare_run(args.run_dir, manifest)
        run_scheduled_fits(
            args.run_dir,
            prepared_root=args.prepared_root,
            python_executable=args.python_executable,
            gpu_selectors=selectors,
            candidate_subset=parse_candidate_selector(args.candidates),
            context_subset=parse_context_selector(args.contexts),
        )
    elif args.command == "internal-fit":
        run_internal_fit(
            args.run_dir,
            args.task_id,
            prepared_root=args.prepared_root,
            gpu_index=args.gpu_index,
            gpu_selector=args.gpu_selector,
            attempt_id=args.attempt_id,
        )
    elif args.command == "status":
        print(json.dumps(status_projection(args.run_dir), sort_keys=True))
    elif args.command == "validate":
        candidates = parse_candidate_selector(args.candidates)
        contexts = parse_context_selector(args.contexts)
        tasks = load_task_grid(args.run_dir)
        required = tuple(
            task.task_id for task in tasks
            if task.candidate_id in candidates and task.context_ordinal in contexts
        )
        validate_run(
            args.run_dir,
            require_complete=not args.allow_partial,
            required_task_ids=required,
        )
    elif args.command == "evaluate":
        evaluate_run(
            args.run_dir,
            candidate_subset=parse_candidate_selector(args.candidates),
            context_subset=parse_context_selector(args.contexts),
            require_full_grid=not args.allow_partial,
        )
    else:
        _fail("unknown command")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except OpenAxesExternalNativeV3ExperimentError as exc:
        print(f"error: {exc}", file=os.sys.stderr)
        raise SystemExit(2) from None

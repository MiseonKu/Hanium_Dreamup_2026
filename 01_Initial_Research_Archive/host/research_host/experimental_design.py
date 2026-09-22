from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


TRIAL_PLAN_FIELDNAMES = [
    "planned_order_index",
    "block_id",
    "condition_id",
    "robot_model",
    "target_id",
    "repeat_id",
    "randomization_seed",
]


@dataclass(frozen=True, slots=True)
class TrialSpec:
    planned_order_index: int
    block_id: str
    condition_id: str
    robot_model: str
    target_id: str
    repeat_id: int
    randomization_seed: int


def generate_balanced_trial_plan(
    *,
    conditions: Sequence[str],
    robot_models: Sequence[str],
    target_ids: Sequence[str],
    repeats: int,
    seed: int,
) -> list[TrialSpec]:
    if not conditions:
        raise ValueError("conditions must not be empty")
    if not robot_models:
        raise ValueError("robot_models must not be empty")
    if not target_ids:
        raise ValueError("target_ids must not be empty")
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    rng = random.Random(seed)
    planned: list[TrialSpec] = []
    previous_condition: str | None = None

    for robot_model in robot_models:
        for target_id in target_ids:
            for repeat_id in range(1, repeats + 1):
                block_id = f"{robot_model}_{target_id}_rep{repeat_id:02d}"
                block_conditions = list(conditions)
                rng.shuffle(block_conditions)
                block_conditions = _rotate_away_from_previous(
                    block_conditions,
                    previous_condition=previous_condition,
                )
                for condition_id in block_conditions:
                    planned.append(
                        TrialSpec(
                            planned_order_index=len(planned),
                            block_id=block_id,
                            condition_id=condition_id,
                            robot_model=robot_model,
                            target_id=target_id,
                            repeat_id=repeat_id,
                            randomization_seed=seed,
                        )
                    )
                    previous_condition = condition_id
    return planned


def _rotate_away_from_previous(
    condition_ids: list[str],
    *,
    previous_condition: str | None,
) -> list[str]:
    if previous_condition is None or len(condition_ids) < 2:
        return condition_ids
    if condition_ids[0] != previous_condition:
        return condition_ids
    for index, condition_id in enumerate(condition_ids[1:], start=1):
        if condition_id != previous_condition:
            return condition_ids[index:] + condition_ids[:index]
    return condition_ids


def trial_plan_to_rows(plan: Iterable[TrialSpec]) -> list[dict[str, object]]:
    return [asdict(spec) for spec in plan]


def trial_plan_from_rows(rows: Iterable[Mapping[str, object]]) -> list[TrialSpec]:
    return [
        TrialSpec(
            planned_order_index=int(row["planned_order_index"]),
            block_id=str(row["block_id"]),
            condition_id=str(row["condition_id"]),
            robot_model=str(row["robot_model"]),
            target_id=str(row["target_id"]),
            repeat_id=int(row["repeat_id"]),
            randomization_seed=int(row["randomization_seed"]),
        )
        for row in rows
    ]


def read_trial_plan_csv(path: str | Path) -> list[TrialSpec]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return trial_plan_from_rows(csv.DictReader(handle))


def plan_digest(plan: Iterable[TrialSpec]) -> str:
    rows = trial_plan_to_rows(plan)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_trial_plan(plan: Sequence[TrialSpec]) -> list[str]:
    issues: list[str] = []
    if not plan:
        return ["trial plan is empty"]

    expected_order = list(range(len(plan)))
    observed_order = [spec.planned_order_index for spec in plan]
    if observed_order != expected_order:
        issues.append("planned_order_index must be contiguous and sorted from 0")

    seeds = {spec.randomization_seed for spec in plan}
    if len(seeds) != 1:
        issues.append("trial plan must record exactly one randomization_seed")

    unique_rows = {
        (
            spec.condition_id,
            spec.robot_model,
            spec.target_id,
            spec.repeat_id,
        )
        for spec in plan
    }
    if len(unique_rows) != len(plan):
        issues.append("trial plan contains duplicate condition/model/target/repeat rows")

    for previous, current in zip(plan, plan[1:]):
        if previous.condition_id == current.condition_id:
            issues.append(
                "adjacent trials repeat condition_id "
                f"{current.condition_id} at order {current.planned_order_index}"
            )
            break

    block_condition_counts: dict[str, Counter[str]] = {}
    for spec in plan:
        block_condition_counts.setdefault(spec.block_id, Counter())
        block_condition_counts[spec.block_id][spec.condition_id] += 1
    for block_id, counter in block_condition_counts.items():
        if len(set(counter.values())) != 1:
            issues.append(f"block {block_id} is not condition-balanced")

    strata_counts = Counter(
        (spec.condition_id, spec.robot_model, spec.target_id) for spec in plan
    )
    if len(set(strata_counts.values())) != 1:
        issues.append("condition/model/target strata are not balanced")

    return issues


def write_trial_plan_csv(path: str | Path, plan: Sequence[TrialSpec]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRIAL_PLAN_FIELDNAMES)
        writer.writeheader()
        for row in trial_plan_to_rows(plan):
            writer.writerow({field: row[field] for field in TRIAL_PLAN_FIELDNAMES})
    return output_path


def write_trial_plan_metadata(
    path: str | Path,
    plan: Sequence[TrialSpec],
    *,
    source: str = "research_host.experimental_design.generate_balanced_trial_plan",
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "source": source,
        "trial_count": len(plan),
        "randomization_seed": plan[0].randomization_seed if plan else None,
        "plan_digest_sha256": plan_digest(plan),
        "validation_issues": validate_trial_plan(plan),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path

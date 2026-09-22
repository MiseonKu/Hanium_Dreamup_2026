from __future__ import annotations

import csv

from research_host.experimental_design import (
    generate_balanced_trial_plan,
    plan_digest,
    validate_trial_plan,
    write_trial_plan_csv,
    write_trial_plan_metadata,
)


def test_trial_plan_is_deterministic_balanced_and_non_adjacent() -> None:
    kwargs = {
        "conditions": ["C1", "C2", "C6"],
        "robot_models": ["long", "short"],
        "target_ids": ["T001", "T002"],
        "repeats": 2,
        "seed": 101,
    }
    plan = generate_balanced_trial_plan(**kwargs)
    same_plan = generate_balanced_trial_plan(**kwargs)

    assert plan == same_plan
    assert len(plan) == 24
    assert validate_trial_plan(plan) == []
    assert plan_digest(plan) == plan_digest(same_plan)


def test_trial_plan_writes_csv_and_metadata(tmp_path) -> None:
    plan = generate_balanced_trial_plan(
        conditions=["C1", "C2"],
        robot_models=["long"],
        target_ids=["T001"],
        repeats=1,
        seed=7,
    )
    csv_path = write_trial_plan_csv(tmp_path / "plan.csv", plan)
    metadata_path = write_trial_plan_metadata(tmp_path / "plan.json", plan)

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["randomization_seed"] == "7"
    assert metadata_path.read_text(encoding="utf-8").count("plan_digest_sha256") == 1


def test_trial_plan_validator_catches_duplicate_order() -> None:
    plan = generate_balanced_trial_plan(
        conditions=["C1", "C2"],
        robot_models=["long"],
        target_ids=["T001"],
        repeats=1,
        seed=7,
    )
    broken = [plan[1], plan[0]]

    assert "planned_order_index" in validate_trial_plan(broken)[0]

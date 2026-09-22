from __future__ import annotations

import argparse
from pathlib import Path

from ..experimental_design import (
    generate_balanced_trial_plan,
    validate_trial_plan,
    write_trial_plan_csv,
    write_trial_plan_metadata,
)


def _csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must contain at least one item")
    return items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic blocked trial randomization plan."
    )
    parser.add_argument("--conditions", type=_csv_list, required=True)
    parser.add_argument("--robot-models", type=_csv_list, required=True)
    parser.add_argument("--targets", type=_csv_list, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = generate_balanced_trial_plan(
        conditions=args.conditions,
        robot_models=args.robot_models,
        target_ids=args.targets,
        repeats=args.repeats,
        seed=args.seed,
    )
    issues = validate_trial_plan(plan)
    write_trial_plan_csv(args.output, plan)
    write_trial_plan_metadata(args.metadata_output, plan)

    print(f"Trial plan rows: {len(plan)}")
    print(f"Output: {args.output}")
    print(f"Metadata: {args.metadata_output}")
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    print("Trial plan OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

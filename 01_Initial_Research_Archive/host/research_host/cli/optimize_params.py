from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..optimization import load_objective_config, select_best_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select the best offline parameter candidate from validation metrics."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--objective-config", type=Path)
    parser.add_argument(
        "--allowed-split",
        help="Override objective-config allowed_split. Defaults to config value or validation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with args.input.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    weights = None
    constraints = None
    allowed_split = args.allowed_split or "validation"
    if args.objective_config is not None:
        objective = load_objective_config(args.objective_config)
        weights = objective.weights
        constraints = objective.constraint_max
        allowed_split = args.allowed_split or objective.allowed_split

    best = select_best_candidate(
        rows,
        weights=weights,
        constraint_max=constraints,
        allowed_split=allowed_split,
    )
    payload = {
        "source_metrics": str(args.input),
        "allowed_split": allowed_split,
        "selected": best.to_jsonable(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..logging import PROCESSED_FIELDNAMES


REQUIRED_FILES = [
    "README.md",
    "AGENTS.md",
    "docs/AI_AGENT_GUIDE.md",
    "docs/CODE_MAP.md",
    "docs/DATA_CONTRACT.md",
    "docs/DECISION_LOG.md",
    "docs/OPTIMIZATION_AND_RANDOMIZATION.md",
    "docs/SO101_IK_REFERENCE.md",
    "docs/SIMULATION_INTEGRATION.md",
    "docs/VALIDITY_THREATS.md",
    "docs/REVIEW_CHECKLIST.md",
    "configs/agent_context.json",
    "configs/experiment_defaults.json",
    "configs/randomization_defaults.json",
    "configs/optimization_objective.json",
    "configs/ik_baseline.json",
    "configs/simulation_targets.json",
    "host/research_host/experimental_design.py",
    "host/research_host/optimization.py",
    "host/research_host/quality.py",
    "host/research_host/calibration_contracts.py",
    "host/research_host/reports.py",
    "host/research_host/sim/exporters.py",
    "host/research_host/sim/replay.py",
    "host/research_host/shapefk.py",
    "host/research_host/logging.py",
    "host/research_host/cli/summarize_quality.py",
    "host/research_host/cli/validate_calibration.py",
    "host/research_host/cli/compare_conditions.py",
    "host/research_host/cli/replay_mujoco_trial.py",
    "host/research_host/cli/replay_ros2_joint_states.py",
    "simulation/README.md",
    "simulation/generated/so101_long_manifest.json",
    "simulation/generated/so101_short_manifest.json",
    "simulation/generated/urdf/so101_long.urdf",
    "simulation/generated/mjcf/so101_long.xml",
]


REQUIRED_AGENT_CONTEXT_KEYS = {
    "schema_version",
    "project",
    "purpose",
    "start_files",
    "hard_rules",
    "operation_classes",
    "task_routing",
    "commands",
    "public_contract_files",
    "safe_extension_points",
    "ask_before_changing",
    "reporting_contract",
}


def _require_string_list(
    context: dict[str, object],
    key: str,
    errors: list[str],
) -> list[str]:
    value = context.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        errors.append(f"configs/agent_context.json {key} must be a list of strings")
        return []
    return value


def _require_path_list(
    root: Path,
    context: dict[str, object],
    key: str,
    errors: list[str],
) -> None:
    for relative in _require_string_list(context, key, errors):
        if not (root / relative).exists():
            errors.append(f"{key} path does not exist: {relative}")


def _validate_agent_context(root: Path, errors: list[str]) -> None:
    context_path = root / "configs" / "agent_context.json"
    if not context_path.exists():
        return

    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid configs/agent_context.json: {exc}")
        return
    if not isinstance(context, dict):
        errors.append("configs/agent_context.json root must be an object")
        return

    missing_keys = sorted(REQUIRED_AGENT_CONTEXT_KEYS - set(context))
    if missing_keys:
        errors.append(
            "configs/agent_context.json missing keys: " + ", ".join(missing_keys)
        )

    _require_path_list(root, context, "start_files", errors)
    _require_path_list(root, context, "public_contract_files", errors)
    _require_path_list(root, context, "safe_extension_points", errors)
    _require_string_list(context, "hard_rules", errors)
    _require_string_list(context, "ask_before_changing", errors)
    _require_string_list(context, "reporting_contract", errors)

    commands = context.get("commands", {})
    if not isinstance(commands, dict) or not all(
        isinstance(name, str) and isinstance(command, str)
        for name, command in commands.items()
    ):
        errors.append("configs/agent_context.json commands must map strings to strings")

    operation_classes = context.get("operation_classes", {})
    if not isinstance(operation_classes, dict):
        errors.append("configs/agent_context.json operation_classes must be an object")
    else:
        for class_name, entries in operation_classes.items():
            if not isinstance(class_name, str) or not isinstance(entries, list) or not all(
                isinstance(entry, str) for entry in entries
            ):
                errors.append(
                    "configs/agent_context.json operation_classes values must be string lists"
                )

    task_routing = context.get("task_routing", {})
    if not isinstance(task_routing, dict):
        errors.append("configs/agent_context.json task_routing must be an object")
    else:
        for task_name, paths in task_routing.items():
            if not isinstance(task_name, str) or not isinstance(paths, list) or not all(
                isinstance(path, str) for path in paths
            ):
                errors.append(
                    "configs/agent_context.json task_routing values must be path lists"
                )
                continue
            for relative in paths:
                if not (root / relative).exists():
                    errors.append(f"task_routing path does not exist: {relative}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check AI-agent reference files and core offline contract."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="01_Initial_Research_Archive root. Defaults to current directory.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    errors: list[str] = []

    for relative in REQUIRED_FILES:
        if not (root / relative).exists():
            errors.append(f"missing required file: {relative}")

    _validate_agent_context(root, errors)

    if not any(name.startswith("P_hat_imu_base_mm") for name in PROCESSED_FIELDNAMES):
        errors.append("processed CSV contract is missing P_hat_imu_base_mm fields")
    if "quality_flags" not in PROCESSED_FIELDNAMES:
        errors.append("processed CSV contract is missing quality_flags")

    if errors:
        print("Agent doctor found problems:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Agent doctor OK")
    print(f"Root: {root}")
    print(f"Required files: {len(REQUIRED_FILES)}")
    print(f"Processed CSV fields: {len(PROCESSED_FIELDNAMES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..logging import read_csv_dicts
from ..sim.replay import (
    OptionalDependencyError,
    build_replay_frames,
    run_ros2_joint_state_replay,
    summarize_replay_frames,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay processed joint states as ROS 2 JointState messages.")
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_csv_dicts(args.processed)
    if args.dry_run:
        payload = summarize_replay_frames(build_replay_frames(rows))
        payload["backend"] = "ros2_joint_states_dry_run"
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    try:
        print(json.dumps(run_ros2_joint_state_replay(rows), indent=2, sort_keys=True))
    except OptionalDependencyError as exc:
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

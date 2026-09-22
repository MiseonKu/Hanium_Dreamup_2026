"""Generate balanced repeated trial schedule."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fkik.geometry import load_config
from workspace_points.balanced_shuffle import generate_trial_schedule


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate balanced trial schedule.")
    parser.add_argument("--config", default="config/robot_config.yaml")
    parser.add_argument("--targets", default="outputs/target_points.csv")
    parser.add_argument("--output", default="outputs/trial_schedule.csv")
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    targets = pd.read_csv(_resolve(args.targets))
    schedule = generate_trial_schedule(targets, config)
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(output, index=False)
    print(f"wrote {output}")
    print(f"trials={len(schedule)} targets={targets['target_id'].nunique()} repeats={config.repeats_per_point}")


if __name__ == "__main__":
    main()

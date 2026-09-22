"""Generate uniform target points from workspace candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fkik.geometry import load_config
from workspace_points.binning import add_bin_labels
from workspace_points.candidate_generator import (
    candidate_metadata_matches,
    generate_candidates,
    write_candidate_metadata,
    write_round_trip_report,
)
from workspace_points.uniform_bin_sampler import sample_uniform_targets


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bin-wise uniform target points.")
    parser.add_argument("--config", default="config/robot_config.yaml")
    parser.add_argument("--candidates", default="outputs/workspace_candidates.csv")
    parser.add_argument("--output", default="outputs/target_points.csv")
    parser.add_argument("--report", default="outputs/point_distribution_report.csv")
    parser.add_argument("--force-candidates", action="store_true")
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    candidates_path = _resolve(args.candidates)
    output = _resolve(args.output)
    report_path = _resolve(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = candidates_path.with_suffix(".meta.json")

    if candidates_path.exists() and not args.force_candidates and candidate_metadata_matches(metadata_path, config):
        candidates = pd.read_csv(candidates_path)
    else:
        candidates = generate_candidates(config)
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        candidates.to_csv(candidates_path, index=False)
        write_candidate_metadata(metadata_path, config, candidates)
        write_round_trip_report(candidates, str(candidates_path.parent / "round_trip_error_report.csv"))

    candidates = add_bin_labels(candidates, config)
    candidates.to_csv(candidates_path, index=False)
    targets, report = sample_uniform_targets(candidates, config)
    targets.to_csv(output, index=False)
    report.to_csv(report_path, index=False)

    print(f"wrote {output}")
    print(f"targets={len(targets)} bins={targets['bin_id'].nunique()} report={report_path}")


if __name__ == "__main__":
    main()

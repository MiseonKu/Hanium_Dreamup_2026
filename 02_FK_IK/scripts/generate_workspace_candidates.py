"""Generate workspace candidate CSV from joint-space samples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fkik.geometry import load_config
from workspace_points.candidate_generator import generate_candidates, write_candidate_metadata, write_round_trip_report


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SO-101 flexible-link workspace candidates.")
    parser.add_argument("--config", default="config/robot_config.yaml")
    parser.add_argument("--output", default="outputs/workspace_candidates.csv")
    parser.add_argument("--n-candidates", type=int, default=None)
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    candidates = generate_candidates(config, n_candidates=args.n_candidates)
    candidates.to_csv(output, index=False)
    write_candidate_metadata(output.with_suffix(".meta.json"), config, candidates)
    report_path = output.parent / "round_trip_error_report.csv"
    write_round_trip_report(candidates, str(report_path))

    print(f"wrote {output}")
    print(f"metadata={output.with_suffix('.meta.json')}")
    print(f"candidates={len(candidates)} valid={int(candidates['valid'].sum())} report={report_path}")


if __name__ == "__main__":
    main()

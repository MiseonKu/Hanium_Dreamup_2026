from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ..experimental_design import plan_digest, read_trial_plan_csv, validate_trial_plan
from ..logging import PROCESSED_FIELDNAMES
from ..quality import load_quality_thresholds, quality_issues, summarize_quality


REQUIRED_METADATA_KEYS = {
    "trial_id",
    "condition_id",
    "robot_model",
    "target_id",
    "repeat_id",
    "date",
    "calibration_versions",
    "filter_params_id",
    "fusion_params_id",
    "quality_thresholds",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate CSV+JSON trial contract.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--trial-plan", type=Path)
    parser.add_argument("--trial-plan-metadata", type=Path)
    parser.add_argument("--optimization-report", type=Path)
    parser.add_argument("--sim-manifest", type=Path)
    parser.add_argument("--quality-thresholds", type=Path, default=Path("configs/quality_thresholds.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    errors: list[str] = []
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    missing_metadata = sorted(REQUIRED_METADATA_KEYS - set(metadata))
    with args.processed.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing_columns = sorted(set(PROCESSED_FIELDNAMES) - set(reader.fieldnames or []))
        rows = list(reader)
        row_count = len(rows)

    if missing_metadata:
        errors.append("Missing metadata keys: " + ", ".join(missing_metadata))
    if missing_columns:
        errors.append("Missing processed columns: " + ", ".join(missing_columns))
    if row_count == 0:
        errors.append("Processed CSV has no rows")

    if args.quality_thresholds.exists():
        summary = summarize_quality(rows)
        errors.extend(
            "Quality issue: " + issue
            for issue in quality_issues(summary, load_quality_thresholds(args.quality_thresholds))
        )

    if args.trial_plan:
        plan = read_trial_plan_csv(args.trial_plan)
        errors.extend("Trial plan issue: " + issue for issue in validate_trial_plan(plan))
        if args.trial_plan_metadata:
            plan_metadata = json.loads(args.trial_plan_metadata.read_text(encoding="utf-8"))
            expected_digest = plan_metadata.get("plan_digest_sha256")
            observed_digest = plan_digest(plan)
            if expected_digest != observed_digest:
                errors.append(
                    f"Trial plan digest mismatch: metadata={expected_digest} observed={observed_digest}"
                )

    if args.optimization_report:
        optimization_report = json.loads(args.optimization_report.read_text(encoding="utf-8"))
        if optimization_report.get("allowed_split") != "validation":
            errors.append("Optimization report allowed_split must be validation")
        if "selected" not in optimization_report:
            errors.append("Optimization report is missing selected candidate")

    if args.sim_manifest:
        sim_manifest = json.loads(args.sim_manifest.read_text(encoding="utf-8"))
        if "model_id" not in sim_manifest or "generated_files" not in sim_manifest:
            errors.append("Simulation manifest must include model_id and generated_files")
        else:
            for name, relative_path in sim_manifest["generated_files"].items():
                candidate = Path(str(relative_path))
                if not candidate.is_absolute():
                    candidate = Path.cwd() / candidate
                if not candidate.exists():
                    errors.append(f"Simulation manifest file missing for {name}: {relative_path}")

    if errors:
        for error in errors:
            print(error)
        return 1

    print("Dataset contract OK")
    print(f"Rows: {row_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..calibration_contracts import (
    load_template_required_keys,
    validate_calibration_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate calibration JSON files before data collection.")
    parser.add_argument("--calibration-dir", type=Path, default=Path("calibration"))
    parser.add_argument("--templates", type=Path, default=Path("calibration/templates"))
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    template_keys = load_template_required_keys(args.templates)
    results = validate_calibration_dir(args.calibration_dir)
    issues = [
        {"calibration_type": result.calibration_type, "path": result.path, "issues": result.issues}
        for result in results
        if result.issues
    ]
    payload = {
        "template_types": sorted(template_keys),
        "checked_count": len(results),
        "issues": issues,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Calibration validation")
        print("Template types:", ", ".join(payload["template_types"]))
        if issues:
            print("Calibration issues:")
            for issue in issues:
                print(f"- {issue['calibration_type']}: " + "; ".join(issue["issues"]))
        else:
            print("Calibration OK")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())

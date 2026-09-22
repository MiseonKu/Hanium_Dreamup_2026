"""Standalone CLI for the isolated OpenAXES external-v1 preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .openaxes_external_v1 import (
    OpenAxesExternalV1Error,
    prepare_openaxes_external_v1,
    validate_openaxes_external_v1,
)
from .openaxes_external_v1_config import (
    OpenAxesExternalV1ConfigError,
    load_openaxes_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m imu_vive_learning.openaxes_external_v1_cli",
        description=(
            "Authenticate and prepare the isolated OpenAXES external-v1 public "
            "benchmark without extracting or executing publisher code."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Create a new immutable preparation-only artifact."
    )
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate", help="Replay source preparation and validate every artifact value."
    )
    validate.add_argument("--source", type=Path, required=True)
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_openaxes_config(args.config)
        if args.command == "prepare":
            manifest = prepare_openaxes_external_v1(
                args.source, config, args.output
            )
            payload = {
                "status": "ok",
                "contract": manifest["contract"],
                "run_dir": str(args.output.resolve()),
                "counts": manifest["counts"],
                "license": manifest["license"],
                "sensitivity_recipe": manifest["preprocessing"][
                    "sensitivity_recipe"
                ],
                "provenance_stage": manifest["provenance"]["stage"],
                "preparation_only": True,
            }
        else:
            payload = validate_openaxes_external_v1(
                args.source, config, args.run_dir
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OpenAxesExternalV1Error, OpenAxesExternalV1ConfigError) as exc:
        print(
            json.dumps(
                {"status": "error", "error": str(exc)},
                ensure_ascii=False,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

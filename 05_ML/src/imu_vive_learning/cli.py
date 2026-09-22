from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .io_utils import sha256_file
from .packaging import package_run
from .pipeline import build_inner_cache, build_run
from .validation import validate_package, validate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imu-vive-mlprep",
        description="Build and validate the leakage-safe AlphaF trial-level ML dataset.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Audit, clean, split, augment, and manifest a run.")
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="Validate an existing unpacked run.")
    validate.add_argument("--run-dir", type=Path, required=True)

    package = subparsers.add_parser("package", help="Validate and create a deterministic release ZIP.")
    package.add_argument("--run-dir", type=Path, required=True)
    package.add_argument("--output", type=Path, required=True)

    inner = subparsers.add_parser(
        "cache-inner",
        help="Generate both synthetic variants for one segment-LOSO inner-train context.",
    )
    inner.add_argument("--run-dir", type=Path, required=True)
    inner.add_argument("--outer-fold", type=int, required=True)
    inner.add_argument("--inner-fold", type=int, required=True)
    inner.add_argument("--output", type=Path, required=True)

    experiment = subparsers.add_parser(
        "run-experiment",
        help="Run the frozen nested-CV endpoint-regression experiment matrix.",
    )
    experiment.add_argument("--release", type=Path, required=True)
    experiment.add_argument("--config", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--jobs", type=int, default=4)
    experiment.add_argument("--resume", action="store_true")

    validate_experiment = subparsers.add_parser(
        "validate-experiment", help="Validate an unpacked endpoint-regression run."
    )
    validate_experiment.add_argument("--run-dir", type=Path, required=True)

    report_experiment = subparsers.add_parser(
        "report-experiment", help="Regenerate the deterministic experiment report."
    )
    report_experiment.add_argument("--run-dir", type=Path, required=True)

    package_experiment = subparsers.add_parser(
        "package-experiment", help="Validate and package endpoint-regression experiment v1."
    )
    package_experiment.add_argument("--run-dir", type=Path, required=True)
    package_experiment.add_argument("--output", type=Path, required=True)

    predict = subparsers.add_parser(
        "predict-endpoint", help="Run strict offline inference from the final model package."
    )
    predict.add_argument("--model-package", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)

    gpu_predict = subparsers.add_parser(
        "predict-gpu-endpoint",
        help="Run safe offline inference from a completed GPU model directory.",
    )
    gpu_predict.add_argument("--run-dir", type=Path, required=True)
    gpu_predict.add_argument("--input", type=Path, required=True)
    gpu_predict.add_argument("--output", type=Path, required=True)
    gpu_predict.add_argument(
        "--device", default="cpu", help="PyTorch device: cpu or cuda[:index]."
    )
    gpu_predict.add_argument(
        "--batch-size", type=int, default=4096, help="Positive inference batch size."
    )

    gpu_smoke = subparsers.add_parser(
        "smoke-gpu-experiment",
        help="Exercise the four-GPU runner with a non-claimable short diagnostic.",
    )
    gpu_smoke.add_argument("--release", type=Path, required=True)
    gpu_smoke.add_argument("--config", type=Path, required=True)
    gpu_smoke.add_argument("--output", type=Path, required=True)
    gpu_smoke.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    gpu_smoke.add_argument("--resume", action="store_true")

    gpu_run = subparsers.add_parser(
        "run-gpu-experiment",
        help="Run the frozen four-GPU nested endpoint-regression experiment.",
    )
    gpu_run.add_argument("--release", type=Path, required=True)
    gpu_run.add_argument("--config", type=Path, required=True)
    gpu_run.add_argument("--output", type=Path, required=True)
    gpu_run.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    gpu_run.add_argument("--resume", action="store_true")

    gpu_validate = subparsers.add_parser(
        "validate-gpu-experiment", help="Validate a completed four-GPU experiment run."
    )
    gpu_validate.add_argument("--run-dir", type=Path, required=True)

    robust_gpu_smoke = subparsers.add_parser(
        "smoke-gpu-robust-experiment",
        help="Exercise robust-v2 GPU training with a short non-claimable diagnostic.",
    )
    robust_gpu_smoke.add_argument("--release", type=Path, required=True)
    robust_gpu_smoke.add_argument("--config", type=Path, required=True)
    robust_gpu_smoke.add_argument("--output", type=Path, required=True)
    robust_gpu_smoke.add_argument(
        "--gpus", type=int, nargs=4, default=[0, 1, 2, 3]
    )
    robust_gpu_smoke.add_argument("--resume", action="store_true")

    robust_gpu_run = subparsers.add_parser(
        "run-gpu-robust-experiment",
        help="Run the frozen four-GPU segment-robust endpoint experiment v2.",
    )
    robust_gpu_run.add_argument("--release", type=Path, required=True)
    robust_gpu_run.add_argument("--config", type=Path, required=True)
    robust_gpu_run.add_argument("--output", type=Path, required=True)
    robust_gpu_run.add_argument(
        "--gpus", type=int, nargs=4, default=[0, 1, 2, 3]
    )
    robust_gpu_run.add_argument("--resume", action="store_true")

    robust_gpu_validate = subparsers.add_parser(
        "validate-gpu-robust-experiment",
        help="Semantically validate a completed segment-robust GPU v2 run.",
    )
    robust_gpu_validate.add_argument("--run-dir", type=Path, required=True)

    outage_gpu_run = subparsers.add_parser(
        "run-gpu-outage-experiment",
        help="Run the prospective four-GPU single-IMU-outage experiment v3.",
    )
    outage_gpu_run.add_argument("--development-release", type=Path, required=True)
    outage_gpu_run.add_argument("--source", type=Path, required=True)
    outage_gpu_run.add_argument("--config", type=Path, required=True)
    outage_gpu_run.add_argument("--output", type=Path, required=True)
    outage_gpu_run.add_argument(
        "--gpus", type=int, nargs=4, default=[0, 1, 2, 3]
    )
    outage_gpu_run.add_argument("--resume", action="store_true")
    outage_gpu_run.add_argument(
        "--finalize-challenge",
        action="store_true",
        help=(
            "Only with --resume on an existing validated sealed-development run: "
            "refit/freeze and perform the one-time challenge-label access. Fresh "
            "runs always stop sealed."
        ),
    )

    outage_gpu_validate = subparsers.add_parser(
        "validate-gpu-outage-experiment",
        help="Validate durable outage-v3 evidence without reopening source labels.",
    )
    outage_gpu_validate.add_argument("--run-dir", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_run(load_config(args.config), args.output)
            validation = validate_run(args.output)
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "contract": manifest["contract"],
                        "run_dir": str(args.output.resolve()),
                        "counts": manifest["counts"],
                        "validation": validation,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "validate":
            result = validate_run(args.run_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "package":
            validate_run(args.run_dir)
            archive = package_run(args.run_dir, args.output)
            result = validate_package(archive)
            result["sha256"] = sha256_file(archive)
            result["path"] = str(archive)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "cache-inner":
            validate_run(args.run_dir)
            manifest = build_inner_cache(
                args.run_dir,
                outer_fold=args.outer_fold,
                inner_fold=args.inner_fold,
                output_dir=args.output,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "contract": manifest["contract"],
                        "split": manifest["split"],
                        "output": str(args.output.resolve()),
                        "augmentation": manifest["augmentation"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "run-experiment":
            from .experiment_config import load_experiment_config
            from .experiment_runner import run_experiment

            config = load_experiment_config(
                args.config, release_override=args.release
            )
            manifest = run_experiment(
                config,
                args.output,
                jobs=args.jobs,
                resume=args.resume,
            )
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "contract": manifest["contract"],
                        "run_dir": str(args.output.resolve()),
                        "counts": manifest["counts"],
                        "final_model": manifest["final_model"]["selection"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "validate-experiment":
            from .experiment_validation import validate_experiment_run

            result = validate_experiment_run(args.run_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "report-experiment":
            from .experiment_reporting import write_experiment_report
            from .experiment_validation import validate_experiment_run

            validate_experiment_run(args.run_dir)
            report = write_experiment_report(args.run_dir)
            validation = validate_experiment_run(args.run_dir)
            print(
                json.dumps(
                    {"status": "ok", "report": report, "validation": validation},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "package-experiment":
            from .experiment_packaging import (
                package_experiment_run,
                validate_experiment_package,
            )
            from .experiment_validation import validate_experiment_run

            validate_experiment_run(args.run_dir)
            archive = package_experiment_run(args.run_dir, args.output)
            result = validate_experiment_package(archive)
            result.pop("manifest", None)
            result["sha256"] = sha256_file(archive)
            result["path"] = str(archive)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "predict-endpoint":
            from .experiment_inference import predict_endpoint

            result = predict_endpoint(
                args.model_package,
                args.input,
                args.output,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "predict-gpu-endpoint":
            from .gpu_inference import predict_gpu_endpoint

            result = predict_gpu_endpoint(
                args.run_dir,
                args.input,
                args.output,
                device=args.device,
                batch_size=args.batch_size,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"smoke-gpu-experiment", "run-gpu-experiment"}:
            from .gpu_experiment import run_gpu_experiment, run_gpu_smoke
            from .gpu_experiment_config import load_gpu_config

            config = load_gpu_config(args.config, release_override=args.release)
            runner = (
                run_gpu_smoke
                if args.command == "smoke-gpu-experiment"
                else run_gpu_experiment
            )
            result = runner(
                config,
                args.output,
                gpu_ids=args.gpus,
                resume=args.resume,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "validate-gpu-experiment":
            from .gpu_experiment import validate_gpu_run

            result = validate_gpu_run(args.run_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command in {
            "smoke-gpu-robust-experiment",
            "run-gpu-robust-experiment",
        }:
            from .gpu_robust_experiment import (
                run_robust_gpu_experiment,
                run_robust_gpu_smoke,
            )
            from .gpu_robust_experiment_config import load_robust_gpu_config

            config = load_robust_gpu_config(
                args.config, release_override=args.release
            )
            runner = (
                run_robust_gpu_smoke
                if args.command == "smoke-gpu-robust-experiment"
                else run_robust_gpu_experiment
            )
            result = runner(
                config,
                args.output,
                gpu_ids=args.gpus,
                resume=args.resume,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "validate-gpu-robust-experiment":
            from .gpu_robust_experiment import validate_robust_gpu_run

            result = validate_robust_gpu_run(args.run_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "run-gpu-outage-experiment":
            from .gpu_outage_experiment import run_outage_gpu_experiment
            from .gpu_outage_experiment_config import load_outage_gpu_config

            config = load_outage_gpu_config(
                args.config,
                development_release_override=args.development_release,
                source_override=args.source,
            )
            result = run_outage_gpu_experiment(
                config,
                args.output,
                gpu_ids=args.gpus,
                resume=args.resume,
                finalize_challenge=args.finalize_challenge,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "validate-gpu-outage-experiment":
            from .gpu_outage_experiment import (
                validate_outage_development_run,
                validate_outage_gpu_run,
            )

            result = (
                validate_outage_gpu_run(args.run_dir)
                if (args.run_dir / "manifest.json").is_file()
                else validate_outage_development_run(args.run_dir)
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
    except Exception as exc:  # CLI boundary: return a stable non-zero status.
        if args.command == "predict-gpu-endpoint":
            print(
                json.dumps(
                    {
                        "status": "error",
                        "command": "predict-gpu-endpoint",
                        "error": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                file=sys.stderr,
            )
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

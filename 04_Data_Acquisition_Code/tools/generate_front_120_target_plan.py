"""Thin CLI wrapper around
`02_FK_IK/scripts/generate_pre_imu_targets.py`
that always runs `--mode front_120` (DECISIONS.md #2) and defaults its
output under this package's own `outputs/` directory instead of that
package's `outputs/`, so this task's artifacts stay separate.

No generation logic lives here - all sampling/FK/gating/binning/CSV/
metadata code is reused as-is from the 02 package by importing and calling
its existing `main()` with a constructed argv, exactly like running that
script directly with `--mode front_120`.
"""
from __future__ import annotations

import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
PKG08_ROOT = TOOLS_DIR.parent
REPO_ROOT = PKG08_ROOT.parent
PKG02_ROOT = REPO_ROOT / "02_FK_IK"
PKG02_SCRIPTS = PKG02_ROOT / "scripts"

if str(PKG02_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(PKG02_SCRIPTS))

import generate_pre_imu_targets as pre_imu_cli  # noqa: E402  (02 package script, same Python env)

DEFAULT_CONFIG = PKG02_ROOT / "config" / "pre_imu_forward_hemisphere_workspace.json"
DEFAULT_STAGE = "P2_TWO_LINK_RIGID_TARGET_VIVE_LABEL"
DEFAULT_OUTPUT = PKG08_ROOT / "outputs" / "pre_imu_front_120_targets.csv"
DEFAULT_METADATA_OUTPUT = PKG08_ROOT / "outputs" / "pre_imu_front_120_targets_metadata.json"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    defaults = [
        "--config", str(DEFAULT_CONFIG),
        "--stage", DEFAULT_STAGE,
        "--output", str(DEFAULT_OUTPUT),
        "--metadata-output", str(DEFAULT_METADATA_OUTPUT),
    ]
    # `--mode front_120` is appended last so it always wins over any
    # user-supplied `--mode` (argparse: last occurrence of a repeated
    # optional flag wins) - this wrapper only ever produces front_120 plans.
    forced_mode = ["--mode", "front_120"]
    sys.argv = ["generate_pre_imu_targets.py", *defaults, *argv, *forced_mode]
    return pre_imu_cli.main()


if __name__ == "__main__":
    raise SystemExit(main())

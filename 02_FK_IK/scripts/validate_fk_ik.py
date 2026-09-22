"""Validate FK/IK consistency on sampled postures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fkik.flex_fk import fk_flex
from fkik.flex_ik import ik_flex
from fkik.geometry import load_config
from fkik.rigid_fk import fk_rigid
from fkik.rigid_ik import ik_rigid
from workspace_points.candidate_generator import sample_joint_space


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate rigid/flex FK-IK round trips.")
    parser.add_argument("--config", default="config/robot_config.yaml")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--output", default="outputs/round_trip_error_report.csv")
    args = parser.parse_args()

    config = load_config(_resolve(args.config))
    q_samples = sample_joint_space(config, args.n)
    rigid_errors: list[float] = []
    flex_errors: list[float] = []
    rigid_success = 0
    flex_success = 0

    for q in q_samples:
        p_rigid = fk_rigid(q, config)
        rigid = ik_rigid(p_rigid, config, q_init=q)
        if rigid.success and rigid.q is not None:
            rigid_success += 1
            rigid_errors.append(float(np.linalg.norm(fk_rigid(rigid.q, config) - p_rigid)))

        p_flex = fk_flex(q, config).p_flex
        flex = ik_flex(p_flex, config, q_init=q)
        if flex.success and flex.q is not None:
            flex_success += 1
            flex_errors.append(float(np.linalg.norm(fk_flex(flex.q, config).p_flex - p_flex)))

    def stats(name: str, errors: list[float], success_count: int) -> dict[str, float | str | int]:
        arr = np.asarray(errors, dtype=float)
        return {
            "solver": name,
            "sample_count": int(len(q_samples)),
            "success_count": int(success_count),
            "mean_mm": float(arr.mean()) if arr.size else np.nan,
            "std_mm": float(arr.std(ddof=0)) if arr.size else np.nan,
            "max_mm": float(arr.max()) if arr.size else np.nan,
            "p95_mm": float(np.quantile(arr, 0.95)) if arr.size else np.nan,
        }

    report = pd.DataFrame([stats("rigid", rigid_errors, rigid_success), stats("flex", flex_errors, flex_success)])
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(report.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

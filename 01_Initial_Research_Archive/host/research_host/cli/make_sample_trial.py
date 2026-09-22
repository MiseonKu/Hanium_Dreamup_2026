from __future__ import annotations

import argparse
import math
from pathlib import Path

from ..adapters.synthetic import synthetic_link_attitudes, synthetic_trial_metadata
from ..analysis import build_analysis_summary
from ..conditions import encoder_imu_fusion, imu_array_only
from ..geometry import get_robot_model
from ..logging import TrialLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a synthetic sample trial fixture.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/samples"),
        help="Output root containing raw/processed/analysis subfolders.",
    )
    parser.add_argument("--samples", type=int, default=40, help="Number of processed rows.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    metadata = synthetic_trial_metadata()
    logger = TrialLogger(args.output, metadata)
    model = get_robot_model(metadata.robot_model)
    link_lengths = model.rod_lengths_mm

    rows: list[dict[str, object]] = []
    for index in range(args.samples):
        t_s = index * 0.02
        bend_deg = 8.0 + (3.0 * math.sin(t_s * math.pi * 2.0))
        link_attitudes = synthetic_link_attitudes(bend_deg=bend_deg)
        imu_estimate, link_shapes = imu_array_only(link_attitudes, link_lengths)
        encoder_estimate = (
            imu_estimate.p_hat_base_mm[0] + 12.0,
            imu_estimate.p_hat_base_mm[1] - 4.0,
            imu_estimate.p_hat_base_mm[2],
        )
        fusion_estimate = encoder_imu_fusion(
            encoder_estimate,
            link_shapes,
            fusion_gain=0.5,
        )
        actual = (
            imu_estimate.p_hat_base_mm[0] + 2.0,
            imu_estimate.p_hat_base_mm[1] - 1.0,
            imu_estimate.p_hat_base_mm[2] + 0.5,
        )
        rows.append(
            {
                "loop_index": index,
                "t_common_s": f"{t_s:.6f}",
                "P_hat_encoder_base_mm_x": f"{encoder_estimate[0]:.6f}",
                "P_hat_encoder_base_mm_y": f"{encoder_estimate[1]:.6f}",
                "P_hat_encoder_base_mm_z": f"{encoder_estimate[2]:.6f}",
                "P_hat_imu_base_mm_x": f"{imu_estimate.p_hat_base_mm[0]:.6f}",
                "P_hat_imu_base_mm_y": f"{imu_estimate.p_hat_base_mm[1]:.6f}",
                "P_hat_imu_base_mm_z": f"{imu_estimate.p_hat_base_mm[2]:.6f}",
                "P_hat_fusion_base_mm_x": f"{fusion_estimate.p_hat_base_mm[0]:.6f}",
                "P_hat_fusion_base_mm_y": f"{fusion_estimate.p_hat_base_mm[1]:.6f}",
                "P_hat_fusion_base_mm_z": f"{fusion_estimate.p_hat_base_mm[2]:.6f}",
                "P_actual_base_mm_x": f"{actual[0]:.6f}",
                "P_actual_base_mm_y": f"{actual[1]:.6f}",
                "P_actual_base_mm_z": f"{actual[2]:.6f}",
                "q_imu_world_w": f"{link_attitudes[0][1][0]:.9f}",
                "q_imu_world_x": f"{link_attitudes[0][1][1]:.9f}",
                "q_imu_world_y": f"{link_attitudes[0][1][2]:.9f}",
                "q_imu_world_z": f"{link_attitudes[0][1][3]:.9f}",
                "q_encoder_rad_1": "0.0",
                "q_encoder_rad_2": "0.0",
                "q_encoder_rad_3": "0.0",
                "latency_encoder_s": "0.001",
                "latency_imu_s": "0.001",
                "latency_tracker_s": "0.002",
                "quality_flags": "ok",
            }
        )

    logger.write_metadata_json(
        {
            "fixture": True,
            "data_contract_version": "v0.1",
            "generated_by": "research_host.cli.make_sample_trial",
        }
    )
    logger.write_processed_rows(rows)
    summary = build_analysis_summary(
        rows,
        trial_id=metadata.trial_id,
        condition_id=metadata.condition_id,
        robot_model=metadata.robot_model,
        target_id=metadata.target_id,
    )
    logger.write_analysis_summary(summary)
    print(f"Wrote sample trial under {args.output}")
    print(f"Metadata: {logger.metadata_path()}")
    print(f"Processed: {logger.processed_csv_path()}")
    print(f"Analysis: {logger.analysis_csv_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

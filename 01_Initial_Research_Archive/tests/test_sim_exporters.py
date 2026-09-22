from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from research_host.sim.exporters import (
    build_sim_model,
    mjcf_xml,
    processed_row_to_joint_state,
    urdf_xml,
    write_sim_assets,
)


def test_sim_model_uses_meter_lengths_and_joint_order() -> None:
    spec = build_sim_model("long")

    assert spec.model_id == "so101_long"
    assert spec.unit_position == "m"
    assert spec.unit_angle == "rad"
    assert spec.joint_names == ("joint_1", "joint_2", "joint_3")
    assert spec.links[0].length_m == 0.2995


def test_urdf_and_mjcf_are_parseable_and_named() -> None:
    spec = build_sim_model("short")
    urdf_root = ET.fromstring(urdf_xml(spec))
    mjcf_root = ET.fromstring(mjcf_xml(spec))

    assert urdf_root.tag == "robot"
    assert urdf_root.attrib["name"] == "so101_short"
    assert len(urdf_root.findall("joint")) >= 3
    assert mjcf_root.tag == "mujoco"
    assert mjcf_root.attrib["model"] == "so101_short"


def test_processed_row_to_joint_state_maps_encoder_columns() -> None:
    state = processed_row_to_joint_state(
        {
            "q_encoder_rad_1": "0.1",
            "q_encoder_rad_2": "-0.2",
            "q_encoder_rad_3": "0.3",
        }
    )

    assert state == {"joint_1": 0.1, "joint_2": -0.2, "joint_3": 0.3}


def test_write_sim_assets_keeps_model_specific_manifests(tmp_path) -> None:
    long_paths = write_sim_assets(tmp_path, robot_model="long")
    short_paths = write_sim_assets(tmp_path, robot_model="short")

    assert long_paths["manifest"].name == "so101_long_manifest.json"
    assert short_paths["manifest"].name == "so101_short_manifest.json"
    assert long_paths["manifest"].exists()
    assert short_paths["manifest"].exists()

    long_manifest = json.loads(long_paths["manifest"].read_text(encoding="utf-8"))
    assert long_manifest["generated_files"]["urdf"].endswith("so101_long.urdf")
    assert (tmp_path / "urdf" / "so101_short.urdf").exists()

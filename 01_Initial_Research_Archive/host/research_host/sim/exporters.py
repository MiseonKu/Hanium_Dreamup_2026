from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.dom import minidom

from ..geometry import RobotModel, get_robot_model, mm_to_m


SIM_JOINT_NAMES = ("joint_1", "joint_2", "joint_3")
IMU_POSITIONS = ("start", "mid", "end")


@dataclass(frozen=True, slots=True)
class SimJointSpec:
    name: str
    parent_link: str
    child_link: str
    axis_xyz: tuple[float, float, float]
    origin_xyz_m: tuple[float, float, float]
    lower_rad: float
    upper_rad: float


@dataclass(frozen=True, slots=True)
class SimLinkSpec:
    name: str
    length_m: float
    radius_m: float
    mass_kg: float


@dataclass(frozen=True, slots=True)
class SimModelSpec:
    model_id: str
    robot_model: str
    unit_position: str
    unit_angle: str
    base_frame: str
    tool_frame: str
    links: tuple[SimLinkSpec, ...]
    joints: tuple[SimJointSpec, ...]

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.joints)


def build_sim_model(
    robot_model_id: str | RobotModel,
    *,
    link_radius_m: float = 0.012,
    link_mass_kg: float = 0.08,
    joint_limit_rad: float = math.pi,
) -> SimModelSpec:
    robot = (
        robot_model_id
        if isinstance(robot_model_id, RobotModel)
        else get_robot_model(robot_model_id)
    )
    lengths_m = tuple(mm_to_m(length) for length in robot.effective_fk_lengths_mm)
    links = tuple(
        SimLinkSpec(
            name=f"link_{index + 1}",
            length_m=length_m,
            radius_m=link_radius_m,
            mass_kg=link_mass_kg,
        )
        for index, length_m in enumerate(lengths_m)
    )
    joints = (
        SimJointSpec(
            name="joint_1",
            parent_link="base_link",
            child_link="link_1",
            axis_xyz=(0.0, 0.0, 1.0),
            origin_xyz_m=(0.0, 0.0, 0.0),
            lower_rad=-joint_limit_rad,
            upper_rad=joint_limit_rad,
        ),
        SimJointSpec(
            name="joint_2",
            parent_link="link_1",
            child_link="link_2",
            axis_xyz=(0.0, 0.0, 1.0),
            origin_xyz_m=(lengths_m[0], 0.0, 0.0),
            lower_rad=-joint_limit_rad,
            upper_rad=joint_limit_rad,
        ),
        SimJointSpec(
            name="joint_3",
            parent_link="link_2",
            child_link="link_3",
            axis_xyz=(0.0, 0.0, 1.0),
            origin_xyz_m=(lengths_m[1], 0.0, 0.0),
            lower_rad=-joint_limit_rad,
            upper_rad=joint_limit_rad,
        ),
    )
    return SimModelSpec(
        model_id=f"so101_{robot.model_id}",
        robot_model=robot.model_id,
        unit_position="m",
        unit_angle="rad",
        base_frame="base_link",
        tool_frame="tool0",
        links=links,
        joints=joints,
    )


def processed_row_to_joint_state(row: Mapping[str, object]) -> dict[str, float]:
    return {
        joint_name: _float(row, f"q_encoder_rad_{index}")
        for index, joint_name in enumerate(SIM_JOINT_NAMES, start=1)
    }


def urdf_xml(spec: SimModelSpec) -> str:
    robot = ET.Element("robot", {"name": spec.model_id})
    _add_urdf_link(robot, "base_link", length_m=0.04, radius_m=0.025, mass_kg=0.2)

    for link in spec.links:
        _add_urdf_link(
            robot,
            link.name,
            length_m=link.length_m,
            radius_m=link.radius_m,
            mass_kg=link.mass_kg,
        )

    for joint in spec.joints:
        joint_el = ET.SubElement(robot, "joint", {"name": joint.name, "type": "revolute"})
        ET.SubElement(joint_el, "parent", {"link": joint.parent_link})
        ET.SubElement(joint_el, "child", {"link": joint.child_link})
        ET.SubElement(joint_el, "origin", {"xyz": _xyz(joint.origin_xyz_m), "rpy": "0 0 0"})
        ET.SubElement(joint_el, "axis", {"xyz": _xyz(joint.axis_xyz)})
        ET.SubElement(
            joint_el,
            "limit",
            {
                "lower": f"{joint.lower_rad:.6f}",
                "upper": f"{joint.upper_rad:.6f}",
                "effort": "10.0",
                "velocity": "3.141593",
            },
        )

    last_link = spec.links[-1]
    tool_link = ET.SubElement(robot, "link", {"name": spec.tool_frame})
    _add_inertial(tool_link, mass_kg=0.001)
    tool_joint = ET.SubElement(robot, "joint", {"name": "tool0_fixed", "type": "fixed"})
    ET.SubElement(tool_joint, "parent", {"link": last_link.name})
    ET.SubElement(tool_joint, "child", {"link": spec.tool_frame})
    ET.SubElement(tool_joint, "origin", {"xyz": f"{last_link.length_m:.6f} 0 0", "rpy": "0 0 0"})

    for link in spec.links:
        for position_name, fraction in zip(IMU_POSITIONS, (0.0, 0.5, 1.0)):
            imu_link_name = f"imu_{link.name}_{position_name}"
            imu_link = ET.SubElement(robot, "link", {"name": imu_link_name})
            _add_inertial(imu_link, mass_kg=0.001)
            imu_joint = ET.SubElement(
                robot,
                "joint",
                {"name": f"{imu_link_name}_fixed", "type": "fixed"},
            )
            ET.SubElement(imu_joint, "parent", {"link": link.name})
            ET.SubElement(imu_joint, "child", {"link": imu_link_name})
            ET.SubElement(
                imu_joint,
                "origin",
                {"xyz": f"{link.length_m * fraction:.6f} 0 0", "rpy": "0 0 0"},
            )

    return _pretty_xml(robot)


def mjcf_xml(spec: SimModelSpec) -> str:
    mujoco = ET.Element("mujoco", {"model": spec.model_id})
    ET.SubElement(mujoco, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(mujoco, "option", {"timestep": "0.001"})
    asset = ET.SubElement(mujoco, "asset")
    ET.SubElement(asset, "material", {"name": "link_mat", "rgba": "0.2 0.4 0.8 1"})
    worldbody = ET.SubElement(mujoco, "worldbody")
    base = ET.SubElement(worldbody, "body", {"name": "base_link", "pos": "0 0 0"})
    ET.SubElement(base, "geom", {"type": "sphere", "size": "0.025", "mass": "0.2"})

    current_body = base
    for index, (link, joint) in enumerate(zip(spec.links, spec.joints), start=1):
        if index == 1:
            body_pos = "0 0 0"
        else:
            previous = spec.links[index - 2]
            body_pos = f"{previous.length_m:.6f} 0 0"
        body = ET.SubElement(current_body, "body", {"name": link.name, "pos": body_pos})
        ET.SubElement(
            body,
            "joint",
            {
                "name": joint.name,
                "type": "hinge",
                "axis": _xyz(joint.axis_xyz),
                "range": f"{joint.lower_rad:.6f} {joint.upper_rad:.6f}",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{link.name}_capsule",
                "type": "capsule",
                "fromto": f"0 0 0 {link.length_m:.6f} 0 0",
                "size": f"{link.radius_m:.6f}",
                "mass": f"{link.mass_kg:.6f}",
                "material": "link_mat",
            },
        )
        for position_name, fraction in zip(IMU_POSITIONS, (0.0, 0.5, 1.0)):
            ET.SubElement(
                body,
                "site",
                {
                    "name": f"imu_{link.name}_{position_name}",
                    "pos": f"{link.length_m * fraction:.6f} 0 0",
                    "size": "0.004",
                },
            )
        current_body = body

    ET.SubElement(
        current_body,
        "site",
        {"name": spec.tool_frame, "pos": f"{spec.links[-1].length_m:.6f} 0 0", "size": "0.006"},
    )
    return _pretty_xml(mujoco)


def rviz_config(spec: SimModelSpec) -> str:
    return f"""Panels:
  - Class: rviz_common/Displays
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/TF
      Enabled: true
      Name: TF
    - Class: rviz_default_plugins/RobotModel
      Description Source: Topic
      Description Topic:
        Value: /robot_description
      Enabled: true
      Name: RobotModel
  Enabled: true
  Global Options:
    Fixed Frame: {spec.base_frame}
  Name: root
Window Geometry:
  Height: 900
  Width: 1400
"""


def ros2_display_launch(spec: SimModelSpec) -> str:
    return f'''from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    urdf_path = LaunchConfiguration("urdf_path")
    rviz_config = LaunchConfiguration("rviz_config")
    return LaunchDescription([
        DeclareLaunchArgument(
            "urdf_path",
            default_value=PathJoinSubstitution([
                FindPackageShare("so101_sim_description"),
                "urdf",
                "{spec.model_id}.urdf",
            ]),
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=PathJoinSubstitution([
                FindPackageShare("so101_sim_description"),
                "rviz",
                "{spec.model_id}.rviz",
            ]),
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{{"robot_description": Command(["xacro ", urdf_path])}}],
        ),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", rviz_config]),
    ])
'''


def isaac_import_manifest(spec: SimModelSpec) -> dict[str, object]:
    return {
        "model_id": spec.model_id,
        "source_format": "URDF",
        "recommended_importer": "Isaac Sim URDF Importer",
        "recommended_extensions": [
            "isaacsim.asset.importer.urdf",
            "isaacsim.ros2.bridge",
        ],
        "urdf_path": f"../urdf/{spec.model_id}.urdf",
        "base_frame": spec.base_frame,
        "tool_frame": spec.tool_frame,
        "joint_names": list(spec.joint_names),
        "units": {"position": spec.unit_position, "angle": spec.unit_angle},
        "notes": [
            "Treat this asset as a research visualization baseline.",
            "Replace placeholder mass/inertia before physics validation.",
            "Keep ROS 2 joint order aligned with q_encoder_rad_1..3.",
        ],
    }


def write_sim_assets(
    output_dir: str | Path,
    *,
    robot_model: str = "long",
    formats: Sequence[str] = ("urdf", "mjcf", "rviz", "ros2_launch", "isaac_manifest"),
) -> dict[str, Path]:
    output_root = Path(output_dir)
    spec = build_sim_model(robot_model)
    written: dict[str, Path] = {}
    format_set = set(formats)

    if "urdf" in format_set:
        path = output_root / "urdf" / f"{spec.model_id}.urdf"
        _write_text(path, urdf_xml(spec))
        written["urdf"] = path
    if "mjcf" in format_set:
        path = output_root / "mjcf" / f"{spec.model_id}.xml"
        _write_text(path, mjcf_xml(spec))
        written["mjcf"] = path
    if "rviz" in format_set:
        path = output_root / "rviz" / f"{spec.model_id}.rviz"
        _write_text(path, rviz_config(spec))
        written["rviz"] = path
    if "ros2_launch" in format_set:
        path = output_root / "ros2" / f"{spec.model_id}_display.launch.py"
        _write_text(path, ros2_display_launch(spec))
        written["ros2_launch"] = path
    if "isaac_manifest" in format_set:
        path = output_root / "isaac" / f"{spec.model_id}_import.json"
        _write_text(
            path,
            json.dumps(isaac_import_manifest(spec), indent=2, sort_keys=True) + "\n",
        )
        written["isaac_manifest"] = path

    manifest_path = output_root / f"{spec.model_id}_manifest.json"
    manifest = {
        "model_id": spec.model_id,
        "robot_model": spec.robot_model,
        "generated_files": {key: str(path.as_posix()) for key, path in written.items()},
        "spec": asdict(spec),
    }
    _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    written["manifest"] = manifest_path
    return written


def _add_urdf_link(
    robot: ET.Element,
    name: str,
    *,
    length_m: float,
    radius_m: float,
    mass_kg: float,
) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    _add_inertial(link, mass_kg=mass_kg)
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", {"xyz": f"{length_m * 0.5:.6f} 0 0", "rpy": "0 1.570796 0"})
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "cylinder", {"radius": f"{radius_m:.6f}", "length": f"{length_m:.6f}"})
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", {"xyz": f"{length_m * 0.5:.6f} 0 0", "rpy": "0 1.570796 0"})
    collision_geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(
        collision_geometry,
        "cylinder",
        {"radius": f"{radius_m:.6f}", "length": f"{length_m:.6f}"},
    )


def _add_inertial(link: ET.Element, *, mass_kg: float) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", {"value": f"{mass_kg:.6f}"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "0.000001",
            "ixy": "0.0",
            "ixz": "0.0",
            "iyy": "0.000001",
            "iyz": "0.0",
            "izz": "0.000001",
        },
    )


def _pretty_xml(element: ET.Element) -> str:
    rough = ET.tostring(element, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _xyz(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def _float(row: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    return float(value)

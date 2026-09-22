from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


PRE_IMU_TARGET_FIELDNAMES = [
    "target_id",
    "experiment_mode",
    "experiment_stage",
    "link_count",
    "tracker_mount",
    "material_preset",
    "q_cmd_rad_1",
    "q_cmd_rad_2",
    "q_cmd_rad_3",
    "q_offset_deg_1",
    "q_offset_deg_2",
    "q_offset_deg_3",
    "p_rigid_target_m_x",
    "p_rigid_target_m_y",
    "p_rigid_target_m_z",
    "p_rigid_target_mm_x",
    "p_rigid_target_mm_y",
    "p_rigid_target_mm_z",
    "workspace_region_type",
    "workspace_mode",
    "is_forward_hemisphere",
    "azimuth_deg",
    "radius_m",
    "stage_bin_primary",
    "stage_bin_secondary",
    "yaw_bin",
    "pitch_bin",
    "azimuth_bin",
    "height_bin",
    "radial_shell",
    "posture_bin",
    "bending_proxy_bin",
    "inside_workspace_region",
    "reject_reason",
]


PRE_IMU_SCHEDULE_FIELDNAMES = [
    "trial_order_index",
    "cycle_id",
    "target_id",
    "repeat_id",
    "experiment_stage",
    "link_count",
    "workspace_mode",
    "stage_bin_primary",
    "stage_bin_secondary",
    "material_preset",
    "q_cmd_rad_1",
    "q_cmd_rad_2",
    "q_cmd_rad_3",
    "p_rigid_target_mm_x",
    "p_rigid_target_mm_y",
    "p_rigid_target_mm_z",
    "randomization_seed",
]


@dataclass(frozen=True, slots=True)
class LinkPreset:
    material_preset: str
    link_1_length_m: float
    link_2_length_m: float
    link_1_cross_section_mm: tuple[float, float]
    link_2_cross_section_mm: tuple[float, float]
    material: str
    wall_count: int
    top_bottom_wall_count: int
    infill_percent: int
    infill_pattern: str
    weak_axis_installation: str


@dataclass(frozen=True, slots=True)
class WorkspaceRegion:
    region_type: str
    center_frame: str
    center_m: tuple[float, float, float]
    forward_axis: tuple[float, float, float]
    default_mode: str
    azimuth_limit_deg: dict[str, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class PreImuStageConfig:
    name: str
    link_count: int
    tracker_mount: str
    q_offset_deg: dict[str, tuple[float, float]]
    basic_q_deg: dict[str, float]
    default_target_count: int
    default_repeats: int
    basic_pose: str
    sign_convention_note: str


@dataclass(frozen=True, slots=True)
class PreImuWorkspaceConfig:
    experiment_mode: str
    imu_available: bool
    material_preset: str
    frame: dict[str, str]
    links: LinkPreset
    workspace: WorkspaceRegion
    stages: dict[str, PreImuStageConfig]
    defaults: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TargetPoint:
    target_id: str
    experiment_mode: str
    experiment_stage: str
    link_count: int
    tracker_mount: str
    material_preset: str
    q_cmd_rad_1: float
    q_cmd_rad_2: float
    q_cmd_rad_3: float
    q_offset_deg_1: float
    q_offset_deg_2: float
    q_offset_deg_3: float
    p_rigid_target_m_x: float
    p_rigid_target_m_y: float
    p_rigid_target_m_z: float
    p_rigid_target_mm_x: float
    p_rigid_target_mm_y: float
    p_rigid_target_mm_z: float
    workspace_region_type: str
    workspace_mode: str
    is_forward_hemisphere: bool
    azimuth_deg: float
    radius_m: float
    stage_bin_primary: str
    stage_bin_secondary: str
    yaw_bin: str
    pitch_bin: str
    azimuth_bin: str
    height_bin: str
    radial_shell: str
    posture_bin: str
    bending_proxy_bin: str
    inside_workspace_region: bool
    reject_reason: str


def load_pre_imu_workspace_config(path: str | Path) -> PreImuWorkspaceConfig:
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"))

    links = data["links"]
    print_preset = data["print_preset"]
    link_preset = LinkPreset(
        material_preset=str(data["material_preset"]),
        link_1_length_m=float(links["link_1"]["length_m"]),
        link_2_length_m=float(links["link_2"]["length_m"]),
        link_1_cross_section_mm=_pair(links["link_1"]["cross_section_mm"]),
        link_2_cross_section_mm=_pair(links["link_2"]["cross_section_mm"]),
        material=str(print_preset["material"]),
        wall_count=int(print_preset["wall_count"]),
        top_bottom_wall_count=int(print_preset["top_bottom_wall_count"]),
        infill_percent=int(print_preset["infill_percent"]),
        infill_pattern=str(print_preset["infill_pattern"]),
        weak_axis_installation=str(print_preset["weak_axis_installation"]),
    )

    workspace = data["workspace"]
    region = WorkspaceRegion(
        region_type=str(workspace["type"]),
        center_frame=str(workspace["center_frame"]),
        center_m=_triple(workspace["center_m"]),
        forward_axis=_normalized(_triple(workspace["forward_axis"])),
        default_mode=str(workspace["default_mode"]),
        azimuth_limit_deg={
            str(name): _pair(value)
            for name, value in workspace["azimuth_limit_deg"].items()
        },
    )

    stages = {
        name: PreImuStageConfig(
            name=name,
            link_count=int(stage["link_count"]),
            tracker_mount=str(stage["tracker_mount"]),
            q_offset_deg={
                str(joint): _pair(bounds)
                for joint, bounds in stage["q_offset_deg"].items()
            },
            basic_q_deg={
                "q1": float(stage.get("basic_q_deg", {}).get("q1", 0.0)),
                "q2": float(stage.get("basic_q_deg", {}).get("q2", 0.0)),
                "q3": float(stage.get("basic_q_deg", {}).get("q3", 0.0)),
            },
            default_target_count=int(stage["default_target_count"]),
            default_repeats=int(stage["default_repeats"]),
            basic_pose=str(stage.get("basic_pose", "")),
            sign_convention_note=str(stage.get("sign_convention_note", "")),
        )
        for name, stage in data["stages"].items()
    }

    return PreImuWorkspaceConfig(
        experiment_mode=str(data["experiment_mode"]),
        imu_available=bool(data["imu_available"]),
        material_preset=str(data["material_preset"]),
        frame={str(key): str(value) for key, value in data["frame"].items()},
        links=link_preset,
        workspace=region,
        stages=stages,
        defaults=dict(data.get("defaults", {})),
        raw=data,
    )


def fk_stage1_rigid(
    q1_rad: float,
    q2_rad: float,
    link1_length_m: float,
) -> tuple[float, float, float]:
    rho = link1_length_m * math.cos(q2_rad)
    z = link1_length_m * math.sin(q2_rad)
    x = rho * math.cos(q1_rad)
    y = rho * math.sin(q1_rad)
    return (x, y, z)


def fk_stage2_rigid(
    q1_rad: float,
    q2_rad: float,
    q3_rad: float,
    link1_length_m: float,
    link2_length_m: float,
) -> tuple[float, float, float]:
    rho = link1_length_m * math.cos(q2_rad) + link2_length_m * math.cos(q2_rad + q3_rad)
    z = link1_length_m * math.sin(q2_rad) + link2_length_m * math.sin(q2_rad + q3_rad)
    x = rho * math.cos(q1_rad)
    y = rho * math.sin(q1_rad)
    return (x, y, z)


def compute_azimuth_deg(
    point_m: Sequence[float],
    center_m: Sequence[float],
) -> float:
    vx = float(point_m[0]) - float(center_m[0])
    vy = float(point_m[1]) - float(center_m[1])
    return math.degrees(math.atan2(vy, vx))


def compute_radius_m(
    point_m: Sequence[float],
    center_m: Sequence[float],
) -> float:
    vx = float(point_m[0]) - float(center_m[0])
    vy = float(point_m[1]) - float(center_m[1])
    vz = float(point_m[2]) - float(center_m[2])
    return math.sqrt(vx * vx + vy * vy + vz * vz)


def is_inside_forward_trimmed_hemisphere(
    point_m: Sequence[float],
    config: PreImuWorkspaceConfig | WorkspaceRegion,
    mode: str = "pilot",
) -> bool:
    region = config.workspace if isinstance(config, PreImuWorkspaceConfig) else config
    _validate_mode(region, mode)
    v = tuple(float(point_m[i]) - region.center_m[i] for i in range(3))
    forward_dot = sum(v[i] * region.forward_axis[i] for i in range(3))
    if forward_dot <= 0.0:
        return False
    azimuth = compute_azimuth_deg(point_m, region.center_m)
    lo, hi = region.azimuth_limit_deg[mode]
    return lo <= azimuth <= hi


def classify_stage1_bins(
    *,
    q1_offset_deg: float,
    q2_offset_deg: float,
    point_m: Sequence[float],
    link1_length_m: float,
) -> dict[str, str]:
    yaw_bin = _signed_three_bin(q1_offset_deg, width_deg=10.0, positive_label="left", negative_label="right")
    if q2_offset_deg <= -20.0:
        pitch_bin = "forward_tilt"
    elif q2_offset_deg <= -8.0:
        pitch_bin = "mid"
    else:
        pitch_bin = "near_vertical"

    z = float(point_m[2])
    if z < 0.78 * link1_length_m:
        height_bin = "low"
    elif z < 0.92 * link1_length_m:
        height_bin = "mid"
    else:
        height_bin = "high"

    q2_rad = math.radians(90.0 + q2_offset_deg)
    bending_proxy = abs(math.cos(q2_rad))
    bending_proxy_bin = _unit_three_bin(bending_proxy)
    return {
        "stage_bin_primary": yaw_bin,
        "stage_bin_secondary": pitch_bin,
        "yaw_bin": yaw_bin,
        "pitch_bin": pitch_bin,
        "azimuth_bin": "",
        "height_bin": height_bin,
        "radial_shell": "",
        "posture_bin": "",
        "bending_proxy_bin": bending_proxy_bin,
    }


def classify_stage2_bins(
    *,
    q1_offset_deg: float,
    q2_rad: float,
    q3_rad: float,
    point_m: Sequence[float],
    link1_length_m: float,
    link2_length_m: float,
    center_m: Sequence[float],
    mode_limit_deg: tuple[float, float],
) -> dict[str, str]:
    radius = compute_radius_m(point_m, center_m)
    reach = link1_length_m + link2_length_m
    radius_ratio = radius / reach if reach > 0.0 else 0.0
    if radius_ratio < 0.68:
        radial_shell = "near"
    elif radius_ratio < 0.84:
        radial_shell = "mid"
    else:
        radial_shell = "far"

    azimuth = compute_azimuth_deg(point_m, center_m)
    azimuth_bin = _azimuth_three_bin(azimuth, mode_limit_deg)

    z = float(point_m[2])
    if z < 0.16:
        height_bin = "low"
    elif z < 0.29:
        height_bin = "mid"
    else:
        height_bin = "high"

    if radius_ratio < 0.68:
        posture_bin = "folded"
    elif radius_ratio < 0.84:
        posture_bin = "neutral"
    else:
        posture_bin = "extended"

    bending_proxy = (
        0.5 * link1_length_m * abs(math.cos(q2_rad))
        + 0.5 * link2_length_m * abs(math.cos(q2_rad + q3_rad))
    ) / max(0.5 * (link1_length_m + link2_length_m), 1e-12)
    bending_proxy_bin = _unit_three_bin(bending_proxy)
    yaw_bin = _signed_three_bin(q1_offset_deg, width_deg=10.0, positive_label="left", negative_label="right")
    return {
        "stage_bin_primary": radial_shell,
        "stage_bin_secondary": posture_bin,
        "yaw_bin": yaw_bin,
        "pitch_bin": "",
        "azimuth_bin": azimuth_bin,
        "height_bin": height_bin,
        "radial_shell": radial_shell,
        "posture_bin": posture_bin,
        "bending_proxy_bin": bending_proxy_bin,
    }


def generate_pre_imu_candidate_targets(
    config: PreImuWorkspaceConfig,
    stage_name: str,
    *,
    mode: str = "pilot",
    count: int | None = None,
    seed: int | None = None,
    candidate_count: int | None = None,
) -> tuple[list[TargetPoint], dict[str, int]]:
    stage = _stage(config, stage_name)
    _validate_mode(config.workspace, mode)
    requested_count = count if count is not None else stage.default_target_count
    rng = random.Random(config.defaults.get("seed", 101) if seed is None else seed)
    multiplier = int(config.defaults.get("candidate_multiplier", 40))
    minimum = int(config.defaults.get("minimum_candidate_count", 300))
    total_candidates = candidate_count or max(minimum, requested_count * multiplier)
    candidates: list[TargetPoint] = []
    rejected_count = 0
    mode_limit = config.workspace.azimuth_limit_deg[mode]

    for index in range(total_candidates):
        offsets = _sample_offsets(stage, rng)
        q_deg = {
            joint: stage.basic_q_deg.get(joint, 0.0) + offsets.get(joint, 0.0)
            for joint in ("q1", "q2", "q3")
        }
        q_rad = {joint: math.radians(value) for joint, value in q_deg.items()}

        if stage.link_count == 1:
            point_m = fk_stage1_rigid(
                q_rad["q1"],
                q_rad["q2"],
                config.links.link_1_length_m,
            )
        elif stage.link_count == 2:
            point_m = fk_stage2_rigid(
                q_rad["q1"],
                q_rad["q2"],
                q_rad["q3"],
                config.links.link_1_length_m,
                config.links.link_2_length_m,
            )
        else:
            raise ValueError(f"unsupported link_count for {stage.name}: {stage.link_count}")

        inside = is_inside_forward_trimmed_hemisphere(point_m, config, mode=mode)
        if not inside:
            rejected_count += 1
            continue

        if stage.link_count == 1:
            bins = classify_stage1_bins(
                q1_offset_deg=offsets.get("q1", 0.0),
                q2_offset_deg=offsets.get("q2", 0.0),
                point_m=point_m,
                link1_length_m=config.links.link_1_length_m,
            )
        else:
            bins = classify_stage2_bins(
                q1_offset_deg=offsets.get("q1", 0.0),
                q2_rad=q_rad["q2"],
                q3_rad=q_rad["q3"],
                point_m=point_m,
                link1_length_m=config.links.link_1_length_m,
                link2_length_m=config.links.link_2_length_m,
                center_m=config.workspace.center_m,
                mode_limit_deg=mode_limit,
            )

        azimuth = compute_azimuth_deg(point_m, config.workspace.center_m)
        radius = compute_radius_m(point_m, config.workspace.center_m)
        candidates.append(
            TargetPoint(
                target_id=f"candidate_{index:05d}",
                experiment_mode=config.experiment_mode,
                experiment_stage=stage.name,
                link_count=stage.link_count,
                tracker_mount=stage.tracker_mount,
                material_preset=config.material_preset,
                q_cmd_rad_1=q_rad["q1"],
                q_cmd_rad_2=q_rad["q2"],
                q_cmd_rad_3=q_rad["q3"] if stage.link_count == 2 else 0.0,
                q_offset_deg_1=offsets.get("q1", 0.0),
                q_offset_deg_2=offsets.get("q2", 0.0),
                q_offset_deg_3=offsets.get("q3", 0.0) if stage.link_count == 2 else 0.0,
                p_rigid_target_m_x=point_m[0],
                p_rigid_target_m_y=point_m[1],
                p_rigid_target_m_z=point_m[2],
                p_rigid_target_mm_x=point_m[0] * 1000.0,
                p_rigid_target_mm_y=point_m[1] * 1000.0,
                p_rigid_target_mm_z=point_m[2] * 1000.0,
                workspace_region_type=config.workspace.region_type,
                workspace_mode=mode,
                is_forward_hemisphere=True,
                azimuth_deg=azimuth,
                radius_m=radius,
                inside_workspace_region=True,
                reject_reason="",
                **bins,
            )
        )

    return candidates, {
        "candidate_count": total_candidates,
        "accepted_count": len(candidates),
        "rejected_count": rejected_count,
    }


def select_bin_balanced_targets(
    candidates: Sequence[TargetPoint],
    count: int,
    *,
    seed: int,
    target_id_prefix: str = "T",
) -> list[TargetPoint]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if not candidates:
        return []

    rng = random.Random(seed)
    grouped: dict[str, list[TargetPoint]] = defaultdict(list)
    for candidate in candidates:
        grouped[_selection_bin_key(candidate)].append(candidate)

    bin_keys = sorted(grouped)
    rng.shuffle(bin_keys)
    for key in bin_keys:
        rng.shuffle(grouped[key])

    selected: list[TargetPoint] = []
    while len(selected) < count:
        progressed = False
        for key in bin_keys:
            if grouped[key]:
                selected.append(grouped[key].pop())
                progressed = True
                if len(selected) >= count:
                    break
        if not progressed:
            break

    return [
        replace(candidate, target_id=f"{target_id_prefix}_T{index:03d}")
        for index, candidate in enumerate(selected, start=1)
    ]


def write_pre_imu_targets_csv(
    path: str | Path,
    targets: Sequence[TargetPoint],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRE_IMU_TARGET_FIELDNAMES)
        writer.writeheader()
        for row in target_rows(targets):
            writer.writerow(row)
    return output_path


def write_pre_imu_targets_metadata(
    path: str | Path,
    *,
    config_path: str | Path,
    config: PreImuWorkspaceConfig,
    stage_name: str,
    mode: str,
    seed: int,
    targets: Sequence[TargetPoint],
    candidate_count: int,
    rejected_count: int,
    validation_issues: Sequence[str] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "source": "workspace_points.pre_imu_targets",
        "config_path": str(config_path),
        "stage": stage_name,
        "mode": mode,
        "seed": seed,
        "target_count": len(targets),
        "candidate_count": candidate_count,
        "rejected_count": rejected_count,
        "workspace_region_type": config.workspace.region_type,
        "material_preset": config.material_preset,
        "link_spec": {
            "link_1_length_m": config.links.link_1_length_m,
            "link_2_length_m": config.links.link_2_length_m,
            "link_1_cross_section_mm": list(config.links.link_1_cross_section_mm),
            "link_2_cross_section_mm": list(config.links.link_2_cross_section_mm),
            "wall_count": config.links.wall_count,
            "top_bottom_wall_count": config.links.top_bottom_wall_count,
            "infill_percent": config.links.infill_percent,
            "infill_pattern": config.links.infill_pattern,
            "weak_axis_installation": config.links.weak_axis_installation,
        },
        "plan_target_digest_sha256": target_digest(targets),
        "validation_issues": list(validation_issues or []),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def read_pre_imu_targets_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def generate_pre_imu_trial_schedule_rows(
    target_rows_input: Sequence[Mapping[str, str]],
    *,
    repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    rng = random.Random(seed)
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in target_rows_input:
        key = "|".join(
            [
                row.get("stage_bin_primary", ""),
                row.get("stage_bin_secondary", ""),
                row.get("height_bin", ""),
                row.get("bending_proxy_bin", ""),
            ]
        )
        grouped[key].append(row)

    rows: list[dict[str, object]] = []
    order_index = 0
    for cycle in range(1, repeats + 1):
        keys = sorted(grouped)
        rng.shuffle(keys)
        cycle_groups = {key: list(grouped[key]) for key in keys}
        for key in keys:
            rng.shuffle(cycle_groups[key])
        cycle_order: list[Mapping[str, str]] = []
        while True:
            progressed = False
            for key in keys:
                if cycle_groups[key]:
                    cycle_order.append(cycle_groups[key].pop())
                    progressed = True
            if not progressed:
                break
        for target in cycle_order:
            rows.append(
                {
                    "trial_order_index": order_index,
                    "cycle_id": cycle,
                    "target_id": target["target_id"],
                    "repeat_id": cycle,
                    "experiment_stage": target["experiment_stage"],
                    "link_count": target["link_count"],
                    "workspace_mode": target["workspace_mode"],
                    "stage_bin_primary": target["stage_bin_primary"],
                    "stage_bin_secondary": target["stage_bin_secondary"],
                    "material_preset": target["material_preset"],
                    "q_cmd_rad_1": target["q_cmd_rad_1"],
                    "q_cmd_rad_2": target["q_cmd_rad_2"],
                    "q_cmd_rad_3": target["q_cmd_rad_3"],
                    "p_rigid_target_mm_x": target["p_rigid_target_mm_x"],
                    "p_rigid_target_mm_y": target["p_rigid_target_mm_y"],
                    "p_rigid_target_mm_z": target["p_rigid_target_mm_z"],
                    "randomization_seed": seed,
                }
            )
            order_index += 1
    return rows


def write_pre_imu_trial_schedule_csv(
    path: str | Path,
    schedule_rows: Sequence[Mapping[str, object]],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRE_IMU_SCHEDULE_FIELDNAMES)
        writer.writeheader()
        for row in schedule_rows:
            writer.writerow({field: row.get(field, "") for field in PRE_IMU_SCHEDULE_FIELDNAMES})
    return output_path


def write_pre_imu_trial_schedule_metadata(
    path: str | Path,
    *,
    target_csv_path: str | Path,
    repeats: int,
    seed: int,
    schedule_rows: Sequence[Mapping[str, object]],
) -> Path:
    payload = {
        "source": "workspace_points.pre_imu_targets.generate_pre_imu_trial_schedule_rows",
        "target_csv_path": str(target_csv_path),
        "repeats": repeats,
        "seed": seed,
        "trial_count": len(schedule_rows),
        "plan_digest_sha256": _rows_digest(schedule_rows),
        "validation_issues": [],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def target_rows(targets: Iterable[TargetPoint]) -> list[dict[str, object]]:
    return [
        {field: _csv_value(asdict(target)[field]) for field in PRE_IMU_TARGET_FIELDNAMES}
        for target in targets
    ]


def target_digest(targets: Sequence[TargetPoint]) -> str:
    return _rows_digest(target_rows(targets))


def _rows_digest(rows: Iterable[Mapping[str, object]]) -> str:
    payload = json.dumps(list(rows), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sample_offsets(stage: PreImuStageConfig, rng: random.Random) -> dict[str, float]:
    offsets = {"q1": 0.0, "q2": 0.0, "q3": 0.0}
    for joint, bounds in stage.q_offset_deg.items():
        offsets[joint] = rng.uniform(bounds[0], bounds[1])
    return offsets


def _stage(config: PreImuWorkspaceConfig, stage_name: str) -> PreImuStageConfig:
    try:
        return config.stages[stage_name]
    except KeyError as exc:
        valid = ", ".join(sorted(config.stages))
        raise ValueError(f"unknown stage {stage_name!r}; valid stages: {valid}") from exc


def _selection_bin_key(target: TargetPoint) -> str:
    if target.link_count == 1:
        values = [
            target.yaw_bin,
            target.pitch_bin,
            target.height_bin,
            target.bending_proxy_bin,
        ]
    else:
        values = [
            target.radial_shell,
            target.azimuth_bin,
            target.height_bin,
            target.posture_bin,
            target.bending_proxy_bin,
        ]
    return "|".join(values)


def _validate_mode(region: WorkspaceRegion, mode: str) -> None:
    if mode not in region.azimuth_limit_deg:
        valid = ", ".join(sorted(region.azimuth_limit_deg))
        raise ValueError(f"unknown workspace mode {mode!r}; valid modes: {valid}")


def _azimuth_three_bin(azimuth_deg: float, mode_limit_deg: tuple[float, float]) -> str:
    lo, hi = mode_limit_deg
    span = hi - lo
    first = lo + span / 3.0
    second = lo + 2.0 * span / 3.0
    if azimuth_deg < first:
        return "right"
    if azimuth_deg > second:
        return "left"
    return "center"


def _signed_three_bin(
    value_deg: float,
    *,
    width_deg: float,
    positive_label: str,
    negative_label: str,
) -> str:
    if value_deg > width_deg:
        return positive_label
    if value_deg < -width_deg:
        return negative_label
    return "center"


def _unit_three_bin(value: float) -> str:
    clamped = max(0.0, min(1.0, value))
    if clamped < 1.0 / 3.0:
        return "low"
    if clamped < 2.0 / 3.0:
        return "mid"
    return "high"


def _pair(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"expected pair, got {values!r}")
    return (float(values[0]), float(values[1]))


def _triple(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"expected triple, got {values!r}")
    return (float(values[0]), float(values[1]), float(values[2]))


def _normalized(values: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = _triple(values)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 0.0:
        raise ValueError("forward_axis must be nonzero")
    return (x / norm, y / norm, z / norm)


def _csv_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.12g}"
        if "e" not in text.lower() and "." not in text:
            return text + ".0"
        return text
    return value

from __future__ import annotations

from dataclasses import dataclass


MM_PER_M = 1000.0


def mm_to_m(value_mm: float) -> float:
    return float(value_mm) / MM_PER_M


def m_to_mm(value_m: float) -> float:
    return float(value_m) * MM_PER_M


@dataclass(frozen=True, slots=True)
class LinkGeometry:
    link_id: str
    rod_length_mm: float
    effective_fk_length_mm: float
    imu_start_s_mm: float
    imu_mid_s_mm: float
    imu_end_s_mm: float


@dataclass(frozen=True, slots=True)
class RobotModel:
    model_id: str
    description: str
    a_rod_mm: float
    b_rod_mm: float
    c_rod_mm: float
    l_a_effective_mm: float
    l_b_effective_mm: float
    l_c_effective_mm: float

    @property
    def rod_lengths_mm(self) -> tuple[float, float, float]:
        return (self.a_rod_mm, self.b_rod_mm, self.c_rod_mm)

    @property
    def effective_fk_lengths_mm(self) -> tuple[float, float, float]:
        return (
            self.l_a_effective_mm,
            self.l_b_effective_mm,
            self.l_c_effective_mm,
        )

    def link_geometries(self) -> tuple[LinkGeometry, LinkGeometry, LinkGeometry]:
        return tuple(
            LinkGeometry(
                link_id=f"link_{index + 1}",
                rod_length_mm=rod_length,
                effective_fk_length_mm=effective_length,
                imu_start_s_mm=0.0,
                imu_mid_s_mm=rod_length * 0.5,
                imu_end_s_mm=rod_length,
            )
            for index, (rod_length, effective_length) in enumerate(
                zip(self.rod_lengths_mm, self.effective_fk_lengths_mm)
            )
        )  # type: ignore[return-value]


ROBOT_MODELS: dict[str, RobotModel] = {
    "long": RobotModel(
        model_id="long",
        description="Long-stroke model, larger reach and higher expected deflection",
        a_rod_mm=233.0,
        b_rod_mm=215.0,
        c_rod_mm=106.0,
        l_a_effective_mm=299.5,
        l_b_effective_mm=299.5,
        l_c_effective_mm=150.3,
    ),
    "short": RobotModel(
        model_id="short",
        description="Short-stroke model, lower expected deflection comparison condition",
        a_rod_mm=183.0,
        b_rod_mm=165.0,
        c_rod_mm=56.0,
        l_a_effective_mm=249.5,
        l_b_effective_mm=249.5,
        l_c_effective_mm=100.3,
    ),
}


def get_robot_model(model_id: str) -> RobotModel:
    key = model_id.strip().lower()
    if key not in ROBOT_MODELS:
        known = ", ".join(sorted(ROBOT_MODELS))
        raise KeyError(f"Unknown robot model {model_id!r}. Known models: {known}")
    return ROBOT_MODELS[key]

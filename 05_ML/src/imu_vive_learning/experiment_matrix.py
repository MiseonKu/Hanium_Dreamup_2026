"""Pre-registered endpoint-regression experiment matrix and feature allow-lists."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .features import PLANNED_COLUMNS


PRIMARY_ALIASES = {"encoder", "four_imu", "fusion"}
DIAGNOSTIC_ALIASES = {"planned_only", "fusion_planned", "fusion_mag"}


@dataclass(frozen=True, order=True)
class ExperimentCell:
    model: str
    feature_view: str
    recipe: str
    purposes: tuple[str, ...]
    diagnostic_only: bool
    deployment_candidate: bool

    @property
    def cell_id(self) -> str:
        return f"{self.model}__{self.feature_view}__{self.recipe}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "model": self.model,
            "feature_view": self.feature_view,
            "recipe": self.recipe,
            "purposes": list(self.purposes),
            "diagnostic_only": self.diagnostic_only,
            "deployment_candidate": self.deployment_candidate,
        }


def build_experiment_cells(config: dict[str, Any]) -> list[ExperimentCell]:
    """Expand and de-duplicate the frozen factorial matrix."""

    matrix = config["experiment_matrix"]
    memberships: dict[tuple[str, str, str], set[str]] = {}
    for purpose in (
        "baseline",
        "sensor_comparison",
        "augmentation_comparison",
        "shortcut_diagnostics",
    ):
        section = matrix[purpose]
        for model in section["models"]:
            for view in section["feature_views"]:
                for recipe in section["recipes"]:
                    memberships.setdefault((str(model), str(view), str(recipe)), set()).add(
                        purpose
                    )

    excluded_models = set(matrix["deployment_exclusions"].get("models", []))
    excluded_recipes = set(matrix["deployment_exclusions"].get("recipes", []))
    excluded_views = set(matrix["deployment_exclusions"].get("feature_views", []))
    cells: list[ExperimentCell] = []
    for (model, view, recipe), purposes in memberships.items():
        diagnostic = "shortcut_diagnostics" in purposes
        deployment = (
            not diagnostic
            and model not in excluded_models
            and recipe not in excluded_recipes
            and view not in excluded_views
        )
        cells.append(
            ExperimentCell(
                model=model,
                feature_view=view,
                recipe=recipe,
                purposes=tuple(sorted(purposes)),
                diagnostic_only=diagnostic,
                deployment_candidate=deployment,
            )
        )
    cells.sort(key=lambda item: item.cell_id)
    if len(cells) != 31:
        raise ValueError(f"experiment matrix must expand to 31 unique cells, got {len(cells)}")
    return cells


def resolve_feature_columns(
    alias: str, *, config: dict[str, Any], release_feature_sets: dict[str, list[str]]
) -> list[str]:
    """Resolve a public matrix alias to an ordered, audited input allow-list."""

    if alias == "none":
        return []
    primary = config["features"]["primary_views"]
    diagnostics = config["features"]["diagnostic_views"]
    if alias in primary:
        release_name = primary[alias]
        columns = list(release_feature_sets[release_name])
        forbidden = tuple(
            str(token).lower() for token in config["features"]["forbidden_primary_tokens"]
        )
        leaked = [
            name
            for name in columns
            if any(token in name.lower() for token in forbidden)
        ]
        if leaked:
            raise ValueError(f"primary feature alias {alias!r} leaks context: {leaked}")
        return columns
    if alias == "planned_only":
        return list(PLANNED_COLUMNS)
    if alias in {"fusion_planned", "fusion_mag"}:
        release_name = diagnostics[alias]
        return list(release_feature_sets[release_name])
    raise KeyError(f"unknown feature-view alias: {alias!r}")


def validate_cell_feature_policy(cell: ExperimentCell, columns: list[str]) -> None:
    if cell.model == "dummy_mean":
        if columns:
            raise ValueError("dummy_mean must not consume input features")
        return
    if not columns:
        raise ValueError(f"non-dummy cell {cell.cell_id} has no input features")
    if cell.feature_view in PRIMARY_ALIASES:
        prohibited = [
            name
            for name in columns
            if name.startswith(("vive_", "planned_"))
            or "tracker" in name.lower()
            or "_mag_" in name.lower()
        ]
        if prohibited:
            raise ValueError(f"primary cell {cell.cell_id} contains prohibited inputs: {prohibited}")
    elif cell.feature_view not in DIAGNOSTIC_ALIASES:
        raise ValueError(f"cell {cell.cell_id} uses an undeclared feature view")


__all__ = [
    "DIAGNOSTIC_ALIASES",
    "ExperimentCell",
    "PRIMARY_ALIASES",
    "build_experiment_cells",
    "resolve_feature_columns",
    "validate_cell_feature_policy",
]

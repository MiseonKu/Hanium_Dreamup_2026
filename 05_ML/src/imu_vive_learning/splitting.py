"""Deterministic, pre-augmentation split assignments."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd


class SplitContractError(ValueError):
    """Raised when samples cannot satisfy the versioned split contract."""


ASSIGNMENT_COLUMNS = (
    "scheme",
    "outer_fold",
    "inner_fold",
    "split_context",
    "trial_uid",
    "segment_id",
    "target_id",
    "target_group",
    "role",
)


def _base_eligible(canonical: pd.DataFrame) -> pd.DataFrame:
    required = {"trial_uid", "segment_id", "target_id", "train_eligible"}
    missing = sorted(required.difference(canonical.columns))
    if missing:
        raise SplitContractError(f"canonical frame is missing split fields: {missing}")
    eligible = canonical.loc[
        canonical["train_eligible"].eq(True), ["trial_uid", "segment_id", "target_id"]
    ].copy()
    if eligible.empty:
        raise SplitContractError("no train-eligible trials are available")
    if eligible["trial_uid"].isna().any() or eligible["trial_uid"].duplicated().any():
        raise SplitContractError("train-eligible trial_uid values must be non-null and unique")
    if eligible[["segment_id", "target_id"]].isna().any().any():
        raise SplitContractError("segment_id and target_id must be non-null")
    return eligible


def _target_numbers(target_ids: pd.Series) -> pd.Series:
    text = target_ids.astype("string")
    valid = text.str.fullmatch(r"T\d+").fillna(False)
    if not valid.all():
        invalid = sorted(text.loc[~valid].astype(str).unique().tolist())
        raise SplitContractError(f"target_id must match T<digits>; invalid values: {invalid}")
    return text.str.slice(1).astype("int64")


def _context_frame(
    base: pd.DataFrame,
    *,
    scheme: str,
    outer_fold: int,
    inner_fold: int,
    role: np.ndarray,
) -> pd.DataFrame:
    result = base.copy()
    result.insert(0, "scheme", scheme)
    result.insert(1, "outer_fold", int(outer_fold))
    result.insert(2, "inner_fold", int(inner_fold))
    inner_label = "refit" if inner_fold == -1 else str(inner_fold)
    result.insert(
        3,
        "split_context",
        f"{scheme}:outer={int(outer_fold)}:inner={inner_label}",
    )
    result["role"] = role
    return result.loc[:, ASSIGNMENT_COLUMNS]


def build_split_assignments(
    canonical: pd.DataFrame, split_config: Mapping[str, Any]
) -> pd.DataFrame:
    """Materialize outer refit and nested segment-validation assignments."""

    required_config = {
        "segment_outer_values",
        "target_group_count",
        "expected_targets_per_group",
        "expected_assignment_rows",
    }
    missing = sorted(required_config.difference(split_config))
    if missing:
        raise SplitContractError(f"split config is missing required fields: {missing}")

    base = _base_eligible(canonical)
    segment_values = [int(value) for value in split_config["segment_outer_values"]]
    if len(segment_values) < 3 or len(segment_values) != len(set(segment_values)):
        raise SplitContractError("segment_outer_values must contain at least three unique segments")
    actual_segments = set(base["segment_id"].astype(int).unique().tolist())
    if actual_segments != set(segment_values):
        raise SplitContractError(
            "eligible segment set changed: "
            f"expected {sorted(segment_values)}, received {sorted(actual_segments)}"
        )

    group_count = int(split_config["target_group_count"])
    if group_count < 2:
        raise SplitContractError("target_group_count must be at least two")
    base["segment_id"] = base["segment_id"].astype("int64")
    base["target_group"] = _target_numbers(base["target_id"]) % group_count
    base.sort_values("trial_uid", kind="mergesort", inplace=True)
    base.reset_index(drop=True, inplace=True)

    contexts: list[pd.DataFrame] = []
    segment_array = base["segment_id"].to_numpy()
    for outer in segment_values:
        outer_role = np.where(segment_array == outer, "test", "train")
        contexts.append(
            _context_frame(
                base,
                scheme="segment_loso",
                outer_fold=outer,
                inner_fold=-1,
                role=outer_role,
            )
        )
        for inner in segment_values:
            if inner == outer:
                continue
            inner_role = np.select(
                [segment_array == outer, segment_array == inner],
                ["test", "validation"],
                default="train",
            )
            contexts.append(
                _context_frame(
                    base,
                    scheme="segment_loso",
                    outer_fold=outer,
                    inner_fold=inner,
                    role=inner_role,
                )
            )

    target_groups = base["target_group"].to_numpy()
    present_groups = set(int(value) for value in np.unique(target_groups))
    expected_groups = set(range(group_count))
    if present_groups != expected_groups:
        raise SplitContractError(
            f"target groups changed: expected {sorted(expected_groups)}, received {sorted(present_groups)}"
        )
    target_counts = base.groupby("target_group", observed=True)["target_id"].nunique().to_dict()
    expected_targets = int(split_config["expected_targets_per_group"])
    if target_counts != {group: expected_targets for group in range(group_count)}:
        raise SplitContractError(
            f"target-group cardinality changed: expected {expected_targets} per group, "
            f"received {target_counts}"
        )
    for outer in range(group_count):
        outer_role = np.where(target_groups == outer, "test", "train")
        contexts.append(
            _context_frame(
                base,
                scheme="target_group",
                outer_fold=outer,
                inner_fold=-1,
                role=outer_role,
            )
        )

    assignments = pd.concat(contexts, ignore_index=True)
    assignments.sort_values(
        ["scheme", "outer_fold", "inner_fold", "trial_uid"],
        kind="mergesort",
        inplace=True,
    )
    assignments.reset_index(drop=True, inplace=True)
    expected_rows = int(split_config["expected_assignment_rows"])
    if len(assignments) != expected_rows:
        raise SplitContractError(
            f"split assignment row count changed: expected {expected_rows}, received {len(assignments)}"
        )
    validate_split_assignments(assignments, expected_trial_count=len(base))
    return assignments


def validate_split_assignments(
    assignments: pd.DataFrame, *, expected_trial_count: int | None = None
) -> None:
    """Check uniqueness and role invariants for every split context."""

    missing = sorted(set(ASSIGNMENT_COLUMNS).difference(assignments.columns))
    if missing:
        raise SplitContractError(f"split assignments are missing columns: {missing}")
    allowed_roles = {"train", "validation", "test"}
    invalid_roles = set(assignments["role"].unique()).difference(allowed_roles)
    if invalid_roles:
        raise SplitContractError(f"unknown split roles: {sorted(invalid_roles)}")

    for context, group in assignments.groupby("split_context", sort=False):
        if group["trial_uid"].duplicated().any():
            raise SplitContractError(f"duplicate trial_uid in split context {context!r}")
        if expected_trial_count is not None and len(group) != int(expected_trial_count):
            raise SplitContractError(
                f"split context {context!r} has {len(group)} rows; expected {expected_trial_count}"
            )
        roles = set(group["role"])
        if not {"train", "test"}.issubset(roles):
            raise SplitContractError(f"split context {context!r} lacks train or test rows")
        inner_fold = int(group["inner_fold"].iloc[0])
        if inner_fold == -1 and "validation" in roles:
            raise SplitContractError(f"outer-refit context {context!r} contains validation rows")
        if inner_fold != -1 and roles != {"train", "validation", "test"}:
            raise SplitContractError(f"nested context {context!r} lacks a required role")


def iter_outer_folds(
    assignments: pd.DataFrame, scheme: str
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield ``(outer_fold, refit_rows)`` for augmentation of outer train only."""

    outer = assignments.loc[
        assignments["scheme"].eq(scheme) & assignments["inner_fold"].eq(-1)
    ]
    if outer.empty:
        raise SplitContractError(f"no outer-refit assignments for scheme {scheme!r}")
    for outer_fold, rows in outer.groupby("outer_fold", sort=True):
        yield int(outer_fold), rows.sort_values("trial_uid", kind="mergesort").reset_index(
            drop=True
        )

"""Nested group-CV assignments for endpoint-regression experiment v1."""

from __future__ import annotations

from collections.abc import Iterator
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from .splitting import ASSIGNMENT_COLUMNS


EXPERIMENT_ASSIGNMENT_ROWS = 385_100
ELIGIBLE_REAL_ROWS = 7_702
TARGET_GROUP_COUNT = 5
SEGMENT_FOLDS = (36, 37, 38, 39, 40)


class ExperimentSplitError(ValueError):
    """Raised when nested evaluation groups can overlap or drift."""


def build_experiment_assignments(
    canonical: pd.DataFrame,
    preparation_assignments: pd.DataFrame,
    *,
    expected_rows: int | None = EXPERIMENT_ASSIGNMENT_ROWS,
    expected_trial_count: int | None = ELIGIBLE_REAL_ROWS,
) -> pd.DataFrame:
    """Add 20 target-group inner contexts to the immutable preparation splits.

    The preparation release already contains all segment outer/inner contexts
    and the five target outer-refit contexts.  For target outer group ``g``,
    each other target group ``h`` becomes validation once while ``g`` remains
    test and the remaining three groups are train.
    """

    base = _eligible_base(canonical)
    if expected_trial_count is not None and len(base) != int(expected_trial_count):
        raise ExperimentSplitError(
            f"eligible real row count changed: expected {expected_trial_count}, received {len(base)}"
        )
    parent = preparation_assignments.copy()
    _validate_parent_assignments(parent, base)

    nested: list[pd.DataFrame] = []
    target_groups = base["target_group"].to_numpy(dtype=np.int64)
    for outer in range(TARGET_GROUP_COUNT):
        for inner in range(TARGET_GROUP_COUNT):
            if inner == outer:
                continue
            roles = np.select(
                [target_groups == outer, target_groups == inner],
                ["test", "validation"],
                default="train",
            )
            context = base.copy()
            context.insert(0, "scheme", "target_group")
            context.insert(1, "outer_fold", outer)
            context.insert(2, "inner_fold", inner)
            context.insert(
                3,
                "split_context",
                f"target_group:outer={outer}:inner={inner}",
            )
            context["role"] = roles
            nested.append(context.loc[:, ASSIGNMENT_COLUMNS])

    result = pd.concat([parent, *nested], ignore_index=True)
    result.sort_values(
        ["scheme", "outer_fold", "inner_fold", "trial_uid"],
        kind="mergesort",
        inplace=True,
    )
    result.reset_index(drop=True, inplace=True)
    validate_experiment_assignments(
        result,
        expected_rows=expected_rows,
        expected_trial_count=len(base),
    )
    return result


def validate_experiment_assignments(
    assignments: pd.DataFrame,
    *,
    expected_rows: int | None = EXPERIMENT_ASSIGNMENT_ROWS,
    expected_trial_count: int = ELIGIBLE_REAL_ROWS,
) -> None:
    """Validate context membership, exhaustive coverage, and group separation."""

    missing = sorted(set(ASSIGNMENT_COLUMNS) - set(assignments.columns))
    if missing:
        raise ExperimentSplitError(f"experiment assignments are missing columns: {missing}")
    if expected_rows is not None and len(assignments) != int(expected_rows):
        raise ExperimentSplitError(
            f"experiment assignment rows changed: expected {expected_rows}, received {len(assignments)}"
        )
    if assignments[list(ASSIGNMENT_COLUMNS)].isna().any().any():
        raise ExperimentSplitError("experiment assignments contain null contract values")
    if set(assignments["scheme"].astype(str)) != {"segment_loso", "target_group"}:
        raise ExperimentSplitError("experiment split scheme set changed")
    if set(assignments["role"].astype(str)) - {"train", "validation", "test"}:
        raise ExperimentSplitError("experiment assignments contain an unknown role")

    context_keys = assignments[
        ["scheme", "outer_fold", "inner_fold", "split_context"]
    ].drop_duplicates()
    if len(context_keys) != 50:
        raise ExperimentSplitError(
            f"experiment must contain exactly 50 contexts; received {len(context_keys)}"
        )
    expected_contexts = _expected_context_keys()
    actual_contexts = {
        (str(row.scheme), int(row.outer_fold), int(row.inner_fold), str(row.split_context))
        for row in context_keys.itertuples(index=False)
    }
    if actual_contexts != expected_contexts:
        missing_contexts = sorted(expected_contexts - actual_contexts)
        unexpected = sorted(actual_contexts - expected_contexts)
        raise ExperimentSplitError(
            f"experiment context set changed; missing={missing_contexts}, unexpected={unexpected}"
        )

    global_ids: set[str] | None = None
    for key, context in assignments.groupby(
        ["scheme", "outer_fold", "inner_fold", "split_context"], sort=False
    ):
        scheme, outer_fold, inner_fold, context_name = key
        if len(context) != int(expected_trial_count):
            raise ExperimentSplitError(
                f"context {context_name!r} has {len(context)} rows; expected {expected_trial_count}"
            )
        ids = context["trial_uid"].astype(str)
        if ids.duplicated().any():
            raise ExperimentSplitError(f"context {context_name!r} contains duplicate trial_uid")
        current_ids = set(ids)
        if global_ids is None:
            global_ids = current_ids
        elif current_ids != global_ids:
            raise ExperimentSplitError(
                f"context {context_name!r} does not cover the same eligible real trials"
            )
        if context["split_context"].nunique() != 1:
            raise ExperimentSplitError("one split key maps to multiple context labels")
        roles = set(context["role"].astype(str))
        if int(inner_fold) == -1:
            if roles != {"train", "test"}:
                raise ExperimentSplitError(
                    f"outer-refit context {context_name!r} must contain train and test only"
                )
        elif roles != {"train", "validation", "test"}:
            raise ExperimentSplitError(
                f"inner context {context_name!r} lacks train/validation/test"
            )
        _validate_group_roles(
            context,
            scheme=str(scheme),
            outer_fold=int(outer_fold),
            inner_fold=int(inner_fold),
        )


def context_assignments(
    assignments: pd.DataFrame,
    *,
    scheme: str,
    outer_fold: int,
    inner_fold: int = -1,
) -> pd.DataFrame:
    """Select one validated context in stable trial order."""

    context = assignments.loc[
        assignments["scheme"].astype(str).eq(str(scheme))
        & assignments["outer_fold"].astype(int).eq(int(outer_fold))
        & assignments["inner_fold"].astype(int).eq(int(inner_fold))
    ].copy()
    if context.empty:
        raise ExperimentSplitError(
            f"unknown split context: {scheme}/outer={outer_fold}/inner={inner_fold}"
        )
    if context["split_context"].nunique() != 1 or context["trial_uid"].duplicated().any():
        raise ExperimentSplitError("selected split context is malformed")
    return context.sort_values("trial_uid", kind="mergesort").reset_index(drop=True)


def context_ids(
    assignments: pd.DataFrame,
    *,
    scheme: str,
    outer_fold: int,
    inner_fold: int = -1,
    role: str,
) -> tuple[str, ...]:
    context = context_assignments(
        assignments,
        scheme=scheme,
        outer_fold=outer_fold,
        inner_fold=inner_fold,
    )
    if role not in {"train", "validation", "test"}:
        raise ExperimentSplitError(f"unknown requested role: {role!r}")
    return tuple(context.loc[context["role"].eq(role), "trial_uid"].astype(str))


def iter_contexts(
    assignments: pd.DataFrame, *, scheme: str, include_refit: bool = True
) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """Yield stable ``(outer, inner, context)`` tuples for one evaluation scheme."""

    selected = assignments.loc[assignments["scheme"].astype(str).eq(str(scheme))]
    if not include_refit:
        selected = selected.loc[selected["inner_fold"].astype(int).ne(-1)]
    if selected.empty:
        raise ExperimentSplitError(f"no contexts exist for scheme {scheme!r}")
    for (outer, inner), _ in selected.groupby(["outer_fold", "inner_fold"], sort=True):
        yield (
            int(outer),
            int(inner),
            context_assignments(
                assignments,
                scheme=scheme,
                outer_fold=int(outer),
                inner_fold=int(inner),
            ),
        )


def assignment_sha256(assignments: pd.DataFrame) -> str:
    """Hash only contract fields after a stable ordering."""

    ordered = assignments.loc[:, ASSIGNMENT_COLUMNS].copy()
    ordered.sort_values(
        ["scheme", "outer_fold", "inner_fold", "trial_uid"],
        kind="mergesort",
        inplace=True,
    )
    records = ordered.to_dict(orient="records")
    encoded = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _eligible_base(canonical: pd.DataFrame) -> pd.DataFrame:
    required = {"trial_uid", "segment_id", "target_id", "train_eligible"}
    missing = sorted(required - set(canonical.columns))
    if missing:
        raise ExperimentSplitError(f"canonical real frame is missing fields: {missing}")
    base = canonical.loc[
        canonical["train_eligible"].eq(True),
        ["trial_uid", "segment_id", "target_id"],
    ].copy()
    if base.empty or base["trial_uid"].isna().any() or base["trial_uid"].duplicated().any():
        raise ExperimentSplitError("eligible real IDs must be non-null, unique, and non-empty")
    target_text = base["target_id"].astype("string")
    valid = target_text.str.fullmatch(r"T\d+").fillna(False)
    if not valid.all():
        raise ExperimentSplitError("target_id must match T<digits>")
    base["segment_id"] = base["segment_id"].astype("int64")
    base["target_group"] = target_text.str.slice(1).astype("int64") % TARGET_GROUP_COUNT
    if set(base["segment_id"].unique()) != set(SEGMENT_FOLDS):
        raise ExperimentSplitError("eligible segment fold set changed")
    if set(base["target_group"].unique()) != set(range(TARGET_GROUP_COUNT)):
        raise ExperimentSplitError("eligible target-group set changed")
    base.sort_values("trial_uid", kind="mergesort", inplace=True)
    base.reset_index(drop=True, inplace=True)
    return base


def _validate_parent_assignments(parent: pd.DataFrame, base: pd.DataFrame) -> None:
    missing = sorted(set(ASSIGNMENT_COLUMNS) - set(parent.columns))
    if missing:
        raise ExperimentSplitError(f"preparation assignments are missing fields: {missing}")
    keys = parent[["scheme", "outer_fold", "inner_fold", "split_context"]].drop_duplicates()
    expected = {
        item
        for item in _expected_context_keys()
        if item[0] == "segment_loso" or (item[0] == "target_group" and item[2] == -1)
    }
    actual = {
        (str(row.scheme), int(row.outer_fold), int(row.inner_fold), str(row.split_context))
        for row in keys.itertuples(index=False)
    }
    if actual != expected:
        raise ExperimentSplitError("preparation assignment context set changed")
    if len(parent) != len(base) * len(expected):
        raise ExperimentSplitError("preparation assignment row count is not context-complete")
    expected_ids = set(base["trial_uid"].astype(str))
    for name, context in parent.groupby("split_context", sort=False):
        ids = context["trial_uid"].astype(str)
        if ids.duplicated().any() or set(ids) != expected_ids:
            raise ExperimentSplitError(
                f"preparation context {name!r} does not cover eligible IDs exactly"
            )
    merged = parent.merge(
        base,
        on="trial_uid",
        how="left",
        suffixes=("", "_canonical"),
        validate="many_to_one",
    )
    if merged["segment_id_canonical"].isna().any():
        raise ExperimentSplitError("preparation assignments contain unknown trial IDs")
    for column in ("segment_id", "target_id", "target_group"):
        if not merged[column].astype(str).equals(merged[f"{column}_canonical"].astype(str)):
            raise ExperimentSplitError(f"preparation assignment {column} metadata drifted")


def _expected_context_keys() -> set[tuple[str, int, int, str]]:
    result: set[tuple[str, int, int, str]] = set()
    for outer in SEGMENT_FOLDS:
        result.add(
            ("segment_loso", outer, -1, f"segment_loso:outer={outer}:inner=refit")
        )
        for inner in SEGMENT_FOLDS:
            if inner != outer:
                result.add(
                    ("segment_loso", outer, inner, f"segment_loso:outer={outer}:inner={inner}")
                )
    for outer in range(TARGET_GROUP_COUNT):
        result.add(
            ("target_group", outer, -1, f"target_group:outer={outer}:inner=refit")
        )
        for inner in range(TARGET_GROUP_COUNT):
            if inner != outer:
                result.add(
                    ("target_group", outer, inner, f"target_group:outer={outer}:inner={inner}")
                )
    return result


def _validate_group_roles(
    context: pd.DataFrame, *, scheme: str, outer_fold: int, inner_fold: int
) -> None:
    if scheme == "segment_loso":
        group_column = "segment_id"
    elif scheme == "target_group":
        group_column = "target_group"
    else:
        raise ExperimentSplitError(f"unknown scheme {scheme!r}")
    groups_by_role = {
        role: set(context.loc[context["role"].eq(role), group_column].astype(int))
        for role in ("train", "validation", "test")
    }
    if groups_by_role["test"] != {outer_fold}:
        raise ExperimentSplitError(
            f"{scheme} outer={outer_fold} test rows do not equal the held-out group"
        )
    if outer_fold in groups_by_role["train"] or outer_fold in groups_by_role["validation"]:
        raise ExperimentSplitError("outer-test group leaked into train or validation")
    if inner_fold == -1:
        if groups_by_role["validation"]:
            raise ExperimentSplitError("outer-refit context contains validation groups")
    else:
        if groups_by_role["validation"] != {inner_fold}:
            raise ExperimentSplitError("inner validation rows do not equal held-out inner group")
        if inner_fold in groups_by_role["train"] or inner_fold in groups_by_role["test"]:
            raise ExperimentSplitError("inner validation group leaked into train or test")
    if groups_by_role["train"] & groups_by_role["test"]:
        raise ExperimentSplitError("train and test groups overlap")


__all__ = [
    "ELIGIBLE_REAL_ROWS",
    "EXPERIMENT_ASSIGNMENT_ROWS",
    "ExperimentSplitError",
    "assignment_sha256",
    "build_experiment_assignments",
    "context_assignments",
    "context_ids",
    "iter_contexts",
    "validate_experiment_assignments",
]

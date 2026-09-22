from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from .augmentation import cmixup_offline_v1, local_interp_v2
from .cleaning import clean_trials
from .config import config_sha256, public_config, resolve_source_archive
from .features import add_feature_columns, feature_views
from .io_utils import relative_file_manifest, sha256_file, write_json, write_parquet, write_text
from .reporting import write_analysis
from .source import load_trials_from_config
from .splitting import build_split_assignments


IMU_PREFIXES = (
    "imu0_base",
    "imu1_link_root",
    "imu2_link_mid",
    "imu3_tip",
)
EULER_COLUMNS = tuple(
    f"{prefix}_euler_{axis}_rad" for prefix in IMU_PREFIXES for axis in "xyz"
)
ACCEL_COLUMNS = tuple(
    f"{prefix}_accel_{axis}_mps2" for prefix in IMU_PREFIXES for axis in "xyz"
)
MAG_COLUMNS = tuple(
    f"{prefix}_mag_{axis}_raw" for prefix in IMU_PREFIXES for axis in "xyz"
)
GYRO_COLUMNS = tuple(f"{prefix}_gyro_rms_rad_s" for prefix in IMU_PREFIXES)
ENCODER_DELTA_COLUMNS = tuple(f"encoder_delta_rad_j{axis}" for axis in range(3))
PLANNED_COLUMNS = ("planned_p_x_mm", "planned_p_y_mm", "planned_p_z_mm")
LABEL_COLUMNS = (
    "vive_pos_raw_x_mm",
    "vive_pos_raw_y_mm",
    "vive_pos_raw_z_mm",
)
PROVENANCE_COLUMNS = (
    "synthetic_id",
    "augmentation_method",
    "split_scheme",
    "outer_fold",
    "inner_fold",
    "split_context",
    "anchor_trial_uid",
    "anchor_replica_index",
    "replacement_for_missing_anchor",
    "parent_a_trial_uid",
    "parent_b_trial_uid",
    "mix_lambda",
    "planned_parent_distance_mm",
    "label_parent_distance_mm",
    "joint_parent_distance_deg",
    "cmixup_sigma_mm",
    "derived_seed",
    "source_hash",
    "config_hash",
    "sample_weight",
)


REFERENCES = [
    {
        "id": "yao2022cmixup",
        "title": "C-Mixup: Improving Generalization in Regression",
        "venue": "NeurIPS 2022",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2022/hash/1626be0ab7f3d7b3c639fbfd5951bc40-Abstract-Conference.html",
        "decision": "Use train-only label-similarity pairing instead of uniform regression mixup.",
    },
    {
        "id": "cmixup_official",
        "title": "Official C-Mixup implementation",
        "url": "https://github.com/huaxiuyao/C-Mixup",
        "decision": "Use Beta(2,2) as the frozen offline materialisation default.",
    },
    {
        "id": "kapoor2023leakage",
        "title": "Leakage and the reproducibility crisis in machine-learning-based science",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/",
        "decision": "Freeze grouped splits before augmentation; validation and test remain real-only.",
    },
    {
        "id": "hughes2021imu_shape",
        "title": "Sensing Soft Robot Shape Using IMUs",
        "url": "https://repository.tudelft.nl/record/uuid%3Aeccca35c-d390-4878-8211-4731a9884dfb",
        "decision": "Do not claim formal shape reconstruction from the current four-sensor endpoint proxy.",
    },
    {
        "id": "kabsch1976",
        "title": "A solution for the best rotation to relate two sets of vectors",
        "url": "https://pure.mpg.de/view/item_1999877",
        "decision": "Do not reuse the legacy all-data rigid registration as an evaluation-safe calibration.",
    },
]


def schema_payload(frame: pd.DataFrame) -> dict[str, Any]:
    arrow_schema = pa.Schema.from_pandas(frame, preserve_index=False)
    fields = [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in arrow_schema
    ]
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "columns": fields,
        "fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _git_revision(workspace_root: Path) -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=workspace_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if dirty:
            raise RuntimeError(
                "refusing a release build from a dirty tracked worktree; commit the implementation first"
            )
        return revision
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _prepare_run_dir(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def validate_frozen_config(config: dict[str, Any]) -> None:
    """Reject resolved settings that disagree with the v1 implementation."""

    if str(config["contract"].get("name")) != "alphaf.ml_preparation.v1":
        raise ValueError("contract.name must be alphaf.ml_preparation.v1")
    if float(config["augmentation"].get("ratio", -1.0)) != 1.0:
        raise ValueError("alphaf.ml_preparation.v1 requires augmentation.ratio=1.0")
    features = config["features"]
    expected_labels = list(LABEL_COLUMNS)
    if features.get("label_columns") != expected_labels:
        raise ValueError(f"features.label_columns must equal {expected_labels}")
    if str(features.get("euler_unit")) != "rad":
        raise ValueError("features.euler_unit must be rad")
    if features.get("include_magnetometer_primary") is not False:
        raise ValueError("magnetometer must remain outside primary v1 features")
    if features.get("include_planned_primary") is not False:
        raise ValueError("planned values must remain outside primary v1 features")


def _column_schema(frame: pd.DataFrame, views: dict[str, list[str]]) -> pd.DataFrame:
    primary = set(views["fusion_v1"])
    ablation = set(views["command_context_ablation_v1"]) | set(views["mag_ablation_v1"])
    rows: list[dict[str, str]] = []
    for column in frame.columns:
        if column in LABEL_COLUMNS:
            role = "label"
        elif column in primary:
            role = "primary_feature"
        elif column in ablation:
            role = "ablation_feature"
        elif column in {"trial_uid", "train_eligible", "train_exclusion_reasons", "qc_warning_flags"}:
            role = "contract_metadata"
        else:
            role = "source_or_audit_metadata"
        if column.endswith("_mm"):
            unit = "mm"
        elif "_rad" in column and not column.endswith(("_sin", "_cos")):
            unit = "rad"
        elif column.endswith("_mps2"):
            unit = "m/s^2"
        elif column.endswith("_rad_s"):
            unit = "rad/s"
        elif column.startswith("vive_pos_") and column.endswith("_m"):
            unit = "m"
        elif "tick" in column:
            unit = "tick"
        else:
            unit = ""
        frame_name = "vive_world" if column.startswith("vive_pos_") else ""
        rows.append(
            {
                "column": column,
                "dtype": str(frame[column].dtype),
                "role": role,
                "unit": unit,
                "frame": frame_name,
            }
        )
    return pd.DataFrame(rows)


def _context_train_ids(
    assignments: pd.DataFrame,
    scheme: str,
    outer_fold: Any,
    inner_fold: int = -1,
) -> set[str]:
    selected = assignments.loc[
        (assignments["scheme"] == scheme)
        & (assignments["outer_fold"].astype(str) == str(outer_fold))
        & (assignments["inner_fold"] == int(inner_fold))
        & (assignments["role"] == "train"),
        "trial_uid",
    ]
    return set(selected.astype(str))


def generate_context_augmentations(
    canonical: pd.DataFrame,
    assignments: pd.DataFrame,
    config: dict[str, Any],
    *,
    scheme: str,
    outer_fold: int,
    inner_fold: int = -1,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Generate both v1 variants from exactly one declared real train context."""

    if scheme not in {"segment_loso", "target_group"}:
        raise ValueError(f"unsupported split scheme: {scheme}")
    if scheme == "target_group" and inner_fold != -1:
        raise ValueError("target_group v1 has no inner-fold context")
    context = assignments.loc[
        assignments["scheme"].eq(scheme)
        & assignments["outer_fold"].eq(int(outer_fold))
        & assignments["inner_fold"].eq(int(inner_fold))
    ]
    if context.empty:
        raise ValueError(
            f"split context does not exist: {scheme}/outer={outer_fold}/inner={inner_fold}"
        )
    train_ids = _context_train_ids(
        assignments, scheme, outer_fold, inner_fold=inner_fold
    )
    eligible = canonical.loc[canonical["train_eligible"].eq(True)]
    train = eligible.loc[eligible["trial_uid"].astype(str).isin(train_ids)].copy()
    if len(train) != len(train_ids):
        raise AssertionError(
            f"split {scheme}/{outer_fold}/{inner_fold} contains unknown or duplicate ids"
        )

    augmentation_cfg = config["augmentation"]
    source_cfg = config["source"]
    trig_map = {name: (f"{name}_sin", f"{name}_cos") for name in EULER_COLUMNS}
    linear_columns = (*ENCODER_DELTA_COLUMNS, *ACCEL_COLUMNS, *MAG_COLUMNS, *GYRO_COLUMNS)
    shared = {
        "planned_columns": PLANNED_COLUMNS,
        "label_columns": LABEL_COLUMNS,
        "linear_columns": linear_columns,
        "euler_columns": EULER_COLUMNS,
        "euler_trig_columns": trig_map,
        "split_scheme": scheme,
        "fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "base_seed": int(config["contract"]["base_seed"]),
        "source_hash": str(source_cfg["member_sha256"]),
        "config_hash": config_sha256(config),
        "planned_max_distance_mm": float(
            augmentation_cfg["planned_max_distance_mm"]
        ),
        "sample_weight": 1.0,
    }
    local = local_interp_v2(
        train,
        neighbor_count=int(augmentation_cfg["local_neighbor_count"]),
        lambdas=tuple(float(value) for value in augmentation_cfg["local_lambdas"]),
        **shared,
    )
    cmixup = cmixup_offline_v1(
        train,
        joint_rad_columns=ENCODER_DELTA_COLUMNS,
        joint_max_distance_deg=float(augmentation_cfg["joint_max_distance_deg"]),
        alpha=float(augmentation_cfg["cmixup_alpha"]),
        **shared,
    )
    return train, {
        "local_interp_v2": local,
        "cmixup_offline_v1": cmixup,
    }


def build_run(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    workspace_root = Path(config["_workspace_root"])
    validate_frozen_config(config)
    code_revision = _git_revision(workspace_root)
    _prepare_run_dir(output)
    source_archive = resolve_source_archive(config)
    cfg_hash = config_sha256(config)
    source_cfg = config["source"]

    source = load_trials_from_config(source_cfg, base_dir=workspace_root)
    canonical = clean_trials(
        source,
        config["cleaning"],
        session_id=str(config["contract"]["session_id"]),
        source_member_sha256=str(source_cfg["member_sha256"]),
    )
    canonical = add_feature_columns(canonical)
    canonical = canonical.sort_values(["segment_id", "t_start_ns", "trial_uid"]).reset_index(
        drop=True
    )
    eligible = canonical.loc[canonical["train_eligible"]].copy()
    assignments = build_split_assignments(eligible, config["splits"])
    assignments = assignments.sort_values(
        ["scheme", "outer_fold", "inner_fold", "role", "trial_uid"],
        kind="mergesort",
    ).reset_index(drop=True)

    views = feature_views()
    write_parquet(output / "canonical" / "canonical_real.parquet", canonical)
    write_json(output / "canonical" / "feature_sets.json", views)
    _column_schema(canonical, views).to_csv(
        output / "canonical" / "column_schema.csv", index=False, lineterminator="\n"
    )
    source_schema = schema_payload(source)
    write_json(output / "canonical" / "source_schema.json", source_schema)
    write_parquet(output / "splits" / "split_assignments.parquet", assignments)
    write_json(output / "config.resolved.json", public_config(config))
    write_json(output / "references.json", REFERENCES)

    augmentation_cfg = config["augmentation"]
    augmentation_rows: list[dict[str, Any]] = []
    provenance_frames: list[pd.DataFrame] = []
    synthetic_schema: dict[str, Any] | None = None
    outer_values = {
        "segment_loso": [int(value) for value in config["splits"]["segment_outer_values"]],
        "target_group": list(range(int(config["splits"]["target_group_count"]))),
    }
    for scheme, folds in outer_values.items():
        for fold in folds:
            train, generated = generate_context_augmentations(
                canonical,
                assignments,
                config,
                scheme=scheme,
                outer_fold=int(fold),
                inner_fold=-1,
            )
            for method, augmented in generated.items():
                relative = Path("augmentation") / scheme / f"fold_{fold}" / f"{method}.parquet"
                write_parquet(output / relative, augmented)
                current_schema = schema_payload(augmented)
                if synthetic_schema is None:
                    synthetic_schema = current_schema
                elif synthetic_schema != current_schema:
                    raise AssertionError("synthetic fold schemas are not identical")
                provenance_frames.append(augmented.loc[:, list(PROVENANCE_COLUMNS)].copy())
                augmentation_rows.append(
                    {
                        "scheme": scheme,
                        "outer_fold": fold,
                        "method": method,
                        "real_train_rows": int(len(train)),
                        "rows": int(len(augmented)),
                        "path": relative.as_posix(),
                        "cmixup_sigma_mm": (
                            float(augmented.attrs["cmixup_sigma_mm"])
                            if method == "cmixup_offline_v1"
                            else None
                        ),
                        "missing_anchor_count": int(
                            augmented.attrs.get("missing_anchor_count", 0)
                        ),
                        "replacement_count": int(
                            augmented.attrs.get("replacement_count", 0)
                        ),
                    }
                )

    expected_synthetic = int(augmentation_cfg["expected_total_synthetic_rows"])
    actual_synthetic = sum(row["rows"] for row in augmentation_rows)
    if actual_synthetic != expected_synthetic:
        raise AssertionError(
            f"synthetic row count changed: expected {expected_synthetic}, got {actual_synthetic}"
        )
    provenance = pd.concat(provenance_frames, ignore_index=True)
    provenance.sort_values("synthetic_id", kind="mergesort", inplace=True)
    provenance.reset_index(drop=True, inplace=True)
    write_parquet(output / "augmentation" / "provenance.parquet", provenance)
    if synthetic_schema is None:
        raise AssertionError("no synthetic schema was produced")
    analysis = write_analysis(
        output / "analysis",
        source,
        canonical,
        assignments,
        augmentation_rows,
        provenance,
    )
    write_text(
        output / "README.md",
        """# AlphaF ML Preparation v1

This package contains canonical real trials once and synthetic-only outer-fold
artifacts. Consumers must join only the real training IDs named by the split
manifest. Validation and test rows are always real measurements.

The label is raw Vive position in `vive_world`, millimetres. The four-IMU view
is an endpoint-regression proxy, not formal three-IMU-per-link ShapeFK.
""",
    )

    manifest = {
        "contract": str(config["contract"]["name"]),
        "config_sha256": cfg_hash,
        "code_revision": code_revision,
        "source": {
            "archive_path": source_cfg["archive"],
            "archive_sha256": sha256_file(source_archive),
            "member": source_cfg["member"],
            "member_sha256": source_cfg["member_sha256"],
            "rows": int(len(source)),
            "columns": int(source.shape[1]),
        },
        "counts": {
            "source_trials": int(len(source)),
            "canonical_real": int(len(canonical)),
            "train_eligible_real": int(canonical["train_eligible"].sum()),
            "split_assignments": int(len(assignments)),
            "synthetic_materialized": actual_synthetic,
        },
        "labels": {"frame": "vive_world", "unit": "mm", "columns": list(LABEL_COLUMNS)},
        "recipes": {
            "trial_uid": "sha256(source_member_sha256|session_id|segment_id|t_start_ns)",
            "encoder_delta_rad": "(raw_tick-reference_tick)*2*pi/4096",
            "seed_derivation": (
                "uint63(sha256(json([base_seed,method,scheme,outer_fold,"
                "inner_fold,anchor_uid,replica])))[:8])"
            ),
        },
        "schemas": {
            "source": source_schema,
            "canonical_real": schema_payload(canonical),
            "split_assignments": schema_payload(assignments),
            "synthetic_fold": synthetic_schema,
            "augmentation_provenance": schema_payload(provenance),
        },
        "augmentation": augmentation_rows,
        "analysis_summary": {
            "target_shortcut_segment_loso": analysis["target_shortcut_segment_loso"]
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    }
    manifest["files"] = relative_file_manifest(output, exclude={"manifest.json"})
    write_json(output / "manifest.json", manifest)
    return manifest


def build_inner_cache(
    run_dir: str | Path,
    *,
    outer_fold: int,
    inner_fold: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Create deterministic segment-LOSO synthetic data for one inner train context.

    The cache is deliberately separate from the release package. Its parents and
    C-Mixup bandwidth are computed from the declared inner-train rows only.
    """

    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    if int(inner_fold) == -1 or int(inner_fold) == int(outer_fold):
        raise ValueError("inner_fold must be non-negative and differ from outer_fold")
    config = json.loads((run / "config.resolved.json").read_text(encoding="utf-8"))
    validate_frozen_config(config)
    parent_manifest_path = run / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    cfg_hash = config_sha256(config)
    if parent_manifest.get("config_sha256") != cfg_hash:
        raise ValueError("parent run config hash does not match config.resolved.json")

    canonical = pd.read_parquet(run / "canonical" / "canonical_real.parquet")
    assignments = pd.read_parquet(run / "splits" / "split_assignments.parquet")
    train, generated = generate_context_augmentations(
        canonical,
        assignments,
        config,
        scheme="segment_loso",
        outer_fold=int(outer_fold),
        inner_fold=int(inner_fold),
    )
    _prepare_run_dir(output)
    provenance_parts: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []
    synthetic_schema: dict[str, Any] | None = None
    for method, frame in generated.items():
        relative = Path(f"{method}.parquet")
        write_parquet(output / relative, frame)
        provenance_parts.append(frame.loc[:, list(PROVENANCE_COLUMNS)].copy())
        current_schema = schema_payload(frame)
        if synthetic_schema is None:
            synthetic_schema = current_schema
        elif synthetic_schema != current_schema:
            raise AssertionError("inner-cache synthetic schemas are not identical")
        rows.append(
            {
                "method": method,
                "path": relative.as_posix(),
                "real_train_rows": int(len(train)),
                "rows": int(len(frame)),
                "cmixup_sigma_mm": (
                    float(frame.attrs["cmixup_sigma_mm"])
                    if method == "cmixup_offline_v1"
                    else None
                ),
                "missing_anchor_count": int(frame.attrs.get("missing_anchor_count", 0)),
                "replacement_count": int(frame.attrs.get("replacement_count", 0)),
            }
        )
    provenance = pd.concat(provenance_parts, ignore_index=True)
    provenance.sort_values("synthetic_id", kind="mergesort", inplace=True)
    provenance.reset_index(drop=True, inplace=True)
    write_parquet(output / "provenance.parquet", provenance)
    manifest = {
        "contract": "alphaf.ml_preparation.v1.inner_cache",
        "parent_contract": parent_manifest.get("contract"),
        "parent_manifest_sha256": sha256_file(parent_manifest_path),
        "config_sha256": cfg_hash,
        "code_revision": parent_manifest.get("code_revision"),
        "split": {
            "scheme": "segment_loso",
            "outer_fold": int(outer_fold),
            "inner_fold": int(inner_fold),
            "train_real_rows": int(len(train)),
        },
        "augmentation": rows,
        "schemas": {
            "synthetic_fold": synthetic_schema,
            "provenance": schema_payload(provenance),
        },
    }
    manifest["files"] = relative_file_manifest(output, exclude={"manifest.json"})
    write_json(output / "manifest.json", manifest)
    return manifest

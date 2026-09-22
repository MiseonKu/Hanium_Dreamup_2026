"""Create workspace validation plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fkik.geometry import load_config


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def _load_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing point file: {path}")
    return pd.read_csv(path)


def plot_workspace_preview(df: pd.DataFrame, output: Path) -> None:
    """Save a 3D scatter of selected p_flex points."""

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    classes = sorted(df["bending_class"].dropna().unique()) if "bending_class" in df else ["all"]
    color_map = {"low": "tab:blue", "mid": "tab:green", "high": "tab:red", "all": "tab:blue"}
    for cls in classes:
        subset = df if cls == "all" else df[df["bending_class"] == cls]
        ax.scatter(subset["p_flex_x"], subset["p_flex_y"], subset["p_flex_z"], s=12, alpha=0.75, label=str(cls), c=color_map.get(cls, None))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title("Flex-aware workspace targets")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_count_bar(counts: pd.Series, output: Path, title: str, xlabel: str) -> None:
    """Save a compact bar plot for categorical counts."""

    fig, ax = plt.subplots(figsize=(9, 4))
    counts.plot(kind="bar", ax=ax, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_bin_distribution(df: pd.DataFrame, output: Path) -> None:
    """Save sorted bin counts without dense x labels."""

    counts = df.groupby("bin_id").size().sort_values().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(counts)), counts["count"], color="tab:purple")
    ax.set_title("Target count per selected bin")
    ax.set_xlabel("sorted bin index")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_q_distribution(df: pd.DataFrame, output: Path) -> None:
    """Save q1/q2/q3 histograms in radians."""

    q_cols = ["q_seed_1", "q_seed_2", "q_seed_3"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, col in zip(axes, q_cols):
        ax.hist(df[col], bins=30, color="tab:orange", alpha=0.85)
        ax.set_title(col)
        ax.set_xlabel("rad")
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate workspace plots.")
    parser.add_argument("--config", default="config/robot_config.yaml")
    parser.add_argument("--points", default="outputs/target_points.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    load_config(_resolve(args.config))
    df = _load_points(_resolve(args.points))
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_workspace_preview(df, output_dir / "workspace_preview.png")
    plot_count_bar(df["radial_shell"].value_counts().sort_index(), output_dir / "radial_shell_distribution.png", "Radial shell distribution", "radial shell")
    plot_bin_distribution(df, output_dir / "bin_count_distribution.png")
    plot_count_bar(df["bending_class"].value_counts().sort_index(), output_dir / "bending_score_distribution.png", "Bending class distribution", "bending class")
    plot_q_distribution(df, output_dir / "q_distribution.png")
    print(f"wrote plots to {output_dir}")


if __name__ == "__main__":
    main()

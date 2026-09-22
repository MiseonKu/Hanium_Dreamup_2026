from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fkik.geometry import load_config
from workspace_points.binning import add_bin_labels
from workspace_points.candidate_generator import generate_candidates
from workspace_points.uniform_bin_sampler import sample_uniform_targets


def small_config():
    cfg = load_config(ROOT / "config" / "robot_config.yaml")
    return replace(cfg, n_candidates=4000, n_targets=180, random_seed=123)


def test_target_points_csv_has_requested_count(tmp_path):
    cfg = small_config()
    candidates = add_bin_labels(generate_candidates(cfg, cfg.n_candidates), cfg)
    targets, report = sample_uniform_targets(candidates, cfg)
    out = tmp_path / "target_points.csv"
    targets.to_csv(out, index=False)
    assert out.exists()
    assert len(targets) == cfg.n_targets
    assert set(["link_length_preset", "link_a_mm", "link_b_mm"]).issubset(targets.columns)
    assert targets["link_a_mm"].eq(cfg.link_a_mm).all()
    assert targets["link_b_mm"].eq(cfg.link_b_mm).all()
    assert report["uniformity_score"].dropna().iloc[0] <= 0.35


def test_selected_bin_counts_are_close_to_uniform():
    cfg = small_config()
    candidates = add_bin_labels(generate_candidates(cfg, cfg.n_candidates), cfg)
    targets, _ = sample_uniform_targets(candidates, cfg)
    counts = targets.groupby("bin_id").size()
    uniformity = counts.std(ddof=0) / counts.mean()
    assert uniformity <= 0.35


def test_bending_classes_all_exist_in_targets():
    cfg = small_config()
    candidates = add_bin_labels(generate_candidates(cfg, cfg.n_candidates), cfg)
    targets, _ = sample_uniform_targets(candidates, cfg)
    assert {"low", "mid", "high"}.issubset(set(targets["bending_class"]))

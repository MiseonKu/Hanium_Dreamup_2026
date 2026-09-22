from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fkik.flex_fk import fk_flex
from fkik.flex_ik import ik_flex
from fkik.geometry import load_config
from fkik.rigid_fk import fk_rigid
from fkik.rigid_ik import ik_rigid


def config():
    return load_config(ROOT / "config" / "robot_config.yaml")


def test_fk_rigid_zero_pose_reaches_l1_plus_l2():
    cfg = config()
    p = fk_rigid(np.array([0.0, 0.0, 0.0]), cfg)
    assert np.allclose(p, np.array([cfg.L1 + cfg.L2, 0.0, 0.0]), atol=1e-9)
    assert cfg.link_a_mm == cfg.L1
    assert cfg.link_b_mm == cfg.L2


def test_link_length_change_updates_fk_automatically():
    cfg = replace(config(), L1=250.0, L2=180.0, link_length_preset="test_AB")
    p = fk_rigid(np.array([0.0, 0.0, 0.0]), cfg)
    assert np.allclose(p, np.array([430.0, 0.0, 0.0]), atol=1e-9)


def test_default_material_preset_matches_latest_print_spec():
    cfg = config()
    assert cfg.material_preset == "TPU95A_40x12_wall4_topbottom4_infill15_honeycomb"
    assert cfg.wall_count == 4
    assert cfg.top_bottom_wall_count == 4
    assert cfg.infill_percent == 15
    assert cfg.infill_type == "honeycomb"
    assert np.allclose(cfg.cross_section_mm, np.array([40.0, 12.0]))


def test_ik_rigid_round_trip_position_error_is_small():
    cfg = config()
    q = np.array([0.3, 0.45, -0.9])
    p = fk_rigid(q, cfg)
    result = ik_rigid(p, cfg, q_init=q)
    assert result.success, result.reason
    assert result.q is not None
    assert np.linalg.norm(fk_rigid(result.q, cfg) - p) < 1e-6


def test_fk_flex_equals_rigid_when_flex_gains_are_zero():
    cfg = config()
    q = np.array([-0.4, 0.55, -1.1])
    flex = fk_flex(q, cfg)
    rigid = fk_rigid(q, cfg)
    assert abs(cfg.k1) == 0.0
    assert abs(cfg.k2) == 0.0
    assert np.allclose(flex.p_flex, rigid, atol=1e-9)
    assert flex.d_flex < 1e-9


def test_ik_flex_matches_rigid_when_flex_gains_are_zero():
    cfg = config()
    q = np.array([0.2, 0.65, -1.2])
    p = fk_rigid(q, cfg)
    rigid = ik_rigid(p, cfg, q_init=q)
    flex = ik_flex(p, cfg, q_init=q)
    assert rigid.success, rigid.reason
    assert flex.success, flex.reason
    assert rigid.q is not None
    assert flex.q is not None
    assert np.allclose(flex.q, rigid.q, atol=1e-7)

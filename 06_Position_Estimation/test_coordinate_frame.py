"""Offline numerical checks; no dataset, serial or Vive connections."""
import importlib.util
import unittest
import numpy as np
from demo_core import DemoError, fk
from fit_coordinate_frame import rigid_fit, effective_fk, fit_model, predict, fitting_mask


class CoordinateFrameTests(unittest.TestCase):
    def test_recovers_known_proper_transform(self):
        a = np.array([[0,0,0],[100,0,0],[0,100,0],[0,0,100],[20,30,40]], float)
        r = np.array([[0,-1,0],[1,0,0],[0,0,1]], float)
        got_r, got_t = rigid_fit(a, a@r.T+[10,20,30])
        np.testing.assert_allclose(got_r, r, atol=1e-10)
        np.testing.assert_allclose(got_t, [10,20,30], atol=1e-10)

    def test_does_not_hide_reflection(self):
        a = np.array([[0,0,0],[100,0,0],[0,100,0],[0,0,100]], float)
        b = a*[-1,1,1]
        r,t = rigid_fit(a,b)
        self.assertAlmostEqual(np.linalg.det(r),1.)
        self.assertGreater(np.linalg.norm(a@r.T+t-b),1.)

    def test_bad_data_rejected(self):
        for a in (np.zeros((4,3)), np.full((4,3),np.nan), np.zeros((3,3))):
            with self.assertRaises(DemoError):
                rigid_fit(a,a)

    def test_nominal_fk_matches_existing_demo(self):
        raw = np.array([[2047,2122,2714],[2698,2531,2425]], float)
        geometry = {'link_lengths_mm':[233,215], 'marker_offset_tip_mm':[0,0,0]}
        np.testing.assert_allclose(effective_fk(raw), np.array([fk(row,geometry) for row in raw]))

    def test_demo_and_later_segments_excluded_from_fit(self):
        mask = fitting_mask([36,37,38,39,40,36], ['T002','T003','T004','T003','T004','T015'],
                            ['T002','T015'])
        np.testing.assert_array_equal(mask,[False,True,True,False,False,False])

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'optional analysis dependency scipy absent')
    def test_synthetic_anchor_recovery(self):
        rng = np.random.default_rng(2026)
        raw = rng.uniform([1400,2150,2050],[2800,2800,3200],size=(24,3))
        p = effective_fk(raw)
        world = effective_fk(raw,[-1.2,.25])+[10,20,30]
        table = np.column_stack((p,raw,world))
        model = fit_model(table,'encoder_anchor_fit')
        self.assertLess(np.max(np.linalg.norm(predict(table,model)-world,axis=1)),1e-4)
        self.assertFalse(model['live_authorized'])
        self.assertFalse(model['independently_calibrated'])


if __name__ == '__main__':
    unittest.main()

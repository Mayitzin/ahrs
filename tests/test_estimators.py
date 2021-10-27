#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

class TestTRIAD(unittest.TestCase):
    def setUp(self) -> None:
        wmm = ahrs.utils.WMM(latitude=ahrs.MUNICH_LATITUDE, longitude=ahrs.MUNICH_LONGITUDE, height=ahrs.MUNICH_HEIGHT)
        g = np.array([0.0, 0.0, -1.0]) + np.random.randn(3)*1e-7
        m = np.array([wmm.X, wmm.Y, wmm.Z]) + np.random.randn(3)*1e-7
        self.R = ahrs.DCM(rpy=np.random.random(3)*90.0-45.0)
        self.Rg = self.R @ g
        self.Rm = self.R @ m

    def tearDown(self) -> None:
        del self.R

    def test_correct_values(self):
        R2 = ahrs.filters.TRIAD(self.Rg, self.Rm)
        np.testing.assert_allclose(self.R, R2.A, atol=1e-6)

    def test_wrong_frame(self):
        self.assertRaises(TypeError, ahrs.filters.TRIAD, frame=1.0)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, frame=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, frame='Up')

    def test_wrong_vectors(self):
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1.0, 2.0], w2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=np.copy(1.0), w2=[2.0, 3.0, 4.0])

if __name__ == '__main__':
    unittest.main()

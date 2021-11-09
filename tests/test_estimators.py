#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

LOCAL_GEOMAGNETIC_FIELD = ahrs.utils.WMM(latitude=ahrs.MUNICH_LATITUDE, longitude=ahrs.MUNICH_LONGITUDE, height=ahrs.MUNICH_HEIGHT)
REFERENCE_MAGNETIC_VECTOR = np.array([LOCAL_GEOMAGNETIC_FIELD.X, LOCAL_GEOMAGNETIC_FIELD.Y, LOCAL_GEOMAGNETIC_FIELD.Z])

class TestTRIAD(unittest.TestCase):
    def setUp(self) -> None:
        wmm = ahrs.utils.WMM(latitude=ahrs.MUNICH_LATITUDE, longitude=ahrs.MUNICH_LONGITUDE, height=ahrs.MUNICH_HEIGHT)
        g = np.array([0.0, 0.0, -1.0]) + np.random.randn(3)*1e-7        # Reference gravity vector + noise
        m = REFERENCE_MAGNETIC_VECTOR + np.random.randn(3)*1e-7   # Reference magnetic field vector + noise
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

class TestSAAM(unittest.TestCase):
    def setUp(self) -> None:
        g = np.array([0.0, 0.0, 1.0])   # Reference gravity vector
        m = REFERENCE_MAGNETIC_VECTOR   # Reference magnetic field vector
        ### 2-dimensional vectors
        num_samples = 1000
        self.Qts = ahrs.QuaternionArray(np.random.random((num_samples, 4)) - 0.5)
        R = self.Qts.to_DCM()
        noises = np.random.randn(2*num_samples, 3)*1e-3
        self.Rg = np.array([r.T @ g + noises[i] for i, r in enumerate(R)])
        self.Rm = np.array([r.T @ m + noises[i+num_samples] for i, r in enumerate(R)])

    def test_correct_values(self):
        R2 = ahrs.filters.SAAM(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, R2.Q)), 6.7e-2)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc="self.Rg", mag="self.Rm")
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

if __name__ == '__main__':
    unittest.main()

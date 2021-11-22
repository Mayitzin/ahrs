#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

wmm = ahrs.utils.WMM(latitude=ahrs.MUNICH_LATITUDE, longitude=ahrs.MUNICH_LONGITUDE, height=ahrs.MUNICH_HEIGHT)
REFERENCE_MAGNETIC_VECTOR = np.array([wmm.X, wmm.Y, wmm.Z])
GRAVITY = ahrs.utils.WGS().normal_gravity(ahrs.MUNICH_LATITUDE, ahrs.MUNICH_HEIGHT)

class TestTRIAD(unittest.TestCase):
    def setUp(self) -> None:
        self.decimal_precision = 1e-7
        g = np.array([0.0, 0.0, -1.0]) + np.random.randn(3)*self.decimal_precision  # Reference gravity vector + noise
        m = REFERENCE_MAGNETIC_VECTOR + np.random.randn(3)*self.decimal_precision   # Reference magnetic field vector + noise
        self.R = ahrs.DCM(rpy=np.random.random(3)*90.0-45.0)
        self.Rg = self.R @ g
        self.Rm = self.R @ m

    def test_correct_values(self):
        R2 = ahrs.filters.TRIAD(self.Rg, self.Rm)
        np.testing.assert_allclose(self.R, R2.A, atol=self.decimal_precision*10.0)

    def test_wrong_frame(self):
        self.assertRaises(TypeError, ahrs.filters.TRIAD, frame=1.0)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, frame=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, frame='Up')

    def test_wrong_vectors(self):
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1.0, 2.0], w2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=np.copy(1.0), w2=[2.0, 3.0, 4.0])

class TestSAAM(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        self.Qts = ahrs.QuaternionArray(np.random.random((num_samples, 4)) - 0.5)
        self.rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        noises = np.random.randn(2*num_samples, 3)*1e-3
        self.Rg = np.array([R.T @ (np.array([0.0, 0.0, GRAVITY]) + noises[i]) for i, R in enumerate(self.rotations)])
        self.Rm = np.array([R.T @ (REFERENCE_MAGNETIC_VECTOR + noises[i+num_samples]) for i, R in enumerate(self.rotations)])
        self.decimal_precision = 7e-2

    def test_single_values(self):
        saam = ahrs.filters.SAAM(self.Rg[0], self.Rm[0])
        self.assertLess(ahrs.utils.metrics.qad(self.Qts[0], saam.Q), self.decimal_precision)

    def test_single_values_as_rotation(self):
        saam = ahrs.filters.SAAM(self.Rg[0], self.Rm[0], representation='rotmat')
        np.testing.assert_allclose(saam.A, self.rotations[0], atol=self.decimal_precision)

    def test_multiple_values(self):
        saam = ahrs.filters.SAAM(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, saam.Q)), self.decimal_precision)

    def test_multiple_values_as_rotations(self):
        saam = ahrs.filters.SAAM(self.Rg, self.Rm, representation='rotmat')
        np.testing.assert_allclose(saam.A, self.rotations, atol=self.decimal_precision*2.0)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc="self.Rg", mag="self.Rm")
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

class TestFAMC(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        self.Qts = ahrs.QuaternionArray(np.random.random((num_samples, 4)) - 0.5)
        self.rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        noises = np.random.randn(2*num_samples, 3) * 1e-6
        self.Rg = np.array([R.T @ (np.array([0.0, 0.0, GRAVITY]) + noises[i]) for i, R in enumerate(self.rotations)])
        self.Rm = np.array([R.T @ (REFERENCE_MAGNETIC_VECTOR + noises[i+num_samples]) for i, R in enumerate(self.rotations)])
        self.decimal_precision = 7e-2

    def test_single_values(self):
        orientation = ahrs.filters.FAMC(self.Rg[0], self.Rm[0])
        self.assertLess(ahrs.utils.metrics.qad(orientation.Q, self.Qts[0]), self.decimal_precision)

    def test_multiple_values(self):
        orientation = ahrs.filters.FAMC(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, self.Qts)), self.decimal_precision)

class TestFLAE(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        self.Qts = ahrs.QuaternionArray(np.random.random((num_samples, 4)) - 0.5)
        self.rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        noises = np.random.randn(2*num_samples, 3) * 1e-6
        self.Rg = np.array([R.T @ (np.array([0.0, 0.0, GRAVITY]) + noises[i]) for i, R in enumerate(self.rotations)])
        self.Rm = np.array([R.T @ (REFERENCE_MAGNETIC_VECTOR + noises[i+num_samples]) for i, R in enumerate(self.rotations)])
        self.decimal_precision = 2e-3

    def test_single_values(self):
        orientation = ahrs.filters.FLAE(self.Rg[0], self.Rm[0])
        self.assertLess(ahrs.utils.metrics.qad(orientation.Q, self.Qts[0]), self.decimal_precision)

    def test_single_values_method_eig(self):
        orientation = ahrs.filters.FLAE(self.Rg[0], self.Rm[0], method='eig')
        self.assertLess(ahrs.utils.metrics.qad(orientation.Q, self.Qts[0]), self.decimal_precision)

    def test_single_values_method_newton(self):
        orientation = ahrs.filters.FLAE(self.Rg[0], self.Rm[0], method='newton')
        self.assertLess(ahrs.utils.metrics.qad(orientation.Q, self.Qts[0]), self.decimal_precision)

    def test_multiple_values(self):
        orientation = ahrs.filters.FLAE(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, self.Qts)), self.decimal_precision)

    def test_multiple_values_method_eig(self):
        orientation = ahrs.filters.FLAE(self.Rg, self.Rm, method='eig')
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, self.Qts)), self.decimal_precision)

    def test_multiple_values_method_newton(self):
        orientation = ahrs.filters.FLAE(self.Rg, self.Rm, method='newton')
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, self.Qts)), self.decimal_precision)

class TestQUEST(unittest.TestCase):     ####### NOT PASSING: ERROR IN IMPLEMENTATION #######
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        self.Qts = ahrs.QuaternionArray(np.random.random((num_samples, 4)) - 0.5)
        self.rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        noises = np.random.randn(2*num_samples, 3)*1e-3
        self.Rg = np.array([R.T @ (np.array([0.0, 0.0, -GRAVITY]) + noises[i]) for i, R in enumerate(self.rotations)])
        self.Rm = np.array([R.T @ (REFERENCE_MAGNETIC_VECTOR + noises[i+num_samples]) for i, R in enumerate(self.rotations)])
        self.decimal_precision = 7e-2

    def test_single_values(self):
        quest = ahrs.filters.QUEST(self.Rg[0], self.Rm[0])
        self.assertLess(ahrs.utils.metrics.qad(self.Qts[0], quest.Q), self.decimal_precision)

class TestDavenport(unittest.TestCase):
    def setUp(self) -> None:
        self.decimal_precision = 1e-7
        # Create random attitudes
        num_samples = 1000
        self.Qts = ahrs.QuaternionArray(np.random.random((num_samples, 4)) - 0.5)
        self.rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        noises = np.random.randn(2*num_samples, 3) * self.decimal_precision * 0.1
        self.Rg = np.array([R.T @ (np.array([0.0, 0.0, GRAVITY]) + noises[i]) for i, R in enumerate(self.rotations)])
        self.Rm = np.array([R.T @ (REFERENCE_MAGNETIC_VECTOR + noises[i+num_samples]) for i, R in enumerate(self.rotations)])

    def test_single_values(self):
        orientation = ahrs.filters.Davenport(self.Rg[0], self.Rm[0])
        self.assertLess(ahrs.utils.metrics.qad(orientation.Q, self.Qts[0]), self.decimal_precision)

    def test_multiple_values(self):
        orientation = ahrs.filters.Davenport(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, self.Qts)), self.decimal_precision)

class TestAQUA(unittest.TestCase):
    def setUp(self) -> None:
        self.decimal_precision = 1e-7
        # Create random attitudes
        num_samples = 1000
        self.Qts = ahrs.QuaternionArray(np.random.random((num_samples, 4)) - 0.5)
        self.rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        noises = np.random.randn(2*num_samples, 3) * self.decimal_precision * 0.1
        self.Rg = np.array([R.T @ (np.array([0.0, 0.0, GRAVITY]) + noises[i]) for i, R in enumerate(self.rotations)])
        self.Rm = np.array([R.T @ (REFERENCE_MAGNETIC_VECTOR + noises[i+num_samples]) for i, R in enumerate(self.rotations)])

    def test_single_values(self):
        orientation = ahrs.filters.AQUA(acc=self.Rg[0], mag=self.Rm[0])
        self.assertLess(ahrs.utils.metrics.qad(orientation.Q, self.Qts[0]), self.decimal_precision)

    def test_multiple_values(self):
        orientation = ahrs.filters.AQUA(acc=self.Rg, mag=self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, self.Qts)), self.decimal_precision)

if __name__ == '__main__':
    unittest.main()

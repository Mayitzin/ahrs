#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

wmm = ahrs.utils.WMM(latitude=ahrs.MUNICH_LATITUDE, longitude=ahrs.MUNICH_LONGITUDE, height=ahrs.MUNICH_HEIGHT)
REFERENCE_MAGNETIC_VECTOR = np.array([wmm.X, wmm.Y, wmm.Z])
NORMAL_GRAVITY = ahrs.utils.WGS().normal_gravity(ahrs.MUNICH_LATITUDE, ahrs.MUNICH_HEIGHT)
REFERENCE_GRAVITY_VECTOR = np.array([0.0, 0.0, NORMAL_GRAVITY])

def __gaussian_filter(input, size = 10, sigma = 1.0):
    """Gaussian filter over an array

    This implementation tries to mimic the behavior of Scipy's function
    ``gaussian_filter1d``, in order to avoid the dependency on Scipy.

    Parameters
    ----------
    input : np.ndarray
        Input array to be filtered.
    size : int, default: 10
        Size of Kernel used over the input array.
    sigma : float, default: 1.0
        Standard deviation of Gaussian Kernel.

    Returns
    -------
    y : np.ndarray
        Filtered array.
    """
    x = np.linspace(-sigma*4, sigma*4, size)
    phi_x = np.exp(-0.5*x**2/sigma**2)
    phi_x /= phi_x.sum()
    if input.ndim < 2:
        return np.correlate(input, phi_x, mode='same')
    return np.array([np.correlate(col, phi_x, mode='same') for col in input.T]).T

def random_angvel(num_samples: int = 500, max_rotations: int = 4, num_axes: int = 3, span: list = None, **kwargs) -> np.ndarray:
    """Random angular velocities

    Create an array of synthetic random angular velocities with reference to a
    local sensor coordinate frame.

    Parameters
    ----------
    num_samples : int, default: 500
        Number of samples to generate. Set it to minimum 50, so that the
        gaussian filter can be applied.
    max_rotations : int, default: 4
        Maximum number of rotations per axis.
    num_axes : int, default: 3
        Number of axes required.
    span : list or tuple, default: None
        Span (minimum to maximum) of the random values.

    Returns
    -------
    angvels: np.ndarray
        Array of angular velocities.
    """
    span = span if isinstance(span, (list, tuple)) else [-0.5*np.pi, 0.5*np.pi]
    all_angs = [np.random.uniform(span[0], span[1], np.random.randint(1, max_rotations)) for _ in np.arange(num_axes)]
    angvels = np.zeros((num_samples, num_axes))
    for j, angs in enumerate(all_angs):
        num_angs = len(angs)
        idxs = np.sort(np.random.randint(0, num_samples, 2*num_angs)).reshape((num_angs, 2))
        for i, idx in enumerate(idxs):
            angvels[idx[0]:idx[1], j] = angs[i]
    return __gaussian_filter(angvels, size=kwargs.pop('gauss_size', 50 if num_samples > 50 else num_samples//5), sigma=5)

class TestTRIAD(unittest.TestCase):
    def setUp(self) -> None:
        # Generate random attitudes
        num_samples = 500
        angular_velocities = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.R = ahrs.QuaternionArray(ahrs.filters.AngularRate(angular_velocities).Q).to_DCM()
        # Rotated reference vectors + noise
        self.noise_sigma = 1e-5
        self.Rg = np.array([R @ REFERENCE_GRAVITY_VECTOR for R in self.R]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rm = np.array([R @ REFERENCE_MAGNETIC_VECTOR for R in self.R]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma

    def test_multiple_values(self):
        R2 = ahrs.filters.TRIAD(self.Rg, self.Rm, v1=REFERENCE_GRAVITY_VECTOR, v2=REFERENCE_MAGNETIC_VECTOR)
        self.assertLess(np.nanmean(ahrs.utils.metrics.chordal(self.R, R2.A)), self.noise_sigma*10)

    def test_wrong_frame(self):
        self.assertRaises(TypeError, ahrs.filters.TRIAD, frame=1.0)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, frame=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, frame='Up')

    def test_wrong_vectors(self):
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1.0, 2.0], w2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=np.copy(1.0), w2=[2.0, 3.0, 4.0])

class TestSAAM(unittest.TestCase):
    def setUp(self) -> None:
        self.decimal_precision = 7e-2
        # Generate random attitudes
        num_samples = 500
        angular_velocities = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(angular_velocities).Q)
        self.R = self.Qts.to_DCM()
        # Rotated reference vectors + noise
        noise_sigma = 1e-5
        self.Rg = np.array([R @ REFERENCE_GRAVITY_VECTOR for R in self.R]) + np.random.standard_normal((num_samples, 3)) * noise_sigma
        self.Rm = np.array([R @ REFERENCE_MAGNETIC_VECTOR for R in self.R]) + np.random.standard_normal((num_samples, 3)) * noise_sigma

    def test_single_values(self):
        saam = ahrs.filters.SAAM(self.Rg[0], self.Rm[0])
        self.assertLess(ahrs.utils.metrics.qad(self.Qts[0], saam.Q), self.decimal_precision)

    def test_single_values_as_rotation(self):
        saam = ahrs.filters.SAAM(self.Rg[0], self.Rm[0], representation='rotmat')
        np.testing.assert_allclose(saam.A, self.R[0], atol=self.decimal_precision)

    def test_multiple_values(self):
        saam = ahrs.filters.SAAM(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, saam.Q)), self.decimal_precision)

    def test_multiple_values_as_rotations(self):
        saam = ahrs.filters.SAAM(self.Rg, self.Rm, representation='rotmat')
        np.testing.assert_allclose(saam.A, self.R, atol=self.decimal_precision*2.0)

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
        num_samples = 500
        angular_velocities = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(angular_velocities).Q)
        self.rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.decimal_precision = 7e-2
        noise_sigma = 1e-5
        self.Rg = np.array([R.T @ REFERENCE_GRAVITY_VECTOR for R in self.rotations]) + np.random.standard_normal((num_samples, 3)) * noise_sigma
        self.Rm = np.array([R.T @ REFERENCE_MAGNETIC_VECTOR for R in self.rotations]) + np.random.standard_normal((num_samples, 3)) * noise_sigma

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
        angular_velocities = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(angular_velocities).Q)
        self.rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.noise_sigma = 1e-5
        self.Rg = np.array([R.T @ REFERENCE_GRAVITY_VECTOR for R in self.rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rm = np.array([R.T @ REFERENCE_MAGNETIC_VECTOR for R in self.rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma

    def test_multiple_values(self):
        orientation = ahrs.filters.FLAE(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, self.Qts)), self.noise_sigma * 700)

    def test_multiple_values_method_eig(self):
        orientation = ahrs.filters.FLAE(self.Rg, self.Rm, method='eig')
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, self.Qts)), self.noise_sigma)

    def test_multiple_values_method_newton(self):
        orientation = ahrs.filters.FLAE(self.Rg, self.Rm, method='newton')
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, self.Qts)), self.noise_sigma)

class TestQUEST(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        angular_velocities = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(angular_velocities).Q)
        self.rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        noise_sigma = 1e-5
        self.decimal_precision = noise_sigma * 10.0
        self.Rg = np.array([R.T @ REFERENCE_GRAVITY_VECTOR for R in self.rotations]) + np.random.standard_normal((num_samples, 3)) * noise_sigma
        self.Rm = np.array([R.T @ REFERENCE_MAGNETIC_VECTOR for R in self.rotations]) + np.random.standard_normal((num_samples, 3)) * noise_sigma
        
    def test_multiple_values(self):
        quest = ahrs.filters.QUEST(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, quest.Q)), self.decimal_precision)

class TestDavenport(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        angular_velocities = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(angular_velocities).Q)
        self.rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        noise_sigma = 1e-5
        self.decimal_precision = noise_sigma * 10.0
        self.Rg = np.array([R.T @ REFERENCE_GRAVITY_VECTOR for R in self.rotations]) + np.random.standard_normal((num_samples, 3)) * noise_sigma
        self.Rm = np.array([R.T @ REFERENCE_MAGNETIC_VECTOR for R in self.rotations]) + np.random.standard_normal((num_samples, 3)) * noise_sigma

    def test_single_values(self):
        orientation = ahrs.filters.Davenport(self.Rg[0], self.Rm[0])
        self.assertLess(ahrs.utils.metrics.qad(orientation.Q, self.Qts[0]), self.decimal_precision)

    def test_multiple_values(self):
        orientation = ahrs.filters.Davenport(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, self.Qts)), self.decimal_precision)

class TestAQUA(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        angular_velocities = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(angular_velocities).Q)
        self.rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.noise_sigma = 1e-2
        self.Rg = np.array([R @ REFERENCE_GRAVITY_VECTOR for R in self.rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma * np.ptp(REFERENCE_GRAVITY_VECTOR)
        self.Rm = np.array([R @ REFERENCE_MAGNETIC_VECTOR for R in self.rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma * np.ptp(REFERENCE_MAGNETIC_VECTOR)

    def test_acc_mag(self):
        orientation = ahrs.filters.AQUA(acc=self.Rg, mag=self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, self.Qts)), self.noise_sigma * 10)

class TestFQA(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        a_ref = np.array([0.0, 0.0, -NORMAL_GRAVITY])
        m_ref = REFERENCE_MAGNETIC_VECTOR
        num_samples = 1000
        angular_velocities = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(angular_velocities).Q)
        rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.noise_sigma = 1e-5
        self.Rg = np.array([R.T @ a_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rm = np.array([R.T @ m_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma

    def test_acc_mag(self):
        orientation = ahrs.filters.FQA(acc=self.Rg, mag=self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, orientation.Q)), self.noise_sigma)

class TestMadgwick(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        a_ref = np.array([0.0, 0.0, NORMAL_GRAVITY])
        m_ref = ahrs.common.frames.ned2enu(REFERENCE_MAGNETIC_VECTOR)
        gyros = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(gyros).Q)
        rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.noise_sigma = 1e-2
        self.gyr = gyros + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rg = np.array([R @ a_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rm = np.array([R @ m_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.gain = np.sqrt(3/4) * self.noise_sigma

    def test_imu(self):
        orientation = ahrs.filters.Madgwick(gyr=self.gyr, acc=self.Rg, gain=self.gain)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, orientation.Q)), self.noise_sigma*10)

    def test_marg(self):
        orientation = ahrs.filters.Madgwick(gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=self.gain)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, orientation.Q)), self.noise_sigma*10)

if __name__ == '__main__':
    unittest.main()

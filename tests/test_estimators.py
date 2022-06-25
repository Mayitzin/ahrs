#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

wmm = ahrs.utils.WMM(latitude=ahrs.MUNICH_LATITUDE, longitude=ahrs.MUNICH_LONGITUDE, height=ahrs.MUNICH_HEIGHT)
REFERENCE_MAGNETIC_VECTOR = np.array([wmm.X, wmm.Y, wmm.Z])
NORMAL_GRAVITY = ahrs.utils.WGS().normal_gravity(ahrs.MUNICH_LATITUDE, ahrs.MUNICH_HEIGHT)
REFERENCE_GRAVITY_VECTOR = np.array([0.0, 0.0, NORMAL_GRAVITY])

def __gaussian_filter(in_array: np.ndarray, size: int = 10, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian filter over an array

    This implementation tries to mimic the behavior of Scipy's function
    ``gaussian_filter1d``, in order to avoid the dependency on Scipy.

    Parameters
    ----------
    in_array : np.ndarray
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
    if in_array.ndim < 2:
        return np.correlate(in_array, phi_x, mode='same')
    return np.array([np.correlate(col, phi_x, mode='same') for col in in_array.T]).T

def random_angvel(num_samples: int = 500, max_rotations: int = 4, num_axes: int = 3, span: list = None, **kwargs) -> np.ndarray:
    """
    Random angular velocities

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

# Generate random attitudes
NUM_SAMPLES = 500
ANGULAR_VELOCITIES = random_angvel(num_samples=NUM_SAMPLES, span=(-np.pi, np.pi))
REFERENCE_QUATERNIONS = ahrs.QuaternionArray(ahrs.filters.AngularRate(ANGULAR_VELOCITIES).Q)
REFERENCE_ROTATIONS = REFERENCE_QUATERNIONS.to_DCM()
ACC_NOISE_STD_DEVIATION = np.linalg.norm(REFERENCE_GRAVITY_VECTOR)/100.0
MAG_NOISE_STD_DEVIATION = np.linalg.norm(REFERENCE_MAGNETIC_VECTOR)/100.0

class TestTRIAD(unittest.TestCase):
    def setUp(self) -> None:
        # Rotated reference vectors + noise
        self.Rg = np.array([R @ REFERENCE_GRAVITY_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * ACC_NOISE_STD_DEVIATION
        self.Rm = np.array([R @ REFERENCE_MAGNETIC_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * MAG_NOISE_STD_DEVIATION

    def test_multiple_values(self):
        R2 = ahrs.filters.TRIAD(self.Rg, self.Rm, v1=REFERENCE_GRAVITY_VECTOR, v2=REFERENCE_MAGNETIC_VECTOR)
        self.assertLess(np.nanmean(ahrs.utils.metrics.chordal(REFERENCE_ROTATIONS, R2.A)), 0.045)

    def test_wrong_frame(self):
        self.assertRaises(TypeError, ahrs.filters.TRIAD, frame=1.0)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, frame=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, frame='Up')

    def test_wrong_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.TRIAD, w1=1.0)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, w2=1.0)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, w1=1.0, w2=2.0)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, w1="[1., 2., 3.]", w2="[2., 3., 4.]")
        self.assertRaises(TypeError, ahrs.filters.TRIAD, w1=True)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, w2=True)
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1.0, 2.0], w2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=np.copy(1.0), w2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[2.0, 3.0, 4.0], w2=np.copy(1.0))
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=np.zeros(3), w2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[2.0, 3.0, 4.0], w2=np.zeros(3))
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1., 2., 3.], w2=[2., 3., 4.], v1=[1.0, 2.0], v2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1., 2., 3.], w2=[2., 3., 4.], v1=np.copy(1.0), v2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1., 2., 3.], w2=[2., 3., 4.], v1=[2.0, 3.0, 4.0], v2=np.copy(1.0))
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1., 2., 3.], w2=[2., 3., 4.], v1=np.zeros(3), v2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1., 2., 3.], w2=[2., 3., 4.], v1=[2.0, 3.0, 4.0], v2=np.zeros(3))

class TestSAAM(unittest.TestCase):
    def setUp(self) -> None:
        # Add noise to reference vectors and rotate them by the random attitudes
        self.Rg = np.array([R @ REFERENCE_GRAVITY_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * ACC_NOISE_STD_DEVIATION
        self.Rm = np.array([R @ REFERENCE_MAGNETIC_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * MAG_NOISE_STD_DEVIATION
        self.threshold = 7.5e-2

    def test_single_values(self):
        saam = ahrs.filters.SAAM(self.Rg[0], self.Rm[0])
        self.assertLess(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS[0], saam.Q), self.threshold*10)

    def test_single_values_as_rotation(self):
        saam = ahrs.filters.SAAM(self.Rg[0], self.Rm[0], representation='rotmat')
        self.assertLess(ahrs.utils.metrics.chordal(saam.A, REFERENCE_ROTATIONS[0]), self.threshold*10)

    def test_multiple_values(self):
        saam = ahrs.filters.SAAM(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, saam.Q)), self.threshold)

    def test_multiple_values_as_rotations(self):
        saam = ahrs.filters.SAAM(self.Rg, self.Rm, representation='rotmat')
        self.assertLess(np.nanmean(ahrs.utils.metrics.chordal(saam.A, REFERENCE_ROTATIONS)), self.threshold*10)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_representation(self):
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.Rg, mag=self.Rm, representation=1.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.Rg, mag=self.Rm, representation=['quaternion'])
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.Rg, mag=self.Rm, representation=None)
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=self.Rg, mag=self.Rm, representation='axisangle')
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=self.Rg, mag=self.Rm, representation='rpy')
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=self.Rg, mag=self.Rm, representation='DCM')
        self.assertRaises(AttributeError, getattr, ahrs.filters.SAAM(self.Rg, self.Rm, representation='quaternion'), 'A')

class TestFAMC(unittest.TestCase):
    def setUp(self) -> None:
        # Add noise to reference vectors and rotate them by the random attitudes
        self.Rg = np.array([R.T @ REFERENCE_GRAVITY_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * ACC_NOISE_STD_DEVIATION
        self.Rm = np.array([R.T @ REFERENCE_MAGNETIC_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * MAG_NOISE_STD_DEVIATION
        self.threshold = 7.5e-2

    def test_single_values(self):
        orientation = ahrs.filters.FAMC(self.Rg[0], self.Rm[0])
        self.assertLess(ahrs.utils.metrics.qad(orientation.Q, REFERENCE_QUATERNIONS[0]), self.threshold*10)

    def test_multiple_values(self):
        orientation = ahrs.filters.FAMC(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, REFERENCE_QUATERNIONS)), self.threshold)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.FAMC, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.FAMC, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_attribute_access(self):
        self.assertRaises(AttributeError, getattr, ahrs.filters.FAMC(self.Rg[0], self.Rm[0]), 'A')

class TestFLAE(unittest.TestCase):
    def setUp(self) -> None:
        # Add noise to reference vectors and rotate them by the random attitudes
        self.Rg = np.array([R.T @ REFERENCE_GRAVITY_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * ACC_NOISE_STD_DEVIATION
        self.Rm = np.array([R.T @ REFERENCE_MAGNETIC_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * MAG_NOISE_STD_DEVIATION
        self.threshold = 3.5e-2

    def test_multiple_values(self):
        orientation = ahrs.filters.FLAE(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, REFERENCE_QUATERNIONS)), self.threshold)

    def test_multiple_values_method_eig(self):
        orientation = ahrs.filters.FLAE(self.Rg, self.Rm, method='eig')
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, REFERENCE_QUATERNIONS)), self.threshold)

    def test_multiple_values_method_newton(self):
        orientation = ahrs.filters.FLAE(self.Rg, self.Rm, method='newton')
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, REFERENCE_QUATERNIONS)), self.threshold)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_method(self):
        self.assertRaises(TypeError, ahrs.filters.FLAE, self.Rg, self.Rm, method=1)
        self.assertRaises(TypeError, ahrs.filters.FLAE, self.Rg, self.Rm, method=3.14159)
        self.assertRaises(TypeError, ahrs.filters.FLAE, self.Rg, self.Rm, method=False)
        self.assertRaises(TypeError, ahrs.filters.FLAE, self.Rg, self.Rm, method=None)
        self.assertRaises(TypeError, ahrs.filters.FLAE, self.Rg, self.Rm, method=['symbolic'])
        self.assertRaises(TypeError, ahrs.filters.FLAE, self.Rg, self.Rm, method=('symbolic',))
        self.assertRaises(ValueError, ahrs.filters.FLAE, self.Rg, self.Rm, method='some_method')

class TestQUEST(unittest.TestCase):
    def setUp(self) -> None:
        # Add noise to reference vectors and rotate them by the random attitudes
        self.Rg = np.array([R.T @ REFERENCE_GRAVITY_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * ACC_NOISE_STD_DEVIATION
        self.Rm = np.array([R.T @ REFERENCE_MAGNETIC_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * MAG_NOISE_STD_DEVIATION
        self.threshold = 3.5e-2

    def test_multiple_values(self):
        quest = ahrs.filters.QUEST(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, quest.Q)), self.threshold)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.QUEST, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.QUEST, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_magnetic_dip(self):
        self.assertRaises(TypeError, ahrs.filters.QUEST, self.Rg, self.Rm, magnetic_dip='34.5')
        self.assertRaises(TypeError, ahrs.filters.QUEST, self.Rg, self.Rm, magnetic_dip=False)
        self.assertRaises(TypeError, ahrs.filters.QUEST, self.Rg, self.Rm, magnetic_dip=['34.5'])
        self.assertRaises(TypeError, ahrs.filters.QUEST, self.Rg, self.Rm, magnetic_dip=('34.5',))

class TestDavenport(unittest.TestCase):
    def setUp(self) -> None:
        # Add noise to reference vectors and rotate them by the random attitudes
        self.Rg = np.array([R.T @ REFERENCE_GRAVITY_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * ACC_NOISE_STD_DEVIATION
        self.Rm = np.array([R.T @ REFERENCE_MAGNETIC_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * MAG_NOISE_STD_DEVIATION
        self.threshold = 7.5e-2

    def test_single_values(self):
        orientation = ahrs.filters.Davenport(self.Rg[0], self.Rm[0])
        self.assertLess(ahrs.utils.metrics.qad(orientation.Q, REFERENCE_QUATERNIONS[0]), self.threshold)

    def test_multiple_values(self):
        orientation = ahrs.filters.Davenport(self.Rg, self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, REFERENCE_QUATERNIONS)), self.threshold)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Davenport, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Davenport, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

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

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, frequency=-100.0)

    def test_wrong_input_frame(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, frame=[100.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, frame=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, frame=True)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, frame='NWU')

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, Dt=-0.01)

    def test_wrong_input_alpha(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, alpha="1.0")
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, alpha=[1.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, alpha=(1.0,))
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, alpha=True)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, alpha=0.0)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, alpha=-1.0)

    def test_wrong_input_beta(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, beta="1.0")
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, beta=[1.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, beta=(1.0,))
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, beta=True)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, beta=0.0)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, beta=-1.0)

    def test_wrong_input_threshold(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, threshold="1.0")
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, threshold=[1.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, threshold=(1.0,))
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, threshold=True)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, threshold=0.0)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, threshold=-1.0)

    def test_wrong_input_adaptive(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, adaptive=1.0)
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, adaptive="1.0")
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, adaptive=[1.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.Rg, mag=self.Rm, adaptive=(1.0,))

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

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.FQA, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=self.Rg, mag=self.Rm, mag_ref=1.0)
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=self.Rg, mag=self.Rm, mag_ref="1.0")
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=self.Rg, mag=self.Rm, mag_ref=['1.0', '2.0', '3.0'])
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=self.Rg, mag=self.Rm, mag_ref=('1.0', '2.0', '3.0'))
        self.assertRaises(ValueError, ahrs.filters.FQA, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.FQA, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ahrs.filters.FQA, acc=self.Rg, mag=self.Rm, mag_ref=[1.0])
        self.assertRaises(ValueError, ahrs.filters.FQA, acc=self.Rg, mag=self.Rm, mag_ref=(1.0,))
        self.assertRaises(ValueError, ahrs.filters.FQA, acc=self.Rg, mag=self.Rm, mag_ref=np.zeros(3))

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

class TestMadgwick(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        a_ref = REFERENCE_GRAVITY_VECTOR
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

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=1.0, acc=self.Rg)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr="self.gyr", acc=self.Rg)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=True, acc=self.Rg)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc="self.Rg")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=True)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr[0], acc=self.Rg[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr[:2], acc=self.Rg, mag=self.Rm)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=['1.0', 2.0, 3.0], acc=self.Rg[0], mag=self.Rm[0])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=['1.0', '2.0', '3.0'], acc=self.Rg[0], mag=self.Rm[0])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr[0], acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr[0], acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr[0], acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr[0], acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=-100.0)

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=-0.01)

    def test_wrong_input_gain(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain="0.1")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain=True)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain="0.1")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=True)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain=-0.1)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=-0.1)

    def test_wrong_input_gain_imu(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain_imu="0.1")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain_imu=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain_imu=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain_imu=True)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain_imu="0.1")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain_imu=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain_imu=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain_imu=True)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain_imu=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain_imu=-0.1)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain_imu=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain_imu=-0.1)

    def test_wrong_input_gain_marg(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain_marg="0.1")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain_marg=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain_marg=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain_marg=True)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain_marg="0.1")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain_marg=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain_marg=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain_marg=True)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain_marg=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, gain_marg=-0.1)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain_marg=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain_marg=-0.1)

    def test_wrong_initial_quaternion(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=1)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=1.0)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=True)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=[1.0])
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=np.zeros(4))

class TestMahony(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        a_ref = REFERENCE_GRAVITY_VECTOR
        m_ref = ahrs.common.frames.ned2enu(REFERENCE_MAGNETIC_VECTOR)
        gyros = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(gyros).Q)
        rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.noise_sigma = 1e-2
        self.gyr = gyros + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rg = np.array([R.T @ a_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rm = np.array([R.T @ m_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma

    def test_imu(self):
        orientation = ahrs.filters.Mahony(gyr=self.gyr, acc=self.Rg)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, orientation.Q)), self.noise_sigma*10)

    def test_marg(self):
        orientation = ahrs.filters.Mahony(gyr=self.gyr, acc=self.Rg, mag=self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, orientation.Q)), self.noise_sigma*10)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=1.0, acc=self.Rg)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr="self.gyr", acc=self.Rg)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=True, acc=self.Rg)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc="self.Rg")
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=True)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr[0], acc=self.Rg[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr[:2], acc=self.Rg, mag=self.Rm)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=['1.0', 2.0, 3.0], acc=self.Rg[0], mag=self.Rm[0])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=['1.0', '2.0', '3.0'], acc=self.Rg[0], mag=self.Rm[0])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr[0], acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr[0], acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr[0], acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr[0], acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=-100.0)

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=-0.01)

    def test_wrong_input_kP(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, k_P="0.01")
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, k_P=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, k_P=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, k_P=True)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, k_P=0.0)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, k_P=-0.01)

    def test_wrong_input_kI(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, k_I="0.01")
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, k_I=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, k_I=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, k_I=True)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, k_I=0.0)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, k_I=-0.01)

    def test_wrong_initial_quaternion(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=1)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=1.0)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=True)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=[1.0])
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=np.zeros(4))

    def test_wrong_initial_bias(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, b0=1)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, b0=1.0)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, b0=True)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, b0="[0.0, 0.0, 0.0]")
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, b0=[0.0])
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyr, acc=self.Rg, mag=self.Rm, b0=np.zeros(4))

class TestFourati(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        a_ref = REFERENCE_GRAVITY_VECTOR
        m_ref = ahrs.common.frames.ned2enu(REFERENCE_MAGNETIC_VECTOR)
        gyros = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(gyros).Q)
        rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.noise_sigma = 1e-2
        self.gyr = gyros + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rg = np.array([R @ a_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rm = np.array([R @ m_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma

    def test_estimation(self):
        orientation = ahrs.filters.Fourati(gyr=self.gyr, acc=self.Rg, mag=self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, orientation.Q)), self.noise_sigma*20)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=1.0, acc=self.Rg, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr="self.gyr", acc=self.Rg, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=True, acc=self.Rg, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr[0], acc=self.Rg[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr[:2], acc=self.Rg, mag=self.Rm)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=['1.0', 2.0, 3.0], acc=self.Rg[0], mag=self.Rm[0])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=['1.0', '2.0', '3.0'], acc=self.Rg[0], mag=self.Rm[0])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr[0], acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr[0], acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr[0], acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr[0], acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=-100.0)

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=-0.01)

    def test_wrong_input_gain(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, gain="0.1")
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, gain=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, gain=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, gain=True)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain="0.1")
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=True)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, gain=0.0)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, gain=-0.1)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=0.0)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=-0.1)

    def test_wrong_magnetic_dip(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, magnetic_dip='34.5')
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, magnetic_dip=False)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, magnetic_dip=['34.5'])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, magnetic_dip=('34.5',))

    def test_wrong_initial_quaternion(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=1)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=1.0)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=True)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=['1.0', '0.0', '0.0', '0.0'])
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=[1.0])
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=np.zeros(4))

class TestEKF(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        a_ref = REFERENCE_GRAVITY_VECTOR
        m_ref = REFERENCE_MAGNETIC_VECTOR
        gyros = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(gyros).Q)
        rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.noise_sigma = 1e-2
        self.gyr = gyros + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rg = np.array([R.T @ a_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rm = np.array([R.T @ m_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma

    def test_gyr_acc(self):
        orientation = ahrs.filters.EKF(gyr=self.gyr, acc=self.Rg)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, orientation.Q)), self.noise_sigma*10.0)

    def test_gyr_acc_mag(self):
        orientation = ahrs.filters.EKF(gyr=self.gyr, acc=self.Rg, mag=self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, orientation.Q)), self.noise_sigma*10.0)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=1.0, acc=self.Rg)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr="self.gyr", acc=self.Rg)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=True, acc=self.Rg)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc="self.Rg")
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=True)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr[0], acc=self.Rg[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr[:2], acc=self.Rg, mag=self.Rm)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=np.zeros(3), mag=self.Rm)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=np.zeros(3))

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=['1.0', 2.0, 3.0], acc=self.Rg[0], mag=self.Rm[0])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=['1.0', '2.0', '3.0'], acc=self.Rg[0], mag=self.Rm[0])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr[0], acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr[0], acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr[0], acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr[0], acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=-100.0)

    def test_wrong_input_frame(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frame=1)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frame=1.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frame=True)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frame=['NED'])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frame=('NED',))
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frame='NWU')

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=-0.01)

    def test_wrong_initial_quaternion(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=1)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=1.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=True)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=[1.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=np.zeros(4))

    def test_wrong_magnetic_reference(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, magnetic_ref='34.5')
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, magnetic_ref=False)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, magnetic_ref=['34.5'])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, magnetic_ref=('34.5',))
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, magnetic_ref=[1.0, 2.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, magnetic_ref=[[1.0], [2.0], [3.0]])

    def test_wrong_initial_state_covariance_matrix(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, P=1)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, P=1.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, P=True)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, P="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, P=[1.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, P=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, P=np.eye(5, 4))

    def test_wrong_spectral_noises_array(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, noises=1)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, noises=1.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, noises=True)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, noises="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, noises=[1.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, noises=[1.0, 2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyr, acc=self.Rg, mag=self.Rm, noises=np.eye(5, 4))

class TestTilt(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        a_ref = REFERENCE_GRAVITY_VECTOR
        m_ref = REFERENCE_MAGNETIC_VECTOR
        gyros = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(gyros).Q)
        rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.noise_sigma = 1e-2
        self.gyr = gyros + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rg = np.array([R.T @ a_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rm = np.array([R.T @ m_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma

    def test_acc_mag_single_sample(self):
        orientation = ahrs.filters.Tilt(acc=self.Rg[0], mag=self.Rm[0])
        self.assertLess(ahrs.utils.metrics.qad(self.Qts[0], orientation.Q), self.noise_sigma*10.0)

    def test_acc_mag_multiple_samples(self):
        orientation = ahrs.filters.Tilt(acc=self.Rg, mag=self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, orientation.Q)), self.noise_sigma*10.0)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc="self.Rg")
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=True)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=self.Rg[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Tilt, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Tilt, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ahrs.filters.Tilt, acc=np.zeros(3), mag=self.Rm[0])
        self.assertRaises(ValueError, ahrs.filters.Tilt, acc=self.Rg[0], mag=np.zeros(3))

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_representation(self):
        self.assertRaises(TypeError, ahrs.filters.Tilt, representation=1)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=self.Rg, mag=self.Rm, representation=1)
        self.assertRaises(ValueError, ahrs.filters.Tilt, representation="some_representation")
        self.assertRaises(ValueError, ahrs.filters.Tilt, acc=self.Rg, mag=self.Rm, representation="some_representation")

class TestComplementary(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        a_ref = REFERENCE_GRAVITY_VECTOR
        m_ref = REFERENCE_MAGNETIC_VECTOR
        gyros = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(gyros).Q)
        rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.noise_sigma = 1e-2
        self.gyr = gyros + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rg = np.array([R.T @ a_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rm = np.array([R.T @ m_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma

    def test_gyr_acc_mag(self):
        orientation = ahrs.filters.Complementary(gyr=self.gyr, acc=self.Rg, mag=self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, orientation.Q)), 0.2)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=1.0, acc=self.Rg)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr="self.gyr", acc=self.Rg)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=True, acc=self.Rg)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc="self.Rg")
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=True)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr[0], acc=self.Rg[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr[:2], acc=self.Rg, mag=self.Rm)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr[0], acc=np.zeros(3), mag=self.Rm[0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr[0], acc=self.Rg[0], mag=np.zeros(3))

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=['1.0', 2.0, 3.0], acc=self.Rg[0], mag=self.Rm[0])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=['1.0', '2.0', '3.0'], acc=self.Rg[0], mag=self.Rm[0])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr[0], acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr[0], acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr[0], acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr[0], acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr[0], acc=self.Rg[0], mag=self.Rm[0], q0=['1.0', '0.0', '0.0', '0.0'])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr[0], acc=self.Rg[0], mag=self.Rm[0], q0=['1.0', 0.0, 0.0, 0.0])

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, frequency=-100.0)

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, Dt=-0.01)

    def test_wrong_input_gain(self):
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain="0.01")
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=True)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=0.0)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=-0.01)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, gain=1.01)

    def test_wrong_initial_quaternion(self):
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=1)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=1.0)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=True)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=['1.0', '0.0', '0.0', '0.0'])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=[1.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=[1.0, 2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=np.zeros(4))
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.gyr, acc=self.Rg, mag=self.Rm, q0=np.identity(4))

class TestOLEQ(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        a_ref = -REFERENCE_GRAVITY_VECTOR
        m_ref = np.array([ahrs.common.mathfuncs.sind(wmm.I), 0.0, ahrs.common.mathfuncs.cosd(wmm.I)])
        gyros = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(gyros).Q)
        rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.noise_sigma = 1e-2
        self.Rg = np.array([R.T @ a_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rm = np.array([R.T @ m_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma

    def test_estimation(self):
        orientation = ahrs.filters.OLEQ(acc=self.Rg, mag=self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, orientation.Q)), self.noise_sigma*10.0)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc="self.Rg")
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=True)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=np.zeros(3), mag=self.Rm)
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=np.zeros(3))

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_magnetic_reference(self):
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, magnetic_ref='34.5')
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, magnetic_ref=False)
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, magnetic_ref=['34.5'])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, magnetic_ref=('34.5',))
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, magnetic_ref=[1.0, 2.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, magnetic_ref=[0.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, magnetic_ref=[[1.0], [2.0], [3.0]])

    def test_wrong_input_frame(self):
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, frame=1)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, frame=1.0)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, frame=True)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, frame=['NED'])
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, frame=('NED',))
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, frame='NWU')

    def test_wrong_weights(self):
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights=1)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights=1.0)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights=True)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights="[1.0, 1.0]")
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights=['1.0', '1.0'])
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights=['1.0', 1.0])
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights=[1.0, '1.0'])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights=[[1.0], [1.0]])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights=[[1.0, 1.0]])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights=[1.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights=[0.5, -0.5])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights=[0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.Rg, mag=self.Rm, weights=np.zeros(4))

class TestROLEQ(unittest.TestCase):
    def setUp(self) -> None:
        # Create random attitudes
        num_samples = 1000
        a_ref = -REFERENCE_GRAVITY_VECTOR
        m_ref = np.array([ahrs.common.mathfuncs.sind(wmm.I), 0.0, ahrs.common.mathfuncs.cosd(wmm.I)])
        gyros = random_angvel(num_samples=num_samples, span=(-np.pi, np.pi))
        self.Qts = ahrs.QuaternionArray(ahrs.filters.AngularRate(gyros).Q)
        rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.noise_sigma = 1e-2
        self.gyros = gyros + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rg = np.array([R.T @ a_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma
        self.Rm = np.array([R.T @ m_ref for R in rotations]) + np.random.standard_normal((num_samples, 3)) * self.noise_sigma

    def test_estimation(self):
        orientation = ahrs.filters.ROLEQ(gyr=self.gyros, acc=self.Rg, mag=self.Rm)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.Qts, orientation.Q)), self.noise_sigma*10.0)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc="self.Rg")
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=True)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=1.0, mag=self.Rm)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc="self.Rg", mag="self.Rm")
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=['1.0', 2.0, 3.0], acc=self.Rg[0], mag=self.Rm[0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=['1.0', '2.0', '3.0'], acc=self.Rg[0], mag=self.Rm[0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros[0], acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros[0], acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros[0], acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros[0], acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros[0], acc=self.Rg[0], mag=self.Rm[0], q0=['1.0', '0.0', '0.0', '0.0'])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros[0], acc=self.Rg[0], mag=self.Rm[0], q0=['1.0', 0.0, 0.0, 0.0])

    def test_wrong_magnetic_reference(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, magnetic_ref='34.5')
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, magnetic_ref=False)
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, magnetic_ref=['34.5'])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, magnetic_ref=('34.5',))
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, magnetic_ref=[1.0, 2.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, magnetic_ref=[0.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, magnetic_ref=[[1.0], [2.0], [3.0]])

    def test_wrong_input_frame(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, frame=1)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, frame=1.0)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, frame=True)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, frame=['NED'])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, frame=('NED',))
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, frame='NWU')

    def test_wrong_weights(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights=1)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights=1.0)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights=True)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights="[1.0, 1.0]")
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights=['1.0', '1.0'])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights=['1.0', 1.0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights=[1.0, '1.0'])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights=[[1.0], [1.0]])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights=[[1.0, 1.0]])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights=[1.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights=[0.5, -0.5])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights=[0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, weights=np.zeros(4))

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, frequency=-100.0)

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, Dt=-0.01)

    def test_wrong_initial_quaternion(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, q0=1)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, q0=1.0)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, q0=True)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, q0="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, q0=['1.0', '0.0', '0.0', '0.0'])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, q0=['1.0', 0.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, q0=[1.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, q0=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyros, acc=self.Rg, mag=self.Rm, q0=np.zeros(4))

if __name__ == '__main__':
    unittest.main()

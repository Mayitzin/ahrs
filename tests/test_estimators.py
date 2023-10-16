#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

# Basic parameters
NUM_SAMPLES = 1000
SAMPLING_FREQUENCY = 100.0
THRESHOLD = 0.5

# Geomagnetic values
wmm = ahrs.utils.WMM(latitude=ahrs.MUNICH_LATITUDE, longitude=ahrs.MUNICH_LONGITUDE, height=ahrs.MUNICH_HEIGHT)
REFERENCE_MAGNETIC_VECTOR = wmm.geodetic_vector
MAG_NOISE_STD_DEVIATION = np.linalg.norm(REFERENCE_MAGNETIC_VECTOR) * 0.005

# Gravitational values
NORMAL_GRAVITY = ahrs.utils.WGS().normal_gravity(ahrs.MUNICH_LATITUDE, ahrs.MUNICH_HEIGHT*1000)
REFERENCE_GRAVITY_VECTOR = np.array([0.0, 0.0, NORMAL_GRAVITY])
ACC_NOISE_STD_DEVIATION = np.linalg.norm(REFERENCE_GRAVITY_VECTOR) * 0.01

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

def random_angpos(num_samples: int = 500, max_positions: int = 4, num_axes: int = 3, span: list = None, **kwargs) -> np.ndarray:
    """
    Random angular positions

    Create an array of synthetic random angular positions with reference to a
    local sensor coordinate frame.

    These angular positions are "simulated" by creating a random number of
    positions per axis, extend them for several samples, and then smoothing
    them with a gaussian filter.

    This creates smooth transitions between the different angular positions.

    Parameters
    ----------
    num_samples : int, default: 500
        Number of samples to generate. Set it to minimum 50, so that the
        gaussian filter can be applied.
    max_positions : int, default: 4
        Maximum number of rotations per axis.
    num_axes : int, default: 3
        Number of axes required.
    span : list or tuple, default: [-pi/2, pi/2]
        Span (minimum to maximum) of the random values.

    Returns
    -------
    angular_positions: np.ndarray
        M-by-3 Array of angular positions.
    """
    span = span if isinstance(span, (list, tuple)) else [-0.5*np.pi, 0.5*np.pi]
    all_angs = [np.random.uniform(span[0], span[1], np.random.randint(1, max_positions)) for _ in np.arange(num_axes)]
    angular_positions = np.zeros((num_samples, num_axes))
    for j, angs in enumerate(all_angs):
        # Create angular positions per axis
        num_angs = len(angs)
        idxs = np.sort(np.random.randint(0, num_samples, 2*num_angs)).reshape((num_angs, 2))
        for i, idx in enumerate(idxs):
            # Extend each angular position for several samples
            angular_positions[idx[0]:idx[1], j] = angs[i]
    smoothed_angular_positions = __gaussian_filter(angular_positions, size=kwargs.pop('gauss_size', 50 if num_samples > 50 else num_samples//5), sigma=5)
    return smoothed_angular_positions

def __centrifugal_force(angular_velocities: np.ndarray) -> np.ndarray:
    """
    Centrifugal force (EXPERIMENTAL)

    Compute the centrifugal force based on the angular velocities.

    Parameters
    ----------
    angular_velocities : np.ndarray
        Array of angular velocities.

    Returns
    -------
    centrifugal_force : np.ndarray
        Array of centrifugal forces.
    """
    Aa = np.zeros_like(angular_velocities)
    Ab = np.zeros_like(angular_velocities)
    Ac = np.zeros_like(angular_velocities)
    Aa[:, 0] = angular_velocities[:, 0]
    Ab[:, 1] = angular_velocities[:, 1]
    Ac[:, 2] = angular_velocities[:, 2]
    return np.c_[np.cross(Ab, Ac)[:, 0], np.cross(Aa, Ac)[:, 1], np.cross(Aa, Ab)[:, 2]]

class Sensors:
    """
    Generate synthetic sensor data of a hypothetical strapdown inertial
    navigation system.

    It generates data of a 9-DOF IMU (3-axes gyroscope, 3-axes accelerometer,
    and 3-axes magnetometer) from a given array of orientations as quaternions.

    The accelerometer data is given as m/s^2, the gyroscope data as deg/s, and
    the magnetometer data as nT.

    If no quaternions are provided, it generates random angular positions and
    computes the corresponding quaternions.

    The sensor data can be accessed as attributes of the object. For example,
    the gyroscope data can be accessed as ``sensors.gyroscopes``.

    Parameters
    ----------
    quaternions : ahrs.QuaternionArray, default: None
        Array of orientations as quaternions.
    num_samples : int, default: 500
        Number of samples to generate.
    freq : float, default: 100.0
        Sampling frequency, in Hz, of the data.
    in_degrees : bool, default: True
        If True, the gyroscope data is generated in degrees per second.
        Otherwise in radians per second.
    normalized_mag : bool, default: False
        If True, the magnetometer data is normalized to unit norm.
    reference_gravitational_vector : np.ndarray, default: None
        Reference gravitational vector. If None, it uses the default reference
        gravitational vector of ``ahrs.utils.WGS()``.
    reference_magnetic_vector : np.ndarray, default: None
        Reference magnetic vector. If None, it uses the default reference
        magnetic vector of ``ahrs.utils.WMM()``.
    gyr_noise : float
        Standard deviation of the gyroscope noise. If None given, it is
        generated from a normal distribution with zero mean. It is then scaled
        to be in the same units as the gyroscope data.
    acc_noise : float
        Standard deviation of the accelerometer noise. If None given, it is
        generated from a normal distribution with zero mean. It is then scaled
        to be in the same units as the accelerometer data.
    mag_noise : float
        Standard deviation of the magnetometer noise. If None given, it is
        generated from a normal distribution with zero mean. It is then scaled
        to be in the same units as the magnetometer data.

    Examples
    --------
    >>> sensors = Sensors(num_samples=1000)
    >>> sensors.gyroscopes.shape
    (1000, 3)
    >>> sensors.accelerometers.shape
    (1000, 3)
    >>> sensors.magnetometers.shape
    (1000, 3)
    >>> sensors.quaternions.shape
    (1000, 4)

    """
    def __init__(self, quaternions: ahrs.QuaternionArray = None, num_samples: int = 500, freq: float = SAMPLING_FREQUENCY, **kwargs):
        self.frequency = freq
        self.in_degrees = kwargs.get('in_degrees', True)
        self.normalized_mag = kwargs.get('normalized_mag', False)

        # Reference earth frames
        self.reference_gravitational_vector = kwargs.get('reference_gravitational_vector', REFERENCE_GRAVITY_VECTOR)
        self.reference_magnetic_vector = kwargs.get('reference_magnetic_vector', REFERENCE_MAGNETIC_VECTOR)

        # Spectral noise density
        self.gyr_noise = kwargs.get('gyr_noise', NOISE_SIGMA)
        self.acc_noise = kwargs.get('acc_noise', ACC_NOISE_STD_DEVIATION)
        self.mag_noise = kwargs.get('mag_noise', MAG_NOISE_STD_DEVIATION)

        # Orientations as quaternions
        if quaternions is None:
            self.num_samples = num_samples
            # Generate orientations (angular positions)
            self.ang_pos = random_angpos(num_samples=self.num_samples, span=(-np.pi, np.pi), max_positions=20)
            self.quaternions = ahrs.QuaternionArray(rpy=self.ang_pos)
            # Estimate angular velocities
            self.ang_vel = self.angular_velocities(self.ang_pos, self.frequency)
        else:
            # Define angular positions and velocities
            self.quaternions = ahrs.QuaternionArray(quaternions)
            self.num_samples = self.quaternions.shape[0]
            self.ang_pos = self.quaternions.to_angles()
            self.ang_vel = np.r_[np.zeros((1, 3)), self.quaternions.angular_velocities(1/self.frequency)]

        # Rotation Matrices
        self.rotations = self.quaternions.to_DCM()

        # Set empty arrays
        self.gyroscopes = None
        self.accelerometers = np.zeros((self.num_samples, 3))
        self.magnetometers = np.zeros((self.num_samples, 3))
        self.magnetometers_nd = np.zeros((self.num_samples, 3))
        self.magnetometers_enu = np.zeros((self.num_samples, 3))

        # Generate MARG data
        self.generate(self.rotations)

    def angular_velocities(self, angular_positions: np.ndarray, frequency: float) -> np.ndarray:
        """Compute angular velocities"""
        Qts = angular_positions if isinstance(angular_positions, ahrs.QuaternionArray) else ahrs.QuaternionArray(rpy=angular_positions)
        angvels = Qts.angular_velocities(1/frequency)
        return np.vstack((angvels[0], angvels))

    def generate(self, rotations: np.ndarray) -> None:
        """Compute synthetic data"""
        # Angular velocities measured in the local frame
        self.gyroscopes = np.copy(self.ang_vel) * ahrs.RAD2DEG
        # Add gyro biases: uniform random constant biases within 1/200th of the full range of the gyroscopes
        self.biases_gyroscopes = (np.random.default_rng().random(3)-0.5) * np.ptp(self.gyroscopes)/200
        self.gyroscopes += self.biases_gyroscopes

        # Accelerometers and magnetometers are measured w.r.t. global frame (inverse of the local frame)
        self.reference_magnetic_vector_nd = np.array([np.cos(wmm.I * ahrs.DEG2RAD), 0.0, np.sin(wmm.I * ahrs.DEG2RAD)])
        self.reference_magnetic_vector_enu = ahrs.common.frames.ned2enu(self.reference_magnetic_vector)
        for i in np.arange(self.num_samples):
            self.accelerometers[i] = rotations[i].T @ self.reference_gravitational_vector
            self.magnetometers[i] = rotations[i].T @ self.reference_magnetic_vector
            self.magnetometers_nd[i] = rotations[i].T @ self.reference_magnetic_vector_nd
            self.magnetometers_enu[i] = rotations[i].T @ self.reference_magnetic_vector_enu

        # # Add centrifugal force based on cross product of angular velocities
        # self.accelerometers -= __centrifugal_force(self.ang_vel)

        # Add noise
        if self.mag_noise < np.ptp(self.magnetometers):
            self.mag_noise = np.linalg.norm(REFERENCE_MAGNETIC_VECTOR) * 0.005
        self.gyroscopes += np.random.standard_normal((self.num_samples, 3)) * self.gyr_noise
        self.accelerometers += np.random.standard_normal((self.num_samples, 3)) * self.acc_noise
        self.magnetometers += np.random.standard_normal((self.num_samples, 3)) * self.mag_noise
        self.magnetometers_nd += np.random.standard_normal((self.num_samples, 3)) * self.mag_noise
        self.magnetometers_enu += np.random.standard_normal((self.num_samples, 3)) * self.mag_noise

        if not self.in_degrees:
            self.gyroscopes *= ahrs.DEG2RAD
            self.biases_gyroscopes *= ahrs.DEG2RAD
        if self.normalized_mag:
            self.magnetometers /= np.linalg.norm(self.magnetometers, axis=1, keepdims=True)
            self.magnetometers_nd /= np.linalg.norm(self.magnetometers_nd, axis=1, keepdims=True)
            self.magnetometers_enu /= np.linalg.norm(self.magnetometers_enu, axis=1, keepdims=True)

# Generate random attitudes
NOISE_SIGMA = abs(np.random.standard_normal(3) * 0.1) * ahrs.RAD2DEG
ANGULAR_POSITIONS = random_angpos(num_samples=NUM_SAMPLES, span=(-np.pi, np.pi), max_positions=20)
SENSOR_DATA = Sensors(num_samples=1000, in_degrees=False)
REFERENCE_QUATERNIONS = SENSOR_DATA.quaternions
REFERENCE_ROTATIONS = SENSOR_DATA.rotations

class TestTRIAD(unittest.TestCase):
    def setUp(self) -> None:
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_multiple_values(self):
        triad = ahrs.filters.TRIAD(self.accelerometers, self.magnetometers, v1=REFERENCE_GRAVITY_VECTOR, v2=REFERENCE_MAGNETIC_VECTOR)
        triad_rotations = np.transpose(triad.A, (0, 2, 1))
        self.assertLess(np.nanmean(ahrs.utils.metrics.chordal(REFERENCE_ROTATIONS, triad_rotations)), THRESHOLD)

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
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_single_values(self):
        saam_quaternion = ahrs.Quaternion(ahrs.filters.SAAM(self.accelerometers[0], self.magnetometers[0]).Q)
        self.assertLess(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS[0], saam_quaternion.conjugate), THRESHOLD)

    def test_single_values_as_rotation(self):
        saam_rotation = ahrs.filters.SAAM(self.accelerometers[0], self.magnetometers[0], representation='rotmat').A
        self.assertLess(ahrs.utils.metrics.chordal(REFERENCE_ROTATIONS[0], saam_rotation.T), THRESHOLD)

    def test_multiple_values(self):
        saam_quaternions = ahrs.QuaternionArray(ahrs.filters.SAAM(self.accelerometers, self.magnetometers).Q)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, saam_quaternions.conjugate())), THRESHOLD)

    def test_multiple_values_as_rotations(self):
        saam_rotations = ahrs.filters.SAAM(self.accelerometers, self.magnetometers, representation='rotmat').A
        saam_rotations = np.transpose(saam_rotations, (0, 2, 1))
        self.assertLess(np.nanmean(ahrs.utils.metrics.chordal(REFERENCE_ROTATIONS, saam_rotations)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc="self.accelerometers", mag="self.magnetometers")
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
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.accelerometers, mag=self.magnetometers, representation=1.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.accelerometers, mag=self.magnetometers, representation=['quaternion'])
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.accelerometers, mag=self.magnetometers, representation=None)
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=self.accelerometers, mag=self.magnetometers, representation='axisangle')
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=self.accelerometers, mag=self.magnetometers, representation='rpy')
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=self.accelerometers, mag=self.magnetometers, representation='DCM')
        self.assertRaises(AttributeError, getattr, ahrs.filters.SAAM(self.accelerometers, self.magnetometers, representation='quaternion'), 'A')

class TestFAMC(unittest.TestCase):
    def setUp(self) -> None:
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_single_values(self):
        quaternion_famc = ahrs.Quaternion(ahrs.filters.FAMC(self.accelerometers[0], self.magnetometers[0]).Q)
        self.assertLess(ahrs.utils.metrics.qad(quaternion_famc, REFERENCE_QUATERNIONS[0]), THRESHOLD)

    def test_multiple_values(self):
        quaternions_famc = ahrs.QuaternionArray(ahrs.filters.FAMC(self.accelerometers, self.magnetometers).Q)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(quaternions_famc, REFERENCE_QUATERNIONS)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc="self.accelerometers", mag="self.magnetometers")
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
        self.assertRaises(AttributeError, getattr, ahrs.filters.FAMC(self.accelerometers[0], self.magnetometers[0]), 'A')

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
        self.threshold = THRESHOLD

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
        # Add noise to reference vectors and rotate them by the random attitudes
        self.Rg = np.array([R @ REFERENCE_GRAVITY_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * ACC_NOISE_STD_DEVIATION
        self.Rm = np.array([R @ REFERENCE_MAGNETIC_VECTOR for R in REFERENCE_ROTATIONS]) + np.random.standard_normal((NUM_SAMPLES, 3)) * MAG_NOISE_STD_DEVIATION
        self.threshold = THRESHOLD

    def test_acc_mag(self):
        orientation = ahrs.filters.AQUA(acc=self.Rg, mag=self.Rm)
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
        self.Qts = REFERENCE_QUATERNIONS
        self.gyr = SENSOR_DATA.gyroscopes
        self.Rg = SENSOR_DATA.accelerometers
        self.Rm = SENSOR_DATA.magnetometers
        self.noise_sigma = np.nanmean(NOISE_SIGMA)
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
        self.frequency = 100.0
        self.a_ref = REFERENCE_GRAVITY_VECTOR
        self.m_ref = REFERENCE_MAGNETIC_VECTOR
        self.angular_positions = random_angpos(num_samples=NUM_SAMPLES, span=(-0.5*np.pi, 0.5*np.pi), max_positions=20)
        self.Qts = ahrs.QuaternionArray(rpy=self.angular_positions)
        angular_velocities = np.vstack((np.zeros(3), self.Qts.angular_velocities(1/self.frequency)))
        rotations = self.Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        self.noise_sigma = 4*np.pi/50.0
        self.gyr = angular_velocities + np.random.standard_normal((NUM_SAMPLES, 3)) * self.noise_sigma
        self.Rg = np.array([R.T @ self.a_ref for R in rotations]) + np.random.standard_normal((NUM_SAMPLES, 3)) * self.noise_sigma
        self.Rm = np.array([R.T @ self.m_ref for R in rotations]) + np.random.standard_normal((NUM_SAMPLES, 3)) * self.noise_sigma

    def test_gyr_acc(self):
        angular_positions = np.c_[self.angular_positions[:, :2], np.zeros(NUM_SAMPLES)]
        angular_positions[:, 2] = 0.0
        Qts = ahrs.QuaternionArray(rpy=angular_positions)
        angular_velocities = np.vstack((np.zeros(3), Qts.angular_velocities(1/self.frequency)))
        rotations = Qts.to_DCM()
        # Add noise to reference vectors and rotate them by the random attitudes
        gyr = angular_velocities + np.random.standard_normal((NUM_SAMPLES, 3)) * self.noise_sigma
        Rg = np.array([R.T @ self.a_ref for R in rotations]) + np.random.standard_normal((NUM_SAMPLES, 3)) * self.noise_sigma
        orientation = ahrs.filters.Complementary(gyr=gyr, acc=Rg)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(Qts, orientation.Q)), 0.2)

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

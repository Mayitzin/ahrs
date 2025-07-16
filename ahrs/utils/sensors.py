# -*- coding: utf-8 -*-
"""
Submodule for INS sensor models.

This module generates synthetic sensor data of a hypothetical strapdown
inertial navigation system (INS.) It mimics a 9-DOF IMU (3-axes gyroscope,
3-axes accelerometer, and 3-axes magnetometer) from a given array of
orientations as quaternions.

"""

import numpy as np
from ..common.constants import DEG2RAD
from ..common.constants import RAD2DEG
from ..common.constants import MUNICH_LATITUDE
from ..common.constants import MUNICH_LONGITUDE
from ..common.constants import MUNICH_HEIGHT
from ..common.frames import ned2enu
from ..common.quaternion import QuaternionArray
from ..utils import WMM
from ..utils import WGS

GENERATOR = np.random.default_rng(42)

# Geomagnetic values
wmm = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT)
REFERENCE_MAGNETIC_VECTOR = wmm.geodetic_vector

# Gravitational values
NORMAL_GRAVITY = WGS().normal_gravity(MUNICH_LATITUDE, MUNICH_HEIGHT)
REFERENCE_GRAVITY_VECTOR = np.array([0.0, 0.0, NORMAL_GRAVITY])

SAMPLING_FREQUENCY = 100.0

def __gaussian_filter(in_array: np.ndarray, size: int = 10, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian filter over an array

    This function tries to mimic the behavior of Scipy's function
    ``gaussian_filter1d``. It is re-implemented here, in order to avoid the
    dependency on Scipy.

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

def random_angpos(num_samples: int = 500, max_positions: int = 4, num_axes: int = 3, span: list = None, **kwargs) -> np.ndarray:
    """
    Random angular positions

    Create an array of synthetic random angular positions with respect to each
    axis of a three-dimensional global coordinate frame.

    These angular positions are "simulated" by creating a random number of
    positions per axis, extend them for several samples, and then smoothing
    them with a gaussian filter.

    This creates smooth transitions between the different angular positions.

    .. warning::

        It must be stressed, these are **ANGULAR** positions, not **linear**.
        They are measured in radians, and **do not** represent translations or
        positions in a three-dimensional space.

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
    rng : np.random.Generator, default: np.random.default_rng(42)
        Random number generator to use. If not given, it uses the default
        random number generator with seed 42.

    Returns
    -------
    angular_positions: np.ndarray
        M-by-3 Array of angular positions.
    """
    rng = kwargs.pop('rng', GENERATOR)
    span = span if isinstance(span, (list, tuple)) else [-0.5*np.pi, 0.5*np.pi]
    all_angs = [rng.uniform(span[0], span[1], rng.integers(1, max_positions)) for _ in np.arange(num_axes)]
    angular_positions = np.zeros((num_samples, num_axes))
    for j, angs in enumerate(all_angs):
        # Create angular positions per axis
        num_angs = len(angs)
        idxs = np.sort(rng.integers(0, num_samples, 2*num_angs)).reshape((num_angs, 2))
        for i, idx in enumerate(idxs):
            # Extend each angular position for several samples
            angular_positions[idx[0]:idx[1], j] = angs[i]
    smoothed_angular_positions = __gaussian_filter(angular_positions, size=kwargs.pop('gauss_size', 50 if num_samples > 50 else num_samples//5), sigma=5)
    return smoothed_angular_positions

class Sensors:
    """
    Generate synthetic sensor data of a hypothetical strapdown inertial
    navigation system.

    It generates data of a 9-DOF IMU (3-axis gyroscopes, 3-axis accelerometers,
    and 3-axis magnetometers) from a given array of orientations as quaternions.

    The accelerometer data is given as m/s^2, the gyroscope data as rad/s, and
    the magnetometer data as nT.

    If no quaternions are provided, it generates random angular positions and
    computes the corresponding quaternions.

    The sensor data can be accessed as attributes of the object. For example,
    the gyroscope data can be accessed as ``sensors.gyroscopes``.

    Simulating N observations, the most used attributes are:

    - ``gyroscopes``: N-by-3 array with gyroscope data, as rad/s.
    - ``accelerometers``: N-by-3 array with accelerometer data, as m/s^2.
    - ``magnetometers``: N-by-3 array with magnetometer data, as nT.
    - ``quaternions``: N-by-4 array with orientations as quaternions.
    - ``rotations``: N-by-3-by-3 array with orientations as 3x3 Rotation
      matrices.
    - ``angular_positions``: N-by-3 array with orientations as Euler angles
      (roll, pitch, yaw.)
    - ``ang_vel``: N-by-3 array with angular velocities around the X-, Y-, and
      Z-axes. Obtained from differentiation of the orientations.
    - ``frequency``: Sampling frequency of the data, in Hz.

    Parameters
    ----------
    quaternions : ahrs.QuaternionArray, default: None
        Array of orientations as quaternions.
    num_samples : int, default: 500
        Number of samples to generate.
    freq : float, default: 100.0
        Sampling frequency, in Hz, of the data.
    in_degrees : bool, default: False
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
    rng : np.random.Generator, default: np.random.default_rng(42)
        Random number generator to use. If not given, it uses the default
        random number generator with seed 42.

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
    def __init__(self, quaternions: QuaternionArray = None, num_samples: int = 500, freq: float = SAMPLING_FREQUENCY, **kwargs):
        self.frequency = freq
        self.in_degrees = kwargs.get('in_degrees', False)
        self.normalized_mag = kwargs.get('normalized_mag', False)
        self.rng = kwargs.pop('rng', GENERATOR)

        # Reference earth frames
        self.reference_gravitational_vector = kwargs.get('reference_gravitational_vector', REFERENCE_GRAVITY_VECTOR)
        self.reference_magnetic_vector = kwargs.get('reference_magnetic_vector', REFERENCE_MAGNETIC_VECTOR)

        # Spectral noise density
        self.gyr_noise_default_std_deviation = abs(self.rng.standard_normal(3) * 0.1) * RAD2DEG
        self.acc_noise_default_std_deviation = np.linalg.norm(REFERENCE_GRAVITY_VECTOR) * 0.01
        self.mag_noise_default_std_deviation = np.linalg.norm(REFERENCE_MAGNETIC_VECTOR) * 0.005
        self.gyr_noise = kwargs.get('gyr_noise', self.gyr_noise_default_std_deviation)
        self.acc_noise = kwargs.get('acc_noise', self.acc_noise_default_std_deviation)
        self.mag_noise = kwargs.get('mag_noise', self.mag_noise_default_std_deviation)

        # Orientations as quaternions
        if quaternions is None:
            # Orientations were NOT given
            self.num_samples = num_samples
            # Generate orientations (angular positions)
            self.angular_positions = random_angpos(num_samples=self.num_samples,
                                                   span=kwargs.get("span", (-np.pi, np.pi)),
                                                   max_positions=20,
                                                   rng=self.rng)
            if 'yaw' in kwargs:
                self.angular_positions[:, 2] = kwargs.get('yaw') * DEG2RAD
            self.quaternions = QuaternionArray(rpy=self.angular_positions)
            # Estimate angular velocities
            self.ang_vel = self.angular_velocities(self.angular_positions, self.frequency)
        else:
            # Orientations were given (as quaternions)
            # Define angular positions and velocities
            self.quaternions = QuaternionArray(quaternions)
            self.num_samples = self.quaternions.shape[0]
            self.angular_positions = self.quaternions.to_angles()
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
        Qts = angular_positions if isinstance(angular_positions, QuaternionArray) else QuaternionArray(rpy=angular_positions)
        angvels = Qts.angular_velocities(1/frequency)
        return np.vstack((angvels[0], angvels))

    def generate(self, rotations: np.ndarray) -> None:
        """Compute synthetic data"""
        # Angular velocities measured in the local frame
        self.gyroscopes = np.copy(self.ang_vel) * RAD2DEG

        # Add gyro biases: uniform random constant biases within 1/200th of the full range of the gyroscopes
        self.biases_gyroscopes = (self.rng.random(3)-0.5) * np.ptp(self.gyroscopes)/200
        if not self.in_degrees:
            self.biases_gyroscopes *= DEG2RAD
        self.gyroscopes += self.biases_gyroscopes

        # Accelerometers and magnetometers are measured w.r.t. global frame (inverse of the local frame)
        self.reference_magnetic_vector_nd = np.array([np.cos(wmm.I * DEG2RAD), 0.0, np.sin(wmm.I * DEG2RAD)])
        self.reference_magnetic_vector_enu = ned2enu(self.reference_magnetic_vector)
        for i in np.arange(self.num_samples):
            self.accelerometers[i] = rotations[i].T @ self.reference_gravitational_vector
            self.magnetometers[i] = rotations[i].T @ self.reference_magnetic_vector
            self.magnetometers_nd[i] = rotations[i].T @ self.reference_magnetic_vector_nd
            self.magnetometers_enu[i] = rotations[i].T @ self.reference_magnetic_vector_enu

        # Add noise
        if self.mag_noise < np.ptp(self.magnetometers):
            self.mag_noise = np.linalg.norm(REFERENCE_MAGNETIC_VECTOR) * 0.005
        self.gyroscopes += self.rng.standard_normal((self.num_samples, 3)) * self.gyr_noise
        self.accelerometers += self.rng.standard_normal((self.num_samples, 3)) * self.acc_noise
        self.magnetometers += self.rng.standard_normal((self.num_samples, 3)) * self.mag_noise
        self.magnetometers_nd += self.rng.standard_normal((self.num_samples, 3)) * self.mag_noise
        self.magnetometers_enu += self.rng.standard_normal((self.num_samples, 3)) * self.mag_noise

        if not self.in_degrees:
            self.gyroscopes *= DEG2RAD
            self.biases_gyroscopes *= DEG2RAD
        if self.normalized_mag:
            self.magnetometers /= np.linalg.norm(self.magnetometers, axis=1, keepdims=True)
            self.magnetometers_nd /= np.linalg.norm(self.magnetometers_nd, axis=1, keepdims=True)
            self.magnetometers_enu /= np.linalg.norm(self.magnetometers_enu, axis=1, keepdims=True)

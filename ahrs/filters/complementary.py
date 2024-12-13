# -*- coding: utf-8 -*-
"""
Attitude obtained with gyroscope and accelerometer-magnetometer measurements,
via complementary filter.

The complementary filter is one of the simplest ways to fuse sensor data from
multiple sensors. It is based on the idea that the errors from one sensor will
be compensated by the other sensor, and vice versa.

The gyroscopes tend to have a low-frequency drift, while the accelerometers
and magnetometers tend to have a high-frequency noise. The complementary filter
simple "combines" these two signals, yielding the benefit of eliminating the
drift from the gyroscope and noise from the accelerometer.

First, the **tilt** angles computed from the accelerometer measurements as:

.. math::
    \\boldsymbol{\\theta}_{am} =
    \\begin{bmatrix}
        \\theta_x \\\\ \\theta_y \\\\ \\theta_z
    \\end{bmatrix} =
    \\begin{bmatrix}
    \\mathrm{arctan2}(a_y, a_z) \\\\
    \\mathrm{arctan2}\\big(-a_x, \\sqrt{a_y^2+a_z^2}\\big) \\\\ 0
    \\end{bmatrix}

Only the roll, :math:`\\theta_x`, and pitch, :math:`\\theta_y`, angles are computed,
leaving the yaw angle, :math:`\\theta_z`, equal to zero.

If a magnetometer sample is available, the **yaw angle**, :math:`\\theta_z` can be
computed. First compensate the measured magnetic field using the tilt:

.. math::
    \\begin{array}{rcl}
    \\mathbf{b} &=&
    \\begin{bmatrix}
        \\cos\\theta_x & \\sin\\theta_x\\sin\\theta_y & \\sin\\theta_x\\cos\\theta_y \\\\
        0 & \\cos\\theta_y & -\\sin\\theta_y \\\\
        -\\sin\\theta_x & \\cos\\theta_x\\sin\\theta_y & \\cos\\theta_x\\cos\\theta_y
    \\end{bmatrix}
    \\begin{bmatrix}m_x \\\\ m_y \\\\ m_z\\end{bmatrix} \\\\
    \\begin{bmatrix}b_x \\\\ b_y \\\\ b_z\\end{bmatrix} &=&
    \\begin{bmatrix}
        m_x\\cos\\theta_x + m_y\\sin\\theta_x\\sin\\theta_y + m_z\\sin\\theta_x\\cos\\theta_y \\\\
        m_y\\cos\\theta_y - m_z\\sin\\theta_y \\\\
        -m_x\\sin\\theta_x + m_y\\cos\\theta_x\\sin\\theta_y + m_z\\cos\\theta_x\\cos\\theta_y
    \\end{bmatrix}
    \\end{array}

Then :math:`\\theta_z` is obtained as:

.. math::
    \\begin{array}{rcl}
    \\theta_z &=& \\mathrm{arctan2}(-b_y, b_x) \\\\
    &=& \\mathrm{arctan2}\\big(m_z\\sin\\theta_y - m_y\\cos\\theta_y, \\; m_x\\cos\\theta_x + \\sin\\theta_x(m_y\\sin\\theta_y + m_z\\cos\\theta_y)\\big)
    \\end{array}

Likewise, the orientation is again estimated at :math:`t`, but this time from a
previous orientation at :math:`t-1`, and a given angular velocity,
:math:`\\boldsymbol{\\omega}=\\begin{bmatrix}\\omega_x & \\omega_y & \\omega_z\\end{bmatrix}^T`,
in rad/s, using the simplest numerical integration:

.. math::
    \\boldsymbol{\\theta}_\\omega =
    \\begin{bmatrix}
    \\theta_{x_t} \\\\ \\theta_{y_t} \\\\ \\theta_{z_t}
    \\end{bmatrix} = 
    \\begin{bmatrix}
    \\theta_{x_{t-1}} + \\omega_x\\Delta_t \\\\
    \\theta_{y_{t-1}} + \\omega_y\\Delta_t \\\\
    \\theta_{z_{t-1}} + \\omega_z\\Delta_t
    \\end{bmatrix}

where :math:`\\Delta_t` is the time interval between the current and previous
measurements, a.k.a. the sampling period or time step. This is merely the
opposite of the sampling frequency: :math:`\\Delta_t=^1/f_s`.

Finally, the estimations are merged using a complementary filter with a
controlling parameter, :math:`\\alpha`, in the range :math:`[0, 1]`:

.. math::
    \\boldsymbol{\\theta} = \\alpha\\boldsymbol{\\theta}_\\omega + (1 - \\alpha)\\boldsymbol{\\theta}_{am}

where :math:`\\boldsymbol{\\theta}_\\omega` is the attitude estimated from the
gyroscope, :math:`\\boldsymbol{\\theta}_{am}` is the attitude estimated from
the accelerometer and the magnetometer, and :math:`\\alpha` is the gain of the
filter.

The filter gain must be a floating value within the range :math:`[0.0, 1.0]`.
It can be seen that when :math:`\\alpha=0`, the attitude is estimated entirely
with the accelerometer and the magnetometer. When :math:`\\alpha=1`, it is
estimated solely with the gyroscope. The values within the range decide how
much of each estimation is "blended" into the quaternion.

"""

import numpy as np
from ..common.quaternion import QuaternionArray
from ..utils.core import _assert_numerical_iterable

class Complementary:
    """
    Complementary filter for attitude estimation as quaternion.

    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity, in rad/s.
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration, in m/s^2.
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field, in mT.
    frequency : float, default: 100.0
        Sampling frequency in Herz.
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if ``frequency`` value is given.
    gain : float, default: 0.9
        Filter gain. Gain equal to 1 uses Angular velocities (gyroscopes) only.
        Gain equal to 0 uses Accelerometer, and Magnetometer if available,
        only. Values greater than zero and less than one blend the two
        estimations proportionally.
    w0 : numpy.ndarray, default: None
        Initial angular position, as roll-pitch-yaw angles, in radians.
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).
    representation : str, default: ``'angles'``
        Attitude representation. Options are ``'angles'``, ``'quaternion'`` or
        ``'rotmat'``.

    Raises
    ------
    ValueError
        When dimension of input arrays ``acc``, ``gyr``, or ``mag`` are not
        equal.

    """
    def __init__(self,
        gyr: np.ndarray = None,
        acc: np.ndarray = None,
        mag: np.ndarray = None,
        frequency: float = 100.0,
        gain: float = 0.9,
        **kwargs):
        self.gyr: np.ndarray = gyr
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.frequency: float = frequency
        self.Dt: float = kwargs.get('Dt', (1.0/self.frequency) if self.frequency else 0.01)
        self.gain: float = gain
        self.q0: np.ndarray = kwargs.get('q0')
        self.w0: np.ndarray = kwargs.get('w0')
        self._assert_validity_of_inputs()
        # Process of given data
        if self.gyr is not None and self.acc is not None:
            self.W = self._compute_all()

    def _assert_validity_of_inputs(self):
        """Asserts the validity of the inputs."""
        # Assert float values
        for item in ['frequency', 'Dt', 'gain']:
            if isinstance(self.__getattribute__(item), bool):
                raise TypeError(f"Parameter '{item}' must be numeric.")
            if not isinstance(self.__getattribute__(item), (int, float)):
                raise TypeError(f"Parameter '{item}' is not a non-zero number.")
            if item in ['frequency', 'Dt'] and self.__getattribute__(item) <= 0.0:
                raise ValueError(f"Parameter '{item}' must be a non-zero number.")
        if not (0.0 <= self.gain <= 1.0):
            raise ValueError(f"Filter gain must be in the range [0, 1]. Got {self.gain}")
        # Assert arrays
        for item in ['gyr', 'acc', 'mag', 'q0', 'w0']:
            if self.__getattribute__(item) is not None:
                _assert_numerical_iterable(self.__getattribute__(item), item)
                self.__setattr__(item, np.copy(self.__getattribute__(item)))
                if item == 'q0':
                    if self.q0.shape != (4,):
                        raise ValueError(f"Parameter 'q0' must be an array of shape (4,). It is {self.q0.shape}.")
                    if not np.allclose(np.linalg.norm(self.q0), 1.0):
                        raise ValueError(f"Parameter 'q0' must be a versor (norm equal to 1.0). Its norm is equal to {np.linalg.norm(self.q0)}.")
                else:
                    if self.__getattribute__(item).ndim > 2:
                        raise ValueError(f"Input '{item}' must be a one- or two-dimensional array.")
                    array_shape = self.__getattribute__(item).shape
                    if array_shape[-1] != 3:
                        raise ValueError(f"Input '{item}' must be of shape (3,) or (N, 3). Got {array_shape}.")

    def _compute_all(self) -> np.ndarray:
        """
        Estimate the quaternions given all data

        Attributes ``gyr``, ``acc`` and, optionally, ``mag`` must contain data.

        Returns
        -------
        W : numpy.ndarray
            M-by-3 Array with all estimated angles, where M is the number of
            samples.

        """
        if self.gyr.ndim < 2:
            raise ValueError(f"All inputs must have at least two observations.")
        if self.acc.shape != self.gyr.shape:
            raise ValueError(f"Could not operate on acc array of shape {self.acc.shape} and gyr array of shape {self.gyr.shape}.")
        W = np.zeros_like(self.acc)
        if self.mag is None:
            # Estimation with IMU only (Gyroscopes and Accelerometers)
            W2 = self.am_estimation(self.acc)
            W[0] = W2[0] if self.w0 is None else self.w0
        else:
            # Estimation with MARG (IMU and Magnetometer)
            if self.mag.shape != self.gyr.shape:
                raise ValueError(f"Could not operate on mag array of shape {self.mag.shape} and gyr array of shape {self.gyr.shape}.")
            W2 = self.am_estimation(self.acc, self.mag)
            W[0] = W2[0] if self.w0 is None else self.w0
        # Complemetary filter
        if self.mag is None:
            # Estimation with IMU only (Gyroscopes and Accelerometers)
            for i in range(1, len(W)):
                W[i, :2] = (W[i-1, :2] + self.gyr[i, :2]*self.Dt)*self.gain + W2[i, :2]*(1.0-self.gain)
            return W
        # Estimation with MARG (IMU and Magnetometer)
        for i in range(1, len(W)):
            W[i] = (W[i-1] + self.gyr[i]*self.Dt)*self.gain + W2[i]*(1.0-self.gain)
        return W

    def am_estimation(self, acc: np.ndarray, mag: np.ndarray = None) -> np.ndarray:
        """
        Attitude estimation from an Accelerometer-Magnetometer architecture.

        Parameters
        ----------
        acc : numpy.ndarray
            N-by-3 array with measurements of gravitational acceleration.
        mag : numpy.ndarray, default: None
            N-by-3 array with measurements of local geomagnetic field.

        Returns
        -------
        q_am : numpy.ndarray
            Estimated attitude.
        """
        acc = np.copy(acc)
        if acc.ndim < 1:
            raise ValueError(f"Input 'acc' must be a one- or two-dimensional array. Got shape {acc.shape}.")
        if acc.ndim < 2:
            # Estimation with one sample of a tri-axial accelerometer
            a_norm = np.linalg.norm(acc)
            if not a_norm > 0:
                raise ValueError("Gravitational acceleration must be non-zero")
            ax, ay, az = acc/a_norm
            # Tilt from Accelerometer
            ex = np.arctan2( ay, az)                        # Roll
            ey = np.arctan2(-ax, np.sqrt(ay**2 + az**2))    # Pitch
            ez = 0.0                                        # Yaw
            if mag is not None:
                _assert_numerical_iterable(mag, 'Geomagnetic field vector')
                if not(np.linalg.norm(mag) > 0):
                    raise ValueError("Magnetic field must be non-zero")
                mx, my, mz = mag/np.linalg.norm(mag)
                # Get tilted reference frame
                by = my*np.cos(ex) - mz*np.sin(ex)
                bx = mx*np.cos(ey) + np.sin(ey)*(my*np.sin(ex) + mz*np.cos(ex))
                ez = np.arctan2(-by, bx)
            return np.array([ex, ey, ez])
        # Estimation for 2-dimensional arrays
        angles = np.zeros_like(acc)   # Allocation of angles array
        # Estimate tilt angles
        a = acc/np.linalg.norm(acc, axis=1)[:, None]
        angles[:, 0] = np.arctan2(a[:, 1], a[:, 2])
        angles[:, 1] = np.arctan2(-a[:, 0], np.sqrt(a[:, 1]**2 + a[:, 2]**2))
        if mag is not None:
            # Estimate heading angle
            m_norm = np.linalg.norm(mag, axis=1)[:, None]
            if np.where(m_norm == 0)[0].size > 0:
                raise ValueError("All Magnetic field measurements must be non-zero.")
            m = mag/m_norm
            by = m[:, 1]*np.cos(angles[:, 0]) - m[:, 2]*np.sin(angles[:, 0])
            bx = m[:, 0]*np.cos(angles[:, 1]) + np.sin(angles[:, 1])*(m[:, 1]*np.sin(angles[:, 0]) + m[:, 2]*np.cos(angles[:, 0]))
            angles[:, 2] = np.arctan2(-by, bx)
        return angles

    @property
    def Q(self) -> np.ndarray:
        """
        Attitudes as quaternions.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number of
            samples.

        """
        if not hasattr(self, 'W'):
            raise ValueError("No data to perform estimation. Attitude is not computed.")
        return QuaternionArray().from_rpy(self.W)

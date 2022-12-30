# -*- coding: utf-8 -*-
"""
Complementary Filter
====================

Attitude quaternion obtained with gyroscope and accelerometer-magnetometer
measurements, via complementary filter.

First, the current orientation is estimated at time :math:`t`, from a previous
orientation at time :math:`t-1`, and a given angular velocity,
:math:`\\omega`, in rad/s.

This orientation is computed by numerically integrating the angular velocity
and adding it to the previous orientation, which is known as an **attitude
propagation**.

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}_\\omega &=& \\Big(\\mathbf{I}_4 + \\frac{\\Delta t}{2}\\boldsymbol\\Omega_t\\Big)\\mathbf{q}_{t-1} \\\\
    &=&
    \\begin{bmatrix}
    1 & -\\frac{\\Delta t}{2}\\omega_x & -\\frac{\\Delta t}{2}\\omega_y & -\\frac{\\Delta t}{2}\\omega_z \\\\
    \\frac{\\Delta t}{2}\\omega_x & 1 & \\frac{\\Delta t}{2}\\omega_z & -\\frac{\\Delta t}{2}\\omega_y \\\\
    \\frac{\\Delta t}{2}\\omega_y & -\\frac{\\Delta t}{2}\\omega_z & 1 & \\frac{\\Delta t}{2}\\omega_x \\\\
    \\frac{\\Delta t}{2}\\omega_z & \\frac{\\Delta t}{2}\\omega_y & -\\frac{\\Delta t}{2}\\omega_x & 1
    \\end{bmatrix}
    \\begin{bmatrix}q_w \\\\ q_x \\\\ q_y \\\\ q_z \\end{bmatrix} \\\\
    &=&
    \\begin{bmatrix}
        q_w - \\frac{\\Delta t}{2} \\omega_x q_x - \\frac{\\Delta t}{2} \\omega_y q_y - \\frac{\\Delta t}{2} \\omega_z q_z\\\\
        q_x + \\frac{\\Delta t}{2} \\omega_x q_w - \\frac{\\Delta t}{2} \\omega_y q_z + \\frac{\\Delta t}{2} \\omega_z q_y\\\\
        q_y + \\frac{\\Delta t}{2} \\omega_x q_z + \\frac{\\Delta t}{2} \\omega_y q_w - \\frac{\\Delta t}{2} \\omega_z q_x\\\\
        q_z - \\frac{\\Delta t}{2} \\omega_x q_y + \\frac{\\Delta t}{2} \\omega_y q_x + \\frac{\\Delta t}{2} \\omega_z q_w
    \\end{bmatrix}
    \\end{array}

Secondly, the *tilt* is computed from the accelerometer measurements as:

.. math::
    \\begin{array}{rcl}
    \\phi &=& \\mathrm{arctan2}(a_y, a_z) \\\\
    \\theta &=& \\mathrm{arctan2}\\big(-a_x, \\sqrt{a_y^2+a_z^2}\\big)
    \\end{array}

Only the pitch, :math:`\\theta`, and roll, :math:`\\phi`, angles are computed,
leaving the yaw angle, :math:`\\psi` equal to zero.

If a magnetometer sample is available, the yaw angle can be computed. First
compensate the measurement using the *tilt*:

.. math::
    \\begin{array}{rcl}
    \\mathbf{b} &=&
    \\begin{bmatrix}
        \\cos\\theta & \\sin\\theta\\sin\\phi & \\sin\\theta\\cos\\phi \\\\
        0 & \\cos\\phi & -\\sin\\phi \\\\
        -\\sin\\theta & \\cos\\theta\\sin\\phi & \\cos\\theta\\cos\\phi
    \\end{bmatrix}
    \\begin{bmatrix}m_x \\\\ m_y \\\\ m_z\\end{bmatrix} \\\\
    \\begin{bmatrix}b_x \\\\ b_y \\\\ b_z\\end{bmatrix} &=&
    \\begin{bmatrix}
        m_x\\cos\\theta + m_y\\sin\\theta\\sin\\phi + m_z\\sin\\theta\\cos\\phi \\\\
        m_y\\cos\\phi - m_z\\sin\\phi \\\\
        -m_x\\sin\\theta + m_y\\cos\\theta\\sin\\phi + m_z\\cos\\theta\\cos\\phi
    \\end{bmatrix}
    \\end{array}

Then, the yaw angle, :math:`\\psi`, is obtained as:

.. math::
    \\begin{array}{rcl}
    \\psi &=& \\mathrm{arctan2}(-b_y, b_x) \\\\
    &=& \\mathrm{arctan2}\\big(m_z\\sin\\phi - m_y\\cos\\phi, \\; m_x\\cos\\theta + \\sin\\theta(m_y\\sin\\phi + m_z\\cos\\phi)\\big)
    \\end{array}

We transform the roll-pitch-yaw angles to a quaternion representation:

.. math::
    \\mathbf{q}_{am} =
    \\begin{pmatrix}q_w\\\\q_x\\\\q_y\\\\q_z\\end{pmatrix} =
    \\begin{pmatrix}
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) + \\sin\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\sin\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) - \\cos\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) + \\sin\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) - \\sin\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big)
    \\end{pmatrix}

Finally, after each orientation is estimated independently, they are fused with
the complementary filter.

.. math::
    \\mathbf{q} = (1 - \\alpha) \\mathbf{q}_\\omega + \\alpha\\mathbf{q}_{am}

where :math:`\\mathbf{q}_\\omega` is the attitude estimated from the gyroscope,
:math:`\\mathbf{q}_{am}` is the attitude estimated from the accelerometer and
the magnetometer, and :math:`\\alpha` is the gain of the filter.

The filter gain must be a floating value within the range :math:`[0.0, 1.0]`.
It can be seen that when :math:`\\alpha=1`, the attitude is estimated entirely
with the accelerometer and the magnetometer. When :math:`\\alpha=0`, it is
estimated solely with the gyroscope. The values within the range decide how
much of each estimation is "blended" into the quaternion.

This is actually a simple implementation of `LERP
<https://en.wikipedia.org/wiki/Linear_interpolation>`_ commonly used to
linearly interpolate quaternions with small differences between them.

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
    gain : float, default: 0.1
        Filter gain.
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
        When dimension of input arrays ``acc``, ``gyr``, or ``mag`` are not equal.

    """
    def __init__(self,
        gyr: np.ndarray = None,
        acc: np.ndarray = None,
        mag: np.ndarray = None,
        frequency: float = 100.0,
        gain: float = 0.1,
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
        if self.acc.shape != self.gyr.shape:
            raise ValueError(f"Could not operate on acc array of shape {self.acc.shape} and gyr array of shape {self.gyr.shape}.")
        W = np.zeros_like(self.acc)
        W1 = self.angle_integration(W[0], self.gyr[1:], self.Dt)
        if self.mag is None:
            # Estimation with IMU only (Gyroscopes and Accelerometers)
            W[0] = self.am_estimation(self.acc[0]) if self.w0 is None else self.w0
            W2 = self.am_estimation(self.acc)
            W2[:, 2] = W1[:, 2].copy()      # Use yaw angle from integrated angular velocity
        else:
            # Estimation with MARG (IMU and Magnetometer)
            if self.mag.shape != self.gyr.shape:
                raise ValueError(f"Could not operate on mag array of shape {self.mag.shape} and gyr array of shape {self.gyr.shape}.")
            W[0] = self.am_estimation(self.acc[0], self.mag[0]) if self.w0 is None else self.w0
            W2 = self.am_estimation(self.acc, self.mag)
        # Complemetary filter
        W = W1*self.gain + W2*(1.0-self.gain)
        return np.unwrap(W, axis=0)         # Remove discontinuity of angles

    def angle_integration(self, w: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        if omega.ndim < 2:
            return w + omega * dt
        return np.cumsum(np.vstack((w, omega))*dt, axis=0)

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
        if acc.ndim < 2:
            a_norm = np.linalg.norm(acc)
            if not a_norm > 0:
                raise ValueError("Gravitational acceleration must be non-zero")
            ax, ay, az = acc/a_norm
            ### Tilt from Accelerometer
            ex = np.arctan2( ay, az)                        # Roll
            ey = np.arctan2(-ax, np.sqrt(ay**2 + az**2))    # Pitch
            ez = 0.0                                        # Yaw
            if mag is not None:
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
            m = mag/np.linalg.norm(mag, axis=1)[:, None]
            by = m[:, 1]*np.cos(angles[:, 0]) - m[:, 2]*np.sin(angles[:, 0])
            bx = m[:, 0]*np.cos(angles[:, 1]) + np.sin(angles[:, 1])*(m[:, 1]*np.sin(angles[:, 0]) + m[:, 2]*np.cos(angles[:, 0]))
            angles[:, 2] = np.arctan2(-by, bx)
        return angles

    @property
    def Q(self) -> np.ndarray:
        """
        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number of
            samples.

        """
        if not hasattr(self, 'W'):
            raise ValueError("No data to perform estimation. Attitude is not computed.")
        return QuaternionArray().from_rpy(self.W)

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
    \\theta &=& \\mathrm{arctan2}(a_y, a_z) \\\\
    \\phi &=& \\mathrm{arctan2}\\big(-a_x, \\sqrt{a_y^2+a_z^2}\\big)
    \\end{array}

Only the pitch, :math:`\\phi`, and roll, :math:`\\theta`, angles are computed,
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
from ..common.orientation import ecompass
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
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).

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
        gain: float = 0.9,
        **kwargs):
        self.gyr: np.ndarray = gyr
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.frequency: float = frequency
        self.Dt: float = kwargs.get('Dt', (1.0/self.frequency) if self.frequency else 0.01)
        self.gain: float = gain
        self.q0: np.ndarray = kwargs.get('q0')
        self._assert_validity_of_inputs()
        # Process of given data
        if self.gyr is not None and self.acc is not None:
            self.Q = self._compute_all()

    def _assert_validity_of_inputs(self):
        """Asserts the validity of the inputs."""
        # Assert float values
        for item in ['frequency', 'Dt', 'gain']:
            if isinstance(self.__getattribute__(item), bool):
                raise TypeError(f"Parameter '{item}' must be numeric.")
            if not isinstance(self.__getattribute__(item), (int, float)):
                raise TypeError(f"Parameter '{item}' is not a non-zero number.")
            if self.__getattribute__(item) <= 0.0:
                raise ValueError(f"Parameter '{item}' must be a non-zero number.")
        if self.gain > 1.0:
            raise ValueError(f"Filter gain must be in the range [0, 1]. Got {self.gain}")
        # Assert arrays
        for item in ['gyr', 'acc', 'mag', 'q0']:
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
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 4))
        if self.mag is None:
            self.mag = [None]*num_samples
        else:
            if self.mag.shape != self.gyr.shape:
                raise ValueError("mag and gyr are not the same size")
        Q[0] = self.am_estimation(self.acc[0], self.mag[0]) if self.q0 is None else self.q0.copy()
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def attitude_propagation(self, q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Attitude propagation of the orientation.

        Estimate the current orientation at time :math:`t`, from a given
        orientation at time :math:`t-1` and a given angular velocity,
        :math:`\\omega`, in rad/s.

        It is computed by numerically integrating the angular velocity and
        adding it to the previous orientation.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        omega : numpy.ndarray
            Tri-axial angular velocity, in rad/s.
        dt : float
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q_omega : numpy.ndarray
            Estimated orientation, as quaternion.
        """
        w = -0.5*dt*omega
        A = np.array([
            [1.0,  -w[0], -w[1], -w[2]],
            [w[0],   1.0,  w[2], -w[1]],
            [w[1], -w[2],   1.0,  w[0]],
            [w[2],  w[1], -w[0],   1.0]])
        q_omega = A @ q
        return q_omega / np.linalg.norm(q_omega)

    def am_estimation(self, acc: np.ndarray, mag: np.ndarray = None) -> np.ndarray:
        """
        Attitude estimation from an Accelerometer-Magnetometer architecture.

        Parameters
        ----------
        acc : numpy.ndarray
            Tri-axial sample of the accelerometer.
        mag : numpy.ndarray, default: None
            Tri-axial sample of the magnetometer.

        Returns
        -------
        q_am : numpy.ndarray
            Estimated attitude.
        """
        return ecompass(acc, mag, frame='NED', representation='quaternion')

    def update(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray = None, dt: float = None) -> np.ndarray:
        """
        Attitude Estimation from given measurements and previous orientation.

        The new orientation is first estimated with the angular velocity, then
        another orientation is computed using the accelerometers and
        magnetometers. The magnetometer is optional.

        Each orientation is estimated independently and fused with a
        complementary filter.

        .. math::
            \\mathbf{q} = (1 - \\alpha) \\mathbf{q}_\\omega + \\alpha\\mathbf{q}_{am}

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2.
        mag : numpy.ndarray, default: None
            Sample of tri-axial Magnetometer in uT.
        dt : float, default: None
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        """
        dt = self.Dt if dt is None else dt
        if gyr is None or not np.linalg.norm(gyr) > 0:
            return q
        q_omega = self.attitude_propagation(q, gyr, dt)
        q_am = self.am_estimation(acc, mag)
        # Complementary Estimation
        if np.linalg.norm(q_omega + q_am) < np.sqrt(2):
            q = (1.0 - self.gain)*q_omega - self.gain*q_am
        else:
            q = (1.0 - self.gain)*q_omega + self.gain*q_am
        return q/np.linalg.norm(q)

# -*- coding: utf-8 -*-
"""
Mahony Orientation Filter
=========================

The filter designed by Robert Mahony [1]_ is formulated as a deterministic
observer in SO(3) mainly driven by angular velocity measurements and
reconstructed attitude.

This observer, termed "explicit complementary filter" (ECF), uses an inertial
measurement :math:`a` and an angular velocity measurement :math:`\\omega`. The
inertial direction obtained from the gravity is a low-frequency normalized
measurement:

.. math::

    a = \\frac{a}{\\|a\\|}

A predicted direction of gravity :math:`v` is expected to be colinear with the
Z-axis of the inertial frame:

.. math::

    \\begin{eqnarray}
    v & = & R(q)^T \\begin{bmatrix}0 & 0 & 1 \\end{bmatrix}^T\\\\
    & = &
    \\begin{bmatrix}
    1 - 2(q_y^2 + q_z^2) & 2(q_xq_y + q_wq_z) & 2(q_xq_z - q_wq_y) \\
    2(q_xq_y - q_wq_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_wq_x + q_yq_z) \\
    2(q_xq_z + q_wq_y) & 2(q_yq_z - q_wq_x) & 1 - 2(q_x^2 + q_y^2)
    \\end{bmatrix}
    \\begin{bmatrix}0 \\ 0 \\ 1\\end{bmatrix} \\\\
    & = &
    \\begin{bmatrix}
    2(q_xq_z - q_wq_y) \\ 2(q_wq_x + q_yq_z) \\ 1 - 2(q_x^2 + q_y^2)
    \\end{bmatrix}
    \\end{eqnarray}

Considering the basic model of a gyroscope :math:`g`:

.. math::

    g = \\omega + b + \\mu

where :math:`\\omega` is the real angular velocity, :math:`b` is a varying
deterministic bias, and :math:`\\mu` is a Gaussian noise.

This implementation is based on simplifications by Mark Euston [2]_ and Tarek
Hamel [3]_ for low-cost inertial measurement units in UAVs.

References
----------
.. [1] Robert Mahony, Tarek Hamel, and Jean-Michel Pflimlin. Non-linear
   complementary filters on the special orthogonal group. IEEE Trans-actions
   on Automatic Control, Institute of Electrical and Electronics Engineers,
   2008, 53 (5), pp.1203-1217.
   (https://hal.archives-ouvertes.fr/hal-00488376/document)
.. [2] Mark Euston, Paul W. Coote, Robert E. Mahony, Jonghyuk Kim, and Tarek
   Hamel. A complementary filter for attitude estimation of a fixed-wing UAV.
   IEEE/RSJ International Conference on Intelligent Robots and Systems,
   340-345. 2008.
   (http://users.cecs.anu.edu.au/~Jonghyuk.Kim/pdf/2008_Euston_iros_v1.04.pdf)
.. [3] Tarek Hamel and Robert Mahony. Attitude estimation on SO(3) based on
   direct inertial measurements. IEEE International Conference on Robotics and
   Automation. ICRA 2006. pp. 2170-2175
   (http://users.cecs.anu.edu.au/~Robert.Mahony/Mahony_Robert/2006_MahHamPfl-C68.pdf)

"""

import numpy as np
from ..common.orientation import q_prod, q_conj, acc2q, am2q, q2R

class Mahony:
    """Mahony's Nonlinear Complementary Filter on SO(3)

    Attributes
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with N gyroscope samples
    acc : numpy.ndarray, default: None
        N-by-3 array with N accelerometer samples
    mag : numpy.ndarray, default: None
        N-by-3 array with N magnetometer samples
    frequency : float
        Sampling frequency in Herz
    Dt : float
        Sampling step in seconds. Inverse of sampling frequency.
    Kp : float
        Proportional filter gain
    Ki : float
        Integral filter gain
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion)

    Methods
    -------
    updateIMU(q, gyr, acc)
        Update given orientation using a gyroscope and an accelerometer sample
    updateMARG(q, gyr, acc, mag)
        Update given orientation using a gyroscope, an accelerometer, and a
        magnetometer gyroscope sample

    Parameters
    ----------
    acc : numpy.ndarray
        N-by-3 array with measurements of acceleration in m/s^2
    gyr : numpy.ndarray
        N-by-3 array with measurements of angular velocity in rad/s
    mag : numpy.ndarray
        N-by-3 array with measurements of magnetic field in mT

    Extra Parameters
    ----------------
    frequency : float, default: 100.0
        Sampling frequency in Herz
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if `frequency` value is given
    Kp : float, default: 1.0
        Proportional filter gain
    Ki : float, default: 0.0
        Integral filter gain
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion).

    Raises
    ------
    ValueError
        When dimension of input array(s) `acc`, `gyr`, or `mag` are not equal.

    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kw):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.frequency = kw.get('frequency', 100.0)
        self.Dt = kw.get('Dt', 1.0/self.frequency)
        self.Kp = kw.get('Kp', 1.0)
        self.Ki = kw.get('Ki', 0.0)
        self.q0 = kw.get('q0')
        self.eInt = np.zeros(3)
        if self.gyr is not None and self.acc is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """Estimate all quaternions with given sensor values"""
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        num_samples = len(self.gyr)
        Q = np.zeros((num_samples, 4))
        # Compute with IMU Architecture
        if self.mag is None:
            Q[0] = acc2q(self.acc[0]) if self.q0 is None else self.q0.copy()
            for t in range(1, num_samples):
                Q[t] = self.updateIMU(Q[t-1], self.gyr[t], self.acc[t])
            return Q
        # Compute with MARG Architecture
        if self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        Q[0] = am2q(self.acc[0], self.mag[0]) if self.q0 is None else self.q0.copy()
        for t in range(1, num_samples):
            Q[t] = self.updateMARG(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def updateIMU(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """Quaternion Estimation with a IMU architecture.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        g = gyr.copy()
        a_norm = np.linalg.norm(acc)
        if a_norm>0:
            v = q2R(q).T@np.array([0.0, 0.0, 1.0])      # Expected Earth's gravity
            e = np.cross(acc/a_norm, v)                 # Difference between expected and measured acceleration (Error)
            self.eInt += e*self.Dt                      # Integrate error
            b = -self.Ki*self.eInt                      # Estimated Gyro bias (eq. 48c)
            d = self.Kp*e + b                           # Innovation
            g += d                                      # Gyro correction
        qDot = 0.5*q_prod(q, [0.0, *g])                 # Rate of change of quaternion (eq. 48b)
        q += qDot*self.Dt                               # Update orientation
        q /= np.linalg.norm(q)
        return q

    def updateMARG(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """Quaternion Estimation with a MARG architecture.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in radians.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in T

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        g = gyr.copy()
        a_norm = np.linalg.norm(acc)
        if a_norm>0:
            m = mag.copy()
            m_norm = np.linalg.norm(m)
            if not m_norm>0:
                return self.updateIMU(q, gyr, acc)
            m /= m_norm
            v = q2R(q).T@np.array([0.0, 0.0, 1.0])              # Expected Earth's gravity
            h = q_prod(q, q_prod([0, *m], q_conj(q)))           # Rotate magnetic measurements to inertial frame
            w = q2R(q).T@np.array([np.sqrt(h[1]**2+h[2]**2), 0.0, h[3]])     # Expected Earth's magnetic field
            e = np.cross(acc/a_norm, v) + np.cross(m, w)        # Difference between expected and measured values
            self.eInt += e*self.Dt                              # Add error
            b = -self.Ki*self.eInt                              # Estimated Gyro bias (eq. 48c)
            g = g - b + self.Kp*e                               # Gyro correction
        qDot = 0.5*q_prod(q, [0.0, *g])                         # Rate of change of quaternion (eq. 48b)
        q += qDot*self.Dt                                       # Update orientation
        q /= np.linalg.norm(q)
        return q

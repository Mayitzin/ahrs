# -*- coding: utf-8 -*-
"""
Mahony Algorithm as proposed by R. Mahony et al [1]_ in 2010.

This implementation is based in the one made by S. Madgwick.

References
----------
.. [1] Nonlinear Complementary Filters on the Special Orthogonal Group; R.
   Mahony et al. 2010. (https://hal.archives-ouvertes.fr/hal-00488376/document)

"""

import numpy as np
from ahrs.common.orientation import *

class Mahony:
    """
    Class of Mahony algorithm

    Parameters
    ----------
    Kp : float
        Proportional filter gain.
    Ki : float
        Integral filter gain.
    samplePeriod : float
        Sampling rate in seconds. Inverse of sampling frequency.

    """
    def __init__(self, *args, **kwargs):
        self.Kp = kwargs.get('Kp', 1.0)
        self.Ki = kwargs.get('Ki', 0.0)
        self.samplePeriod = kwargs.get('samplePeriod', 1.0/256.0)
        # Integral Error
        self.eInt = np.array([0.0, 0.0, 0.0])

    def updateIMU(self, gyr, acc, q):
        """
        Mahony's AHRS algorithm with an IMU architecture.

        Adapted to Python from original implementation by Sebastian Madgwick.

        Parameters
        ----------
        gyr : array
            Sample of tri-axial Gyroscope in radians.
        acc : array
            Sample of tri-axial Accelerometer.
        q : array
            A-priori quaternion.

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        g = gyr.copy()
        a = acc.copy()
        # Normalise accelerometer measurement
        a_norm = np.linalg.norm(a)
        if a_norm == 0:     # handle NaN
            return q
        a /= a_norm
        # Assert values
        q /= np.linalg.norm(q)
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        # Estimated direction of gravity and magnetic flux
        v = np.array([2.0*(qx*qz - qw*qy),
                    2.0*(qw*qx + qy*qz),
                    qw**2 - qx**2 - qy**2 + qz**2])
        e = np.cross(a, v)
        self.eInt = self.eInt + e*self.samplePeriod if self.Ki > 0 else np.array([0.0, 0.0, 0.0])
        # Apply feedback term
        g += self.Kp*e + self.Ki*self.eInt
        # Compute rate of change of quaternion
        qDot = 0.5*q_prod(q, [0.0, g[0], g[1], g[2]])
        # Integrate to yield Quaternion
        q += qDot*self.samplePeriod
        q /= np.linalg.norm(q)
        return q

    def updateMARG(self, gyr, acc, mag, q):
        """
        Mahony's AHRS algorithm with a MARG architecture.

        Adapted to Python from original implementation by Sebastian Madgwick.

        Parameters
        ----------
        gyr : array
            Sample of tri-axial Gyroscope in radians.
        acc : array
            Sample of tri-axial Accelerometer.
        mag : array
            Sample of tri-axial Magnetometer.
        q : array
            A-priori quaternion.

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        g = gyr.copy()
        a = acc.copy()
        m = mag.copy()
        # Normalise accelerometer measurement
        a_norm = np.linalg.norm(a)
        if a_norm == 0:     # handle NaN
            return q
        a /= a_norm
        # Normalise magnetometer measurement
        m_norm = np.linalg.norm(m)
        if m_norm == 0:     # handle NaN
            return q
        m /= m_norm
        # Assert values
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        # Reference direction of Earth's magnetic feild
        h = q_prod(q, q_prod([0, m[0], m[1], m[2]], q_conj(q)))
        b = [0.0, np.linalg.norm([h[1], h[2]]), 0.0, h[3]]
        # Estimated direction of gravity and magnetic flux
        v = np.array([2.0*(qx*qz - qw*qy),
                      2.0*(qw*qx + qy*qz),
                      qw**2 - qx**2 - qy**2 + qz**2])
        w = np.array([b[1]*(0.5 - qy**2 - qz**2) + b[3]*(qx*qz - qw*qy),
                      b[1]*(qx*qy - qw*qz)       + b[3]*(qw*qx + qy*qz),
                      b[1]*(qw*qy + qx*qz)       + b[3]*(0.5 - qx**2 - qy**2)])
        # Error is sum of cross product between estimated direction and measured direction of fields
        e = np.cross(a, v) + np.cross(m, 2.0*w)
        self.eInt = self.eInt + e*self.samplePeriod if self.Ki > 0 else np.array([0.0, 0.0, 0.0])
        # Apply feedback term
        g += self.Kp*e + self.Ki*self.eInt
        # Compute rate of change of quaternion
        qDot = 0.5*q_prod(q, [0.0, g[0], g[1], g[2]])
        # Integrate to yield Quaternion
        q += qDot*self.samplePeriod
        q /= np.linalg.norm(q)
        return q

# -*- coding: utf-8 -*-
"""
Madgwick Algorithm

References
----------
.. [1] http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/

"""

import numpy as np
from ahrs.common.orientation import *

class Madgwick:
    """
    Class of Madgwick filter algorithm

    Parameters
    ----------
    beta : float
        Filter gain of a quaternion derivative.
    samplePeriod : float
        Sampling rate in seconds. Inverse of sampling frequency.

    """
    def __init__(self, *args, **kwargs):
        self.beta = kwargs.get('beta', 0.1)
        self.frequency = kwargs.get('frequency', 256.0)
        self.samplePeriod = kwargs.get('samplePeriod', 1.0/self.frequency)

    def updateIMU(self, gyr, acc, q):
        """
        Madgwick's AHRS algorithm with an IMU architecture.

        Adapted to Python from original implementation by Sebastian Madgwick.

        Parameters
        ----------
        g : array
            Sample of tri-axial Gyroscope in radians.
        a : array
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
        # Gradient decent algorithm corrective step
        F = np.array([2.0*(qx*qz - qw*qy)   - a[0],
                      2.0*(qw*qx + qy*qz)   - a[1],
                      2.0*(0.5-qx**2-qy**2) - a[2]])
        J = np.array([[-qy, qz, -qw, qx],
                      [ qx, qw,  qz, qy],
                      [ 0.0, -2.0*qx, -2.0*qy, 0.0]])
        step = 2.0*J.T@F
        step /= np.linalg.norm(step)
        # Compute rate of change of quaternion
        qDot = 0.5 * q_prod(q, [0, g[0], g[1], g[2]]) - self.beta * step.T
        # Integrate to yield Quaternion
        q += qDot*self.samplePeriod
        q /= np.linalg.norm(q)
        return q

    def updateMARG(self, gyr, acc, mag, q):
        """
        Madgwick's AHRS algorithm with a MARG architecture.

        Adapted to Python from original implementation by Sebastian Madgwick.

        Parameters
        ----------
        g : array
            Sample of tri-axial Gyroscope in radians.
        a : array
            Sample of tri-axial Accelerometer.
        m : array
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
        q /= np.linalg.norm(q)
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        # Reference direction of Earth's magnetic field
        h = q_prod(q, q_prod([0, m[0], m[1], m[2]], q_conj(q)))
        b = [0.0, np.linalg.norm([h[1], h[2]]), 0.0, h[3]]
        # Gradient decent algorithm corrective step
        F = np.array([(qx * qz - qw * qy)   - 0.5*a[0],
                      (qw * qx + qy * qz)   - 0.5*a[1],
                      (0.5 - qx**2 - qy**2) - 0.5*a[2],
                      b[1]*(0.5 - qy**2 - qz**2) + b[3]*(qx*qz - qw*qy)       - 0.5*m[0],
                      b[1]*(qx*qy - qw*qz)       + b[3]*(qw*qx + qy*qz)       - 0.5*m[1],
                      b[1]*(qw*qy + qx*qz)       + b[3]*(0.5 - qx**2 - qy**2) - 0.5*m[2]])
        J = np.array([[-qy,               qz,                  -qw,                    qx],
                    [ qx,                 qw,                   qz,                    qy],
                    [ 0.0,               -2.0*qx,              -2.0*qy,                0.0],
                    [-b[3]*qy,            b[3]*qz,             -2.0*b[1]*qy-b[3]*qw,  -2.0*b[1]*qz+b[3]*qx],
                    [-b[1]*qz+2*b[3]*qx,  b[1]*qy+b[3]*qw,      b[1]*qx+b[3]*qz,      -b[1]*qw+b[3]*qy],
                    [ b[1]*qy,            b[1]*qz-2.0*b[3]*qx,  b[1]*qw-2.0*b[3]*qy,   b[1]*qx]])
        step = 4.0*J.T@F
        step /= np.linalg.norm(step)    # normalise step magnitude
        # Compute rate of change of quaternion
        qDot = 0.5 * q_prod(q, [0, g[0], g[1], g[2]]) - self.beta * step.T
        # Integrate to yield quaternion
        q += qDot*self.samplePeriod
        q /= np.linalg.norm(q) # normalise quaternion
        return q


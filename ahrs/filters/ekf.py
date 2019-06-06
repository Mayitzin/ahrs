# -*- coding: utf-8 -*-
"""
** UNDER CONSTRUCTION **

References
----------
.. [1] Simo Särkkä (2013). Bayesian Filtering and Smoothing. Cambridge University Press.
  https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
.. [2] Wikipedia: Extended Kalman Filter.
  https://en.wikipedia.org/wiki/Extended_Kalman_filter
.. [3] Yan-Bin Jia (2018). Quaternions.
  http://web.cs.iastate.edu/~cs577/handouts/quaternion.pdf
.. [4] Thibaud Michel (2016). On Attitude Estimation with Smartphones.
  http://tyrex.inria.fr/mobile/benchmarks-attitude/

"""

import numpy as np
from ahrs.common.orientation import *
from ahrs.common.mathfuncs import *

class EKF:
    """
    Class of Extended Kalman Filter

    Parameters
    ----------
    samplePeriod : float
        Sampling rate in seconds. Inverse of sampling frequency.
    noises : array
        List of noise variance for each type of sensor, whose order is:

        [gyroscope, accelerometer, magnetometer]

        Default values: [0.3^2, 0.5^2, 0.8^2]

    """
    def __init__(self, *args, **kwargs):
        self.samplePeriod = kwargs['samplePeriod'] if 'samplePeriod' in kwargs else 1.0/256.0
        self.noises = kwargs['noises'] if 'noises' in kwargs else [0.3**2, 0.5**2, 0.8**2]
        self.g_noise = self.noises[0]*np.identity(3)
        self.a_noise = self.noises[1]*np.identity(3)
        self.m_noise = self.noises[2]*np.identity(3)
        self.q = np.array([1., 0., 0., 0.])
        self.P = np.identity(4)

    def jacobian(self, q, v):
        """
        Jacobian
        """
        qw, qx, qy, qz = q
        vx, vy, vz = v
        H = np.array([[qy*vz - qz*vy,             qy*vy + qz*vz, qw*vz + qx*vy - 2.0*qy*vx, qx*vz - qw*vy - 2.0*qz*vx],
                      [qz*vx - qx*vz, qy*vx - 2.0*qx*vy - qw*vz,             qx*vx + qz*vz, qw*vx + qy*vz - 2.0*qz*vy],
                      [qx*vy - qy*vx, qw*vy - 2.0*qx*vz + qz*vx, qz*vy - 2.0*qy*vz - qw*vx,             qx*vx + qy*vy]])
        return H

    def update(self, g, a, m, q):
        """
        Update State.

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
            Estimated (A-posteriori) quaternion.

        """
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
        # Normalize input quaternion
        q /= np.linalg.norm(q)

        # ----- Prediction -----
        # Approximate apriori quaternion
        F = q_mult_L(np.insert(0.5*self.samplePeriod*g, 0, 1.0))
        q_apriori = F@q
        # Estimate apriori Covariance Matrix
        E = np.vstack((-q[1:], skew(q[1:]) + q[0]*np.identity(3)))
        Qk = 0.25*self.samplePeriod**2 * (E@self.g_noise@E.T)
        P_apriori = F@self.P@F.T + Qk

        # ----- Correction -----
        q_apriori_conj = q_conj(q_apriori)
        dz = np.concatenate((q2R(q_apriori_conj)@m, q2R(q_apriori_conj)@a))
        H = np.vstack((self.jacobian(q_apriori_conj, m), self.jacobian(q_apriori_conj, a)))
        R = np.zeros((6, 6))
        R[:3, :3] = self.m_noise
        R[3:, 3:] = self.a_noise
        K = P_apriori@H.T@np.linalg.inv(H@P_apriori@H.T + R)
        q = q_apriori + K@dz
        P = (np.identity(4) - K@H)@P_apriori

        self.q = q/np.linalg.norm(q)
        self.P = P

        return self.q


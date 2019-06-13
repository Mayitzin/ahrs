# -*- coding: utf-8 -*-
"""
Attitude estimation using an Extended Kalman Filter

References
----------
.. [1] João Luís Marins, Xiaoping Yun, Eric R. Bachmann, Robert B. McGhee, and
  Michael J.Zyda. An Extended Kalman Filter for Quaternion-Based Orientation
  Estimation Using MARG Sensors. Proceedings of the 2001 IEEE/RSJ International
  Conference on Intelligent Robots and Systems, Maui, Hawaii, USA, Oct. 29 -
  Nov. 03, 2001, pp. 2003-2011.
  https://calhoun.nps.edu/handle/10945/41567
.. [2] Simo Särkkä (2013). Bayesian Filtering and Smoothing. Cambridge University Press.
  https://users.aalto.fi/~ssarkka/pub/cup_book_online_20131111.pdf
.. [3] Wikipedia: Extended Kalman Filter.
  https://en.wikipedia.org/wiki/Extended_Kalman_filter
.. [4] Thibaud Michel (2016). On Attitude Estimation with Smartphones.
  http://tyrex.inria.fr/mobile/benchmarks-attitude/

"""

import numpy as np
from ahrs.common.orientation import *
from ahrs.common.mathfuncs import *

class EKF:
    """
    Extended Kalman Filter

    Parameters
    ----------
    samplePeriod : float
        Sampling rate in seconds. Inverse of sampling frequency.
    noises : array
        List of noise variances :math:`\\sigma` for each type of sensor.
        Default values:


    .. math::

        \\sigma =
        \\begin{bmatrix}
        \\sigma_\\mathrm{acc} \\\\
        \\sigma_\\mathrm{gyr} \\\\
        \\sigma_\\mathrm{mag}
        \\end{bmatrix} =
        \\begin{bmatrix} 0.3^2 \\\\ 0.5^2 \\\\ 0.8^2 \\end{bmatrix}

    """
    def __init__(self, *args, **kwargs):
        self.frequency = kwargs.get('frequency', 256.0)
        self.samplePeriod = kwargs.get('samplePeriod', 1.0/self.frequency)
        self.noises = kwargs.get('noises', [0.3**2, 0.5**2, 0.8**2])
        self.g_noise = self.noises[0]*np.identity(3)
        self.a_noise = self.noises[1]*np.identity(3)
        self.m_noise = self.noises[2]*np.identity(3)
        self.q = np.array([1., 0., 0., 0.])
        self.P = np.identity(4)

    def jacobian(self, q, v):
        """
        Jacobian of vector :math:`\\mathbf{v}` with respect to quaternion :math:`\\mathbf{q}`.

        Parameters
        ----------
        q : array
            Quaternion.
        v : array
            vector to build a Jacobian from.

        """
        qw, qx, qy, qz = q
        vx, vy, vz = v
        J = 2.0*np.array([[qy*vz - qz*vy,             qy*vy + qz*vz, qw*vz + qx*vy - 2.0*qy*vx, qx*vz - qw*vy - 2.0*qz*vx],
                          [qz*vx - qx*vz, qy*vx - 2.0*qx*vy - qw*vz,             qx*vx + qz*vz, qw*vx + qy*vz - 2.0*qz*vy],
                          [qx*vy - qy*vx, qw*vy - 2.0*qx*vz + qz*vx, qz*vy - 2.0*qy*vz - qw*vx,             qx*vx + qy*vy]])
        return J

    def update(self, gyr, acc, mag, q):
        """
        Perform an update of the state.

        Parameters
        ----------
        gyr : array
            Sample of tri-axial Gyroscope in radians.
        aacc : array
            Sample of tri-axial Accelerometer.
        mag : array
            Sample of tri-axial Magnetometer.
        q : array
            A-priori quaternion.

        Returns
        -------
        q : array
            Estimated (A-posteriori) quaternion.

        """
        g = gyr.copy()
        a = acc.copy()
        m = mag.copy()
        # handle NaNs
        a_norm = np.linalg.norm(a)
        if a_norm == 0:
            return q
        m_norm = np.linalg.norm(m)
        if m_norm == 0:
            return q
        # Normalize vectors
        a /= a_norm
        m /= m_norm
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
        z = np.concatenate((q2R(q_apriori_conj)@m, q2R(q_apriori_conj)@a))
        H = np.vstack((self.jacobian(q_apriori_conj, m), self.jacobian(q_apriori_conj, a)))
        R = np.zeros((6, 6))
        R[:3, :3] = self.m_noise
        R[3:, 3:] = self.a_noise
        K = P_apriori@H.T@np.linalg.inv(H@P_apriori@H.T + R)
        q = q_apriori + K@z
        P = (np.identity(4) - K@H)@P_apriori

        self.q = q/np.linalg.norm(q)
        self.P = P

        return self.q


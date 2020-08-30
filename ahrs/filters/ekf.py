# -*- coding: utf-8 -*-
"""
Extended Kalman Filter
======================

References
----------
.. [Marins] João Luís Marins, Xiaoping Yun, Eric R. Bachmann, Robert B. McGhee,
    and Michael J.Zyda. An Extended Kalman Filter for Quaternion-Based
    Orientation Estimation Using MARG Sensors. Proceedings of the 2001 IEEE/RSJ
    International Conference on Intelligent Robots and Systems, Maui, Hawaii,
    USA, Oct. 29 - Nov. 03, 2001, pp. 2003-2011.
    (https://calhoun.nps.edu/handle/10945/41567)
.. [WikiEKF] Wikipedia: Extended Kalman Filter.
    (https://en.wikipedia.org/wiki/Extended_Kalman_filter)
.. [Michel] Thibaud Michel (2016). On Attitude Estimation with Smartphones.
    (http://tyrex.inria.fr/mobile/benchmarks-attitude/)

"""

import numpy as np
from ..common.orientation import *
from ..common.mathfuncs import *
from ..common import DEG2RAD

class EKF:
    """
    Extended Kalman Filter

    Parameters
    ----------
    Dt : float
        Sampling rate in seconds. Inverse of sampling frequency.
    noises : numpy.ndarray
        List of noise variances :math:`\\sigma` for each type of sensor.
        Default values: :math:`\\sigma = \\begin{bmatrix} 0.3^2 & 0.5^2 & 0.8^2 \\end{bmatrix}`

    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kwargs):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.frequency = kwargs.get('frequency', 100.0)
        self.Dt = kwargs.get('Dt', 1.0/self.frequency)
        self.q0 = kwargs.get('q0')
        self.noises = kwargs.get('noises', [0.3**2, 0.5**2, 0.8**2])
        self.g_noise = self.noises[0]*np.identity(3)
        self.a_noise = self.noises[1]*np.identity(3)
        self.m_noise = self.noises[2]*np.identity(3)
        self.P = np.identity(4)
        # Process of data is given
        if self.acc is not None and self.gyr is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """
        Estimate the quaternions given all data in class Data.

        Class Data must have, at least, ``acc`` and ``mag`` attributes.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        if self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 4))
        Q[0] = am2q(self.acc[0], self.mag[0]) if self.q0 is None else self.q0/np.linalg.norm(self.q0)
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def jacobian(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Jacobian of vector :math:`\\mathbf{v}` with respect to quaternion :math:`\\mathbf{q}`.

        Parameters
        ----------
        q : numpy.ndarray
            Quaternion.
        v : numpy.ndarray
            vector to build a Jacobian from.

        """
        qw, qx, qy, qz = q
        vx, vy, vz = v
        J = 2.0*np.array([[qy*vz - qz*vy,             qy*vy + qz*vz, qw*vz + qx*vy - 2.0*qy*vx, qx*vz - qw*vy - 2.0*qz*vx],
                          [qz*vx - qx*vz, qy*vx - 2.0*qx*vy - qw*vz,             qx*vx + qz*vz, qw*vx + qy*vz - 2.0*qz*vy],
                          [qx*vy - qy*vx, qw*vy - 2.0*qx*vz + qz*vx, qz*vy - 2.0*qy*vz - qw*vx,             qx*vx + qy*vy]])
        return J

    def update(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Perform an update of the state.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2.
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in mT.

        Returns
        -------
        q : numpy.ndarray
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
        F = q_mult_L(np.insert(0.5*self.Dt*g, 0, 1.0))
        q_apriori = F@q
        # Estimate apriori Covariance Matrix
        E = np.vstack((-q[1:], skew(q[1:]) + q[0]*np.identity(3)))
        Qk = 0.25*self.Dt**2 * (E@self.g_noise@E.T)
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


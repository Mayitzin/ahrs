# -*- coding: utf-8 -*-
"""
Extended Kalman Filter
======================

References
----------
.. [Kalman1960] Rudolf Kalman. A New Approach to Linear Filtering and Prediction
    Problems. 1960.
.. [Hartikainen] J. Hartikainen, A. Solin and S. Särkkä. Optimal Filtering with
    Kalman Filters and Smoothers. 2011
.. [Marins] João Luís Marins, Xiaoping Yun, Eric R. Bachmann, Robert B. McGhee,
    and Michael J.Zyda. An Extended Kalman Filter for Quaternion-Based
    Orientation Estimation Using MARG Sensors. Proceedings of the 2001 IEEE/RSJ
    International Conference on Intelligent Robots and Systems, Maui, Hawaii,
    USA, Oct. 29 - Nov. 03, 2001, pp. 2003-2011.
    (https://calhoun.nps.edu/handle/10945/41567)
.. [WikiEKF] Wikipedia: Extended Kalman Filter.
    (https://en.wikipedia.org/wiki/Extended_Kalman_filter)
.. [Labbe2015] Roger R. Labbe Jr. Kalman and Bayesian Filters in Python.
    (https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)

"""

import numpy as np
from ..common.orientation import q2R, ecompass
from ..common.mathfuncs import skew
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
    def __init__(self,
        gyr: np.ndarray = None,
        acc: np.ndarray = None,
        mag: np.ndarray = None,
        frequency: float = 100.0, **kwargs):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.frequency = frequency
        self.Dt = kwargs.get('Dt', 1.0/self.frequency)
        self.q0 = kwargs.get('q0')
        self.noises = kwargs.get('noises', [0.3**2, 0.5**2, 0.8**2])
        self.g_noise = self.noises[0]*np.identity(3)
        self.R = np.diag(np.repeat(self.noises[1:], 3))
        self.P = np.identity(4)
        # Process of data is given
        if all([x is not None for x in [self.gyr, self.acc, self.mag]]):
            self.Q = self._compute_all()

    def _compute_all(self):
        """
        Estimate the quaternions given all data in class Data.

        Attributes ``gyr``, ``acc`` and ``mag`` must contain data.

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
        Q[0] = ecompass(self.acc[0], self.mag[0], representation='quaternion') if self.q0 is None else self.q0/np.linalg.norm(self.q0)
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def dfdq(self, omega):
        """Linearization of state with partial derivative (Jacobian.)

        Parameters
        ----------
        omega : numpy.ndarray
            Angular velocity in rad/s.

        Returns
        -------
        F : numpy.ndarray
            Jacobian of state.
        """
        f = [1.0, *0.5*self.Dt*omega]
        return np.array([
            [f[0], -f[1], -f[2], -f[3]],
            [f[1],  f[0], -f[3],  f[2]],
            [f[2],  f[3],  f[0], -f[1]],
            [f[3], -f[2],  f[1],  f[0]]])

    def dhdq(self, q, v):
        """Linearization of observations with partial derivative (Jacobian.)

        Parameters
        ----------
        q : numpy.ndarray
            Predicted state estimate.
        v : numpy.ndarray
            Observations.

        Returns
        -------
        J : numpy.ndarray
            Jacobian of observations.
        """
        qw, qx, qy, qz = q
        J = np.array([
            [qy*v[2] - qz*v[1],               qy*v[1] + qz*v[2], qw*v[2] + qx*v[1] - 2.0*qy*v[0], qx*v[2] - qw*v[1] - 2.0*qz*v[0]],
            [qz*v[0] - qx*v[2], qy*v[0] - 2.0*qx*v[1] - qw*v[2],               qx*v[0] + qz*v[2], qw*v[0] + qy*v[2] - 2.0*qz*v[1]],
            [qx*v[1] - qy*v[0], qw*v[1] - 2.0*qx*v[2] + qz*v[0], qz*v[1] - 2.0*qy*v[2] - qw*v[0],               qx*v[0] + qy*v[1]],
            [qy*v[5] - qz*v[4],               qy*v[4] + qz*v[5], qw*v[5] + qx*v[4] - 2.0*qy*v[3], qx*v[5] - qw*v[4] - 2.0*qz*v[3]],
            [qz*v[3] - qx*v[5], qy*v[3] - 2.0*qx*v[4] - qw*v[5],               qx*v[3] + qz*v[5], qw*v[3] + qy*v[5] - 2.0*qz*v[4]],
            [qx*v[4] - qy*v[3], qw*v[4] - 2.0*qx*v[5] + qz*v[3], qz*v[4] - 2.0*qy*v[5] - qw*v[3],               qx*v[3] + qy*v[4]]])
        return 2.0*J

    def update(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Perform an update of the state.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori orientation as quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2.
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in uT.

        Returns
        -------
        q : numpy.ndarray
            Estimated a-posteriori orientation as quaternion.

        """
        if not np.isclose(np.linalg.norm(q), 1.0):
            raise ValueError("A-priori quaternion must have norm equal to 1.")
        g = np.copy(gyr)
        a = np.copy(acc)
        m = np.copy(mag)
        # handle NaNs
        a_norm = np.linalg.norm(a)
        m_norm = np.linalg.norm(m)
        if a_norm == 0 or m_norm == 0:
            return q
        # Normalize vectors
        a /= a_norm
        m /= m_norm
        # ----- Prediction -----
        F = self.dfdq(g)    # Linearize apriori quaternion
        q_k = F@q
        # Apriori Covariance Matrix (P_k)
        E = np.r_[[-q[1:]], skew(q[1:]) + q[0]*np.identity(3)]
        Q_k = 0.25*self.Dt**2 * (E@self.g_noise@E.T)
        P_k = F@self.P@F.T + Q_k
        # ----- Correction -----
        dcm_k = q2R(q_k)
        z = np.r_[dcm_k@a, dcm_k@m]
        H = self.dhdq(q_k, np.r_[a, m])     # Linearize Observations
        v = z - H@q_k
        S = H@P_k@H.T + self.R
        K = P_k@H.T@np.linalg.inv(S)
        self.P = (np.identity(4) - K@H)@P_k
        self.q = q_k + K@v
        self.q /= np.linalg.norm(self.q)
        return self.q

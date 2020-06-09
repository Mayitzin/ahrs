# -*- coding: utf-8 -*-
"""
Fast Linear Quaternion Attitude Estimator.

Quaternion obtained via an eigenvalue-based solution to Wahba's
problem.

References
----------
.. [1] Jin Wu, Zebo Zhou, Bin Gao, Rui Li, Yuhua Cheng, et al. Fast Linear
       Quaternion Attitude Estimator Using Vector Observations. IEEE
       Transactions on Automation Science and Engineering, Institute of
       Electrical and Electronics Engineers, 2018.
       https://hal.inria.fr/hal-01513263 and https://github.com/zarathustr/FLAE

"""

import numpy as np
from ..common.mathfuncs import *
# from ..common.constants import *
from ..common.orientation import am2q

class FLAE:
    """Fast Linear Attitude Estimator algorithm

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None, **kw):
        self.acc = acc
        self.mag = mag
        self.method = kw.get('method', 'eig')
        mdip = kw.get('magnetic_dip', 64.22)    # Magnetic dip, in degrees, in Munich, Germany.
        self.w = kw.get('weights', np.array([0.5, 0.5]))    # Weights of sensors
        self.a_ref = np.array([0.0, 0.0, 1.0])
        self.m_ref = np.array([cosd(mdip), 0.0, -sind(mdip)])
        self.ref = np.vstack((self.a_ref, self.m_ref))
        if self.acc is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """
        Estimate the quaternions given all data in class Data.

        Class Data must have, at least, `acc` and `mag` attributes.

        Returns
        -------
        Q : array
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape != self.mag.shape:
            raise ValueError("acc and mag are not the same size")
        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 4))
        for t in range(num_samples):
            Q[t] = self.update(self.acc[t], self.mag[t])
        return Q

    def Hx(self, h):
        return np.array([
            [ h[0],   0.0, -h[2],  h[1]],
            [ 0.0,   h[0],  h[1],  h[2]],
            [-h[2],  h[1], -h[0],  0.0],
            [ h[1],  h[2],  0.0,  -h[0]]])

    def Hy(self, h):
        return np.array([
            [ h[1],  h[2],  0.0, -h[0]],
            [ h[2], -h[1],  h[0],  0.0],
            [ 0.0,  h[0],  h[1],  h[2]],
            [-h[0],  0.0,  h[2], -h[1]]])

    def Hz(self, h):
        return np.array([
            [ h[2], -h[1],  h[0],  0.0],
            [-h[1], -h[2],  0.0,  h[0]],
            [ h[0],  0.0, -h[2],  h[1]],
            [ 0.0,  h[0],  h[1],  h[2]]])

    def update(self, acc, mag):
        """
        Estimate a quaternion with th given measurements and weights.

        Parameters
        ----------
        a : array
            Sample of tri-axial Accelerometer.
        m : array
            Sample of tri-axial Magnetometer.
        w : array
            Weights

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        a = acc.copy()/np.linalg.norm(acc)
        m = mag.copy()/np.linalg.norm(mag)
        body = np.vstack((a, m))
        mm = self.w*body.T@self.ref
        W = self.Hx(mm[0, :]) + self.Hy(mm[1, :]) + self.Hz(mm[2, :])
        if self.method.lower()=='eig':
            V, D = np.linalg.eig(W)
            q = D[:, 3]
            return q/np.linalg.norm(q)
        if self.method.lower()=='symbolic':
            # Polynomial parameters                     (eq. 48)
            t1 = -2*np.trace(mm@mm.T)
            t2 = -8*np.linalg.det(mm.T)
            t3 = np.linalg.det(W)
            # Symbolic solutions
            T0 = 2*t1**3 + 27*t2**2 - 72*t1*t3          # (eq. 53)
            T1 = np.cbrt(T0+np.sqrt(abs(-4*(t1**2 + 12*t3)**3 + T0**2)))
            T2 = np.sqrt(-4*t1+2**(4/3)*(t1**2+12*t3)/T1 + 2**(2/3)*T1)
            f = 12*np.sqrt(6)
            inv_f = 1/(2*np.sqrt(6))
            L = np.ones(4)                              # (eq. 52)
            L[0] = inv_f * (T2 - np.sqrt(-T2**2 - 12*t1 - f*t2/T2))
            L[1] = inv_f * (T2 + np.sqrt(-T2**2 - 12*t1 - f*t2/T2))
            L[2] = inv_f * (T2 + np.sqrt(-T2**2 - 12*t1 + f*t2/T2))
            L[3] = inv_f * (T2 - np.sqrt(-T2**2 - 12*t1 + f*t2/T2))
            lam = max(L)
            N = W - lam*np.identity(4)                  # (eq. 54)
            # Elementary row operations
            k = N[0, 0]
            N[0] /= k
            N[1] = N[1] - N[1, 0]*N[0]
            N[2] = N[2] - N[2, 0]*N[0]
            N[3] = N[3] - N[3, 0]*N[0]
            k = N[1, 1]
            N[1] /= k
            N[0] = N[0] - N[0, 1]*N[1]
            N[2] = N[2] - N[2, 1]*N[1]
            N[3] = N[3] - N[3, 1]*N[1]
            k = N[2, 2]
            N[2] /= k
            N[0] = N[0] - N[0, 2]*N[2]
            N[1] = N[1] - N[1, 2]*N[2]
            N[3] = N[3] - N[3, 2]*N[2]
            q = np.array([N[0, 3], N[1, 3], N[2, 3], -1])
            return q/np.linalg.norm(q)

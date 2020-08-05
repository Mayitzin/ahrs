# -*- coding: utf-8 -*-
"""
Fast Linear Attitude Estimator
==============================

The Fast Linear Attitude Estimator (FLAE) obtains the attitude quaternion with
an eigenvalue-based solution.

A symbolic solutions to the corresponding characteristic polynomial is also
derived for a higher computation speed.

References
----------
.. [1] Jin Wu, Zebo Zhou, Bin Gao, Rui Li, Yuhua Cheng, et al. Fast Linear
       Quaternion Attitude Estimator Using Vector Observations. IEEE
       Transactions on Automation Science and Engineering, Institute of
       Electrical and Electronics Engineers, 2018.
       https://hal.inria.fr/hal-01513263 and https://github.com/zarathustr/FLAE

"""

import numpy as np
from scipy import sqrt
from ..common.mathfuncs import *

# Reference Observations in Munich, Germany
from ..utils.wmm import WMM
MAG = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT).magnetic_elements

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
        self.method = kw.get('method', 'symbolic')
        if self.method.lower() not in ['eig', 'symbolic', 'newton']:
            print(f"Given method '{self.method}' is not valid. Setting to 'symbolic'")
            self.method = 'symbolic'
        self.w = kw.get('weights', np.array([0.5, 0.5]))    # Weights of sensors
        # Reference measurements
        mdip = kw.get('magnetic_dip')                       # Magnetic dip, in degrees
        self.m_ref = np.array([MAG['X'], MAG['Y'], MAG['Z']]) if mdip is None else np.array([cosd(mdip), 0., sind(mdip)])
        # self.m_ref = np.array([cosd(mdip), 0.0, -sind(mdip)])
        self.m_ref /= np.linalg.norm(self.m_ref)
        self.a_ref = np.array([0.0, 0.0, 1.0])
        self.ref = np.vstack((self.a_ref, self.m_ref))
        if self.acc is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
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
            Q[t] = self.estimate(self.acc[t], self.mag[t])
        return Q

    def row_reduction(self, A: np.ndarray) -> np.ndarray:
        """Gaussian elimination
        """
        for i in range(3):
            A[i] /= A[i, i]
            for j in range(4):
                if i!=j:
                    A[j] -= A[j, i]*A[i]
        return A

    def Hx(self, h: np.ndarray) -> np.ndarray:
        return np.array([
            [ h[0],   0.0, -h[2],  h[1]],
            [ 0.0,   h[0],  h[1],  h[2]],
            [-h[2],  h[1], -h[0],  0.0],
            [ h[1],  h[2],  0.0,  -h[0]]])

    def Hy(self, h: np.ndarray) -> np.ndarray:
        return np.array([
            [ h[1],  h[2],  0.0, -h[0]],
            [ h[2], -h[1],  h[0],  0.0],
            [ 0.0,  h[0],  h[1],  h[2]],
            [-h[0],  0.0,  h[2], -h[1]]])

    def Hz(self, h: np.ndarray) -> np.ndarray:
        return np.array([
            [ h[2], -h[1],  h[0],  0.0],
            [-h[1], -h[2],  0.0,  h[0]],
            [ h[0],  0.0, -h[2],  h[1]],
            [ 0.0,  h[0],  h[1],  h[2]]])

    def estimate(self, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
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
        body = np.r_[[acc.copy()/np.linalg.norm(acc)], [mag.copy()/np.linalg.norm(mag)]]
        mm = self.w*body.T@self.ref
        W = self.Hx(mm[0, :]) + self.Hy(mm[1, :]) + self.Hz(mm[2, :])   # (eq. 44)
        if self.method.lower()=='eig':
            V, D = np.linalg.eig(W)
            q = D[:, np.argmax(V)]
            return q/np.linalg.norm(q)
        # Polynomial parameters                             (eq. 49)
        t1 = -2*np.trace(mm@mm.T)
        t2 = -8*np.linalg.det(mm.T)
        t3 = np.linalg.det(W)
        if self.method.lower()=='symbolic':
            # Parameters                                    (eq. 53)
            T0 = 2*t1**3 + 27*t2**2 - 72*t1*t3
            T1 = np.cbrt(T0 + np.sqrt(abs(-4*(t1**2 + 12*t3)**3 + T0**2)))
            T2 = np.sqrt(abs(-4*t1 + 2**(4/3)*(t1**2 + 12*t3)/T1 + 2**(2/3)*T1))
            # Solutions to polynomial                       (eq. 52)
            L = np.zeros(4)
            k1 = -T2**2 - 12*t1
            k2 = 12*np.sqrt(6)*t2/T2
            L[0] =   T2 - sqrt(abs(k1 - k2))
            L[1] =   T2 + sqrt(abs(k1 - k2))
            L[2] = -(T2 + sqrt(abs(k1 + k2)))
            L[3] = -(T2 - sqrt(abs(k1 + k2)))
            L /= 2*np.sqrt(6)
            lam = max(L)                    # Choose eigenvalue closest to 1
        if self.method.lower()=='newton':
            lam = lam_old = 1.0
            i = 0
            while abs(lam_old-lam)>1e-8 or i<=30:
                lam_old = lam
                f = lam**4 + t1*lam**2 + t2*lam + t3        # (eq. 48)
                fp = 4*lam**3 + 2*t1*lam + t2               # (eq. 50)
                lam -= f/fp                                 # (eq. 51)
                i += 1
        N = W - lam*np.identity(4)                          # (eq. 54)
        N = self.row_reduction(N)                           # (eq. 55)
        q = np.array([N[0, 3], N[1, 3], N[2, 3], -1])       # (eq. 58)
        return q/np.linalg.norm(q)

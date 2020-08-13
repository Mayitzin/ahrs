# -*- coding: utf-8 -*-
"""
QUEST
=====

QUaternion ESTimator as described by Shuster in [Shuster1978]_ and [Shuster1981]_.

References
----------
.. [Shuster1978] Shuster, Malcom D. Approximate Algorithms for Fast Optimal
    Attitude Computation, AIAA Guidance and Control Conference. August 1978.
    (http://www.malcolmdshuster.com/Pub_1978b_C_PaloAlto_scan.pdf)
.. [Shuster1981] Shuster, M.D. and Oh, S.D. "Three-Axis Attitude Determination
    from Vector Observations," Journal of Guidance and Control, Vol.4, No.1,
    Jan.-Feb. 1981, pp. 70-77.

"""

import numpy as np
from ..common.mathfuncs import *

from ..utils.wmm import WMM
from ..utils.wgs84 import WGS
# Reference Observations in Munich, Germany
MAG = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT).magnetic_elements
GRAVITY = WGS().normal_gravity(MUNICH_LATITUDE, MUNICH_HEIGHT)

class QUEST:
    """
    QUaternion ESTimator

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT
    weights : array-like
        Array with two weights used in each observation.
    magnetic_dip : float
        Magnetic Inclination angle, in degrees.
    gravity : float
        Normal gravity, in m/s^2.

    Attributes
    ----------
    acc : numpy.ndarray
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    w : numpy.ndarray
        Weights for each observation.

    Raises
    ------
    ValueError
        When dimension of input arrays ``acc`` and ``mag`` are not equal.

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None, **kw):
        self.acc = acc
        self.mag = mag
        self.w = kw.get('weights', np.ones(2))
        # Reference measurements
        mdip = kw.get('magnetic_dip')                           # Magnetic dip, in degrees
        self.m_q = np.array([MAG['X'], MAG['Y'], MAG['Z']]) if mdip is None else np.array([cosd(mdip), 0., sind(mdip)])
        g = kw.get('gravity', GRAVITY)                          # Earth's normal gravity in m/s^2
        self.g_q = np.array([0.0, 0.0, g])                      # Normal Gravity vector
        if self.acc is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """Estimate the quaternions given all data.

        Attributes ``acc`` and ``mag`` must contain data.

        Returns
        -------
        Q : numpy.ndarray
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

    def estimate(self, acc: np.ndarray = None, mag: np.ndarray = None) -> np.ndarray:
        """Attitude Estimation.

        Parameters
        ----------
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in T

        Returns
        -------
        q : numpy.ndarray
            Estimated attitude as a quaternion.

        """
        B = self.w[0]*np.outer(acc, self.g_q) + self.w[1]*np.outer(mag, self.m_q)   # Attitude profile matrix
        S = B + B.T
        z = np.array([B[1, 2]-B[2, 1], B[2, 0]-B[0, 2], B[0, 1]-B[1, 0]])   # Pseudovector (Axial vector)
        # Parameters of characeristic equation (eq. 63)
        sigma = B.trace()
        Delta = np.linalg.det(S)
        kappa = (Delta*np.linalg.inv(S)).trace()
        # (eq. 71)
        a = sigma**2 - kappa
        b = sigma**2 + z@z
        c = Delta + z@S@z
        d = z@S**2@z
        # Newton-Raphson method (eq. 70)
        k = a*b + c*sigma - d
        lam = lam_0 = self.w.sum()
        while abs(lam-lam_0)>=1e-12:
            lam_0 = lam
            phi = lam**4 - (a+b)*lam**2 - c*lam + k
            phi_prime = 4*lam**3 - 2*(a+b)*lam - c
            lam -= phi/phi_prime
        # (eq. 66)
        alpha = lam**2 - sigma**2 + kappa
        beta = lam - sigma
        gamma = alpha*(lam+sigma) - Delta
        Chi = (alpha*np.eye(3) + beta*S + S**2)@z       # (eq. 68)
        # Optimal Quaternion (eq. 69)
        q = np.array([gamma, *Chi])
        return q/np.linalg.norm(q)

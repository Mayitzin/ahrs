# -*- coding: utf-8 -*-
"""
Davenport's q-Method
====================

Paul Davenport's q-method to estimate attitude as proposed in [Davenport1968]_.

References
----------
.. [Davenport1968] Paul B. Davenport. A Vector Approach to the Algebra of Rotations
    with Applications. NASA Technical Note D-4696. August 1968.
    (https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19680021122.pdf)

"""

import numpy as np
from ..common.mathfuncs import *

# Reference Observations in Munich, Germany
from ..utils.wmm import WMM
from ..utils.wgs84 import WGS
MAG = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT).magnetic_elements
GRAVITY = WGS().normal_gravity(MUNICH_LATITUDE, MUNICH_HEIGHT)

class Davenport:
    """
    Davenport's q-Method for attitude estimation

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT
    weights : array-like
        Array with two weights used in each observation.
    magnetic_dip : float
        Magnetic Inclination angle, in degrees. Defaults to magnetic dip of
        Munich, Germany.
    gravity : float
        Normal gravity, in m/s^2. Defaults to normal gravity of Munich,
        Germany.

    Attributes
    ----------
    acc : numpy.ndarray
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    w : numpy.ndarray
        Weights of each observation.

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
        mdip = kw.get('magnetic_dip')           # Magnetic dip, in degrees
        self.m_q = np.array([MAG['X'], MAG['Y'], MAG['Z']]) if mdip is None else np.array([cosd(mdip), 0., sind(mdip)])
        g = kw.get('gravity', GRAVITY)          # Earth's normal gravity, in m/s^2
        self.g_q = np.array([0.0, 0.0, g])      # Normal Gravity vector
        if self.acc is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """
        Estimate all quaternions given all data.

        Attributes ``acc`` and ``mag`` must contain data.

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

    def estimate(self, acc: np.ndarray = None, mag: np.ndarray = None) -> np.ndarray:
        """
        Attitude Estimation

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
        sigma = B.trace()
        z = np.array([B[1, 2]-B[2, 1], B[2, 0]-B[0, 2], B[0, 1]-B[1, 0]])
        S = B+B.T
        K = np.zeros((4, 4))
        K[0, 0] = sigma
        K[1:, 1:] = S - sigma*np.eye(3)
        K[0, 1:] = K[1:, 0] = z
        w, v = np.linalg.eig(K)
        return v[:, np.argmax(w)]       # Eigenvector associated to largest eigenvalue is normalized quaternion

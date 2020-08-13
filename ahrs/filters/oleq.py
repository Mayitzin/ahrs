# -*- coding: utf-8 -*-
"""
Optimal Linear Estimator of Quaternion
======================================

References
----------
.. [Zhou] Zhou, Z.; Wu, J.; Wang, J.; Fourati, H. Optimal, Recursive and
    Sub-Optimal Linear Solutions to Attitude Determination from Vector
    Observations for GNSS/Accelerometer/Magnetometer Orientation Measurement.
    Remote Sens. 2018, 10, 377.
    (https://www.mdpi.com/2072-4292/10/3/377)

"""

import numpy as np
from ..common.mathfuncs import *        # Import constants and special functions

# Reference Observations in Munich, Germany
from ..utils.wmm import WMM
from ..utils.wgs84 import WGS
MAG = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT).magnetic_elements
GRAVITY = WGS().normal_gravity(MUNICH_LATITUDE, MUNICH_HEIGHT)

class OLEQ:
    """
    Optimal Linear Estimator of Quaternion

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT

    Attributes
    ----------
    acc : numpy.ndarray
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    Q : numpy.array, default: None
        M-by-4 Array with all estimated quaternions, where M is the number of
        samples. Equal to None when no estimation is performed.

    Raises
    ------
    ValueError
        When dimension of input arrays ``acc`` and ``mag`` are not equal.

    Examples
    --------
    >>> acc_data.shape, mag_data.shape      # NumPy arrays with sensor data
    ((1000, 3), (1000, 3))
    >>> from ahrs.filters import OLEQ
    >>> orientation = OLEQ(acc=acc_data, mag=mag_data)
    >>> orientation.Q.shape                 # Estimated attitude
    (1000, 4)

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None, **kwargs):
        self.acc = acc
        self.mag = mag
        self.Q = None
        self.w = kwargs.get('weights', np.ones(2))
        # Reference measurements
        mdip = kwargs.get('magnetic_dip')   # Magnetic dip, in degrees
        self.m_ref = np.array([MAG['X'], MAG['Y'], MAG['Z']]) if mdip is None else np.array([cosd(mdip), 0., sind(mdip)])
        self.g_ref = np.array([0.0, 0.0, kwargs.get('gravity', GRAVITY)])   # Earth's Normal Gravity vector
        if self.acc is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """Estimate the quaternions given all data.

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

    def WW(self, b, r):
        """W Matrix

        Parameters
        ----------
        b : numpy.ndarray
            Normalized observations vector
        r : numpy.ndarray
            Normalized vector of reference frame

        Returns
        -------
        W_matrix : numpy.ndarray
            W Matrix of equation K^T b = W q
        """
        bx, by, bz = b
        rx, ry, rz = r
        M1 = np.array([[bx, 0.0, bz, -by], [0.0, bx, by, bz], [bz, by, -bx, 0.0], [-by, bz, 0.0, -bx]]) # (eq. 18a)
        M2 = np.array([[by, -bz, 0.0, bx], [-bz, -by, bx, 0.0], [0.0, bx, by, bz], [bx, 0.0, bz, -by]]) # (eq. 18b)
        M3 = np.array([[bz, by, -bx, 0.0], [by, -bz, 0.0, bx], [-bx, 0.0, -bz, by], [0.0, bx, by, bz]]) # (eq. 18c)
        return rx*M1 + ry*M2 + rz*M3    # (eq. 20)

    def estimate(self, acc: np.ndarray = None, mag: np.ndarray = None) -> np.ndarray:
        """Attitude Estimation

        Parameters
        ----------
        a : array
            Sample of tri-axial Accelerometer.
        m : array
            Sample of tri-axial Magnetometer.

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        # Normalize measurements (eq. 1)
        a_norm = np.linalg.norm(acc)
        m_norm = np.linalg.norm(mag)
        if not a_norm>0 or not m_norm>0:      # handle NaN
            return None
        a = acc/a_norm
        m = mag/m_norm
        W = self.w[0]*self.WW(a, self.g_ref) + self.w[1]*self.WW(m, self.m_ref) # (eq. 20)
        G = 0.5*(W + np.eye(4))
        q = np.ones(4)      # "random" quaternion
        last_q = np.array([1., 0., 0., 0.])
        i = 0
        while np.linalg.norm(q-last_q)>1e-8 and i<=20:
            last_q = q
            q = G@last_q                    # (eq. 25)
            q /= np.linalg.norm(q)
            i += 1
        return q/np.linalg.norm(q)

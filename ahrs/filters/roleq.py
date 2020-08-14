# -*- coding: utf-8 -*-
"""
Recursive Optimal Linear Estimator of Quaternion
================================================

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

class ROLEQ:
    """
    Recursive Optimal Linear Estimator of Quaternion

    Uses OLEQ to estimate the initial attitude.

    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s.
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2.
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT.

    Attributes
    ----------
    gyr : numpy.ndarray
        N-by-3 array with N gyroscope samples.
    acc : numpy.ndarray
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    frequency : float
        Sampling frequency in Herz
    Dt : float
        Sampling step in seconds. Inverse of sampling frequency.
    Q : numpy.array, default: None
        M-by-4 Array with all estimated quaternions, where M is the number of
        samples. Equal to None when no estimation is performed.

    Raises
    ------
    ValueError
        When dimension of input arrays ``gyr``, ``acc`` or ``mag`` are not
        equal.

    Examples
    --------
    >>> gyr_data.shape, acc_data.shape, mag_data.shape      # NumPy arrays with sensor data
    ((1000, 3), (1000, 3), (1000, 3))
    >>> from ahrs.filters import ROLEQ
    >>> orientation = ROLEQ(gyr=gyr_data, acc=acc_data, mag=mag_data)
    >>> orientation.Q.shape                 # Estimated attitude
    (1000, 4)

    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kwargs):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.Q = None
        self.frequency = kwargs.get('frequency', 100.0)
        self.Dt = kwargs.get('Dt', 1.0/self.frequency)
        self.w = kwargs.get('weights', np.ones(2))
        self.q0 = kwargs.get('q0')
        # Reference measurements
        mdip = kwargs.get('magnetic_dip', MAG['I'])   # Magnetic dip, in degrees
        # self.m_ref = np.array([MAG['X'], MAG['Y'], MAG['Z']]) if mdip is None else np.array([cosd(mdip), 0., sind(mdip)])
        self.m_ref = np.array([cosd(mdip), 0., sind(mdip)])
        self.g_ref = np.array([0.0, 0.0, kwargs.get('gravity', GRAVITY)])   # Earth's Normal Gravity vector
        # Estimate all quaternions if data is given
        if self.acc is not None and self.gyr is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """Estimate the quaternions given all data.

        Attributes ``gyr``, ``acc`` and ``mag`` must contain data.

        Returns
        -------
        Q : array
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        if self.acc.shape != self.mag.shape:
            raise ValueError("acc and mag are not the same size")
        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 4))
        Q[0] = self.estimate(self.acc[0], self.mag[0]) if self.q0 is None else self.q0.copy()
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def WW(self, b: np.ndarray, r: np.ndarray) -> np.ndarray:
        """W Matrix

        Parameters
        ----------
        b : numpy.ndarray
            Normalized observations vector
        r : numpy.ndarray
            Normalized vector of reference frame

        Returns
        -------
        W : numpy.ndarray
            Matrix :math:`\\mathbf{W}` of equation :math:`\\mathbf{K}^T \\mathbf{b} = \\mathbf{Wq}`
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
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer.
        mag : numpy.ndarray
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
        W = self.w[0]*self.WW(a, self.g_ref) + self.w[1]*self.WW(m, self.m_ref)
        G = 0.5*(W + np.eye(4))
        q = np.ones(4)
        last_q = np.array([1., 0., 0., 0.])
        i = 0
        while np.linalg.norm(q-last_q)>1e-8 and i<=20:
            last_q = q
            q = G@last_q                    # (eq. 25)
            q /= np.linalg.norm(q)
            i += 1
        return q/np.linalg.norm(q)

    def update(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """Update Attitude

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray, default: None
            Sample of angular velocity in rad/s
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in mT

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        """
        # Rotate given quaternion with gyroscope measurements
        gx, gy, gz = gyr
        Omega = np.array([[0.0, -gx, -gy, -gz], [gx, 0.0, gz, -gy], [gy, -gz, 0.0, gx], [gz, gy, -gx, 0.0]])
        Phi = np.eye(4) + 0.5*self.Dt*Omega     # (eq. 37)
        q_g = Phi@q
        q_g /= np.linalg.norm(q_g)
        # Second stage: Estimate with OLEQ
        a_norm = np.linalg.norm(acc)
        m_norm = np.linalg.norm(mag)
        if not a_norm>0 or not m_norm>0:    # handle NaN
            return q_g
        # Rotation operator (eq. 33)
        R = np.zeros((4, 4))
        R += 0.5*self.w[0]*(self.WW(acc/a_norm, self.g_ref) + np.eye(4))
        R += 0.5*self.w[1]*(self.WW(mag/m_norm, self.m_ref) + np.eye(4))
        q = R@q_g                                   # (eq. 25)
        q /= np.linalg.norm(q)
        return q

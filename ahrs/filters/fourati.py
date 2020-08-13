# -*- coding: utf-8 -*-
"""
Fourati's nonlinear attitude estimation
=======================================

Fourati Filter Algorithm as proposed by Hassen Fourati et al [Fourati]_.

References
----------
.. [Fourati] Hassen Fourati, Noureddine Manamanni, Lissan Afilal, Yves
    Handrich. A Nonlinear Filtering Approach for the Attitude and Dynamic Body
    Acceleration Estimation Based on Inertial and Magnetic Sensors: Bio-Logging
    Application. IEEE Sensors Journal, Institute of Electrical and Electronics
    Engineers, 2011, 11 (1), pp. 233-244. 10.1109/JSEN.2010.2053353.
    (https://hal.archives-ouvertes.fr/hal-00624142/file/Papier_IEEE_Sensors_Journal.pdf)

"""

import numpy as np
from ..common.orientation import q_prod, q_conj, am2q
from ..common.mathfuncs import *

# Reference Observations in Munich, Germany
from ..utils.wmm import WMM
from ..utils.wgs84 import WGS
MAG = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT).magnetic_elements
GRAVITY = WGS().normal_gravity(MUNICH_LATITUDE, MUNICH_HEIGHT)

class Fourati:
    """
    Fourati's attitude estimation

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT
    frequency : float, default: 100.0
        Sampling frequency in Herz.
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if `frequency` value is given.
    gain : float, default: 0.1
        Filter gain factor.
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).
    magnetic_dip : float
        Magnetic Inclination angle, in degrees.
    gravity : float
        Normal gravity, in m/s^2.

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
    gain : float
        Filter gain factor.
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion).

    Raises
    ------
    ValueError
        When dimension of input array(s) ``acc``, ``gyr``, or ``mag`` are not equal.

    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kwargs):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.frequency = kwargs.get('frequency', 100.0)
        self.Dt = kwargs.get('Dt', 1.0/self.frequency)
        self.gain = kwargs.get('gain', 0.1)
        self.q0 = kwargs.get('q0')
        # Reference measurements
        mdip = kwargs.get('magnetic_dip')             # Magnetic dip, in degrees
        self.m_q = np.array([0.0, MAG['X'], MAG['Y'], MAG['Z']]) if mdip is None else np.array([0.0, cosd(mdip), 0.0, sind(mdip)])
        self.m_q /= np.linalg.norm(self.m_q)
        self.g_q = np.array([0.0, 0.0, 0.0, 1.0])     # Normalized Gravity vector
        # Process of given data
        if self.acc is not None and self.gyr is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """
        Estimate the quaternions given all data

        Attributes ``gyr``, ``acc`` and ``mag`` must contain data.

        Returns
        -------
        Q : array
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        if self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        num_samples = len(self.gyr)
        Q = np.zeros((num_samples, 4))
        Q[0] = am2q(self.acc[0], self.mag[0]) if self.q0 is None else self.q0.copy()
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def update(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Quaternion Estimation with a MARG architecture.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in mT

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        """
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        qDot = 0.5 * q_prod(q, [0, *gyr])                           # (eq. 5)
        a_norm = np.linalg.norm(acc)
        m_norm = np.linalg.norm(mag)
        if a_norm>0 and m_norm>0 and self.gain>0:
            # Levenberg Marquardt Algorithm
            fhat = q_prod(q_conj(q), q_prod(self.g_q, q))           # (eq. 21)
            hhat = q_prod(q_conj(q), q_prod(self.m_q, q))           # (eq. 22)
            y = np.r_[acc/a_norm, mag/m_norm]                       # Measurements (eq. 6)
            yhat = np.r_[fhat[1:], hhat[1:]]                        # Estimated values (eq. 8)
            dq = y - yhat                                           # Modeling Error
            X = -2*np.c_[skew(fhat[1:]), skew(hhat[1:])].T          # Jacobian Matrix (eq. 23)
            lam = 1e-8                                              # Deviation to guarantee inversion
            K = self.gain*np.linalg.inv(X.T@X + lam*np.eye(3))@X.T  # Filter gain (eq. 24)
            Delta = [1, *K@dq]                                      # Correction term (eq. 25)
            qDot = q_prod(qDot, Delta)                              # Corrected quaternion rate (eq. 7)
        q += qDot*self.Dt
        return q/np.linalg.norm(q)

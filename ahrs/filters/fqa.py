# -*- coding: utf-8 -*-
"""
Factored Quaternion Algorithm
=============================

The factored quaternion algorithm (FQA) produces a quaternion output to
represent the orientation, which restricts the use of magnetic data to the
determination of the rotation about the vertical axis.

Magnetic variations cause only azimuth errors in FQA attitude estimation. A
singularity avoidance method is used, which allows the algorithm to track
through all orientations.

References
----------
.. [Yun] Xiaoping Yun et al. (2008) A Simplified Quaternion-Based Algorithm for
    Orientation Estimation From Earth Gravity and Magnetic Field Measurements.
    https://ieeexplore.ieee.org/document/4419916

"""

import numpy as np
from ..common.constants import *
from ..common.orientation import q_prod, q_conj

# Reference Observations in Munich, Germany
from ..utils.wmm import WMM
MAG = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT).magnetic_elements

class FQA:
    """Factored Quaternion Algorithm

    Attributes
    ----------
    acc : numpy.ndarray
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    m_ref : numpy.ndarray
        Reference local magnetic field.

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT

    Extra Parameters
    ----------------
    magnetic_dip : float
        Magnetic Inclination angle, in degrees.

    Raises
    ------
    ValueError
        When dimension of input arrays `acc` and `mag` are not equal.

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None, **kw):
        self.acc = acc
        self.mag = mag
        # Reference measurements
        mdip = kw.get('magnetic_dip')             # Magnetic dip, in degrees
        self.m_ref = np.array([MAG['X'], MAG['Y'], MAG['Z']]) if mdip is None else np.array([cosd(mdip), 0., sind(mdip)])
        self.m_ref = self.m_ref[:2]/np.linalg.norm(self.m_ref[:2])
        if self.acc is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """Estimate the quaternions given all data.

        Attributes `acc` and `mag` must contain data.

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
        """Attitude Estimation.

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
        a_norm = np.linalg.norm(acc)
        if a_norm == 0:     # handle NaN
            return np.array([1., 0., 0., 0.])
        a = acc/a_norm
        # Elevation Quaternion
        s_theta = a[0]                                              # (eq. 21)
        c_theta = np.sqrt(1.0-s_theta**2)                           # (eq. 22)
        s_theta_2 = np.sign(s_theta)*np.sqrt((1.0-c_theta)/2.0)     # (eq. 23)
        c_theta_2 = np.sqrt((1.0+c_theta)/2.0)                      # (eq. 24)
        q_e = np.array([c_theta_2, 0.0, s_theta_2, 0.0])            # (eq. 25)
        q_e /= np.linalg.norm(q_e)
        # Roll Quaternion
        is_singular = c_theta==0.0
        s_phi = 0.0 if is_singular else -a[1]/c_theta               # (eq. 30)
        c_phi = 0.0 if is_singular else -a[2]/c_theta               # (eq. 31)
        s_phi_2 = np.sign(s_phi)*np.sqrt((1.0-c_phi)/2.0)
        c_phi_2 = np.sqrt((1.0+c_phi)/2.0)
        q_r = np.array([c_phi_2, s_phi_2, 0.0, 0.0])                # (eq. 32)
        q_r /= np.linalg.norm(q_r)
        q_er = q_prod(q_e, q_r)
        q_er /= np.linalg.norm(q_er)
        # Azimuth Quaternion
        m_norm = np.linalg.norm(mag)
        if not m_norm>0:
            return q_er
        q_a = np.array([1., 0., 0., 0.])
        m_norm = np.linalg.norm(mag)
        if m_norm>0:
            m = mag/m_norm
            bm = np.array([0.0, *m])
            em = q_prod(q_e, q_prod(q_r, q_prod(bm, q_prod(q_conj(q_r), q_conj(q_e)))))     # (eq. 34)
            # em = [0.0, *q2R(q_e)@q2R(q_r)@m]
            # N = self.m_ref[:2].copy()                               # (eq. 36)
            N = self.m_ref.copy()                               # (eq. 36)
            _, Mx, My, _ = em/np.linalg.norm(em)                    # (eq. 37)
            c_psi, s_psi = np.array([[Mx, My], [-My, Mx]])@N        # (eq. 39)
            s_psi_2 = np.sign(s_psi)*np.sqrt((1.0-c_psi)/2.0)
            c_psi_2 = np.sqrt((1.0+c_psi)/2.0)
            q_a = np.array([c_psi_2, 0.0, 0.0, s_psi_2])            # (eq. 40)
            q_a /= np.linalg.norm(q_a)
        # Final Quaternion
        q = q_prod(q_a, q_er)                                       # (eq. 41)
        return q/np.linalg.norm(q)

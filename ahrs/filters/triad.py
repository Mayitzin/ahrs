# -*- coding: utf-8 -*-
"""
TRIAD
=====

The Tri-Axial Attitude Determination was first described by Gerald M. Lerner
in [Lerner]_ to algebraically estimate an attitude represented as a Direction
Cosine Matrix directly from two orthogonal vector observations.

References
----------
.. [Lerner] Lerner, G.M. "Three-Axis Attitude Determination" in Spacecraft
    Attitude Determination and Control, edited by J.R. Wertz. 1978. p. 420-428.
.. [Garcia] H. Garcia de Marina et al. UAV attitude estimation using Unscented
    Kalman Filter and TRIAD. IEE 2016. (https://arxiv.org/pdf/1609.07436.pdf)
.. [Hall] Chris Hall. Spacecraft Attitude Dynamics and Control. Chapter 4:
    Attitude Determination. 2003.
    (http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf)
.. [Makley] F.L. Makley et al. Fundamentals of Spacecraft Attitude
    Determination and Control. 2014. Pages 184-186.

"""

import numpy as np
from ..common.orientation import chiaverini
from ..common.mathfuncs import *

# Reference Observations in Munich, Germany
from ..utils.wmm import WMM
MAG = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT).magnetic_elements

class TRIAD:
    """Tri-Axial Attitude Determination

    Originally TRIAD estimates the Direction Cosine Matrix describing the
    attitude. This implementation, however, will return its equivalent
    quaternion by default. To return it as DCM, set the flag `as_dcm` to True.

    Parameters
    ----------
    acc : numpy.ndarray
        First 3-by-1 observation vector in body frame. Usually is normalized
        acceleration vector a = [ax ay az]
    mag : numpy.ndarray
        Second 3-by-1 observation vector in body frame. Usually is normalized
        magnetic field vector m = [mx my mz]
    V1 : numpy.ndarray, optional.
        3-by-1 Reference vector 1. Defaults to gravity g = [0 0 1]
    V2 : numpy.ndarray, optional.
        3-by-1 Reference vector 2. Defaults to normalized geomagnetic field
        h = [hx, hy, hz]

    Extra Parameters
    ----------------
    as_dcm : bool, default: False
        Whether to return attitude as a Direction Cosine Matrix.
    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None, **kw):
        self.acc = acc
        self.mag = mag
        self.V1 = kw.get('V1', np.array([0.0, 0.0, 1.0]))
        self.V2 = kw.get('V2', np.array([MAG['X'], MAG['Y'], MAG['Z']]))
        self.V2 /= np.linalg.norm(self.V2)
        self.as_dcm = kw.get('as_dcm', False)
        if self.acc is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """
        Estimate the quaternions given all data.

        Attributes `gyr`, `acc` and `mag` must contain data.

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

    def estimate(self, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """Attitude Estimation.

        Parameters
        ----------
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in T

        Returns
        -------
        attitude : numpy.ndarray
            Estimated attitude as 3-by-3 Direction Cosine Matrix if `as_dcm` is
            set to True. Otherwise as a quaternion.

        """
        # Normalized Observations
        W1 = acc.copy()/np.linalg.norm(acc)                 # (eq. 12-39a)
        W2 = mag.copy()/np.linalg.norm(mag)
        # First Triad
        W1xW2 = np.cross(W1, W2)
        s2 = W1xW2 / np.linalg.norm(W1xW2)                  # (eq. 12-39b)
        s3 = np.cross(W1, W1xW2) / np.linalg.norm(W1xW2)    # (eq. 12-39c)
        # Second Triad
        V1xV2 = np.cross(self.V1, self.V2)
        r2 = V1xV2 / np.linalg.norm(V1xV2)
        r3 = np.cross(self.V1, V1xV2) / np.linalg.norm(V1xV2)
        # Solve TRIAD
        Mb = np.c_[W1, s2, s3]                              # (eq- 12-41)
        Mr = np.c_[self.V1, r2, r3]                         # (eq. 12-42)
        dcm = Mb@Mr.T                                       # (eq. 12-45)
        if self.as_dcm:
            return dcm
        return chiaverini(dcm)

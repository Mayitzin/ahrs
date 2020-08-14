# -*- coding: utf-8 -*-
"""
Attitude from angular rate
==========================

Attitude quaternion obtained via angular rate measurements.

Integrate the given angular veolcity to obtain the angular position as a
quaternion representation [Jia]_.

References
----------
.. [Jia] Yan-Bin Jia. Quaternions. 2018.
    http://web.cs.iastate.edu/~cs577/handouts/quaternion.pdf

"""

import numpy as np
from ..common.orientation import q_prod

class AngularRate:
    """
    Quaternion Estimation based on angular velocity

    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s
    frequency : float, default: 100.0
        Sampling frequency in Herz.
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if `frequency` value is given.
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion).

    Attributes
    ----------
    gyr : numpy.ndarray
        N-by-3 array with N gyroscope samples.

    """
    def __init__(self, gyr: np.ndarray = None, **kw):
        self.gyr = gyr
        self.frequency = kw.get('frequency', 100.0)
        self.Dt = kw.get('Dt', 1.0/self.frequency)
        self.q0 = kw.get('q0')
        if self.gyr is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """Estimate all quaternions with given sensor values"""
        num_samples = len(self.gyr)
        Q = np.zeros((num_samples, 4))
        Q[0] = np.array([1.0, 0.0, 0.0, 0.0]) if self.q0 is None else self.q0.copy()
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t])
        return Q

    def update(self, q: np.ndarray, gyr: np.ndarray):
        """Update the quaternion estimation

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray, default: None
            Array with triaxial measurements of angular velocity in rad/s

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        """
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        q += 0.5*q_prod(q, [0, *gyr])*self.Dt
        return q/np.linalg.norm(q)


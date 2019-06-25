# -*- coding: utf-8 -*-
"""
Quaternion from gravity acceleration

Attitude quaternion obtained via gravity acceleration measurements.

"""

import numpy as np
from ahrs.common.orientation import *
from ahrs.common import DEG2RAD

class GravityQuaternion:
    """
    Class of Gravity-based estimation of quaternion.

    Parameters
    ----------

    """
    def __init__(self, *args, **kwargs):
        self.input = args[0] if args else None
        # Data is given
        if self.input:
            self.Q = self.estimate_all()

    def estimate_all(self):
        data = self.input
        Q = np.zeros((data.num_samples, 4))
        for t in range(data.num_samples):
            Q[t] = self.estimate(data.acc[t])
        return Q

    def estimate(self, a):
        """
        Estimate the quaternion

        Parameters
        ----------
        a : array
            Sample of tri-axial Accelerometer in m/s^2.

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        a_norm = np.linalg.norm(a)
        if a_norm == 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        a /= a_norm
        ax, ay, az = a
        # Euler Angles from Gravity vector
        ex = np.arctan2( ay, az)
        ey = np.arctan2(-ax, np.sqrt(ay*ay + az*az))
        # Euler to Quaternion
        q = np.array([1.0, 0.0, 0.0, 0.0])
        cx = np.cos(ex/2.0)
        sx = np.sin(ex/2.0)
        cy = np.cos(ey/2.0)
        sy = np.sin(ey/2.0)
        q = np.array([cx*cy, sx*cy, cx*sy, -sx*sy])
        return q/np.linalg.norm(q)

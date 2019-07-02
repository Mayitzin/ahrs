# -*- coding: utf-8 -*-
"""
Complementary filter for Quaternion estimation

Attitude quaternion obtained with gyroscope and acceleration measurements, via
complementary filter.

References
----------
.. [JWu] Jin Wu. Generalized Linear Quaternion Complementary Filter for
    Attitude Estimation from Multi-Sensor Observations: An Optimization
    Approach. IEEE Transactions on Automation Science and Engineering. 2019.
    https://ram-lab.com/papers/2018/tase_2018.pdf

"""

import numpy as np
from ahrs.common.orientation import *
from ahrs.common import DEG2RAD, RAD2DEG

class Complementary:
    """
    Class of complementary filter for quaternion estimation.

    Parameters
    ----------

    """
    def __init__(self, *args, **kwargs):
        self.input = args[0] if args else None
        self.alpha = kwargs.get('alpha', 0.9)
        self.frequency = kwargs.get('frequency', 100.0)
        self.Dt = kwargs.get('Dt', 1.0/self.frequency)
        self.pitch = 0.0
        self.roll = 0.0
        # Data is given
        if self.input:
            self.Q = self.estimate_all()

    def estimate_all(self):
        data = self.input
        d2r = 1.0 if data.in_rads else DEG2RAD
        Q = np.tile([1., 0., 0., 0.], (data.num_samples, 1))
        for t in range(data.num_samples):
            Q[t] = self.estimate(Q[t-1], data.acc[t].copy(), d2r*data.gyr[t].copy())
        return Q

    def estimate(self, q, a, g):
        """
        Estimate the quaternion from the tilting read by an orthogonal
        tri-axial array of accelerometers.

        The orientation of the roll and pitch angles is estimated using the
        measurements of the accelerometers, and finally converted to a
        quaternion representation according to [WKDCM2Q]_

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
        # Predict with Gyroscope
        q_omega = q + 0.5*q_prod(q, np.concatenate(([0], g)))*self.Dt
        q_omega /= np.linalg.norm(q_omega)
        # Correct with Euler Angles from Acceleration
        ex = np.arctan2( ay, az)
        ey = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        # Roll
        cx = np.cos(ex/2.0)
        sx = np.sin(ex/2.0)
        # Pitch
        cy = np.cos(ey/2.0)
        sy = np.sin(ey/2.0)
        q_acc = np.array([cy*cx + sy*sx, cy*sx - sy*cx, cy*sx + sy*cx, cy*cx - sy*sx])
        # Complementary Filter
        q = (1.0-self.alpha)*q_acc + self.alpha*q_omega
        return q/np.linalg.norm(q)

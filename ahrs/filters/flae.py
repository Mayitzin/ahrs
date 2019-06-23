# -*- coding: utf-8 -*-
"""
Fast Linear Quaternion Attitude Estimator.

Quaternion obtained via an eigenvalue-based solution to Wahba's
problem.

References
----------
.. [Wu] Jin Wu, Zebo Zhou, Bin Gao, Rui Li, Yuhua Cheng, et al. Fast Linear
    Quaternion Attitude Estimator Using Vector Observations. IEEE Transactions
    on Automation Science and Engineering, Institute of Electrical and
    Electronics Engineers, 2018, in press. 10.1109/TASE.2017.2699221.
    https://hal.inria.fr/hal-01513263 and https://github.com/zarathustr/FLAE

"""

import numpy as np
from ahrs.common.orientation import *
from ahrs.common.mathfuncs import *
from ahrs.common import DEG2RAD

class FLAE:
    """
    Class of Fast Linear Attitude Estimator algorithm

    Parameters
    ----------

    """
    def __init__(self, *args, **kwargs):
        self.input = args[0] if args else None
        mdip = kwargs.get('magnetic_dip', 64.22)    # Magnetic dip, in degrees, in Munich, Germany.
        self.w = kwargs.get('weights', np.array([0.5, 0.5]))    # Weights of sensors
        self.a_ref = np.array([0.0, 0.0, 1.0])
        self.m_ref = np.array([cosd(mdip), 0.0, -sind(mdip)])
        self.ref = np.vstack((self.a_ref, self.m_ref))
        # Process of data is given
        if self.input:
            self.Q = self.estimate_all()

    def estimate_all(self):
        """
        Estimate the quaternions given all data in class Data.

        Class Data must have, at least, `acc` and `mag` attributes.

        Returns
        -------
        Q : array
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        data = self.input
        Q = np.zeros((data.num_samples, 4))
        for t in range(data.num_samples):
            Q[t] = self.update(data.acc[t], data.mag[t])
        return Q

    def H1_matrix(self, h):
        Hx1, Hy1, Hz1 = h
        H = np.array([
            [ Hx1,  0.0, -Hz1,  Hy1],
            [ 0.0,  Hx1,  Hy1,  Hz1],
            [-Hz1,  Hy1, -Hx1,  0.0],
            [ Hy1,  Hz1,  0.0, -Hx1]])
        return H

    def H2_matrix(self, h):
        Hx2, Hy2, Hz2 = h
        H = np.array([
            [ Hy2,  Hz2,  0.0, -Hx2],
            [ Hz2, -Hy2,  Hx2,  0.0],
            [ 0.0,  Hx2,  Hy2,  Hz2],
            [-Hx2,  0.0,  Hz2, -Hy2]])
        return H

    def H3_matrix(self, h):
        Hx3, Hy3, Hz3 = h
        H = np.array([
            [ Hz3, -Hy3,  Hx3,  0.0],
            [-Hy3, -Hz3,  0.0,  Hx3],
            [ Hx3,  0.0, -Hz3,  Hy3],
            [ 0.0,  Hx3,  Hy3,  Hz3]])
        return H

    def update(self, acc, mag):
        """
        Estimate a quaternion with th given measurements and weights.

        Parameters
        ----------
        a : array
            Sample of tri-axial Accelerometer.
        m : array
            Sample of tri-axial Magnetometer.
        w : array
            Weights

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        a = acc.copy()
        m = mag.copy()
        a /= np.linalg.norm(a)
        m /= np.linalg.norm(m)
        body = np.vstack((a, m))
        mm = self.w*body.T@self.ref
        hx = mm[0, :]
        hy = mm[1, :]
        hz = mm[2, :]
        ww = self.H1_matrix(hx) + self.H2_matrix(hy) + self.H3_matrix(hz)
        V, D = np.linalg.eig(ww)
        q = D[:, 3]
        return q/np.linalg.norm(q)


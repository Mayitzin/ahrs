# -*- coding: utf-8 -*-
"""
Algebraic Quaternion Algorithm

References
----------
.. [Valenti1] Roberto Valenti et al. (2015) A Linear Kalman Filter for MARG
    Orientation Estimation Using the Algebraic Quaternion Algorithm.
    https://ieeexplore.ieee.org/document/7345567
.. [Valenti2] Valenti, R.G.; Dryanovski, I.; Xiao, J. Keeping a Good Attitude:
    A Quaternion-Based Orientation Filter for IMUs and MARGs. Sensors 2015, 15,
    19302-19330.
    https://res.mdpi.com/sensors/sensors-15-19302/article_deploy/sensors-15-19302.pdf

"""

import numpy as np
from ahrs.common.orientation import *
from ahrs.common.mathfuncs import *
from ahrs.common import DEG2RAD

class AQUA:
    """
    Class of Algebraic Quaternion Algorithm

    Parameters
    ----------
    frequency : float
        Sampling frequency in Herz.
    samplePeriod : float
        Sampling rate in seconds. Inverse of sampling frequency.

    """
    def __init__(self, *args, **kwargs):
        self.input = args[0] if args else None
        self.frequency = kwargs.get('frequency', 100.0)
        self.Dt = kwargs.get('Dt', 1.0/self.frequency)
        self.alpha = kwargs.get('alpha', 0.1)
        self.threshold = kwargs.get('threshold', 0.1)
        self.mdip = kwargs.get('magnetic_dip', 64.22)    # Magnetic dip, in degrees, in Munich, Germany.
        self.q_I = np.array([1.0, 0.0, 0.0, 0.0])
        self.m_ref = np.array([cosd(self.mdip), 0.0, -sind(self.mdip)])
        # Process of data is given
        if self.input:
            self.Q = self.estimate_all()

    def estimate_all(self):
        """
        Estimate the quaternions given all data in class Data.

        Class Data must have, at least, `gyr`, `acc` and `mag` attributes.

        Returns
        -------
        Q : array
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        data = self.input
        Q = np.tile([1., 0., 0., 0.], (data.num_samples, 1))
        if data.in_rads:
            data.gyr *= DEG2RAD
        for t in range(1, data.num_samples):
            Q[t] = self.update(Q[t-1], data.gyr[t], data.acc[t], data.mag[t])
        return Q

    def update(self, q, gyr, acc, mag):
        """
        Update Quaternion

        Parameters
        ----------
        q : array
            A-priori quaternion.
        gyr : array
            Sample of tri-axial Gyroscope in radians.
        a : array
            Sample of tri-axial Accelerometer.
        m : array
            Sample of tri-axial Magnetometer.

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        a = acc.copy()
        m = mag.copy()
        # Normalise acceleration and magnetic field measurements
        a_norm = np.linalg.norm(a)
        if a_norm == 0:
            return q
        a /= a_norm
        m_norm = np.linalg.norm(m)
        if m_norm == 0:
            return q
        m /= m_norm
        # Predict
        q_omega = q + 0.5*q_prod(q, np.concatenate(([0], gyr)))*self.Dt
        q_omega /= np.linalg.norm(q_omega)
        # # Acceleration Quaternion
        gx, gy, gz = q2R(q_omega).T@a
        Dq_acc = np.array([np.sqrt((gz+1.0)/2.0), -gy/np.sqrt(2.0*(gz+1.0)), gx/np.sqrt(2.0*(gz+1.0)), 0.0])
        Dq_acc /= np.linalg.norm(Dq_acc)
        # Correct
        if Dq_acc[0] > self.threshold:
            # Use LERP
            Dq_acc = (1.0-self.alpha)*self.q_I + self.alpha*Dq_acc
            Dq_acc /= np.linalg.norm(Dq_acc)
        else:
            # Use SLERP
            Omega = np.arccos(np.dot(self.q_I, Dq_acc))
            Dq_acc = self.q_I*np.sin(abs(1.0-self.alpha)*Omega)/np.sin(Omega) + Dq_acc*np.sin(self.alpha*Omega)/np.sin(Omega)
        q = q_prod(q_omega, Dq_acc)
        q /= np.linalg.norm(q)
        # if az >= 0:
        #     q_acc = np.array([np.sqrt((az+1.0)/2.0), -ay/np.sqrt(2.0*(az+1.0)), ax/np.sqrt(2.0*(az+1.0)), 0.0])
        # else:
        #     q_acc = np.array([-ay/np.sqrt(2.0*(1.0-az)), np.sqrt((1.0-az)/2.0), 0.0, ax/np.sqrt(2.0*(1.0-az))])
        # # Magnetometer Quaternion
        # mx, my, mz = m
        # lx, ly, lz = q2R(q_acc).T@m
        # G = lx**2 + ly**2
        # if lx >= 0:
        #     q_mag = np.array([np.sqrt(G+lx*np.sqrt(G))/np.sqrt(2.0*G), 0.0, 0.0, ly/(np.sqrt(2.0)*np.sqrt(G+lx*np.sqrt(G)))])
        # else:
        #     q_mag = np.array([ly/(np.sqrt(2.0)*np.sqrt(G-lx*np.sqrt(G))), 0.0, 0.0, np.sqrt(G-lx*np.sqrt(G))/np.sqrt(2.0*G)])
        return q


# -*- coding: utf-8 -*-
"""
Algebraic Quaternion Algorithm

References
----------
.. [Valenti1] Roberto Valenti et al. (2015) A Linear Kalman Filter for MARG
    Orientation Estimation Using the Algebraic Quaternion Algorithm.
    https://ieeexplore.ieee.org/document/7345567
.. []
    https://robertogl.github.io/

"""

import numpy as np
from ahrs.common.orientation import *
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
        self.samplePeriod = kwargs.get('samplePeriod', 1.0/self.frequency)
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
        d2r = 1.0 if data.in_rads else DEG2RAD
        Q = np.tile([1., 0., 0., 0.], (data.num_samples, 1))
        for t in range(1, data.num_samples):
            Q[t] = self.update(data.acc[t], data.mag[t])
        return Q

    def update(self, acc, mag):
        """
        Update Quaternion

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
        a = acc.copy()
        m = mag.copy()
        # Normalise acceleration and magnetic field measurements
        a_norm = np.linalg.norm(a)
        if a_norm == 0:     # handle NaN
            return np.array([1., 0., 0., 0.])
        a /= a_norm
        m_norm = np.linalg.norm(m)
        if m_norm == 0:     # handle NaN
            return np.array([1., 0., 0., 0.])
        m /= m_norm
        # Acceleration Quaternion
        ax, ay, az = a
        if az >= 0:
            q_acc = np.array([np.sqrt((az+1.0)/2.0), -ay/np.sqrt(2.0*(az+1.0)), ax/np.sqrt(2.0*(az+1.0)), 0.0])
        else:
            q_acc = np.array([-ay/np.sqrt(2.0*(1.0-az)), np.sqrt((1.0-az)/2.0), 0.0, ax/np.sqrt(2.0*(1.0-az))])
        # Magnetometer Quaternion
        mx, my, mz = m
        lx, ly, lz = q2R(q_acc).T@m
        G = lx**2 + ly**2
        if lx >= 0:
            q_mag = np.array([np.sqrt(G+lx*np.sqrt(G))/np.sqrt(2.0*G), 0.0, 0.0, ly/(np.sqrt(2.0)*np.sqrt(G+lx*np.sqrt(G)))])
        else:
            q_mag = np.array([ly/(np.sqrt(2.0)*np.sqrt(G-lx*np.sqrt(G))), 0.0, 0.0, np.sqrt(G-lx*np.sqrt(G))/np.sqrt(2.0*G)])
        # Final Quaternion
        return q_prod(q_acc, q_mag)


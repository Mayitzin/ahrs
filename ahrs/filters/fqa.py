# -*- coding: utf-8 -*-
"""
Factored Quaternion Algorithm

References
----------
.. [Yun] Xiaoping Yun et al. (2008) A Simplified Quaternion-Based Algorithm for
    Orientation Estimation From Earth Gravity and Magnetic Field Measurements.
    https://ieeexplore.ieee.org/document/4419916

"""

import numpy as np
from ahrs.common.orientation import *
from ahrs.common.mathfuncs import *
from ahrs.common import DEG2RAD

class FQA:
    """
    Class of Factored Quaternion Algorithm

    Parameters
    ----------

    """
    def __init__(self, *args, **kwargs):
        self.input = args[0] if args else None
        mdip = kwargs.get('magnetic_dip', 64.22)    # Magnetic dip, in degrees, in Munich, Germany.
        self.a_ref = np.array([0.0, 0.0, 1.0])
        self.m_ref = np.array([cosd(mdip), 0.0, -sind(mdip)])
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
        # Elevation Quaternion
        s_theta = a[0]
        c_theta = np.sqrt(1.0-s_theta**2)
        s_theta_2 = np.sign(s_theta)*np.sqrt((1.0-c_theta)/2.0)
        c_theta_2 = np.sqrt((1.0+c_theta)/2.0)
        q_e = np.array([c_theta_2, 0.0, s_theta_2, 0.0])
        # Roll Quaternion
        is_singular = c_theta == 0.0
        s_phi = 0.0 if is_singular else -a[1]/c_theta
        c_phi = 0.0 if is_singular else -a[2]/c_theta
        s_phi_2 = np.sign(s_phi)*np.sqrt((1.0-c_phi)/2.0)
        c_phi_2 = np.sqrt((1.0+c_phi)/2.0)
        q_r = np.array([c_phi_2, s_phi_2, 0.0, 0.0])
        # Azimuth Quaternion
        bm = np.array([0.0, m[0], m[1], m[2]])
        em = q_prod(q_e, q_prod(q_r, q_prod(bm, q_prod(q_conj(q_r), q_conj(q_e)))))
        nx, ny = self.m_ref[0], self.m_ref[2]
        N = np.array([nx, ny])/np.linalg.norm([nx, ny])
        M = em[1:3]/np.linalg.norm(em[1:3])
        c_psi, s_psi = np.array([[M[0], M[1]], [-M[1], M[0]]])@N
        s_psi_2 = np.sign(s_psi)*np.sqrt((1.0-c_psi)/2.0)
        c_psi_2 = np.sqrt((1.0+c_psi)/2.0)
        q_a = np.array([c_psi_2, 0.0, 0.0, s_psi_2])
        # Final Quaternion
        return q_prod(q_a, q_prod(q_e, q_r))


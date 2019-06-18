# -*- coding: utf-8 -*-
"""
Fast Accelerometer-Magnetometer Combination algorithm

References
----------
.. [Liu] Zhuohua Liu, Wei Liu, Xiangyang Gong, and Jin Wu, “Simplified Attitude
    Determination Algorithm Using Accelerometer and Magnetometer with Extremely
    Low Execution Time,” Journal of Sensors, vol. 2018, Article ID 8787236,
    11 pages, 2018. https://doi.org/10.1155/2018/8787236.

"""

import numpy as np
from ahrs.common.orientation import *

class FAMC:
    """
    Class of Fast Accelerometer-Magnetometer Combination algorithm

    """
    def __init__(self, *args, **kwargs):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, acc, mag):
        """
        FACM algorithm with a 6-axis Accelerometer-Magnetometer architecture.

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
        # Normalise measurements
        a_norm = np.linalg.norm(a)
        m_norm = np.linalg.norm(m)
        if a_norm == 0 or m_norm == 0:     # handle NaN
            return None
        a /= a_norm
        m /= m_norm
        # Store measured values
        ax, ay, az = a  # A^b = (ax, ay, az)  in article
        mx, my, mz = m  # M^b = (mx, my, mz)  in article
        # Dynamic magnetometer reference vector
        m_D = ax*mx + ay*my + az*mz
        m_N = np.sqrt(1.0-m_D**2)
        # Compute parameters
        t1 = 1.0-m_D**2
        t2 = 2.0+2.0*m_D*()
        B_11 = 0.5*m_N*mx
        B_21 = 0.5*m_N*my
        B_31 = 0.5*m_N*mz
        B_13 = 0.5*(ax+m_D*mx)
        B_23 = 0.5*(ay+m_D*my)
        B_33 = 0.5*(az+m_D*mz)
        tau = B_13+B_31
        alpha_1 = B_11-B_33-1.0
        alpha_2 = B_21**2/alpha_1-B_11-B_33-1.0
        alpha_3 = t1
        return None


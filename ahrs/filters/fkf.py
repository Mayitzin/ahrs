# -*- coding: utf-8 -*-
"""
Fast Kalman Filter algorithm

References
----------
.. [Guo] Siwen Guo, Jin Wu, Zuocai Wang, and Jide Qian, “Novel MARG-Sensor
    Orientation Estimation Algorithm UsingFast Kalman Filter.” Journal of
    Sensors, vol. 2017, Article ID 8542153, 12 pages.
    https://doi.org/10.1155/2017/8542153 and https://github.com/zarathustr/FKF

"""

import numpy as np
from ahrs.common.orientation import *
from ahrs.common import DEG2RAD

class FKF:
    """
    Class of Fast Kalman Filter algorithm

    Parameters
    ----------

    """
    def __init__(self, *args, **kwargs):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.Ar = np.array([0.0, 0.0, 1.0])
        self.Mr = np.array([0.0, 0.0, 1.0])

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
        Ab = acc.copy()
        Mb = mag.copy()
        return self.q


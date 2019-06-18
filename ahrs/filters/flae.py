# -*- coding: utf-8 -*-
"""
Fast Linear Quaternion Attitude Estimator Using Vector Observations.

Attitude quaternion obtained via an eigenvalue-based solution to Wahba's
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
from ahrs.common import DEG2RAD

class FLAE:
    """
    Class of Fast Linear Attitude Estimator algorithm

    Parameters
    ----------

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
        return self.q


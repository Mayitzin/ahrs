#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Attitude Filters Testing
========================

References
----------
.. [Crassidis] John L. Crassidis (2007) A Survey of Nonlinear Attitude
    Estimation Methods.
.. [Teage] Harris Teage (2016) Comparison of Attitude Estimation Techniques for
    Low-cost Unmanned Aerial Vehicles.
    https://arxiv.org/pdf/1602.07733.pdf
    http://ancs.eng.buffalo.edu/pdf/ancs_papers/2007/att_survey07.pdf
.. [Cirillo] A. Cirillo et al. (2016) A comparison of multisensor attitude
    estimation algorithms.
    https://www.researchgate.net/profile/Pasquale_Cirillo/publication/303738116_A_comparison_of_multisensor_attitude_estimation_algorithms/links/5750181208aeb753e7b4a0c0/A-comparison-of-multisensor-attitude-estimation-algorithms.pdf

"""

import numpy as np
import ahrs

RAD2DEG = ahrs.common.RAD2DEG
DEG2RAD = ahrs.common.DEG2RAD

class Test_Filter:
    def __init__(self, file_name, **kwargs):
        self.file = file_name
        self.data = ahrs.utils.io.load(self.file)

    def check_integrity(self, Q):
        sz = Q.shape
        qts_ok = not np.allclose(np.sum(Q, axis=0), sz[0]*np.array([1., 0., 0., 0.]))
        qnm_ok = np.allclose(np.linalg.norm(Q, axis=1).mean(), 1.0)
        return qts_ok and qnm_ok

    def allocate_arrays(self):
        self.Q = np.zeros((self.data.num_samples, 4))
        self.Q[:, 0] = 1.0

    def fourati(self):
        self.allocate_arrays()
        fourati = ahrs.filters.Fourati()
        for t in range(1, self.data.num_samples):
            self.Q[t] = fourati.update(DEG2RAD*self.data.gyr[t], self.data.acc[t], self.data.mag[t], self.Q[t-1])
        return self.check_integrity(self.Q)

    def ekf(self):
        self.allocate_arrays()
        ekf = ahrs.filters.EKF()
        for t in range(1, self.data.num_samples):
            self.Q[t] = ekf.update(DEG2RAD*self.data.gyr[t], self.data.acc[t], self.data.mag[t], self.Q[t-1])
        return self.check_integrity(self.Q)

    def mahony(self):
        self.allocate_arrays()
        mahony = ahrs.filters.Mahony()
        for t in range(1, self.data.num_samples):
            # self.Q[t] = mahony.updateIMU(DEG2RAD*self.data.gyr[t], self.data.acc[t], self.Q[t-1])
            self.Q[t] = mahony.updateMARG(DEG2RAD*self.data.gyr[t], self.data.acc[t], self.data.mag[t], self.Q[t-1])
        return self.check_integrity(self.Q)

    def madgwick(self):
        self.allocate_arrays()
        madgwick = ahrs.filters.Madgwick()
        for t in range(1, self.data.num_samples):
            # self.Q[t] = madgwick.updateIMU(DEG2RAD*self.data.gyr[t], self.data.acc[t], self.Q[t-1])
            self.Q[t] = madgwick.updateMARG(DEG2RAD*self.data.gyr[t], self.data.acc[t], self.data.mag[t], self.Q[t-1])
        return self.check_integrity(self.Q)

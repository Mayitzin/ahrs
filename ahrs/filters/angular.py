# -*- coding: utf-8 -*-
"""
Quaternion from angular rate.

Attitude quaternion obtained via angular rate measurements.

References
----------
.. [Wu] Yan-Bin Jia. Quaternions. 2018.
    http://web.cs.iastate.edu/~cs577/handouts/quaternion.pdf

"""

import numpy as np
from ahrs.common.orientation import *
from ahrs.common import DEG2RAD

class AngularRate:
    """
    Class of Angular rate based estimation of quaternion.

    Parameters
    ----------

    """
    def __init__(self, *args, **kwargs):
        self.input = args[0] if args else None
        self.frequency = kwargs.get('frequency', 100.0)
        self.Dt = kwargs.get('Dt', 1.0/self.frequency)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        # Data is given
        if self.input:
            self.Q = self.estimate_all()

    def estimate_all(self):
        data = self.input
        d2r = 1.0 if data.in_rads else DEG2RAD
        Q = np.tile(self.q, (data.num_samples, 1))
        if data.q_ref is not None:
            Q[0] = data.q_ref[0]
        for t in range(1, data.num_samples):
            Q[t] = self.update(Q[t-1], d2r*data.gyr[t])
        return Q

    def update(self, q, g):
        """
        Update the quaternion

        Parameters
        ----------
        g : array
            Sample of tri-axial Gyroscope in radians.

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        q += 0.5*q_prod(q, [0, g[0], g[1], g[2]])*self.Dt
        return q

if __name__ == '__main__':
    from ahrs.utils import plot
    data = np.genfromtxt('../../tests/repoIMU.csv', dtype=float, delimiter=';', skip_header=2)
    q_ref = data[:, 1:5]
    gyr = data[:, 8:11]
    num_samples = data.shape[0]
    # Estimate Orientations with IMU
    q = np.tile([1., 0., 0., 0.], (num_samples, 1))
    angular = AngularRate()
    for i in range(1, num_samples):
        q[i] = angular.update(q[i-1], gyr[i])
    # Compute Error
    sqe = abs(q_ref - q).sum(axis=1)**2
    # Plot results
    plot(q_ref, q, sqe,
        title="Angular rate integration",
        subtitles=["Reference Quaternions", "Estimated Quaternions", "Squared Errors"],
        yscales=["linear", "linear", "log"],
        labels=[[], [], ["MSE = {:.3e}".format(sqe.mean())]])

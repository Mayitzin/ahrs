# -*- coding: utf-8 -*-
"""
Mahony Algorithm as proposed by R. Mahony et al [Mahony]_ in 2010.

This implementation is based on the one made by S. Madgwick.

References
----------
.. [Mahony] Mahony et al. Nonlinear Complementary Filters on the Special
   Orthogonal Group; R. 2010.
   (https://hal.archives-ouvertes.fr/hal-00488376/document)

"""

import numpy as np
from ahrs.common.orientation import q_prod, q_conj
from ahrs.common import DEG2RAD

class Mahony:
    """
    Class of Mahony algorithm

    Parameters
    ----------
    Kp : float
        Proportional filter gain.
    Ki : float
        Integral filter gain.
    frequency : float
        Sampling frequency in Herz.
    samplePeriod : float
        Sampling rate in seconds. Inverse of sampling frequency.

    """
    def __init__(self, *args, **kwargs):
        self.input = args[0] if args else None
        self.Kp = kwargs.get('Kp', 1.0)
        self.Ki = kwargs.get('Ki', 0.0)
        self.frequency = kwargs.get('frequency', 100.0)
        self.samplePeriod = kwargs.get('samplePeriod', 1.0/self.frequency)
        # Integral Error
        self.eInt = np.array([0.0, 0.0, 0.0])
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
        if data.q_ref is not None:
            Q[0] = data.q_ref[0]
        if data.mag is None:
            for t in range(1, data.num_samples):
                Q[t] = self.updateIMU(Q[t-1], d2r*data.gyr[t], data.acc[t])
        else:
            for t in range(1, data.num_samples):
                Q[t] = self.updateMARG(Q[t-1], d2r*data.gyr[t], data.acc[t], data.mag[t])
        return Q

    def updateIMU(self, q, gyr, acc):
        """
        Mahony's AHRS algorithm with an IMU architecture.

        Adapted to Python from original implementation by Sebastian Madgwick.

        Parameters
        ----------
        gyr : array
            Sample of tri-axial Gyroscope in radians per second.
        acc : array
            Sample of tri-axial Accelerometer.
        q : array
            A-priori quaternion.

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        g = gyr.copy()
        a = acc.copy()
        # Normalise accelerometer measurement
        a_norm = np.linalg.norm(a)
        if a_norm == 0:     # handle NaN
            return q
        a /= a_norm
        # Assert values
        q /= np.linalg.norm(q)
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        # Estimate orientation error
        v = np.array([2.0*(qx*qz - qw*qy),
                      2.0*(qw*qx + qy*qz),
                      qw**2 - qx**2 - qy**2 + qz**2])
        e = np.cross(a, v)
        self.eInt = self.eInt + e*self.samplePeriod if self.Ki > 0 else np.array([0.0, 0.0, 0.0])
        # Apply feedback term
        g += self.Kp*e + self.Ki*self.eInt
        # Compute rate of change of quaternion
        qDot = 0.5*q_prod(q, [0.0, g[0], g[1], g[2]])
        # Integrate to yield Quaternion
        q += qDot*self.samplePeriod
        q /= np.linalg.norm(q)
        return q

    def updateMARG(self, q, gyr, acc, mag):
        """
        Mahony's AHRS algorithm with a MARG architecture.

        Adapted to Python from original implementation by Sebastian Madgwick.

        Parameters
        ----------
        gyr : array
            Sample of tri-axial Gyroscope in radians per second.
        acc : array
            Sample of tri-axial Accelerometer.
        mag : array
            Sample of tri-axial Magnetometer.
        q : array
            A-priori quaternion.

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        g = gyr.copy()
        a = acc.copy()
        m = mag.copy()
        # Normalise accelerometer measurement
        a_norm = np.linalg.norm(a)
        if a_norm == 0:     # handle NaN
            return q
        a /= a_norm
        # Normalise magnetometer measurement
        m_norm = np.linalg.norm(m)
        if m_norm == 0:     # handle NaN
            return q
        m /= m_norm
        # Assert values
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        # Reference direction of Earth's magnetic feild
        h = q_prod(q, q_prod([0, m[0], m[1], m[2]], q_conj(q)))
        b = [0.0, np.linalg.norm([h[1], h[2]]), 0.0, h[3]]
        # Estimated direction of gravity and magnetic flux
        v = np.array([2.0*(qx*qz - qw*qy),
                      2.0*(qw*qx + qy*qz),
                      qw**2 - qx**2 - qy**2 + qz**2])
        w = np.array([b[1]*(0.5 - qy**2 - qz**2) + b[3]*(qx*qz - qw*qy),
                      b[1]*(qx*qy - qw*qz)       + b[3]*(qw*qx + qy*qz),
                      b[1]*(qw*qy + qx*qz)       + b[3]*(0.5 - qx**2 - qy**2)])
        # Error is sum of cross product between estimated direction and measured direction of fields
        e = np.cross(a, v) + np.cross(m, 2.0*w)
        self.eInt = self.eInt + e*self.samplePeriod if self.Ki > 0 else np.array([0.0, 0.0, 0.0])
        # Apply feedback term
        g += self.Kp*e + self.Ki*self.eInt
        # Compute rate of change of quaternion
        qDot = 0.5*q_prod(q, [0.0, g[0], g[1], g[2]])
        # Integrate to yield Quaternion
        q += qDot*self.samplePeriod
        q /= np.linalg.norm(q)
        return q

if __name__ == '__main__':
    from ahrs.utils import plot
    data = np.genfromtxt('../../tests/repoIMU.csv', dtype=float, delimiter=';', skip_header=2)
    q_ref = data[:, 1:5]
    acc = data[:, 5:8]
    gyr = data[:, 8:11]
    mag = data[:, 11:14]
    num_samples = data.shape[0]
    # Estimate Orientations with IMU
    q_imu = np.tile([1., 0., 0., 0.], (num_samples, 1))
    mahony = Mahony()
    for i in range(1, num_samples):
        q_imu[i] = mahony.updateIMU(q_imu[i-1], gyr[i], acc[i])
    # Estimate Orientations with MARG
    q_marg = np.tile([1., 0., 0., 0.], (num_samples, 1))
    mahony = Mahony()
    for i in range(1, num_samples):
        q_marg[i] = mahony.updateMARG(q_marg[i-1], gyr[i], acc[i], mag[i])
    # Compute Error
    sqe_imu = abs(q_ref - q_imu).sum(axis=1)**2
    sqe_marg = abs(q_ref - q_marg).sum(axis=1)**2
    # Plot results
    plot(data[:, 1:5], q_imu, q_marg, [sqe_imu, sqe_marg],
        title="Mahony's algorithm",
        subtitles=["Reference Quaternions", "Estimated Quaternions (IMU)", "Estimated Quaternions (MARG)", "Squared Errors"],
        yscales=["linear", "linear", "linear", "log"],
        labels=[[], [], [], ["MSE (IMU) = {:.3e}".format(sqe_imu.mean()), "MSE (MARG) = {:.3e}".format(sqe_marg.mean())]])

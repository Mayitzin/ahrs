# -*- coding: utf-8 -*-
"""
Madgwick Algorithm

Orientation filter applicable to IMUs consisting of tri-axis gyroscopes and
accelerometers, and MARG sensor arrays that also include tri-axis magnetometers.

The MARG implementation incorporates magnetic distortion and gyroscope bias
drift compensation. The filter uses a quaternion representation, allowing
accelerometer and magnetometer data to be used in an analytically derived and
optimised gradient-descent algorithm to compute the direction of the gyroscope
measurement error as a quaternion derivative.

References
----------
.. [Madgwick] Sebastian Madgwick. An efficient orientation filter for inertial
    and inertial/magnetic sensor arrays. Internal Report. 2010.
    http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/

"""

import numpy as np
from ahrs.common.orientation import *
from ahrs.common import DEG2RAD

class Madgwick:
    """
    Class of Madgwick filter algorithm

    Parameters
    ----------
    beta : float
        Filter gain of a quaternion derivative.
    frequency : float
        Sampling frequency in Herz.
    Dt : float
        Sampling rate in seconds. Inverse of sampling frequency.

    """
    def __init__(self, *args, **kwargs):
        self.input = args[0] if args else None
        self.beta = kwargs.get('beta', 0.1)
        self.frequency = kwargs.get('frequency', 100.0)
        self.Dt = kwargs.get('Dt', 1.0/self.frequency)
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
        Madgwick's AHRS algorithm with an IMU architecture.

        Adapted to Python from original implementation by Sebastian Madgwick.

        Parameters
        ----------
        gyr : array
            Sample of tri-axial Gyroscope in radians.
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
        # Gradient decent algorithm corrective step
        f = np.array([2.0*(qx*qz - qw*qy)   - a[0],
                      2.0*(qw*qx + qy*qz)   - a[1],
                      2.0*(0.5-qx**2-qy**2) - a[2]])
        J = np.array([[-qy,       qz,     -qw,  qx],
                      [ qx,       qw,      qz,  qy],
                      [ 0.0, -2.0*qx, -2.0*qy, 0.0]])
        step = 2.0*J.T@f
        step /= np.linalg.norm(step)
        # Compute rate of change of quaternion
        qDot = 0.5 * q_prod(q, [0, g[0], g[1], g[2]]) - self.beta * step.T
        # Integrate to yield Quaternion
        q += qDot*self.Dt
        q /= np.linalg.norm(q)
        return q

    def updateMARG(self, q, gyr, acc, mag):
        """
        Madgwick's AHRS algorithm with a MARG architecture.

        Adapted to Python from original implementation by Sebastian Madgwick.

        Parameters
        ----------
        gyr : array
            Sample of tri-axial Gyroscope in radians.
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
        q /= np.linalg.norm(q)
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        # Reference direction of Earth's magnetic field
        h = q_prod(q, q_prod([0, m[0], m[1], m[2]], q_conj(q)))
        b = [0.0, np.linalg.norm([h[1], h[2]]), 0.0, h[3]]
        # Gradient decent algorithm corrective step
        F = np.array([(qx * qz - qw * qy)   - 0.5*a[0],
                      (qw * qx + qy * qz)   - 0.5*a[1],
                      (0.5 - qx**2 - qy**2) - 0.5*a[2],
                      b[1]*(0.5 - qy**2 - qz**2) + b[3]*(qx*qz - qw*qy)       - 0.5*m[0],
                      b[1]*(qx*qy - qw*qz)       + b[3]*(qw*qx + qy*qz)       - 0.5*m[1],
                      b[1]*(qw*qy + qx*qz)       + b[3]*(0.5 - qx**2 - qy**2) - 0.5*m[2]])
        J = np.array([[-qy,               qz,                  -qw,                    qx],
                    [ qx,                 qw,                   qz,                    qy],
                    [ 0.0,               -2.0*qx,              -2.0*qy,                0.0],
                    [-b[3]*qy,            b[3]*qz,             -2.0*b[1]*qy-b[3]*qw,  -2.0*b[1]*qz+b[3]*qx],
                    [-b[1]*qz+2*b[3]*qx,  b[1]*qy+b[3]*qw,      b[1]*qx+b[3]*qz,      -b[1]*qw+b[3]*qy],
                    [ b[1]*qy,            b[1]*qz-2.0*b[3]*qx,  b[1]*qw-2.0*b[3]*qy,   b[1]*qx]])
        step = 4.0*J.T@F
        step /= np.linalg.norm(step)    # normalise step magnitude
        # Compute rate of change of quaternion
        qDot = 0.5 * q_prod(q, [0, g[0], g[1], g[2]]) - self.beta * step.T
        # Integrate to yield quaternion
        q += qDot*self.Dt
        q /= np.linalg.norm(q) # normalise quaternion
        return q

if __name__ == '__main__':
    data = np.genfromtxt('../../tests/repoIMU.csv', dtype=float, delimiter=';', skip_header=2)
    q_ref = data[:, 1:5]
    acc = data[:, 5:8]
    gyr = data[:, 8:11]
    mag = data[:, 11:14]
    num_samples = data.shape[0]
    # Estimate Orientations with IMU
    q_imu = np.tile([1., 0., 0., 0.], (num_samples, 1))
    madgwick = Madgwick()
    for i in range(1, num_samples):
        q_imu[i] = madgwick.updateIMU(q_imu[i-1], gyr[i], acc[i])
    # Estimate Orientations with MARG
    q_marg = np.tile([1., 0., 0., 0.], (num_samples, 1))
    madgwick = Madgwick()
    for i in range(1, num_samples):
        q_marg[i] = madgwick.updateMARG(q_marg[i-1], gyr[i], acc[i], mag[i])
    # Compute Error
    sqe_imu = abs(q_ref - q_imu).sum(axis=1)**2
    sqe_marg = abs(q_ref - q_marg).sum(axis=1)**2
    # Plot results
    from ahrs.utils import plot
    plot(data[:, 1:5], q_imu, q_marg, [sqe_imu, sqe_marg],
        title="Madgwick's algorithm",
        subtitles=["Reference Quaternions", "Estimated Quaternions (IMU)", "Estimated Quaternions (MARG)", "Squared Errors"],
        yscales=["linear", "linear", "linear", "log"],
        labels=[[], [], [], ["MSE (IMU) = {:.3e}".format(sqe_imu.mean()), "MSE (MARG) = {:.3e}".format(sqe_marg.mean())]])

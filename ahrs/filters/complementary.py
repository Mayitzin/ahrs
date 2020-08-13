# -*- coding: utf-8 -*-
"""
Complementary filter for Quaternion estimation
==============================================

Attitude quaternion obtained with gyroscope and acceleration measurements, via
complementary filter as described by [Wu]_.

References
----------
.. [Wu] Jin Wu. Generalized Linear Quaternion Complementary Filter for Attitude
    Estimation from Multi-Sensor Observations: An Optimization Approach. IEEE
    Transactions on Automation Science and Engineering. 2019.
    (https://ram-lab.com/papers/2018/tase_2018.pdf)

"""

import numpy as np
from ..common.orientation import acc2q, q_prod, am2q

class ComplementaryQ:
    """
    Class of complementary filter for quaternion estimation.

    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT
    frequency : float, default: 100.0
        Sampling frequency in Herz.
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if ``frequency`` value is given.
    gain : float, default: 0.9
        Filter gain.
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).

    Attributes
    ----------
    gyr : numpy.ndarray
        N-by-3 array with N gyroscope samples.
    acc : numpy.ndarray
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    frequency : float
        Sampling frequency in Herz
    Dt : float
        Sampling step in seconds. Inverse of sampling frequency.
    gain : float
        Filter gain.
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion).

    Raises
    ------
    ValueError
        When dimension of input arrays `acc`, `gyr`, or `mag` are not equal.

    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kw):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.frequency = kw.get('frequency', 100.0)
        self.Dt = kw.get('Dt', 1.0/self.frequency)
        self.gain = kw.get('gain', 0.9)
        self.q0 = kw.get('q0')
        # Process of given data
        if self.gyr is not None and self.acc is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """Estimate the quaternions given all data

        Attributes `gyr`, `acc` and, optionally, `mag` must contain data.

        Returns
        -------
        Q : array
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape!=self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 4))
        # Compute with IMU architecture
        if self.mag is None:
            Q[0] = acc2q(self.acc[0]) if self.q0 is None else self.q0.copy()
            for t in range(1, num_samples):
                Q[t] = self.updateIMU(Q[t-1], self.gyr[t], self.acc[t])
            return Q
        # Compute with MARG architecture
        if self.mag.shape!=self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        Q[0] = am2q(self.acc[0], self.mag[0]) if self.q0 is None else self.q0.copy()
        for t in range(1, num_samples):
            Q[t] = self.updateMARG(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def updateIMU(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """"Quaternion Estimation with a IMU architecture.

        The orientation of the roll and pitch angles is estimated using the
        measurements of the gyroscopes and accelerometers, and converted to a
        quaternion representation.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        # Predict with Gyroscope
        q_omega = q + 0.5*q_prod(q, [0, *gyr])*self.Dt
        q_omega /= np.linalg.norm(q_omega)
        a_norm = np.linalg.norm(acc)
        if a_norm>0:
            a = acc.copy()/a_norm
            ax, ay, az = a
            # Estimate Pitch and Roll from Accelerometers
            ex = np.arctan2( ay, az)                        # Roll
            ey = np.arctan2(-ax, np.sqrt(ay**2 + az**2))    # Pitch
            cx, sx = np.cos(ex/2.0), np.sin(ex/2.0)
            cy, sy = np.cos(ey/2.0), np.sin(ey/2.0)
            q_am = np.array([cy*cx, cy*sx, sy*cx, -sy*sx])
            q_am /= np.linalg.norm(q_am)
        # Complementary Estimation
        q = (1.0-self.gain)*q_omega + self.gain*q_am
        return q/np.linalg.norm(q)

    def updateMARG(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """Quaternion Estimation with a MARG architecture.

        The orientation of the roll and pitch angles is estimated using the
        measurements of the gyroscopes, accelerometers and magnetometers, and
        converted to a quaternion representation.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in T

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        # Predict with Gyroscope
        q_omega = q + 0.5*q_prod(q, [0, *gyr])*self.Dt
        q_omega /= np.linalg.norm(q_omega)
        a_norm = np.linalg.norm(acc)
        if a_norm>0:
            a = acc.copy()/a_norm
            ax, ay, az = a
            # Estimate Pitch and Roll from Accelerometers
            ex = np.arctan2( ay, az)                        # Roll
            ey = np.arctan2(-ax, np.sqrt(ay**2 + az**2))    # Pitch
            cx, sx = np.cos(ex/2.0), np.sin(ex/2.0)
            cy, sy = np.cos(ey/2.0), np.sin(ey/2.0)
            ez = 0.0                                        # Yaw
            if mag is not None:
                # Estimate Yaw from compensated compass
                my2 = mag[2]*np.sin(ex) - mag[1]*np.cos(ex)
                mz2 = mag[1]*np.sin(ex) + mag[2]*np.cos(ex)
                mx3 = mag[0]*np.cos(ey) + mz2*np.sin(ey)
                ez = np.arctan2(my2, mx3)
            cz, sz = np.cos(ez/2.0), np.sin(ez/2.0)
            q_am = np.array([
                cz*cy*cx + sz*sy*sx,
                cz*cy*sx - sz*sy*cx,
                sz*cy*sx + cz*sy*cx,
                sz*cy*cx - cz*sy*sx])
            q_am /= np.linalg.norm(q_am)
        # Complementary Estimation
        q = (1.0-self.gain)*q_omega + self.gain*q_am
        return q/np.linalg.norm(q)

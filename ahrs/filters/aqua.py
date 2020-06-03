# -*- coding: utf-8 -*-
"""
Algebraic Quaternion Algorithm
==============================

References
----------
.. [1] Valenti, R.G.; Dryanovski, I.; Xiao, J. Keeping a Good Attitude: A
   Quaternion-Based Orientation Filter for IMUs and MARGs. Sensors 2015, 15,
   19302-19330.
   (https://res.mdpi.com/sensors/sensors-15-19302/article_deploy/sensors-15-19302.pdf)
.. [2] R. G. Valenti, I. Dryanovski and J. Xiao, "A Linear Kalman Filter for
   MARG Orientation Estimation Using the Algebraic Quaternion Algorithm," in
   IEEE Transactions on Instrumentation and Measurement, vol. 65, no. 2, pp.
   467-481, 2016.
   (https://ieeexplore.ieee.org/document/7345567)

"""

import numpy as np
from ..common.orientation import q_prod, q2R

GRAVITY = 9.80665

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
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kw):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.frequency = kw.get('frequency', 100.0)
        self.Dt = kw.get('Dt', 1.0/self.frequency)
        self.q0 = kw.get('q0')
        self.alpha = kw.get('alpha', 0.01)
        self.beta = kw.get('beta', 0.01)
        self.threshold = kw.get('threshold', 0.9)
        if self.acc is not None and self.gyr is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """Estimate all quaternions with given sensor values"""
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        num_samples = len(self.gyr)
        Q = np.zeros((num_samples, 4))
        # Compute with IMU architecture
        if self.mag is None:
            Q[0] = self.init_q(self.acc[0]) if self.q0 is None else self.q0.copy()
            for t in range(1, num_samples):
                Q[t] = self.updateIMU(Q[t-1], self.gyr[t], self.acc[t])
            return Q
        # Compute with MARG architecture
        if self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        Q[0] = self.init_q(self.acc[0], self.mag[0]) if self.q0 is None else self.q0.copy()
        for t in range(1, num_samples):
            Q[t] = self.updateMARG(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def init_q(self, acc, mag: np.ndarray = None):
        ax, ay, az = acc.copy()/np.linalg.norm(acc)
        # Quaternion from Accelerometer Readings (eq. 25)
        if az>=0:
            q_acc = np.array([np.sqrt((az+1)/2), -ay/np.sqrt(2*(1-ax)), ax/np.sqrt(2*(az+1)), 0.0])
        else:
            q_acc = np.array([-ay/np.sqrt(2*(1-az)), np.sqrt((1-az)/2.0), 0.0, ax/np.sqrt(2*(1-az))])
        if mag is not None:
            lx, ly, lz = q2R(q_acc).T@mag
            Gamma = lx**2 + ly**2
            # Quaternion from Magnetometer Readings (eq. 35)
            if lx>=0:
                q_mag = np.array([np.sqrt(Gamma+lx*np.sqrt(Gamma))/np.sqrt(2*Gamma), 0.0, 0.0, ly/np.sqrt(2)*np.sqrt(Gamma+lx*np.sqrt(Gamma))])
            else:
                q_mag = np.array([ly/np.sqrt(2)*np.sqrt(Gamma-lx*np.sqrt(Gamma)), 0.0, 0.0, np.sqrt(Gamma-lx*np.sqrt(Gamma))/np.sqrt(2*Gamma)])
            # Generalized Quaternion Orientation (eq. 36)
            q = q_prod(q_acc, q_mag)
            return q/np.linalg.norm(q)
        return q_acc

    def _slerp(self, q, ratio, t):
        q_I = np.array([1.0, 0.0, 0.0, 0.0])
        if q[0]>t:
            # LERP
            q = (1.0-ratio)*q_I + ratio*q   # (eq. 50)
        else:
            # SLERP
            angle = np.arccos(q[0])
            q = q_I*np.sin(abs(1.0-ratio)*angle)/np.sin(angle) + q*np.sin(ratio*angle)/np.sin(angle)    # (eq. 52)
        q /= np.linalg.norm(q)              # (eq. 51)
        return q

    def _adaptive_gain(self, gain, norm):
        error = abs(norm-GRAVITY)/GRAVITY
        e1, e2 = 0.1, 0.2
        factor = 0.0
        if error<e1:
            factor = 1.0
        if error<e2:
            factor = (error-e1)/(e1-e2) + 1.0
        return factor*gain

    def updateIMU(self, q, gyr, acc):
        """Update Quaternion

        Parameters
        ----------
        q : array
            A-priori quaternion.
        gyr : array
            Sample of tri-axial Gyroscope in radians.
        acc : array
            Sample of tri-axial Accelerometer.

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        if not np.linalg.norm(gyr)>0:
            return q
        # PREDICTION
        qDot = -0.5*q_prod([0, *gyr], q)                    # Quaternion derivative (eq. 38)
        qInt = q + qDot*self.Dt                             # Quaternion integration (eq. 42)
        qInt /= np.linalg.norm(qInt)
        # CORRECTION
        a_norm = np.linalg.norm(acc)
        if not a_norm>0:
            return qInt
        a = acc.copy()/a_norm
        gx, gy, gz = q2R(qInt).T@a                          # Predicted gravity (eq. 44)
        q_acc = np.array([np.sqrt((gz+1)/2.0), -gy/np.sqrt(2.0*(gz+1)), gx/np.sqrt(2.0*(gz+1)), 0.0])     # Delta Quaternion (eq. 47)
        # self.alpha = self._adaptive_gain(self.alpha, a_norm)
        q_acc = self._slerp(q_acc, self.alpha, self.threshold)
        q_prime = q_prod(qInt, q_acc)                       # (eq. 53)
        return q_prime/np.linalg.norm(q_prime)

    def updateMARG(self, q, gyr, acc, mag):
        """
        Update Quaternion

        Parameters
        ----------
        q : array
            A-priori quaternion.
        gyr : array
            Sample of tri-axial Gyroscope in radians.
        acc : array
            Sample of tri-axial Accelerometer.
        mag : array
            Sample of tri-axial Magnetometer.

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        if not np.linalg.norm(gyr)>0:
            return q
        # PREDICTION
        qDot = -0.5*q_prod([0, *gyr], q)                    # Quaternion derivative (eq. 38)
        qInt = q + qDot*self.Dt                             # Quaternion integration (eq. 42)
        qInt /= np.linalg.norm(qInt)
        # CORRECTION
        # Accelerometer-Based
        a_norm = np.linalg.norm(acc)
        if not a_norm>0:
            return qInt
        a = acc.copy()/a_norm
        gx, gy, gz = q2R(qInt).T@a                          # Predicted gravity (eq. 44)
        q_acc = np.array([np.sqrt((gz+1)/2.0), -gy/np.sqrt(2.0*(gz+1)), gx/np.sqrt(2.0*(gz+1)), 0.0])     # Delta Quaternion (eq. 47)
        # self.alpha = self._adaptive_gain(self.alpha, a_norm)
        q_acc = self._slerp(q_acc, self.alpha, self.threshold)
        q_prime = q_prod(qInt, q_acc)                       # (eq. 53)
        q_prime /= np.linalg.norm(q_prime)
        # Magnetometer-Based
        m_norm = np.linalg.norm(mag)
        if not m_norm>0:
            return q_prime
        lx, ly, lz = q2R(q_prime).T@mag                     # Predicted gravity (eq. 54)
        Gamma = lx**2 + ly**2
        q_mag = np.array([np.sqrt(Gamma+lx*np.sqrt(Gamma))/np.sqrt(2*Gamma), 0.0, 0.0, ly/np.sqrt(2*(Gamma+lx*np.sqrt(Gamma)))])    # (eq. 58)
        q_mag = self._slerp(q_mag, self.beta, self.threshold)
        q = q_prod(q_prime, q_mag)                          # (eq. 59)
        return q/np.linalg.norm(q)

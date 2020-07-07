# -*- coding: utf-8 -*-
"""
Fast Accelerometer-Magnetometer Combination
===========================================

References
----------
.. [1] Zhuohua Liu, Wei Liu, Xiangyang Gong, and Jin Wu, "Simplified Attitude
    Determination Algorithm Using Accelerometer and Magnetometer with Extremely
    Low Execution Time," Journal of Sensors, vol. 2018, Article ID 8787236,
    11 pages, 2018. https://doi.org/10.1155/2018/8787236.

"""

import numpy as np

class FAMC:
    """Fast Accelerometer-Magnetometer Combination

    Attributes
    ----------
    acc : numpy.ndarray
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    Q : numpy.array, default: None
        M-by-4 Array with all estimated quaternions, where M is the number of
        samples. Equal to None when no estimation is performed.

    Methods
    -------
    estimate(acc, mag)
        Estimate orientation `q` using an accelerometer, and a magnetometer
        sample.

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT

    Raises
    ------
    ValueError
        When dimension of input arrays `acc` and `mag` are not equal.

    Examples
    --------
    >>> acc_data.shape, mag_data.shape      # NumPy arrays with sensor data
    ((1000, 3), (1000, 3))
    >>> from ahrs.filters import FAMC
    >>> orientation = FAMC(acc=acc_data, mag=mag_data)
    >>> orientation.Q.shape                 # Estimated
    (1000, 4)

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None):
        self.acc = acc
        self.mag = mag
        self.Q = None
        if self.acc is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """Estimate the quaternions given all data.

        Attributes `acc` and `mag` must contain data.

        Returns
        -------
        Q : array
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape != self.mag.shape:
            raise ValueError("acc and mag are not the same size")
        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 4))
        for t in range(num_samples):
            Q[t] = self.estimate(self.acc[t], self.mag[t])
        return Q

    def estimate(self, acc: np.ndarray = None, mag: np.ndarray = None) -> np.ndarray:
        """Attitude Estimation

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
        # Normalize measurements (eq. 10)
        a_norm = np.linalg.norm(acc)
        m_norm = np.linalg.norm(mag)
        if not a_norm>0 or not m_norm>0:      # handle NaN
            return None
        a = acc/a_norm                  # A^b = [ax, ay, az]
        m = mag/m_norm                  # M^b = [mx, my, mz]
        # Dynamic magnetometer reference vector
        m_D = a[0]*m[0] + a[1]*m[1] + a[2]*m[2]     # (eq. 13)
        m_N = np.sqrt(1.0-m_D**2)
        # Parameters
        B = np.zeros((3, 3))            # (eq. 18)
        B[:, 0] = m_N*m
        B[:, 2] = m_D*m + a
        B *= 0.5
        tau = B[0, 2] + B[2, 0]
        p = np.zeros(3)
        Y = np.zeros((3, 3))
        p[0] = B[0, 0] - B[2, 2] - 1
        Y[0] = np.array([-1, B[1, 0], tau])/p[0]
        p[1] = B[1, 0]**2/p[0] - B[0, 0] - B[2, 2] - 1
        Y[1] = np.array([-B[1, 0]/p[0], -1, B[1, 2]+B[1, 0]*Y[0, 2]])/p[1]
        p[2] = p[0] - 2 + tau**2/p[0] + Y[1, 2]**2*p[1]
        Y[2] = np.array([(tau+B[1, 0]*Y[1, 2])/p[0], Y[1, 2], 1])/p[2]
        # Quaternion Elements
        q = -np.ones(4)
        q[1] = B[1, 2]*(Y[0, 0] + Y[0, 1]*(Y[1, 2]*Y[2, 0] + Y[1, 0]) + Y[0, 2]*Y[2, 0]) - (B[0, 2]-B[2, 0])*(Y[1, 2]*Y[2, 0] + Y[1, 0]) - Y[2, 0]*B[1, 0]
        q[2] = B[1, 2]*(          Y[0, 1]*(Y[1, 2]*Y[2, 1] + Y[1, 1]) + Y[0, 2]*Y[2, 1]) - (B[0, 2]-B[2, 0])*(Y[1, 2]*Y[2, 1] + Y[1, 1]) - Y[2, 1]*B[1, 0]
        q[3] = B[1, 2]*(          Y[0, 1]* Y[1, 2]*Y[2, 2]            + Y[0, 2]*Y[2, 2]) - (B[0, 2]-B[2, 0])*(Y[1, 2]*Y[2, 2])           - Y[2, 2]*B[1, 0]
        return q/np.linalg.norm(q)

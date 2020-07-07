# -*- coding: utf-8 -*-
"""
Super-fast Attitude of Accelerometer and Magnetometer
=====================================================

References
----------
.. [1] Jin Wu, Zebo Zhou, Hassen Fourati, Yuhua Cheng. A Super Fast Attitude
       Determination Algorithm for Consumer-Level Accelerometer and
       Magnetometer. IEEE Transactions on Con-sumer Electronics, Institute of
       Electrical and Electronics Engineers, 2018, 64 (3), pp.375
       381.10.1109/tce.2018.2859625. hal-01922922
       (https://hal.inria.fr/hal-01922922/document)

"""

import numpy as np

class SAAM:
    """Super-fast Attitude of Accelerometer and Magnetometer

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
    >>> from ahrs.filters import SAAM
    >>> orientation = SAAM(acc=acc_data, mag=mag_data)
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
        # Normalize measurements (eq. 1)
        a_norm = np.linalg.norm(acc)
        m_norm = np.linalg.norm(mag)
        if not a_norm>0 or not m_norm>0:      # handle NaN
            return None
        ax, ay, az = acc/a_norm
        mx, my, mz = mag/m_norm
        # Dynamic magnetometer reference vector (eq. 12)
        mD = ax*mx + ay*my + az*mz
        mN = np.sqrt(1-mD**2)
        # Quaternion components (eq. 16)
        q = np.array([ax*my - ay*(mN+mx), (az-1)*(mN+mx) + ax*(mD-mz), (az-1)*my + ay*(mD-mz), az*mD - ax*mN-mz])
        return q/np.linalg.norm(q)

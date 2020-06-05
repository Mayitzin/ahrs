# -*- coding: utf-8 -*-
"""
Madgwick Orientation Filter
===========================

Orientation filter applicable to IMUs consisting of tri-axial gyroscopes and
accelerometers, and MARG arrays that also include tri-axial magnetometers.

The filter employs a quaternion representation of orientation to describe the
nature of orientations in three-dimensions and is not subject to the
singularities associated with an Euler angle representation, allowing
accelerometer and magnetometer data to be used in an analytically derived and
optimised gradient-descent algorithm to compute the direction of the gyroscope
measurement error as a quaternion derivative.

Innovative aspects of this filter include:

- A single adjustable parameter defined by observable systems characteristics
- An analytically derived and optimised gradient-descent algorithm enabling performance at low sampling rates
- On-line magnetic distortion compensation algorithm.
- Gyroscope bias drift compensation.

Adapted to Python from original implementation by Sebastian Madgwick.

References
----------
.. [1] Sebastian Madgwick. An efficient orientation filter for inertial
    and inertial/magnetic sensor arrays. April 30, 2010.
    http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/

"""

import numpy as np
from ..common.orientation import q_prod, q_conj, acc2q, am2q

class Madgwick:
    """Madgwick's Gradient Descent Orientation Filter

    If `acc` and `gyr` are given as parameters, the orientations will be
    immediately computed with method `updateIMU`

    If `acc`, `gyr` and `mag` are given as parameters, the orientations will be
    immediately computed with method `updateMARG`

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
        Filter gain of a quaternion derivative.
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion).

    Methods
    -------
    updateIMU(q, gyr, acc)
        Update orientation `q` using a gyroscope and an accelerometer sample.
    updateMARG(q, gyr, acc, mag)
        Update orientation `q` using a gyroscope, an accelerometer, and a
        magnetometer gyroscope sample.

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT

    Extra Parameters
    ----------------
    frequency : float, default: 100.0
        Sampling frequency in Herz.
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if `frequency` value is given.
    gain : float, default: {0.033, 0.041}
        Filter gain of a quaternion derivative. Defaults to 0.033 for IMU
        implementations, or to 0.041 for MARG implementations.
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).

    Raises
    ------
    ValueError
        When dimension of input array(s) `acc`, `gyr`, or `mag` are not equal.

    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kw):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.frequency = kw.get('frequency', 100.0)
        self.Dt = kw.get('Dt', 1.0/self.frequency)
        self.q0 = kw.get('q0')
        self.gain = kw.get('beta')  # Setting gain with `beta` will be removed in the future.
        if self.gain is None:
            self.gain = kw.get('gain', 0.033 if self.mag is None else 0.041)
        if self.acc is not None and self.gyr is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """Estimate all quaternions with given sensor values"""
        if self.acc.shape != self.gyr.shape:
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
        if self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        Q[0] = am2q(self.acc[0], self.mag[0]) if self.q0 is None else self.q0.copy()
        for t in range(1, num_samples):
            Q[t] = self.updateMARG(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def updateIMU(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """Quaternion Estimation with a IMU architecture.

        Adapted to Python from original implementation by Sebastian Madgwick.

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
        qEst = 0.5 * q_prod(q, [0, *gyr])                           # (eq. 12)
        a_norm = np.linalg.norm(acc)
        if a_norm>0:
            a = acc/a_norm
            qw, qx, qy, qz = q/np.linalg.norm(q)
            # Gradient objective function and Jacobian
            f = np.array([2.0*(qx*qz - qw*qy)   - a[0],
                          2.0*(qw*qx + qy*qz)   - a[1],
                          2.0*(0.5-qx**2-qy**2) - a[2]])            # (eq. 25)
            J = np.array([[-2.0*qy,  2.0*qz, -2.0*qw, 2.0*qx],
                          [ 2.0*qx,  2.0*qw,  2.0*qz, 2.0*qy],
                          [ 0.0,    -4.0*qx, -4.0*qy, 0.0   ]])     # (eq. 26)
            # Objective Function Gradient
            gradient = J.T@f                                        # (eq. 34)
            gradient /= np.linalg.norm(gradient)
            qEst -= self.gain*gradient.T                            # (eq. 33)
        q += qEst*self.Dt                                           # (eq. 13)
        q /= np.linalg.norm(q)
        return q

    def updateMARG(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """Quaternion Estimation with a MARG architecture.

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
        q : numpy.ndarray, default: None
            Estimated quaternion.

        """
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        qEst = 0.5 * q_prod(q, [0, *gyr])                           # (eq. 12)
        a_norm = np.linalg.norm(acc)
        if a_norm>0:
            a = acc/a_norm
            if mag is None or not np.linalg.norm(mag)>0:
                return self.updateIMU(q, gyr, acc)
            m = mag/np.linalg.norm(mag)
            h = q_prod(q, q_prod([0, *m], q_conj(q)))               # (eq. 45)
            bx = np.linalg.norm([h[1], h[2]])                       # (eq. 46)
            bz = h[3]
            qw, qx, qy, qz = q/np.linalg.norm(q)
            # Gradient objective function (eq. 31) and Jacobian (eq. 32)
            f = np.array([2.0*(qx*qz - qw*qy)   - a[0],
                          2.0*(qw*qx + qy*qz)   - a[1],
                          2.0*(0.5-qx**2-qy**2) - a[2],
                          2.0*bx*(0.5 - qy**2 - qz**2) + 2.0*bz*(qx*qz - qw*qy)       - m[0],
                          2.0*bx*(qx*qy - qw*qz)       + 2.0*bz*(qw*qx + qy*qz)       - m[1],
                          2.0*bx*(qw*qy + qx*qz)       + 2.0*bz*(0.5 - qx**2 - qy**2) - m[2]])  # (eq. 31)
            J = np.array([[-2.0*qy,               2.0*qz,              -2.0*qw,               2.0*qx             ],
                          [ 2.0*qx,               2.0*qw,               2.0*qz,               2.0*qy             ],
                          [ 0.0,                 -4.0*qx,              -4.0*qy,               0.0                ],
                          [-2.0*bz*qy,            2.0*bz*qz,           -4.0*bx*qy-2.0*bz*qw, -4.0*bx*qz+2.0*bz*qx],
                          [-2.0*bx*qz+2.0*bz*qx,  2.0*bx*qy+2.0*bz*qw,  2.0*bx*qx+2.0*bz*qz, -2.0*bx*qw+2.0*bz*qy],
                          [ 2.0*bx*qy,            2.0*bx*qz-4.0*bz*qx,  2.0*bx*qw-4.0*bz*qy,  2.0*bx*qx          ]]) # (eq. 32)
            gradient = J.T@f                                        # (eq. 34)
            gradient /= np.linalg.norm(gradient)
            qEst -= self.gain*gradient.T                            # (eq. 33)
        q += qEst*self.Dt                                           # (eq. 13)
        q /= np.linalg.norm(q)
        return q

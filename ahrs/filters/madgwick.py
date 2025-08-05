# -*- coding: utf-8 -*-
"""
Madgwick Orientation Filter
===========================

Orientation filter applicable to IMUs consisting of tri-axial gyroscopes and
accelerometers, and MARG arrays, which also include tri-axial magnetometers,
proposed by Sebastian Madgwick [Madgwick]_.

The filter employs a quaternion representation of orientation to describe the
nature of orientations in three-dimensions and is not subject to the
singularities associated with an Euler angle representation, allowing
accelerometer and magnetometer data to be used in an analytically derived and
optimised gradient-descent algorithm to compute the direction of the gyroscope
measurement error as a quaternion derivative.

Innovative aspects of this filter include:

- A single adjustable parameter defined by observable systems characteristics.
- An analytically derived and optimised gradient-descent algorithm enabling
  performance at low sampling rates.
- On-line magnetic distortion compensation algorithm.
- Gyroscope bias drift compensation.

Adapted to Python from original implementation by Sebastian Madgwick.

References
----------
.. [Madgwick] Sebastian Madgwick. An efficient orientation filter for inertial 
    and inertial/magnetic sensor arrays. April 30, 2010.
    http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/

"""

import numpy as np
from ..common.orientation import q_prod, q_conj, acc2q, am2q

class Madgwick:
    """Madgwick's Gradient Descent Orientation Filter

    If ``acc`` and ``gyr`` are given as parameters, the orientations will be
    immediately computed with method ``updateIMU``.

    If ``acc``, ``gyr`` and ``mag`` are given as parameters, the orientations
    will be immediately computed with method ``updateMARG``.

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
        if `frequency` value is given.
    gain : float, default: {0.033, 0.041}
        Filter gain. Defaults to 0.033 for IMU implementations, or to 0.041 for
        MARG implementations.
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).

    Attributes
    ----------
    gyr : numpy.ndarray
        N-by-3 array with N tri-axial gyroscope samples.
    acc : numpy.ndarray
        N-by-3 array with N tri-axial accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N tri-axial magnetometer samples.
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
        When dimension of input array(s) ``acc``, ``gyr``, or ``mag`` are not
        equal.

    Examples
    --------
    Assuming we have 3-axis sensor data in N-by-3 arrays, we can simply give
    these samples to their corresponding type. The Madgwick algorithm can work
    solely with gyroscope and accelerometer samples.

    The easiest way is to directly give the full array of samples to their
    matching parameters.

    >>> from ahrs.filters import Madgwick
    >>> orientation = Madgwick(gyr=gyro_data, acc=acc_data)     # Using IMU

    The estimated quaternions are saved in the attribute ``Q``.

    >>> type(orientation.Q), orientation.Q.shape
    (<class 'numpy.ndarray'>, (1000, 4))

    If we desire to estimate each sample independently, we call the
    corresponding method.

    .. code:: python

        orientation = Madgwick()
        Q = np.tile([1., 0., 0., 0.], (num_samples, 1)) # Allocate for quaternions
        for t in range(1, num_samples):
            Q[t] = orientation.updateIMU(Q[t-1], gyr=gyro_data[t], acc=acc_data[t])

    Further on, we can also use magnetometer data.

    >>> orientation = Madgwick(gyr=gyro_data, acc=acc_data, mag=mag_data)   # Using MARG

    This algorithm is dynamically adding the orientation, instead of estimating
    it from static observations. Thus, it requires an initial attitude to build
    on top of it. This can be set with the parameter ``q0``:

    >>> orientation = Madgwick(gyr=gyro_data, acc=acc_data, q0=[0.7071, 0.0, 0.7071, 0.0])

    If no initial orientation is given, then an attitude using the first
    samples is estimated. This attitude is computed assuming the sensors are
    straped to a system in a quasi-static state.

    A constant sampling frequency equal to 100 Hz is used by default. To change
    this value we set it in its parameter ``frequency``. Here we set it, for
    example, to 150 Hz.

    >>> orientation = Madgwick(gyr=gyro_data, acc=acc_data, frequency=150.0)

    Or, alternatively, setting the sampling step (:math:`\\Delta t = \\frac{1}{f}`):

    >>> orientation = Madgwick(gyr=gyro_data, acc=acc_data, Dt=1/150)

    This is specially useful for situations where the sampling rate is variable:

    .. code:: python

        orientation = Madgwick()
        Q = np.tile([1., 0., 0., 0.], (num_samples, 1)) # Allocate for quaternions
        for t in range(1, num_samples):
            orientation.Dt = new_sample_rate
            Q[t] = orientation.updateIMU(Q[t-1], gyr=gyro_data[t], acc=acc_data[t])

    Madgwick's algorithm uses a gradient descent method to correct the
    estimation of the attitude. The **step size**, a.k.a.
    `learning rate <https://en.wikipedia.org/wiki/Learning_rate>`_, is
    considered a gain of this algorithm and can be set in the parameters too:

    >>> orientation = Madgwick(gyr=gyro_data, acc=acc_data, gain=0.01)

    Following the original article, the gain defaults to ``0.033`` for IMU
    arrays, and to ``0.041`` for MARG arrays.

    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kwargs):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.frequency = kwargs.get('frequency', 100.0)
        self.Dt = kwargs.get('Dt', 1.0/self.frequency)
        self.q0 = kwargs.get('q0')
        self.gain = kwargs.get('beta')  # Setting gain with `beta` will be removed in the future.
        if self.gain is None:
            self.gain = kwargs.get('gain', 0.033 if self.mag is None else 0.041)
        if self.acc is not None and self.gyr is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """Estimate the quaternions given all data.

        Attributes ``gyr`` and ``acc`` must contain data. If ``mag`` contains
        data, the updateMARG() method is used.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 4))
        # Compute with IMU architecture
        if self.mag is None:
            Q[0] = acc2q(self.acc[0]) if self.q0 is None else self.q0/np.linalg.norm(self.q0)
            for t in range(1, num_samples):
                Q[t] = self.updateIMU(Q[t-1], self.gyr[t], self.acc[t])
            return Q
        # Compute with MARG architecture
        if self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        Q[0] = am2q(self.acc[0], self.mag[0]) if self.q0 is None else self.q0/np.linalg.norm(self.q0)
        for t in range(1, num_samples):
            Q[t] = self.updateMARG(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def updateIMU(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """
        Quaternion Estimation with IMU architecture.

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
        q : numpy.ndarray
            Estimated quaternion.

        """
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        qEst = 0.5 * q_prod(q, [0, *gyr])                           # (eq. 12)
        a_norm = np.linalg.norm(acc)
        if a_norm>0:
            a = acc/a_norm
            qw, qx, qy, qz = q/np.linalg.norm(q)
            # Gradient objective function (eq. 25) and Jacobian (eq. 26)
            f = np.array([2.0*(qx*qz - qw*qy)   - a[0],
                          2.0*(qw*qx + qy*qz)   - a[1],
                          2.0*(0.5-qx**2-qy**2) - a[2]])            # (eq. 25)
            J = np.array([[-2.0*qy,  2.0*qz, -2.0*qw, 2.0*qx],
                          [ 2.0*qx,  2.0*qw,  2.0*qz, 2.0*qy],
                          [ 0.0,    -4.0*qx, -4.0*qy, 0.0   ]])     # (eq. 26)
            # Objective Function Gradient
            gradient = J.T@f                                        # (eq. 34)
            gradient /= np.linalg.norm(gradient)
            qEst -= self.gain*gradient                              # (eq. 33)
        q += qEst*self.Dt                                           # (eq. 13)
        q /= np.linalg.norm(q)
        return q

    def updateMARG(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Quaternion Estimation with a MARG architecture.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in mT

        Returns
        -------
        q : numpy.ndarray
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
            qEst -= self.gain*gradient                              # (eq. 33)
        q += qEst*self.Dt                                           # (eq. 13)
        q /= np.linalg.norm(q)
        return q

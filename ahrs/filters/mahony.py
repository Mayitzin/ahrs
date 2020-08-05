# -*- coding: utf-8 -*-
"""
Mahony Orientation Filter
=========================

The filter designed by Robert Mahony [Mahony]_ is formulated as a deterministic
observer in SO(3) mainly driven by angular velocity measurements and
reconstructed attitude.

This observer, termed "explicit complementary filter" (ECF), uses an inertial
measurement :math:`a` and an angular velocity measurement :math:`\\omega`. The
inertial direction obtained from the gravity is a low-frequency normalized
measurement:

.. math::

    a = \\frac{a}{\\|a\\|}

A predicted direction of gravity :math:`v` is expected to be colinear with the
Z-axis of the inertial frame:

.. math::

    \\begin{array}{lll}
    v & = & R(q)^T \\begin{bmatrix}0 & 0 & 1 \\end{bmatrix}^T\\\\
    & = &
    \\begin{bmatrix}
    1 - 2(q_y^2 + q_z^2) & 2(q_xq_y + q_wq_z) & 2(q_xq_z - q_wq_y) \\\\
    2(q_xq_y - q_wq_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_wq_x + q_yq_z) \\\\
    2(q_xq_z + q_wq_y) & 2(q_yq_z - q_wq_x) & 1 - 2(q_x^2 + q_y^2)
    \\end{bmatrix}
    \\begin{bmatrix}0 \\\\ 0 \\\\ 1\\end{bmatrix} \\\\
    & = &
    \\begin{bmatrix}
    2(q_xq_z - q_wq_y) \\\\ 2(q_wq_x + q_yq_z) \\\\ 1 - 2(q_x^2 + q_y^2)
    \\end{bmatrix}
    \\end{array}

Considering the basic model of a gyroscope :math:`g`:

.. math::

    g = \\omega + b + \\mu

where :math:`\\omega` is the real angular velocity, :math:`b` is a varying
deterministic bias, and :math:`\\mu` is a Gaussian noise.

This implementation is based on simplifications by [Euston]_ and [Hamel]_ for
low-cost inertial measurement units in UAVs.

References
----------
.. [Mahony] Robert Mahony, Tarek Hamel, and Jean-Michel Pflimlin. Nonlinear
   Complementary Filters on the Special Orthogonal Group. IEEE Transactions
   on Automatic Control, Institute of Electrical and Electronics Engineers,
   2008, 53 (5), pp.1203-1217.
   (https://hal.archives-ouvertes.fr/hal-00488376/document)
.. [Euston] Mark Euston, Paul W. Coote, Robert E. Mahony, Jonghyuk Kim, and
   Tarek Hamel. A complementary filter for attitude estimation of a fixed-wing
   UAV. IEEE/RSJ International Conference on Intelligent Robots and Systems,
   340-345. 2008.
   (http://users.cecs.anu.edu.au/~Jonghyuk.Kim/pdf/2008_Euston_iros_v1.04.pdf)
.. [Hamel] Tarek Hamel and Robert Mahony. Attitude estimation on SO(3) based on
   direct inertial measurements. IEEE International Conference on Robotics and
   Automation. ICRA 2006. pp. 2170-2175
   (http://users.cecs.anu.edu.au/~Robert.Mahony/Mahony_Robert/2006_MahHamPfl-C68.pdf)

"""

import numpy as np
from ..common.orientation import q_prod, q_conj, acc2q, am2q, q2R

class Mahony:
    """Mahony's Nonlinear Complementary Filter on SO(3)

    If ``acc`` and ``gyr`` are given as parameters, the orientations will be
    immediately computed with method ``updateIMU``.

    If ``acc``, ``gyr`` and ``mag`` are given as parameters, the orientations
    will be immediately computed with method ``updateMARG``.

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in m/s^2
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT
    frequency : float, default: 100.0
        Sampling frequency in Herz
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if `frequency` value is given
    kp : float, default: 1.0
        Proportional filter gain
    ki : float, default: 0.3
        Integral filter gain
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
        Sampling frequency in Herz.
    Dt : float
        Sampling step in seconds. Inverse of sampling frequency.
    kp : float
        Proportional filter gain.
    ki : float
        Integral filter gain.
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion)

    Raises
    ------
    ValueError
        When dimension of input array(s) ``acc``, ``gyr``, or ``mag`` are not
        equal.

    Examples
    --------
    Assuming we have 3-axis sensor data in N-by-3 arrays, we can simply give
    these samples to their corresponding type. The Mahony algorithm can work
    solely with gyroscope samples, although the use of accelerometer samples is
    much encouraged.

    The easiest way is to directly give the full array of samples to their
    matching parameters.

    >>> from ahrs.filters import Mahony
    >>> orientation = Mahony(gyr=gyro_data, acc=acc_data)   # Using IMU

    The estimated quaternions are saved in the attribute ``Q``.

    >>> type(orientation.Q), orientation.Q.shape
    (<class 'numpy.ndarray'>, (1000, 4))

    If we desire to estimate each sample independently, we call the
    corresponding method.

    .. code:: python

        orientation = Mahony()
        Q = np.tile([1., 0., 0., 0.], (num_samples, 1)) # Allocate for quaternions
        for t in range(1, num_samples):
            Q[t] = orientation.updateIMU(Q[t-1], gyr=gyro_data[t], acc=acc_data[t])

    Further on, we can also use magnetometer data.

    >>> orientation = Mahony(gyr=gyro_data, acc=acc_data, mag=mag_data)   # Using MARG

    This algorithm is dynamically adding the orientation, instead of estimating
    it from static observations. Thus, it requires an initial attitude to build
    on top of it. This can be set with the parameter ``q0``:

    >>> orientation = Mahony(gyr=gyro_data, acc=acc_data, q0=[0.7071, 0.0, 0.7071, 0.0])

    If no initial orientation is given, then an attitude using the first
    samples is estimated. This attitude is computed assuming the sensors are
    straped to a system in a quasi-static state.

    A constant sampling frequency equal to 100 Hz is used by default. To change
    this value we set it in its parameter ``frequency``. Here we set it, for
    example, to 150 Hz.

    >>> orientation = Mahony(gyr=gyro_data, acc=acc_data, frequency=150.0)

    Or, alternatively, setting the sampling step (:math:`\\Delta t = \\frac{1}{f}`):

    >>> orientation = Mahony(gyr=gyro_data, acc=acc_data, Dt=1/150)

    This specially useful for situations where the sampling rate is variable:

    .. code:: python

        orientation = Mahony()
        Q = np.tile([1., 0., 0., 0.], (num_samples, 1)) # Allocate for quaternions
        for t in range(1, num_samples):
            orientation.Dt = new_sample_rate
            Q[t] = orientation.updateIMU(Q[t-1], gyr=gyro_data[t], acc=acc_data[t])

    Mahony's algorithm uses an explicit complementary filter with two gains
    :math:`k_P` and :math:`k_I` to correct the estimation of the attitude.
    These gains can be set in the parameters too:

    >>> orientation = Mahony(gyr=gyro_data, acc=acc_data, Kp=0.5, Ki=0.1)

    Following the experimental settings of the original article, the gains are,
    by default, :math:`k_P=1` and :math:`k_I=0.3`.


    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kw):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.frequency = kw.get('frequency', 100.0)
        self.Dt = kw.get('Dt', 1.0/self.frequency)
        self.kp = kw.get('kp', 1.0)
        self.ki = kw.get('ki', 0.3)
        self.q0 = kw.get('q0')
        self.eInt = np.zeros(3)
        if self.gyr is not None and self.acc is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """
        Estimate the quaternions given all data

        Attributes ``gyr``, ``acc`` and, optionally, ``mag`` must contain data.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        num_samples = len(self.gyr)
        Q = np.zeros((num_samples, 4))
        # Compute with IMU Architecture
        if self.mag is None:
            Q[0] = acc2q(self.acc[0]) if self.q0 is None else self.q0/np.linalg.norm(self.q0)
            for t in range(1, num_samples):
                Q[t] = self.updateIMU(Q[t-1], self.gyr[t], self.acc[t])
            return Q
        # Compute with MARG Architecture
        if self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        Q[0] = am2q(self.acc[0], self.mag[0]) if self.q0 is None else self.q0/np.linalg.norm(self.q0)
        for t in range(1, num_samples):
            Q[t] = self.updateMARG(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def updateIMU(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """
        Attitude Estimation with a IMU architecture.

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
        g = gyr.copy()
        a_norm = np.linalg.norm(acc)
        if a_norm>0:
            v = q2R(q).T@np.array([0.0, 0.0, 1.0])      # Expected Earth's gravity
            e = np.cross(acc/a_norm, v)                 # Difference between expected and measured acceleration (Error)
            self.eInt += e*self.Dt                      # Integrate error
            b = -self.ki*self.eInt                      # Estimated Gyro bias (eq. 48c)
            d = self.kp*e + b                           # Innovation
            g += d                                      # Gyro correction
        qDot = 0.5*q_prod(q, [0.0, *g])                 # Rate of change of quaternion (eq. 48b)
        q += qDot*self.Dt                               # Update orientation
        q /= np.linalg.norm(q)
        return q

    def updateMARG(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """
        Attitude Estimation with a MARG architecture.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in radians.
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
        g = gyr.copy()
        a_norm = np.linalg.norm(acc)
        if a_norm>0:
            m = mag.copy()
            m_norm = np.linalg.norm(m)
            if not m_norm>0:
                return self.updateIMU(q, gyr, acc)
            m /= m_norm
            v = q2R(q).T@np.array([0.0, 0.0, 1.0])              # Expected Earth's gravity
            # h = q_prod(q, q_prod([0, *m], q_conj(q)))           # Rotate magnetic measurements to inertial frame
            h = q_prod(q_conj(q), q_prod([0, *m], q))           # Rotate magnetic measurements to inertial frame
            w = q2R(q).T@np.array([np.sqrt(h[1]**2+h[2]**2), 0.0, h[3]])     # Expected Earth's magnetic field
            e = np.cross(acc/a_norm, v) + np.cross(m, w)        # Difference between expected and measured values
            self.eInt += e*self.Dt                              # Add error
            b = -self.ki*self.eInt                              # Estimated Gyro bias (eq. 48c)
            g = g - b + self.kp*e                               # Gyro correction
        qDot = 0.5*q_prod(q, [0.0, *g])                         # Rate of change of quaternion (eq. 48b)
        q += qDot*self.Dt                                       # Update orientation
        q /= np.linalg.norm(q)
        return q

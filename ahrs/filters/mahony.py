# -*- coding: utf-8 -*-
"""
Mahony Orientation Filter
=========================

This estimator, termed "Explicit Complementary Filter" (ECF) by Robert Mahony
[Mahony]_, uses an inertial measurement :math:`a`, and an angular velocity
measurement :math:`\\Omega`.

The **gyroscopes** measure angular velocity in the body-fixed frame, whose error
model is:

.. math::
    \\Omega^y = \\Omega + b + \\mu \\in\\mathbb{R}^3

where :math:`\\Omega` is the true angular velocity, :math:`b` is a constant (or
slow time-varying) **bias**, and :math:`\\mu` is an additive **measurement
noise**.

The observer's main objective is to provide a set of dynamics for an estimated
orientation :math:`\\hat{\\mathbf{R}}\\in SO(3)`, and to drive such estimation
towards the real attitude: :math:`\\hat{\\mathbf{R}}\\to\\mathbf{R}\\in SO(3)`.

The proposed observer equations include a prediction term based on the measured
angular velocity :math:`\\Omega`, and an innovation correction term
:math:`\\omega(\\tilde{\\mathbf{R}})` derived from the error :math:`\\tilde{\\mathbf{R}}`.

The correction term :math:`\\omega` can be thought of as a non-linear
approximation of the error between :math:`\\mathbf{R}` and :math:`\\hat{\\mathbf{R}}`
as measured from the frame of reference associated with :math:`\\hat{\\mathbf{R}}`.

Let :math:`\\mathbf{v}_{0i}\\in\\mathbb{R}^3` denote a set of :math:`n\\geq 2`
known directions in the inertial (fixed) frame of reference, where the
directions are not collinear, and :math:`\\mathbf{v}_{i}\\in\\mathbb{R}^3` are
their associated measurements. The measurements are body-fixed frame
observations of the fixed inertial directions:

.. math::
    \\mathbf{v}_i = \\mathbf{R}^T\\mathbf{v}_{0i} + \\mu_i

where :math:`\\mu_i` is a process noise. We assume that :math:`|\\mathbf{v}_{0i}|=1`
and normalize all measurements to force :math:`|\\mathbf{v}_i|=1`.

We declare :math:`\\hat{\\mathbf{R}}` to be an estimate of :math:`\\mathbf{R}`

They are linked by:

.. math::
    \\hat{\\mathbf{v}}_i = \\hat{\\mathbf{R}}^T\\mathbf{v}_{0i}

Low cost IMUs measure vectors :math:`\\mathbf{a}` and :math:`\\mathbf{m}`
representing the gravitational and magnetic vector fields respectively.

.. math::
    \\begin{array}{rcl}
    \\mathbf{a} &=& \\mathbf{R}^T\\mathbf{a}_0 \\\\ && \\\\
    \\mathbf{m} &=& \\mathbf{R}^T\\mathbf{m}_0
    \\end{array}

For a single direction, the chosen error is

.. math::
    \\begin{array}{rcl}
    E_i &=& 1-\\cos(\\angle\\hat{\\mathbf{v}}_i, \\mathbf{v}_i) \\\\
    &=& 1-\\langle\\hat{\\mathbf{v}}_i, \\mathbf{v}_i\\rangle \\\\
    &=& 1-\\mathrm{tr}(\\hat{\\mathbf{R}}^Tv_{0i}v_{0i}^T\\mathbf{R}) \\\\
    &=& 1-\\mathrm{tr}(\\tilde{\\mathbf{R}}\\mathbf{R}^Tv_{0i}v_{0i}^T\\mathbf{R})
    \\end{array}

If :math:`\\tilde{\\mathbf{R}}=\\mathbf{I}`, then :math:`\\hat{\\mathbf{R}}`
already converges to :math:`\\mathbf{R}`.

For :math:`n` measures, the global cost becomes:

.. math::
    E_\\mathrm{mes} = \\sum_{i=1}^nk_iE_{vi} = \\sum_{i=1}^nk_i-\\mathrm{tr}(\\tilde{\\mathbf{R}}\\mathbf{M})

where :math:`\\mathbf{M}>0` is a positive definite matrix if :math:`n>2`, or
positive definite for :math:`n\\leq 2`:

.. math::
    \\mathbf{M} = \\mathbf{R}^T\\mathbf{M}_0\\mathbf{R}

with:

.. math::
    \\mathbf{M}_0 = \\sum_{i=1}^nk_iv_{0i}v_{0i}^T

The weights :math:`k_i>0` are chosen depending on the relative confidence in
the measured directions.

the goal of the observer design is to find a simple expression for :math:`\\omega`
that leads to robust convergence of 

The kinematics of the true system are:

.. math::
    \\dot{\\mathbf{R}} = \\mathbf{R}\\Omega_\\times = (\\mathbf{R}\\Omega)_\\times\\mathbf{R}

for a time-varying :math:`\\mathbf{R}(t)\\in SO(3)` and with measurements given
by:

.. math::
    \\Omega_y \\approx \\Omega + b

with a constant bias :math:`b`. It is assumed that there are :math:`n\\geq 2`
measurements :math:`v_i` available, expressing the kinematics of the Explicit
Complementary Filter as quaternions:

.. math::
    \\begin{array}{rcl}
    \\dot{\\hat{\\mathbf{q}}} &=& \\frac{1}{2}\\hat{\\mathbf{q}}\\mathbf{p}\\Big(\\lfloor\\Omega_y-\\hat{b}\\rfloor_\\times + k_P(\\omega_\\mathrm{mes})\\Big) \\\\ && \\\\
    \\dot{\\hat{b}} &=& -k_I\\omega_\\mathrm{mes} \\\\ && \\\\
    \\omega_\\mathrm{mes} &=& -\\mathrm{vex}\\Big(\\displaystyle\\sum_{i=1}^n\\frac{k_i}{2}(\\mathbf{v}_i\\hat{\\mathbf{v}}_i^T-\\hat{\\mathbf{v}}_i\\mathbf{v}_i^T)\\Big)
    \\end{array}

The estimated attitude rate of change :math:`\\dot{\\hat{\\mathbf{q}}}` is
multiplied with the sample-rate :math:`\\Delta t` to integrate the angular
displacement, which is finally added to the previous attitude, obtaining a more
robust attitude.

.. math::
    \\mathbf{q}_{t+1} = \\mathbf{q}_t + \\Delta t\\dot{\\hat{\\mathbf{q}}}_t

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

    This is specially useful for situations where the sampling rate is variable:

    .. code:: python

        orientation = Mahony()
        Q = np.tile([1., 0., 0., 0.], (num_samples, 1)) # Allocate for quaternions
        for t in range(1, num_samples):
            orientation.Dt = new_sample_rate
            Q[t] = orientation.updateIMU(Q[t-1], gyr=gyro_data[t], acc=acc_data[t])

    Mahony's algorithm uses an explicit complementary filter with two gains
    :math:`k_P` and :math:`k_I` to correct the estimation of the attitude.
    These gains can be set in the parameters too:

    >>> orientation = Mahony(gyr=gyro_data, acc=acc_data, kp=0.5, ki=0.1)

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

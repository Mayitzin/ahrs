# -*- coding: utf-8 -*-
"""
This estimator proposed by Robert Mahony et al. :cite:p:`mahony2008` is formulated
as a deterministic kinematic observer on the Special Orthogonal group SO(3)
driven by an instantaneous attitude and angular velocity measurements
:cite:p:`hamel2006`.

By exploiting the geometry of the special orthogonal group a related observer,
the *passive complementary filter*, is derived that decouples the gyro
measurements from the reconstructed attitude in the observer inputs
:cite:p:`mahony2005`.

Direct and passive filters are extended to estimate gyro bias on-line. This
leads to an observer on SO(3), termed the **Explicit Complementary Filter**
:cite:p:`euston2008`, that requires only accelerometer and gyro outputs,
suitable for hardware implementation, and providing good estimates as well as
online gyro bias computation.

Sensor Models
-------------

The `gyroscopes <https://en.wikipedia.org/wiki/Gyroscope>`_ measure angular
velocity in the body-fixed frame, whose error
model is:

.. math::
    \\Omega_y = \\Omega + b + \\mu \\in\\mathbb{R}^3

where :math:`\\Omega` is the true angular velocity, :math:`b` is a constant (or
slow time-varying) **bias**, and :math:`\\mu` is an additive **measurement
noise**.

An ideal `accelerometer <https://en.wikipedia.org/wiki/Accelerometer>`_
measures the instantaneous linear acceleration :math:`\\dot{\\mathbf{v}}` of
the body frame *minus* the gravitational acceleration field :math:`\\mathbf{g}_0`. In
practice, the output :math:`\\mathbf{a}` of a tri-axial accelerometer has also
an added bias :math:`\\mathbf{b}_a` and noise :math:`\\mu_a`.

.. math::
    \\mathbf{a} = \\mathbf{R}^T(\\dot{\\mathbf{v}}-\\mathbf{g}_0) + \\mathbf{b}_a + \\mu_a

where :math:`|\\mathbf{g}_0|\\approx 9.8`. Under quasi-static conditions it is
common to normalize this vector so that:

.. math::
    \\mathbf{v}_a = \\frac{\\mathbf{a}}{|\\mathbf{a}|} \\approx -\\mathbf{R}^Te_3

where :math:`e_3=\\frac{\\mathbf{g}_0}{|\\mathbf{g}_0|}=\\begin{bmatrix}0 & 0 & 1\\end{bmatrix}^T`.

A `magnetometer <https://en.wikipedia.org/wiki/Magnetometer>`_ provides
measurements for the magnetic field

.. math::
    \\mathbf{m} = \\mathbf{R}^T\\mathbf{h} + \\mathbf{B}_m + \\mu_b

where :math:`\\mathbf{h}` is `Earth's magnetic field <https://en.wikipedia.org/wiki/Earth%27s_magnetic_field>`_
as measured at the inertial frame, :math:`\\mathbf{B}_m` is the local magnetic
disturbance and :math:`\\mu_b` is the measurement noise.

The magnetic intensity is irrelevant for the estimation of the attitude, and
only the **direction** of the geomagnetic field will be used. Normally, this
measurement is also normalized:

.. math::
    \\mathbf{v}_m = \\frac{\\mathbf{m}}{|\\mathbf{m}|}

The measurement vectors :math:`\\mathbf{v}_a` and :math:`\\mathbf{v}_m` are
used to build an instantaneous algebraic rotation :math:`\\mathbf{R}`:

.. math::
    \\mathbf{R} \\approx \\mathbf{R}_y=\\underset{\\mathbf{R}\\in SO(3)}{\\operatorname{arg\\,min}} (\\lambda_1|e_3-\\mathbf{Rv}_a|^2 + \\lambda_2|\\mathbf{v}_m^*-\\mathbf{Rv}_m|^2)

where :math:`\\mathbf{v}_m^*` is the referential direction of the local
magnetic field.

The corresponding weights :math:`\\lambda_1` and :math:`\\lambda_2` are chosen
depending on the relative confidence in the sensor outputs.

Two degrees of freedom in the rotation matrix are resolved using the
acceleration readings (*tilt*) and the final degree of freedom is resolved
using the magnetometer (*heading*.)

The system considered is the kinematics:

.. math::
    \\dot{\\mathbf{R}} = \\mathbf{R}\\lfloor\\Omega\\rfloor_\\times

where :math:`\\lfloor\\Omega\\rfloor_\\times` denotes the `skew-symmetric
matrix <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_ of
:math:`\\Omega=\\begin{bmatrix}\\Omega_X & \\Omega_Y & \\Omega_Z\\end{bmatrix}^T`:

.. math::
    \\lfloor\\Omega\\rfloor_\\times = \\begin{bmatrix}
    0 & -\\Omega_Z & \\Omega_Y\\\\
    \\Omega_Z & 0 & -\\Omega_X\\\\
    -\\Omega_Y & \\Omega_X & 0
    \\end{bmatrix}

The inverse operation taking the skew-symmetric matrix into its associated
vector is :math:`\\Omega=\\mathrm{vex}(\\lfloor\\Omega\\rfloor_\\times)`.

The kinematics can also be written in terms of the quaternion representation in
SO(3):

.. math::
    \\dot{\\mathbf{q}} = \\frac{1}{2}\\mathbf{qp}(\\Omega)

where :math:`\\mathbf{q}=\\begin{pmatrix}q_w & \\mathbf{q}_v\\end{pmatrix}=\\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}`
represents a unit quaternion, and :math:`\\mathbf{p}(\\Omega)` represents the
unitary pure quaternion associated to the angular velocity
:math:`\\mathbf{p}(\\Omega)=\\begin{pmatrix}0 & \\Omega_X & \\Omega_Y & \\Omega_Z\\end{pmatrix}`

.. warning::
    The product of two quaternions :math:`\\mathbf{p}` and :math:`\\mathbf{q}`
    is the Hamilton product defined as:

    .. math::
        \\mathbf{pq} =
        \\begin{bmatrix}
        p_w q_w - p_x q_x - p_y q_y - p_z q_z \\\\
        p_w q_x + p_x q_w + p_y q_z - p_z q_y \\\\
        p_w q_y - p_x q_z + p_y q_w + p_z q_x \\\\
        p_w q_z + p_x q_y - p_y q_x + p_z q_w
        \\end{bmatrix}

Error Criteria
--------------

We denote :math:`\\hat{\\mathbf{R}}` as the *estimation* of the body-fixed
rotation matrix :math:`\\mathbf{R}`. The used estimation error is the relative
rotation from the body-fixed frame to the estimator frame:

.. math::
    \\tilde{\\mathbf{R}} := \\hat{\\mathbf{R}}^T\\mathbf{R}

Mahony's proposed observer, based on `Lyapunov stability analysis
<https://en.wikipedia.org/wiki/Lyapunov_stability>`_, yields the **cost
function**:

.. math::
    E_\\mathrm{tr} = \\frac{1}{2}\\mathrm{tr}(\\mathbf{I}_3-\\tilde{\\mathbf{R}})

The goal of attitude estimation is to provide a set of dynamics for an estimate
:math:`\\hat{\\mathbf{R}}(t)\\in SO(3)` to drive the error rotation
:math:`\\tilde{\\mathbf{R}}(t)\\to \\mathbf{I}_3`, which in turn would drive
:math:`\\hat{\\mathbf{R}}\\to\\mathbf{R}`.

The general form of the observer is termed as a **Complementary Filter on SO(3)**:

.. math::
    \\dot{\\hat{\\mathbf{R}}} = \\lfloor\\mathbf{R}\\Omega + k_P\\hat{\\mathbf{R}}\\omega\\rfloor_\\times\\hat{\\mathbf{R}}

where :math:`k_P>0` and the term :math:`\\mathbf{R}\\Omega + k_P\\hat{\\mathbf{R}}\\omega`
is expressed in the inertial frame.

The *innovation* or *correction* term :math:`\\omega`, derived from the error
:math:`\\tilde{\\mathbf{R}}`, can be thought of as a non-linear
approximation of the error between :math:`\\mathbf{R}` and :math:`\\hat{\\mathbf{R}}`
as measured from the frame of reference associated with :math:`\\hat{\\mathbf{R}}`.

When no correction term is used (:math:`\\omega=0`) the error rotation
:math:`\\tilde{\\mathbf{R}}` will be constant.

If :math:`\\tilde{\\mathbf{q}}=\\begin{pmatrix}\\tilde{q}_w & \\tilde{\\mathbf{q}}_v\\end{pmatrix}`
is the quaternion related to :math:`\\tilde{\\mathbf{R}}`, then:

.. math::
    E_\\mathrm{tr} = 2|\\tilde{\\mathbf{q}}_v|^2 = 2(1-\\tilde{q}_w^2)

Explicit Complementary Filter
-----------------------------

Let :math:`\\mathbf{v}_{0i}\\in\\mathbb{R}^3` denote a set of :math:`n\\geq 2`
known directions in the inertial (fixed) frame of reference, where the
directions are not collinear, and :math:`\\mathbf{v}_{i}\\in\\mathbb{R}^3` are
their associated measurements. The measurements are body-fixed frame
observations of the fixed inertial directions:

.. math::
    \\mathbf{v}_i = \\mathbf{R}^T\\mathbf{v}_{0i} + \\mu_i

where :math:`\\mu_i` is a process noise. We assume that :math:`|\\mathbf{v}_{0i}|=1`
and normalize all measurements to force :math:`|\\mathbf{v}_i|=1`.

For :math:`n` measures, the **global cost** becomes:

.. math::
    E_\\mathrm{mes} = \\sum_{i=1}^nk_i-\\mathrm{tr}(\\tilde{\\mathbf{R}}\\mathbf{M})

where :math:`\\mathbf{M}>0` is a positive definite matrix if :math:`n\\geq 3`,
or positive semi-definite if :math:`n=2`:

.. math::
    \\mathbf{M} = \\mathbf{R}^T\\mathbf{M}_0\\mathbf{R}

with:

.. math::
    \\mathbf{M}_0 = \\sum_{i=1}^nk_i\\mathbf{v}_{0i}\\mathbf{v}_{0i}^T

The weights :math:`k_i>0` are chosen depending on the relative confidence in
the measured directions.

Low-cost IMUs typically measure gravitational, :math:`\\mathbf{a}`, and
magnetic, :math:`\\mathbf{m}`, vector fields.

.. math::
    \\begin{array}{rcl}
    \\mathbf{v}_a &=& \\mathbf{R}^T\\frac{\\mathbf{a}_0}{|\\mathbf{a}_0|} \\\\ && \\\\
    \\mathbf{v}_m &=& \\mathbf{R}^T\\frac{\\mathbf{m}_0}{|\\mathbf{m}_0|}
    \\end{array}

In this case, the cost function :math:`E_\\mathrm{mes}` becomes:

.. math::
    E_\\mathrm{mes} = k_1(1-\\langle\\hat{\\mathbf{v}}_a, \\mathbf{v}_a\\rangle) + k_2(1-\\langle\\hat{\\mathbf{v}}_m, \\mathbf{v}_m\\rangle)

.. tip::
    When the IMU is subject to high magnitude accelerations (takeoff, landing
    manoeuvres, etc.) it may be wise to reduce the relative weighing of the
    accelerometer data (:math:`k_1 \\ll k_2`) compared to the magnetometer data.
    Conversely, when the IMU is mounted in the proximity of powerful electric
    motors leading to low confidence in the magnetometer readings choose
    :math:`k_1 \\gg k_2`.

Expressing the kinematics of the Explicit Complementary Filter as quaternions
we get:

.. math::
    \\begin{array}{rcl}
    \\dot{\\hat{\\mathbf{q}}} &=& \\frac{1}{2}\\hat{\\mathbf{q}}\\mathbf{p}\\Big(\\Omega_y-\\hat{b} + k_P\\omega_\\mathrm{mes}\\Big) \\\\ && \\\\
    \\dot{\\hat{b}} &=& -k_I\\omega_\\mathrm{mes} \\\\ && \\\\
    \\omega_\\mathrm{mes} &=& \\displaystyle\\sum_{i=1}^nk_i\\mathbf{v}_i\\times\\hat{\\mathbf{v}}_i
    \\end{array}

The estimated attitude rate of change :math:`\\dot{\\hat{\\mathbf{q}}}` is
multiplied with the sample-rate :math:`\\Delta t` to integrate the angular
displacement, which is finally added to the previous attitude, obtaining the
newest estimated attitude.

.. math::
    \\mathbf{q}_t = \\mathbf{q}_{t-1} + \\dot{\\hat{\\mathbf{q}}}_t\\Delta t

"""

import numpy as np
from ..common.orientation import q_prod
from ..common.orientation import acc2q
from ..common.orientation import am2q
from ..common.orientation import q2R
from ..utils.core import _assert_numerical_iterable

class Mahony:
    """
    Mahony's Nonlinear Complementary Filter on SO(3)

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
    k_P : float, default: 1.0
        Proportional filter gain
    k_I : float, default: 0.3
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
    k_P : float
        Proportional filter gain.
    k_I : float
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
    def __init__(self,
        gyr: np.ndarray = None,
        acc: np.ndarray = None,
        mag: np.ndarray = None,
        frequency: float = 100.0,
        k_P: float = 1.0,
        k_I: float = 0.3,
        q0: np.ndarray = None,
        b0: np.ndarray = None,
        **kwargs):
        self.gyr: np.ndarray = gyr
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.frequency: float = frequency
        self.q0: np.ndarray = q0
        self.b = b0 if b0 is not None else np.zeros(3)
        self.k_P: float = k_P
        self.k_I: float = k_I
        # Old parameter names for backward compatibility
        self.k_P: float = kwargs.get('kp', k_P)
        self.k_I: float = kwargs.get('ki', k_I)
        self.Dt: float = kwargs.get('Dt', 1.0/self.frequency if self.frequency else 0.01)
        self._assert_validity_of_inputs()
        # Estimate all orientations if sensor data is given
        if self.gyr is not None and self.acc is not None:
            self.Q = self._compute_all()

    def _assert_validity_of_inputs(self):
        """Asserts the validity of the inputs."""
        for item in ["frequency", "Dt", "k_P", "k_I"]:
            if isinstance(self.__getattribute__(item), bool):
                raise TypeError(f"Parameter '{item}' must be numeric.")
            if not isinstance(self.__getattribute__(item), (int, float)):
                raise TypeError(f"Parameter '{item}' is not a non-zero number.")
            if self.__getattribute__(item) <= 0.0:
                raise ValueError(f"Parameter '{item}' must be a non-zero number.")
        for item in ['q0', 'b']:
            if self.__getattribute__(item) is not None:
                if isinstance(self.__getattribute__(item), bool):
                    raise TypeError(f"Parameter '{item}' must be an array.")
                if not isinstance(self.__getattribute__(item), (list, tuple, np.ndarray)):
                    raise TypeError(f"Parameter '{item}' is not an array. Got {type(self.__getattribute__(item))}.")
                if np.copy(self.__getattribute__(item)).ndim != 1:
                    raise ValueError(f"Parameter '{item}' must be a 1-dimensional array.")
                self.__setattr__(item, np.copy(self.__getattribute__(item)))
        if self.q0 is not None:
            if self.q0.shape != (4,):
                raise ValueError("Parameter 'q0' must be an array with 4 elements.")
            if not np.allclose(np.linalg.norm(self.q0), 1.0):
                raise ValueError("Parameter 'q0' must be a unit quaternion.")
        if self.b is not None:
            if self.b.shape != (3,):
                raise ValueError("Parameter 'b' must be an array with 3 elements.")

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
        _assert_numerical_iterable(self.gyr, 'Angular velocity vector')
        _assert_numerical_iterable(self.acc, 'Gravitational acceleration vector')
        self.gyr = np.copy(self.gyr)
        self.acc = np.copy(self.acc)
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
        _assert_numerical_iterable(self.mag, 'Geomagnetic field vector')
        self.mag = np.copy(self.mag)
        if self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        Q[0] = am2q(self.acc[0], self.mag[0]) if self.q0 is None else self.q0/np.linalg.norm(self.q0)
        for t in range(1, num_samples):
            Q[t] = self.updateMARG(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def updateIMU(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Attitude Estimation with a IMU architecture.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2.
        dt : float, default: None
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        Examples
        --------
        >>> orientation = Mahony()
        >>> Q = np.tile([1., 0., 0., 0.], (num_samples, 1)) # Allocate for quaternions
        >>> for t in range(1, num_samples):
        ...     Q[t] = orientation.updateIMU(Q[t-1], gyr=gyro_data[t], acc=acc_data[t])

        """
        dt = self.Dt if dt is None else dt
        if gyr is None or not np.linalg.norm(gyr) > 0:
            return q
        Omega = np.copy(gyr)
        a_norm = np.linalg.norm(acc)
        if a_norm > 0:
            R = q2R(q)
            v_a = R.T@np.array([0.0, 0.0, 1.0])     # Expected Earth's gravity
            # ECF
            omega_mes = np.cross(acc/a_norm, v_a)   # Cost function (eqs. 32c and 48a)
            bDot = -self.k_I*omega_mes                   # Estimated change in Gyro bias (eqs.32b and 48c)
            self.b += bDot * dt                          # Estimated Gyro bias
            Omega = Omega - self.b + self.k_P*omega_mes  # Gyro correction
        p = np.array([0.0, *Omega])
        qDot = 0.5*q_prod(q, p)                     # Rate of change of quaternion (eqs. 45 and 48b)
        q += qDot*dt                                # Update orientation
        q /= np.linalg.norm(q)                      # Normalize Quaternion (Versor)
        return q

    def updateMARG(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray, dt: float = None) -> np.ndarray:
        """
        Attitude Estimation with a MARG architecture.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2.
        mag : numpy.ndarray
            Sample of tri-axail Magnetometer in uT.
        dt : float, default: None
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        Examples
        --------
        >>> orientation = Mahony()
        >>> Q = np.tile([1., 0., 0., 0.], (num_samples, 1)) # Allocate for quaternions
        >>> for t in range(1, num_samples):
        ...     Q[t] = orientation.updateMARG(Q[t-1], gyr=gyro_data[t], acc=acc_data[t], mag=mag_data[t])

        """
        dt = self.Dt if dt is None else dt
        if gyr is None or not np.linalg.norm(gyr) > 0:
            return q
        Omega = np.copy(gyr)
        a_norm = np.linalg.norm(acc)
        if a_norm > 0:
            m_norm = np.linalg.norm(mag)
            if not m_norm>0:
                return self.updateIMU(q, gyr, acc)
            a = np.copy(acc)/a_norm
            m = np.copy(mag)/m_norm
            R = q2R(q)
            v_a = R.T@np.array([0.0, 0.0, 1.0])     # Expected Earth's gravity
            # Rotate magnetic field to inertial frame
            h = R@m
            v_m = R.T@np.array([0.0, np.linalg.norm([h[0], h[1]]), h[2]])
            v_m /= np.linalg.norm(v_m)
            # ECF
            omega_mes = np.cross(a, v_a) + np.cross(m, v_m) # Cost function (eqs. 32c and 48a)
            bDot = -self.k_I*omega_mes                   # Estimated change in Gyro bias (eqs.32b and 48c)
            self.b += bDot * dt                          # Estimated Gyro bias
            Omega = Omega - self.b + self.k_P*omega_mes  # Gyro correction (eq. 48b)
        p = np.array([0.0, *Omega])
        qDot = 0.5*q_prod(q, p)                     # Rate of change of quaternion (eqs. 45 and 48b)
        q += qDot*dt                                # Update orientation
        q /= np.linalg.norm(q)                      # Normalize Quaternion (Versor)
        return q

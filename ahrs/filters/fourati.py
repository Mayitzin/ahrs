# -*- coding: utf-8 -*-
"""
Fourati's nonlinear attitude estimation
=======================================

Attitude estimation algorithm as proposed by [Fourati]_, whose approach
combines a quaternion-based nonlinear filter with the Levenberg Marquardt
Algorithm (LMA.)

The estimation algorithm has a complementary structure that exploits
measurements from an accelerometer, a magnetometer and a gyroscope, combined in
a strap-down system, based on the time integral of the angular velocity, using
the Earth's magnetic field and gravity vector to compensate the attitude
predicted by the gyroscope.

The **rigid body attitude** in space is determined when the body's orientation
frame :math:`(X_B, Y_B, Z_B)` is specified with respect to the navigation frame
:math:`(Y_N, Y_N, Z_N)`, where the navigation frame follows the NED convention
(North-East-Down.)

The unit quaternion, :math:`\\mathbf{q}`, is defined as a scalar-vector pair of
the form:

.. math::
    \\mathbf{q} = \\begin{pmatrix}s & \\mathbf{v}\\end{pmatrix}^T

where :math:`s` is the scalar part and :math:`\\mathbf{v}=\\begin{pmatrix}v_x & v_y & v_z\\end{pmatrix}^T`
is the vector part of the quaternion.

.. note::
    Most literature, and this package's documentation, use the notation
    :math:`\\mathbf{q}=\\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}` to
    define a quaternion, but this algorithm uses a different one, and it will
    preserved to keep the coherence with the original document.

The sensor configuration consists of a three-axis gyroscope, a three-axis
accelerometer, a three-axis magnetometer. Their outputs can be modelled,
respectively, as:

.. math::
    \\begin{array}{rcl}
    \\omega_G =& \\begin{bmatrix}\\omega_{GX} & \\omega_{GY} & \\omega_{GZ}\\end{bmatrix}^T &= \\omega + b + \\delta_G \\\\&&\\\\
    \\mathbf{f} =& \\begin{bmatrix}f_x & f_y & f_z\\end{bmatrix}^T &= M_N^B(\\mathbf{q}) (g+a) + \\delta_f \\\\&&\\\\
    \\mathbf{h} =& \\begin{bmatrix}h_x & h_y & h_z\\end{bmatrix}^T &= M_N^B(\\mathbf{q}) m + \\delta_h
    \\end{array}

where :math:`b\\in\\mathbb{R}^3` is the unknown gyro-bias vector and :math:`\\delta_G`,
:math:`\\delta_f` and :math:`\\delta_h\\in\\mathbb{R}^3` are assumed `white
Gaussian noises <https://en.wikipedia.org/wiki/Additive_white_Gaussian_noise>`_.
:math:`\\omega` is the *real* angular velocity, :math:`g` is the gravity vector,
:math:`a` denotes the Dynamic Body Acceleration (DBA), :math:`m` describes the
direction of the Earth's magnetic field on the local position, and
:math:`M_N^B(\\mathbf{q})` is the orthogonal matrix describing the attitude of
the body frame.

.. math::
    \\mathbf{M}_N^B(\\mathbf{q}) =
    \\begin{bmatrix}
    2(s^2 + v_x^2) - 1 & 2(v_xv_y + sv_z) & 2(v_xv_z - sv_y) \\\\
    2(v_xv_y - sv_z) & 2(s^2 + v_y^2) - 1 & 2(sv_x + v_yv_z) \\\\
    2(sv_y + v_xv_z) & 2(v_yv_z - sv_x) & 2(s^2 + v_z^2) - 1
    \\end{bmatrix}

The kinematic differential equation, in terms of the unit quaternion, that
describes the relationship between the rigid body attitude variation and the
angular velocity in the body frame is represented by:

.. math::
    \\begin{array}{rcl}
    \\dot{\\mathbf{q}} &=& \\frac{1}{2}\\mathbf{q}\\omega_\\mathbf{q} \\\\
    \\begin{bmatrix}\\dot{s}\\\\ \\dot{v}_x \\\\ \\dot{v}_y \\\\ \\dot{v}_z\\end{bmatrix}
    &=& \\frac{1}{2}\\begin{bmatrix}-\\mathbf{v}^T \\\\ \\mathbf{I}_3s+\\lfloor\\mathbf{v}\\rfloor_\\times\\end{bmatrix}
    \\begin{bmatrix}\\omega_x \\\\ \\omega_y \\\\ \\omega_z\\end{bmatrix}
    \\end{array}

where :math:`\\omega_\\mathbf{q}=\\begin{bmatrix}0 & \\omega^T\\end{bmatrix}^T`
is the equivalent to the angular velocity :math:`\\omega\\in\\mathbb{R}^3` of
the rigid body measured in :math:`B` and relative to :math:`N`, :math:`\\mathbf{I}_3`
is the :math:`3\\times 3` identity matrix, and :math:`\\lfloor\\mathbf{v}\\rfloor_\\times`
is the `Skew symmetric matrix <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_
of the vector :math:`\\mathbf{v}`.

.. math::
    \\lfloor\\mathbf{v}\\rfloor_\\times =
    \\begin{bmatrix}0 & -v_z & v_y \\\\ v_z & 0 & -v_x \\\\ -v_y & v_x & 0\\end{bmatrix}

.. note::
    Any vector :math:`\\mathbf{x}=\\begin{bmatrix}x_1 & x_2 & x_3\\end{bmatrix}^T\\in\\mathbb{R}^3`
    that multiplies with a quaternion must be considered a `pure quaternion
    <https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternion_as_rotations>`_,
    :math:`\\mathbf{x_q}=\\begin{bmatrix}0 & x_1 & x_2 & x_3\\end{bmatrix}^T\\in\\mathbb{R}^4`,
    so that they operate with a *Hamilton product*.

To achieve an optimal attitude estimation, a nonlinear system is developed,
whose **output** is the stack of the accelerometer and magnetometer measurements:

.. math::
    \\mathbf{y} = \\begin{bmatrix}f_x & f_y & f_z & h_x & h_y & h_z\\end{bmatrix}^T

The `World Magnetic Model <../wmm.html>`_ considers a magnetic vector
:math:`\\mathbf{m}=\\begin{bmatrix}m_x & m_y & m_z \\end{bmatrix}\\in\\mathbb{R}^3`
at any location on Earth to describe the geomagnetic field. For practical
purposes, the vector is simplified to
:math:`\\mathbf{m}=\\begin{bmatrix}m\\cos\\theta & 0 & m\\sin\\theta\\end{bmatrix}`,
with a dip angle :math:`\\theta` and a magnetic intensity :math:`m`, which
varies between 23000 and 67000 nT, depending on the region on Earth. This
simplified vector discards the Easterly magnetic field (:math:`m_y`), although
for an accurate reference, it is preferred to use it.

Similar to :math:`\\mathbf{y}`, the estimated values :math:`\\hat{\\mathbf{y}}`
are given by:

.. math::
    \\hat{\\mathbf{y}} = \\begin{bmatrix}\\hat{f}_x & \\hat{f}_y & \\hat{f}_z & \\hat{h}_x & \\hat{h}_y & \\hat{h}_z\\end{bmatrix}^T

whose components are calculated as:

.. math::
    \\begin{array}{rcl}
    \\hat{\\mathbf{f}} &=& \\begin{bmatrix}\\hat{f}_x & \\hat{f}_y & \\hat{f}_z \\end{bmatrix}^T = \\hat{\\mathbf{q}}^{-1}\\mathbf{g_q}\\hat{\\mathbf{q}} \\\\ && \\\\
    \\hat{\\mathbf{h}} &=& \\begin{bmatrix}\\hat{h}_x & \\hat{h}_y & \\hat{h}_z \\end{bmatrix}^T = \\hat{\\mathbf{q}}^{-1}\\mathbf{m_q}\\hat{\\mathbf{q}}
    \\end{array}

where :math:`\\mathbf{g_q}=\\begin{bmatrix}0 & 0 & 0 & 9.8\\end{bmatrix}^T` is
the **reference gravity vector** as a pure quaternion, and
:math:`\\mathbf{m_q}=\\begin{bmatrix}0 & m\\cos\\theta & 0 & m\\sin\\theta\\end{bmatrix}^T`
is the local **reference geomagnetic field** also represented as a pure
quaternion.

The modeling error, :math:`\\delta(\\hat{\\mathbf{q}})=\\mathbf{y}-\\hat{\\mathbf{y}}`,
represents the difference between the real measurements :math:`\\mathbf{y}` and
the estimated values :math:`\\hat{\\mathbf{y}}`.

The nonlinear filter of this model takes the form:

.. math::
    \\dot{\\mathbf{q}} =
    \\begin{bmatrix}\\dot{s}\\\\ \\dot{v}_x \\\\ \\dot{v}_y \\\\ \\dot{v}_z\\end{bmatrix} =
    \\frac{1}{2}\\hat{\\mathbf{q}}\\omega_\\mathbf{q}
    \\begin{bmatrix}1 \\\\ \\mathbf{K}\\end{bmatrix}

where :math:`\\hat{\\mathbf{q}}=\\begin{bmatrix}\\hat{s}& \\hat{v}_x & \\hat{v}_y & \\hat{v}_z\\end{bmatrix}^T\\in\\mathbb{R}^4`
is the **estimated state**, and :math:`\\mathbf{K}\\in\\mathbb{R}^{3\\times 6}`
is the **observer gain**.

This gain :math:`\\mathbf{K}` is used to correct the modeling error
:math:`\\delta(\\hat{\\mathbf{q}})`, which can be done if we locate the minimum
of the squared error function :math:`\\xi(\\hat{\\mathbf{q}})=\\delta(\\hat{\\mathbf{q}})^T\\delta(\\hat{\\mathbf{q}})`.

For this attitude estimator the `Levenberg-Marquardt Algorithm
<https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`_ (LMA)
is used to minimize the nonlinear function :math:`\\xi(\\hat{\\mathbf{q}})`.
So, the unique minimum can be computed with:

.. math::
    \\begin{array}{rcl}
    \\eta(\\hat{\\mathbf{q}}) &=& \\mathbf{K}\\delta(\\hat{\\mathbf{q}}) \\\\
    &=& k[\\mathbf{X}^T\\mathbf{X} + \\lambda\\mathbf{I}_3]^{-1}\\mathbf{X}^T\\delta(\\hat{\\mathbf{q}})
    \\end{array}

where the tiny value :math:`\\lambda` guarantees the inversion of the matrix,
the gain factor :math:`k` tunes the balance between measurement noise
supression and the filter's response time, and
:math:`\\mathbf{X}\\in\\mathbb{R}^{6\\times 3}` is the Jacobian matrix:

.. math::
    \\begin{array}{rcl}
    \\mathbf{X} &=& -2\\begin{bmatrix}\\lfloor\\hat{\\mathbf{f}}\\rfloor_\\times & \\lfloor\\hat{\\mathbf{h}}\\rfloor_\\times\\end{bmatrix} \\\\
    &=& -2\\begin{bmatrix}
    0 & -\\hat{f}_z & \\hat{f}_y & 0 & -\\hat{h}_z & \\hat{h}_y \\\\
    \\hat{f}_z & 0 & -\\hat{f}_x & \\hat{h}_z & 0 & -\\hat{h}_x \\\\
    -\\hat{f}_y & \\hat{f}_x & 0 & -\\hat{h}_y & \\hat{h}_x & 0
    \\end{bmatrix}
    \\end{array}

The resulting structure of the nonlinear filter is complementary: it blends the
low-frequency region (low bandwidth) of the accelerometer and magnetometer data,
where the attitude is typically more accurate, with the high-frequency region
(high bandwidth) of the gyroscope data, where the integration of the angular
velocity yields better attitude estimates.

By filtering the high-frequency components of the signals from the
accelerometer (DBA) and the low-frequency components of the gyroscope signal
(slow-moving drift), the nonlinear filter produces an accurate estimate of the
attitude.

The correction term, :math:`\\Delta\\in\\mathbb{R}^{4\\times 7}`, is computed
using the gain :math:`K` such as:

.. math::
    \\Delta =
    \\begin{bmatrix}1 & \\mathbf{0} \\\\ \\mathbf{0} & \\mathbf{K}\\end{bmatrix}
    \\begin{bmatrix}1 \\\\ \\delta(\\hat{\\mathbf{q}})\\end{bmatrix}

It is used to correct the estimated angular velocity, :math:`\\dot{\\hat{\\mathbf{q}}}`,
as:

.. math::
    \\dot{\\hat{\\mathbf{q}}} = \\big(\\frac{1}{2}\\hat{\\mathbf{q}}\\omega_\\mathbf{q}\\big)\\Delta

With the corrected angular velocity, we integrate it using the sampling step
:math:`\\Delta_t` and add it to the previous quaternion :math:`\\mathbf{q}_{t-1}`
to obtain the new attitude :math:`\\mathbf{q}_t`:

.. math::
    \\mathbf{q}_t = \\mathbf{q}_{t-1} + \\dot{\\hat{\\mathbf{q}}}\\Delta_t

.. warning::
    Do not confuse the correction term :math:`\\Delta` with the sampling step
    :math:`\\Delta_t`, which is actually the inverse of the sampling frequency
    :math:`f=\\frac{1}{\\Delta_t}`.

References
----------
.. [Fourati] Hassen Fourati, Noureddine Manamanni, Lissan Afilal, Yves
    Handrich. A Nonlinear Filtering Approach for the Attitude and Dynamic Body
    Acceleration Estimation Based on Inertial and Magnetic Sensors: Bio-Logging
    Application. IEEE Sensors Journal, Institute of Electrical and Electronics
    Engineers, 2011, 11 (1), pp. 233-244. 10.1109/JSEN.2010.2053353.
    (https://hal.archives-ouvertes.fr/hal-00624142/file/Papier_IEEE_Sensors_Journal.pdf)

"""

import numpy as np
from ..common.orientation import q_prod, q_conj, am2q
from ..common.mathfuncs import *

# Reference Observations in Munich, Germany
from ..utils.wmm import WMM
from ..utils.wgs84 import WGS
MAG = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT).magnetic_elements
GRAVITY = WGS().normal_gravity(MUNICH_LATITUDE, MUNICH_HEIGHT)

class Fourati:
    """
    Fourati's attitude estimation

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT
    frequency : float, default: 100.0
        Sampling frequency in Herz.
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if `frequency` value is given.
    gain : float, default: 0.1
        Filter gain factor.
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).
    magnetic_dip : float
        Magnetic Inclination angle, in degrees.
    gravity : float
        Normal gravity, in m/s^2.

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
        Filter gain factor.
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion).

    Raises
    ------
    ValueError
        When dimension of input array(s) ``acc``, ``gyr``, or ``mag`` are not equal.

    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kwargs):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.frequency = kwargs.get('frequency', 100.0)
        self.Dt = kwargs.get('Dt', 1.0/self.frequency)
        self.gain = kwargs.get('gain', 0.1)
        self.q0 = kwargs.get('q0')
        # Reference measurements
        mdip = kwargs.get('magnetic_dip')             # Magnetic dip, in degrees
        self.m_q = np.array([0.0, MAG['X'], MAG['Y'], MAG['Z']]) if mdip is None else np.array([0.0, cosd(mdip), 0.0, sind(mdip)])
        self.m_q /= np.linalg.norm(self.m_q)
        self.g_q = np.array([0.0, 0.0, 0.0, 1.0])     # Normalized Gravity vector
        # Process of given data
        if self.acc is not None and self.gyr is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """
        Estimate the quaternions given all data

        Attributes ``gyr``, ``acc`` and ``mag`` must contain data.

        Returns
        -------
        Q : array
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        if self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        num_samples = len(self.gyr)
        Q = np.zeros((num_samples, 4))
        Q[0] = am2q(self.acc[0], self.mag[0]) if self.q0 is None else self.q0.copy()
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def update(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray, dt: float = None) -> np.ndarray:
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
        dt : float, default: None
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        """
        dt = self.Dt if dt is None else dt
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        qDot = 0.5 * q_prod(q, [0, *gyr])                           # (eq. 5)
        a_norm = np.linalg.norm(acc)
        m_norm = np.linalg.norm(mag)
        if a_norm>0 and m_norm>0 and self.gain>0:
            # Levenberg Marquardt Algorithm
            fhat = q_prod(q_conj(q), q_prod(self.g_q, q))           # (eq. 21)
            hhat = q_prod(q_conj(q), q_prod(self.m_q, q))           # (eq. 22)
            y = np.r_[acc/a_norm, mag/m_norm]                       # Measurements (eq. 6)
            yhat = np.r_[fhat[1:], hhat[1:]]                        # Estimated values (eq. 8)
            dq = y - yhat                                           # Modeling Error
            X = -2*np.c_[skew(fhat[1:]), skew(hhat[1:])].T          # Jacobian Matrix (eq. 23)
            lam = 1e-8                                              # Deviation to guarantee inversion
            K = self.gain*np.linalg.inv(X.T@X + lam*np.eye(3))@X.T  # Filter gain (eq. 24)
            Delta = [1, *K@dq]                                      # Correction term (eq. 25)
            qDot = q_prod(qDot, Delta)                              # Corrected quaternion rate (eq. 7)
        q += qDot*dt
        return q/np.linalg.norm(q)

# -*- coding: utf-8 -*-
"""
Attitude estimation algorithm as proposed in :cite:p:`fourati2011`, whose
approach combines a quaternion-based nonlinear filter with the
`Levenberg-Marquardt Algorithm (LMA.)
<https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`_.

The estimation algorithm has a complementary structure that exploits
measurements from an accelerometer, a magnetometer and a gyroscope, combined in
a strap-down system, based on the time integral of the angular velocity, using
the Earth's magnetic field and gravity vector to compensate the attitude
predicted by the gyroscope.

The **rigid body attitude** in space is determined when the body's orientation
frame :math:`(X_B, Y_B, Z_B)` is specified with respect to the navigation frame
:math:`(X_N, Y_N, Z_N)`, where the navigation frame follows the NED convention
(North-East-Down.)

The unit quaternion, :math:`\\mathbf{q}`, is defined as a scalar-vector pair of
the form:

.. math::
    \\mathbf{q} = \\begin{pmatrix}s & \\mathbf{v}\\end{pmatrix}^T

where :math:`s` is the scalar part and :math:`\\mathbf{v}=\\begin{pmatrix}v_x & v_y & v_z\\end{pmatrix}^T`
is the vector part of the quaternion.

.. note::
    Most literature, and this Python package's documentation, use the notation
    :math:`\\mathbf{q}=\\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}` to
    define a quaternion, but this algorithm uses a different one. The latter
    will preserved in this page to keep coherence with the original document.

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
whose **output** is the stack of the accelerometer and magnetometer
measurements:

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
    \\mathbf{X} &=& -2\\begin{bmatrix}\\lfloor\\hat{\\mathbf{f}}\\rfloor_\\times & \\lfloor\\hat{\\mathbf{h}}\\rfloor_\\times\\end{bmatrix}^T \\\\
    &=& -2\\begin{bmatrix}
    0 & -\\hat{f}_z & \\hat{f}_y & 0 & -\\hat{h}_z & \\hat{h}_y \\\\
    \\hat{f}_z & 0 & -\\hat{f}_x & \\hat{h}_z & 0 & -\\hat{h}_x \\\\
    -\\hat{f}_y & \\hat{f}_x & 0 & -\\hat{h}_y & \\hat{h}_x & 0
    \\end{bmatrix}^T
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

"""

from typing import Union
import numpy as np
from ..common.quaternion import Quaternion
from ..common.orientation import ecompass
from ..common.constants import MUNICH_LONGITUDE
from ..common.constants import MUNICH_LATITUDE
from ..common.constants import MUNICH_HEIGHT
from ..common.mathfuncs import cosd
from ..common.mathfuncs import sind
from ..common.mathfuncs import skew
from ..utils.core import _assert_numerical_iterable
from ..utils.core import _assert_numerical_positive_variable

# Reference Observations in Munich, Germany
from ..utils.wmm import WMM
wmm = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT)
REFERENCE_MAGNETIC_VECTOR = np.array([wmm.X, wmm.Y, wmm.Z])

def _set_magnetic_field_vector(magnetic_dip: Union[int, float, list, tuple, np.ndarray]):
    """
    Set the magnetic reference vector.

    Parameters
    ----------
    magnetic_dip : float, array-like
        Magnetic dip, in degrees, or local geomagnetic reference as
        three-dimensional vector.

    """
    if isinstance(magnetic_dip, bool):
        raise TypeError("magnetic_dip must be given as a float, list, tuple or NumPy array. Got bool")
    elif isinstance(magnetic_dip, (float, int)):
        magnetic_field = np.array([0.0, cosd(magnetic_dip), 0., sind(magnetic_dip)])
    elif isinstance(magnetic_dip, (list, tuple, np.ndarray)):
        if not all(isinstance(x, (float, int)) for x in magnetic_dip):
            raise TypeError("magnetic_dip must be an array of floats. Contains non-numeric values.")
        magnetic_field = np.copy(magnetic_dip)
    elif magnetic_dip is None:
        magnetic_field = np.array([0.0, *REFERENCE_MAGNETIC_VECTOR])
    else:
        raise TypeError(f"magnetic_dip must be given as a float, list, tuple or NumPy array. Got {type(magnetic_dip)}")
    if magnetic_field.shape != (4,):
        raise ValueError(f"magnetic_dip array must contain 4 elements. Got {magnetic_field.shape}")
    return magnetic_field / np.linalg.norm(magnetic_field)

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
    magnetic_dip : float, array-like, default: None
        Magnetic Inclination angle, in degrees, or as a local geomagnetic
        vector in a pure quaternion of the form [0, mx, my, mz], where mx, my,
        and mz are the three-dimensional components of the local geomagnetic
        vector.

    Raises
    ------
    ValueError
        When dimension of input array(s) ``acc``, ``gyr``, or ``mag`` are not equal.

    Examples
    --------
    >>> gyro_data.shape, acc_data.shape, mag_data.shape      # NumPy arrays with sensor data
    ((1000, 3), (1000, 3), (1000, 3))
    >>> from ahrs.filters import Fourati
    >>> fourati = Fourati(gyr=gyro_data, acc=acc_data, mag=mag_data)
    >>> fourati.Q       # Estimated attitudes as Quaternions
    array([[-0.82311077,  0.45760535, -0.33408929, -0.0383452 ],
           [-0.82522048,  0.4547043 , -0.33277675, -0.03892033],
           [-0.82463698,  0.4546915 , -0.33422422, -0.03903417],
           ...,
           [-0.82420642,  0.56217735,  0.02548005, -0.06317571],
           [-0.82364606,  0.56311099,  0.0241655 , -0.06268338],
           [-0.81844766,  0.57077781,  0.02532182, -0.06095017]])
    >>> fourati.Q.shape
    (1000, 4)

    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kwargs):
        self.gyr: np.ndarray = gyr
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.frequency: float = kwargs.get('frequency', 100.0)
        self.Dt: float = kwargs.get('Dt', 1.0/self.frequency if self.frequency else 0.01)
        self.gain: float = kwargs.get('gain', 0.1)
        self.q0: np.ndarray = kwargs.get('q0')
        # Reference vectors
        self.m_q: Quaternion = Quaternion(_set_magnetic_field_vector(kwargs.get('magnetic_dip')))   # Gravity vector as pure quaternion
        self.g_q: Quaternion = Quaternion([0.0, 0.0, 0.0, 1.0])                                     # Magnetic field vector as pure quaternion
        # Process of given data
        self._assert_validity_of_inputs()
        if self.acc is not None and self.gyr is not None and self.mag is not None:
            self.Q = self._compute_all()

    def _assert_validity_of_inputs(self):
        """Asserts the validity of the inputs."""
        for item in ["frequency", "Dt", "gain"]:
            _assert_numerical_positive_variable(getattr(self, item), item)
        if self.q0 is not None:
            self.q0 = Quaternion(self.q0).to_array()

    def _compute_all(self):
        """
        Estimate the quaternions given all data

        Attributes ``gyr``, ``acc`` and ``mag`` must contain data.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        _assert_numerical_iterable(self.gyr, 'Angular velocity vector')
        _assert_numerical_iterable(self.acc, 'Gravitational acceleration vector')
        _assert_numerical_iterable(self.mag, 'Geomagnetic field vector')
        self.gyr = np.copy(self.gyr)
        self.acc = np.copy(self.acc)
        self.mag = np.copy(self.mag)
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        if self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        num_samples = len(self.gyr)
        Q = np.zeros((num_samples, 4))
        Q[0] = ecompass(self.acc[0], self.mag[0], frame='NED', representation='quaternion')
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
        _assert_numerical_iterable(q, 'Quaternion')
        _assert_numerical_iterable(gyr, 'Tri-axial gyroscope sample')
        _assert_numerical_iterable(acc, 'Tri-axial accelerometer sample')
        _assert_numerical_iterable(mag, 'Tri-axial magnetometer sample')
        dt = self.Dt if dt is None else dt
        if gyr is None or np.linalg.norm(gyr) == 0:
            return q
        q = Quaternion(q)
        qDot = 0.5 * q.product([0, *gyr])                           # (eq. 5)
        if self.gain > 0:
            a_norm = np.linalg.norm(acc)
            if a_norm == 0:
                raise ValueError("Accelerometer data is null. Cannot estimate quaternion.")
            m_norm = np.linalg.norm(mag)
            if m_norm == 0:
                raise ValueError("Magnetometer data is null. Cannot estimate quaternion.")
            # Levenberg Marquardt Algorithm
            fhat = Quaternion(q.conj).product(self.g_q.product(q))  # (eq. 21)
            hhat = Quaternion(q.conj).product(self.m_q.product(q))  # (eq. 22)
            y = np.r_[acc/a_norm, mag/m_norm]                       # Measurements (eq. 6)
            yhat = np.r_[fhat[1:], hhat[1:]]                        # Estimated values (eq. 8)
            dq = y - yhat                                           # Modeling Error
            X = -2*np.c_[skew(fhat[1:]), skew(hhat[1:])].T          # Jacobian Matrix (eq. 23)
            lam = 1e-5                                              # Deviation to guarantee inversion
            K = self.gain*np.linalg.inv(X.T@X + lam*np.eye(3))@X.T  # Gain (eq. 24)
            eta = K @ dq                                            # Unique minimum (eq. 10)
            Delta = [1, *eta]                                       # Correction term (eq. 25)
            qDot = Quaternion(qDot, versor=False).product(Delta)    # Corrected quaternion rate (eq. 7)
        q += qDot*dt
        return q/np.linalg.norm(q)

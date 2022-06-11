# -*- coding: utf-8 -*-
"""
Factored Quaternion Algorithm
=============================

The factored quaternion algorithm (FQA) produces a quaternion output to
represent the orientation, restricting the use of magnetic data to the
determination of the rotation about the vertical axis.

The FQA and the `TRIAD <../triad.html>`_ algorithm produce an equivalent
solution to the same problem, with the difference that the former produces a
quaternion, and the latter a rotation matrix.

Magnetic variations cause only azimuth errors in FQA attitude estimation. A
singularity avoidance method is used, which allows the algorithm to track
through all orientations.

.. warning::
    This algorithm is not applicable to situations in which relatively large
    linear accelerations due to dynamic motion are present, unless it is used
    in a complementary or optimal filter together with angular rate information.

The *Earth-fixed coordinate system* :math:`(^ex,\\,^ey,\\,^ez)` is defined following
the `North-East-Down <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates#Local_north,_east,_down_(NED)_coordinates>`_
(NED) convention.

A *body coordinate system* :math:`(^bx,\\,^by,\\,^bz)` is attached to the rigid
body whose orientation is being measured, and the *sensor frame* :math:`(^sx,\\,^sy,\\,^sz)`
corresponds to the sensor module conformed by the accelerometer/magnetometer
system.

The body coordinate system differs from the sensor frame by a constant offset,
if they are not coinciding. For this method *they are assumed to occupy the same
place*.

From Euler's theorem of rotations, we can use a unit quaternion :math:`\\mathbf{q}`
as an `axis-angle rotation <https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Unit_quaternions>`_:

.. math::
    \\mathbf{q} =
    \\begin{pmatrix}
    \\cos\\frac{\\beta}{2} \\\\
    u_x\\sin\\frac{\\beta}{2} \\\\
    u_y\\sin\\frac{\\beta}{2} \\\\
    u_z\\sin\\frac{\\beta}{2}
    \\end{pmatrix}

where :math:`\\mathbf{u}=\\begin{bmatrix}u_x & u_y & u_z\\end{bmatrix}^T` is
the rotation axis, and :math:`\\beta` is the rotation angle. See the
`Quaternion <../quaternion.html>`_ reference page for further details of it.

A rigid body can be placed in an arbitrary orientation by first rotating it
about its Z-axis by an angle :math:`\\psi` (azimuth or yaw rotation), then
about its Y-axis by angle :math:`\\theta` (elevation or pitch rotation), and
finally about its X-axis by angle :math:`\\phi` (bank or roll rotation).

Elevation Quaternion
--------------------

The X-axis accelerometer senses only the component of gravity along the X-axis,
and this component, in turn, depends only on the elevation angle.

Starting with the rigid body in its reference orientation, the X-axis
accelerometer is perpendicular to gravity and thus registers zero acceleration.
The Y-axis accelerometer also reads zero, while the Z-axis accelerometer reads
:math:`-g`. If the body is then rotated in azimuth about its Z-axis, the X-axis
accelerometer still reads zero, regardless of the azimuth angle.

If the rigid body is next rotated up through an angle :math:`\\theta`, the
X-axis accelerometer will instantaneously read

.. math::
    a_x = g\\sin\\theta

and the Z-axis will read

.. math::
    a_z = -g\\cos\\theta

where :math:`9.81 \\frac{m}{s^2}` is the gravitational acceleration and
:math:`\\mathbf{a}=\\begin{bmatrix}a_x & a_y & a_z\\end{bmatrix}^T` is the
**measured acceleration vector** in the body coordinate system.

For convenience, the accelerometer and magnetometer outputs are normalized to
unit vectors, so that:

.. math::
    \\begin{array}{rcl}
    \\mathbf{a} &=& \\frac{\\mathbf{a}}{\\|\\mathbf{a}\\|} \\\\
    \\mathbf{m} &=& \\frac{\\mathbf{m}}{\\|\\mathbf{m}\\|}
    \\end{array}

From the normalized accelerometer measurements, we can get:

.. math::
    \\sin\\theta = a_x

In order to obtain an elevation quaternion, a value is needed for
:math:`\\sin\\frac{\\theta}{2}` and :math:`\\cos\\frac{\\theta}{2}`. From
trigonometric half-angle formulas, **half-angle values** are given by

.. math::
    \\begin{array}{rcl}
    \\sin\\frac{\\theta}{2} &=& \\mathrm{sgn}(\\sin\\theta) \\sqrt{\\frac{1-\\cos\\theta}{2}} \\\\
    \\cos\\frac{\\theta}{2} &=& \\sqrt{\\frac{1+\\cos\\theta}{2}}
    \\end{array}

where :math:`\\mathrm{sgn}()` is the `sign <https://en.wikipedia.org/wiki/Sign_function>`_
function.

Because elevation is a rotation about the Y-axis, the unit quaternion
representing it will be expressed as:

.. math::
    \\mathbf{q}_e =
    \\begin{pmatrix}\\cos\\frac{\\theta}{2} \\\\ 0 \\\\ \\sin\\frac{\\theta}{2} \\\\ 0\\end{pmatrix}

Roll Quaternion
---------------

Changing the roll angle alters the measurements, so that the accelerometer
readings are:

.. math::
    \\begin{array}{rcl}
    a_y &=& -\\cos\\theta\\sin\\phi \\\\
    a_z &=& -\\cos\\theta\\cos\\phi
    \\end{array}

.. note::
    In reality the measurements are :math:`-g\\cos\\theta\\sin\\phi` and
    :math:`-g\\cos\\theta\\cos\\phi`, with a magnitude equal to :math:`g`, but
    when normalized their magnitude is equal to :math:`1`, and :math:`g` is
    overlooked.

If :math:`\\cos\\theta\\neq 0`, the values of :math:`\\sin\\phi` and :math:`\\cos\\phi`
are determined by:

.. math::
    \\begin{array}{rcl}
    \\sin\\phi &=& -\\frac{a_y}{\\cos\\theta} \\\\
    \\cos\\phi &=& -\\frac{a_z}{\\cos\\theta}
    \\end{array}

But if :math:`\\cos\\theta=0` the roll angle :math:`\\phi` is undefined and can
be assumed to be equal to zero. We obtain the half-angles similar to the
elevation quaternion, and roll quaternion is then computed as:

.. math::
    \\mathbf{q}_r =
    \\begin{pmatrix}\\cos\\frac{\\phi}{2} \\\\ \\sin\\frac{\\phi}{2} \\\\ 0 \\\\ 0\\end{pmatrix}

Azimuth Quaternion
------------------

Since the azimuth rotation has no effect on accelerometer data, the strategy is
to use the readings of the magnetometer, but first we have to rotate the
normalized magnetic readings :math:`^b\\mathbf{m}` into an intermediate
coordinate system through the elevation and roll quaternions:

.. math::
    ^e\\mathbf{m} = \\mathbf{q}_e\\mathbf{q}_r \\,^b\\mathbf{m}\\mathbf{q}_r^{-1}\\mathbf{q}_e^{-1}

where :math:`^b\\mathbf{m}=\\begin{pmatrix}0 & ^bm_x & ^bm_y & ^bm_z\\end{pmatrix}`
is the magnetic field measured in the body frame, and represented as a pure
quaternion.

The rotated magnetic measurements should correspond to the **normalized known
local geomagnetic field** [#]_ vector :math:`\\mathbf{n}=\\begin{bmatrix}n_x & n_y & n_z\\end{bmatrix}`,
except for the azimuth:

.. math::
    \\begin{bmatrix}n_x \\\\ n_y\\end{bmatrix}=
    \\begin{bmatrix}\\cos\\psi & -\\sin\\psi \\\\ \\sin\\psi & \\cos\\psi\\end{bmatrix}
    \\begin{bmatrix}^em_x \\\\ ^em_y\\end{bmatrix}

where :math:`\\psi` is the azimuth angle. We normalize both sides to enforce
equal length of its vectors:

.. math::
    \\begin{bmatrix}N_x \\\\ N_y\\end{bmatrix}=
    \\begin{bmatrix}\\cos\\psi & -\\sin\\psi \\\\ \\sin\\psi & \\cos\\psi\\end{bmatrix}
    \\begin{bmatrix}M_x \\\\ M_y\\end{bmatrix}

with:

.. math::
    \\begin{array}{rcl}
    \\begin{bmatrix}N_x \\\\ N_y\\end{bmatrix} &=& \\frac{1}{\\sqrt{n_x^2+n_y^2}} \\begin{bmatrix}n_x \\\\ n_y\\end{bmatrix} \\\\
    \\begin{bmatrix}M_x \\\\ M_y\\end{bmatrix} &=& \\frac{1}{\\sqrt{^em_x^2+^em_y^2}} \\begin{bmatrix}^em_x \\\\ ^em_y\\end{bmatrix}
    \\end{array}

And now we just solve for the azimuth angle with:

.. math::
    \\begin{bmatrix}\\cos\\psi \\\\ \\sin\\psi \\end{bmatrix} =
    \\begin{bmatrix}M_x & M_y \\\\ -My & M_x \\end{bmatrix}
    \\begin{bmatrix}N_x \\\\ N_y \\end{bmatrix}

In the same manner as with the elevation and roll, we estimate the half-angle
values and define the azimuth quaternion as:

.. math::
    \\mathbf{q}_a =
    \\begin{pmatrix}\\cos\\frac{\\psi}{2} \\\\ 0 \\\\ 0 \\\\ \\sin\\frac{\\psi}{2} \\end{pmatrix}

Final Quaternion
----------------

Having computed all three quaternions, the estimation representing the
orientation of the rigid body is given by their product:

.. math::
    \\mathbf{q} = \\mathbf{q}_a\\mathbf{q}_e\\mathbf{q}_r

It should be noted that this algorithm does not evaluate any trigonometric
function at any step, although a singularity occurs in the FQA if the elevation
angle is :math:`\\pm 90°`, making :math:`\\cos\\theta=0`, but that is dealt
with at the computation of the first quaternion.

Footnotes
---------
.. [#] The local geomagnetic field can be obtained with the World Magnetic
    Model. See the `WMM documentation <../WMM.html>`_ page for further details.

References
----------
.. [Yun] Xiaoping Yun et al. (2008) A Simplified Quaternion-Based Algorithm for
    Orientation Estimation From Earth Gravity and Magnetic Field Measurements.
    https://ieeexplore.ieee.org/document/4419916

"""

import numpy as np
from ..common.constants import MUNICH_LATITUDE
from ..common.constants import MUNICH_LONGITUDE
from ..common.constants import MUNICH_HEIGHT
from ..common.orientation import q_prod, q_conj
from ..utils.core import _assert_numerical_iterable

# Reference Observations in Munich, Germany at current date
from ..utils.wmm import WMM
wmm = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT)
REFERENCE_MAGNETIC_VECTOR = np.array([wmm.X, wmm.Y, wmm.Z])

def _assert_same_shapes(item1, item2, item_names: list = None) -> None:
    for item in [item1, item2]:
        if not isinstance(item, (list, tuple, np.ndarray)):
            raise TypeError(f"{item} must be an array. Got {type(item)}")
    if item_names is None:
        item_names = ['item1', 'item2']
    item1, item2 = np.copy(item1), np.copy(item2)
    if item1.shape != item2.shape:
        raise ValueError(f"{item_names[0]} and {item_names[1]} must have the same shape. Got {item1.shape} and {item2.shape}")

def _assert_magnetic_reference(m: np.ndarray) -> None:
    _assert_numerical_iterable(m, 'mag_ref')
    m = np.copy(m)
    for item in m:
        if not isinstance(item, (int, float)):
            raise TypeError(f"mag_ref must be given as an array of numbers. Got {type(item)}")
    if m.shape != REFERENCE_MAGNETIC_VECTOR.shape:
        raise ValueError(f"mag_ref must have shape (3,). Got {m.shape}")
    if np.linalg.norm(m) == 0:
        raise ValueError("mag_ref must be non-zero.")

class FQA:
    """
    Factored Quaternion Algorithm

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with N measurements of the gravitational acceleration.
    mag : numpy.ndarray, default: None
        N-by-3 array with N measurements of the geomagnetic field.
    mag_ref : numpy.ndarray, default: None
        Reference geomagnetic field. If None is given, defaults to the
        geomagnetic field of Munich, Germany.

    Attributes
    ----------
    acc : numpy.ndarray
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    m_ref : numpy.ndarray
        Normalized reference geomagnetic field.
    Q : numpy.ndarray
        Estimated attitude as quaternion.

    Raises
    ------
    ValueError
        When dimension of input arrays ``acc`` and ``mag`` are not equal.

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None, mag_ref: np.ndarray = None):
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        # Reference measurements
        self.m_ref: np.ndarray = REFERENCE_MAGNETIC_VECTOR if mag_ref is None else mag_ref
        _assert_magnetic_reference(self.m_ref)
        self.m_ref = np.array(self.m_ref)/np.linalg.norm(self.m_ref)
        if self.acc is not None:
            self.Q: np.ndarray = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """
        Estimate the quaternions given all data.

        Attributes ``acc`` and ``mag`` must contain data.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        _assert_numerical_iterable(self.acc, 'Gravitational acceleration vector')
        _assert_numerical_iterable(self.mag, 'Geomagnetic field vector')
        self.acc = np.copy(self.acc)
        self.mag = np.copy(self.mag)
        if self.acc.ndim < 2:
            _assert_numerical_iterable(self.acc, 'acc')
            if self.mag is not None:
                _assert_numerical_iterable(self.mag, 'mag')
                _assert_same_shapes(self.acc, self.mag, ['acc', 'mag'])
                return self.estimate(self.acc, self.mag)
            return self.estimate(self.acc)
        # Estimate multiple observations
        num_samples = len(self.acc)
        if self.mag is not None:
            _assert_same_shapes(self.acc, self.mag, ['acc', 'mag'])
            return np.array([self.estimate(self.acc[t], self.mag[t]) for t in range(num_samples)])
        return np.array([self.estimate(self.acc[t]) for t in range(num_samples)])

    def estimate(self, acc: np.ndarray, mag: np.ndarray = None) -> np.ndarray:
        """
        Attitude Estimation.

        Parameters
        ----------
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer.
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer.

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        """
        _assert_numerical_iterable(acc, 'Gravitational acceleration vector')
        _assert_numerical_iterable(mag, 'Geomagnetic field vector')
        a_norm = np.linalg.norm(acc)
        if a_norm == 0:     # handle NaN
            return np.array([1., 0., 0., 0.])
        a_x, a_y, a_z = acc/a_norm                                  # (eq. 20)
        # Elevation Quaternion
        s_theta = np.clip(a_x, -1.0, 1.0)                           # (eq. 21)
        c_theta = np.sqrt(1.0-s_theta**2)                           # (eq. 22)
        s_theta_2 = np.sign(s_theta)*np.sqrt((1.0-c_theta)/2.0)     # (eq. 23)
        c_theta_2 = np.sqrt((1.0+c_theta)/2.0)                      # (eq. 24)
        q_e = np.array([c_theta_2, 0.0, s_theta_2, 0.0])            # (eq. 25)
        q_e /= np.linalg.norm(q_e)
        # Roll Quaternion
        is_singular = c_theta == 0.0        # X-axis is vertically oriented
        s_phi = 0.0 if is_singular else -a_y/c_theta                # (eq. 30)
        c_phi = 0.0 if is_singular else -a_z/c_theta                # (eq. 31)
        c_phi = np.clip(c_phi, -1.0, 1.0)
        sign_s_phi = np.sign(s_phi)
        if c_phi == -1.0 and s_phi == 0.0:
            sign_s_phi = 1
        s_phi_2 = sign_s_phi*np.sqrt((1.0-c_phi)/2.0)
        c_phi_2 = np.sqrt((1.0+c_phi)/2.0)
        q_r = np.array([c_phi_2, s_phi_2, 0.0, 0.0])                # (eq. 32)
        q_r /= np.linalg.norm(q_r)
        q_er = q_prod(q_e, q_r)
        q_er /= np.linalg.norm(q_er)
        if mag is None:
            return q_er
        # Azimuth Quaternion
        m_norm = np.linalg.norm(mag)
        if not m_norm > 0:
            return q_er
        mag /= np.linalg.norm(mag)
        bm = np.array([0.0, *mag])
        em = q_prod(q_e, q_prod(q_r, q_prod(bm, q_prod(q_conj(q_r), q_conj(q_e)))))     # (eq. 34)
        nx, ny, _ = self.m_ref
        N = np.array([nx, ny])/np.linalg.norm([nx, ny])             # (eq. 36)
        Mx, My = em[1:-1] / np.linalg.norm(em[1:-1])                # (eq. 37)
        c_psi, s_psi = np.array([[Mx, My], [-My, Mx]])@N            # (eq. 39)
        c_psi = np.clip(c_psi, -1.0, 1.0)
        s_psi_2 = np.sign(s_psi)*np.sqrt((1.0-c_psi)/2.0)
        c_psi_2 = np.sqrt((1.0+c_psi)/2.0)
        q_a = np.array([c_psi_2, 0.0, 0.0, s_psi_2])                # (eq. 40)
        q_a /= np.linalg.norm(q_a)
        # Final Quaternion (eq. 41)
        q = q_prod(q_a, q_er)
        return q/np.linalg.norm(q)

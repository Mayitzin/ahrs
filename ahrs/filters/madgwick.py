# -*- coding: utf-8 -*-
"""
Madgwick Orientation Filter
===========================

.. contents:: Table of Contents
    :local:
    :depth: 2

This is an orientation filter applicable to IMUs consisting of tri-axial
gyroscopes and accelerometers, and MARG arrays, which also include tri-axial
magnetometers, proposed by Sebastian Madgwick [Madgwick]_.

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

Rewritten in Python from the `original implementation <https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/>`_
conceived by Sebastian Madgwick.

Orientation from angular rate
-----------------------------

The orientation of the Earth frame relative to the sensor frame
:math:`\\mathbf{q}_{\\omega, t}=\\begin{bmatrix}q_w & q_x & q_y & q_z\\end{bmatrix}`
at time :math:`t` can be computed by numerically integrating the quaternion
derivative :math:`\\dot{\\mathbf{q}}_t=\\frac{1}{2}\\mathbf{q}_{t-1}\\mathbf{\\omega}_t` as:

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}_{\\omega, t} &=& \\,\\mathbf{q}_{t-1} + \\,\\dot{\\mathbf{q}}_{\\omega, t}\\Delta t\\\\
    &=& \\,\\mathbf{q}_{t-1} + \\frac{1}{2}\\big(\\,\\mathbf{q}_{t-1}\\mathbf{\\,^S\\omega_t}\\big)\\Delta t
    \\end{array}

where :math:`\\Delta t` is the sampling period and :math:`^S\\omega=\\begin{bmatrix}0 & \\omega_x & \\omega_y & \\omega_z\\end{bmatrix}`
is the tri-axial angular rate, in rad/s, measured in the sensor frame and
represented as a pure quaternion.

.. note::
    The multiplication of quaternions (included pure quaternions) is performed
    as a `Hamilton product <https://en.wikipedia.org/wiki/Quaternion#Hamilton_product>`_.
    All quaternion products explained here follow this procedure. For further
    details on how to compute it, see the `quaternion <../quaternion.html>`_
    documentation page.

The sub-script :math:`\\omega` in :math:`\\mathbf{q}_\\omega` indicates that it
is calculated from angular rates.

A more detailed explanation of the orientation estimation solely based on
angular rate can be found in the documentation of the `AngularRate
<./angular.html>`_ estimator.

Orientation as solution of Gradient Descent
-------------------------------------------

A quaternion representation requires a complete solution to be found. This may
be achieved through the formulation of an optimization problem where an
orientation of the sensor, :math:`\\mathbf{q}`, is that which aligns any
*predefined reference* in the Earth frame,
:math:`^E\\mathbf{d}=\\begin{bmatrix}0 & d_x & d_y & d_z\\end{bmatrix}`, with
its corresponding *measured* direction in the sensor frame,
:math:`^S\\mathbf{s}=\\begin{bmatrix}0 & s_x & s_y & s_z\\end{bmatrix}`.

Thus, the `objective function <https://en.wikipedia.org/wiki/Loss_function>`_
is:

.. math::
    \\begin{array}{rcl}
    f( \\mathbf{q}, \\,^E\\mathbf{d}, \\,^S\\mathbf{s}) &=&  \\mathbf{q}^*\\,^E\\mathbf{d} \\,\\mathbf{q}-\\,^S\\mathbf{s} \\\\
    &=&\\begin{bmatrix}
    2d_x(\\frac{1}{2}-q_y^2-q_z^2) + 2d_y(q_wq_z+q_xq_y) + 2d_z(q_xq_z-q_wq_y) - s_x \\\\
    2d_x(q_xq_y-q_wq_z) + 2d_y(\\frac{1}{2}-q_x^2-q_z^2) + 2d_z(q_wq_x+q_yq_z) - s_y \\\\
    2d_x(q_wq_y+q_xq_z) + 2d_y(q_yq_z-q_wq_x) + 2d_z(\\frac{1}{2}-q_x^2-q_y^2) - s_z
    \\end{bmatrix}
    \\end{array}

where :math:`\\mathbf{q}^*` is the `conjugate <https://mathworld.wolfram.com/QuaternionConjugate.html>`_
of :math:`\\mathbf{q}`. Consequently, :math:`\\mathbf{q}` is found as
the solution to:

.. math::
    \\mathrm{min}\\; f( \\mathbf{q}, \\,^E\\mathbf{d}, \\,^S\\mathbf{s})

The suggested approach of this estimator is to use the `Gradient Descent
Algorithm <https://en.wikipedia.org/wiki/Gradient_descent>`_ to compute the
solution.

From an *initial guess* :math:`\\mathbf{q}_0` and a step-size :math:`\\mu`,
the GDA for :math:`n` iterations, which estimates :math:`n+1` orientations, is
described as:

.. math::
     \\mathbf{q}_{k+1} =  \\mathbf{q}_k-\\mu\\frac{\\nabla f( \\mathbf{q}_k, \\,^E\\mathbf{d}, \\,^S\\mathbf{s})}{\\|\\nabla f( \\mathbf{q}_k, \\,^E\\mathbf{d}, \\,^S\\mathbf{s})\\|}

where :math:`k=0,1,2\\dots n`, and the `gradient <https://en.wikipedia.org/wiki/Gradient>`_
of the solution is defined by the objective function and its Jacobian:

.. math::
    \\nabla f( \\mathbf{q}_k, \\,^E\\mathbf{d}, \\,^S\\mathbf{s}) = J( \\mathbf{q}_k, \\,^E\\mathbf{d})^T f( \\mathbf{q}_k, \\,^E\\mathbf{d}, \\,^S\\mathbf{s})

The `Jacobian <https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_
of the objective function is:

.. math::
    \\begin{array}{rcl}
    J( \\mathbf{q}_k, \\,^E\\mathbf{d}) &=&\\begin{bmatrix}
    \\frac{\\partial f( \\mathbf{q}, \\,^E\\mathbf{d}, \\,^S\\mathbf{s})}{\\partial q_w} &
    \\frac{\\partial f( \\mathbf{q}, \\,^E\\mathbf{d}, \\,^S\\mathbf{s})}{\\partial q_x} &
    \\frac{\\partial f( \\mathbf{q}, \\,^E\\mathbf{d}, \\,^S\\mathbf{s})}{\\partial q_y} &
    \\frac{\\partial f( \\mathbf{q}, \\,^E\\mathbf{d}, \\,^S\\mathbf{s})}{\\partial q_z} &
    \\end{bmatrix}\\\\
    &=& \\begin{bmatrix}
    2d_yq_z-2d_zq_y  & 2d_yq_y+2d_zq_z         & -4d_xq_y+2d_yq_x-2d_zq_w & -4d_xq_z+2d_yq_w+2d_zq_x \\\\
    -2d_xq_z+2d_zq_x & 2d_xq_y-4d_yq_x+2d_zq_w & 2d_xq_x+2d_zq_z          & -2d_xq_w-4d_yq_z+2d_zq_y \\\\
    2d_xq_y-2d_yq_x  & 2d_xq_z-2d_yq_w-4d_zq_x & 2d_xq_w+2d_yq_z-4d_zq_y  & 2d_xq_x+2d_yq_y
    \\end{bmatrix}
    \\end{array}

This general form of the algorithm can be applied to a field predefined in any
direction, as it will be shown for IMU and MARG systems.

The gradient quaternion :math:`\\mathbf{q}_{\\nabla, t}`, computed at time
:math:`t`, is based on a previous estimation :math:`\\mathbf{q}_{t-1}` and
the *objective function gradient* :math:`\\nabla f`:

.. math::
     \\mathbf{q}_{\\nabla, t} =  \\mathbf{q}_{t-1}-\\mu_t\\frac{\\nabla f}{\\|\\nabla f\\|}

An optimal value of :math:`\\mu_t` ensures that the convergence rate of
:math:`\\mathbf{q}_{\\nabla, t}` is limited to the physical orientation
rate. It can be calculated with:

.. math::
    \\mu_t = \\alpha\\|\\,\\dot{\\mathbf{q}}_{\\omega, t}\\|\\Delta t

with :math:`\\dot{\\mathbf{q}}_{\\omega, t}` being the physical orientation
rate measured by the gyroscopes, and :math:`\\alpha>1` is an augmentation of
:math:`\\mu` dealing with the noise of accelerometers and magnetometers.

An estimated orientation of the sensor frame relative to the earth frame,
:math:`\\mathbf{q}_t`, is obtained through the *weighted fusion of the
orientation calculations*, :math:`\\mathbf{q}_{\\omega, t}` and :math:`\\mathbf{q}_{\\nabla, t}`
with a simple **complementary filter**:

.. math::
     \\mathbf{q}_t = \\gamma_t  \\mathbf{q}_{\\nabla, t} + (1-\\gamma_t) \\mathbf{q}_{\\omega, t}

where :math:`\\gamma_t` and :math:`(1-\\gamma_t)` are the weights, ranging
between 0 and 1, applied to each orientation calculation. An optimal value of
:math:`\\gamma_t` ensures that the weighted divergence of :math:`\\mathbf{q}_{\\omega, t}`
is equal to the weighted convergence of :math:`\\mathbf{q}_{\\nabla, t}`.
This is expressed with:

.. math::
    (1-\\gamma_t)\\beta = \\gamma_t\\frac{\\mu_t}{\\Delta t}

defining :math:`\\frac{\\mu_t}{\\Delta t}` as the *convergence rate* of
:math:`\\mathbf{q}_\\nabla`, and :math:`\\beta` as the *divergence rate* of
:math:`\\mathbf{q}_\\omega` expressed as the magnitude of a quaternion
derivative corresponding to the gyroscope measurement error.

If :math:`\\alpha` is very large then :math:`\\mu` becomes very large making
:math:`\\mathbf{q}_{t-1}` negligible in the **objective function gradient**
simplifying it to the approximation:

.. math::
     \\mathbf{q}_{\\nabla, t} \\approx -\\mu_t\\frac{\\nabla f}{\\|\\nabla f\\|}

This also simplifies the relation of :math:`\\gamma` and :math:`\\beta`:

.. math::
    \\gamma \\approx \\frac{\\beta\\Delta t}{\\mu_t}

which further reduces the estimation to:

.. math::
    \\begin{array}{rcl}
     \\mathbf{q}_t &=&  \\mathbf{q}_{t-1} +  \\dot{\\mathbf{q}}_t\\Delta t \\\\
    &=&  \\mathbf{q}_{t-1} + \\big( \\dot{\\mathbf{q}}_{\\omega, t} - \\beta \\dot{\\mathbf{q}}_{\\epsilon, t}\\big)\\Delta t \\\\
    &=&  \\mathbf{q}_{t-1} + \\big( \\dot{\\mathbf{q}}_{\\omega, t} - \\beta\\frac{\\nabla f}{\\|\\nabla f\\|}\\big)\\Delta t
    \\end{array}

where :math:`\\dot{\\mathbf{q}}_t` is the **estimated rate of change of
orienation** defined by :math:`\\beta` and its direction error
:math:`\\dot{\\mathbf{q}}_{\\epsilon, t}=\\frac{\\nabla f}{\\|\\nabla f\\|}`.

In summary, the filter calculates the orientation :math:`\\mathbf{q}_{t}`
by numerically integrating the estimated orientation rate :math:`\\dot{\\mathbf{q}}_t`.
It computes :math:`\\dot{\\mathbf{q}}_t` as the rate of change of
orientation measured by the gyroscopes, :math:`\\dot{\\mathbf{q}}_{\\omega, t}`,
with the magnitude of the gyroscope measurement error, :math:`\\beta`, removed
in the direction of the estimated error, :math:`\\dot{\\mathbf{q}}_{\\epsilon, t}`,
computed from accelerometer and magnetometer measurements.

Orientation from IMU
--------------------

Two main geodetic properties can be used to build Earth's reference:

- The `gravitational force <https://en.wikipedia.org/wiki/Gravity_of_Earth>`_
  :math:`^E\\mathbf{g}` represented as a vector and measured with a tri-axial
  accelerometer.
- The `geomagnetic field <https://en.wikipedia.org/wiki/Earth%27s_magnetic_field>`_
  :math:`^E\\mathbf{b}` represented as a vector and measured with a tri-axial
  magnetometer.

Earth's shape is not uniform and a geographical location is usually provided to
obtain the references' true values. Madgwick's filter, however, uses normalized
references, making their magnitudes, irrespective of their location, always
equal to 1.

.. note::
    All real vectors in a three-dimensional euclidean operating with
    quaternions will be considered pure quaternions. That is, given a
    three-dimensional vector :math:`\\mathbf{x}=\\begin{bmatrix}x&y&z\\end{bmatrix}\\in\\mathbb{R}^3`,
    it will be redefined as :math:`\\mathbf{x}=\\begin{bmatrix}0&x&y&z\\end{bmatrix}\\in\\mathbf{H}^4`.

To obtain the objective function of the **gravitational acceleration**, we
assume, by convention, that the vertical Z-axis is defined by the direction of
the gravity :math:`^E\\mathbf{g}=\\begin{bmatrix}0 & 0 & 0 & 1\\end{bmatrix}`.

Substituting :math:`^E\\mathbf{g}` and the *normalized* accelerometer
measurement :math:`^S\\mathbf{a}=\\begin{bmatrix}0 & a_x & a_y & a_z\\end{bmatrix}`
for :math:`^E\\mathbf{d}` and :math:`^S\\mathbf{s}`, respectively, yields a new
objective function and its Jacobian particular to the acceleration:

.. math::
    \\begin{array}{c}
    f_g( \\mathbf{q}, \\,^S\\mathbf{a}) = \\begin{bmatrix}
    2(q_xq_z-q_wq_y)-a_x \\\\ 2(q_wq_x+q_yq_z)-a_y \\\\ 2(\\frac{1}{2}-q_x^2-q_y^2)-a_z
    \\end{bmatrix} \\\\ \\\\
    J_g( \\mathbf{q})=\\begin{bmatrix}
    -2q_y & 2q_z & -2q_w & 2q_x \\\\
    2q_x & 2q_w & 2q_z & 2q_y \\\\
    0 & -4q_x & -4q_y & 0
    \\end{bmatrix}
    \\end{array}

The function gradient is defined by the sensor measurements at time :math:`t`:

.. math::
    \\nabla f = J_g^T( \\mathbf{q}_{t-1})f_g( \\mathbf{q}_{t-1}, \\,^S\\mathbf{a}_t)

So, the estimation of the orientation using inertial sensors only (gyroscopes
and accelerometers) becomes:

.. math::
    \\begin{array}{rcl}
     \\mathbf{q}_t &=&  \\mathbf{q}_{t-1} +  \\dot{\\mathbf{q}}_t\\Delta t \\\\
    &=&  \\mathbf{q}_{t-1} + \\Big( \\dot{\\mathbf{q}}_{\\omega, t} - \\beta\\frac{\\nabla f}{\\|\\nabla f\\|}\\Big) \\Delta t \\\\
    &=&  \\mathbf{q}_{t-1} + \\Big( \\dot{\\mathbf{q}}_{\\omega, t} - \\beta\\frac{J_g^T( \\mathbf{q}_{t-1})f_g( \\mathbf{q}_{t-1}, \\,^S\\mathbf{a}_t)}{\\|J_g^T( \\mathbf{q}_{t-1})f_g( \\mathbf{q}_{t-1}, \\,^S\\mathbf{a}_t)\\|}\\Big) \\Delta t
    \\end{array}

Orientation from MARG
---------------------

The gravity and the angular velocity are good parameters for an estimation over
a short period of time. But they don't hold for longer periods of time,
especially estimating the heading orientation of the system, as the gyroscope
measurements, prone to drift, are instantaneous and local, while the
accelerometer computes the roll and pitch orientations only.

Therefore, it is always very convenient to add a reference that provides
constant information about the heading angle (a.k.a. yaw). Earth's magnetic
field is usually the chosen reference, as it fairly keeps a constant reference [#]_.

The mix of Magnetic, Angular Rate and Gravity (MARG) is the most prevalent
solution in the majority of attitude estimation systems.

The reference magnetic field :math:`^E\\mathbf{b}=\\begin{bmatrix}0 & b_x & b_y & b_z\\end{bmatrix}`
in Earth's frame, has components along the three axes of `NED coordinates
<https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates#Local_north,_east,_down_(NED)_coordinates>`_
(North-East-Down), which can be obtained using the `World Magnetic Model
<https://www.ngdc.noaa.gov/geomag/WMM/>`_.

Madgwick's estimator, nonetheless, assumes the East component of the magnetic
field (along Y-axis) is negligible, further reducing the reference magnetic
vector to:

.. math::
    \\mathbf{b}=\\begin{bmatrix}0 & b_x & 0 & b_z\\end{bmatrix}

The *measured* direction of Earth's magnetic field in the Earth frame at time
:math:`t`, :math:`^E\\mathbf{h}_t`, can be computed as the **normalized**
magnetometer measurement, :math:`^S\\mathbf{m}_t`, rotated by the orientation
of the sensor computed in the previous estimation, :math:`\\mathbf{q}_{t-1}`.

.. math::
    ^E\\mathbf{h}_t = \\begin{bmatrix}0 & h_x & h_y & h_z\\end{bmatrix} =
    \\,\\mathbf{q}_{t-1}\\,^S\\mathbf{m}_t\\,\\mathbf{q}_{t-1}^*

The effect of an erroneous inclination of the measured direction Earth's
magnetic field, :math:`^E\\mathbf{h}_t`, can be corrected if the filter's
reference direction of the geomagnetic field, :math:`^E\\mathbf{b}_t`, is of
the same inclination. This is achieved by computing :math:`^E\\mathbf{b}_t` as
a normalized :math:`^E\\mathbf{h}_t` to have only components in X- and Z-axes
of the Earth frame.

.. math::
    ^E\\mathbf{b}_t = \\begin{bmatrix}0 & \\sqrt{h_x^2+h_y^2} & 0 & h_z\\end{bmatrix}

This way ensures that magnetic disturbances are limited to only affect the
estimated heading component of orientation. It also eliminates the need for the
reference direction of the Earth's magnetic field to be predefined.

Substituting :math:`^E\\mathbf{b}` and the normalized magnetometer normalized
:math:`^S\\mathbf{m}` to form the *objective function* and *Jacobian* we get:

.. math::
    \\begin{array}{c}
    f_b( \\mathbf{q}, \\,^E\\mathbf{b}, \\,^S\\mathbf{m}) = \\begin{bmatrix}
    2b_x(\\frac{1}{2}-q_y^2-q_z^2) + 2b_z(q_xq_z-q_wq_y)-m_x \\\\
    2b_x(q_xq_y-q_wq_z) + 2b_z(q_wq_x+q_yq_z)-m_y \\\\
    2b_x(q_wq_y+q_xq_z) + 2b_z(\\frac{1}{2}-q_x^2-q_y^2)-m_z
    \\end{bmatrix} \\\\ \\\\
    J_b( \\mathbf{q}, \\,^E\\mathbf{b})=\\begin{bmatrix}
    -2b_zq_y          & 2b_zq_z         & -4b_xq_y-2b_zq_w & -4b_xq_z+2b_zq_x \\\\
    -2b_xq_z+2b_zq_x  & 2b_xq_y+2b_zq_w & 2b_xq_x+2b_zq_z  & -2b_xq_w+2b_zq_y \\\\
    2b_xq_y           & 2b_xq_z-4b_zq_x & 2b_xq_w-4b_zq_y  & 2b_xq_x
    \\end{bmatrix}
    \\end{array}

The measurements and reference directions of both fields, gravity and magnetic
field, are combined, where the solution surface has a minimum defined by a
single point, as long as the northerly magnetic intensity is defined (:math:`b_x\\neq 0`):

.. math::
    \\begin{array}{c}
    f_{g,b}( \\mathbf{q}, \\,^S\\mathbf{a}, \\,^E\\mathbf{b}, \\,^S\\mathbf{m})=
    \\begin{bmatrix}f_g( \\mathbf{q}, \\,^S\\mathbf{a}) \\\\ f_b( \\mathbf{q}, \\,^E\\mathbf{b}, \\,^S\\mathbf{m})\\end{bmatrix}\\\\ \\\\
    J_{g,b}( \\mathbf{q}, \\,^E\\mathbf{b})=
    \\begin{bmatrix}J_g^T( \\mathbf{q}) \\\\ J_b^T( \\mathbf{q}, \\,^E\\mathbf{b})\\end{bmatrix}
    \\end{array}

Simliar to the implementation with IMU, the estimation of the new quaternion
will be:

.. math::
     \\mathbf{q}_t =  \\mathbf{q}_{t-1} + \\Big( \\dot{\\mathbf{q}}_{\\omega, t} - \\beta\\frac{J_{g,b}^T( \\mathbf{q}_{t-1}, \\,^E\\mathbf{b})f_{g,b}( \\mathbf{q}_{t-1}, \\,^S\\mathbf{a}, \\,^E\\mathbf{b}, \\,^S\\mathbf{m})}{\\|J_{g,b}^T( \\mathbf{q}_{t-1}, \\,^E\\mathbf{b})f_{g,b}( \\mathbf{q}_{t-1}, \\,^S\\mathbf{a}, \\,^E\\mathbf{b}, \\,^S\\mathbf{m})\\|}\\Big) \\Delta t

Filter gain
-----------

The gain :math:`\\beta` represents all mean zero gyroscope measurement errors,
expressed as the magnitude of a quaternion derivative. It is defined using the
angular velocity:

.. math::
    \\beta = \\sqrt{\\frac{3}{4}}\\bar{\\omega}_\\beta

where :math:`\\bar{\\omega}_\\beta` is the estimated mean zero gyroscope
measurement error of each axis.

Footnotes
---------
.. [#] In reality, Earth's magnetic field varies slowly over time, which is a
    phenomenon known as `Geomagnetic secular variation <https://en.wikipedia.org/wiki/Geomagnetic_secular_variation>`_,
    but such shift can be omited for practical purposes.

References
----------
.. [Madgwick] Sebastian Madgwick. An efficient orientation filter for inertial
    and inertial/magnetic sensor arrays. April 30, 2010.
    http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/

"""

import numpy as np
from ..common.orientation import q_prod
from ..common.orientation import q_conj
from ..common.orientation import acc2q
from ..common.orientation import ecompass
from ..utils.core import _assert_numerical_iterable

class Madgwick:
    """
    Madgwick's Gradient Descent Orientation Filter

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
    gain_imu : float, default: 0.033
        Filter gain for IMU implementation.
    gain_marg : float, default: 0.041
        Filter gain for MARG implementation.
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
        When dimension of input arrays ``acc``, ``gyr``, or ``mag`` are not
        equal.

    Examples
    --------

    This algorithm can work solely with gyroscope and accelerometer samples.
    The easiest way is to directly give the full array of samples to their
    matching parameters. The estimated quaternions are saved in the attribute
    ``Q``.

    >>> from ahrs.filters import Madgwick
    >>> madgwick = Madgwick(gyr=gyro_data, acc=acc_data)     # Using IMU
    >>> type(madgwick.Q), madgwick.Q.shape
    (<class 'numpy.ndarray'>, (1000, 4))

    If we desire to estimate each sample independently, we call the
    corresponding ``update`` method.

    >>> madgwick = Madgwick()
    >>> Q = np.tile([1., 0., 0., 0.], (num_samples, 1)) # Allocate for quaternions
    >>> for t in range(1, num_samples):
    ...     Q[t] = madgwick.updateIMU(Q[t-1], gyr=gyro_data[t], acc=acc_data[t])


    This algorithm requires a valid initial attitude, as a versor. This can be
    set with the parameter ``q0``:

    >>> madgwick = Madgwick(gyr=gyro_data, acc=acc_data, q0=[0.7071, 0.0, 0.7071, 0.0])

    .. warning::
        Do **NOT** initialize the filter with an empty array. The initial
        quaternion **must** be a versor, which is a quaternion, whose norm is
        equal to ``1.0``.

    If no initial orientation is given, an attitude is estimated using the
    first sample of each sensor. This initial attitude is computed assuming the
    sensors are straped to a body in a quasi-static state.

    Further on, we can also use magnetometer data.

    >>> madgwick = Madgwick(gyr=gyro_data, acc=acc_data, mag=mag_data)   # Using MARG

    A constant sampling frequency equal to 100 Hz is used by default. To change
    this value we set it in its parameter ``frequency``.

    >>> madgwick = Madgwick(gyr=gyro_data, acc=acc_data, frequency=150.0)   # 150 Hz

    Or, alternatively, setting the sampling step (:math:`\\Delta t = \\frac{1}{f}`):

    >>> madgwick = Madgwick(gyr=gyro_data, acc=acc_data, Dt=1/150)

    This is specially useful for situations where the sampling rate is variable:

    >>> madgwick = Madgwick()
    >>> Q = np.zeros((num_samples, 4))      # Allocation of quaternions
    >>> Q[0] = [1.0, 0.0, 0.0, 0.0]         # Initial attitude as a quaternion
    >>> for t in range(1, num_samples):
    >>>     madgwick.Dt = new_sample_rate
    ...     Q[t] = madgwick.updateIMU(Q[t-1], gyr=gyro_data[t], acc=acc_data[t])

    Madgwick's algorithm uses a gradient descent method to correct the
    estimation of the attitude. The **step size**, a.k.a.
    `learning rate <https://en.wikipedia.org/wiki/Learning_rate>`_, is
    considered a *gain* of this algorithm and can be set in the parameters too:

    >>> madgwick = Madgwick(gyr=gyro_data, acc=acc_data, gain=0.01)

    Following the original article, the gain defaults to ``0.033`` for IMU
    arrays, and to ``0.041`` for MARG arrays. Alternatively, the individual
    gains can be also set separately:

    >>> madgwick = Madgwick(gyr=gyro_data, acc=acc_data, mag=mag_data, gain_imu=0.01, gain_marg=0.05)

    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kwargs):
        self.gyr: np.ndarray = gyr
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.frequency: float = kwargs.get('frequency', 100.0)
        self.Dt: float = kwargs.get('Dt', (1.0/self.frequency) if self.frequency else 0.01)
        self.q0: np.ndarray = kwargs.get('q0')
        self._set_gain(**kwargs)
        self._assert_validity_of_inputs()
        if self.acc is not None and self.gyr is not None:
            self.Q: np.ndarray = self._compute_all()

    def _set_gain(self, **kwargs) -> None:
        """Set the gain parameter."""
        self.gain_imu: float = kwargs.get('gain_imu', 0.033)
        self.gain_marg: float = kwargs.get('gain_marg', 0.041)
        self.gain: float = kwargs.get('beta')  # Setting gain with `beta` will be removed in the future.
        if self.gain is None:
            self.gain: float = kwargs.get('gain', self.gain_imu if self.mag is None else self.gain_marg)

    def _assert_validity_of_inputs(self):
        """Asserts the validity of the inputs."""
        for item in ["frequency", "Dt", "gain", "gain_imu", "gain_marg"]:
            if isinstance(self.__getattribute__(item), bool):
                raise TypeError(f"Parameter '{item}' must be numeric.")
            if not isinstance(self.__getattribute__(item), (int, float)):
                raise TypeError(f"Parameter '{item}' is not a non-zero number.")
            if self.__getattribute__(item) <= 0.0:
                raise ValueError(f"Parameter '{item}' must be a non-zero number.")
        if self.q0 is not None:
            if not isinstance(self.q0, (list, tuple, np.ndarray)):
                raise TypeError(f"Parameter 'q0' must be an array. Got {type(self.q0)}.")
            self.q0 = np.copy(self.q0)
            if self.q0.shape != (4,):
                raise ValueError(f"Parameter 'q0' must be an array of shape (4,). It is {self.q0.shape}.")
            if not np.allclose(np.linalg.norm(self.q0), 1.0):
                raise ValueError(f"Parameter 'q0' must be a versor (norm equal to 1.0). Its norm is equal to {np.linalg.norm(self.q0)}.")

    def _compute_all(self) -> np.ndarray:
        """
        Estimate the quaternions given all data.

        Attributes ``gyr`` and ``acc`` must contain data. If ``mag`` contains
        data, the updateMARG() method is used.

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
        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 4))
        # Compute with IMU architecture
        if self.mag is None:
            Q[0] = acc2q(self.acc[0]) if self.q0 is None else self.q0/np.linalg.norm(self.q0)
            for t in range(1, num_samples):
                Q[t] = self.updateIMU(Q[t-1], self.gyr[t], self.acc[t])
            return Q
        # Compute with MARG architecture
        _assert_numerical_iterable(self.mag, 'Geomagnetic field vector')
        self.mag = np.copy(self.mag)
        if self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        Q[0] = ecompass(self.acc[0], self.mag[0], frame='NED', representation='quaternion')
        for t in range(1, num_samples):
            Q[t] = self.updateMARG(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def updateIMU(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, dt: float = None) -> np.ndarray:
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
        dt : float, default: None
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        Examples
        --------

        Assuming we have a tri-axial gyroscope array with 1000 samples, and
        1000 samples of a tri-axial accelerometer. We get the attitude with the
        Madgwick algorithm as:

        >>> from ahrs.filters import Madgwick
        >>> madgwick = Madgwick()
        >>> Q = np.tile([1., 0., 0., 0.], (len(gyro_data), 1)) # Allocate for quaternions
        >>> for t in range(1, num_samples):
        ...   Q[t] = madgwick.updateIMU(Q[t-1], gyr=gyro_data[t], acc=acc_data[t])
        ...

        Or giving the data directly in the class constructor will estimate all
        attitudes at once and store the estimated quaternions in the attribute
        ``Q``:

        >>> madgwick = Madgwick(gyr=gyro_data, acc=acc_data)
        >>> madgwick.Q.shape
        (1000, 4)

        """
        _assert_numerical_iterable(q, 'Quaternion')
        _assert_numerical_iterable(gyr, 'Tri-axial gyroscope sample')
        _assert_numerical_iterable(acc, 'Tri-axial accelerometer sample')
        dt = self.Dt if dt is None else dt
        if gyr is None or not np.linalg.norm(gyr) > 0:
            return q
        qDot = 0.5 * q_prod(q, [0, *gyr])                           # (eq. 12)
        a_norm = np.linalg.norm(acc)
        if a_norm > 0:
            a = acc/a_norm
            qw, qx, qy, qz = q/np.linalg.norm(q)
            # Objective function (eq. 25)
            f = np.array([2.0*(qx*qz - qw*qy)   - a[0],
                          2.0*(qw*qx + qy*qz)   - a[1],
                          2.0*(0.5-qx**2-qy**2) - a[2]])
            if np.linalg.norm(f) > 0:
                # Jacobian (eq. 26)
                J = np.array([[-2.0*qy,  2.0*qz, -2.0*qw, 2.0*qx],
                              [ 2.0*qx,  2.0*qw,  2.0*qz, 2.0*qy],
                              [ 0.0,    -4.0*qx, -4.0*qy, 0.0   ]])
                # Objective Function Gradient
                gradient = J.T@f                                    # (eq. 34)
                gradient /= np.linalg.norm(gradient)
                qDot -= self.gain*gradient                          # (eq. 33)
        q_new = q + qDot*dt                                         # (eq. 13)
        q_new /= np.linalg.norm(q_new)
        return q_new

    def updateMARG(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray, dt: float = None) -> np.ndarray:
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
            Sample of tri-axial Magnetometer in nT
        dt : float, default: None
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        Examples
        --------

        Assuming we have a tri-axial gyroscope array with 1000 samples, a
        second array with 1000 samples of a tri-axial accelerometer, and a
        third array with 1000 samples of a tri-axial magnetometer. We get the
        attitude with the Madgwick algorithm as:

        >>> from ahrs.filters import Madgwick
        >>> madgwick = Madgwick()
        >>> Q = np.tile([1., 0., 0., 0.], (len(gyro_data), 1)) # Allocate for quaternions
        >>> for t in range(1, num_samples):
        ...   Q[t] = madgwick.updateMARG(Q[t-1], gyr=gyro_data[t], acc=acc_data[t], mag=mag_data[t])
        ...

        Or giving the data directly in the class constructor will estimate all
        attitudes at once and store the estimated quaternions in the attribute
        ``Q``:

        >>> madgwick = Madgwick(gyr=gyro_data, acc=acc_data, mag=mag_data)
        >>> madgwick.Q.shape
        (1000, 4)

        """
        _assert_numerical_iterable(q, 'Quaternion')
        _assert_numerical_iterable(gyr, 'Tri-axial gyroscope sample')
        _assert_numerical_iterable(acc, 'Tri-axial accelerometer sample')
        _assert_numerical_iterable(mag, 'Tri-axial magnetometer sample')
        dt = self.Dt if dt is None else dt
        if gyr is None or not np.linalg.norm(gyr) > 0:
            return q
        if mag is None or not np.linalg.norm(mag) > 0:
            return self.updateIMU(q, gyr, acc)
        qDot = 0.5 * q_prod(q, [0, *gyr])                           # (eq. 12)
        a_norm = np.linalg.norm(acc)
        if a_norm > 0:
            a = acc/a_norm
            m = mag/np.linalg.norm(mag)
            # Rotate normalized magnetometer measurements
            h = q_prod(q, q_prod([0, *m], q_conj(q)))               # (eq. 45)
            bx = np.linalg.norm([h[1], h[2]])                       # (eq. 46)
            bz = h[3]
            qw, qx, qy, qz = q/np.linalg.norm(q)
            # Objective function (eq. 31)
            f = np.array([2.0*(qx*qz - qw*qy)   - a[0],
                          2.0*(qw*qx + qy*qz)   - a[1],
                          2.0*(0.5-qx**2-qy**2) - a[2],
                          2.0*bx*(0.5 - qy**2 - qz**2) + 2.0*bz*(qx*qz - qw*qy)       - m[0],
                          2.0*bx*(qx*qy - qw*qz)       + 2.0*bz*(qw*qx + qy*qz)       - m[1],
                          2.0*bx*(qw*qy + qx*qz)       + 2.0*bz*(0.5 - qx**2 - qy**2) - m[2]])
            # Jacobian (eq. 32)
            J = np.array([[-2.0*qy,               2.0*qz,              -2.0*qw,               2.0*qx             ],
                          [ 2.0*qx,               2.0*qw,               2.0*qz,               2.0*qy             ],
                          [ 0.0,                 -4.0*qx,              -4.0*qy,               0.0                ],
                          [-2.0*bz*qy,            2.0*bz*qz,           -4.0*bx*qy-2.0*bz*qw, -4.0*bx*qz+2.0*bz*qx],
                          [-2.0*bx*qz+2.0*bz*qx,  2.0*bx*qy+2.0*bz*qw,  2.0*bx*qx+2.0*bz*qz, -2.0*bx*qw+2.0*bz*qy],
                          [ 2.0*bx*qy,            2.0*bx*qz-4.0*bz*qx,  2.0*bx*qw-4.0*bz*qy,  2.0*bx*qx          ]])
            gradient = J.T@f                                        # (eq. 34)
            gradient /= np.linalg.norm(gradient)
            qDot -= self.gain*gradient                              # (eq. 33)
        q_new = q + qDot*dt                                         # (eq. 13)
        q_new /= np.linalg.norm(q_new)
        return q_new

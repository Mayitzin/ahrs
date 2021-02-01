# -*- coding: utf-8 -*-
"""
Attitude from angular rate
==========================

A quaternion is updated via integration of angular rate measurements of a
gyroscope. The easiest way to update the quaternions is by integrating the
differential equation for a local rotation rate [Sola]_.

In a kinematic system, the angular velocity :math:`\\boldsymbol\\omega` of a
rigid body at any instantaneous time is described with respect to a fixed frame
coinciding instantaneously with its body frame. Thus, this angular
velocity is *in terms of* the body frame [Jia]_.

First, let's remember the definition of a vector or `matrix exponential
<https://en.wikipedia.org/wiki/Matrix_exponential>`_ as a power series:

.. math::
    e^\\mathbf{X} = \\sum_{k=0}^\\infty \\frac{1}{k!} \\mathbf{X}^k

Letting :math:`\\mathbf{v}=\\mathbf{u}\\theta` be the **rotation vector**,
representing a rotation of :math:`\\theta` radians around the unitary [#]_ axis
:math:`\\mathbf{u}=\\begin{bmatrix}u_x & u_y & u_z\\end{bmatrix}^T`, we can
get its exponential series as:

.. math::
    e^\\mathbf{v} = e^{\\mathbf{u}\\theta} = \\Big(1 - \\frac{\\theta^2}{2!} + \\frac{\\theta^4}{4!} + \\cdots\\Big)
    + \\Big(\\mathbf{u}\\theta - \\frac{\\mathbf{u}\\theta^3}{3!} + \\frac{\\mathbf{u}\\theta^5}{5!} + \\cdots\\Big)

We recognize the power-series expansion of `Euler's formula
<https://en.wikipedia.org/wiki/Euler%27s_formula#Using_power_series>`_, which
helps to map the quaternion :math:`\\mathbf{q}` from a rotation vector
:math:`\\mathbf{v}`. This **exponential map** [Sola]_ is formerly defined as:

.. math::
    \\mathbf{q} = e^\\mathbf{v} =
    \\begin{bmatrix}\\cos\\frac{\\theta}{2} \\\\
    \\mathbf{u}\\sin\\frac{\\theta}{2}\\end{bmatrix}

Accumulating rotation over time in quaternion form is done by integrating the
differential equation of :math:`\\mathbf{q}` to the rotation rate definition.
This constant augmentation is sometimes termed the **Attitude Propagation**.

Because exact closed-form solutions of the integration are not available, an
approximation method is required. Numerical solutions, however, don't give a
continuous solution for :math:`\\mathbf{q}`, but a discrete set of values
:math:`\\mathbf{q}_t`, :math:`n=1,2,\\dots`

In our case, the angular rates are measured by `gyroscopes
<https://en.wikipedia.org/wiki/Gyroscope>`_, providing local measurements,
:math:`\\boldsymbol\\omega(t_n)=\\begin{bmatrix}\\omega_x&\\omega_y&\\omega_z\\end{bmatrix}^T`,
in *rad/s*, at discrete times :math:`t_n = n\\Delta t` in the local sensor
frame. The parameter :math:`\\Delta t` is called *time step* or *step size* of
the numerical integration.

Quaternion Derivative
---------------------

An orientation is described by :math:`\\mathbf{q} (t+\\Delta t)` at a time
:math:`t+\\Delta t`. This is after a rotation change :math:`\\Delta\\mathbf{q}`
during :math:`\\Delta t` seconds is performed on the local frame [Jia]_.

This rotation change about the instantaneous axis
:math:`\\mathbf{u}=\\frac{\\boldsymbol\\omega}{\\|\\boldsymbol\\omega\\|}`
through the angle :math:`\\theta=\\|\\boldsymbol\\omega\\|\\Delta t` can be
described by a quaternion too:

.. math::
    \\begin{array}{rcl}
    \\Delta\\mathbf{q} &=& \\cos\\frac{\\theta}{2} + \\mathbf{u}\\sin\\frac{\\theta}{2} \\\\
    &=& \\cos\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2} + \\frac{\\boldsymbol\\omega}{\\|\\boldsymbol\\omega\\|}\\sin\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2}
    \\end{array}

implying that :math:`\\mathbf{q}(t+\\Delta t)=\\Delta\\mathbf{qq}(t)`. To
obtain the derivative we consider :math:`\\mathbf{q} = \\mathbf{q}(t)` as the
original state at a time :math:`t`, and the difference between states as:

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}(t+\\Delta t)-\\mathbf{q}(t)
    &=& \\Big(\\cos\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2} + \\frac{\\boldsymbol\\omega}{\\|\\boldsymbol\\omega\\|}\\sin\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2}\\Big)\\mathbf{q} - \\mathbf{q} \\\\
    &=& \\Big(-2\\sin^2\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{4} + \\frac{\\boldsymbol\\omega}{\\|\\boldsymbol\\omega\\|}\\sin\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2}\\Big)\\mathbf{q}
    \\end{array}

The development of the `time-derivative <https://en.wikipedia.org/wiki/Differential_calculus#Derivative>`_
of the quaternions [#]_ follows the formal definition:

.. math::
    \\begin{array}{rcl}
    \\dot{\\mathbf{q}}
    &=& \\underset{\\Delta t\\to 0}{\\mathrm{lim}} \\frac{\\mathbf{q}(t+\\Delta t)-\\mathbf{q}(t)}{\\Delta t} \\\\
    &=& \\underset{\\Delta t\\to 0}{\\mathrm{lim}} \\frac{1}{\\Delta t}\\Big(-2\\sin^2\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{4} + \\frac{\\boldsymbol\\omega}{\\|\\boldsymbol\\omega\\|}\\sin\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2}\\Big)\\mathbf{q} \\\\
    &=& \\frac{\\boldsymbol\\omega}{\\|\\boldsymbol\\omega\\|} \\underset{\\Delta t\\to 0}{\\mathrm{lim}} \\frac{1}{\\Delta t}\\sin\\big(\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2}\\big) \\mathbf{q} \\\\
    &=& \\frac{\\boldsymbol\\omega}{\\|\\boldsymbol\\omega\\|} \\frac{d}{dt} \\sin\\big(\\frac{\\|\\boldsymbol\\omega\\|}{2}t\\big) \\Big |_{t=0} \\mathbf{q} \\\\
    &=& \\frac{1}{2}\\boldsymbol\\omega \\mathbf{q} \\\\
    &=& \\frac{1}{2}
    \\begin{bmatrix}
    -\\omega_x q_x -\\omega_y q_y - \\omega_z q_z\\\\
    \\omega_x q_w + \\omega_z q_y - \\omega_y q_z\\\\
    \\omega_y q_w - \\omega_z q_x + \\omega_x q_z\\\\
    \\omega_z q_w + \\omega_y q_x - \\omega_x q_y
    \\end{bmatrix}
    \\end{array}

.. warning::
    The product between the angular velocity :math:`\\boldsymbol\\omega\\in\\mathbb{R}^3`
    and the quaternion :math:`\\mathbf{q}\\in\\mathbb{H}^4` is done by
    representing the former as a pure quaternion :math:`\\boldsymbol\\omega=
    \\begin{bmatrix}0 & \\omega_x & \\omega_y & \\omega_z\\end{bmatrix}\\in\\mathbb{H}^4`,
    so that a `Hamilton Product <https://en.wikipedia.org/wiki/Quaternion#Hamilton_product>`_
    can be applied.

Defining the omega operator as

.. math::
    \\boldsymbol\\Omega(\\boldsymbol\\omega) =
    \\begin{bmatrix}0 & -\\boldsymbol\\omega^T \\\\ \\boldsymbol\\omega & \\lfloor\\boldsymbol\\omega\\rfloor_\\times\\end{bmatrix} =
    \\begin{bmatrix}
    0 & -\\omega_x & -\\omega_y & -\\omega_z \\\\
    \\omega_x & 0 & \\omega_z & -\\omega_y \\\\
    \\omega_y & -\\omega_z & 0 & \\omega_x \\\\
    \\omega_z & \\omega_y & -\\omega_x & 0
    \\end{bmatrix}

.. note::
    The expression :math:`\\lfloor\\mathbf{x}\\rfloor_\\times` expands a vector
    :math:`\\mathbf{x}\\in\\mathbb{R}^3` into a :math:`3\\times 3`
    `skew-symmetric matrix <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_:

    .. math::
        \\lfloor\\mathbf{x}\\rfloor_\\times =
        \\begin{bmatrix} 0 & x_2 & -x_1 \\\\ -x_2 & 0 & x_0 \\\\ x_1 & -x_0 & 0\\end{bmatrix}

we get an equivalent matrix expression for the derivative:

.. math::
    \\dot{\\mathbf{q}} = \\frac{1}{2}\\boldsymbol\\Omega(\\boldsymbol\\omega)\\mathbf{q}

This definition does **not** require a Hamilton product and the Omega operator
can be used to simply multiply with the quaternion as any other matrix
operation.

Quaternion Integration
----------------------

Now we develop an :math:`n^{th}`-order polynomial linearization method built
from `Taylor series <https://en.wikipedia.org/wiki/Taylor_series>`_ of
:math:`\\mathbf{q}(t+\\Delta t)` around the time :math:`t` for the quaternion:

.. math::
    \\mathbf{q}_{t+1} = \\mathbf{q}_t + \\dot{\\mathbf{q}}_t\\Delta t +
    \\frac{1}{2!}\\ddot{\\mathbf{q}}_t\\Delta t^2 +
    \\frac{1}{3!}\\dddot{\\mathbf{q}}_t\\Delta t^3 + \\cdots

Using the definition of :math:`\\dot{\\mathbf{q}}` the new orientation
:math:`\\mathbf{q}_{t+1}` is written as [Wertz]_:

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}_{t+1} &=& \\Bigg[\\mathbf{I}_4 + \\frac{1}{2}\\boldsymbol\\Omega(\\boldsymbol\\omega)\\Delta t +
    \\frac{1}{2!}\\Big(\\frac{1}{2}\\boldsymbol\\Omega(\\boldsymbol\\omega)\\Delta t\\Big)^2 +
    \\frac{1}{3!}\\Big(\\frac{1}{2}\\boldsymbol\\Omega(\\boldsymbol\\omega)\\Delta t\\Big)^3 + \\cdots\\Bigg]\\mathbf{q}_t \\\\
    && \\qquad{} + \\frac{1}{4}\\dot{\\boldsymbol\\Omega}(\\boldsymbol\\omega)\\Delta t^2\\mathbf{q}_t
    + \\Big[\\frac{1}{12}\\dot{\\boldsymbol\\Omega}(\\boldsymbol\\omega)\\boldsymbol\\Omega(\\boldsymbol\\omega)
    + \\frac{1}{24}\\boldsymbol\\Omega(\\boldsymbol\\omega)\\dot{\\boldsymbol\\Omega}(\\boldsymbol\\omega)
    + \\frac{1}{12}\\ddot{\\boldsymbol\\Omega}(\\boldsymbol\\omega)\\Big]\\Delta t^3\\mathbf{q}_t
    + \\cdots
    \\end{array}

Assuming the angular rate is constant over the period :math:`[t, t+1]`, we have
:math:`\\dot{\\boldsymbol\\omega}=0`, and we can prescind from the derivatives
of :math:`\\boldsymbol\\Omega` reducing the series to:

.. math::
    \\mathbf{q}_{t+1} = \\Bigg[\\mathbf{I}_4 + \\frac{1}{2}\\boldsymbol\\Omega(\\boldsymbol\\omega)\\Delta t +
    \\frac{1}{2!}\\Big(\\frac{1}{2}\\boldsymbol\\Omega(\\boldsymbol\\omega)\\Delta t\\Big)^2 +
    \\frac{1}{3!}\\Big(\\frac{1}{2}\\boldsymbol\\Omega(\\boldsymbol\\omega)\\Delta t\\Big)^3 + \\cdots\\Bigg]\\mathbf{q}_t

The error of the approximation vanishes rapidly at higher orders, or when the
time step :math:`\\Delta t \\to 0`. The more terms we have, the better our
approximation should be, assuming the sensor signals are unbiased and noiseless,
with the downside of a big computational demand.

Notice the series has the known form of the matrix exponential:

.. math::
    e^{\\frac{\\Delta t}{2}\\boldsymbol\\Omega(\\boldsymbol\\omega)} =
    \\sum_{k=0}^\\infty \\frac{1}{k!} \\Big(\\frac{\\Delta t}{2}\\boldsymbol\\Omega(\\boldsymbol\\omega)\\Big)^k

For our purpose a truncation up to the second term, making it of first order
(:math:`k=1`), is implemented.

.. math::
    \\mathbf{q}_{t+1} = \\Bigg[\\mathbf{I}_4 + \\frac{1}{2}\\boldsymbol\\Omega(\\boldsymbol\\omega)\\Delta t\\Bigg]\\mathbf{q}_t

Two good reasons for this truncation are:

- Simple architectures (like embedded systems) avoid the burden of the heavy
  computation, and still achieve fairly good results.
- If the sensor signals are not properly filtered and corrected, the estimation
  will not converge to a good result, and even worsening it at higher orders.

The resulting quaternion must be normalized to operate as a versor, and to
represent a valid orientation.

.. math::
    \\mathbf{q}_{t+1} \\gets \\frac{\\mathbf{q}_{t+1}}{\\|\\mathbf{q}_{t+1}\\|}

Assuming the gyroscope data is sampled at a fixed rate and that the angular
velocity vector, :math:`\\boldsymbol\\omega`, in local body coordinates is
constant over the sampling interval :math:`\\Delta t`, we can alternatively
integrate :math:`\\dot{\\mathbf{q}}` to obtain:

.. math::
    \\mathbf{q}_{t+1} = e^{\\frac{1}{2}\\boldsymbol\\Omega(\\boldsymbol\\omega)\\Delta t}\\mathbf{q}_t

Using the `Euler-Rodrigues rotation formula
<https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternion_as_rotations>`_
and the exponential map from above we find a *closed-form solution* [Wertz]_:

.. math::
    \\mathbf{q}_{t+1} =
    \\Bigg[
    \\cos\\Big(\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2}\\Big)\\mathbf{I}_4 +
    \\frac{1}{\\|\\boldsymbol\\omega\\|}\\sin\\Big(\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2}\\Big)\\boldsymbol\\Omega(\\boldsymbol\\omega)
    \\Bigg]\\mathbf{q}_t

where :math:`\\mathbf{I}_4` is a :math:`4\\times 4` Identity matrix. The large
term inside the brackets, multiplying :math:`\\mathbf{q}_t`, is an orthogonal
rotation retaining the normalization of the propagated attitude quaternions.
Thus, it is not necessary to normalize :math:`\\mathbf{q}_{t+1}`, but it is
highly recommended to do so in order to avoid any `round-off errors
<https://en.wikipedia.org/wiki/Round-off_error>`_ inherent to all computers.

Numerical Integration based on Runge-Kutta methods can be employed to increase
the accuracy, and are shown to be more effective. See [Sola]_ and [Zhao]_ for a
comparison of the different methods, their accuracy, and their computational
load.

Footnotes
---------
.. [#] A vector :math:`\\mathbf{x}` is unitary if its norm is equal to one:
    :math:`\\|\\mathbf{x}\\|=1`.
.. [#] The successive derivatives of :math:`\\mathbf{q}_n` are obtained by
    repeatedly applying the expression of the quaternion derivative,
    with :math:`\\ddot{\\boldsymbol\\omega}=0`.

    .. math::
        \\begin{array}{rcl}
        \\dot{\\mathbf{q}} &=& \\frac{1}{2}\\mathbf{q}\\boldsymbol\\omega \\\\
        \\ddot{\\mathbf{q}} &=& \\frac{1}{4}\\mathbf{q}\\boldsymbol\\omega^2 + \\frac{1}{2}\\mathbf{q}\\dot{\\boldsymbol\\omega} \\\\
        \\dddot{\\mathbf{q}} &=& \\frac{1}{6}\\mathbf{q}\\boldsymbol\\omega^3 + \\frac{1}{4}\\mathbf{q}\\dot{\\boldsymbol\\omega}\\boldsymbol\\omega + \\frac{1}{2}\\mathbf{q}\\boldsymbol\\omega\\dot{\\boldsymbol\\omega} \\\\
        \\mathbf{q}^{i\\geq 4} &=& \\frac{1}{2^i}\\mathbf{q}\\boldsymbol\\omega^i + \\cdots
        \\end{array}

References
----------
.. [Jia] Yan-Bin Jia. Quaternions. 2018.
    (http://web.cs.iastate.edu/~cs577/handouts/quaternion.pdf)
.. [Sola] Solà, Joan. Quaternion kinematics for the error-state Kalman Filter.
    October 12, 2017.
    (http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf)
.. [Zhao] F. Zhao and B.G.M. van Wachem. A novel Quaternion integration
    approach for describing the behaviour of non-spherical particles.
    (https://link.springer.com/content/pdf/10.1007/s00707-013-0914-2.pdf)

"""

import numpy as np

class AngularRate:
    """
    Quaternion update by integrating angular velocity

    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s.
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion).
    frequency : float, default: 100.0
        Sampling frequency in Herz.
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if ``frequency`` value is given.
    method : str, default: ``'closed'``
        Estimation method to use. Options are: ``'series'`` or ``'closed'``.
    order : int
        Truncation order, if method ``'series'`` is used.

    Attributes
    ----------
    gyr : numpy.ndarray
        N-by-3 array with N gyroscope samples.
    Q : numpy.ndarray
        Estimated quaternions.
    method : str
        Used estimation method.
    order : int
        Truncation order.

    Examples
    --------
    >>> gyro_data.shape             # NumPy arrays with gyroscope data in rad/s
    (1000, 3)
    >>> from ahrs.filters import AngularRate
    >>> angular_rate = AngularRate(gyr=gyro_data)
    >>> angular_rate.Q
    array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
           [ 9.99999993e-01,  2.36511228e-06, -1.12991334e-04,  4.28771947e-05],
           [ 9.99999967e-01,  1.77775173e-05, -2.43529706e-04,  8.33144162e-05],
           ...,
           [-0.92576208, -0.23633121,  0.19738534, -0.2194337 ],
           [-0.92547793, -0.23388968,  0.19889139, -0.22187479],
           [-0.92504595, -0.23174096,  0.20086376, -0.22414251]])
    >>> angular_rate.Q.shape        # Estimated attitudes as Quaternions
    (1000, 4)

    The estimation of each attitude is built upon the previous attitude. This
    estimator sets the initial attitude equal to the unit quaternion
    ``[1.0, 0.0, 0.0, 0.0]`` by default, because we cannot obtain the first
    orientation with gyroscopes only.

    We can use the class :class:`Tilt` to estimate the initial attitude with a
    simple measurement of a tri-axial accelerometer:

    >>> from ahrs.filter import Tilt
    >>> tilt = Tilt()
    >>> q_initial = tilt.estimate(acc=acc_sample)  # One tridimensional sample suffices
    >>> angular_rate = AngularRate(gyr=gyro_data, q0=q_initial)
    >>> angular_rate.Q
    array([[ 0.77547502,  0.6312126 ,  0.01121595, -0.00912944],
           [ 0.77547518,  0.63121388,  0.01110125, -0.00916754],
           [ 0.77546726,  0.63122508,  0.01097435, -0.00921875],
           ...,
           [-0.92576208, -0.23633121,  0.19738534, -0.2194337 ],
           [-0.92547793, -0.23388968,  0.19889139, -0.22187479],
           [-0.92504595, -0.23174096,  0.20086376, -0.22414251]])

    The :class:`Tilt` can also use a magnetometer to improve the estimation
    with the heading orientation.

    >>> q_initial = tilt.estimate(acc=acc_sample, mag=mag_sample)
    >>> angular_rate = AngularRate(gyr=gyro_data, q0=q_initial)
    >>> angular_rate.Q
    array([[ 0.66475674,  0.55050651, -0.30902706, -0.39942875],
           [ 0.66473764,  0.5504497 , -0.30912672, -0.39946172],
           [ 0.66470495,  0.55039529, -0.30924191, -0.39950193],
           ...,
           [-0.90988476, -0.10433118,  0.28970402,  0.27802214],
           [-0.91087203, -0.1014633 ,  0.28977124,  0.2757716 ],
           [-0.91164416, -0.09861271,  0.2903888 ,  0.27359606]])

    """
    def __init__(self, gyr: np.ndarray = None, q0: np.ndarray = None, frequency: float = 100.0, order: int = 1, **kw):
        self.gyr = gyr
        self.frequency = frequency
        self.order = order
        self.method = kw.get('method', 'closed')
        self.q0 = kw.get('q0', np.array([1.0, 0.0, 0.0, 0.0]))
        self.Dt = kw.get('Dt', 1.0/self.frequency)
        if self.gyr is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """Estimate all quaternions with given sensor values"""
        num_samples = len(self.gyr)
        Q = np.zeros((num_samples, 4))
        Q[0] = self.q0
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t], method=self.method, order=self.order)
        return Q

    def update(self, q: np.ndarray, gyr: np.ndarray, method: str = 'closed', order: int = 1) -> np.ndarray:
        """Update the quaternion estimation

        Estimate quaternion :math:`\\mathbf{q}_{t+1}` from given a-priori
        quaternion :math:`\\mathbf{q}_t` with a given angular rate measurement
        :math:`\\mathbf{\\omega}`.

        If ``method='closed'``, the new orienation is computed with the
        closed-form solution:

        .. math::
            \\mathbf{q}_{t+1} =
            \\Bigg[
            \\cos\\Big(\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2}\\Big)\\mathbf{I}_4 +
            \\frac{1}{\\|\\boldsymbol\\omega\\|}\\sin\\Big(\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2}\\Big)\\boldsymbol\\Omega(\\boldsymbol\\omega)
            \\Bigg]\\mathbf{q}_t

        Otherwise, if ``method='series'``, it is computed with a series of the
        form:

        .. math::
            \\mathbf{q}_{t+1} =
            \\Bigg[\\sum_{k=0}^\\infty \\frac{1}{k!} \\Big(\\frac{\\Delta t}{2}\\boldsymbol\\Omega(\\boldsymbol\\omega)\\Big)^k\\Bigg]\\mathbf{q}_t

        where the order :math:`k` in the series has to be set as non-negative
        integer in the parameter ``order``. By default it is set equal to 1.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Array with triaxial measurements of angular velocity in rad/s.
        method : str, default: ``'closed'``
            Estimation method to use. Options are: ``'series'`` or ``'closed'``.
        order : int
            Truncation order, if method ``'series'`` is used.

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        Examples
        --------
        >>> from ahrs.filters import AngularRate
        >>> gyro_data.shape
        (1000, 3)
        >>> num_samples = gyro_data.shape[0]
        >>> Q = np.zeros((num_samples, 4))      # Allocation of quaternions
        >>> Q[0] = [1.0, 0.0, 0.0, 0.0]         # Initial attitude as a quaternion
        >>> angular_rate = AngularRate()
        >>> for t in range(1, num_samples):
        ...     Q[t] = angular_rate.update(Q[t-1], gyro_data[t])
        ...
        >>> Q
        array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
               [ 9.99999993e-01,  2.36511228e-06, -1.12991334e-04,  4.28771947e-05],
               [ 9.99999967e-01,  1.77775173e-05, -2.43529706e-04,  8.33144162e-05],
               ...,
               [-0.92576208, -0.23633121,  0.19738534, -0.2194337 ],
               [-0.92547793, -0.23388968,  0.19889139, -0.22187479],
               [-0.92504595, -0.23174096,  0.20086376, -0.22414251]])

        """
        if method.lower() not in ['series', 'closed']:
            raise ValueError(f"Invalid method '{method}'. Try 'series' or 'closed'")
        q = np.copy(q)
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        Omega = np.array([
            [   0.0, -gyr[0], -gyr[1], -gyr[2]],
            [gyr[0],     0.0,  gyr[2], -gyr[1]],
            [gyr[1], -gyr[2],     0.0,  gyr[0]],
            [gyr[2],  gyr[1], -gyr[0],     0.0]])
        if method.lower() == 'closed':
            w = np.linalg.norm(gyr)
            A = np.cos(w*self.Dt/2.0)*np.eye(4) + np.sin(w*self.Dt/2.0)*Omega/w
        else:
            if order < 0:
                raise ValueError(f"The order must be an int equal or greater than 0. Got {order}")
            S = 0.5 * self.Dt * Omega
            A = np.identity(4)
            for i in range(1, order+1):
                A += S**i / np.math.factorial(i)
        q = A @ q
        return q / np.linalg.norm(q)

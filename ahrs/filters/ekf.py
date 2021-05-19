# -*- coding: utf-8 -*-
"""
Extended Kalman Filter
======================

The Extended Kalman Filter is one of the most used algorithms in the world, and
this module will use it to compute the attitude as a quaternion with the
observations of tri-axial gyroscopes, accelerometers and magnetometers.

The **state** is the physical state, which can be described by dynamic
variables. The **noise** in the measurements means that there is a certain
degree of uncertainty in them [Hartikainen2011]_.

A `dynamical system <https://en.wikipedia.org/wiki/Dynamical_system>`_ is a
system whose state evolves over time, so differential equations are normally
used to model them [Labbe2015]_. There is also noise in the dynamics of the
system, **process noise**, which means we cannot be entirely deterministic, but
we can get indirect noisy measurements.

Here, the term **filtering** refers to the process of *filtering out* the noise
in the measurements to provide an optimal estimate for the state, given the
observed measurements.

The instantaneous state of the system is represented with a vector updated
through discrete time increments to generate the next state. The simplest of
the state space models are linear models [Hartikainen2011]_, which can be
expressed with equations of the following form:

.. math::
    \\begin{array}{rcl}
    \\mathbf{x}_t &=& \\mathbf{Fx}_{t-1} + \\mathbf{w}_x \\\\
    \\mathbf{z}_t &=& \\mathbf{Hx}_t + \\mathbf{w}_z
    \\end{array}

where

- :math:`\\mathbf{x}_t\\in\\mathbb{R}^n` is the **state** of the system
  describing the condition of :math:`n` elements at time :math:`t`.
- :math:`\\mathbf{z}_t\\in\\mathbb{R}^m` are the **measurements** at time :math:`t`.
- :math:`\\mathbf{w}_x\\sim\\mathcal{N}(\\mathbf{0}, \\mathbf{Q}_t)` is the
  **process noise** at time :math:`t`.
- :math:`\\mathbf{w}_z\\sim\\mathcal{N}(\\mathbf{0}, \\mathbf{R}_t)` is the
  **measurement noise** at time :math:`t`.
- :math:`\\mathbf{F}\\in\\mathbb{R}^{n\\times n}` is called either the **State
  Transition Matrix** or the **Fundamental Matrix**, and sometimes is
  represented with :math:`\\mathbf{\\Phi}`. It depends on the literature.
- :math:`\\mathbf{H}` is the measurement model matrix.

Many linear models are also described with continuous-time state equations of
the form:

.. math::
    \\dot{\\mathbf{x}}_t = \\frac{d\\mathbf{x}_t}{dt} = \\mathbf{Ax}_t + \\mathbf{Lw}_t

where :math:`\\mathbf{A}` and :math:`\\mathbf{L}` are **constant** matrices
characterizing the behaviour of the model, and :math:`\\mathbf{w}_t` is a
`white noise <https://en.wikipedia.org/wiki/White_noise#Mathematical_definitions>`_
with a `power spectral density <https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density>`_
:math:`\\sigma_\\mathbf{w}^2`.

The main difference is that :math:`\\mathbf{A}` models a set of `linear
differential equations <https://en.wikipedia.org/wiki/Linear_differential_equation>`_,
and is continuous. :math:`\\mathbf{F}` is discrete, and represents a set of
linear equations (not differential equations) which transitions :math:`\\mathbf{x}_{t-1}`
to :math:`\\mathbf{x}_t` over a discrete time step :math:`\\Delta t`.

A common way to obtain :math:`\\mathbf{F}` uses the `matrix exponential
<https://en.wikipedia.org/wiki/Matrix_exponential>`_, which can be expanded
with a `Taylor series <https://en.wikipedia.org/wiki/Taylor_series>`_ [Sola]_:

.. math::
    \\mathbf{F} = e^{\\mathbf{A}\\Delta t} = \\mathbf{I} + \\mathbf{A}\\Delta t + \\frac{(\\mathbf{A}\\Delta t)^2}{2!} + \\frac{(\\mathbf{A}\\Delta t)^3}{3!} + \\cdots = \\sum_{k=0}^\\infty\\frac{(\\mathbf{A}\\Delta t)^k}{k!}

The main goal is to find an equation that recursively finds the value of
:math:`\\mathbf{x}_t` in terms of :math:`\\mathbf{x}_{t-1}`.

Kalman Filter
-------------

The solution proposed by [Kalman1960]_ models a system with a set of
:math:`n^{th}`-order differential equations, converts them into an equivalent
set of first-order differential equations, and puts them into the matrix form
:math:`\\dot{\\mathbf{x}}=\\mathbf{Ax}`. Once in this form several techniques
are used to convert these linear differential equations into the recursive
equation :math:`\\mathbf{x}_t = \\mathbf{Fx}_{t-1}`.

The Kalman filter has two steps:

1. The **prediction step** estimates the next state, and its covariance, at
time :math:`t` of the system given the previous state at time :math:`t-1`.

.. math::
    \\begin{array}{rcl}
    \\hat{\\mathbf{x}}_t &=& \\mathbf{F}\\mathbf{x}_{t-1} + \\mathbf{Bu}_t \\\\
    \\hat{\\mathbf{P}}_t &=& \\mathbf{F}\\mathbf{P}_{t-1}\\mathbf{F}^T + \\mathbf{Q}_t
    \\end{array}

2. The **correction step** rectifies the estimation with a set of measurements
:math:`\\mathbf{z}` at time :math:`t`.

.. math::
    \\begin{array}{rcl}
    \\mathbf{v}_t &=& \\mathbf{z}_t - \\mathbf{H}\\hat{\\mathbf{x}}_t \\\\
    \\mathbf{S}_t &=& \\mathbf{H} \\hat{\\mathbf{P}}_t \\mathbf{H}^T + \\mathbf{R} \\\\
    \\mathbf{K}_t &=& \\hat{\\mathbf{P}}_t \\mathbf{H}^T \\mathbf{S}_t^{-1} \\\\
    \\mathbf{x}_t &=& \\hat{\\mathbf{x}}_t + \\mathbf{K}_t \\mathbf{v}_t \\\\
    \\mathbf{P}_t &=& \\hat{\\mathbf{P}}_t - \\mathbf{K}_t \\mathbf{S}_t \\mathbf{K}_t^T
    \\end{array}

where:

- :math:`\\hat{\\mathbf{P}}_t\\in\\mathbb{R}^{n\\times n}` is the **Predicted
  Covariance** of the state before seeing the measurements :math:`\\mathbf{z}_t`.
- :math:`\\mathbf{P}_t\\in\\mathbb{R}^{n\\times n}` is the **Estimated
  Covariance** of the state after seeing the measurements :math:`\\mathbf{z}_t`.
- :math:`\\mathbf{u}_t\\in\\mathbb{R}^k` is a **Control input vector** defining the expected
  behaviour of the system.
- :math:`\\mathbf{B}\\in\\mathbb{R}^{n\\times k}` is the **Control input model**.
- :math:`\\mathbf{H}\\in\\mathbb{R}^{m\\times n}` is the **Observation model**
  linking the predicted state and the measurements.
- :math:`\\mathbf{v}_t\\in\\mathbb{R}^m` is the **Innovation** or **Measurement
  residual**.
- :math:`\\mathbf{S}_t\\in\\mathbb{R}^{m\\times m}` is the **Measurement
  Prediction Covariance**.
- :math:`\\mathbf{K}_t\\in\\mathbb{R}^{n\\times m}` is the filter *gain*,
  a.k.a. the **Kalman Gain**, telling how much the predictions should be
  corrected.

The *predicted* state :math:`\\hat{\\mathbf{x}}_t` is estimated in the first
step based on the previously computed state :math:`\\mathbf{x}_{t-1}`, and
later is *"corrected"* during the second step to obtain the final estimation
:math:`\\mathbf{x}_t`. A similar computation happens with its covariance
:math:`\\mathbf{P}`.

.. note::
    The `hat notation <https://en.wikipedia.org/wiki/Hat_operator#Estimated_value>`_
    is (mostly) used to indicate an *estimated value*. It marks here the
    calculation of the state in the prediction step at time :math:`t`.

The loop starts with prior mean :math:`\\mathbf{x}_0` and prior covariance
:math:`\\mathbf{P}_0`, which are defined by the system model.

Extended Kalman Filter
----------------------

The functions have been Gaussian and linear so far and, thus, the output was
always another Gaussian, but Gaussians are not closed under nonlinear functions.

The EKF handles nonlinearity by forming a Gaussian approximation to the joint
distribution of state :math:`\\mathbf{x}` and measurements :math:`\\mathbf{z}`
using `Taylor series <https://en.wikipedia.org/wiki/Taylor_series>`_ based
transformations [Hartikainen2011]_.

Likewise, the EKF is split into two steps:

**Prediction**

.. math::
    \\begin{array}{rcl}
    \\hat{\\mathbf{x}}_t &=& \\mathbf{f}(\\mathbf{x}_{t-1}, \\mathbf{u}_t) \\\\
    \\hat{\\mathbf{P}}_t &=& \\mathbf{F}(\\mathbf{x}_{t-1}, \\mathbf{u}_t)\\mathbf{P}_{t-1}\\mathbf{F}^T(\\mathbf{x}_{t-1}, \\mathbf{u}_t) + \\mathbf{Q}_t
    \\end{array}

**Correction**

.. math::
    \\begin{array}{rcl}
    \\mathbf{v}_t &=& \\mathbf{z}_t - \\mathbf{h}(\\mathbf{x}_t) \\\\
    \\mathbf{S}_t &=& \\mathbf{H}(\\mathbf{x}_t) \\hat{\\mathbf{P}}_t \\mathbf{H}^T(\\mathbf{x}_t) + \\mathbf{R}_t \\\\
    \\mathbf{K}_t &=& \\hat{\\mathbf{P}}_t \\mathbf{H}^T(\\mathbf{x}_t) \\mathbf{S}_t^{-1} \\\\
    \\mathbf{x}_t &=& \\hat{\\mathbf{x}}_t + \\mathbf{K}_t \\mathbf{v}_t \\\\
    \\mathbf{P}_t &=& \\big(\\mathbf{I}_4 - \\mathbf{K}_t\\mathbf{H}(\\mathbf{x}_t)\\big)\\hat{\\mathbf{P}}_t
    \\end{array}

where :math:`\\mathbf{f}` is the nonlinear dynamic model function, and
:math:`\\mathbf{h}` is the nonlinear measurement model function. The matrices
:math:`\\mathbf{F}` and :math:`\\mathbf{H}` are the `Jacobians
<https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`_ of :math:`\\mathbf{f}`
and :math:`\\mathbf{h}`, respectively:

.. math::
    \\begin{array}{rcl}
    \\mathbf{F}(\\mathbf{x}_{t-1}, \\mathbf{u}_t) &=& \\frac{\\partial \\mathbf{f}(\\mathbf{x}_{t-1}, \\mathbf{u}_t)}{\\partial \\mathbf{x}} \\\\
    \\mathbf{H}(\\mathbf{x}_t) &=& \\frac{\\partial \\mathbf{h}(\\mathbf{x}_t)}{\\partial \\mathbf{x}}
    \\end{array}

Notice that the matrices :math:`\\mathbf{F}_{t-1}` and :math:`\\mathbf{H}_t` of
the normal KF are replaced with Jacobian matrices
:math:`\\mathbf{F}(\\mathbf{x}_{t-1}, \\mathbf{u}_t)` and
:math:`\\mathbf{H}(\\hat{\\mathbf{x}}_t)` in the EKF, respectively. The
predicted state, :math:`\\hat{\\mathbf{x}}_t`, and the residual of the
prediction, :math:`\\mathbf{v}_t`, are also calculated differently.

.. attention::
    The state transition and the observation models **must** be `differentiable
    <https://en.wikipedia.org/wiki/Differentiable_function>`_ functions.

**Quaternion EKF**

In this case, we will use the EKF to estimate an orientation represented as a
quaternion :math:`\\mathbf{q}`. First, we *predict* the new state (newest
orientation) using the immediate measurements of the gyroscopes, then we
*correct* this state using the measurements of the accelerometers and
magnetometers.

All sensors are assumed to have a fixed sampling rate (:math:`f=\\frac{1}{\\Delta t}`),
even though the time step can be computed at any sample :math:`n` as
:math:`\\Delta t = t_n - t_{n-1}`. Numerical integration will give a discrete
set of :math:`n` values, that are approximations at discrete times :math:`t_n =
t_0 + n\\Delta t`.

Gyroscope data are treated as external inputs to the filter rather than as
measurements, and their measurement noises enter the filter as *process noise*
rather than as measurement noise [Sabatini2011]_.

For this model, the quaternion :math:`\\mathbf{q}` will be the **state vector**,
and the angular velocity :math:`\\boldsymbol\\omega`, in *rad/s*, will be the
**control vector**:

.. math::
    \\begin{array}{rcl}
    \\mathbf{x} &\\triangleq & \\mathbf{q}
    = \\begin{bmatrix}q_w & \\mathbf{q}_v\\end{bmatrix}^T = \\begin{bmatrix}q_w & q_x & q_y & q_z\\end{bmatrix}^T
    \\\\ && \\\\
    \\mathbf{u} &\\triangleq & \\boldsymbol\\omega
    = \\begin{bmatrix}\\omega_x & \\omega_y & \\omega_z\\end{bmatrix}^T
    \\end{array}

Therefore, transition models are described as:

.. math::
    \\begin{array}{rcl}
    \\dot{\\mathbf{q}}_t &=& \\mathbf{Aq}_{t-1} + \\mathbf{B}\\boldsymbol\\omega_t + \\mathbf{w}_\\mathbf{q} \\\\ && \\\\
    \\hat{\\mathbf{q}}_t &=& \\mathbf{Fq}_{t-1} = e^{\\mathbf{A}\\Delta t}\\mathbf{q}_{t-1}
    \\end{array}

with :math:`\\mathbf{w}_\\mathbf{q}` being the **process noise**.

Prediction Step
---------------

In the first step, the quaternion at time :math:`t` is **predicted** by
integrating the angular rate :math:`\\boldsymbol\\omega`, and adding it to the
formerly computed quaternion :math:`\\mathbf{q}_{t-1}`:

.. math::
    \\hat{\\mathbf{q}}_t = \\mathbf{q}_{t-1} + \\int_{t-1}^t\\boldsymbol\\omega\\, dt

This constant augmentation is sometimes termed the **Attitude Propagation**.
Because exact closed-form solutions of the integration are not available, an
approximation method is required.

**Discretization**

Using the `Euler-Rodrigues rotation formula
<https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Using_quaternion_as_rotations>`_
to redefine the quaternion [Sabatini2011]_ we find:

.. math::
    \\hat{\\mathbf{q}}_t =
    \\Bigg[
    \\cos\\Big(\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2}\\Big)\\mathbf{I}_4 +
    \\frac{2}{\\|\\boldsymbol\\omega\\|}\\sin\\Big(\\frac{\\|\\boldsymbol\\omega\\|\\Delta t}{2}\\Big)\\boldsymbol\\Omega_t
    \\Bigg]\\mathbf{q}_{t-1}

where :math:`\\mathbf{I}_4` is a :math:`4\\times 4` Identity matrix, and:

.. math::
    \\boldsymbol\\Omega_t =
    \\begin{bmatrix}
    0 & -\\boldsymbol\\omega^T\\\\ \\boldsymbol\\omega & \\lfloor\\boldsymbol\\omega\\rfloor_\\times
    \\end{bmatrix} =
    \\begin{bmatrix}
    0 & -\\omega_x & -\\omega_y & -\\omega_z \\\\
    \\omega_x & 0 & \\omega_z & -\\omega_y \\\\
    \\omega_y & -\\omega_z & 0 & \\omega_x \\\\
    \\omega_z & \\omega_y & -\\omega_x & 0
    \\end{bmatrix}

The expression :math:`\\lfloor\\mathbf{x}\\rfloor_\\times` expands a vector
:math:`\\mathbf{x}\\in\\mathbb{R}^3` into a :math:`3\\times 3` `skew-symmetric
matrix <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_. The large term
inside the brackets, multiplying :math:`\\mathbf{q}_{t-1}`, is an orthogonal
rotation retaining the normalization of the propagated attitude quaternions,
and we might be tempted to consider it equal to :math:`\\mathbf{F}`, but it is
not yet linear, although it is **already discrete**. It must be linearized to
be used in the EKF.

**Linearization**

:math:`n^{th}`-order polynomial linearization methods can be built from `Taylor
series <https://en.wikipedia.org/wiki/Taylor_series>`_ of
:math:`\\mathbf{q}(t_n+\\Delta t)` around the time :math:`t=t_n`:

.. math::
    \\mathbf{q}_t = \\mathbf{q}_{t-1} + \\dot{\\mathbf{q}}_{t-1}\\Delta t +
    \\frac{1}{2!}\\ddot{\\mathbf{q}}_{t-1}\\Delta t^2 +
    \\frac{1}{3!}\\dddot{\\mathbf{q}}_{t-1}\\Delta t^3 + \\cdots

where :math:`\\dot{\\mathbf{q}}=\\frac{d\\mathbf{q}}{dt}` is the derivative
of the quaternion [#]_:

.. math::
    \\begin{array}{rcl}
    \\dot{\\mathbf{q}}
    &=& \\underset{\\Delta t\\to 0}{\\mathrm{lim}} \\frac{\\mathbf{q}(t+\\Delta t) - \\mathbf{q}(t)}{\\Delta t} \\\\
    &=& \\frac{1}{2}\\boldsymbol\\Omega_t\\mathbf{q}_{t-1} \\\\\\
    &=& \\frac{1}{2}
    \\begin{bmatrix}
    -\\omega_x q_x -\\omega_y q_y - \\omega_z q_z\\\\
    \\omega_x q_w + \\omega_z q_y - \\omega_y q_z\\\\
    \\omega_y q_w - \\omega_z q_x + \\omega_x q_z\\\\
    \\omega_z q_w + \\omega_y q_x - \\omega_x q_y
    \\end{bmatrix}
    \\end{array}

The angular rates :math:`\\boldsymbol\\omega` are measured by the gyroscopes in
the local *sensor frame*. Hence, this term describes the evolution of the
orientation with respect to the local frame [Sola]_.

Using the definition of :math:`\\dot{\\mathbf{q}}`, the predicted state,
:math:`\\hat{\\mathbf{q}}_t` is written as [Wertz]_:

.. math::
    \\begin{array}{rcl}
    \\hat{\\mathbf{q}}_t &=& \\Bigg(\\mathbf{I}_4 + \\frac{1}{2}\\boldsymbol\\Omega_t\\Delta t +
    \\frac{1}{2!}\\Big(\\frac{1}{2}\\boldsymbol\\Omega_t\\Delta t\\Big)^2 +
    \\frac{1}{3!}\\Big(\\frac{1}{2}\\boldsymbol\\Omega_t\\Delta t\\Big)^3 + \\cdots\\Bigg)\\mathbf{q}_{t-1} \\\\
    && \\qquad{} + \\frac{1}{4}\\dot{\\boldsymbol\\Omega}_t\\Delta t^2\\mathbf{q}_{t-1}
    + \\Big[\\frac{1}{12}\\dot{\\boldsymbol\\Omega}_t\\boldsymbol\\Omega_t
    + \\frac{1}{24}\\boldsymbol\\Omega_t\\dot{\\boldsymbol\\Omega}_t
    + \\frac{1}{12}\\ddot{\\boldsymbol\\Omega}_t\\Big]\\Delta t^3\\mathbf{q}_{t-1}
    + \\cdots
    \\end{array}

Assuming the angular rate is constant over the period :math:`[t-1, t]`, we have
:math:`\\dot{\\boldsymbol\\omega}=0`, and we can prescind from the derivatives
of :math:`\\boldsymbol\\Omega` reducing the series to:

.. math::
    \\hat{\\mathbf{q}}_t = \\Bigg(\\mathbf{I}_4 + \\frac{1}{2}\\boldsymbol\\Omega_t\\Delta t +
    \\frac{1}{2!}\\Big(\\frac{1}{2}\\boldsymbol\\Omega_t\\Delta t\\Big)^2 +
    \\frac{1}{3!}\\Big(\\frac{1}{2}\\boldsymbol\\Omega_t\\Delta t\\Big)^3 + \\cdots\\Bigg)\\mathbf{q}_{t-1}

Notice the series has the known form of the matrix exponential:

.. math::
    e^{\\frac{\\Delta t}{2}\\boldsymbol\\Omega_t} =
    \\sum_{k=0}^\\infty \\frac{1}{k!} \\Big(\\frac{\\Delta t}{2}\\boldsymbol\\Omega_t\\Big)^k

The error of the approximation vanishes rapidly at higher orders, or when the
time step :math:`\\Delta t \\to 0`. The more terms we have, the better our
approximation becomes, with the downside of a big computational demand. For
simple architectures (like embedded systems) we can reduce this burden by
truncating the series to its second term making it a **First Order EKF** [#]_,
and achieving fairly good results. Thus, our **process model** shortens to:

.. math::
    \\begin{array}{rcl}
    \\hat{\\mathbf{q}}_t &=& \\mathbf{f}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t) \\\\
    &=&\\Big(\\mathbf{I}_4 + \\frac{\\Delta t}{2}\\boldsymbol\\Omega_t\\Big)\\mathbf{q}_{t-1} \\\\
    \\begin{bmatrix}\\hat{q_w} \\\\ \\hat{q_x} \\\\ \\hat{q_y} \\\\ \\hat{q_z}\\end{bmatrix}
    &=&
    \\begin{bmatrix}
        q_w - \\frac{\\Delta t}{2} \\omega_x q_x - \\frac{\\Delta t}{2} \\omega_y q_y - \\frac{\\Delta t}{2} \\omega_z q_z\\\\
        q_x + \\frac{\\Delta t}{2} \\omega_x q_w - \\frac{\\Delta t}{2} \\omega_y q_z + \\frac{\\Delta t}{2} \\omega_z q_y\\\\
        q_y + \\frac{\\Delta t}{2} \\omega_x q_z + \\frac{\\Delta t}{2} \\omega_y q_w - \\frac{\\Delta t}{2} \\omega_z q_x\\\\
        q_z - \\frac{\\Delta t}{2} \\omega_x q_y + \\frac{\\Delta t}{2} \\omega_y q_x + \\frac{\\Delta t}{2} \\omega_z q_w
    \\end{bmatrix}
    \\end{array}

And now, we simply compute :math:`\\mathbf{F}` as [#]_:

.. math::
    \\begin{array}{rcl}
    \\mathbf{F}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)
    &=& \\frac{\\partial\\mathbf{f}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)}{\\partial\\mathbf{q}} \\\\
    &=&\\begin{bmatrix}
    \\frac{\\partial\\mathbf{f}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)}{\\partial q_w} &
    \\frac{\\partial\\mathbf{f}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)}{\\partial q_x} &
    \\frac{\\partial\\mathbf{f}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)}{\\partial q_y} &
    \\frac{\\partial\\mathbf{f}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)}{\\partial q_z}
    \\end{bmatrix} \\\\ &=&
    \\begin{bmatrix}
        1 & - \\frac{\\Delta t}{2} \\omega_x & - \\frac{\\Delta t}{2} \\omega_y & - \\frac{\\Delta t}{2} \\omega_z\\\\
        \\frac{\\Delta t}{2} \\omega_x & 1 & \\frac{\\Delta t}{2} \\omega_z & - \\frac{\\Delta t}{2} \\omega_y\\\\
        \\frac{\\Delta t}{2} \\omega_y & - \\frac{\\Delta t}{2} \\omega_z & 1 & \\frac{\\Delta t}{2} \\omega_x\\\\
        \\frac{\\Delta t}{2} \\omega_z & \\frac{\\Delta t}{2} \\omega_y & - \\frac{\\Delta t}{2} \\omega_x & 1
    \\end{bmatrix}
    \\end{array}

**Process Noise Covariance**

Per definition, the predicted error state covariance is calculated as:

.. math::
    \\hat{\\mathbf{P}}_t = \\mathbf{F}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)\\mathbf{P}_{t-1}\\mathbf{F}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)^T + \\mathbf{Q}_t

We already know how to compute :math:`\\mathbf{F}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)`,
but we still need to compute the *Process Noise Covariance Matrix*
:math:`\\mathbf{Q}_t`.

The noise at the prediction step lies mainy in the control input. Consequently,
:math:`\\mathbf{Q}_t` is derived mainly from the gyroscope.

We define :math:`\\boldsymbol\\sigma_\\boldsymbol\\omega^2=
\\begin{bmatrix}\\sigma_{\\omega x}^2 & \\sigma_{\\omega y}^2 & \\sigma_{\\omega z}^2\\end{bmatrix}^T`
as the `spectral density <https://en.wikipedia.org/wiki/Noise_spectral_density>`_
of the gyroscopic noises on each axis, whose `standard deviation
<https://en.wikipedia.org/wiki/Standard_deviation>`_,
:math:`\\boldsymbol\\sigma_\\boldsymbol\\omega`, is specified as a scalar in
*rad/s*.

Without boring too much into `DSP <https://en.wikipedia.org/wiki/Digital_signal_processing>`_
analysis, we can frankly say that the smaller the spectral density of a sensor
is, the less noisy its signals are, and we would tend to trust its readings a
bit more. Normally, these noises can be found already in the datasheets
provided by the sensor manufacturers.

Taking for granted that the gyroscope is the same kind on its axes with the
manufacturer guaranteeing perfect orthogonality and uncorrelation between them,
we build the *spectral noise covariance matrix* as:

.. math::
    \\boldsymbol\\Sigma_\\boldsymbol\\omega =
    \\begin{bmatrix}
    \\sigma_{\\boldsymbol\\omega x}^2 & 0 & 0 \\\\
    0 & \\sigma_{\\boldsymbol\\omega y}^2 & 0 \\\\
    0 & 0 & \\sigma_{\\boldsymbol\\omega z}^2
    \\end{bmatrix}

The easiest solution is to assume that this noise is consistent, and our
process noise covariance would simply be :math:`\\mathbf{Q}_t =
\\boldsymbol\\Sigma_\\boldsymbol\\omega`, but in a real system this covariance
changes depending on the angular velocity, so it will need to be recomputed for
every prediction.

We need to know how much noise is added to the system over a discrete interval
:math:`\\Delta t` and, therefore, we integrate the process noise over the
interval :math:`[0, \\Delta t]`

.. math::
    \\mathbf{Q}_t = \\int_0^{\\Delta t} e^{\\mathbf{A}_t} \\boldsymbol\\Sigma_\\boldsymbol\\omega e^{\\mathbf{A}_t^T} dt

As we have seen, the matrix exponentials tend to be computationally demanding,
so, we opt to use the Jacobian of the prediction, but with respect to the
angular rate:

.. math::
    \\begin{array}{rcl}
    \\mathbf{W}_t &=& \\frac{\\partial\\mathbf{f}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)}{\\partial\\boldsymbol\\omega} \\\\
    &=& \\begin{bmatrix}
    \\frac{\\partial\\mathbf{f}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)}{\\partial\\omega_x} &
    \\frac{\\partial\\mathbf{f}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)}{\\partial\\omega_y} &
    \\frac{\\partial\\mathbf{f}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)}{\\partial\\omega_z}
    \\end{bmatrix} \\\\
    &=& \\frac{\\Delta t}{2}
    \\begin{bmatrix}
    -q_x & -q_y & -q_z \\\\
    q_w & -q_z & q_y \\\\
    q_z & q_w & -q_x \\\\
    -q_y & q_x & q_w
    \\end{bmatrix}
    \\end{array}

Notice this Jacobian is very similar to :math:`\\mathbf{F}`, but this one has
partial derivatives with respect to :math:`\\boldsymbol\\omega`. Having the
noise values and the Jacobian :math:`\\mathbf{W}_t`, the Process Noise
Covariance is computed at each time :math:`t` with:

.. math::
    \\mathbf{Q}_t = \\mathbf{W}_t\\boldsymbol\\Sigma_\\boldsymbol\\omega\\mathbf{W}_t^T

For convenience, it is assumed that the noises are equal on each axis, and
don't influence each other, yielding a white, uncorrelated and `isotropic
<https://en.wikipedia.org/wiki/Isotropic_position>`_ noise [#]_:

.. math::
    \\boldsymbol\\Sigma_\\boldsymbol\\omega = \\sigma_\\boldsymbol\\omega^2\\mathbf{I}_3

further simplifying the computation to:

.. math::
    \\mathbf{Q}_t = \\sigma_\\boldsymbol\\omega^2\\mathbf{W}_t\\mathbf{W}_t^T

.. warning::
    The assumption that the noise variances of the gyroscope axes are all equal
    (:math:`\\sigma_{wx}=\\sigma_{wy}=\\sigma_{wz}`) is almost never true in
    reality. It is possible to infer the individual variances through a careful
    modeling and calibration process [Lam2003]_. If these three different
    values are at hand, it is then recommended to compute the Process Noise
    Covariance with :math:`\\mathbf{W}_t\\boldsymbol\\Sigma_\\boldsymbol\\omega\\mathbf{W}_t^T`.

Finally, the prediction step of this model would propagate the covariance
matrix like:

.. math::
    \\hat{\\mathbf{P}}_t = \\mathbf{F}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)\\mathbf{P}_{t-1}\\mathbf{F}(\\mathbf{q}_{t-1}, \\boldsymbol\\omega_t)^T + \\sigma_\\boldsymbol\\omega^2\\mathbf{W}_t\\mathbf{W}_t^T

Correction Step
---------------

The gyroscope measurements have been used, so far, to predict the new state of
the quaternion, but there are also accelerometer and magnetometer readings
available, that can be used to *correct* the estimation.

We know, from the equations above, that the corrected state can be computed as:

.. math::
    \\mathbf{q}_t = \\hat{\\mathbf{q}}_t + \\mathbf{K}_t \\big(\\mathbf{z}_t - \\mathbf{h}(\\mathbf{q}_t)\\big)

where

- :math:`\\mathbf{z}_t\\in\\mathbb{R}^6` is the current measurement.
- :math:`\\mathbf{h}_t\\in\\mathbb{R}^6` is the predicted measurement.
- :math:`\\mathbf{K}_t\\in\\mathbb{R}^{4\\times 6}` is the Kalman gain.

.. tip::
    The Kalman Gain can be thought of as a *blending value* within the range
    :math:`[0.0, 1.0]`, that decides how much of the innovation
    :math:`\\mathbf{v}_t = \\mathbf{z}_t - \\mathbf{h}(\\mathbf{q}_t)` will be
    considered. You can see, for example, that:

    - If :math:`\\mathbf{K}=0`, there is no correction: :math:`\\mathbf{q}_t=\\hat{\\mathbf{q}}_t`.
    - If :math:`\\mathbf{K}=1`, the state will be corrected with *all* the
      innovation: :math:`\\mathbf{q}_t = \\hat{\\mathbf{q}}_t + \\mathbf{v}_t`.

We start by defining the measurement vector as:

.. math::
    \\mathbf{z}_t = \\begin{bmatrix}\\mathbf{a}_t \\\\ \\mathbf{m}_t\\end{bmatrix} =
    \\begin{bmatrix}a_x \\\\ a_y \\\\ a_z \\\\ m_x \\\\ m_y \\\\ m_z\\end{bmatrix}

These are the values obtained from the sensors, where :math:`\\mathbf{a}\\in\\mathbb{R}^3`
is a tri-axial accelerometer sample, in :math:`\\frac{m}{s^2}`, and
:math:`\\mathbf{m}\\in\\mathbb{R}^3` is a tri-axial magnetometer sample, in
:math:`\\mu T`.

Their noises :math:`\\mathbf{w}_\\mathbf{a}` and :math:`\\mathbf{w}_\\mathbf{m}`
are assumed independent white Gaussian with zero mean of covariances
:math:`\\boldsymbol\\Sigma_\\mathbf{a}=\\boldsymbol\\sigma_\\mathbf{a}^2\\mathbf{I}_3`
and :math:`\\boldsymbol\\Sigma_\\mathbf{m}=\\boldsymbol\\sigma_\\mathbf{m}^2\\mathbf{I}_3`.

However, while the gyroscopes give measurements in the *sensor frame*, the
accelerometers and magnetometers deliver measurements of the *global frame*. We
must represent the accelerometers and magnetometer readings, if we want to
correct the estimated orientation in the *sensor frame*.

The predicted quaternion :math:`\\hat{\\mathbf{q}}_t` describes the orientation
of the sensor frame with respect to the global frame. For that reason, it will
be used to rotate the sensor measurements.

To rotate any vector :math:`\\mathbf{x}\\in\\mathbb{R}^3` through a quaternion
:math:`\\mathbf{q}\\in\\mathbb{H}^4`, we can use its representation as a
rotation matrix :math:`\\mathbf{C}(\\mathbf{q})\\in\\mathbb{R}^{3\\times 3}`.
For the estimated quaternion :math:`\\hat{\\mathbf{q}}_t` this becomes:

.. math::
    \\begin{array}{rcl}
    \\mathbf{x}' &=& \\mathbf{C}(\\hat{\\mathbf{q}})\\mathbf{x} \\\\ &=&
    \\begin{bmatrix}
    1-2(\\hat{q}_y^2+\\hat{q}_z^2) & 2(\\hat{q}_x\\hat{q}_y-\\hat{q}_w\\hat{q}_z) & 2(\\hat{q}_x\\hat{q}_z+\\hat{q}_w\\hat{q}_y) \\\\
    2(\\hat{q}_x\\hat{q}_y+\\hat{q}_w\\hat{q}_z) & 1-2(\\hat{q}_x^2+\\hat{q}_z^2) & 2(\\hat{q}_y\\hat{q}_z-\\hat{q}_w\\hat{q}_x) \\\\
    2(\\hat{q}_x\\hat{q}_z-\\hat{q}_w\\hat{q}_y) & 2(\\hat{q}_w\\hat{q}_x+\\hat{q}_y\\hat{q}_z) & 1-2(\\hat{q}_x^2+\\hat{q}_y^2)
    \\end{bmatrix}
    \\begin{bmatrix}x_1 \\\\ x_2 \\\\ x_3 \\end{bmatrix}
    \\end{array}

**Global References**

There are two main global reference frames based on the `local tangent plane
<https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates>`_:

- **NED** defines the X-, Y-, and Z-axis colinear to the geographical *North*,
  *East*, and *Down* directions, respectively.
- **ENU** defines the X-, Y-, and Z-axis colinear to the geographical  *East*,
  *North*, and *Up* directions, respectively.

The gravitational acceleration vector in a global *NED* frame is normally
defined as :math:`\\mathbf{g}_\\mathrm{NED}=\\begin{bmatrix}0 & 0 & -9.81\\end{bmatrix}^T`,
where the normal acceleration, :math:`g_0\\approx 9.81 \\frac{m}{s^2}`, acts
solely on the Z-axis [#]_. For the *ENU* frame, it simply flips the sign along
the Z-axis, :math:`\\mathbf{g}_\\mathrm{ENU}=\\begin{bmatrix}0 & 0 & 9.81\\end{bmatrix}^T`.

Earth's magnetic field is also represented with a 3D vector, :math:`\\mathbf{r}
=\\begin{bmatrix}r_x & r_y & r_z\\end{bmatrix}^T`, whose values indicate the
direction of the magnetic flow. In an ideal case only the *North* component
holds a value, yielding the options :math:`\\mathbf{r}_\\mathrm{NED}=\\begin{bmatrix}r_x & 0 & 0\\end{bmatrix}^T`
or :math:`\\mathbf{r}_\\mathrm{ENU}=\\begin{bmatrix}0 & r_y & 0\\end{bmatrix}^T`

The geomagnetic field is, however, not regular on the planet, and it even
changes over time (see `WMM <../wmm.html>`_ for more details.) Usually, certain
authors prefer to define this referencial vector discarding the eastwardly
magnetic information, sometimes projecting its magnitude against the XY plane
with the help of the `magnetic dip <https://en.wikipedia.org/wiki/Magnetic_dip>`_
angle, :math:`\\theta`, resulting in :math:`\\mathbf{r}_\\mathrm{NED} =
\\begin{bmatrix}\\cos\\theta & 0 & \\sin\\theta\\end{bmatrix}^T` and
:math:`\\mathbf{r}_\\mathrm{ENU} = \\begin{bmatrix}0 & \\cos\\theta & -\\sin\\theta\\end{bmatrix}^T`.

From the acceleration and magnetic field merely their directions are required,
not really their magnitudes. We can, therefore, use their normalized values,
simplifying their definition to:

.. math::
    \\begin{array}{rcl}
    \\mathbf{g} &=&
    \\left\\{
    \\begin{array}{ll}
        \\begin{bmatrix}0 & 0 & -1\\end{bmatrix}^T & \\mathrm{if}\\; \\mathrm{NED} \\\\
        \\begin{bmatrix}0 & 0 & 1\\end{bmatrix}^T & \\mathrm{if}\\; \\mathrm{ENU}
    \\end{array}
    \\right.\\\\ && \\\\
    \\mathbf{r} &=&
    \\left\\{
    \\begin{array}{ll}
        \\frac{1}{\\sqrt{\\cos^2\\theta+\\sin^2\\theta}}\\begin{bmatrix}\\cos\\theta & 0 & \\sin\\theta\\end{bmatrix}^T & \\mathrm{if}\\; \\mathrm{NED} \\\\
        \\frac{1}{\\sqrt{\\cos^2\\theta+\\sin^2\\theta}}\\begin{bmatrix}0 & \\cos\\theta & -\\sin\\theta\\end{bmatrix}^T & \\mathrm{if}\\; \\mathrm{ENU}
    \\end{array}
    \\right.
    \\end{array}

To compare these vectors against their corresponding observations, we must
**also normalize** the sensors' measurements:

.. math::
    \\begin{array}{rcl}
    \\mathbf{a} &=& \\frac{1}{\\sqrt{a_x^2+a_y^2+a_z^2}} \\begin{bmatrix}a_x & a_y & a_z\\end{bmatrix}^T \\\\ && \\\\
    \\mathbf{m} &=& \\frac{1}{\\sqrt{m_x^2+m_y^2+m_z^2}} \\begin{bmatrix}m_x & m_y & m_z\\end{bmatrix}^T
    \\end{array}

**Measurement Model**

The *expected* gravitational acceleration in the sensor frame, :math:`\\hat{\\mathbf{a}}`,
can be estimated from the *estimated orientation*:

.. math::
    \\hat{\\mathbf{a}} = \\mathbf{C}(\\hat{\\mathbf{q}})^T \\mathbf{g}

Similarly, the *expected magnetic field* in sensor frame, :math:`\\hat{\\mathbf{m}}`
is:

.. math::
    \\hat{\\mathbf{m}} = \\mathbf{C}(\\hat{\\mathbf{q}})^T \\mathbf{r}

The measurement model, :math:`\\mathbf{h}(\\hat{\\mathbf{q}}_t)`, and its
Jacobian, :math:`\\mathbf{H}(\\hat{\\mathbf{q}}_t)`, can be used to correct the
predicted model. The *Measurement model* is directly defined with these
transformations as:

.. math::
    \\begin{array}{rcl}
    \\mathbf{h}(\\hat{\\mathbf{q}}_t) &=& \\begin{bmatrix}\\hat{\\mathbf{a}} \\\\ \\hat{\\mathbf{m}}\\end{bmatrix}
    = \\begin{bmatrix}\\mathbf{C}(\\hat{\\mathbf{q}})^T \\mathbf{g} \\\\ \\mathbf{C}(\\hat{\\mathbf{q}})^T \\mathbf{r}\\end{bmatrix} \\\\
    &=&
    2 \\begin{bmatrix}
    g_x (\\frac{1}{2} - q_y^2 - q_z^2) + g_y (q_wq_z + q_xq_y) + g_z (q_xq_z - q_wq_y) \\\\
    g_x (q_xq_y - q_wq_z) + g_y (\\frac{1}{2} - q_x^2 -  q_z^2) + g_z (q_wq_x + q_yq_z) \\\\
    g_x (q_wq_y + q_xq_z) + g_y (q_yq_z - q_wq_x) + g_z (\\frac{1}{2} - q_x^2 -  q_y^2) \\\\
    r_x (\\frac{1}{2} - q_y^2 - q_z^2) + r_y (q_wq_z + q_xq_y) + r_z (q_xq_z - q_wq_y) \\\\
    r_x (q_xq_y - q_wq_z) + r_y (\\frac{1}{2} - q_x^2 -  q_z^2) + r_z (q_wq_x + q_yq_z) \\\\
    r_x (q_wq_y + q_xq_z) + r_y (q_yq_z - q_wq_x) + r_z (\\frac{1}{2} - q_x^2 -  q_y^2)
    \\end{bmatrix}
    \\end{array}

The measurement equations are nonlinear, which forces to, accordingly, compute
their Jacobian matrix as:

.. math::
    \\begin{array}{rcl}
    \\mathbf{H}(\\hat{\\mathbf{q}}_t) &=& \\frac{\\partial\\mathbf{h}(\\hat{\\mathbf{q}}_t)}{\\partial\\mathbf{q}} \\\\
    &=&\\begin{bmatrix}
    \\frac{\\partial\\mathbf{h}(\\hat{\\mathbf{q}}_t)}{\\partial q_w} &
    \\frac{\\partial\\mathbf{h}(\\hat{\\mathbf{q}}_t)}{\\partial q_x} &
    \\frac{\\partial\\mathbf{h}(\\hat{\\mathbf{q}}_t)}{\\partial q_y} &
    \\frac{\\partial\\mathbf{h}(\\hat{\\mathbf{q}}_t)}{\\partial q_z}
    \\end{bmatrix} \\\\
    &=&
    2
    \\begin{bmatrix}
    g_yq_z - g_zq_y & g_yq_y + g_zq_z & - 2g_xq_y + g_yq_x - g_zq_w & - 2g_xq_z + g_yq_w + g_zq_x \\\\
    -g_xq_z + g_zq_x & g_xq_y - 2g_yq_x + g_zq_w & g_xq_x + g_zq_z & -g_xq_w - 2g_yq_z + g_zq_y \\\\
    g_xq_y - g_yq_x & g_xq_z - g_yq_w - 2g_zq_x & g_xq_w + g_yq_z - 2g_zq_y & g_xq_x + g_yq_y \\\\
    r_yq_z - r_zq_y & r_yq_y + r_zq_z & - 2r_xq_y + r_yq_x - r_zq_w & - 2r_xq_z + r_yq_w + r_zq_x \\\\
    - r_xq_z + r_zq_x & r_xq_y - 2r_yq_x + r_zq_w & r_xq_x + r_zq_z & - r_xq_w - 2r_yq_z + r_zq_y \\\\
    r_xq_y - r_yq_x & r_xq_z - r_yq_w - 2r_zq_x & r_xq_w + r_yq_z - 2r_zq_y & r_xq_x + r_yq_y
    \\end{bmatrix}
    \\end{array}

This Jacobian can be refactored as:

.. math::
    \\mathbf{H}(\\hat{\\mathbf{q}}_t) =
    \\begin{bmatrix}
    \\frac{\\partial\\hat{\\mathbf{a}}}{\\partial\\mathbf{q}} \\\\
    \\frac{\\partial\\hat{\\mathbf{m}}}{\\partial\\mathbf{q}}
    \\end{bmatrix} = 2
    \\begin{bmatrix}
    \\mathbf{u}_g & \\lfloor\\mathbf{u}_g+\\hat{q}_w\\mathbf{g}\\rfloor_\\times + (\\hat{\\mathbf{q}}_v\\cdot\\mathbf{g})\\mathbf{I}_3 - \\mathbf{g}\\hat{\\mathbf{q}}_v^T \\\\
    \\mathbf{u}_r & \\lfloor\\mathbf{u}_r+\\hat{q}_w\\mathbf{r}\\rfloor_\\times + (\\hat{\\mathbf{q}}_v\\cdot\\mathbf{r})\\mathbf{I}_3 - \\mathbf{r}\\hat{\\mathbf{q}}_v^T
    \\end{bmatrix}

with

.. math::
    \\begin{array}{rcl}
    \\mathbf{u}_g &=& \\lfloor\\mathbf{g}\\rfloor_\\times\\hat{\\mathbf{q}}_v = \\mathbf{g}\\times\\hat{\\mathbf{q}}_v \\\\
    \\mathbf{u}_r &=& \\lfloor\\mathbf{r}\\rfloor_\\times\\hat{\\mathbf{q}}_v = \\mathbf{r}\\times\\hat{\\mathbf{q}}_v
    \\end{array}

The measurement noise covariance matrix, :math:`\\mathbf{R}\\in\\mathbb{R}^{6\\times 6}`,
is expressed directly in terms of the statistics of the measurement noise
affecting each sensor [Sabatini2011]_. The sensor noises are considered as
uncorrelated and isotropic, which creates a diagonal matrix:

.. math::
    \\mathbf{R} =
    \\begin{bmatrix}
    \\boldsymbol\\sigma_\\mathbf{a}^2\\mathbf{I}_3 & \\mathbf{0}_3 \\\\
    \\mathbf{0}_3 & \\boldsymbol\\sigma_\\mathbf{m}^2\\mathbf{I}_3
    \\end{bmatrix}

This definition allows us to obtain simple expressions for :math:`\\mathbf{P}_t`.
The rest of the EKF elements in the correction step are obtained as defined
originally:

.. math::
    \\begin{array}{rcl}
    \\mathbf{v}_t &=& \\mathbf{z}_t - \\mathbf{h}(\\hat{\\mathbf{q}}_t) \\\\
    \\mathbf{S}_t &=& \\mathbf{H}(\\hat{\\mathbf{q}}_t) \\hat{\\mathbf{P}}_t \\mathbf{H}^T(\\hat{\\mathbf{q}}_t) + \\mathbf{R} \\\\
    \\mathbf{K}_t &=& \\hat{\\mathbf{P}}_t \\mathbf{H}^T(\\hat{\\mathbf{q}}_t) \\mathbf{S}_t^{-1}
    \\end{array}

Lastly, the corrected state (quaternion) and its covariance at time :math:`t`
are computed with:

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}_t &=& \\hat{\\mathbf{q}}_t + \\mathbf{K}_t \\mathbf{v}_t \\\\
    \\mathbf{P}_t &=& \\big(\\mathbf{I}_4 - \\mathbf{K}_t\\mathbf{H}(\\hat{\\mathbf{q}}_t)\\big)\\hat{\\mathbf{P}}_t
    \\end{array}

The EKF is not an optimal estimator and might diverge quickly with an incorrect
model. The origin of the problem lies in the lack of independence of the four
components of the quaternion, since they are related by the unit-norm
constraint. The easiest solution is to normalize the corrected state.

.. math::
    \\mathbf{q}_t \\leftarrow \\frac{1}{\\|\\mathbf{q}_t\\|}\\mathbf{q}_t

Even though it is neither elegant nor optimal, this "brute-force" approach to
compute the final quaternion is proven to work generally well [Sabatini2011]_.

Initial values
--------------

The EKF state vector represents the quaternion, and we must provide an
appropriate initial state, :math:`\\mathbf{q}_0`, that corresponds to a valid
quaternion. That means, the initial state must have a norm equal to one
(:math:`\\|\\mathbf{q}_0\\|=1`)

To estimate this initial orientation a simple `ecompass
<https://en.wikipedia.org/wiki/Ecompass>`_ method is used, which requires
single observations of a tri-axial accelerometer and a tri-axial magnetometer,
in a motionless state.

Similar to `TRIAD <.triad.html>`_, the orientation can be estimated with only
one observation of both sensors. A rotation matrix :math:`\\mathbf{C}\\in\\mathbb{R}^{3\\times 3}`
of this estimation is computed as:

.. math::
    \\mathbf{C} =
    \\begin{bmatrix}
    (\\mathbf{a}_0\\times\\mathbf{m}_0)\\times\\mathbf{a}_0 & \\mathbf{a}_0\\times\\mathbf{m}_0 & \\mathbf{a}_0
    \\end{bmatrix}

where :math:`\\mathbf{a}_0` and :math:`\\mathbf{m}_0` are the accelerometer and
magnetometer measurements. Each column of this rotation matrix should be
normalized. Then, we get the initial quaternion with [Chiaverini]_:

.. math::
    \\mathbf{q}_0 =
    \\begin{bmatrix}
    \\frac{1}{2}\\sqrt{c_{11} + c_{22} + c_{33} + 1} \\\\
    \\frac{1}{2}\\mathrm{sgn}(c_{32} - c_{23}) \\sqrt{c_{11}-c_{22}-c_{33}+1} \\\\
    \\frac{1}{2}\\mathrm{sgn}(c_{13} - c_{31}) \\sqrt{c_{22}-c_{33}-c_{11}+1} \\\\
    \\frac{1}{2}\\mathrm{sgn}(c_{21} - c_{12}) \\sqrt{c_{33}-c_{11}-c_{22}+1}
    \\end{bmatrix}

The initial state covariance matrix, :math:`\\mathbf{P}_0` is set equal to a
:math:`4\\times 4` identity matrix :math:`\\mathbf{I}_4`.

Appropriate values of :math:`\\boldsymbol\\sigma_\\boldsymbol\\omega^2`,
:math:`\\boldsymbol\\sigma_\\mathbf{a}^2` and :math:`\\boldsymbol\\sigma_\\mathbf{m}^2`
should also be provided. These values can be normally found in the datasheet of
each sensor. The accelerometer's and magnetometers measurement noise variances
are increased in value, thus giving more weight to the filter predictions. For
our case, the default values are:

.. math::
    \\begin{array}{rcl}
    \\boldsymbol\\sigma_\\boldsymbol\\omega^2 &=& 0.3^2 \\\\
    \\boldsymbol\\sigma_\\mathbf{a}^2 &=& 0.5^2 \\\\
    \\boldsymbol\\sigma_\\mathbf{m}^2 &=& 0.8^2
    \\end{array}

Final Notes
-----------

The model described and implemented here does not consider any biases in the
signal, which can be also added to the state vector to be dynamically computed.
But this extra analysis is left out to provide a simple EKF. Therefore, the
given **sensor signals are considered to be calibrated already**.

No compensation for magnetic disturbances is considered either. It is assumed
that the magnetometer has been calibrated in the enviroment, where it is used,
so that scaling and bias errors are neglected too.

The modularity of the class allows for scalability in the estimation. It is
possible, for example, to expand the state vector (and its covariance) to
include the bias terms, if we need to, by simply setting the attribute ``q``.

Footnotes
---------
.. [#] The successive derivatives of :math:`\\mathbf{q}_n` are obtained by
    repeatedly applying the expression of the quaternion derivative,
    with :math:`\\ddot{\\boldsymbol\\omega}=0`.

    .. math::
        \\begin{array}{rcl}
        \\dot{\\mathbf{q}}_n &=& \\frac{1}{2}\\mathbf{q}_n\\boldsymbol\\omega_n \\\\
        \\ddot{\\mathbf{q}}_n &=& \\frac{1}{4}\\mathbf{q}_n\\boldsymbol\\omega_n^2 + \\frac{1}{2}\\mathbf{q}_n\\dot{\\boldsymbol\\omega} \\\\
        \\dddot{\\mathbf{q}}_n &=& \\frac{1}{6}\\mathbf{q}_n\\boldsymbol\\omega_n^3 + \\frac{1}{4}\\mathbf{q}_n\\dot{\\boldsymbol\\omega}\\boldsymbol\\omega_n + \\frac{1}{2}\\mathbf{q}\\boldsymbol\\omega_n\\dot{\\boldsymbol\\omega} \\\\
        \\mathbf{q}_n^{i\\geq 4} &=& \\frac{1}{2^i}\\mathbf{q}_n\\boldsymbol\\omega_n^i + \\cdots
        \\end{array}
.. [#] Other applications of the EKF truncate the series after the second term,
    producing a 2nd, 3rd, or nth- order EKF. This slightly increases the
    accuracy of the model, with the disadvantage of an elevated computational
    demand.
.. [#] Numerically, the :math:`\\mathbf{F}` can be *approximated* with the
    `matrix exponential <https://en.wikipedia.org/wiki/Matrix_exponential>`_
    :math:`e^{\\mathbf{A}\\Delta t}`. This function has been largerly
    implemented in several packages, including `Scipy
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html>`_:
    ``F=scipy.linalg.expm(A*Dt)``; however, the Jacobian of :math:`\\mathbf{f}`
    is always prefered.
.. [#] A three-dimensional isotropic covariance describes a sphere centered at
    the origin, which means that its mean and covariance matrix are invariant
    upon rotations. In algebraic terms, an isotropic covariance matrix is an
    identity matrix multiplied by a scalar.
.. [#] The magnitude of the normal gravity also changes with respect to the
    location on Earth. Simpler models consider only the latitude and height
    above sea level to estimate this magnitude. For the purpose of analysis a
    common value of *9.81* is given.

References
----------
.. [Kalman1960] Rudolf Kalman. A New Approach to Linear Filtering and Prediction
    Problems. 1960.
.. [Hartikainen2011] J. Hartikainen, A. Solin and S. Särkkä. Optimal Filtering with
    Kalman Filters and Smoothers. 2011
.. [Sabatini2011] Sabatini, A.M. Kalman-Filter-Based Orientation Determination
    Using Inertial/Magnetic Sensors: Observability Analysis and Performance
    Evaluation. Sensors 2011, 11, 9182-9206.
    (https://www.mdpi.com/1424-8220/11/10/9182)
.. [Labbe2015] Roger R. Labbe Jr. Kalman and Bayesian Filters in Python.
    (https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
.. [Lam2003] Lam, Quang & Stamatakos, Nick & Woodruff, Craig & Ashton, Sandy.
    Gyro Modeling and Estimation of Its Random Noise Sources. AAIA 2003.
    DOI: 10.2514/6.2003-5562.
    (https://www.researchgate.net/publication/268554081)


"""

import numpy as np
from ..common.orientation import q2R, ecompass, acc2q
from ..common.mathfuncs import cosd, sind, skew

class EKF:
    """
    Extended Kalman Filter to estimate orientation as Quaternion.

    Examples
    --------
    >>> import numpy as np
    >>> from ahrs.filters import EKF
    >>> from ahrs.common.orientation import acc2q
    >>> ekf = EKF()
    >>> num_samples = 1000              # Assuming sensors have 1000 samples each
    >>> Q = np.zeros((num_samples, 4))  # Allocate array for quaternions
    >>> Q[0] = acc2q(acc_data[0])       # First sample of tri-axial accelerometer
    >>> for t in range(1, num_samples):
    ...     Q[t] = ekf.update(Q[t-1], gyr_data[t], acc_data[t])

    The estimation is simplified by giving the sensor values at the
    construction of the EKF object. This will perform all steps above and store
    the estimated orientations, as quaternions, in the attribute ``Q``.

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data)
    >>> ekf.Q.shape
    (1000, 4)

    In this case, the measurement vector, set in the attribute ``z``, is equal
    to the measurements of the accelerometer. If extra information from a
    magnetometer is available, it will also be considered to estimate the
    attitude.

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data)
    >>> ekf.Q.shape
    (1000, 4)

    For this case, the measurement vector contains the accelerometer and
    magnetometer measurements together: ``z = [acc_data, mag_data]`` at each
    time :math:`t`.

    The most common sampling frequency is 100 Hz, which is used in the filter.
    If that is different in the given sensor data, it can be changed too.

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, frequency=200.0)

    Normally, when using the magnetic data, a referencial magnetic field must
    be given. This filter computes the local magnetic field in Munich, Germany,
    but it can also be set to a different reference with the parameter
    ``mag_ref``.

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, magnetic_ref=[17.06, 0.78, 34.39])

    If the full referencial vector is not available, the magnetic dip angle, in
    degrees, can be also used.

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, magnetic_ref=60.0)

    The initial quaternion is estimated with the first observations of the
    tri-axial accelerometers and magnetometers, but it can also be given
    directly in the parameter ``q0``.

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, q0=[0.7071, 0.0, -0.7071, 0.0])

    Measurement noise variances must be set from each sensor, so that the
    Process and Measurement Covariance Matrix can be built. They are set in an
    array equal to ``[0.3**2, 0.5**2, 0.8**2]`` for the gyroscope,
    accelerometer and magnetometer, respectively. If a different set of noise
    variances is used, they can be set with the parameter ``noises``:

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, noises=[0.1**2, 0.3**2, 0.5**2])

    or the individual variances can be set separately too:

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, var_acc=0.3**2)

    This class can also differentiate between NED and ENU frames. By default it
    estimates the orientations using the NED frame, but ENU is used if set in
    its parameter:

    >>> ekf = EKF(gyr=gyr_data, acc=acc_data, mag=mag_data, frame='ENU')

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
    frame : str, default: 'NED'
        Local tangent plane coordinate frame. Valid options are right-handed
        ``'NED'`` for North-East-Down and ``'ENU'`` for East-North-Up.
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).
    magnetic_ref : float or numpy.ndarray
        Local magnetic reference.
    noises : numpy.ndarray
        List of noise variances for each type of sensor. Default values:
        ``[0.3**2, 0.5**2, 0.8**2]``.
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. NOT required
        if ``frequency`` value is given.

    """
    def __init__(self,
        gyr: np.ndarray = None,
        acc: np.ndarray = None,
        mag: np.ndarray = None,
        frequency: float = 100.0,
        frame: str = 'NED',
        **kwargs):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.frequency = frequency
        self.frame = frame                          # Local tangent plane coordinate frame
        self.Dt = kwargs.get('Dt', 1.0/self.frequency)
        self.q0 = kwargs.get('q0')
        self.P = np.identity(4)                     # Initial state covariance
        self.R = self._set_measurement_noise_covariance(**kwargs)
        self._set_reference_frames(kwargs.get('magnetic_ref'), self.frame)
        # Process of data is given
        if self.gyr is not None and self.acc is not None:
            self.Q = self._compute_all(self.frame)

    def _set_measurement_noise_covariance(self, **kw) -> np.ndarray:
        self.noises = np.array(kw.get('noises', [0.3**2, 0.5**2, 0.8**2]))
        if 'var_gyr' in kw:
            self.noises[0] = kw.get('var_gyr')
        if 'var_acc' in kw:
            self.noises[1] = kw.get('var_acc')
        if 'var_mag' in kw:
            self.noises[2] = kw.get('var_mag')
        self.g_noise, self.a_noise, self.m_noise = self.noises
        return np.diag(np.repeat(self.noises[1:], 3))

    def _set_reference_frames(self, mref: float, frame: str = 'NED') -> None:
        if frame.upper() not in ['NED', 'ENU']:
            raise ValueError(f"Invalid frame '{frame}'. Try 'NED' or 'ENU'")
        # Magnetic Reference Vector
        if mref is None:
            # Local magnetic reference of Munich, Germany
            from ..common.mathfuncs import MUNICH_LATITUDE, MUNICH_LONGITUDE, MUNICH_HEIGHT
            from ..utils.wmm import WMM
            wmm = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT)
            self.m_ref = np.array([wmm.X, wmm.Y, wmm.Z]) if frame.upper() == 'NED' else np.array([wmm.Y, wmm.X, -wmm.Z])
        elif isinstance(mref, (int, float)):
            cd, sd = cosd(mref), sind(mref)
            self.m_ref = np.array([cd, 0.0, sd]) if frame.upper() == 'NED' else np.array([0.0, cd, -sd])
        else:
            self.m_ref = np.copy(mref)
        self.m_ref /= np.linalg.norm(self.m_ref)
        # Gravitational Reference Vector
        self.a_ref = np.array([0.0, 0.0, -1.0]) if frame.upper() == 'NED' else np.array([0.0, 0.0, 1.0])

    def _compute_all(self, frame: str) -> np.ndarray:
        """
        Estimate the quaternions given all sensor data.

        Attributes ``gyr``, ``acc`` MUST contain data. Attribute ``mag`` is
        optional.

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
        Q[0] = self.q0
        if self.mag is not None:
            ###### Compute attitude with MARG architecture ######
            if self.mag.shape != self.gyr.shape:
                raise ValueError("mag and gyr are not the same size")
            if self.q0 is None:
                Q[0] = ecompass(self.acc[0], self.mag[0], frame=frame, representation='quaternion')
            Q[0] /= np.linalg.norm(Q[0])
            # EKF Loop over all data
            for t in range(1, num_samples):
                Q[t] = self.update(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
            return Q
        ###### Compute attitude with IMU architecture ######
        if self.q0 is None:
            Q[0] = acc2q(self.acc[0])
        Q[0] /= np.linalg.norm(Q[0])
        # EKF Loop over all data
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t], self.acc[t])
        return Q

    def Omega(self, x: np.ndarray) -> np.ndarray:
        """Omega operator.

        Given a vector :math:`\\mathbf{x}\\in\\mathbb{R}^3`, return a
        :math:`4\\times 4` matrix of the form:

        .. math::
            \\boldsymbol\\Omega(\\mathbf{x}) =
            \\begin{bmatrix}
            0 & -\\mathbf{x}^T \\\\ \\mathbf{x} & \\lfloor\\mathbf{x}\\rfloor_\\times
            \\end{bmatrix} =
            \\begin{bmatrix}
            0 & -x_1 & -x_2 & -x_3 \\\\
            x_1 & 0 & x_3 & -x_2 \\\\
            x_2 & -x_3 & 0 & x_1 \\\\
            x_3 & x_2 & -x_1 & 0
            \\end{bmatrix}

        This operator is constantly used at different steps of the EKF.

        Parameters
        ----------
        x : numpy.ndarray
            Three-dimensional vector.

        Returns
        -------
        Omega : numpy.ndarray
            Omega matrix.
        """
        return np.array([
            [0.0,  -x[0], -x[1], -x[2]],
            [x[0],   0.0,  x[2], -x[1]],
            [x[1], -x[2],   0.0,  x[0]],
            [x[2],  x[1], -x[0],   0.0]])

    def f(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Linearized function of Process Model (Prediction.)

        .. math::
            \\mathbf{f}(\\mathbf{q}_{t-1}) = \\Big(\\mathbf{I}_4 + \\frac{\\Delta t}{2}\\boldsymbol\\Omega_t\\Big)\\mathbf{q}_{t-1} =
            \\begin{bmatrix}
            q_w - \\frac{\\Delta t}{2} \\omega_x q_x - \\frac{\\Delta t}{2} \\omega_y q_y - \\frac{\\Delta t}{2} \\omega_z q_z\\\\
            q_x + \\frac{\\Delta t}{2} \\omega_x q_w - \\frac{\\Delta t}{2} \\omega_y q_z + \\frac{\\Delta t}{2} \\omega_z q_y\\\\
            q_y + \\frac{\\Delta t}{2} \\omega_x q_z + \\frac{\\Delta t}{2} \\omega_y q_w - \\frac{\\Delta t}{2} \\omega_z q_x\\\\
            q_z - \\frac{\\Delta t}{2} \\omega_x q_y + \\frac{\\Delta t}{2} \\omega_y q_x + \\frac{\\Delta t}{2} \\omega_z q_w
            \\end{bmatrix}

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        omega : numpy.ndarray
            Angular velocity, in rad/s.

        Returns
        -------
        q : numpy.ndarray
            Linearized estimated quaternion in **Prediction** step.
        """
        Omega_t = self.Omega(omega)
        return (np.identity(4) + 0.5*self.Dt*Omega_t) @ q

    def dfdq(self, omega: np.ndarray) -> np.ndarray:
        """Jacobian of linearized predicted state.

        .. math::
            \\mathbf{F} = \\frac{\\partial\\mathbf{f}(\\mathbf{q}_{t-1})}{\\partial\\mathbf{q}} =
            \\begin{bmatrix}
            1 & - \\frac{\\Delta t}{2} \\omega_x & - \\frac{\\Delta t}{2} \\omega_y & - \\frac{\\Delta t}{2} \\omega_z\\\\
            \\frac{\\Delta t}{2} \\omega_x & 1 & \\frac{\\Delta t}{2} \\omega_z & - \\frac{\\Delta t}{2} \\omega_y\\\\
            \\frac{\\Delta t}{2} \\omega_y & - \\frac{\\Delta t}{2} \\omega_z & 1 & \\frac{\\Delta t}{2} \\omega_x\\\\
            \\frac{\\Delta t}{2} \\omega_z & \\frac{\\Delta t}{2} \\omega_y & - \\frac{\\Delta t}{2} \\omega_x & 1
            \\end{bmatrix}

        Parameters
        ----------
        omega : numpy.ndarray
            Angular velocity in rad/s.

        Returns
        -------
        F : numpy.ndarray
            Jacobian of state.
        """
        x = 0.5*self.Dt*omega
        return np.identity(4) + self.Omega(x)

    def h(self, q: np.ndarray) -> np.ndarray:
        """Measurement Model

        If only the gravitational acceleration is used to correct the
        estimation, a vector with 3 elements is used:

        .. math::
            \\mathbf{h}(\\hat{\\mathbf{q}}_t) =
            2 \\begin{bmatrix}
            g_x (\\frac{1}{2} - q_y^2 - q_z^2) + g_y (q_wq_z + q_xq_y) + g_z (q_xq_z - q_wq_y) \\\\
            g_x (q_xq_y - q_wq_z) + g_y (\\frac{1}{2} - q_x^2 - q_z^2) + g_z (q_wq_x + q_yq_z) \\\\
            g_x (q_wq_y + q_xq_z) + g_y (q_yq_z - q_wq_x) + g_z (\\frac{1}{2} - q_x^2 - q_y^2)
            \\end{bmatrix}

        If the gravitational acceleration and the geomagnetic field are used,
        then a vector with 6 elements is used:

        .. math::
            \\mathbf{h}(\\hat{\\mathbf{q}}_t) =
            2 \\begin{bmatrix}
            g_x (\\frac{1}{2} - q_y^2 - q_z^2) + g_y (q_wq_z + q_xq_y) + g_z (q_xq_z - q_wq_y) \\\\
            g_x (q_xq_y - q_wq_z) + g_y (\\frac{1}{2} - q_x^2 - q_z^2) + g_z (q_wq_x + q_yq_z) \\\\
            g_x (q_wq_y + q_xq_z) + g_y (q_yq_z - q_wq_x) + g_z (\\frac{1}{2} - q_x^2 - q_y^2) \\\\
            r_x (\\frac{1}{2} - q_y^2 - q_z^2) + r_y (q_wq_z + q_xq_y) + r_z (q_xq_z - q_wq_y) \\\\
            r_x (q_xq_y - q_wq_z) + r_y (\\frac{1}{2} - q_x^2 - q_z^2) + r_z (q_wq_x + q_yq_z) \\\\
            r_x (q_wq_y + q_xq_z) + r_y (q_yq_z - q_wq_x) + r_z (\\frac{1}{2} - q_x^2 - q_y^2)
            \\end{bmatrix}

        Parameters
        ----------
        q : numpy.ndarray
            Predicted Quaternion.

        Returns
        -------
        numpy.ndarray
            Expected Measurements.
        """
        C = q2R(q).T
        if len(self.z) < 4:
            return C @ self.a_ref
        return np.r_[C @ self.a_ref, C @ self.m_ref]

    def dhdq(self, q: np.ndarray, mode: str = 'normal') -> np.ndarray:
        """Linearization of observations with Jacobian.

        If only the gravitational acceleration is used to correct the
        estimation, a :math:`3\\times 4` matrix:

        .. math::
            \\mathbf{H}(\\hat{\\mathbf{q}}_t) =
            2 \\begin{bmatrix}
            g_yq_z - g_zq_y & g_yq_y + g_zq_z & - 2g_xq_y + g_yq_x - g_zq_w & - 2g_xq_z + g_yq_w + g_zq_x \\\\
            -g_xq_z + g_zq_x & g_xq_y - 2g_yq_x + g_zq_w & g_xq_x + g_zq_z & -g_xq_w - 2g_yq_z + g_zq_y \\\\
            g_xq_y - g_yq_x & g_xq_z - g_yq_w - 2g_zq_x & g_xq_w + g_yq_z - 2g_zq_y & g_xq_x + g_yq_y
            \\end{bmatrix}

        If the gravitational acceleration and the geomagnetic field are used,
        then a :math:`6\\times 4` matrix is used:

        .. math::
            \\mathbf{H}(\\hat{\\mathbf{q}}_t) =
            2 \\begin{bmatrix}
            g_yq_z - g_zq_y & g_yq_y + g_zq_z & - 2g_xq_y + g_yq_x - g_zq_w & - 2g_xq_z + g_yq_w + g_zq_x \\\\
            -g_xq_z + g_zq_x & g_xq_y - 2g_yq_x + g_zq_w & g_xq_x + g_zq_z & -g_xq_w - 2g_yq_z + g_zq_y \\\\
            g_xq_y - g_yq_x & g_xq_z - g_yq_w - 2g_zq_x & g_xq_w + g_yq_z - 2g_zq_y & g_xq_x + g_yq_y \\\\
            r_yq_z - r_zq_y & r_yq_y + r_zq_z & - 2r_xq_y + r_yq_x - r_zq_w & - 2r_xq_z + r_yq_w + r_zq_x \\\\
            - r_xq_z + r_zq_x & r_xq_y - 2r_yq_x + r_zq_w & r_xq_x + r_zq_z & - r_xq_w - 2r_yq_z + r_zq_y \\\\
            r_xq_y - r_yq_x & r_xq_z - r_yq_w - 2r_zq_x & r_xq_w + r_yq_z - 2r_zq_y & r_xq_x + r_yq_y
            \\end{bmatrix}

        If ``mode`` is equal to ``'refactored'``, the computation is carried
        out as:

        .. math::
            \\mathbf{H}(\\hat{\\mathbf{q}}_t) = 2
            \\begin{bmatrix}
            \\mathbf{u}_g & \\lfloor\\mathbf{u}_g+\\hat{q}_w\\mathbf{g}\\rfloor_\\times + (\\hat{\\mathbf{q}}_v\\cdot\\mathbf{g})\\mathbf{I}_3 - \\mathbf{g}\\hat{\\mathbf{q}}_v^T \\\\
            \\mathbf{u}_r & \\lfloor\\mathbf{u}_r+\\hat{q}_w\\mathbf{r}\\rfloor_\\times + (\\hat{\\mathbf{q}}_v\\cdot\\mathbf{r})\\mathbf{I}_3 - \\mathbf{r}\\hat{\\mathbf{q}}_v^T
            \\end{bmatrix}

        .. warning::
            The refactored mode might lead to slightly different results as it
            employs more and different operations than the normal mode,
            created by the nummerical capabilities of the host system.

        Parameters
        ----------
        q : numpy.ndarray
            Predicted state estimate.
        mode : str, default: 'normal'
            Computation mode for Observation matrix.

        Returns
        -------
        H : numpy.ndarray
            Jacobian of observations.
        """
        if mode.lower() not in ['normal', 'refactored']:
            raise ValueError(f"Mode '{mode}' is invalid. Try 'normal' or 'refactored'.")
        qw, qx, qy, qz = q
        if mode.lower() == 'refactored':
            t = skew(self.a_ref)@q[1:]
            H = np.c_[t, q[1:]*self.a_ref*np.identity(3) + skew(t + qw*self.a_ref) - np.outer(self.a_ref, q[1:])]
            if len(self.z) == 6:
                t = skew(self.m_ref)@q[1:]
                H_2 = np.c_[t, q[1:]*self.m_ref*np.identity(3) + skew(t + qw*self.m_ref) - np.outer(self.m_ref, q[1:])]
                H = np.vstack((H, H_2))
            return 2.0*H
        v = np.r_[self.a_ref, self.m_ref]
        H = np.array([[-qy*v[2] + qz*v[1],  qy*v[1] + qz*v[2], -qw*v[2] + qx*v[1] - 2.0*qy*v[0],  qw*v[1] + qx*v[2] - 2.0*qz*v[0]],
                      [ qx*v[2] - qz*v[0],  qw*v[2] - 2.0*qx*v[1] + qy*v[0],  qx*v[0] + qz*v[2], -qw*v[0] + qy*v[2] - 2.0*qz*v[1]],
                      [-qx*v[1] + qy*v[0], -qw*v[1] - 2.0*qx*v[2] + qz*v[0],  qw*v[0] - 2.0*qy*v[2] + qz*v[1],  qx*v[0] + qy*v[1]]])
        if len(self.z) == 6:
            H_2 = np.array([[-qy*v[5] + qz*v[4],                qy*v[4] + qz*v[5], -qw*v[5] + qx*v[4] - 2.0*qy*v[3],  qw*v[4] + qx*v[5] - 2.0*qz*v[3]],
                            [ qx*v[5] - qz*v[3],  qw*v[5] - 2.0*qx*v[4] + qy*v[3],                qx*v[3] + qz*v[5], -qw*v[3] + qy*v[5] - 2.0*qz*v[4]],
                            [-qx*v[4] + qy*v[3], -qw*v[4] - 2.0*qx*v[5] + qz*v[3],  qw*v[3] - 2.0*qy*v[5] + qz*v[4],  qx*v[3] + qy*v[4]]])
            H = np.vstack((H, H_2))
        return 2.0*H

    def update(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray = None) -> np.ndarray:
        """
        Perform an update of the state.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori orientation as quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2.
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in uT.

        Returns
        -------
        q : numpy.ndarray
            Estimated a-posteriori orientation as quaternion.

        """
        if not np.isclose(np.linalg.norm(q), 1.0):
            raise ValueError("A-priori quaternion must have a norm equal to 1.")
        # Current Measurements
        g = np.copy(gyr)                # Gyroscope data as control vector
        a = np.copy(acc)
        a_norm = np.linalg.norm(a)
        if a_norm == 0:
            return q
        a /= a_norm
        self.z = np.copy(a)
        if mag is not None:
            m_norm = np.linalg.norm(mag)
            if m_norm == 0:
                raise ValueError(f"Invalid geomagnetic field. Its magnitude must be greater than zero.")
            self.z = np.r_[a, mag/m_norm]
        self.R = np.diag(np.repeat(self.noises[1:] if mag is not None else self.noises[1], 3))
        # ----- Prediction -----
        q_t = self.f(q, g)                  # Predicted State
        F   = self.dfdq(g)                  # Linearized Fundamental Matrix
        W   = 0.5*self.Dt * np.r_[[-q[1:]], q[0]*np.identity(3) + skew(q[1:])]  # Jacobian W = df/dω
        Q_t = 0.5*self.Dt * self.g_noise * W@W.T    # Process Noise Covariance
        P_t = F@self.P@F.T + Q_t            # Predicted Covariance Matrix
        # ----- Correction -----
        y   = self.h(q_t)                   # Expected Measurement function
        v   = self.z - y                    # Innovation (Measurement Residual)
        H   = self.dhdq(q_t)                # Linearized Measurement Matrix
        S   = H@P_t@H.T + self.R            # Measurement Prediction Covariance
        K   = P_t@H.T@np.linalg.inv(S)      # Kalman Gain
        self.P = (np.identity(4) - K@H)@P_t
        self.q = q_t + K@v                  # Corrected state
        self.q /= np.linalg.norm(self.q)
        return self.q

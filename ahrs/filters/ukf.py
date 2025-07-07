# -*- coding: utf-8 -*-
"""
.. versionadded:: 0.4.0

The Unscented Kaman Filter (UKF) was first proposed by S. Julier and J. Uhlmann
:cite:p:`julier1997` as an alternative to the Kalman Fiter for nonlinear
systems.

The UKF approximates the mean and covariance of the state distribution using a
set of discretely sampled points, called the **Sigma Points**, obtained through
a deterministic sampling technique called the `Unscented Transform
<https://en.wikipedia.org/wiki/Unscented_transform>`_.

Contrary to the EKF, the UKF does not linearize the models, but uses each of
the sigma points as an input to the state transition and measurement functions
to get a new set of transformed state points, thus avoiding the need for
Jacobians, and yielding an accuracy similar to the KF for linear systems.

The UKF offers significant advantages over the EKF in terms of handling
nonlinearities, achieving higher-order accuracy, robustness to initial
estimates, and consistent performance.

However, the UKF has disadvantages related to computational complexity, memory
requirements, and parameter tuning. These factors can make the UKF less
suitable for certain applications, particularly those with limited
computational resources.

The implementation in this module is based on the UKF algorithm for nonlinear
estimations proposed by Wan and van de Merwe :cite:p:`wan2000`, and further
developed by Kraft :cite:p:`kraft2003` and Klingbeil :cite:p:`klingbeil2006`
for orientation estimation using quaternions.

**Kalman Filter**

We have a `discrete system <https://en.wikipedia.org/wiki/Discrete_system>`_,
whose `states <https://en.wikipedia.org/wiki/State_(computer_science)>`_ are
described by a vector :math:`\\mathbf{x}_t` at each time :math:`t`.

This vector has :math:`n` items, which quantify the position, velocity,
orientation, etc. Basically, anything that can be measured or estimated can be
a state, as long as it can be described numerically.

Knowing how the state was at time :math:`t-1`, we want to predict how the state
is at time :math:`t`. In addition, we also have a set of measurements
:math:`\\mathbf{z}_t`, that can be used to improve the prediction of the state.

The traditional `Kalman filter <https://en.wikipedia.org/wiki/Kalman_filter>`_,
as described by :cite:p:`kalman1960` computes a state in two steps:

1. The **prediction step** computes a guess of the current state,
   :math:`\\hat{\\mathbf{x}}_t`, and its covariance :math:`\\hat{\\mathbf{P}}_t`,
   at time :math:`t`, given the previous state and covariance at time :math:`t-1`.

.. math::
    \\begin{array}{rcl}
    \\hat{\\mathbf{x}}_t &=& \\mathbf{F}\\mathbf{x}_{t-1} + \\mathbf{Bu}_t \\\\
    \\hat{\\mathbf{P}}_t &=& \\mathbf{F}\\mathbf{P}_{t-1}\\mathbf{F}^T + \\mathbf{Q}_t
    \\end{array}

2. The **correction step** improves this prediction using a measurement (or set
   of measurements) :math:`\\mathbf{z}_t` at time :math:`t`.

.. math::
    \\begin{array}{rcl}
    \\mathbf{v}_t &=& \\mathbf{z}_t - \\mathbf{H}\\hat{\\mathbf{x}}_t \\\\
    \\mathbf{S}_t &=& \\mathbf{H} \\hat{\\mathbf{P}}_t \\mathbf{H}^T + \\mathbf{R} \\\\
    \\mathbf{K}_t &=& \\hat{\\mathbf{P}}_t \\mathbf{H}^T \\mathbf{S}_t^{-1} \\\\
    \\mathbf{x}_t &=& \\hat{\\mathbf{x}}_t + \\mathbf{K}_t \\mathbf{v}_t \\\\
    \\mathbf{P}_t &=& \\hat{\\mathbf{P}}_t - \\mathbf{K}_t \\mathbf{S}_t \\mathbf{K}_t^T
    \\end{array}

The Kalman filter, however, is limited to `linear systems
<https://en.wikipedia.org/wiki/Linear_system>`_, rendering the process above
inapplicable to `nonlinear systems <https://en.wikipedia.org/wiki/Nonlinear_system>`_
like our attitude estimation problem.

**Extended Kalman Filter**

A common solution to the nonlinearity issue is the `Extended Kalman Filter
<./ekf.html>`_ (EKF), which linearizes the system model and measurement
functions to __approximate__ the terms, allowing the use of the Kalman filter
as if it were a linear system.

In this approach the predicted mean and covariance are computed using the
linearized models:

.. math::

    \\begin{array}{rcl}
    \\hat{\\mathbf{x}}_t &=& \\mathbf{f}(\\mathbf{x}_{t-1}, \\mathbf{u}_t) \\\\
    \\hat{\\mathbf{P}}_t &=& \\mathbf{F}(\\mathbf{x}_{t-1}, \\mathbf{u}_t)\\mathbf{P}_{t-1}\\mathbf{F}^T(\\mathbf{x}_{t-1}, \\mathbf{u}_t) + \\mathbf{Q}_t
    \\end{array}

where :math:`\\mathbf{f}(\\mathbf{x}_{t-1}, \\mathbf{u}_t)` is the nonlinear
dynamic model function, whose Jacobian is:

.. math::

    \\mathbf{F}(\\mathbf{x}_{t-1}, \\mathbf{u}_t) = \\frac{\\partial \\mathbf{f}(\\mathbf{x}_{t-1}, \\mathbf{u}_t)}{\\partial \\mathbf{x}}

whereas the measurement model [#]_ is linearized as:

.. math::

    \\begin{array}{rcl}
    \\mathbf{v}_t &=& \\mathbf{z}_t - \\mathbf{h}(\\mathbf{x}_t) \\\\
    \\mathbf{S}_t &=& \\mathbf{H}(\\mathbf{x}_t) \\hat{\\mathbf{P}}_t \\mathbf{H}^T(\\mathbf{x}_t) + \\mathbf{R}_t \\\\
    \\mathbf{K}_t &=& \\hat{\\mathbf{P}}_t \\mathbf{H}^T(\\mathbf{x}_t) \\mathbf{S}_t^{-1} \\\\
    \\mathbf{x}_t &=& \\hat{\\mathbf{x}}_t + \\mathbf{K}_t \\mathbf{v}_t \\\\
    \\mathbf{P}_t &=& \\big(\\mathbf{I}_4 - \\mathbf{K}_t\\mathbf{H}(\\mathbf{x}_t)\\big)\\hat{\\mathbf{P}}_t
    \\end{array}

where :math:`\\mathbf{h}(\\mathbf{x}_t)` is the nonlinear measurement model
function, whose Jacobian is:

.. math::

    \\mathbf{H}(\\hat{\\mathbf{x}}_t) = \\frac{\\partial \\mathbf{h}(\\mathbf{x}_t)}{\\partial \\mathbf{x}}

Unfortunately, these approximations can introduce large errors in the posterior
mean and covariance of the transformed random variable, which may lead to
sub-optimal performance.

To avoid these issues, a solution using unscented transforms was proposed by
Julier and Uhlmann :cite:p:`julier1997`, which is the basis for the Unscented
Kalman Filter (UKF).

Unscented Kalman Filter
------------------------

**Unscented Transform**

The UKF is a type of Kalman filter that replaces the linearization with a
deterministic sampling technique called the `Unscented Transform
<https://en.wikipedia.org/wiki/Unscented_transform>`_.

This transformation generates a set of points that capture the mean and
covariance of the state distribution, called the **Sigma Points**.

Each of the sigma points is used as an input to the state transition and
measurement functions to get a new set of transformed state points.

.. epigraph::

   The unscented transformation ... is founded on the intuition that it is
   easier to approximate a Gaussian distribution than it is to approximate an
   arbitrary nonlinear function or transformation.

   -- Jeffrey K. Uhlmann

Imagine there is a set of random points :math:`\\mathbf{x}` with mean
:math:`\\bar{\\mathbf{x}}`, and covariance :math:`\\mathbf{P_{xx}}`, and there
is another set of random points :math:`\\mathbf{y}` related to
:math:`\\mathbf{x}` by a nonlinear function :math:`\\mathbf{y} = f(\\mathbf{x})`.

Our goal is to find the mean :math:`\\bar{\\mathbf{y}}` and covariance
:math:`\\mathbf{P_{yy}}` of :math:`\\mathbf{y}`. The unscented transform
approximates them by sampling a set of points from :math:`\\mathbf{x}` and
applying the nonlinear function :math:`f` to each of the sampled points.

Information about the distribution can be captured using a small number of
points :cite:p:`julier1997`. The samples are not drawn at random but according
to a deterministic method.

**Sigma Points**

The :math:`n`-dimensional random variable :math:`\\mathbf{x}` with mean
:math:`\\bar{\\mathbf{x}}` and covariance :math:`\\mathbf{P_{xx}}` is
approximated by :math:`2n + 1` points computed with:

.. math::

    \\begin{array}{rcl}
    \\mathcal{X}_0 &=& \\bar{\\mathbf{x}} \\\\
    \\mathcal{X}_i &=& \\bar{\\mathbf{x}} + \\Big(\\sqrt{(n + \\lambda)\\mathbf{P_{xx}}}\\Big)_i\\\\
    \\mathcal{X}_{i+n} &=& \\bar{\\mathbf{x}} - \\Big(\\sqrt{(n + \\lambda)\\mathbf{P_{xx}}}\\Big)_i
    \\end{array}

where :math:`(\\sqrt{(n + \\lambda)\\mathbf{P_{xx}}})_i` is the :math:`i`-th
column of the `matrix square root <https://en.wikipedia.org/wiki/Square_root_of_a_matrix>`_,
and :math:`\\lambda=\\alpha^2(n + \\kappa) - n` is a scaling parameter.

:math:`\\alpha` determines the spread of the sigma points around the mean,
usually set to :math:`0.001`, and :math:`\\kappa` is a secondary scaling
parameter, usually set to :math:`0` :cite:p:`wan2000`.

But, how do we obtain the matrix form of the `square root
<https://en.wikipedia.org/wiki/Square_root_of_a_matrix>`_ of
:math:`(n + \\lambda)\\mathbf{P_{xx}}`?

Because :math:`\\mathbf{P_{xx}}` is a covariance matrix, it means it is
symmetric and positive-definite.

The `Cholesky decomposition <https://en.wikipedia.org/wiki/Cholesky_decomposition>`_
of a `positive-definite matrix <https://en.wikipedia.org/wiki/Definite_matrix>`_
:math:`\\mathbf{A}` is a lower triangular matrix :math:`\\mathbf{L}` such that
:math:`\\mathbf{A} = \\mathbf{LL}^T`.

The square root of :math:`(n + \\lambda)\\mathbf{P_{xx}}` is then
:math:`\\mathbf{L}=\\mathrm{chol}((n + \\lambda)\\mathbf{P_{xx}})`.

The Cholesky decomposition is preferred because:

- It efficiently computes (roughly :math:`\\frac{n^3}{3}` operations for an
  :math:`n\\times n` matrix) the lower triangular matrix :math:`\\mathbf{L}`.
- It's numerically stable.
- It naturally handles the positive-definiteness requirement of covariance
  matrices.

Therefore, we first calculate the Cholesky decomposition of
:math:`(n + \\lambda)\\mathbf{P_{xx}}`, and then we obtain the Sigma Points by
adding and subtracting the columns of :math:`\\mathbf{L}` to the mean.

.. math::
   :label: sigma_points

    \\begin{array}{rcl}
    \\mathcal{X}_0 &=& \\bar{\\mathbf{x}} \\\\
    \\mathcal{X}_i &=& \\bar{\\mathbf{x}} + \\mathbf{L}_i \\\\
    \\mathcal{X}_{i+n} &=& \\bar{\\mathbf{x}} - \\mathbf{L}_i
    \\end{array}

where :math:`\\mathbf{L}` is the Cholesky decomposition of
:math:`(n + \\lambda)\\mathbf{P_{xx}}`, as mentioned above.

We pass these sigma points through the nonlinear function :math:`f` to get the
transformed points :math:`\\mathcal{Y}`.

.. math::
   :label: ukf_process_model

    \\mathcal{Y} = f(\\mathcal{X})

Their **mean** is given by their wieghted sum:

.. math::
   :label: ukf_predicted_state_mean

    \\bar{\\mathbf{y}} = \\sum_{i=0}^{2n} w_i^{(m)} \\mathcal{Y}_i

And their **covariance** by their weighted outer product:

.. math::
   :label: ukf_predicted_state_covariance

    \\mathbf{P_{yy}} = \\sum_{i=0}^{2n} w_i^{(c)} (\\mathcal{Y}_i - \\bar{\\mathbf{y}})(\\mathcal{Y}_i - \\bar{\\mathbf{y}})^T + \\mathbf{Q}

with :math:`\\mathbf{Q}` being the :math:`n\\times n` process noise covariance
matrix.

The weights :math:`\\mathbf{w}` are computed as:

.. math::

    \\begin{array}{rcl}
    w_0^{(m)} &=& \\frac{\\lambda}{n + \\lambda} \\\\
    w_0^{(c)} &=& \\frac{\\lambda}{n + \\lambda} + (1 - \\alpha^2 + \\beta) \\\\
    w_i^{(m)} = w_i^{(c)} &=& \\frac{1}{2(n + \\lambda)} \\quad \\text{for} \\quad i=1,2,\\ldots,2n
    \\end{array}

The weights :math:`\\mathbf{w}^{(m)}` are used to compute the mean, and the
weights :math:`\\mathbf{w}^{(c)}` are used to compute the covariance.

The constant :math:`\\beta` is used to incorporate prior knowledge about the
distribution of the random variable, and is usually set to :math:`2` for
Gaussian distributions :cite:p:`wan2000`.

**UKF Summary**

Given the previous state :math:`\\mathbf{x}_{t-1}`, its covariance matrix
:math:`\\mathbf{P}_{t-1}`, a control vector :math:`\\mathbf{u}_t`, and a vector
with the most recent :math:`m` measurements :math:`\\mathbf{z}_t`, the UKF
algorithm can be summarized as follows:

**Prediction**:

1. Compute Sigma Points.

.. math::

    \\mathcal{X}(\\mathbf{x}_{t-1}, \\mathbf{P}_{t-1}) = \\Big\\{ \\mathcal{X}_0 \\; , \\quad\\mathcal{X}_i \\; , \\quad\\mathcal{X}_{i+n} \\Big\\}

2. Propagate Sigma Points through Process model to get Transformed Sigma Points.

.. math::

    \\mathcal{Y} = f(\\mathcal{X})

3. Predict State Mean and Covariance.

.. math::

    \\begin{array}{rcl}
    \\bar{\\mathbf{y}} &=& \\sum_{i=0}^{2n} w_i^{(m)} \\mathcal{Y}_i \\\\ \\\\
    \\mathbf{P}_{yy} &=& \\sum_{i=0}^{2n} w_i^{(c)} (\\mathcal{Y}_i - \\bar{\\mathbf{y}})(\\mathcal{Y}_i - \\bar{\\mathbf{y}})^T + \\mathbf{Q}
    \\end{array}

**Correction**:

4. Instantiate each Transformed Sigma Point through Measurement model.

.. math::

    \\mathcal{Z} = h(\\mathcal{Y})

5. Compute Predicted Measurement Mean and Covariance.

.. math::

    \\begin{array}{rcl}
    \\bar{\\mathbf{z}} &=& \\sum_{i=0}^{2n} w_i^{(m)} \\mathcal{Z}_i \\\\ \\\\
    \\mathbf{P}_{zz} &=& \\sum_{i=0}^{2n} w_i^{(c)} (\\mathcal{Z}_i - \\bar{\\mathbf{z}})(\\mathcal{Z}_i - \\bar{\\mathbf{z}})^T + \\mathbf{R}
    \\end{array}

6. Compute Cross-Covariance.

.. math::

    \\mathbf{P}_{yz} = \\sum_{i=0}^{2n} w_i^{(c)} (\\mathcal{Y}_i - \\bar{\\mathbf{y}})(\\mathcal{Z}_i - \\bar{\\mathbf{z}})^T

7. Compute Kalman gain.

.. math::

    \\mathbf{K} = \\mathbf{P}_{yz} \\mathbf{P}_{zz}^{-1}

8. Compute Innovation (residual.)

.. math::

    \\mathbf{v}_t = \\mathbf{z}_t - \\bar{\\mathbf{z}}

9. Update (Correct) State and Covariance

.. math::

    \\begin{array}{rcl}
    \\mathbf{x}_t &=& \\bar{\\mathbf{y}} + \\mathbf{K} \\mathbf{v}_t \\\\ \\\\
    \\mathbf{P}_t &=& \\mathbf{P}_{yy} - \\mathbf{K} \\mathbf{P}_{zz} \\mathbf{K}^T
    \\end{array}

UKF for Attitude Estimation
---------------------------

In this implementation, we build the simplest UKF for attitude estimation using
gyroscopes and accelerometers (and magnetometer, if available), so that we can
focus on the details of the algorithm. Once the basic structure is understood,
we could extend the model to create more complex systems.

**PREDICTION MODEL**

We start by defining the vectors of the Prediction Model, a.k.a. **Process
Model**:

.. math::

    \\begin{array}{rcl}
    \\mathbf{x} &=& \\begin{bmatrix} q_w & q_x & q_y & q_z \\end{bmatrix}^T \\\\ \\\\
    \\mathbf{u} &=& \\begin{bmatrix} \\omega_x & \\omega_y & \\omega_z \\end{bmatrix}^T
    \\end{array}

The state vector :math:`\\mathbf{x}_t\\in\\mathbb{R}^4` has the elements of a
quaternion representing the orientation at any time :math:`t`.

We have added the **control vector** :math:`\\mathbf{u}_t\\in\\mathbb{R}^3`
containing the angular velocity readings from a tri-axial gyroscope. This
control vector is used to propagate the state vector through the process model.
Normally, it is ignored when the gyroscope is not available or when we assume
the model is not affected by external forces, but this is not the case in our
implementation.

The quaternion describing the orientation is also known as a `versor
<https://en.wikipedia.org/wiki/Versor>`_, and it is a unit quaternion, meaning
:math:`\\|\\mathbf{q}\\|=1`. Thus, we normalize the quaternion after each
transformation.

.. note::

    Notice we don't extend the state vector to include the gyroscope biases
    like other models do. For the sake of simplicity we don't estimate these
    biases, and assume the sensor readings are already calibrated.

**Sigma Points**

Given the previous state and covariance, the sigma points are computed first.

Using the cholesky decomposition we obtain the **matrix square root**:

.. math::

    \\mathbf{L} = \\mathrm{chol}\\Big(\\sqrt{(n + \\lambda)\\mathbf{P_{xx}}}\\Big)

where the previous covariance is set as :math:`\\mathbf{P_{xx}}=\\mathbf{P}_{t-1}`,
:math:`n=4` is the number of items in the state vector :math:`\\mathbf{x}`, and
:math:`\\lambda=\\alpha^2(n + \\kappa) - n` is the scaling parameter.

.. hint::

    Using the default values :math:`\\alpha=0.001`, and :math:`\\kappa=0`, we
    get:

    .. math::

        \\lambda = 0.001^2 (4 + 0) - 4 = -3.999996

    which yields :math:`\\mathbf{L} = \\mathrm{chol}\\big(\\sqrt{0.000004\\mathbf{P}_{t-1}}\\big)`.

Then, we compute the sigma points using the equations:

.. math::

    \\begin{array}{rcl}
    \\mathcal{X}_0 &=& \\mathbf{x}_{t-1} \\\\
    \\mathcal{X}_i &=& \\mathbf{x}_{t-1} + \\mathbf{L}_i \\\\
    \\mathcal{X}_{i+n} &=& \\mathbf{x}_{t-1} - \\mathbf{L}_i
    \\end{array}

The first sigma point :math:`\\mathcal{X}_0` is always equal to the previous
state :math:`\\mathbf{x}_{t-1}`. The rest are obtained by adding and
subtracting the columns of :math:`\\mathbf{L}` to the mean.

Because the state vector has 4 items, we obtain a set of 9 sigma points:

.. math::

    \\begin{array}{rcl}
    \\mathcal{X} &=&
    \\begin{Bmatrix}
        \\big| & \\big| & \\big| & \\big| & \\big| & \\big| & \\big| & \\big| & \\big| \\\\
        \\mathcal{X}_0 & \\mathcal{X}_1 & \\mathcal{X}_2 & \\mathcal{X}_3 & \\mathcal{X}_4 & \\mathcal{X}_5 & \\mathcal{X}_6 & \\mathcal{X}_7 & \\mathcal{X}_8 \\\\
        \\big| & \\big| & \\big| & \\big| & \\big| & \\big| & \\big| & \\big| & \\big|
    \\end{Bmatrix} \\\\ \\\\
    &=&
    \\begin{Bmatrix}
        q_w & q_w + \\mathbf{L}_{1,1} & q_w + \\mathbf{L}_{1,2} & q_w + \\mathbf{L}_{1,3} & q_w + \\mathbf{L}_{1,4} & q_w - \\mathbf{L}_{1,1} & q_w - \\mathbf{L}_{1,2} & q_w - \\mathbf{L}_{1,3} & q_w - \\mathbf{L}_{1,4} \\\\
        q_x & q_x + \\mathbf{L}_{2,1} & q_x + \\mathbf{L}_{2,2} & q_x + \\mathbf{L}_{2,3} & q_x + \\mathbf{L}_{2,4} & q_x - \\mathbf{L}_{2,1} & q_x - \\mathbf{L}_{2,2} & q_x - \\mathbf{L}_{2,3} & q_x - \\mathbf{L}_{2,4} \\\\
        q_y & q_y + \\mathbf{L}_{3,1} & q_y + \\mathbf{L}_{3,2} & q_y + \\mathbf{L}_{3,3} & q_y + \\mathbf{L}_{3,4} & q_y - \\mathbf{L}_{3,1} & q_y - \\mathbf{L}_{3,2} & q_y - \\mathbf{L}_{3,3} & q_y - \\mathbf{L}_{3,4} \\\\
        q_z & q_z + \\mathbf{L}_{4,1} & q_z + \\mathbf{L}_{4,2} & q_z + \\mathbf{L}_{4,3} & q_z + \\mathbf{L}_{4,4} & q_z - \\mathbf{L}_{4,1} & q_z - \\mathbf{L}_{4,2} & q_z - \\mathbf{L}_{4,3} & q_z - \\mathbf{L}_{4,4}
    \\end{Bmatrix}
    \\end{array}

The estimation process is done as a two-step filter consisting of an `attitude
propagation <./angular.html>`_ (using the gyroscope) and a correction (using
the accelerometer.)

**Attitude Propagation**

Based on the time spent between :math:`t-1` and :math:`t` (known as
the time step :math:`\\Delta t`) we can compute the angular displacement
:math:`\\boldsymbol\\Delta_\\theta` using the measured angular velocities
:math:`\\boldsymbol\\omega` from the gyroscopes, and add it to the previous
attitude :math:`\\mathbf{x}_{t-1}` to obtain the predicted attitude
:math:`\\hat{\\mathbf{x}}_t`:

.. math::

    \\begin{array}{rcl}
    \\hat{\\mathbf{x}}_t &=& \\mathbf{x}_{t-1} + \\boldsymbol\\Delta_\\theta \\\\
    &=& \\mathbf{x}_{t-1} + \\int_{t-1}^t\\boldsymbol\\omega\\, dt
    \\end{array}

This is called **attitude propagation**. However, this is a nonlinear operation
and we cannot use it in the Kalman Filter. Therefore, we approximate it to
define our required linear **Process Model**:

.. math::

    \\begin{array}{rcl}
    \\hat{\\mathbf{x}}_t &=& f(\\mathbf{x}_{t-1}, \\boldsymbol\\omega_t) \\\\
    &=&\\Big[\\mathbf{I}_4 + \\frac{\\Delta t}{2}\\boldsymbol\\Omega_t(\\boldsymbol\\omega_t)\\Big]\\mathbf{x}_{t-1} \\\\
    &=&
    \\begin{bmatrix}
    1 & -\\frac{\\Delta t}{2}\\omega_x & -\\frac{\\Delta t}{2}\\omega_y & -\\frac{\\Delta t}{2}\\omega_z \\\\
    \\frac{\\Delta t}{2}\\omega_x & 1 & \\frac{\\Delta t}{2}\\omega_z & -\\frac{\\Delta t}{2}\\omega_y \\\\
    \\frac{\\Delta t}{2}\\omega_y & -\\frac{\\Delta t}{2}\\omega_z & 1 & \\frac{\\Delta t}{2}\\omega_x \\\\
    \\frac{\\Delta t}{2}\\omega_z & \\frac{\\Delta t}{2}\\omega_y & -\\frac{\\Delta t}{2}\\omega_x & 1
    \\end{bmatrix}
    \\begin{bmatrix}
        q_w \\\\ q_x \\\\ q_y \\\\ q_z
    \\end{bmatrix} \\\\
    \\begin{bmatrix}\\hat{q_w} \\\\ \\hat{q_x} \\\\ \\hat{q_y} \\\\ \\hat{q_z}\\end{bmatrix}
    &=&
    \\begin{bmatrix}
        q_w - \\frac{\\Delta t}{2} \\omega_x q_x - \\frac{\\Delta t}{2} \\omega_y q_y - \\frac{\\Delta t}{2} \\omega_z q_z\\\\
        q_x + \\frac{\\Delta t}{2} \\omega_x q_w - \\frac{\\Delta t}{2} \\omega_y q_z + \\frac{\\Delta t}{2} \\omega_z q_y\\\\
        q_y + \\frac{\\Delta t}{2} \\omega_x q_z + \\frac{\\Delta t}{2} \\omega_y q_w - \\frac{\\Delta t}{2} \\omega_z q_x\\\\
        q_z - \\frac{\\Delta t}{2} \\omega_x q_y + \\frac{\\Delta t}{2} \\omega_y q_x + \\frac{\\Delta t}{2} \\omega_z q_w
    \\end{bmatrix}
    \\end{array}

where the rotation operator :math:`\\big[\\mathbf{I}_4 + \\frac{\\Delta t}{2}
\\boldsymbol\\Omega_t(\\boldsymbol\\omega_t)\\big]` is a truncation up to the
second term of the Taylor series expansion of
:math:`\\int_{t-1}^t\\boldsymbol\\omega\\, dt`.

We assume in this description, that the time step :math:`\\Delta t` is constant.
However, it can be changed at any time during the implementation.

.. note::

    For more details about this linear operation, please refer to the
    documentation of the `Attitude from Angular Rate <./angular.html>`_.

We propagate each of the sigma points through the process model
:math:`f` to get a new set of transformed state points :math:`\\mathcal{Y}`:

.. math::

    \\mathcal{Y} =
    \\begin{Bmatrix}
        \\big| & \\big| & & \\big| \\\\
        f(\\mathcal{X}_0, \\boldsymbol\\omega_t) &
        f(\\mathcal{X}_1, \\boldsymbol\\omega_t) &
        \\cdots &
        f(\\mathcal{X}_8, \\boldsymbol\\omega_t) \\\\
        \\big| & \\big| & & \\big|
    \\end{Bmatrix}

Every :math:`\\mathcal{Y}_i` describes a quaternion. They must be normalized after the transformation, so that :math:`\\forall i \\in
\\{0, \\ldots, 2n\\} \\;, \\|\\mathcal{Y}_i\\|=1`.

Now we compute the **Predicted State Mean**.

.. math::

    \\boxed{\\bar{\\mathbf{y}} = \\sum_{i=0}^{2n} w_i^{(m)} \\mathcal{Y}_i}

This predicted state represents the mean of the transformed state points as a
quaternion. Therefore, we must normalize it:

.. math::

    \\bar{\\mathbf{y}} \\leftarrow \\frac{\\bar{\\mathbf{y}}}{\\|\\bar{\\mathbf{y}}\\|}

We proceed to compute the **Predicted State Covariance**.

.. math::

    \\boxed{\\mathbf{P}_{yy} = \\sum_{i=0}^{2n} w_i^{(c)} (\\mathcal{Y}_i - \\bar{\\mathbf{y}})(\\mathcal{Y}_i - \\bar{\\mathbf{y}})^T + \\mathbf{Q}}

Notice the product :math:`(\\mathcal{Y}_i - \\bar{\\mathbf{y}})(\\mathcal{Y}_i
- \\bar{\\mathbf{y}})^T` is the `outer product
<https://en.wikipedia.org/wiki/Outer_product>`_ of the vector
:math:`(\\mathcal{Y}_i - \\bar{\\mathbf{y}})`, which results in a
:math:`4\\times 4` matrix.

**MEASUREMENT MODEL**

Our strategy is to use Earth's known physical reference values, rotate them
around the set of predicted orientations :math:`\\mathcal{Y}`, and compare
these against actual sensor readings :math:`\\mathbf{z}`.

Their difference tells us how "off" the predicted orientation is. To ease the
comparison both the references and the sensor readings are `normalized
<https://en.wikipedia.org/wiki/Unit_vector>`_.

The two main physical references are commonly used in attitude estimation, and
we will use them in our **Measurement Model**:

- Earth's gravitational vector :math:`\\mathbf{g}`.
- Earth's geomagnetic vector :math:`\\mathbf{r}`.

In order to perform the rotations, we use the `direction cosine matrix
<../dcm.html>`_, a.k.a. rotation matrix, built **from each predicted
orientation** (transformed points) to rotate the reference vectors to the
sensor frame.

For `Earth's gravitational vector <https://en.wikipedia.org/wiki/Gravity_of_Earth#Direction>`_,
we set the normalized reference on `local tangent plane coordinates
<https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates>`_ (LTP) as:

.. math::

    \\mathbf{g} = \\begin{bmatrix} 0 \\\\ 0 \\\\ 1 \\end{bmatrix}

We assume Earth's gravitational vector is aligned with the Z-axis of the LTP
and, therefore, the X- and Y-axes are equal to zero.

`Earth's Magnetic Field <https://en.wikipedia.org/wiki/Earth%27s_magnetic_field>`_
is not aligned with a particular axis of `Earth's ellipsoid
<https://en.wikipedia.org/wiki/Earth_ellipsoid>`_, and it is fully described
with:

.. math::

    \\mathbf{r} = \\begin{bmatrix} r_x \\\\ r_y \\\\ r_z \\end{bmatrix}

where :math:`r_x`, :math:`r_y`, and :math:`r_z` are the components of the
reference geomagnetic vector, which can be obtained from the `World Magnetic
Model <https://en.wikipedia.org/wiki/World_Magnetic_Model>`_ (WMM)
based on the sensor's geographical location.

This vector represents the direction and intensity of the Earth's magnetic
field at the sensor's location.

The reference geomagnetic vector :math:`\\mathbf{r}` is also normalized to
unit length, so that we can use it as a direction vector too:

.. math::

    \\mathbf{r} = \\frac{\\mathbf{r}}{\\|\\mathbf{r}\\|} = \\begin{bmatrix} r_x \\\\ r_y \\\\ r_z \\end{bmatrix}

**Sensor Readings**

If only a tri-axial accelerometer is available to correct (update) the
predicted attitude, the measurement vector :math:`\\mathbf{z}_t` has normalized
accelerometer readings only.

.. math::

    \\mathbf{z} = \\mathbf{a} = \\begin{bmatrix} a_x \\\\ a_y \\\\ a_z \\end{bmatrix}

However, if both tri-axial accelerometer and magnetometer are available, the
measurement vector :math:`\\mathbf{z}_t` includes both:

.. math::

    \\mathbf{z} = \\begin{bmatrix} a_x \\\\ a_y \\\\ a_z \\\\ m_x \\\\ m_y \\\\ m_z \\end{bmatrix}

Given one predicted orientation :math:`\\mathcal{Y}_i` we rotate a reference
vector :math:`\\mathbf{g}` or :math:`\\mathbf{r}` to get its corresponding
expected sensor reading. We call this the **Measurement Model Function**.

If only the accelerometer is available, the measurement model function yields:

.. math::

    \\begin{array}{rcl}
    h(\\hat{\\mathbf{x}}) &=& \\mathbf{R}(\\hat{\\mathbf{x}})^T \\mathbf{g} \\\\
    &=& \\begin{bmatrix}
        1 - 2(\\hat{q_y}^2 + \\hat{q_z}^2) & 2(\\hat{q_x} \\hat{q_y} + \\hat{q_w} \\hat{q_z}) & 2(\\hat{q_x} \\hat{q_z} - \\hat{q_w} \\hat{q_y}) \\\\
        2(\\hat{q_x} \\hat{q_y} - \\hat{q_w} \\hat{q_z}) & 1 - 2(\\hat{q_x}^2 + \\hat{q_z}^2) & 2(\\hat{q_y} \\hat{q_z} + \\hat{q_w} \\hat{q_x}) \\\\
        2(\\hat{q_x} \\hat{q_z} + \\hat{q_w} \\hat{q_y}) & 2(\\hat{q_y} \\hat{q_z} - \\hat{q_w} \\hat{q_x}) & 1 - 2(\\hat{q_x}^2 + \\hat{q_y}^2)
    \\end{bmatrix}
    \\begin{bmatrix} 0 \\\\ 0 \\\\ 1 \\end{bmatrix} \\\\
    &=& \\begin{bmatrix}
        2(\\hat{q_x} \\hat{q_z} - \\hat{q_w} \\hat{q_y}) \\\\
        2(\\hat{q_y} \\hat{q_z} + \\hat{q_w} \\hat{q_x}) \\\\
        1 - 2(\\hat{q_x}^2 + \\hat{q_y}^2)
    \\end{bmatrix}
    \\end{array}

If both accelerometer and magnetometer are available, it is stacked as:

.. math::

    \\begin{array}{rcl}
    h(\\hat{\\mathbf{x}}) &=& \\begin{bmatrix}
        \\mathbf{R}(\\hat{\\mathbf{x}})^T \\mathbf{g} \\\\
        \\mathbf{R}(\\hat{\\mathbf{x}})^T \\ \\mathbf{r}
    \\end{bmatrix} \\\\
    &=& \\begin{bmatrix}
        2(\\hat{q_x} \\hat{q_z} - \\hat{q_w} \\hat{q_y}) \\\\
        2(\\hat{q_y} \\hat{q_z} + \\hat{q_w} \\hat{q_x}) \\\\
        1 - 2(\\hat{q_x}^2 + \\hat{q_y}^2) \\\\
        r_x\\big(1 - 2(\\hat{q_y}^2 + \\hat{q_z}^2)\\big) + 2r_y(\\hat{q_x} \\hat{q_y} + \\hat{q_w} \\hat{q_z}) + 2r_z(\\hat{q_x} \\hat{q_z} - \\hat{q_w} \\hat{q_y}) \\\\
        2r_x(\\hat{q_x} \\hat{q_y} - \\hat{q_w} \\hat{q_z}) + r_y\\big(1 - 2(\\hat{q_x}^2 + \\hat{q_z}^2)\\big) + 2r_z(\\hat{q_y} \\hat{q_z} + \\hat{q_w} \\hat{q_x}) \\\\
        2r_x(\\hat{q_x} \\hat{q_z} + \\hat{q_w} \\hat{q_y}) + 2r_y(\\hat{q_y} \\hat{q_z} - \\hat{q_w} \\hat{q_x}) + r_z\\big(1 - 2(\\hat{q_x}^2 + \\hat{q_y}^2)\\big)
    \\end{bmatrix}
    \\end{array}

.. note::

    Notice we use the transpose of the rotation matrix to rotate the gravity
    vector from the global frame to the sensor frame (the opposite of what it
    describes.) We do this, so that we can compare it against the accelerometer
    readings in sensor frame.

We execute the measurement model function over each of the predicted sigma
points :math:`\\mathcal{Y}` to get the expected accelerometer readings
:math:`\\mathcal{Z}`:

.. math::

    \\mathcal{Z} =
    \\begin{Bmatrix}
        \\big| & \\big| & & \\big| \\\\
        h(\\mathcal{Y}_0) &
        h(\\mathcal{Y}_1) &
        \\cdots &
        h(\\mathcal{Y}_8) \\\\
        \\big| & \\big| & & \\big|
    \\end{Bmatrix}

With this set of expected accelerometer readings :math:`\\mathcal{Z}` we can
compute the **Measurement Mean**:

.. math::

    \\boxed{\\bar{\\mathbf{z}} = \\sum_{i=0}^{2n} w_i^{(m)} \\mathcal{Z}_i}

And the **Measurement Covariance**:

.. math::

    \\boxed{\\mathbf{P}_{zz} = \\sum_{i=0}^{2n} w_i^{(c)} (\\mathcal{Z}_i - \\bar{\\mathbf{z}})(\\mathcal{Z}_i - \\bar{\\mathbf{z}})^T + \\mathbf{R}}

The **Measurement Noise Covariance Matrix** :math:`\\mathbf{R}` is built based
on whether we have accelerometer readings only, or both accelerometer and
magnetometer readings.

.. math::

    \\mathbf{R} &=&
    \\left\\{
    \\begin{array}{ll}
        \\begin{bmatrix}
            \\sigma_{a_x}^2 & 0 & 0 \\\\
            0 & \\sigma_{a_y}^2 & 0 \\\\
            0 & 0 & \\sigma_{a_z}^2
        \\end{bmatrix}
        & \\mathrm{if}\\; \\mathrm{Acc} \\\\ \\\\
        \\begin{bmatrix}
            \\sigma_{a_x}^2 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & \\sigma_{a_y}^2 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & \\sigma_{a_z}^2 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & \\sigma_{m_x}^2 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & \\sigma_{m_y}^2 & 0 \\\\
            0 & 0 & 0 & 0 & 0 & \\sigma_{m_z}^2
        \\end{bmatrix}
        & \\mathrm{if}\\; \\mathrm{Acc+Mag}
    \\end{array}
    \\right.\\\\

where :math:`\\sigma_{a_x}`, :math:`\\sigma_{a_y}`, and :math:`\\sigma_{a_z}`
are the `standard deviations <https://en.wikipedia.org/wiki/Standard_deviation>`_
of the accelerometer readings, and :math:`\\sigma_{m_x}`, :math:`\\sigma_{m_y}`,
and :math:`\\sigma_{m_z}` are the standard deviations of the magnetometer
readings.

The **Cross-Covariance Matrix** :math:`\\mathbf{P}_{yz}` represents how changes
in the state variables correlate with changes in the measurement variables.
Specifically, it quantifies how errors in the predicted states are related to
errors in the expected measurements.

.. math::

    \\boxed{\\mathbf{P}_{yz} = \\sum_{i=0}^{2n} w_i^{(c)} (\\mathcal{Y}_i - \\bar{\\mathbf{y}})(\\mathcal{Z}_i - \\bar{\\mathbf{z}})^T}

With these matrices we compute the **Kalman Gain**.

.. math::

    \\boxed{\\mathbf{K} = \\mathbf{P}_{yz} \\mathbf{P}_{zz}^{-1}}

Notice this is a much simpler operation than the Extended Kalman Filter (EKF)
where we need to compute the Jacobian matrix.

We compare the expected measurement mean :math:`\\bar{\\mathbf{z}}` against
the actual measurement reading :math:`\\mathbf{z}` to get the **innovation** at
the curent time step :math:`t`:

.. math::

    \\boxed{\\mathbf{v}_t = \\mathbf{z}_t - \\bar{\\mathbf{z}}_t}

This tells us, basically, how "far off" the predicted state is from the actual
measurements.

Finally, we use all this information to correct the state and covariance:

.. math::

    \\boxed{
    \\begin{array}{rcl}
    \\mathbf{x}_t &=& \\bar{\\mathbf{y}} + \\mathbf{K} \\mathbf{v}_t \\\\ \\\\
    \\mathbf{P}_t &=& \\mathbf{P}_{yy} - \\mathbf{K} \\mathbf{P}_{zz} \\mathbf{K}^T
    \\end{array}}

Footnotes
---------
.. [#] The Measurement Model is sometimes called **Observation Model**.

.. seealso::

   - `EKF <./ekf.html>`_ - Extended Kalman Filter for orientation estimation.
   - `AngularRate <./angular.html>`_ - Attitude propagation using angular rate.

"""

import numpy as np
from ..common.quaternion import Quaternion
from ..common.orientation import ecompass
from ..common.orientation import acc2q
from ..utils.core import _assert_numerical_iterable
# Local magnetic reference of Munich, Germany
from ..common.constants import MUNICH_LATITUDE, MUNICH_LONGITUDE, MUNICH_HEIGHT
from ..utils.wmm import WMM

class UKF:
    """
    Unscented Kalman Filter to estimate orientation as Quaternion.

    Examples
    --------
    >>> import numpy as np
    >>> from ahrs.filters import UKF
    >>> from ahrs.common.orientation import acc2q
    >>> ukf = UKF()
    >>> num_samples = 1000              # Assuming sensors have 1000 samples each
    >>> Q = np.zeros((num_samples, 4))  # Allocate array for quaternions
    >>> Q[0] = acc2q(acc_data[0])       # First sample of tri-axial accelerometer
    >>> for t in range(1, num_samples):
    ...     Q[t] = ukf.update(Q[t-1], gyr_data[t], acc_data[t])

    The estimation can be simplified by giving all sensor values at the
    construction of the UKF object.

    >>> ukf = UKF(gyr=gyr_data, acc=acc_data)
    >>> ukf.Q.shape
    (1000, 4)

    This will perform all steps above and store the estimated orientations, as
    quaternions, in the attribute ``Q``.

    The most common sampling frequency is 100 Hz, which is used in the filter.
    If the sampling frequency is different in the given sensor data, it can be
    changed too.

    >>> ukf = UKF(gyr=gyr_data, acc=acc_data, frequency=200.0)  # Sampling frequency is 200 Hz

    The initial quaternion is estimated with the first observations of the
    tri-axial accelerometers, but it can also be given directly in the
    parameter ``q0``.

    >>> ukf = UKF(gyr=gyr_data, acc=acc_data, mag=mag_data, q0=[0.7071, 0.0, -0.7071, 0.0])

    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    frequency : float, default: 100.0
        Sampling frequency in Herz.
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. NOT required
        if ``frequency`` value is given.
    alpha : float, default: 1e-3
        Parameter controlling spread of Sigma points.
    beta : float, default: 2
        Parameter controlling distribution of Sigma points.
    kappa : float, default: 0
        Secondary scaling parameter. Usually set to 0.

    """
    def __init__(self,
            gyr: np.ndarray = None,
            acc: np.ndarray = None,
            mag: np.ndarray = None,
            frequency: float = 100.0,
            alpha: float = 1e-3,
            beta: float = 2,
            kappa: float = 0,
            **kwargs):
        self.gyr: np.ndarray = gyr
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.frequency: float = frequency
        self.Dt: float = kwargs.get('Dt', (1.0/self.frequency) if self.frequency else 0.01)
        self.q0: np.ndarray = kwargs.get('q0')
        # UKF parameters
        self.state_dimension: int = 4      # n : State dimension (Quaternion items)
        self.sigma_point_count: int = 2 * self.state_dimension + 1   # 2*n+1 sigma points
        self.alpha: float = alpha          # Spread parameter
        self.beta: float = beta            # Distribution parameter
        self.kappa: float = kappa          # Secondary scaling parameter
        # Lambda parameter: λ = α²(n + κ) - n
        self.lambda_param: float = self.alpha**2 * (self.state_dimension + self.kappa) - self.state_dimension
        # Weights for sigma points
        self.weight_mean, self.weight_covariance = self.set_weights()
        # Process and measurement noise covariances
        self.P: np.ndarray = kwargs.get('P', np.eye(self.state_dimension) * 0.01)    # Initial state covariance
        self.Q_t : np.ndarray = kwargs.get('process_noise_covariance', np.eye(4) * 0.0001)
        self.R : np.ndarray = kwargs.get('measurement_noise_covariance', np.eye(3) * 0.01)

        # Reference gravitational acceleration
        self.a_ref: np.ndarray = kwargs.get('a_ref', np.array([0., 0., 1.]))
        # Reference magnetic field vector
        wmm = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT)
        self.m_ref = np.array([wmm.X, wmm.Y, wmm.Z])
        self.m_ref /= np.linalg.norm(self.m_ref)  # Normalize magnetic reference
        # Sensor data is given. Compute all
        if self.gyr is not None and self.acc is not None:
            self.Q: np.ndarray = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        # Assert input types and values
        _assert_numerical_iterable(self.gyr, 'Angular velocity vector')
        _assert_numerical_iterable(self.acc, 'Gravitational acceleration vector')
        if self.mag is not None:
            _assert_numerical_iterable(self.mag, 'Magnetic field vector')
        self.gyr = np.array(self.gyr)
        self.acc = np.array(self.acc)
        if self.mag is not None:
            self.mag = np.array(self.mag)
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        if self.mag is not None and self.acc.shape != self.mag.shape:
            raise ValueError("acc and mag are not the same size")
        num_samples = len(self.acc)

        # Normalize sensor data
        self.acc = self.acc / np.linalg.norm(self.acc, axis=1, keepdims=True)
        if self.mag is not None:
            self.mag = self.mag / np.linalg.norm(self.mag, axis=1, keepdims=True)

        # Loop over all data
        Q = np.zeros((num_samples, 4))
        if self.mag is not None:
            # Estimation with MARG
            Q[0] = ecompass(self.acc[0], self.mag[0], frame='NED', representation='quaternion')
            for t in range(1, num_samples):
                Q[t] = self.update(q=Q[t-1], gyr=self.gyr[t], acc=self.acc[t], mag=self.mag[t])
        else:
            # Estimation with IMU
            Q[0] = acc2q(self.acc[0]) if self.q0 is None else self.q0
            Q[0] /= np.linalg.norm(Q[0])
            for t in range(1, num_samples):
                Q[t] = self.update(q=Q[t-1], gyr=self.gyr[t], acc=self.acc[t])
        return Q

    def set_weights(self) -> tuple:
        """
        Set weights for mean and covariance computation

        The weights :math:`\\mathbf{w}=\\begin{bmatrix}\\mathbf{w}^{(m)}&
        \\mathbf{w}^{(c)}\\end{bmatrix}` are computed as:

        .. math::

            \\begin{array}{rcl}
            w_0^{(m)} &=& \\frac{\\lambda}{n + \\lambda} \\\\
            w_0^{(c)} &=& \\frac{\\lambda}{n + \\lambda} + (1 - \\alpha^2 + \\beta) \\\\
            w_i^{(m)} = w_i^{(c)} &=& \\frac{1}{2(n + \\lambda)} \\quad \\text{for} \\quad i=1,2,\\ldots,2n
            \\end{array}

        The weights :math:`\\mathbf{w}^{(m)}` are used to compute the mean, and
        the weights :math:`\\mathbf{w}^{(c)}` are used to compute the
        covariance.

        The constant :math:`\\beta` is used to incorporate prior knowledge
        about the distribution of the random variable, and is usually set to
        :math:`2`.

        The scaling parameter :math:`\\alpha` determines the spread of the
        sigma points around the mean. A small value of :math:`\\alpha` results
        in sigma points that are close to the mean, while a larger value
        results in sigma points that are more spread out. The value of
        :math:`\\alpha` is usually set to a small positive number, typically in
        the range of :math:`10^{-3}` to :math:`10^{-1}`.

        Returns
        -------
        tuple
            Weights for mean and covariance.

            - ``weight_mean``: Weights for mean.
            - ``weight_covariance``: Weights for covariance.

        """
        # Weights for sigma points
        weight_mean = np.zeros(self.sigma_point_count)
        weight_covariance = np.zeros(self.sigma_point_count)
        # Set weights
        weight_mean[0] = self.lambda_param / (self.state_dimension + self.lambda_param)
        weight_covariance[0] = weight_mean[0] + (1 - self.alpha**2 + self.beta)
        weight_covariance[1:] = weight_mean[1:] = 1.0 / (2 * (self.state_dimension + self.lambda_param))
        return weight_mean, weight_covariance

    def compute_sigma_points(self, state: np.ndarray, state_covariance: np.ndarray) -> np.ndarray:
        """
        Sigma Points computation.

        Given a state :math:`\\mathbf{x}` and its covariance
        :math:`\\mathbf{P}_{xx}`, compute the sigma points
        :math:`\\mathcal{X}` as:

        .. math::

            \\begin{array}{rcl}
            \\mathcal{X}_0 &=& \\mathbf{x}_{t-1} \\\\
            \\mathcal{X}_i &=& \\mathbf{x}_{t-1} + \\mathbf{L}_i \\\\
            \\mathcal{X}_{i+n} &=& \\mathbf{x}_{t-1} - \\mathbf{L}_i
            \\end{array}

        where the **Square Root** covariance matrix :math:`\\mathbf{L}` is
        obtained using the `Cholesky decomposition
        <https://en.wikipedia.org/wiki/Cholesky_decomposition>`_:

        .. math::

            \\mathbf{L} = \\mathrm{chol}\\Big(\\sqrt{(n + \\lambda)\\mathbf{P_{xx}}}\\Big)

        Returns
        -------
        sigma_points : numpy.ndarray
            Sigma points array of shape `(2n+1, n)`, where `n` is the state
            dimension.
        """
        try:
            sqrt_covariance = np.linalg.cholesky((self.state_dimension + self.lambda_param) * state_covariance)
        except np.linalg.LinAlgError:
            # Add small regularization if Cholesky decomposition fails
            regularized_covariance = state_covariance + np.eye(self.state_dimension) * 1e-8
            sqrt_covariance = np.linalg.cholesky((self.state_dimension + self.lambda_param) * regularized_covariance)
        sigma_points = np.zeros((self.state_dimension, 2 * self.state_dimension + 1))
        sigma_points[:, 0] = state
        for i in range(self.state_dimension):
            sigma_points[:, i + 1] = state + sqrt_covariance[:, i]
            sigma_points[:, self.state_dimension + i + 1] = state - sqrt_covariance[:, i]
        return sigma_points

    def Omega(self, x: np.ndarray) -> np.ndarray:
        """
        Omega operator.

        Given a vector :math:`\\mathbf{x}\\in\\mathbb{R}^3`, return a
        :math:`4\\times 4` matrix of the form:

        .. math::
            \\boldsymbol\\Omega(\\mathbf{x}) =
            \\begin{bmatrix}
                0 & -\\mathbf{x}^T \\\\
                \\mathbf{x} & -\\lfloor\\mathbf{x}\\rfloor_\\times
            \\end{bmatrix} =
            \\begin{bmatrix}
                0   & -x_1 & -x_2 & -x_3 \\\\
                x_1 &    0 &  x_3 & -x_2 \\\\
                x_2 & -x_3 &    0 &  x_1 \\\\
                x_3 &  x_2 & -x_1 & 0
            \\end{bmatrix}

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

    def update(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray = None, dt: float = None) -> np.ndarray:
        """
        Perform an update of the state.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori state describing orientation as quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2.
        dt : float, default: None
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        updated_quaternion : numpy.ndarray
            Estimated state describing orientation as quaternion.

        """
        _assert_numerical_iterable(q, 'Quaternion')
        _assert_numerical_iterable(gyr, 'Tri-axial gyroscope sample')
        _assert_numerical_iterable(acc, 'Tri-axial accelerometer sample')
        if mag is not None:
            _assert_numerical_iterable(mag, 'Tri-axial magnetometer sample')
        dt = self.Dt if dt is None else dt

        ## Prediction Step
        # 1. Generate Sigma Points
        sigma_points = self.compute_sigma_points(q, self.P)

        # 2. Process model - propagate sigma points with gyro data
        rotation_operator = np.eye(4) + 0.5 * self.Omega(gyr) * dt
        sigma_points_propagated = np.zeros_like(sigma_points)
        for i in range(self.sigma_point_count):
            sigma_points_propagated[:, i] = Quaternion(rotation_operator @ sigma_points[:, i])

        # 3.1. Predicted state mean (y_bar)
        predicted_state_mean = Quaternion(np.sum(sigma_points_propagated * self.weight_mean, axis=1))

        # 3.2 Predicted States difference and Predicted state covariance
        # Quaternion variation using direct difference, although less robust for large rotations
        predicted_covariance = np.zeros((self.state_dimension, self.state_dimension))
        for i in range(self.sigma_point_count):
            diff = sigma_points_propagated[:, i] - predicted_state_mean
            predicted_covariance += self.weight_covariance[i] * np.outer(diff, diff)
        predicted_covariance += self.Q_t # Add process noise

        ## Correction
        # 4. Measurement Model: Transform Sigma Points into Measurement Space (expected sensor readings)
        measurement_dimension = 6 if mag is not None else 3
        sigma_points_measurement_space = np.zeros((measurement_dimension, self.sigma_point_count))
        for i in range(self.sigma_point_count):
            # Rotation matrix from inertial to body frame
            R_i_to_b = Quaternion(sigma_points_propagated[:, i]).to_DCM().T
            # Expected accelerometer reading in body frame
            sigma_points_measurement_space[:3, i] = R_i_to_b @ self.a_ref
            if mag is not None:
                # Expected magnetometer reading in body frame
                sigma_points_measurement_space[3:, i] = R_i_to_b @ self.m_ref

        # 5.1. Predicted measurement mean
        predicted_measurement_mean = np.sum(sigma_points_measurement_space * self.weight_mean, axis=1)

        # 5.2. Measurement Covariance
        measurement_covariance = np.zeros((measurement_dimension, measurement_dimension))
        # 6. Cross-Covariance
        cross_covariance = np.zeros((self.state_dimension, measurement_dimension))
        for i in range(self.sigma_point_count):
            diff_measurement = sigma_points_measurement_space[:, i] - predicted_measurement_mean
            diff_state = sigma_points_propagated[:, i] - predicted_state_mean
            measurement_covariance += self.weight_covariance[i] * np.outer(diff_measurement, diff_measurement)
            cross_covariance += self.weight_covariance[i] * np.outer(diff_state, diff_measurement)
        # Kronecker product is used to extend covariance if magnetometer is present.
        measurement_noise_covariance = np.kron(np.eye(2), self.R) if mag is not None else self.R
        measurement_covariance += measurement_noise_covariance # Add measurement noise

        # 7. Kalman Gain
        kalman_gain = cross_covariance @ np.linalg.inv(measurement_covariance)

        # 8. Innovation (measurement residual)
        actual_measurement = np.concatenate((acc, mag)) if mag is not None else acc
        innovation = actual_measurement - predicted_measurement_mean

        # 9.1. Update State Estimate
        correction = kalman_gain @ innovation
        updated_quaternion = Quaternion(predicted_state_mean + correction)

        # 9.2. Update Covariance Estimate
        self.P = predicted_covariance - kalman_gain @ measurement_covariance @ kalman_gain.T
        return updated_quaternion

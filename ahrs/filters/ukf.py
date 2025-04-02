# -*- coding: utf-8 -*-
"""
.. attention::

    The UKF algorithm and its documentation are **under development**. The
    current implementation is functional for IMU-architecture only, but is not
    yet finalized.

    Wait until pypi release 0.4.0 for a fully tested version.

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
   at time :math:`t`, given the previous state at time :math:`t-1`.

.. math::
    \\begin{array}{rcl}
    \\hat{\\mathbf{x}}_t &=& \\mathbf{F}\\mathbf{x}_{t-1} + \\mathbf{Bu}_t \\\\
    \\hat{\\mathbf{P}}_t &=& \\mathbf{F}\\mathbf{P}_{t-1}\\mathbf{F}^T + \\mathbf{Q}_t
    \\end{array}

2. The **correction step** improves the prediction with a measurement (or set
   of measurements) :math:`\\mathbf{z}_t` at time :math:`t`.

.. math::
    \\begin{array}{rcl}
    \\mathbf{v}_t &=& \\mathbf{z}_t - \\mathbf{H}\\hat{\\mathbf{x}}_t \\\\
    \\mathbf{S}_t &=& \\mathbf{H} \\hat{\\mathbf{P}}_t \\mathbf{H}^T + \\mathbf{R} \\\\
    \\mathbf{K}_t &=& \\hat{\\mathbf{P}}_t \\mathbf{H}^T \\mathbf{S}_t^{-1} \\\\
    \\mathbf{x}_t &=& \\hat{\\mathbf{x}}_t + \\mathbf{K}_t \\mathbf{v}_t \\\\
    \\mathbf{P}_t &=& \\hat{\\mathbf{P}}_t - \\mathbf{K}_t \\mathbf{S}_t \\mathbf{K}_t^T
    \\end{array}

The Kalman filter, however, is limited to linear systems, rendering the process
above inapplicable to nonlinear systems like our attitude estimation problem.

**Extended Kalman Filter**

A common solution to this issue is the `Extended Kalman Filter <./ekf.html>`_
(EKF), which linearizes the system model and measurement functions around the
current estimate to approximate the terms, allowing the use of the Kalman
filter equations as if it were a linear system.

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

whereas the measurement model is linearized as:

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

Therefore, before computing the sigma points, we first calculate the Cholesky
decomposition of :math:`(n + \\lambda)\\mathbf{P_{xx}}`, and then we obtain
them by adding and subtracting the columns of :math:`\\mathbf{L}` to the mean.

.. math::

    \\begin{array}{rcl}
    \\mathcal{X}_0 &=& \\bar{\\mathbf{x}} \\\\
    \\mathcal{X}_i &=& \\bar{\\mathbf{x}} + \\mathbf{L}_i \\\\
    \\mathcal{X}_{i+n} &=& \\bar{\\mathbf{x}} - \\mathbf{L}_i
    \\end{array}

where :math:`\\mathbf{L}` is the Cholesky decomposition of
:math:`(n + \\lambda)\\mathbf{P_{xx}}`.

We pass these sigma points through the nonlinear function :math:`f` to get the
transformed points :math:`\\mathcal{Y}`.

.. math::

    \\mathcal{Y} = f(\\mathcal{X})

Their **mean** is given by their wieghted sum:

.. math::

    \\boxed{\\bar{\\mathbf{y}} = \\sum_{i=0}^{2n} W_i^{(m)} \\mathcal{Y}_i}

And their **covariance** by their weighted outer product:

.. math::

    \\boxed{\\mathbf{P_{yy}} = \\sum_{i=0}^{2n} W_i^{(c)} (\\mathcal{Y}_i - \\bar{\\mathbf{y}})(\\mathcal{Y}_i - \\bar{\\mathbf{y}})^T + \\mathbf{Q}}

with :math:`\\mathbf{Q}` being the :math:`n\\times n` process noise covariance
matrix.

The weights :math:`W` are computed as:

.. math::

    \\begin{array}{rcl}
    W_0^{(m)} &=& \\frac{\\lambda}{n + \\lambda} \\\\
    W_0^{(c)} &=& \\frac{\\lambda}{n + \\lambda} + (1 - \\alpha^2 + \\beta) \\\\
    W_i^{(m)} = W_i^{(c)} &=& \\frac{1}{2(n + \\lambda)} \\quad \\text{for} \\quad i=1,2,\\ldots,2n
    \\end{array}

The weights :math:`W^{(m)}` are used to compute the mean, and the weights
:math:`W^{(c)}` are used to compute the covariance.

The constant :math:`\\beta` is used to incorporate prior knowledge about the
distribution of the random variable, and is usually set to :math:`2` for
Gaussian distributions :cite:p:`wan2000`.

**UKF Summary**

Given the initial state :math:`\\mathbf{x}_0`, and its covariance matrix
:math:`\\mathbf{P}_0\\in\\mathbb{R}^{n\\times n}`, the UKF algorithm can be
summarized as follows:

**Prediction**:

1. Calculate the sigma points

.. math::

    \\mathcal{X} = \\Big\\{ \\mathcal{X}_0 \\; , \\quad\\mathcal{X}_i \\; , \\quad\\mathcal{X}_{i+n} \\Big\\}

2. Propagate the sigma points through the process model

.. math::

    \\mathcal{Y} = f(\\mathcal{X})

3. Compute the predicted state mean and covariance

.. math::

    \\begin{array}{rcl}
    \\bar{\\mathbf{y}} &=& \\sum_{i=0}^{2n} W_i^{(m)} \\mathcal{Y}_i \\\\ \\\\
    \\mathbf{P}_{yy} &=& \\sum_{i=0}^{2n} W_i^{(c)} (\\mathcal{Y}_i - \\bar{\\mathbf{y}})(\\mathcal{Y}_i - \\bar{\\mathbf{y}})^T + \\mathbf{Q}
    \\end{array}

**Correction**:

4. Transform the predicted sigma points to the measurement space

.. math::

    \\mathcal{Z} = h(\\mathcal{Y})

5. Compute the predicted measurement mean and covariance

.. math::

    \\begin{array}{rcl}
    \\bar{\\mathbf{z}} &=& \\sum_{i=0}^{2n} W_i^{(m)} \\mathcal{Y}_i \\\\ \\\\
    \\mathbf{P}_{yy} &=& \\sum_{i=0}^{2n} W_i^{(c)} (\\mathcal{Y}_i - \\bar{\\mathbf{y}})(\\mathcal{Y}_i - \\bar{\\mathbf{y}})^T + \\mathbf{R}
    \\end{array}

6. Compute the cross-covariance

.. math::

    \\mathbf{P}_{xy} = \\sum_{i=0}^{2n} W_i^{(c)} (\\mathcal{X}_i - \\bar{\\mathbf{x}})(\\mathcal{Y}_i - \\bar{\\mathbf{y}})^T

7. Compute the Kalman gain

.. math::

    \\mathbf{K} = \\mathbf{P}_{xy} \\mathbf{P}_{yy}^{-1}

8. Update the state and covariance

.. math::

    \\begin{array}{rcl}
    \\mathbf{x}_t &=& \\bar{\\mathbf{x}} + \\mathbf{K} (\\mathbf{z}_t - \\bar{\\mathbf{z}}) \\\\ \\\\
    \\mathbf{P}_t &=& \\mathbf{P}_{xx} - \\mathbf{K} \\mathbf{P}_{yy} \\mathbf{K}^T
    \\end{array}

UKF for Attitude Estimation
---------------------------

In this implementation, we build a simple UKF for the attitude estimation, so
that we can focus on the details of the algorithm. Once the basic structure is
understood, we could extend the model to include more complex systems.

We start by defining the main vectors:

.. math::

    \\begin{array}{rcl}
    \\mathbf{x} &=& \\begin{bmatrix} q_w & q_x & q_y & q_z \\end{bmatrix}^T \\\\ \\\\
    \\mathbf{u} &=& \\begin{bmatrix} \\omega_x & \\omega_y & \\omega_z \\end{bmatrix}^T \\\\ \\\\
    \\mathbf{z} &=& \\begin{bmatrix} a_x & a_y & a_z \\end{bmatrix}^T
    \\end{array}

The state vector :math:`\\mathbf{x}_t\\in\\mathbb{R}^4` is the quaternion
representing the orientation at time :math:`t`, the input vector
:math:`\\mathbf{u}_t\\in\\mathbb{R}^3` contains the angular velocity readings
from a tri-axial gyroscope, and the measurement vector
:math:`\\mathbf{z}_t\\in\\mathbb{R}^3` has the readings of a tri-axial
accelerometer.

Notice we don't extend the state vector to include the gyroscope biases like
others do. For the sake of simplicity we don't estimate these biases, and
assume the sensor readings are already calibrated.

**Sigma Points**

The sigma points are computed first, given the previous state and covariance.

Using the cholesky decomposition we obtain the **matrix square root**:

.. math::

    \\mathbf{L} = \\mathrm{chol}\\Big(\\sqrt{(n + \\lambda)\\mathbf{P_{xx}}}\\Big)

where :math:`n=4` is the number of items in the state vector :math:`\\mathbf{x}`,
and :math:`\\lambda=\\alpha^2(n + \\kappa) - n` is the scaling parameter.

Using the default values :math:`\\alpha=0.001`, and :math:`\\kappa=0`, we get:

.. math::

    \\lambda = 0.001^2 (4 + 0) - 4 = -3.999996

which yields :math:`\\mathbf{L} = \\mathrm{chol}\\big(\\sqrt{0.000004\\mathbf{P_{xx}}}\\big)`.

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

The estimation process is done as a two-step filter consisting of an attitude
propagation (using the gyroscope) and a correction (using the accelerometer.)

**Attitude Propagation**

For the propagation we use the the gyroscope data to measure the angular
velocity. Based on the time spent between :math:`t-1` and :math:`t` (known as
the time step :math:`\\Delta t`) we can obtain the angular displacement
:math:`\\boldsymbol\\theta_t`, and add it to the previous attitude
:math:`\\mathbf{x}_{t-1}` to get the new attitude :math:`\\hat{\\mathbf{q}}_t`:

.. math::

    \\begin{array}{rcl}
    \\hat{\\mathbf{x}}_t &=& \\mathbf{x}_{t-1} + \\boldsymbol\\theta_t \\\\
    &=& \\mathbf{x}_{t-1} + \\int_{t-1}^t\\boldsymbol\\omega\\, dt
    \\end{array}

However, this operation is not linear, and we cannot use it in the Kalman
filter. We need to use a linear operation to propagate the attitude. This
common operation, known as **attitude propagation**, defines the **Process
model as**:

.. math::
    \\begin{array}{rcl}
    \\hat{\\mathbf{x}}_t &=& \\mathbf{f}(\\mathbf{x}_{t-1}, \\boldsymbol\\omega_t) \\\\
    &=&\\Big(\\mathbf{I}_4 + \\frac{\\Delta t}{2}\\boldsymbol\\Omega_t\\Big)\\mathbf{x}_{t-1} \\\\
    \\begin{bmatrix}\\hat{q_w} \\\\ \\hat{q_x} \\\\ \\hat{q_y} \\\\ \\hat{q_z}\\end{bmatrix}
    &=&
    \\begin{bmatrix}
        q_w - \\frac{\\Delta t}{2} \\omega_x q_x - \\frac{\\Delta t}{2} \\omega_y q_y - \\frac{\\Delta t}{2} \\omega_z q_z\\\\
        q_x + \\frac{\\Delta t}{2} \\omega_x q_w - \\frac{\\Delta t}{2} \\omega_y q_z + \\frac{\\Delta t}{2} \\omega_z q_y\\\\
        q_y + \\frac{\\Delta t}{2} \\omega_x q_z + \\frac{\\Delta t}{2} \\omega_y q_w - \\frac{\\Delta t}{2} \\omega_z q_x\\\\
        q_z - \\frac{\\Delta t}{2} \\omega_x q_y + \\frac{\\Delta t}{2} \\omega_y q_x + \\frac{\\Delta t}{2} \\omega_z q_w
    \\end{bmatrix}
    \\end{array}

where the term :math:`\\mathbf{I}_4 + \\frac{\\Delta t}{2}\\boldsymbol\\Omega_t`
is a linearized approximation of the attitude propagation.

.. tip::

    For more details about this linear operation, please refer to the `Attitude
    from Angular Rate <./angular.html>`_ documentation.

.. seealso::

   `EKF <./ekf.html>`_ - Extended Kalman Filter for orientation estimation.

"""

import numpy as np
from ..common.quaternion import Quaternion

class UKF:
    def __init__(self, alpha=1e-3, beta=2, kappa=0, **kwargs):
        # UKF parameters
        self.state_dimension = 4    # n : State dimension (Quaternion items)
        self.sigma_point_count = 2 * self.state_dimension + 1   # 2*n+1 sigma points
        self.alpha = alpha          # Spread parameter
        self.beta = beta            # Distribution parameter
        self.kappa = kappa          # Secondary scaling parameter
        # Lambda parameter: λ = α²(n + κ) - n
        self.lambda_param = self.alpha**2 * (self.state_dimension + self.kappa) - self.state_dimension
        # Weights for sigma points
        self.weight_mean, self.weight_covariance = self.set_weights()
        # Process and measurement noise covariances
        self.Q = kwargs.get('process_noise_covariance', np.eye(4) * 0.0001)
        self.R = kwargs.get('measurement_noise_covariance', np.eye(3) * 0.01)
        # Initial state covariance
        self.P = np.eye(self.state_dimension) * 0.01

    def set_weights(self):
        # Weights for sigma points
        weight_mean = np.zeros(self.sigma_point_count)
        weight_covariance = np.zeros(self.sigma_point_count)
        # Set weights
        weight_mean[0] = self.lambda_param / (self.state_dimension + self.lambda_param)
        weight_covariance[0] = weight_mean[0] + (1 - self.alpha**2 + self.beta)
        weight_covariance[1:] = weight_mean[1:] = 1.0 / (2 * (self.state_dimension + self.lambda_param))
        return weight_mean, weight_covariance

    def compute_sigma_points(self, state, state_covariance):
        # Calculate square root of scaled covariance (eq. 36)
        try:
            sqrt_covariance = np.linalg.cholesky((self.state_dimension + self.lambda_param) * state_covariance)
        except np.linalg.LinAlgError:
            # Add small regularization if Cholesky decomposition fails
            regularized_covariance = state_covariance + np.eye(self.state_dimension) * 1e-8
            sqrt_covariance = np.linalg.cholesky((self.state_dimension + self.lambda_param) * regularized_covariance)
        sigma_points = np.zeros((self.sigma_point_count, self.state_dimension)) # Initialize sigma points array
        sigma_points[0] = state                                      # Set mean as the first sigma point
        # Set remaining sigma points as Quaternions (eq. 33)
        for i in range(1, self.state_dimension+1):
            sigma_points[i] = state + sqrt_covariance[i-1]
            sigma_points[i+self.state_dimension] = state - sqrt_covariance[i-1]
        return sigma_points

    def Omega(self, x: np.ndarray) -> np.ndarray:
        return np.array([
            [0.0,  -x[0], -x[1], -x[2]],
            [x[0],   0.0,  x[2], -x[1]],
            [x[1], -x[2],   0.0,  x[0]],
            [x[2],  x[1], -x[0],   0.0]])

    def update(self, q, gyro, acc, dt):
        ## Prediction
        # 1. Normalize accelerometer data
        acc_normalized = acc / np.linalg.norm(acc)

        # 2. Generate sigma points
        sigma_points = self.compute_sigma_points(q, self.P)

        # 3. Process model - propagate sigma points with gyro data (eq. 37)
        rotation_operator = np.eye(4) + 0.5 * self.Omega(gyro) * dt
        predicted_sigma_points = [Quaternion(rotation_operator @ point) for point in sigma_points]

        # 4. Predicted state mean (x_bar) (eq. 38)
        predicted_state_mean = Quaternion(np.sum(self.weight_mean[:, None] * predicted_sigma_points, axis=0))

        # Predicted States difference: x_i - x_bar
        predicted_state_diffs = [points.product(predicted_state_mean.conjugate) * 2.0 for points in predicted_sigma_points]

        # 5. Predicted state covariance (using error quaternions)
        predicted_state_covariance = np.zeros((3, 3))   # 3x3 for orientation error
        for i, eq in enumerate(predicted_state_diffs):
            predicted_state_covariance += self.weight_covariance[i] * np.outer(eq[1:], eq[1:])
        predicted_state_covariance += self.Q[1:, 1:]  # Add process noise (eq. 45)

        ## Correction
        # 6. Transform sigma points to measurement space (predicted accelerometer readings) (eq. 16)
        predicted_measurements = [point.to_DCM().T @ np.array([0, 0, 1]) for point in predicted_sigma_points]

        # 7. Predicted measurement mean (eq. 17)
        predicted_measurement_mean = np.sum(self.weight_mean[:, None] * predicted_measurements, axis=0)

        # Predicted measurements difference: Z_i - z_i
        predicted_measurements_diff = predicted_measurements - predicted_measurement_mean

        # 8. Predicted measurement covariance (eq. 18) (eq. 68)
        predicted_measurement_covariance = np.zeros((3, 3))
        for i, measured_difference in enumerate(predicted_measurements_diff):
            predicted_measurement_covariance += self.weight_covariance[i] * np.outer(measured_difference, measured_difference)
        predicted_measurement_covariance += self.R      # Add measurement noise (eq. 45)

        # 9. Cross-covariance (eq. 71)
        cross_covariance = np.sum(self.weight_covariance[i] * np.outer(predicted_state_diffs[i][1:], predicted_measurements_diff[i]) for i in range(self.sigma_point_count))

        # 10. Calculate Kalman gain (eq. 72)
        kalman_gain = cross_covariance @ np.linalg.inv(predicted_measurement_covariance)

        # 11. Update state with measurement
        innovation = acc_normalized - predicted_measurement_mean                    # Innovation (measurement residual) (eq. 44)
        correction_vector = kalman_gain @ innovation                                # Correction as a rotation vector
        theta = np.linalg.norm(correction_vector)  # Angle of rotation
        correction_quaternion = Quaternion([np.cos(theta/2.0), *(np.sin(theta/2.0) * correction_vector/theta)])  # Convert to quaternion
        updated_quaternion = predicted_state_mean.product(correction_quaternion)    # Apply correction to predicted state

        # 12. Re-define state covariance
        self.P = np.zeros((self.state_dimension, self.state_dimension))  # Reset covariance
        self.P[1:, 1:] = predicted_state_covariance - kalman_gain @ predicted_measurement_covariance @ kalman_gain.T

        return updated_quaternion

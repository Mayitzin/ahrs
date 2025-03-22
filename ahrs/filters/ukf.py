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

If we know how the state was at time :math:`t-1`, we predict how the state will
be at time :math:`t`. In addition, we also have a set of measurements
:math:`\\mathbf{z}_t` that can be used to improve the prediction of the state.

The traditional `Kalman filter <https://en.wikipedia.org/wiki/Kalman_filter>`_,
as described by :cite:p:`kalman1960` computes a state in two steps:

1. The **prediction step** computes a guess of the current state,
   :math:`\\hat{\\mathbf{x}}_t`, and its covariance :math:`\\hat{\\mathbf{P}}_t`,
   at time :math:`t`, given the previous state at time :math:`t-1`.

.. math::
    \\begin{array}{rcl}
    \\hat{\\mathbf{x}}_t &=& \\mathbf{F}\\mathbf{x}_{t-1} + \\mathbf{Bu}_t \\\\
    \\hat{\\mathbf{P}}_t &=& \\mathbf{F}(\\mathbf{x}_{t-1}, \\mathbf{u}_t)\\mathbf{P}_{t-1}\\mathbf{F}^T(\\mathbf{x}_{t-1}, \\mathbf{u}_t) + \\mathbf{Q}_t
    \\end{array}

2. The **correction step** rectifies the estimation with a measurement (or set
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

**Sigma Points**

Imagine there is a set of random points :math:`\\mathbf{x}` with mean
:math:`\\bar{\\mathbf{x}}`, and covariance :math:`\\mathbf{P_{xx}}`, and there
is another set of random points :math:`\\mathbf{y}` related to
:math:`\\mathbf{x}` by a nonlinear function :math:`\\mathbf{y} = f(\\mathbf{x})`.

Our goal is to find the mean :math:`\\bar{\\mathbf{y}}` and covariance
:math:`\\mathbf{P_{yy}}` of :math:`\\mathbf{y}`. The unscented transform
approximates the by sampling a set of points from :math:`\\mathbf{x}` and
applying the nonlinear function :math:`f` to each of the sampled points.

The samples are not drawn at random but according to a deterministic method.
Information about the distribution can be captured using only a small number of
points.

The :math:`n`-dimensional random variable :math:`\\mathbf{x}` with mean
:math:`\\bar{\\mathbf{x}}` and covariance :math:`\\mathbf{P_{xx}}` is
approximated by :math:`2n + 1` points given by

.. math::

    \\begin{array}{rcl}
    \\mathcal{X}_0 &=& \\bar{\\mathbf{x}} \\\\
    \\mathcal{X}_i &=& \\bar{\\mathbf{x}} + \\Big(\\sqrt{(n + \\lambda)\\mathbf{P_{xx}}}\\Big)_i\\\\
    \\mathcal{X}_{i+n} &=& \\bar{\\mathbf{x}} - \\Big(\\sqrt{(n + \\lambda)\\mathbf{P_{xx}}}\\Big)_i
    \\end{array}

where :math:`(\\sqrt{(n + \\lambda)\\mathbf{P_{xx}}})_i` is the :math:`i`-th
column of the matrix square root, and :math:`\\lambda=\\alpha^2(n + \\kappa) - n`
is a scaling parameter.

:math:`\\alpha` determines the spread of the sigma points around the mean,
usually set to :math:`0.001`, and :math:`\\kappa` is a secondary scaling
parameter, usually set to :math:`0` :cite:p:`wan2000`.

We start the computation of the sigma points by passing them through the
nonlinear function :math:`f` to get the transformed points :math:`\\mathcal{Y}`.

.. math::

    \\mathcal{Y}_i = f(\\mathcal{X}_i)

Their mean is given by their wieghted sum:

.. math::

    \\bar{\\mathbf{y}} = \\sum_{i=0}^{2n} W_i^{(m)} \\mathcal{Y}_i

And their covariance is given by their weighted outer product:

.. math::

    \\mathbf{P_{yy}} = \\sum_{i=0}^{2n} W_i^{(c)} (\\mathcal{Y}_i - \\bar{\\mathbf{y}})(\\mathcal{Y}_i - \\bar{\\mathbf{y}})^T

The weights :math:`W` are computed as:

.. math::

    \\begin{array}{rcl}
    W_0^{(m)} &=& \\frac{\\lambda}{n + \\lambda} \\\\
    W_0^{(c)} &=& \\frac{\\lambda}{n + \\lambda} + (1 - \\alpha^2 + \\beta) \\\\
    W_i^{(m)} = W_i^{(c)} &=& \\frac{1}{2(n + \\lambda)} \\quad \\text{for} \\quad i=1,2,\\ldots,2n
    \\end{array}

UKF for Attitude Estimation
---------------------------

We start by defining the state vector :math:`\\mathbf{x}_t`, and the
measurement vector :math:`\\mathbf{z}_t` as:

.. math::

    \\begin{array}{rcl}
    \\mathbf{x}_t &=& \\begin{bmatrix} q_w & q_x & q_y & q_z \\end{bmatrix}^T \\\\ \\\\
    \\mathbf{z}_t &=& \\begin{bmatrix} a_x & a_y & a_z \\end{bmatrix}^T
    \\end{array}

In this case, we will try to have a very simple model for the UKF, so that we
can focus on the implementation details.

Given the initial state :math:`\\mathbf{x}_0`, its covariance matrix
:math:`\\mathbf{P}_0` the UKF algorithm can be summarized as follows:

**Prediction**:

1. Calculate the sigma points

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

    def compute_sigma_points(self, quaternion_state, state_covariance):
        # Calculate square root of scaled covariance (eq. 36)
        try:
            sqrt_covariance = np.linalg.cholesky((self.state_dimension + self.lambda_param) * state_covariance)
        except np.linalg.LinAlgError:
            # Add small regularization if Cholesky decomposition fails
            regularized_covariance = state_covariance + np.eye(self.state_dimension) * 1e-8
            sqrt_covariance = np.linalg.cholesky((self.state_dimension + self.lambda_param) * regularized_covariance)
        sigma_points = np.zeros((self.sigma_point_count, self.state_dimension)) # Initialize sigma points array
        sigma_points[0] = quaternion_state                                      # Set mean as the first sigma point
        # Set remaining sigma points as Quaternions (eq. 33)
        for i in range(1, self.state_dimension+1):
            sigma_points[i] = Quaternion(quaternion_state + sqrt_covariance[i-1])
            sigma_points[i+self.state_dimension] = Quaternion(quaternion_state - sqrt_covariance[i-1])
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

        # 4. Predicted state mean (x_i) (eq. 38)
        predicted_state_mean = Quaternion(np.sum(self.weight_mean[:, None] * predicted_sigma_points, axis=0))

        # Predicted States difference: X_i - x_i
        predicted_state_diffs = [points.product(predicted_state_mean.conjugate) * 2.0 for points in predicted_sigma_points]

        # 5. Predicted state covariance (using error quaternions) (eq. 70)
        predicted_state_covariance = np.zeros((3, 3))   # 3x3 for orientation error
        for i, eq in enumerate(predicted_state_diffs):
            predicted_state_covariance += self.weight_covariance[i] * np.outer(eq[1:], eq[1:])
        # Add process noise to orientation part
        predicted_state_covariance += self.Q[1:4, 1:4]

        ## Correction
        # 6. Transform sigma points to measurement space (predicted accelerometer readings) (eq. 16)
        predicted_measurements = [point.to_DCM().T @ np.array([0, 0, 1]) for point in predicted_sigma_points]

        # 7. Predicted measurement mean (eq. 17)
        predicted_measurement_mean = np.sum(self.weight_mean[:, None] * predicted_measurements, axis=0)

        # Predicted measurements difference: Z_i - z_i
        predicted_measurements_diff = predicted_measurements - predicted_measurement_mean

        # 8. Predicted measurement covariance (eq. 18) (eq. 68)
        predicted_measurement_covariance = np.zeros((3, 3))
        for i in range(self.sigma_point_count):
            predicted_measurement_covariance += self.weight_covariance[i] * np.outer(predicted_measurements_diff[i], predicted_measurements_diff[i])
        predicted_measurement_covariance += self.R      # Add measurement noise (eq. 45)

        # 9. Cross-covariance (eq. 71)
        cross_covariance = np.zeros((3, 3))
        for i in range(self.sigma_point_count):
            cross_covariance += self.weight_covariance[i] * np.outer(predicted_state_diffs[i][1:], predicted_measurements_diff[i]) # Update cross-covariance with vector part of error quaternion

        # 10. Calculate Kalman gain (eq. 72)
        kalman_gain = cross_covariance @ np.linalg.inv(predicted_measurement_covariance)

        # 11. Update state with measurement
        innovation = acc_normalized - predicted_measurement_mean                    # Innovation (measurement residual) (eq. 44)
        correction_vector = kalman_gain @ innovation                                # Correction as a rotation vector
        correction_quaternion = Quaternion([1.0, *(correction_vector/2.0)])         # Convert to quaternion (small angle approximation)
        updated_quaternion = predicted_state_mean.product(correction_quaternion)    # Apply correction to predicted state

        # 12. Re-define state covariance
        self.P = np.zeros((4, 4))
        self.P[1:, 1:] = predicted_state_covariance - kalman_gain @ predicted_measurement_covariance @ kalman_gain.T

        return updated_quaternion

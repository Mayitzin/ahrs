# -*- coding: utf-8 -*-
"""
.. attention::

    The UKF algorithm and its documentation are **under development**. The
    current implementation is functional for IMU-architecture only. It may not
    work as expected.

    Wait until pypi release 0.4.0 for a fully tested version.

The Unscented Kaman filter was first proposed by S. Julier and J. Uhlmann
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

Kalman Filter
-------------

We have a `discrete system <https://en.wikipedia.org/wiki/Discrete_system>`_,
whose `states <https://en.wikipedia.org/wiki/State_(computer_science)>`_ are
described by a vector :math:`\\mathbf{x}_t` at each time :math:`t`.

This vector has :math`n` items describing an object in the system. These items
could be the position, velocity, orientation, etc. Basically, anything that can
be measured or estimated can be a state, as long as it can be described
numerically.

Given that we know how the state was at time :math:`t-1`, we would like to
predict how the state will be at time :math:`t`. We also have a set of
measurements :math:`\\mathbf{z}_t` that can be used to improve the prediction
of the state.

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

where:

- :math:`\\hat{\\mathbf{F}}\\in\\mathbb{R}^{n\\times n}` is the **State
  Transition Matrix**.
- :math:`\\hat{\\mathbf{P}}_t\\in\\mathbb{R}^{n\\times n}` is the **Predicted
  Covariance** of the state at time :math:`t`.
- :math:`\\mathbf{B}\\in\\mathbb{R}^{n\\times k}` is the **Control input model**.
- :math:`\\mathbf{u}_t\\in\\mathbb{R}^k` is a **Control input vector**.
- :math:`\\mathbf{Q}_t\\in\\mathbb{R}^{n\\times n}` is the **Process Noise
  Covariance**.
- :math:`\\mathbf{H}\\in\\mathbb{R}^{m\\times n}` is the **Observation model**.
- :math:`\\mathbf{R}_t\\in\\mathbb{R}^{m\\times m}` is the **Measurement Noise
  Covariance**.
- :math:`\\mathbf{K}_t\\in\\mathbb{R}^{n\\times m}` is the filter *gain*,
  a.k.a. the **Kalman Gain**.
- :math:`\\mathbf{P}_t\\in\\mathbb{R}^{n\\times n}` is the **Updated
  Covariance** of the corrected state :math:`\\mathbf{x}_t`.

The Kalman filter, however, is limited to linear systems, rendering the process
above inapplicable to nonlinear systems, like our attitude estimation problem.

A common solution to this issue is the `Extended Kalman Filter  <./ekf.html>`_
(EKF), which linearizes the system model and measurement functions around the
current estimate to approximate the terms llinearly, allowing the use of the
Kalman filter equations as if it were a linear system.

In this approach we modify the state transition and measurement models to
include the Jacobian matrices of the nonlinear functions, which are used to
linearize the system.

The predicted mean and covariance are computed using the linearized models:

.. math::

    \\begin{array}{rcl}
    \\hat{\\mathbf{x}}_t &=& \\mathbf{F}(\\mathbf{x}_{t-1}, \\mathbf{u}_t) \\\\
    \\hat{\\mathbf{P}}_t &=& \\mathbf{F}(\\mathbf{P}_{t-1}, \\mathbf{Q}_t) \\mathbf{F}^T + \\mathbf{Q}_t \\\\
    \\end{array}

with the Jacobian:

.. math::

    \\mathbf{F}(\\mathbf{x}_{t-1}, \\mathbf{u}_t) = \\frac{\\partial \\mathbf{F}}{\\partial \\mathbf{x}} \\bigg|_{\\mathbf{x}_{t-1}, \\mathbf{u}_t}

whereas the measurement model is linearized as:

.. math::

    \\begin{array}{rcl}
    \\mathbf{v}_t &=& \\mathbf{z}_t - \\mathbf{h}(\\mathbf{x}_t) \\\\
    \\mathbf{S}_t &=& \\mathbf{H}(\\mathbf{x}_t) \\hat{\\mathbf{P}}_t \\mathbf{H}^T(\\mathbf{x}_t) + \\mathbf{R}_t \\\\
    \\mathbf{K}_t &=& \\hat{\\mathbf{P}}_t \\mathbf{H}^T(\\mathbf{x}_t) \\mathbf{S}_t^{-1} \\\\
    \\mathbf{x}_t &=& \\hat{\\mathbf{x}}_t + \\mathbf{K}_t \\mathbf{v}_t \\\\
    \\mathbf{P}_t &=& \\big(\\mathbf{I}_4 - \\mathbf{K}_t\\mathbf{H}(\\mathbf{x}_t)\\big)\\hat{\\mathbf{P}}_t
    \\end{array}

with the Jacobian:

.. math::

    \\mathbf{H}(\\hat{\\mathbf{x}}_t) = \\frac{\\partial \\mathbf{H}}{\\partial \\mathbf{x}} \\bigg|_{\\hat{\\mathbf{x}}_t}

Unfortunately these approximations can introduce large errors in the posterior
mean and covariance of the transformed random variable, which may lead to
sub-optimal performance.

To avoid these issues, a solution using unscented transforms was proposed by
Julier and Uhlmann :cite:p:`julier1997`, which is the basis for the Unscented
Kalman Filter (UKF).

Unscented Kalman Filter
------------------------

The UKF is a type of Kalman filter that replaces the linearization with a
deterministic sampling technique called the `Unscented Transform
<https://en.wikipedia.org/wiki/Unscented_transform>`_.

This transformation generates a set of points that capture the mean and
covariance of the state distribution, called the **Sigma Points**.

Each of the sigma points is used as an input to the state transition
and measurement functions to get a new set of transformed state points. The
mean and covariance of the transformed points is then used to obtain state
estimates and state estimation error covariance.

This propagation captures the posterior mean and covariance of the state
distribution more accurately than the EKF.

.. seealso::

   `EKF <./ekf.html>`_ - Extended Kalman Filter for orientation estimation.

"""

import numpy as np
from ..common.quaternion import Quaternion

class UKF:
    def __init__(self, alpha=1e-3, beta=2, kappa=0, **kwargs):
        # UKF parameters
        self.state_dimension = 4    # L : State dimension (Quaternion items)
        self.sigma_point_count = 2 * self.state_dimension + 1   # 2*L+1 sigma points
        self.alpha = alpha          # Spread parameter
        self.beta = beta            # Distribution parameter
        self.kappa = kappa          # Secondary scaling parameter
        # Lambda parameter: λ = α²(L + κ) - L
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

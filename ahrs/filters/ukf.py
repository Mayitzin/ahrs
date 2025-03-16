# -*- coding: utf-8 -*-
"""
.. attention::

    The UKF algorithm and its documentation are **under development**. The
    current implementation is functional for IMU-architecture only, but may not
    work as expected.

    Wait until release 0.4.0 for a stable version.

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
estimations proposed by Wan and Rudolph van de Merwe :cite:p:`wan2000`, and 
urther developed by Kraft :cite:p:`kraft2003` and Klingbeil
:cite:p:`klingbeil2006` for orientation estimation using quaternions.

.. seealso::

   `EKF <./ekf.html>`_ - Extended Kalman Filter for orientation
   estimation.

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
        for i in range(1, self.sigma_point_count):
            weight_mean[i] = 1.0 / (2 * (self.state_dimension + self.lambda_param))
            weight_covariance[i] = weight_mean[i]
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
        for i in range(self.state_dimension):
            sigma_points[i+1] = Quaternion(quaternion_state + sqrt_covariance[i])
            sigma_points[i+1+self.state_dimension] = Quaternion(quaternion_state - sqrt_covariance[i])
        return sigma_points

    def Omega(self, x: np.ndarray) -> np.ndarray:
        return np.array([
            [0.0,  -x[0], -x[1], -x[2]],
            [x[0],   0.0,  x[2], -x[1]],
            [x[1], -x[2],   0.0,  x[0]],
            [x[2],  x[1], -x[0],   0.0]])

    def updateIMU(self, q, gyro, acc, dt):
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
        error_quaternion = [points.product(predicted_state_mean.conjugate) * 2.0 for points in predicted_sigma_points]

        # 5. Predicted state covariance (using error quaternions) (eq. 70)
        predicted_state_covariance = np.zeros((3, 3))   # 3x3 for orientation error
        for i in range(self.sigma_point_count):
            # Error quaternion between sigma point and mean
            predicted_state_covariance += self.weight_covariance[i] * np.outer(error_quaternion[i][1:], error_quaternion[i][1:])
        # Add process noise to orientation part
        predicted_state_covariance += self.Q[1:4, 1:4]

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
            cross_covariance += self.weight_covariance[i] * np.outer(error_quaternion[i][1:], predicted_measurements_diff[i]) # Update cross-covariance with vector part of error quaternion

        # 10. Calculate Kalman gain (eq. 72)
        kalman_gain = cross_covariance @ np.linalg.inv(predicted_measurement_covariance)

        # 11. Update state with measurement
        innovation = acc_normalized - predicted_measurement_mean                    # Innovation (measurement residual) (eq. 44)
        correction_vector = kalman_gain @ innovation                                # Correction as a rotation vector
        correction_quaternion = Quaternion([1.0, *(correction_vector/2.0)])         # Convert to quaternion (small angle approximation)
        updated_quaternion = predicted_state_mean.product(correction_quaternion)    # Apply correction to predicted state

        # 12. Update covariance
        updated_covariance_orientation = predicted_state_covariance - kalman_gain @ predicted_measurement_covariance @ kalman_gain.T
        # Rebuild full state covariance (4x4)
        full_covariance = np.zeros((4, 4))
        full_covariance[1:4, 1:4] = updated_covariance_orientation
        # Store updated covariance
        self.P = full_covariance

        return updated_quaternion

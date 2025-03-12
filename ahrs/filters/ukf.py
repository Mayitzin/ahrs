"""
Unscented Kalman Filter (UKF) for orientation estimation.

This implementation is based on the following paper:

E. Kraft, "A quaternion-based unscented Kalman filter for orientation tracking,"
Sixth International Conference of Information Fusion, 2003. Proceedings of the,
Cairns, QLD, Australia, 2003, pp. 47-54, doi: 10.1109/ICIF.2003.177425.

"""

import numpy as np

def qprod(q1, q2):
    """
    Multiply two quaternions.

    Args:
        q1: First quaternion [qw, qx, qy, qz]
        q2: Second quaternion [qw, qx, qy, qz]

    Returns:
        Quaternion product
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def qnorm(q):
    """Normalize a quaternion to unit magnitude."""
    return q / np.linalg.norm(q)

def _angvel2q(omega, delta_t):
    """
    Convert angular velocity to quaternion.

    Args:
        omega: Angular velocity vector
        delta_t: Time interval

    Returns:
        Quaternion
    """
    angle = np.linalg.norm(omega) * delta_t
    axis = omega / np.linalg.norm(omega)
    return np.array([
        np.cos(angle/2),
        axis[0] * np.sin(angle/2),
        axis[1] * np.sin(angle/2),
        axis[2] * np.sin(angle/2)
    ])

def _quaternion_to_gravity(q):
    """
    Convert quaternion to expected gravity vector in sensor frame.

    Args:
        q: Quaternion [qw, qx, qy, qz]

    Returns:
        Expected gravity vector in sensor frame
    """
    qw, qx, qy, qz = q
    # Rotation of gravity vector [0, 0, 1] by quaternion
    # This is a simplified rotation calculation for the specific case of [0, 0, 1]
    return np.array([
        2 * (qx*qz - qw*qy),
        2 * (qw*qx + qy*qz),
        qw*qw - qx*qx - qy*qy + qz*qz])


class UKF:
    def __init__(self, alpha=1e-3, beta=2, kappa=0, **kwargs):
        # UKF parameters
        self.state_dimension = 4  # Quaternion state dimension
        self.sigma_point_count = 2 * self.state_dimension + 1
        self.alpha = alpha  # Spread parameter
        self.beta = beta    # Distribution parameter
        self.kappa = kappa  # Secondary scaling parameter
        # Lambda parameter for sigma point calculation
        self.lambda_param = self.alpha**2 * (self.state_dimension + self.kappa) - self.state_dimension
        # Process and measurement noise covariances
        self.Q = kwargs.get('process_noise_covariance', np.eye(4) * 0.0001)
        self.R = kwargs.get('measurement_noise_covariance', np.eye(3) * 0.01)
        # Weights for sigma points
        self.weight_mean = np.zeros(self.sigma_point_count)
        self.weight_covariance = np.zeros(self.sigma_point_count)
        # Set weights
        self.weight_mean[0] = self.lambda_param / (self.state_dimension + self.lambda_param)
        self.weight_covariance[0] = self.weight_mean[0] + (1 - self.alpha**2 + self.beta)
        for i in range(1, self.sigma_point_count):
            self.weight_mean[i] = 1.0 / (2 * (self.state_dimension + self.lambda_param))
            self.weight_covariance[i] = self.weight_mean[i]
        # Initial state covariance
        self.P = np.eye(self.state_dimension) * 0.01

    def compute_sigma_points(self, quaternion_state, state_covariance):
        # Calculate square root of scaled covariance
        try:
            sqrt_covariance = np.linalg.cholesky((self.state_dimension + self.lambda_param) * state_covariance)
        except np.linalg.LinAlgError:
            # Add small regularization if Cholesky decomposition fails
            regularized_covariance = state_covariance + np.eye(self.state_dimension) * 1e-8
            sqrt_covariance = np.linalg.cholesky((self.state_dimension + self.lambda_param) * regularized_covariance)

        # Initialize sigma points array
        sigma_points = np.zeros((self.sigma_point_count, self.state_dimension))

        # Set mean as the first sigma point
        sigma_points[0] = quaternion_state

        # Set remaining sigma points
        for i in range(self.state_dimension):
            sigma_points[i+1] = quaternion_state + sqrt_covariance[i]
            sigma_points[i+1+self.state_dimension] = quaternion_state - sqrt_covariance[i]

            # Normalize all quaternions
            sigma_points[i+1] = qnorm(sigma_points[i+1])
            sigma_points[i+1+self.state_dimension] = qnorm(sigma_points[i+1+self.state_dimension])
        return sigma_points

    def updateIMU(self, q, gyro, acc, dt):
        # Store current state
        quaternion_state = q.copy()

        # 1. Normalize accelerometer data
        acc_normalized = acc / np.linalg.norm(acc)

        # 2. Generate sigma points
        sigma_points = self.compute_sigma_points(quaternion_state, self.P)

        # 3. Process model - propagate sigma points with gyro data (eq. 37)
        rotation_quaternion = _angvel2q(gyro, dt)
        predicted_sigma_points = np.zeros_like(sigma_points)
        for i in range(self.sigma_point_count):
            predicted_sigma_points[i] = qprod(sigma_points[i], rotation_quaternion)
            predicted_sigma_points[i] = qnorm(predicted_sigma_points[i])

        # 4. Calculate predicted state mean
        predicted_state_mean = np.zeros(self.state_dimension)
        for i in range(self.sigma_point_count):
            predicted_state_mean += self.weight_mean[i] * predicted_sigma_points[i]
        predicted_state_mean = qnorm(predicted_state_mean)

        # 5. Calculate predicted state covariance (using error quaternions)
        predicted_state_covariance = np.zeros((3, 3))  # 3x3 for orientation error
        for i in range(self.sigma_point_count):
            # Calculate error quaternion between sigma point and mean
            error_quaternion = qprod(
                predicted_sigma_points[i],
                qconj(predicted_state_mean))
            # Small angle approximation - use vector part scaled by 2
            orientation_error = error_quaternion[1:4] * 2.0
            # Update covariance with weighted outer product
            predicted_state_covariance += self.weight_covariance[i] * np.outer(orientation_error, orientation_error)
        # Add process noise to orientation part
        predicted_state_covariance += self.Q[1:4, 1:4]

        # 6. Transform sigma points to measurement space (predicted accelerometer readings)
        predicted_measurements = np.zeros((self.sigma_point_count, 3))
        for i in range(self.sigma_point_count):
            predicted_measurements[i] = _quaternion_to_gravity(predicted_sigma_points[i])

        # 7. Calculate predicted measurement mean
        predicted_measurement_mean = np.zeros(3)
        for i in range(self.sigma_point_count):
            predicted_measurement_mean += self.weight_mean[i] * predicted_measurements[i]

        # 8. Calculate predicted measurement covariance
        predicted_measurement_covariance = np.zeros((3, 3))
        for i in range(self.sigma_point_count):
            measurement_diff = predicted_measurements[i] - predicted_measurement_mean
            predicted_measurement_covariance += self.weight_covariance[i] * np.outer(measurement_diff, measurement_diff)
        # Add measurement noise
        predicted_measurement_covariance += self.R

        # 9. Calculate cross-covariance
        cross_covariance = np.zeros((3, 3))
        for i in range(self.sigma_point_count):
            # Error quaternion
            error_quaternion = qprod(
                predicted_sigma_points[i],
                qconj(predicted_state_mean)
            )
            # Orientation error (vector part)
            orientation_error = error_quaternion[1:4] * 2.0
            # Measurement difference
            measurement_diff = predicted_measurements[i] - predicted_measurement_mean
            # Update cross-covariance
            cross_covariance += self.weight_covariance[i] * np.outer(orientation_error, measurement_diff)

        # 10. Calculate Kalman gain
        kalman_gain = cross_covariance @ np.linalg.inv(predicted_measurement_covariance)

        # 11. Update state with measurement
        # Innovation (measurement residual)
        innovation = acc_normalized - predicted_measurement_mean
        # Correction as a rotation
        correction_vector = kalman_gain @ innovation
        # Convert to quaternion (small angle approximation)
        correction_quaternion = np.array([
            1.0,
            correction_vector[0] / 2.0,
            correction_vector[1] / 2.0,
            correction_vector[2] / 2.0
        ])
        correction_quaternion = qnorm(correction_quaternion)
        # Apply correction to predicted state
        updated_quaternion = qprod(predicted_state_mean, correction_quaternion)
        updated_quaternion = qnorm(updated_quaternion)

        # 12. Update covariance
        updated_covariance_orientation = predicted_state_covariance - kalman_gain @ predicted_measurement_covariance @ kalman_gain.T
        # Ensure symmetry and positive definiteness
        updated_covariance_orientation = (updated_covariance_orientation + updated_covariance_orientation.T) / 2.0
        # Small regularization if needed
        min_eigenvalue = np.min(np.real(np.linalg.eigvals(updated_covariance_orientation)))
        if min_eigenvalue < 0:
            updated_covariance_orientation += np.eye(3) * (abs(min_eigenvalue) + 1e-8)
        # Rebuild full state covariance (4x4)
        full_covariance = np.zeros((4, 4))
        full_covariance[1:4, 1:4] = updated_covariance_orientation
        # Store updated covariance
        self.P = full_covariance

        return updated_quaternion

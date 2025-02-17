# -*- coding: utf-8 -*-
"""
Fast Kalman Filter
==================

Implementation of the Fast Kalman Filter algorithm for orientation estimation
described in "Novel MARG-Sensor Orientation Algorithm Using Fast Kalman Filter"
:cite:p:`guo2017`.

The Fast Kalman Filter (FKF) is a quaternion-based orientation estimation
algorithm that focuses on the fusion of accelerometer and magnetometer
measurements.

In the FKF, the quaternion kinematic equation is the process model, similar to
the EKF. Likewise, the measurement model is based on the accelerometer and
magnetometer readings.

The FKF algorithm is computationally efficient, because it symbolically solves
the linear system of equations of the observation model.

The steps of the KF are summarized as:

1. **State Prediction**

.. math::
    \\mathbf{q}_t^- = \\Phi_k \\mathbf{q}_{k-1}

2. **Covariance Prediction**

.. math::
    \\mathbf{\\Sigma}_k = \\Phi_k \\mathbf{P}_{k-1} \\Phi_k^T + \\Xi_k

3. **Kalman Gain**

.. math::
    \\mathbf{G}_k = \\mathbf{P}_k (\\mathbf{P}_k + \\boldsymbol\\varepsilon_k)^{-1}

4. **State Update**

.. math::
    \\mathbf{q}_k = \\mathbf{q}_k + \\mathbf{G}_k (\\mathbf{y}_k - \\mathbf{q}_k)

5. **Covariance Update**

.. math::
    \\mathbf{P}_k = (\\mathbf{I} - \\mathbf{G}_k) \\mathbf{P}_k

As we can see, the date vector is defined merely as the quaternion itself,
without considering biases over the sensors.

"""

from typing import Tuple
from typing import Optional
import numpy as np
from ..utils.core import _assert_numerical_iterable
from ..common.orientation import ecompass

class FKF:
    """
    Class of Fast Kalman Filter algorithm

    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in nT
    frequency : float, default: 100.0
        Sampling frequency in Hz.
    Dt : float, default: 0.01
        Sampling period in seconds. If ``frequency`` is given, ``Dt`` is
        computed as 1/frequency.
    sigma_g : float, default: 0.01
        Standard deviation of the gyroscope noise.
    sigma_a : float, default: 0.01
        Standard deviation of the accelerometer noise.
    sigma_m : float, default: 0.01
        Standard deviation of the magnetometer noise.
    Pk : float, default: 0.01
        Initial value of the diagonal values of covariance matrix.

    """
    def __init__(self, gyr: Optional[np.ndarray] = None, acc: Optional[np.ndarray] = None, mag: Optional[np.ndarray] = None, **kwargs):
        self.gyr: Optional[np.ndarray] = gyr
        self.acc: Optional[np.ndarray] = acc
        self.mag: Optional[np.ndarray] = mag
        self.frequency: float = kwargs.get('frequency', 100.0)
        self.Dt: float = kwargs.get('Dt', (1.0/self.frequency) if self.frequency else 0.01)
        self.sigma_g: float = kwargs.get('sigma_g', 0.01)
        self.sigma_a: float = kwargs.get('sigma_a', 0.01)
        self.sigma_m: float = kwargs.get('sigma_m', 0.01)
        self.Pk: np.ndarray = np.identity(4)*kwargs.get('Pk', 0.01)
        if all(x is not None for x in[self.gyr, self.acc, self.mag]):
            self.Q: np.ndarray = self._compute_all()

    def Omega4(self, x: np.ndarray) -> np.ndarray:
        """
        Omega operator.

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

    def kalman_update(self, q_1: np.ndarray, q_am: np.ndarray, Pk_1: np.ndarray, Phi: np.ndarray, Sigma_eps: np.ndarray, Sigma_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman update step.

        Parameters
        ----------
        q_1 : numpy.ndarray
            Previous state vector.
        q_am : numpy.ndarray
            Measurement vector.
        Pk_1 : numpy.ndarray
            Previous covariance matrix.
        Phi : numpy.ndarray
            State transition matrix.
        Sigma_eps : numpy.ndarray
            Process noise covariance matrix.
        Sigma_v : numpy.ndarray
            Measurement noise covariance matrix.

        Returns
        -------
        xk : numpy.ndarray
            Updated state vector.
        Pk : numpy.ndarray
            Updated covariance matrix.

        """
        # Prediction
        q_ = Phi @ q_1                                          # Predicted State
        Sigma_q_ = Phi @ (Pk_1 @ Phi.transpose()) + Sigma_eps   # Predicted Covariance
        # Update
        Gk = Sigma_q_ @ (np.linalg.inv(Sigma_q_ + Sigma_v))     # Kalman Gain
        Sigma_q = (np.identity(4) - Gk) @ Sigma_q_              # Updated Covariance
        q = q_ + Gk @ (q_am - q_)                               # Updated State
        return q, Sigma_q

    def measurement_quaternion_acc_mag(self, q: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Measurement model of Accelerometer and Magnetometer.

        Parameters
        ----------
        q : numpy.ndarray
            Predicted quaternion.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer.
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer.

        Returns
        -------
        q : numpy.ndarray
            Measurement quaternion.

        """
        # Assert arrays and normalize observations
        _assert_numerical_iterable(q, 'Quaternion')
        _assert_numerical_iterable(acc, 'Tri-axial accelerometer sample')
        _assert_numerical_iterable(mag, 'Tri-axial magnetometer sample')
        ax, ay, az = acc / np.linalg.norm(acc)
        mx, my, mz = mag / np.linalg.norm(mag)
        qw, qx, qy, qz = q
        # Dynamic magnetometer reference vector (eq. 4)
        mD = ax*mx + ay*my + az*mz
        mN = np.sqrt(1.0 - mD**2)
        # Build measurement quaternion (eq. 25)
        q_new = np.zeros(4)
        q_new[0] =     (ay*mD*my + (1 + az)*(1 + mN*mx + mD*mz) + ax*(mD*mx - mN*mz))*qw              + ((mD + az*mD - ax*mN)*my + ay*(1 + mN*mx - mD*mz))*qx + (ay*mN*my + ax*(-1 + mN*mx + mD*mz) + (1 + az)*(-(mD*mx) + mN*mz))*qy                  + (-((ax*mD + mN + az*mN)*my) + ay*(mD*mx + mN*mz))*qz
        q_new[1] =     ((mD - az*mD - ax*mN)*my             + ay*(1 + mN*mx + mD*mz))*qw + (ay*mD*my - (-1 + az)*(1 + mN*mx - mD*mz) + ax*(mD*mx + mN*mz))*qx                  + ((ax*mD + mN - az*mN)*my + ay*(-(mD*mx) + mN*mz))*qy + (-(ay*mN*my) + ax*(1 - mN*mx + mD*mz) - (-1 + az)*(mD*mx + mN*mz))*qz
        q_new[2] = (-(ay*mN*my) - ax*(1 + mN*mx + mD*mz) + (-1 + az)*(mD*mx - mN*mz))*qw               + ((-(ax*mD) + mN - az*mN)*my + ay*(mD*mx + mN*mz))*qx   + (ay*mD*my + (-1 + az)*(-1 + mN*mx + mD*mz) + ax*(mD*mx - mN*mz))*qy                 + ((mD - az*mD + ax*mN)*my + ay*(1 - mN*mx + mD*mz))*qz
        q_new[3] = ax*(qx + mN*mx*qx + mN*my*qy + mN*mz*qz + mD*(my*qw - mz*qx + mx*qz)) + (1 + az)*(mD*mx*qx + mD*my*qy + qz + mD*mz*qz - mN*(my*qw - mz*qx + mx*qz)) + ay*(mN*mz*qw + mN*my*qx + qy - mN*mx*qy - mD*(mx*qw + mz*qy - my*qz))
        q_new = 0.25 * q_new / np.linalg.norm(q_new)
        # Build Jacobian matrix (eq. 27)
        J = np.zeros((4, 6))
        J[0, 0]= -qy - mN*(mz*qw + my*qx - mx*qy) + mD*(mx*qw + mz*qy - my*qz)
        J[0, 1]=  qx + mN*mx*qx + mN*my*qy + mN*mz*qz + mD*(my*qw - mz*qx + mx*qz)
        J[0, 2]=  qw + mN*mx*qw + mD*mz*qw + mD*my*qx - mD*mx*qy + mN*mz*qy - mN*my*qz
        J[0, 3]= (ax*mD + mN + az*mN)*qw + ay*mN*qx + (-((1 + az)*mD) + ax*mN)*qy + ay*mD*qz
        J[0, 4]= ay*mD*qw + (mD + az*mD - ax*mN)*qx + ay*mN*qy - (ax*mD + mN + az*mN)*qz
        J[0, 5]= mD*(qw + az*qw - ay*qx + ax*qy) + mN*(-(ax*qw) + qy + az*qy + ay*qz)
        J[1, 0]= qz - mN*(my*qw - mz*qx + mx*qz) + mD*(mx*qx + my*qy + mz*qz)
        J[1, 1]= qw + mN*mx*qw + mD*mz*qw + mD*my*qx - mD*mx*qy + mN*mz*qy - mN*my*qz
        J[1, 2]= -((1 + mN*mx)*qx) - mD*(my*qw - mz*qx + mx*qz) - mN*(my*qy + mz*qz)
        J[1, 3]= ay*(mN*qw - mD*qy) - (-1 + az)*(mN*qx + mD*qz) + ax*(mD*qx - mN*qz)
        J[1, 4]= mD*(qw - az*qw + ay*qx + ax*qy) - mN*(ax*qw + (-1 + az)*qy + ay*qz)
        J[1, 5]= ay*(mD*qw + mN*qy) + mD*((-1 + az)*qx + ax*qz) + mN*(ax*qx + qz - az*qz)
        J[2, 0]= -((1 + mN*mx + mD*mz)*qw) - mD*my*qx + mD*mx*qy - mN*mz*qy + mN*my*qz
        J[2, 1]= qz - mN*(my*qw - mz*qx + mx*qz) + mD*(mx*qx + my*qy + mz*qz)
        J[2, 2]= -qy - mN*(mz*qw + my*qx - mx*qy) + mD*(mx*qw + mz*qy - my*qz)
        J[2, 3]= mD*((-1 + az)*qw + ay*qx + ax*qy) - mN*(ax*qw + qy - az*qy + ay*qz)
        J[2, 4]= ay*(-(mN*qw) + mD*qy) - (-1 + az)*(mN*qx + mD*qz) + ax*(-(mD*qx) + mN*qz)
        J[2, 5]= mN*(qw - az*qw + ay*qx) - ax*(mD*qw + mN*qy) + mD*((-1 + az)*qy + ay*qz)
        J[3, 0]= qx + mN*mx*qx + mN*my*qy + mN*mz*qz + mD*(my*qw - mz*qx + mx*qz)
        J[3, 1]= qy + mN*(mz*qw + my*qx - mx*qy) - mD*(mx*qw + mz*qy - my*qz)
        J[3, 2]= qz - mN*(my*qw - mz*qx + mx*qz) + mD*(mx*qx + my*qy + mz*qz)
        J[3, 3]= -(ay*(mD*qw + mN*qy)) + ax*(mN*qx + mD*qz) + (1 + az)*(mD*qx - mN*qz)
        J[3, 4]= (1 + az)*(-(mN*qw) + mD*qy) + ax*(mD*qw + mN*qy) + ay*(mN*qx + mD*qz)
        J[3, 5]= ay*(mN*qw - mD*qy) + (1 + az)*(mN*qx + mD*qz) + ax*(-(mD*qx) + mN*qz)
        # J = 0.25 * J
        return q_new, J

    def _compute_all(self) -> np.ndarray:
        """
        Compute all steps of the FKF algorithm.

        Returns
        -------
        Q : numpy.ndarray
            Copmuted quaternions.

        """
        _assert_numerical_iterable(self.gyr, 'Angular velocity vector')
        _assert_numerical_iterable(self.acc, 'Gravitational acceleration vector')
        _assert_numerical_iterable(self.mag, 'Geomagnetic field vector')
        gyr = np.atleast_2d(self.gyr)
        acc = np.atleast_2d(self.acc)
        mag = np.atleast_2d(self.mag)
        if gyr.shape != acc.shape or gyr.shape != mag.shape:
            raise ValueError("All input arrays must have the same shape.")
        num_samples = gyr.shape[0]
        Sigma_g = self.sigma_g * np.identity(3)
        Sigma_am = np.diag([self.sigma_a]*3 + [self.sigma_m]*3)     # (eq. 28)
        Q = np.zeros((num_samples, 4))
        # Initial quaternion from the accelerometer and magnetometer
        Q[0] = ecompass(acc[0], mag[0], frame='NED', representation='quaternion')
        for t in range(1, num_samples):
            q_ = Q[t-1]                                             # Previous quaternion
            # PROCESS MODEL
            omega4 = self.Omega4(gyr[t])                            # Skew symmetric matrix (eq. 20)
            Phi = np.identity(4) + 0.5 * self.Dt * omega4           # State transition matrix (eq. 21)
            Xi = np.array([[ q_[1],  q_[2],  q_[3]],
                           [-q_[0], -q_[3], -q_[2]],
                           [ q_[2], -q_[0], -q_[1]],
                           [-q_[2],  q_[1], -q_[0]]])               # (eq. 24)
            Sigma_eps = (self.Dt/2.0)**2 * Xi @ Sigma_g @ Xi.transpose()   # Process noise covariance (eq. 23)
            # MEASUREMENT MODEL
            qy, J = self.measurement_quaternion_acc_mag(q_, acc[t], mag[t])
            Sigma_v = J @ Sigma_am @ J.transpose()                  # Measurement quaternion's covariance (eq. 26)
            # Kalman Update
            q, self.Pk = self.kalman_update(q_, qy, self.Pk, Phi, Sigma_eps, Sigma_v)
            Q[t] = q
        return Q

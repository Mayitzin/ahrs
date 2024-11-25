# -*- coding: utf-8 -*-
"""
Fast Kalman Filter
==================

Quaternion-based Fast Kalman Filter algorithm for orientation estimation
:cite:p:`guo2017`.

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

    def kalman_update(self, xk_1: np.ndarray, yk: np.ndarray, Pk_1: np.ndarray, Phi_k: np.ndarray, Xi_k: np.ndarray, Eps_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kalman update step.

        Parameters
        ----------
        xk_1 : numpy.ndarray
            Previous state vector.
        yk : numpy.ndarray
            Measurement vector.
        Pk_1 : numpy.ndarray
            Previous covariance matrix.
        Phi_k : numpy.ndarray
            State transition matrix.
        Xi_k : numpy.ndarray
            Process noise covariance matrix.
        Eps_k : numpy.ndarray
            Measurement noise covariance matrix.

        Returns
        -------
        xk : numpy.ndarray
            Updated state vector.
        Pk : numpy.ndarray
            Updated covariance matrix.

        """
        x_ = Phi_k @ xk_1
        Pk_ = Phi_k @ (Pk_1 @ Phi_k.transpose()) + Xi_k
        Gk = Pk_ @ (np.linalg.inv(Pk_ + Eps_k))
        Pk = (np.identity(4) - Gk) @ Pk_
        xk = x_ + Gk @ (yk - x_)
        return xk, Pk

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
        q_new = q_new / np.linalg.norm(q_new)
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
        J = 0.25 * J
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
        for i in range(1, num_samples):
            q_ = Q[i-1]                                             # Previous quaternion
            # PROCESS MODEL
            omega4 = self.Omega4(gyr[i])                            # Skew symmetric matrix (eq. 20)
            Phi = np.identity(4) + 0.5 * self.Dt * omega4           # State transition matrix (eq. 15)
            Dk = np.array([[ q_[1],  q_[2],  q_[3]],
                           [-q_[0], -q_[3], -q_[2]],
                           [ q_[2], -q_[0], -q_[1]],
                           [-q_[2],  q_[1], -q_[0]]])               # (eq. 24)
            Xi = self.Dt**2 * 0.25 * Dk @ Sigma_g @ Dk.transpose()  # Process noise covariance (eq. 23)
            # MEASUREMENT MODEL
            qy, J = self.measurement_quaternion_acc_mag(q_, acc[i], mag[i])
            Eps = J @ Sigma_am @ J.transpose()                      # Measurement quaternion's covariance (eq. 26)
            q, self.Pk = self.kalman_update(q_, qy, self.Pk, Phi, Xi, Eps)
            Q[i] = q
        return Q

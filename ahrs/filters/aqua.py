# -*- coding: utf-8 -*-
"""
Algebraic Quaternion Algorithm
==============================

Roberto Valenti's Algebraic Quaterion Algorithm (AQUA) [Valenti2015]_ estimates
a quaternion with the algebraic solution of a system from inertial/magnetic
observations.

AQUA computes the "tilt" quaternion and the "heading" quaternion separately in
two sub-parts. This avoids the impact of the magnetic disturbances on the roll
and pitch components of the orientation.

It uses a complementary filter that fuses together gyroscope data with
accelerometer and magnetic field readings. The correction part of the filter is
based on the independently estimated quaternions and works for both IMU
(Inertial Measurement Unit) and MARG (Magnetic, Angular Rate, and Gravity)
sensors [Valenti2016]_.

References
----------
.. [Valenti2015] Valenti, R.G.; Dryanovski, I.; Xiao, J. Keeping a Good
    Attitude: A Quaternion-Based Orientation Filter for IMUs and MARGs. Sensors
    2015, 15, 19302-19330.
    (https://res.mdpi.com/sensors/sensors-15-19302/article_deploy/sensors-15-19302.pdf)
.. [Valenti2016] R. G. Valenti, I. Dryanovski and J. Xiao, "A Linear Kalman
    Filter for MARG Orientation Estimation Using the Algebraic Quaternion
    Algorithm," in IEEE Transactions on Instrumentation and Measurement, vol.
    65, no. 2, pp. 467-481, 2016.
    (https://ieeexplore.ieee.org/document/7345567)

"""

import numpy as np
from ..common.orientation import q_prod, q2R
from ..common.constants import MUNICH_LATITUDE, MUNICH_HEIGHT

# Reference Observations in Munich, Germany
from ..utils.wgs84 import WGS
GRAVITY = WGS().normal_gravity(MUNICH_LATITUDE, MUNICH_HEIGHT)

def slerp_I(q: np.ndarray, ratio: float, t: float) -> np.ndarray:
    """
    Quaternion Interpolation with Identity

    Interpolate a given quaternion with identity quaternion
    :math:`\\mathbf{q}=\\begin{bmatrix}1 & 0 & 0 & 0\\end{bmatrix}` to
    scale it to closest versor.

    The interpolation can be with either LERP (Linear) or SLERP (Spherical
    Linear) methods, decided by a threshold value :math:`t`, which lies
    between ``0.0`` and ``1.0``.

    .. math::
        \\mathrm{method} = \\left\\{
        \\begin{array}{ll}
            \\mathrm{LERP} & \\: q_w > t \\\\
            \\mathrm{SLERP} & \\: \\mathrm{otherwise}
        \\end{array}
        \\right.

    For LERP, a simple equation is implemented:

    .. math::

        \\hat{\\mathbf{q}} = (1-\\alpha)\\mathbf{q}_I + \\alpha\\Delta \\mathbf{q}

    where :math:`\\alpha\\in [0, 1]` is the gain characterizing the cut-off
    frequency of the filter. It basically decides how "close" to the given
    quaternion or to the identity quaternion the interpolation is.

    If the scalar part :math:`q_w` of the given quaternion is below the
    threshold :math:`t`, SLERP is used:

    .. math::
        \\hat{\\mathbf{q}} = \\frac{\\sin([1-\\alpha]\\Omega)}{\\sin\\Omega} \\mathbf{q}_I + \\frac{\\sin(\\alpha\\Omega)}{\\sin\\Omega} \\mathbf{q}

    where :math:`\\Omega=\\arccos(q_w)` is the subtended arc between the
    quaternions.

    Parameters
    ----------
    q : numpy.array
        Quaternion to inerpolate with.
    ratio : float
        Gain characterizing the cut-off frequency of the filter.
    t : float
        Threshold deciding interpolation method. LERP when q_w>t or SLERP

    Returns
    -------
    q : numpy.array
        Interpolated quaternion
    """
    q_I = np.array([1.0, 0.0, 0.0, 0.0])
    if q[0]>t:  # LERP
        q = (1.0-ratio)*q_I + ratio*q   # (eq. 50)
    else:       # SLERP
        angle = np.arccos(q[0])
        q = q_I*np.sin(abs(1.0-ratio)*angle)/np.sin(angle) + q*np.sin(ratio*angle)/np.sin(angle)    # (eq. 52)
    q /= np.linalg.norm(q)              # (eq. 51)
    return q

def adaptive_gain(a: float, a_norm: float, t1: float = 0.1, t2: float = 0.2, g: float = GRAVITY) -> float:
    """
    Adaptive filter gain factor

    The estimated gain :math:`\\alpha` is dependent on the gain factor
    :math:`f` determined by the magnitude error :math:`e_m`:

    .. math::
        \\alpha = a f(e_m)

    where the magnitude error is defined by the measured acceleration
    :math:`\\mathbf{a}=\\begin{bmatrix}a_x & a_y & a_z\\end{bmatrix}^T` and the
    reference gravity :math:`g\\approx 9.809196 \\frac{m}{s^2}`:

    .. math::
        e_m = \\frac{|\\|\\mathbf{a}\\|-g|}{g}

    The gain factor is constant and equal to 1 when the magnitude of the
    nongravitational acceleration is not high enough to overcome gravity.
    If nongravitational acceleration rises and :math:`e_m` exceeds the
    first threshold, the gain factor :math:`f` decreases linearly with the
    increase of the magnitude until reaching zero at the second threshold
    and above it.

    Empirically, both thresholds have been defined at ``0.1`` and ``0.2``,
    respectively. They can be, however, changed by setting the values of
    input parameters ``t1`` and ``t2``.

    Parameters
    ----------
    a : float
        Constant gain yielding best results in static conditions.
    a_norm : float
        Norm of measured local acceleration vector.
    t1 : float, default: 0.1
        First threshold
    t2 : float, default: 0.2
        Second threshold
    g : float, default: 9.809196
        Reference gravitational acceleration in m/s^2. The estimated gravity in
        Munich, Germany (``9.809196``) is used as default reference value.

    Returns
    -------
    alpha : float
        Gain factor

    Examples
    --------
    >>> alpha = adaptive_gain(a, 9.71)
    """
    em = abs(a_norm-GRAVITY)/GRAVITY      # Magnitude error (eq. 60)
    f = 0.0
    if e1<em<t2:
        f = (t2-em)/t1
    if em<=e1:
        f = 1.0
    return f*a                          # Filtering gain (eq. 61)

class AQUA:
    """
    Algebraic Quaternion Algorithm

    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in g
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT
    frequency : float, default: 100.0
        Sampling frequency in Herz
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if ``frequency`` value is given.
    alpha : float, default: 0.01
        Gain characterizing cut-off frequency for accelerometer quaternion
    beta : float, default: 0.01
        Gain characterizing cut-off frequency for magnetometer quaternion
    threshold : float, default: 0.9
        Threshold to discriminate between LERP and SLERP interpolation
    adaptive : bool, default: False
        Whether to use an adaptive gain for each sample
    q0 : numpy.ndarray, default: None
        Initial orientation, as a versor (normalized quaternion).

    Attributes
    ----------
    gyr : numpy.ndarray
        N-by-3 array with N gyroscope samples.
    acc : numpy.ndarra
        N-by-3 array with N accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N magnetometer samples.
    frequency : float
        Sampling frequency in Herz
    Dt : float
        Sampling step in seconds. Inverse of sampling frequency.
    alpha : float
        Gain characterizing cut-off frequency for accelerometer quaternion.
    beta : float
        Gain characterizing cut-off frequency for magnetometer quaternion.
    threshold : float
        Threshold to discern between LERP and SLERP interpolation.
    adaptive : bool
        Flag indicating use of adaptive gain.
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion).

    """
    def __init__(self, gyr: np.ndarray = None, acc: np.ndarray = None, mag: np.ndarray = None, **kw):
        self.gyr = gyr
        self.acc = acc
        self.mag = mag
        self.Q = None
        self.frequency = kw.get('frequency', 100.0)
        self.Dt = kw.get('Dt', 1.0/self.frequency)
        self.alpha = kw.get('alpha', 0.01)
        self.beta = kw.get('beta', 0.01)
        self.threshold = kw.get('threshold', 0.9)
        self.adaptive = kw.get('adaptive', False)
        self.q0 = kw.get('q0')
        if self.acc is not None and self.gyr is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """Estimate all quaternions with given sensor values"""
        if self.acc.shape != self.gyr.shape:
            raise ValueError("acc and gyr are not the same size")
        num_samples = len(self.gyr)
        Q = np.zeros((num_samples, 4))
        # Compute with IMU architecture
        if self.mag is None:
            Q[0] = self.init_q(self.acc[0]) if self.q0 is None else self.q0.copy()
            for t in range(1, num_samples):
                Q[t] = self.updateIMU(Q[t-1], self.gyr[t], self.acc[t])
            return Q
        # Compute with MARG architecture
        if self.mag.shape != self.gyr.shape:
            raise ValueError("mag and gyr are not the same size")
        Q[0] = self.init_q(self.acc[0], self.mag[0]) if self.q0 is None else self.q0.copy()
        for t in range(1, num_samples):
            Q[t] = self.updateMARG(Q[t-1], self.gyr[t], self.acc[t], self.mag[t])
        return Q

    def init_q(self, acc: np.ndarray, mag: np.ndarray = None) -> np.ndarray:
        """
        Quaternion from Earth-Field Observations

        Algebraic estimation of a quaternion as a function of an observation of
        the Earth's gravitational and magnetic fields.

        It decomposes the quaternion :math:`\\mathbf{q}` into two auxiliary
        quaternions :math:`\\mathbf{q}_{\\mathrm{acc}}` and
        :math:`\\mathbf{q}_{\\mathrm{mag}}`, such that:

        .. math::
            \\mathbf{q} = \\mathbf{q}_{\\mathrm{acc}}\\mathbf{q}_{\\mathrm{mag}}

        Parameters
        ----------
        acc : numpy.ndarray, default: None
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray, default: None
            Sample of tri-axial Magnetometer in mT

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.
        """
        ax, ay, az = acc/np.linalg.norm(acc)
        # Quaternion from Accelerometer Readings (eq. 25)
        if az>=0:
            q_acc = np.array([np.sqrt((az+1)/2), -ay/np.sqrt(2*(1-ax)), ax/np.sqrt(2*(az+1)), 0.0])
        else:
            q_acc = np.array([-ay/np.sqrt(2*(1-az)), np.sqrt((1-az)/2.0), 0.0, ax/np.sqrt(2*(1-az))])
        q_acc /= np.linalg.norm(q_acc)
        # m_norm = np.linalg.norm(mag)
        if mag is not None and not (np.linalg.norm(mag)>0):
            lx, ly, lz = q2R(q_acc).T@(mag/np.linalg.norm(mag)) # (eq. 26)
            Gamma = lx**2 + ly**2                               # (eq. 28)
            # Quaternion from Magnetometer Readings (eq. 35)
            if lx>=0:
                q_mag = np.array([np.sqrt(Gamma+lx*np.sqrt(Gamma))/np.sqrt(2*Gamma), 0.0, 0.0, ly/np.sqrt(2)*np.sqrt(Gamma+lx*np.sqrt(Gamma))])
            else:
                q_mag = np.array([ly/np.sqrt(2)*np.sqrt(Gamma-lx*np.sqrt(Gamma)), 0.0, 0.0, np.sqrt(Gamma-lx*np.sqrt(Gamma))/np.sqrt(2*Gamma)])
            # Generalized Quaternion Orientation (eq. 36)
            q = q_prod(q_acc, q_mag)
            return q/np.linalg.norm(q)
        return q_acc

    def updateIMU(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """Quaternion Estimation with a IMU architecture.

        The estimation is made in two steps: a _prediction_ is done with the
        angular rate (gyroscope) to integrate and estimate the current
        orientation; then a _correction_ step uses the measured accelerometer
        to infer the expected gravity vector and use it to correct the
        predicted quaternion.

        If the gyroscope data is invalid, it returns the given a-priori
        quaternion. Secondly, if the accelerometer data is invalid the
        predicted quaternion (using gyroscopes) is returned.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        """
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        # PREDICTION
        qDot = -0.5*q_prod([0, *gyr], q)                    # Quaternion derivative (eq. 38)
        qInt = q + qDot*self.Dt                             # Quaternion integration (eq. 42)
        qInt /= np.linalg.norm(qInt)
        # CORRECTION
        a_norm = np.linalg.norm(acc)
        if not a_norm>0:
            return qInt
        a = acc/a_norm
        gx, gy, gz = q2R(qInt).T@a                          # Predicted gravity (eq. 44)
        q_acc = np.array([np.sqrt((gz+1)/2.0), -gy/np.sqrt(2.0*(gz+1)), gx/np.sqrt(2.0*(gz+1)), 0.0])     # Delta Quaternion (eq. 47)
        if self.adaptive:
            self.alpha = self.adaptive_gain(self.alpha, a_norm)
        q_acc = slerp_I(q_acc, self.alpha, self.threshold)
        q_prime = q_prod(qInt, q_acc)                       # (eq. 53)
        return q_prime/np.linalg.norm(q_prime)

    def updateMARG(self, q: np.ndarray, gyr: np.ndarray, acc: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """Quaternion Estimation with a MARG architecture.

        The estimation is made in two steps: a _prediction_ is done with the
        angular rate (gyroscope) to integrate and estimate the current
        orientation; then a _correction_ step uses the measured accelerometer
        and magnetic field to infer the expected geodetic values. Its
        divergence is used to correct the predicted quaternion.

        If the gyroscope data is invalid, it returns the given a-priori
        quaternion. Secondly, if the accelerometer data is invalid the
        predicted quaternion (using gyroscopes) is returned. Finally, if the
        magnetometer measurements are invalid, returns a quaternion corrected
        by the accelerometer only.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Sample of tri-axial Gyroscope in rad/s.
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2
        mag : numpy.ndarray
            Sample of tri-axial Magnetometer in T

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        """
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        # PREDICTION
        qDot = -0.5*q_prod([0, *gyr], q)                    # Quaternion derivative (eq. 38)
        qInt = q + qDot*self.Dt                             # Quaternion integration (eq. 42)
        qInt /= np.linalg.norm(qInt)
        # CORRECTION
        a_norm = np.linalg.norm(acc)
        if not a_norm>0:
            return qInt
        a = acc/a_norm
        gx, gy, gz = q2R(qInt).T@a                          # Predicted gravity (eq. 44)
        # Accelerometer-Based Quaternion
        q_acc = np.array([np.sqrt((gz+1)/2.0), -gy/np.sqrt(2.0*(gz+1)), gx/np.sqrt(2.0*(gz+1)), 0.0])     # Delta Quaternion (eq. 47)
        if self.adaptive:
            self.alpha = self.adaptive_gain(self.alpha, a_norm)
        q_acc = slerp_I(q_acc, self.alpha, self.threshold)
        q_prime = q_prod(qInt, q_acc)                       # (eq. 53)
        q_prime /= np.linalg.norm(q_prime)
        # Magnetometer-Based Quaternion
        m_norm = np.linalg.norm(mag)
        if not m_norm>0:
            return q_prime
        lx, ly, lz = q2R(q_prime).T@(mag/m_norm)            # World frame magnetic vector (eq. 54)
        Gamma = lx**2 + ly**2                               # (eq. 28)
        q_mag = np.array([np.sqrt(Gamma+lx*np.sqrt(Gamma))/np.sqrt(2*Gamma), 0.0, 0.0, ly/np.sqrt(2*(Gamma+lx*np.sqrt(Gamma)))])    # (eq. 58)
        q_mag = slerp_I(q_mag, self.beta, self.threshold)
        # Generalized Quaternion
        q = q_prod(q_prime, q_mag)                          # (eq. 59)
        return q/np.linalg.norm(q)

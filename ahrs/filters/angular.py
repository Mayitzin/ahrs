# -*- coding: utf-8 -*-
"""
Attitude from angular rate
==========================

A quaternion is updated via integration of angular rate measurements of a
gyroscope. The easiest way to update the quaternions is by integrating the
differential equation for a local rotation rate [Sola]_.

In a kinematic system, the angular velocity :math:`\\mathbf{\\omega}` of a
rigid body at any instantaneous time is described with respect to a fixed frame
coinciding instantaneously with its body frame [#]_. Thus, this angular
velocity is *in terms of* the body frame [Jia]_.

The angular rates are measured by local sensors, providing local measurements
:math:`\\omega_t` at discrete times. The differential equation of a quaternion
at time :math:`t` is:

.. math::
    \\dot{\\mathbf{q}}_t = \\frac{1}{2}\\mathbf{q}_t\\mathbf{\\omega}_t

where :math:`\\omega_t` is the measured angular velocity, in radians per
second, at the instant :math:`t`, and represented as a pure quaternion
:math:`\\mathbf{\\omega}=\\begin{bmatrix}0 & \\omega_x & \\omega_y & \\omega_z\\end{bmatrix}`.

At :math:`t+\\Delta t`, the orientation is described by :math:`\\mathbf{q}(t+\\Delta t)`,
after a rotation was performed during :math:`\\Delta t` on the frame. The
simplest assumption is that :math:`\\omega` is constant over the integration
period :math:`\\Delta t`. Thus, making the differential equation **time
invariant** [Trawny]_.

The quickest integration is done performing a `right Riemann sum
<https://en.wikipedia.org/wiki/Riemann_sum#Right_Riemann_sum>`_ with the
computed derivative:

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}_{t+1} &=& \\mathbf{q}_t + \\dot{\\mathbf{q}}_t\\Delta t \\\\
    &=& \\mathbf{q}_t + \\frac{1}{2}\\mathbf{q}_t\\mathbf{\\omega}_t\\Delta t
    \\end{array}

The resulting quaternion must be normalized to operate as a
`versor <https://en.wikipedia.org/wiki/Versor>`_, and to represent a valid
orientation.

.. math::
    \\mathbf{q}_{t+1} \\gets \\frac{\\mathbf{q}_{t+1}}{\\|\\mathbf{q}_{t+1}\\|}

This is the simplest approximation to integrate the quaternions over time. That
will, naturally, yield a drift due to faulty sensors (no sensor is perfect) and
the truncation of the numerical solution.

Integration methods based on the Taylor series and Runge-Kutta can be employed
to increase the accuracy, and are shown to be more effective. See [Sola]_ and
[Zhao]_ for a comparison of the different methods, their accuracy, and their
computational load.

Footnotes
---------
.. [#] At a different time instant, the angular velocity is measured in a
    different fixed frame due to the rotation of the body frame.

References
----------
.. [Jia] Yan-Bin Jia. Quaternions. 2018.
    (http://web.cs.iastate.edu/~cs577/handouts/quaternion.pdf)
.. [Sola] Solà, Joan. Quaternion kinematics for the error-state Kalman Filter.
    October 12, 2017.
    (http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf)
.. [Zhao] F. Zhao and B.G.M. van Wachem. A novel Quaternion integration
    approach for describing the behaviour of non-spherical particles.
    (https://link.springer.com/content/pdf/10.1007/s00707-013-0914-2.pdf)
.. [Trawny] N. Trawny and S.I. Roumeliotis. Indirect Kalman Filter for 3D
    Attitude Estimation. Technical Report No. 2005-002. March 2005.
    (http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf)

"""

import numpy as np
from ..common.orientation import q_prod

class AngularRate:
    """
    Quaternion update by integrating angular velocity

    Parameters
    ----------
    gyr : numpy.ndarray, default: None
        N-by-3 array with measurements of angular velocity in rad/s.
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion).
    frequency : float, default: 100.0
        Sampling frequency in Herz.
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if ``frequency`` value is given.

    Attributes
    ----------
    gyr : numpy.ndarray
        N-by-3 array with N gyroscope samples.
    Q : numpy.ndarray
        Estimated quaternions.

    Examples
    --------
    >>> gyro_data.shape             # NumPy arrays with gyroscope data in rad/s
    (1000, 3)
    >>> from ahrs.filters import AngularRate
    >>> angular_rate = AngularRate(gyr=gyro_data)
    >>> angular_rate.Q
    array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
           [ 9.99999993e-01,  2.36511228e-06, -1.12991334e-04,  4.28771947e-05],
           [ 9.99999967e-01,  1.77775173e-05, -2.43529706e-04,  8.33144162e-05],
           ...,
           [-0.92576208, -0.23633121,  0.19738534, -0.2194337 ],
           [-0.92547793, -0.23388968,  0.19889139, -0.22187479],
           [-0.92504595, -0.23174096,  0.20086376, -0.22414251]])
    >>> angular_rate.Q.shape        # Estimated attitudes as Quaternions
    (1000, 4)

    The estimation of each attitude is built upon the previous attitude. This
    estimator sets the initial attitude equal to the unit quaternion
    ``[1.0, 0.0, 0.0, 0.0]`` by default, because we cannot obtain the first
    orientation with gyroscopes only.

    We can use the class :class:`Tilt` to estimate the initial attitude with a
    simple measurement of a tri-axial accelerometer:

    >>> from ahrs.filter import Tilt
    >>> tilt = Tilt()
    >>> q_initial = tilt.estimate(acc=acc_sample)  # One tridimensional sample suffices
    >>> angular_rate = AngularRate(gyr=gyro_data, q0=q_initial)
    >>> angular_rate.Q
    array([[ 0.77547502,  0.6312126 ,  0.01121595, -0.00912944],
           [ 0.77547518,  0.63121388,  0.01110125, -0.00916754],
           [ 0.77546726,  0.63122508,  0.01097435, -0.00921875],
           ...,
           [-0.92576208, -0.23633121,  0.19738534, -0.2194337 ],
           [-0.92547793, -0.23388968,  0.19889139, -0.22187479],
           [-0.92504595, -0.23174096,  0.20086376, -0.22414251]])

    The :class:`Tilt` can also use a magnetometer to improve the estimation
    with the heading orientation.

    >>> q_initial = tilt.estimate(acc=acc_sample, mag=mag_sample)
    >>> angular_rate = AngularRate(gyr=gyro_data, q0=q_initial)
    >>> angular_rate.Q
    array([[ 0.66475674,  0.55050651, -0.30902706, -0.39942875],
           [ 0.66473764,  0.5504497 , -0.30912672, -0.39946172],
           [ 0.66470495,  0.55039529, -0.30924191, -0.39950193],
           ...,
           [-0.90988476, -0.10433118,  0.28970402,  0.27802214],
           [-0.91087203, -0.1014633 ,  0.28977124,  0.2757716 ],
           [-0.91164416, -0.09861271,  0.2903888 ,  0.27359606]])

    """
    def __init__(self, gyr: np.ndarray = None, q0: np.ndarray = None, **kw):
        self.gyr = gyr
        self.q0 = q0
        self.frequency = kw.get('frequency', 100.0)
        self.Dt = kw.get('Dt', 1.0/self.frequency)
        if self.gyr is not None:
            self.Q = self._compute_all()

    def _compute_all(self):
        """Estimate all quaternions with given sensor values"""
        num_samples = len(self.gyr)
        Q = np.zeros((num_samples, 4))
        Q[0] = np.array([1.0, 0.0, 0.0, 0.0]) if self.q0 is None else self.q0.copy()
        for t in range(1, num_samples):
            Q[t] = self.update(Q[t-1], self.gyr[t])
        return Q

    def update(self, q: np.ndarray, gyr: np.ndarray) -> np.ndarray:
        """Update the quaternion estimation

        Estimate quaternion :math:`\\mathbf{q}_{t+1}` from given a-priori
        quaternion :math:`\\mathbf{q}_t` with a given angular rate measurement
        :math:`\\mathbf{\\omega}`:

        .. math::
            \\mathbf{q}_{t+1} \\gets \\mathbf{q}_t + \\frac{1}{2}\\mathbf{q}_t\\mathbf{\\omega}_t\\Delta t

        where :math:`\\|\\mathbf{q}_{t+1}\\|=1` means it is a versor.

        Parameters
        ----------
        q : numpy.ndarray
            A-priori quaternion.
        gyr : numpy.ndarray
            Array with triaxial measurements of angular velocity in rad/s

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        Examples
        --------
        >>> from ahrs.filters import AngularRate
        >>> gyro_data.shape
        (1000, 3)
        >>> num_samples = gyro_data.shape[0]
        >>> Q = np.zeros((num_samples, 4))      # Allocation of quaternions
        >>> Q[0] = [1.0, 0.0, 0.0, 0.0]         # Initial attitude as a quaternion
        >>> angular_rate = AngularRate()
        >>> for t in range(1, num_samples):
        ...     Q[t] = angular_rate.update(Q[t-1], gyro_data[t])
        ...
        >>> Q
        array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
               [ 9.99999993e-01,  2.36511228e-06, -1.12991334e-04,  4.28771947e-05],
               [ 9.99999967e-01,  1.77775173e-05, -2.43529706e-04,  8.33144162e-05],
               ...,
               [-0.92576208, -0.23633121,  0.19738534, -0.2194337 ],
               [-0.92547793, -0.23388968,  0.19889139, -0.22187479],
               [-0.92504595, -0.23174096,  0.20086376, -0.22414251]])

        """
        q = np.copy(q)
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        qDot = 0.5*q_prod(q, [0, *gyr])     # Quaternion derivative
        q += qDot*self.Dt
        return q/np.linalg.norm(q)


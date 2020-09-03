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
        N-by-3 array with measurements of angular velocity in rad/s
    q0 : numpy.ndarray
        Initial orientation, as a versor (normalized quaternion).
    frequency : float, default: 100.0
        Sampling frequency in *Herz*.
    Dt : float, default: 0.01
        Sampling step in seconds. Inverse of sampling frequency. Not required
        if ``frequency`` value is given.

    Attributes
    ----------
    gyr : numpy.ndarray
        N-by-3 array with N gyroscope samples.
    Q : numpy.ndarray
        Estimated quaternions.

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

        """
        if gyr is None or not np.linalg.norm(gyr)>0:
            return q
        qDot = 0.5*q_prod(q, [0, *gyr])     # Quaternion derivative
        q += qDot*self.Dt
        return q/np.linalg.norm(q)


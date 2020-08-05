# -*- coding: utf-8 -*-
"""
Attitude from gravity (Tilt)
============================

Attitude estimation via gravity acceleration measurements.

The simplest approach using `geometric quadrants <https://en.wikipedia.org/wiki/Quadrant_(plane_geometry)>`_
to find a vector in a 3D space offers the simplest solution using the gravity
vector.

Although some methods use ``arctan`` to estimate the angles [ST-AN4509]_ [AD-AN1057]_,
it is preferred to use ``arctan2`` to explore all quadrants estimating the tilt
angles.

First, we normalize the gravity vector, so that it has magnitude equal to 1.
Then we the angles to the main axes with `arctan2 <https://en.wikipedia.org/wiki/Atan2>`_
[FS-AN3461]_ [Trimpe]_:

.. math::
    \\begin{array}{ll}
    \\theta &= \\mathrm{arctan2}(a_y, a_z) \\\\
    \\psi &= \\mathrm{arctan2}\\big(-a_z, \\sqrt{a_y^2+a_z^2}\\big)
    \\end{array}

where :math:`\\beta` is the **roll** angle, :math:`\\psi` is the **pitch**
angle, and :math:`\\mathbf{a}=\\begin{bmatrix}a_x & a_y & a_z\\end{bmatrix}^T`
is the normalized vector of measured accelerations, which means
:math:`\\|\\mathbf{a}\\|=1`.

The attitude in terms of these two angles is called the **tilt**.

**Heading angle**

The heading angle, a.k.a. **yaw**, cannot be obtained from the measured
acceleration, and a different reference shall be used to obtain it. The most
common is the use of the geomagnetic information, in other words, `Earth's
magnetic field <https://en.wikipedia.org/wiki/Earth%27s_magnetic_field>`_.

With the pitch and roll angles estimated from the accelerometer, we can rotate
a magnetometer reading :math:`\\mathbf{m}`, and estimate the yaw angle
:math:`\\psi` to update the orientation.

The vector :math:`\\mathbf{b}=\\begin{bmatrix}b_x & b_y & b_z\\end{bmatrix}^T`
represents the magnetometer readings after *rotating them back* to the plane,
where :math:`\\theta = \\phi = 0`.

.. math::
    \\begin{bmatrix}b_x \\\\ b_y \\\\ b_z\\end{bmatrix} =
    \\begin{bmatrix}
        m_x\\cos\\theta + m_y\\sin\\theta\\sin\\phi + m_z\\sin\\theta\\cos\\phi \\\\
        m_y\\cos\\phi - m_z\\sin\\phi \\\\
        -m_x\\sin\\theta + m_y\\cos\\theta\\sin\\phi + m_z\\cos\\theta\\cos\\phi
    \\end{bmatrix}

Where :math:`\\mathbf{m}=\\begin{bmatrix}m_x & m_y & m_z\\end{bmatrix}^T` is
the normalized vector of the measured magnetometic field, which means
:math:`\\|\\mathbf{m}\\|=1`.

The yaw angle :math:`\\psi` is the tilt-compensated heading angle relative to
magnetic North, computed as [FS-AN4248]_:

.. math::
    \\begin{array}{ll}
    \\psi &= \\mathrm{arctan2}(-b_y, b_x) \\\\
    &= \\mathrm{arctan2}\\big(m_z\\sin\\phi - m_y\\cos\\phi, \\; m_x\\cos\\theta + \\sin\\theta(m_y\\sin\\phi + m_z\\cos\\phi)\\big)
    \\end{array}

Finally, we transform the roll-pitch-yaw angles to a quaternion representation:

.. math::
    \\mathbf{q} =
    \\begin{bmatrix}
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) + \\sin\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\sin\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) - \\cos\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) + \\sin\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) - \\sin\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big)
    \\end{bmatrix}

Setting the property ``as_angles`` to ``True`` will avoid this last conversion
and return the attitude as angles.

References
----------
.. [Trimpe] Sebastian Trimpe and Raffaello D'Andrea. The Balancing cube. A
    dynamic sculpture as test bed for distributed estimation and control. IEEE
    Control Systems Magazine. December 2012.
    (http://trimpe.is.tuebingen.mpg.de/publications/trimpe-CSM12.pdf)
.. [FS-AN3461] Mark Pedley. Tilt Sensing Using a Three-Axis Accelerometer.
    Freescale Semiconductor Application Note. Document Number: AN3461. 2013.
    (https://www.nxp.com/files-static/sensors/doc/app_note/AN3461.pdf)
.. [FS-AN4248] Talat Ozyagcilar. Implementing a Tilt-Compensated eCompass using
    Accelerometer and Magnetometer sensors. Freescale Semoconductor Application
    Note. Document Number: AN4248. 2015.
    (https://www.nxp.com/files-static/sensors/doc/app_note/AN4248.pdf)
.. [AD-AN1057] Christopher J. Fisher. Using an Accelerometer for Inclination
    Sensing. Analog Devices. Application Note. AN-1057.
    (https://www.analog.com/media/en/technical-documentation/application-notes/AN-1057.pdf)
.. [ST-AN4509] Tilt measurement using a low-g 3-axis accelerometer.
    STMicroelectronics. Application note AN4509. 2014.
    (https://www.st.com/resource/en/application_note/dm00119046.pdf)
.. [WikiConversions] Wikipedia Conversion between quaternions and Euler angles.
    (https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles)

"""

import numpy as np
from ..common.constants import *

class Tilt:
    """
    Gravity-based estimation of attitude.

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of acceleration in in m/s^2
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of magnetic field in mT
    as_angles : bool, default: False
        Whether to return the attitude as rpy angles.

    Attributes
    ----------
    acc : numpy.ndarray
        N-by-3 array with N tri-axial accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N tri-axial magnetometer samples.
    Q : numpy.ndarray, default: None
        N-by-4 or N-by-3 array with
    as_angles : bool, default: False
        Whether to return the attitude as rpy angles.

    Raises
    ------
    ValueError
        When shape of input array ``acc`` is not (N, 3)

    Examples
    --------
    Assuming we have 3-axis accelerometer data in N-by-3 arrays, we can simply
    give these samples to the constructor. The tilt estimation works solely
    with accelerometer samples.

    >>> from ahrs.filters import Tilt
    >>> orientation = Tilt(acc_data)

    The estimated quaternions are saved in the attribute ``Q``.

    >>> type(orientation.Q), orientation.Q.shape
    (<class 'numpy.ndarray'>, (1000, 4))

    If we desire to estimate each sample independently, we call the
    corresponding method.

    .. code:: python

        orientation = Tilt()
        Q = np.zeros((num_samples, 4))  # Allocate quaternions array
        for t in range(num_samples):
            Q[t] = orientation.estimate(acc_data[t])

    Originally, this estimation computes first the Roll-Pitch-Yaw angles and
    then converts them to Quaternions. If we desire the angles instead, we set
    it so in the parameters.

    .. note::
        It will return the angles, in degrees, following the standard order
        ``Roll->Pitch->Yaw``.

    >>> orientation = Tilt(acc_data, as_angles=True)
    >>> type(orientation.Q), orientation.Q.shape
    (<class 'numpy.ndarray'>, (1000, 3))
    >>> orientation.Q[:5]
    array([[8.27467200e-04,  4.36167791e-06, 0.00000000e+00],
           [9.99352822e-04,  8.38015258e-05, 0.00000000e+00],
           [1.30423484e-03,  1.72201573e-04, 0.00000000e+00],
           [1.60337482e-03,  8.53081042e-05, 0.00000000e+00],
           [1.98459171e-03, -8.34729603e-05, 0.00000000e+00]])

    The yaw angle is, expectedly, equal to zero, because the heading cannot be
    estimated with the gravity acceleration only.

    For this reason, the use of magnetometer data can be used to estimate the
    yaw. This is also implemented in :class:`Tilt` and the magnetometer is
    taken into account when given as parameter.

    >>> orientation = Tilt(acc=acc_data, mag=mag_data, as_angles=True)
    >>> orientation.Q[:5]
    array([[8.27467200e-04,  4.36167791e-06, -4.54352439e-02],
           [9.99352822e-04,  8.38015258e-05, -4.52836926e-02],
           [1.30423484e-03,  1.72201573e-04, -4.49355365e-02],
           [1.60337482e-03,  8.53081042e-05, -4.44276770e-02],
           [1.98459171e-03, -8.34729603e-05, -4.36931634e-02]])

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None, **kwargs):
        self.acc = acc
        self.mag = mag
        self.as_angles = kwargs.get('as_angles', False)
        if self.acc is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """
        Estimate the orientation given all data.

        Attribute ``acc`` must contain data. Additiona

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 Array with all estimated quaternions, where M is the number
            of samples.

        """
        if self.acc.shape[-1]!=3:
            raise ValueError("Input data must be of shape (N, 3). Got shape {}".format(self.acc.shape))
        num_samples = len(self.acc)
        Q = np.zeros((num_samples, 3)) if self.as_angles else np.zeros((num_samples, 4))
        for t in range(num_samples):
            Q[t] = self.estimate(self.acc[t]) if self.mag is None else self.estimate(self.acc[t], self.mag[t])
        return Q
        #######################################
        # # Normalization of 2D arrays
        # a = self.acc/np.linalg.norm(self.acc, axis=1)[:, None]
        # angles = np.zeros((len(a), 3))   # Allocation of angles array
        # # Estimate tilt angles
        # angles[:, 0] = np.arctan2(a[:, 1], a[:, 2])
        # angles[:, 1] = np.arctan2(-a[:, 0], np.sqrt(a[:, 1]**2 + a[:, 2]**2))
        # if self.mag is not None:
        #     # Estimate heading angle
        #     m = self.mag/np.linalg.norm(self.mag, axis=1)[:, None]
        #     my2 = m[:, 2]*np.sin(angles[:, 0]) - m[:, 1]*np.cos(angles[:, 0])
        #     mz2 = m[:, 1]*np.sin(angles[:, 0]) + m[:, 2]*np.cos(angles[:, 0])
        #     mx3 = m[:, 0]*np.cos(angles[:, 1]) + mz2*np.sin(angles[:, 1])
        #     angles[:, 2] = np.arctan2(my2, mx3)
        #     # Return in degrees or in radians
        #     if in_deg:
        #         return angles*dptk.maths.RAD2DEG
        # return angles

    def estimate(self, acc: np.ndarray, mag: np.ndarray = None) -> np.ndarray:
        """
        Estimate the quaternion from the tilting read by an orthogonal
        tri-axial array of accelerometers.

        The orientation of the roll and pitch angles is estimated using the
        measurements of the accelerometers, and finally converted to a
        quaternion representation according to [WikiConversions]_

        Parameters
        ----------
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2.

        Returns
        -------
        q : numpy.ndarray
            Estimated quaternion.

        """
        a_norm = np.linalg.norm(acc)
        if not a_norm>0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        ax, ay, az = acc/a_norm
        ### Tilt from Accelerometer
        # Roll
        ex = np.arctan2( ay, az)
        cx = np.cos(0.5*ex)
        sx = np.sin(0.5*ex)
        # Pitch
        ey = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        cy = np.cos(0.5*ey)
        sy = np.sin(0.5*ey)
        # Yaw
        ez = 0.0
        # m_norm = np.linalg.norm(self.mag)
        if mag is not None and np.linalg.norm(mag)>0:
            mx, my, mz = mag/np.linalg.norm(mag)
            # Get tilted reference frame
            my2 = mz*np.sin(ex) - my*np.cos(ex)
            mx3 = mx*np.cos(ey) + np.sin(ey)*(my*np.sin(ex) + mz*np.cos(ex))
            ez = np.arctan2(my2, mx3)
        if self.as_angles:
            return np.array([ex, ey, ez])*RAD2DEG
        #### Euler to Quaternion
        cp = np.cos(0.5*ey)
        sp = np.sin(0.5*ey)
        cr = np.cos(0.5*ex)
        sr = np.sin(0.5*ex)
        cy = np.cos(0.5*ez)
        sy = np.sin(0.5*ez)
        q = np.zeros(4)
        q[0] = cy*cp*cr + sy*sp*sr
        q[1] = cy*cp*sr - sy*sp*cr
        q[2] = sy*cp*sr + cy*sp*cr
        q[3] = sy*cp*cr - cy*sp*sr
        return q/np.linalg.norm(q)

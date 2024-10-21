# -*- coding: utf-8 -*-
"""
Attitude from gravity (Tilt)
============================

Attitude estimation via gravity acceleration measurements.

The simplest way to estimate the attitude from the gravitational acceleration
is using 3D `geometric quadrants <https://en.wikipedia.org/wiki/Quadrant_(plane_geometry)>`_.

Although some methods use ``arctan`` to estimate the angles :cite:p:`stm4509`
:cite:p:`fisher2010` it is preferred to use ``arctan2`` to explore all
quadrants searching the tilt angles.

First, we normalize the gravity vector, so that it has magnitude equal to 1.
Then, we get the angles to the main axes with `arctan2
<https://en.wikipedia.org/wiki/Atan2>`_ :cite:p:`pedley2013` :cite:p:`trimpe2012`:

.. math::
    \\begin{array}{ll}
    \\theta &= \\mathrm{arctan2}(a_y, a_z) \\\\
    \\phi &= \\mathrm{arctan2}\\big(-a_x, \\sqrt{a_y^2+a_z^2}\\big)
    \\end{array}

where :math:`\\theta` is the **roll** angle, :math:`\\phi` is the **pitch**
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
a magnetometer reading :math:`\\mathbf{m}=\\begin{bmatrix}m_x & m_y & m_z\\end{bmatrix}^T`,
and estimate the yaw angle :math:`\\psi` to update the orientation.

The vector :math:`\\mathbf{b}=\\begin{bmatrix}b_x & b_y & b_z\\end{bmatrix}^T`
represents the magnetometer readings after *rotating them back* to the plane,
where :math:`\\theta = \\phi = 0`.

.. math::
    \\begin{array}{cl}
    \\mathbf{b} &=
    R_y(-\\theta)R_x(-\\phi)\\mathbf{m} = R_y(\\theta)^TR_x(\\phi)^T\\mathbf{m} \\\\
    &=
    \\begin{bmatrix}
        \\cos\\theta & \\sin\\theta\\sin\\phi & \\sin\\theta\\cos\\phi \\\\
        0 & \\cos\\phi & -\\sin\\phi \\\\
        -\\sin\\theta & \\cos\\theta\\sin\\phi & \\cos\\theta\\cos\\phi
    \\end{bmatrix}
    \\begin{bmatrix}m_x \\\\ m_y \\\\ m_z\\end{bmatrix} \\\\
    \\begin{bmatrix}b_x \\\\ b_y \\\\ b_z\\end{bmatrix} &=
    \\begin{bmatrix}
        m_x\\cos\\theta + m_y\\sin\\theta\\sin\\phi + m_z\\sin\\theta\\cos\\phi \\\\
        m_y\\cos\\phi - m_z\\sin\\phi \\\\
        -m_x\\sin\\theta + m_y\\cos\\theta\\sin\\phi + m_z\\cos\\theta\\cos\\phi
    \\end{bmatrix}
    \\end{array}

Where :math:`\\mathbf{m}=\\begin{bmatrix}m_x & m_y & m_z\\end{bmatrix}^T` is
the *normalized* vector of the measured magnetic field, which means
:math:`\\|\\mathbf{m}\\|=1`.

The yaw angle :math:`\\psi` is the tilt-compensated heading angle relative to
magnetic North, computed as :cite:p:`ozyagcilar2015`:

.. math::
    \\begin{array}{ll}
    \\psi &= \\mathrm{arctan2}(-b_y, b_x) \\\\
    &= \\mathrm{arctan2}\\big(m_z\\sin\\phi - m_y\\cos\\phi, \\; m_x\\cos\\theta + \\sin\\theta(m_y\\sin\\phi + m_z\\cos\\phi)\\big)
    \\end{array}

Finally, we transform the roll-pitch-yaw angles to a quaternion representation:

.. math::
    \\mathbf{q} =
    \\begin{pmatrix}q_w\\\\q_x\\\\q_y\\\\q_z\\end{pmatrix} =
    \\begin{pmatrix}
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) + \\sin\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\sin\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) - \\cos\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big) + \\sin\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) \\\\
        \\cos\\Big(\\frac{\\phi}{2}\\Big)\\cos\\Big(\\frac{\\theta}{2}\\Big)\\sin\\Big(\\frac{\\psi}{2}\\Big) - \\sin\\Big(\\frac{\\phi}{2}\\Big)\\sin\\Big(\\frac{\\theta}{2}\\Big)\\cos\\Big(\\frac{\\psi}{2}\\Big)
    \\end{pmatrix}

Setting the property ``as_angles`` to ``True`` will avoid this last conversion
returning the attitude as angles.

"""

import numpy as np
from ..common.constants import RAD2DEG
from ..common.orientation import q2R
from ..utils.core import _assert_numerical_iterable

def _assert_representation(representation: str):
    if not isinstance(representation, str):
        raise TypeError(f"Representation must be a string. Got {type(representation)}.")
    if representation not in ['quaternion', 'angles', 'rotmat']:
        raise ValueError(f"Given representation '{representation}' is NOT valid. Try 'rotmat', 'angles' or 'quaternion'")

class Tilt:
    """
    Gravity-based estimation of attitude.

    Parameters
    ----------
    acc : numpy.ndarray, default: None
        N-by-3 array with measurements of gravitational acceleration.
    mag : numpy.ndarray, default: None
        N-by-3 array with measurements of local geomagnetic field.
    representation : str, default: ``'quaternion'``
        Attitude representation. Options are ``'quaternion'``, ``'angles'`` or
        ``'rotmat'``.

    Attributes
    ----------
    acc : numpy.ndarray
        N-by-3 array with N tri-axial accelerometer samples.
    mag : numpy.ndarray
        N-by-3 array with N tri-axial magnetometer samples.
    Q : numpy.ndarray, default: None
        N-by-4 or N-by-3 array with N quaternions.

    Raises
    ------
    ValueError
        When shape of input arrays are not (3,) or (N, 3).

    Examples
    --------
    Assuming we have 3-axis accelerometer data in N-by-3 arrays, we can simply
    give these samples to the constructor. The tilt estimation works solely
    with accelerometer samples.

    >>> from ahrs.filters import Tilt
    >>> tilt = Tilt(acc_data)

    The estimated quaternions are saved in the attribute ``Q``.

    >>> tilt.Q
    array([[0.76901856, 0.60247641, -0.16815772, 0.13174072],
           [0.77310283, 0.59724644, -0.16900433, 0.1305612 ],
           [0.7735134,  0.59644005, -0.1697294,  0.1308748 ],
           ...,
           [0.7800751,  0.59908629, -0.14315079, 0.10993772],
           [0.77916118, 0.59945374, -0.14520157, 0.11171197],
           [0.77038613, 0.61061868, -0.14375869, 0.11394512]])
    >>> tilt.Q.shape
    (1000, 4)

    If we desire to estimate each sample independently, we call the
    corresponding method with each sample individually.

    >>> tilt = Tilt()
    >>> num_samples = len(acc_data)
    >>> Q = np.zeros((num_samples, 4))  # Allocate quaternions array
    >>> for t in range(num_samples):
    ...     Q[t] = tilt.estimate(acc_data[t])
    ...
    >>> tilt.Q[:5]
    array([[0.76901856, 0.60247641, -0.16815772, 0.13174072],
           [0.77310283, 0.59724644, -0.16900433, 0.1305612 ],
           [0.7735134,  0.59644005, -0.1697294,  0.1308748 ],
           [0.77294791, 0.59913005, -0.16502363, 0.12791369],
           [0.76936935, 0.60323746, -0.16540014, 0.12968487]])

    Originally, this estimation computes first the Roll-Pitch-Yaw angles and
    then converts them to Quaternions. If we desire the angles instead, we set
    it so in the parameters.

    >>> tilt = Tilt(acc_data, representation='angles')
    >>> type(tilt.Q), tilt.Q.shape
    (<class 'numpy.ndarray'>, (1000, 3))
    >>> tilt.Q[:5]
    array([[8.27467200e-04,  4.36167791e-06, 0.00000000e+00],
           [9.99352822e-04,  8.38015258e-05, 0.00000000e+00],
           [1.30423484e-03,  1.72201573e-04, 0.00000000e+00],
           [1.60337482e-03,  8.53081042e-05, 0.00000000e+00],
           [1.98459171e-03, -8.34729603e-05, 0.00000000e+00]])

    .. note::
        It will return the angles, in radians, following the standard order
        ``Roll->Pitch->Yaw``.

    The yaw angle is, expectedly, equal to zero, because the heading cannot be
    estimated with the gravity acceleration only.

    For this reason, magnetometer data can be used to estimate the yaw. This is
    also implemented and the magnetometer will be taken into account when given
    as parameter.

    >>> tilt = Tilt(acc=acc_data, mag=mag_data, representation='angles')
    >>> tilt.Q[:5]
    array([[8.27467200e-04,  4.36167791e-06, -4.54352439e-02],
           [9.99352822e-04,  8.38015258e-05, -4.52836926e-02],
           [1.30423484e-03,  1.72201573e-04, -4.49355365e-02],
           [1.60337482e-03,  8.53081042e-05, -4.44276770e-02],
           [1.98459171e-03, -8.34729603e-05, -4.36931634e-02]])

    """
    def __init__(self, acc: np.ndarray = None, mag: np.ndarray = None, **kwargs):
        self.acc: np.ndarray = acc
        self.mag: np.ndarray = mag
        self.representation: str = kwargs.get('representation', 'quaternion')
        self.as_angles: bool = kwargs.get('as_angles', self.representation == 'angles') # Old parameter. Backwards compatiblity.
        _assert_representation(self.representation)
        self.angles: np.ndarray = None
        if self.acc is not None:
            self.Q = self._compute_all()

    def _compute_all(self) -> np.ndarray:
        """
        Estimate the orientation given all data.

        Attributes ``acc`` and ``mag`` must contain data. It is assumed that
        these attributes have the same shape (M, 3), where M is the number of
        observations.

        The full estimation is vectorized, to avoid the use of a time-wasting
        loop.

        Returns
        -------
        Q : numpy.ndarray
            M-by-4 array with all estimated quaternions, where M is the number
            of samples. It returns an M-by-3 array if ``representation`` is set
            to ``angles``, or M-by-3-by-3 array if set to ``rotmat``.

        """
        _assert_numerical_iterable(self.acc, 'Gravitational acceleration vector')
        self.acc = np.copy(self.acc)
        if self.acc.ndim < 2:
            return self.estimate(self.acc, self.mag, self.representation)
        # Normalization of 2D arrays
        a = self.acc/np.linalg.norm(self.acc, axis=1)[:, None]
        self.angles = np.zeros((len(a), 3))   # Allocation of angles array
        # Estimate tilt angles
        self.angles[:, 0] = np.arctan2(a[:, 1], a[:, 2])
        self.angles[:, 1] = np.arctan2(-a[:, 0], np.sqrt(a[:, 1]**2 + a[:, 2]**2))
        if self.mag is not None:
            _assert_numerical_iterable(self.mag, 'Geomagnetic field vector')
            # Estimate heading angle
            m = self.mag/np.linalg.norm(self.mag, axis=1)[:, None]
            my2 = m[:, 2]*np.sin(self.angles[:, 0]) - m[:, 1]*np.cos(self.angles[:, 0])
            mz2 = m[:, 1]*np.sin(self.angles[:, 0]) + m[:, 2]*np.cos(self.angles[:, 0])
            mx3 = m[:, 0]*np.cos(self.angles[:, 1]) + mz2*np.sin(self.angles[:, 1])
            self.angles[:, 2] = np.arctan2(my2, mx3)
        # Return angles in radians
        if self.representation == 'angles':
            return self.angles
        # RPY to Quaternion
        cp = np.cos(0.5*self.angles[:, 1])
        sp = np.sin(0.5*self.angles[:, 1])
        cr = np.cos(0.5*self.angles[:, 0])
        sr = np.sin(0.5*self.angles[:, 0])
        cy = np.cos(0.5*self.angles[:, 2])
        sy = np.sin(0.5*self.angles[:, 2])
        Q = np.zeros((len(self.acc), 4))
        Q[:, 0] = cy*cp*cr + sy*sp*sr
        Q[:, 1] = cy*cp*sr - sy*sp*cr
        Q[:, 2] = sy*cp*sr + cy*sp*cr
        Q[:, 3] = sy*cp*cr - cy*sp*sr
        if self.representation == 'quaternion':
            return Q/np.linalg.norm(Q, axis=1)[:, None]
        return q2R(Q)

    def estimate(self, acc: np.ndarray, mag: np.ndarray = None, representation: str = 'quaternion') -> np.ndarray:
        """
        Estimate the quaternion from the tilting read by an orthogonal
        tri-axial array of accelerometers.

        The orientation of the roll and pitch angles is estimated using the
        measurements of the accelerometers, and finally converted to a
        quaternion representation according to :cite:p:`Wiki_Conversions`.

        Parameters
        ----------
        acc : numpy.ndarray
            Sample of tri-axial Accelerometer in m/s^2.
        mag : numpy.ndarray, default: None
            N-by-3 array with measurements of magnetic field in mT.

        Returns
        -------
        q : numpy.ndarray
            Estimated attitude.

        Examples
        --------
        >>> acc_data = np.array([4.098297, 8.663757, 2.1355896])
        >>> mag_data = np.array([-28.71550512, -25.92743566, 4.75683931])
        >>> from ahrs.filters import Tilt
        >>> tilt = Tilt()
        >>> tilt.estimate(acc=acc_data, mag=mag_data)   # Estimate attitude as quaternion
        array([0.09867706 0.33683592 0.52706394 0.77395607])

        Optionally, the attitude can be retrieved as roll-pitch-yaw angles, in
        radians.

        >>> tilt = Tilt(as_angles=True)
        >>> tilt.estimate(acc=acc_data, mag=mag_data)
        array([ 76.15281566 -24.66891862 146.02634429])

        """
        _assert_representation(representation)
        _assert_numerical_iterable(acc, 'Gravitational acceleration vector')
        acc = np.copy(acc)
        a_norm = np.linalg.norm(acc)
        if not a_norm > 0:
            raise ValueError("Gravitational acceleration must be non-zero")
        ax, ay, az = acc/a_norm
        ### Tilt from Accelerometer
        ex = np.arctan2( ay, az)                        # Roll
        ey = np.arctan2(-ax, np.sqrt(ay**2 + az**2))    # Pitch
        ez = 0.0                                        # Yaw
        if mag is not None:
            _assert_numerical_iterable(mag, 'Geomagnetic field vector')
            if not(np.linalg.norm(mag) > 0):
                raise ValueError("Magnetic field must be non-zero")
            mx, my, mz = mag/np.linalg.norm(mag)
            # Get tilted reference frame
            by = my*np.cos(ex) - mz*np.sin(ex)
            bx = mx*np.cos(ey) + np.sin(ey)*(my*np.sin(ex) + mz*np.cos(ex))
            ez = np.arctan2(-by, bx)
        if representation == 'angles':
            return np.array([ex, ey, ez])
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
        if representation == 'quaternion':
            return q/np.linalg.norm(q)
        return q2R(q)

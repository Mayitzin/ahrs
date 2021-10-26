# -*- coding: utf-8 -*-
"""
TRIAD
=====

The Tri-Axial Attitude Determination (`TRIAD <https://en.wikipedia.org/wiki/Triad_method>`_)
was first described in [Black]_ to algebraically estimate an attitude
represented as a Direction Cosine Matrix from two orthogonal vector
observations.

Given two non-parallel reference *unit vectors* :math:`\\mathbf{v}_1` and
:math:`\\mathbf{v}_2` and their corresponding *unit vectors* :math:`\\mathbf{w}_1`
and :math:`\\mathbf{w}_2`, it is required to find an orthogonal matrix
:math:`\\mathbf{A}` satisfying:

.. math::
    \\mathbf{Av}_1

Two vectors :math:`\\mathbf{v}_1` and :math:`\\mathbf{v}_2` define an
orthogonal coordinate system with the **normalized** basis vectors
:math:`\\mathbf{q}`, :math:`\\mathbf{r}`, and :math:`\\mathbf{s}` as the
following triad:

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}_r &=& \\mathbf{v}_1 \\\\
    \\mathbf{r}_r &=& \\frac{\\mathbf{v}_1\\times\\mathbf{v}_2}{|\\mathbf{v}_1\\times\\mathbf{v}_2|} \\\\
    \\mathbf{s}_r &=& \\mathbf{q}_r\\times\\mathbf{r}_r
    \\end{array}

The TRIAD method, initially developed to estimate the attitude of spacecrafts
[Shuster2007]_, uses the position of the sun (using a `star tracker
<https://en.wikipedia.org/wiki/Star_tracker>`_) and the magnetic field of Earth
as references [Hall]_ [Makley]_. These are represented as vectors to build an
appropriate *reference* frame :math:`\\mathbf{M}_r`:

.. math::
    \\mathbf{M}_r = \\begin{bmatrix} \\mathbf{q}_r & \\mathbf{r}_r & \\mathbf{s}_r \\end{bmatrix}

Similarly, at any given time, two measured vectors in the spacecraft's **body
frame** :math:`\\mathbf{w}_1` and :math:`\\mathbf{w}_2` determine the
:math:`3\\times 3` body matrix :math:`\\mathbf{M}_b`:

.. math::
    \\mathbf{M}_b = \\begin{bmatrix} \\mathbf{q}_b & \\mathbf{r}_b & \\mathbf{s}_b \\end{bmatrix}

where, like the first triad, the second triad is built as:

.. math::
    \\begin{array}{rcl}
    \\mathbf{q}_b &=& \\mathbf{w}_1 \\\\
    \\mathbf{r}_b &=& \\frac{\\mathbf{w}_1\\times\\mathbf{w}_2}{|\\mathbf{w}_1\\times\\mathbf{w}_2|} \\\\
    \\mathbf{s}_b &=& \\mathbf{q}_b\\times\\mathbf{r}_b
    \\end{array}

The attitude matrix :math:`\\mathbf{A}\\in\\mathbb{R}^{3\\times 3}` defines the
coordinate transformation,

.. math::
    \\mathbf{AM}_r = \\mathbf{M}_b

Solving for :math:`\\mathbf{A}` we obtain:

.. math::
    \\mathbf{A} = \\mathbf{M}_b\\mathbf{M}_r^{-1}

But we also know that :math:`\\mathbf{M}_r` is orthogonal. So, the solution is
simply:

.. math::
    \\mathbf{A} = \\mathbf{M}_b\\mathbf{M}_r^T

Inverse trigonometric functions are not required, a unique attitude is obtained,
and computational requirements are minimal.

It is only required that :math:`\\mathbf{M}_r` has an inverse, but that is
already ensured, since :math:`\\mathbf{q}_r`, :math:`\\mathbf{r}_r`,
and :math:`\\mathbf{s}_r` are linearly independent [Lerner1]_.

Strapdown INS
-------------

For estimations using a Strapdown INS on Earth, we identify two main reference
vectors: gravity :math:`\\mathbf{g}=\\begin{bmatrix}g_x & g_y & g_z\\end{bmatrix}`
and magnetic field :math:`\\mathbf{h}=\\begin{bmatrix}h_x & h_y & h_z\\end{bmatrix}`.

A common convention sets the *gravity vector* equal to :math:`0` along the X-
and Y-axis, and equal to :math:`\\sim 9.81` along the Z-axis. This assumes the
direction of the gravity is parallel to the vertical axis. Because TRIAD uses
normalized vectors, the Z-axis will turn out to be equal to :math:`1`:

.. math::
    \\mathbf{g} = \\begin{bmatrix}0 \\\\ 0 \\\\ 1 \\end{bmatrix}

The *magnetic field* is defined from the geographical position of the
measurement. Using the `World Magnetic Model <https://www.ngdc.noaa.gov/geomag/WMM/>`_,
we can estimate the magnetic field elements of our location on a given date.

The class :class:`ahrs.utils.WMM` can help us to retrieve it. Let's say we want
to know the geomagnetic field elements of Munich, Germany [#]_ on the 3rd of
October, 2020.

The city's location is 48.137154° N and 11.576124° E at 519 m above sea level.
We obtain its magnetic elements as:

.. code-block:: python

    >>> import datetime
    >>> from ahrs.utils import WMM
    >>> wmm = WMM(latitude=48.137154, longitude=11.576124, height=0.519, date=datetime.date(2020, 10, 3))
    >>> wmm.magnetic_elements
    {'X': 21009.66924050522, 'Y': 1333.4601319284525, 'Z': 43731.849938722924, 'H': 21051.943319296533, 'F': 48535.13177670226, 'I': 64.2944417667441, 'D': 3.631627635223863, 'GV': 3.631627635223863}

For further explanation of class :class:`WMM`, please check its `page <../WMM.html>`_.
Of our interest are only the values of ``X``, ``Y`` and ``Z`` representing the
magnetic field intensity, in nT, along the X-, Y- and Z-axis, respectively.

.. math::
    \\mathbf{h} = \\begin{bmatrix} 21009.66924 \\\\ 1333.46013 \\\\ 43731.84994 \\end{bmatrix}

But, again, TRIAD works with normalized vectors, so the reference magnetic
vector becomes:

.. math::
    \\mathbf{h} = \\begin{bmatrix} 0.43288 \\\\ 0.02747 \\\\ 0.90103 \\end{bmatrix}

.. code-block:: python

    >>> import numpy as np
    >>> h = np.array([wmm.magnetic_elements[x] for x in list('XYZ')])
    >>> h /= np.linalg.norm(h)      # Reference geomagnetic field (h)
    >>> h
    array([0.4328755 , 0.02747412, 0.90103495])

Both normalized vectors :math:`\\mathbf{g}` and :math:`\\mathbf{h}` build the
*reference triad* :math:`\\mathbf{M}_r`

Then, we have to measure their equivalent vectors, for which we use the
accelerometer to obtain :math:`\\mathbf{a} = \\begin{bmatrix}a_x & a_y & a_z \\end{bmatrix}`,
and the magnetometer for :math:`\\mathbf{m} = \\begin{bmatrix}m_x & m_y & m_z \\end{bmatrix}`.

Both measurement vectors are also normalized, meaning :math:`\\|\\mathbf{a}\\|=\\|\\mathbf{m}\\|=1`,
so that they can build the *body's measurement triad*  :math:`\\mathbf{M}_b`.

To get the Direction Cosine Matrix we simply call the method ``estimate`` with
the normalized measurement vectors:

.. code-block:: python

    >>> triad = ahrs.filters.TRIAD()
    >>> triad.v1 = np.array([0.0, 0.0, 1.0])                    # Reference gravity vector (g)
    >>> triad.v2 = h                                            # Reference geomagnetic field (h)
    >>> a = np.array([-2.499e-04, 4.739e-02, 0.9988763])        # Measured acceleration (normalized)
    >>> a /= np.linalg.norm(a)
    >>> m = np.array([-0.36663061, 0.17598138, -0.91357132])    # Measured magnetic field (normalized)
    >>> m /= np.linalg.norm(m)
    >>> triad.estimate(w1=a, w2=m)
    array([[-8.48320410e-01, -5.29483162e-01, -2.49900033e-04],
           [ 5.28878238e-01, -8.47373587e-01,  4.73900062e-02],
           [-2.53039690e-02,  4.00697428e-02,  9.98876431e-01]])

Optionally, it can return the estimation as a quaternion representation setting
``representation`` to ``'quaternion'``.

.. code-block:: python

    >>> triad.estimate(w1=a, w2=m, representation='quaternion')
    array([ 0.27531002, -0.00664729,  0.02275078,  0.96106327])

Giving the observation vector to the constructor, the attitude estimation
happens automatically, and is stored in the attribute ``A``.

.. code-block:: python

    >>> triad = ahrs.filters.TRIAD(w1=np.array([-2.499e-04, 4.739e-02, 0.9988763]), w2=np.array([-0.36663061, 0.17598138, -0.91357132]), v2=h)
    >>> triad.A
    array([[-8.48320410e-01, -5.29483162e-01, -2.49900033e-04],
           [ 5.28878238e-01, -8.47373587e-01,  4.73900062e-02],
           [-2.53039690e-02,  4.00697428e-02,  9.98876431e-01]])
    >>> triad = ahrs.filters.TRIAD(w1=np.array([-2.499e-04, 4.739e-02, 0.9988763]), w2=np.array([-0.36663061, 0.17598138, -0.91357132]), v2=h, representation='quaternion')
    >>> triad.A
    array([ 0.27531002, -0.00664729,  0.02275078,  0.96106327])

If the input data contains many observations, all will be estimated at once.

.. code-block:: python

    >>> a = np.array([[-0.000249905733, 0.0473926177, 0.998876307],
    ... [-0.00480145530, 0.0572267567, 0.998349660],
    ... [-0.00986626329, 0.0746539896, 0.997160688]])
    >>> m = np.array([[-0.36663061, 0.17598138, -0.91357132],
    ... [-0.37726367, 0.18069746, -0.90830642],
    ... [-0.3874741, 0.18536454, -0.9030525]])
    >>> triad = ahrs.filters.TRIAD(w1=a, w2=m, v2=h)
    >>> triad.A
    array([[[-8.48317898e-01, -5.29487187e-01, -2.49905733e-04],
            [ 5.28882192e-01, -8.47370974e-01,  4.73926177e-02],
            [-2.53055467e-02,  4.00718352e-02,  9.98876307e-01]],

           [[-8.43678607e-01, -5.36827117e-01, -4.80145530e-03],
            [ 5.35721702e-01, -8.42453178e-01,  5.72267567e-02],
            [-3.47658761e-02,  4.57087466e-02,  9.98349660e-01]],

           [[-8.32771974e-01, -5.53528225e-01, -9.86626329e-03],
            [ 5.51396878e-01, -8.30896061e-01,  7.46539896e-02],
            [-4.95209297e-02,  5.67295235e-02,  9.97160688e-01]]])
    >>> triad = ahrs.filters.TRIAD(w1=a, w2=m, representation='quaternion')
    >>> triad.A
    array([[ 0.27531229, -0.00664771,  0.02275202,  0.96106259],
           [ 0.2793823 , -0.01030667,  0.0268131 ,  0.95975016],
           [ 0.28874411, -0.01551933,  0.03433374,  0.95666461]])

The first disadvantage is that TRIAD can only use two observations per
estimation. If there are more observations, we must discard part of them
(losing accuracy), or mix them in such a way that we obtain only two
representative observations.

The second disadvantage is its loss of accuracy in a heavily dynamic state of
the measuring device. TRIAD assumes a quasi-static state of the body frame and,
therefore, its use is limited to motionless objects, preferably.

Footnotes
---------
.. [#] This package's author resides in Munich, and examples of geographical
    locations will take it as a reference.

References
----------
.. [Black] Black, Harold. "A Passive System for Determining the Attitude of a
    Satellite," AIAA Journal, Vol. 2, July 1964, pp. 1350–1351.
.. [Lerner1] Lerner, G. M. "Three-Axis Attitude Determination" in Spacecraft
    Attitude Determination and Control, edited by J.R. Wertz. 1978. p. 420-426.
.. [Hall] Chris Hall. Spacecraft Attitude Dynamics and Control. Chapter 4:
    Attitude Determination. 2003.
    (http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf)
.. [Makley] F.L. Makley et al. Fundamentals of Spacecraft Attitude
    Determination and Control. 2014. Pages 184-186.
.. [Shuster2007] Shuster, Malcolm D. The optimization of TRIAD. The Journal of
    the Astronautical Sciences, Vol. 55, No 2, April – June 2007, pp. 245–257.
    (http://www.malcolmdshuster.com/Pub_2007f_J_OptTRIAD_AAS.pdf)

"""

import numpy as np
from ..common.orientation import chiaverini
from ..common.mathfuncs import *

# Reference Observations in Munich, Germany
from ..utils.wmm import WMM
MAG = WMM(latitude=MUNICH_LATITUDE, longitude=MUNICH_LONGITUDE, height=MUNICH_HEIGHT).magnetic_elements

class TRIAD:
    """
    Tri-Axial Attitude Determination

    TRIAD estimates the attitude as a Direction Cosine Matrix. To return it as
    a quaternion, set the parameter ``as_quaternion`` to ``True``.

    Parameters
    ----------
    w1 : numpy.ndarray
        First tri-axial observation vector in body frame. Usually a normalized
        acceleration vector :math:`\\mathbf{a} = \\begin{bmatrix}a_x & a_y & a_z \\end{bmatrix}`
    w2 : numpy.ndarray
        Second tri-axial observation vector in body frame. Usually a normalized
        magnetic field vector :math:`\\mathbf{m} = \\begin{bmatrix}m_x & m_y & m_z \\end{bmatrix}`
    v1 : numpy.ndarray, optional.
        First tri-axial reference vector. Defaults to normalized gravity vector
        :math:`\\mathbf{g} = \\begin{bmatrix}0 & 0 & 1 \\end{bmatrix}`
    v2 : numpy.ndarray, optional.
        Second tri-axial reference vector. Defaults to normalized geomagnetic
        field :math:`\\mathbf{h} = \\begin{bmatrix}h_x & h_y & h_z \\end{bmatrix}`
        in Munich, Germany.
    representation : str, default: ``'rotmat'``
        Attitude representation. Options are ``rotmat'`` or ``'quaternion'``.
    frame : str, default: 'NED'
        Local tangent plane coordinate frame. Valid options are right-handed
        ``'NED'`` for North-East-Down and ``'ENU'`` for East-North-Up.

    Attributes
    ----------
    w1 : numpy.ndarray
        First tri-axial observation vector in body frame.
    w2 : numpy.ndarray
        Second tri-axial observation vector in body frame.
    v1 : numpy.ndarray, optional.
        First tri-axial reference vector.
    v2 : numpy.ndarray, optional.
        Second tri-axial reference vector.
    A : numpy.ndarray
        Estimated attitude.

    Examples
    --------
    >>> from ahrs.filters import TRIAD
    >>> triad = TRIAD()
    >>> triad.v1 = np.array([0.0, 0.0, 1.0])                    # Reference gravity vector (g)
    >>> triad.v2 = np.array([21.0097, 1.3335, 43.732])          # Reference geomagnetic field (h)
    >>> a = np.array([-2.499e-04, 4.739e-02, 0.9988763])        # Measured acceleration (normalized)
    >>> a /= np.linalg.norm(a)
    >>> m = np.array([-0.36663061, 0.17598138, -0.91357132])    # Measured magnetic field (normalized)
    >>> m /= np.linalg.norm(m)
    >>> triad.estimate(w1=a, w2=m)
    array([[-8.48320410e-01, -5.29483162e-01, -2.49900033e-04],
           [ 5.28878238e-01, -8.47373587e-01,  4.73900062e-02],
           [-2.53039690e-02,  4.00697428e-02,  9.98876431e-01]])

    It also works by passing each array to its corresponding parameter. They
    will be normalized too.

    >>> triad = TRIAD(w1=a, w2=m, v1=[0.0, 0.0, 1.0], v2=[-0.36663061, 0.17598138, -0.91357132])

    """
    def __init__(self,
        w1: np.ndarray = None,
        w2: np.ndarray = None,
        v1: np.ndarray = None,
        v2: np.ndarray = None,
        representation: str = 'rotmat',
        frame: str = 'NED'):
        self.w1: np.ndarray = np.copy(w1)
        self.w2: np.ndarray = np.copy(w2)
        self.representation: str = representation
        if representation.lower() not in ['rotmat', 'quaternion']:
            raise ValueError("Wrong representation type. Try 'rotmat', or 'quaternion'")
        if frame.upper() not in ['NED', 'ENU']:
            raise ValueError(f"Given frame {frame} is NOT valid. Try 'NED' or 'ENU'")
        # Reference frames
        self.v1 = self._set_first_triad_reference(v1, frame)
        self.v2 = self._set_second_triad_reference(v2, frame)
        # Compute values if samples given
        if self.w1 is not None and self.w2 is not None:
            self.A = self._compute_all(self.representation)

    def _set_first_triad_reference(self, value, frame):
        if value is None:
            ref = np.array([0.0, 0.0, 1.0]) if frame.upper == 'NED' else np.array([0.0, 0.0, -1.0])
        else:
            ref = np.copy(value)
            ref /= np.linalg.norm(ref)
        return ref

    def _set_second_triad_reference(self, value, frame):
        ref = np.array([MAG['X'], MAG['Y'], MAG['Z']])
        if isinstance(value, float):
            if abs(value) > 90:
                raise ValueError(f"Dip Angle must be within range [-90, 90]. Got {value}")
            ref = np.array([cosd(value), 0.0, sind(value)]) if frame.upper() == 'NED' else np.array([0.0, cosd(value), -sind(value)])
        if isinstance(value, (np.ndarray, list)):
            ref = np.copy(value)
        return ref/np.linalg.norm(ref)

    def _compute_all(self, representation: str) -> np.ndarray:
        """
        Estimate the attitude given all data.

        Attributes ``w1`` and ``w2`` must contain data.

        Parameters
        ----------
        representation : str
            Attitude representation. Options are ``'rotmat'`` or ``'quaternion'``.

        Returns
        -------
        A : numpy.ndarray
            M-by-3-by-3 with all estimated attitudes as direction cosine
            matrices, where M is the number of samples. It is an N-by-4 array
            if ``representation`` is set to ``'quaternion'``.

        """
        if self.w1.shape != self.w2.shape:
            raise ValueError("w1 and w2 are not the same size")
        if self.w1.ndim == 1:
            return self.estimate(self.w1, self.w2, representation)
        num_samples = len(self.w1)
        A = np.zeros((num_samples, 4)) if representation.lower() == 'quaternion' else np.zeros((num_samples, 3, 3))
        for t in range(num_samples):
            A[t] = self.estimate(self.w1[t], self.w2[t], representation)
        return A

    def estimate(self, w1: np.ndarray, w2: np.ndarray, representation: str = 'rotmat') -> np.ndarray:
        """
        Attitude Estimation.

        The equation numbers in the code refer to [Lerner1]_.

        Parameters
        ----------
        w1 : numpy.ndarray
            Sample of first tri-axial sensor.
        w2 : numpy.ndarray
            Sample of second tri-axial sensor.
        representation : str, default: ``'rotmat'``
            Attitude representation. Options are ``rotmat'`` or ``'quaternion'``.

        Returns
        -------
        A : numpy.ndarray
            Estimated attitude as 3-by-3 Direction Cosine Matrix. If
            ``representation`` is set to ``'quaternion'``, it is returned as a
            quaternion.

        Examples
        --------
        >>> triad = ahrs.filters.TRIAD()
        >>> triad.v1 = [0.0, 0.0, 1.0]                              # Normalized reference gravity vector (g)
        >>> triad.v2 = [0.4328755, 0.02747412, 0.90103495]          # Normalized reference geomagnetic field (h)
        >>> a = [4.098297, 8.663757, 2.1355896]                     # Measured acceleration
        >>> m = [-28715.50512, -25927.43566, 4756.83931]            # Measured magnetic field
        >>> triad.estimate(w1=a, w2=m)                              # Estimated attitude as DCM
        array([[-7.84261e-01  ,  4.5905718e-01,  4.1737417e-01],
               [ 2.2883429e-01, -4.1126404e-01,  8.8232463e-01],
               [ 5.7668844e-01,  7.8748232e-01,  2.1749032e-01]])
        >>> triad.estimate(w1=a, w2=m, representation='quaternion')          # Estimated attitude as quaternion
        array([ 0.07410345, -0.3199659, -0.53747247, -0.77669417])

        """
        if representation.lower() not in ['rotmat', 'quaternion']:
            raise ValueError("Wrong representation type. Try 'rotmat', or 'quaternion'")
        w1, w2 = np.copy(w1), np.copy(w2)
        # Normalized Vectors
        w1 /= np.linalg.norm(w1)                            # (eq. 12-39a)
        w2 /= np.linalg.norm(w2)
        # First Triad
        w1xw2 = np.cross(w1, w2)
        s2 = w1xw2 / np.linalg.norm(w1xw2)                  # (eq. 12-39b)
        s3 = np.cross(w1, w1xw2) / np.linalg.norm(w1xw2)    # (eq. 12-39c)
        # Second Triad
        v1xv2 = np.cross(self.v1, self.v2)
        r2 = v1xv2 / np.linalg.norm(v1xv2)
        r3 = np.cross(self.v1, v1xv2) / np.linalg.norm(v1xv2)
        # Solve TRIAD
        Mb = np.c_[w1, s2, s3]                              # (eq. 12-41)
        Mr = np.c_[self.v1, r2, r3]                         # (eq. 12-42)
        A = Mb@Mr.T                                         # (eq. 12-45)
        # Return according to desired representation
        if representation.lower() == 'quaternion':
            return chiaverini(A)
        return A

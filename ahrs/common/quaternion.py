# -*- coding: utf-8 -*-
"""
Quaternion
==========

Quaternions were initially defined by `William Hamilton <https://en.wikipedia.org/wiki/History_of_quaternions>`_
in 1843 to describe a `Cayley-Dickson construction <https://en.wikipedia.org/wiki/Cayley%E2%80%93Dickson_construction>`_
in four dimensions.

Since then, many interpretations have appeared for different applications. The
most common definition of a quaternion :math:`\\mathbf{q}` is as an ordered
expression of the form:

.. math::
    \\mathbf{q} = w + xi + yj + zk

where :math:`w`, :math:`x`, :math:`y` and :math:`z` are real numbers, and
:math:`i`, :math:`j` and :math:`k` are three imaginary unit numbers defined so
that [Sola]_ [Kuipers]_ [WikiQuaternion]_ :

.. math::
    i^2 = j^2 = k^2 = ijk = -1

It is helpful to notice that quaternions are arrays with the same structure: a
real number :math:`w` followed by three pairs of complex numbers. Thus, we can
split it to find a second definition:

.. math::
    \\mathbf{q} = w + v

where :math:`v=(xi + yj + zk)\\in\\mathbb{Z}` and :math:`w\\in\\mathbb{R}`.
Recognizing that :math:`i`, :math:`j` and :math:`k` are always the same, we can
omit them to redefine the quaternion as a pair of a scalar and a vector:

.. math::
    \\mathbf{q} = q_w + \\mathbf{q}_v

where :math:`q_w\\in\\mathbb{R}` is called the *real* or *scalar* part, while
:math:`\\mathbf{q}_v = (q_x, q_y, q_z)\\in\\mathbb{R}^3` is the *imaginary* or
*vector* part.

For practical reasons, most literature will use the **vector definition** of a
quaternion [#]_:

.. math::
    \\mathbf{q}\\triangleq
    \\begin{bmatrix}q_w \\\\ \\mathbf{q}_v\\end{bmatrix} = 
    \\begin{bmatrix}q_w \\\\ q_x \\\\ q_y \\\\ q_z\\end{bmatrix}

Sadly, many authors use different notations for the same type of quaternions.
Some even invert their order, with the vector part first followed by the scalar
part, increasing the confusion among readers. Here, the definition above will
be used throughout the package.

Let's say, for example, we want to use the quaternion :math:`\\mathbf{q}=\\begin{pmatrix}0.7071 & 0 & 0.7071 & 0\\end{pmatrix}`
with this class:

.. code:: python

    >>> from ahrs import Quaternion
    >>> q = Quaternion([0.7071, 0.0, 0.7071, 0.0])
    >>> q
    Quaternion([0.70710678, 0.        , 0.70710678, 0.        ])

This will *extend* the values of the quaternion, because it is handled as a
rotation operator. Something explained in a moment.

By the way, if you want, you can have a *pretty* formatting of the quaternion
if typecasted as a string:

.. code:: python

    >>> str(q)
    '(0.7071 +0.0000i +0.7071j +0.0000k)'

As Rotation Operators
---------------------

Quaternions can be defined in the geometric space as an alternative form of a
**rotation operator**, so that we can find the image :math:`\\mathbf{a}'` of
some vector :math:`\\mathbf{a}` using a product similar to other euclidean
transformations in 3D:

.. math::
    \\mathbf{a}' = \\mathbf{qa}

Back in the XVIII century `Leonhard Euler <https://en.wikipedia.org/wiki/Leonhard_Euler>`_
showed that any **rotation around the origin** can be described by a
three-dimensional axis and a rotation magnitude.
`Euler's Rotation Theorem <https://en.wikipedia.org/wiki/Euler%27s_rotation_theorem>`_
is the basis for most three-dimensional rotations out there.

Later, in 1840, `Olinde Rodrigues <https://en.wikipedia.org/wiki/Olinde_Rodrigues>`_
came up with a formula using Euler's principle in vector form:

.. math::
    \\mathbf{a}' = \\mathbf{a}\\cos\\theta + (\\mathbf{v}\\times\\mathbf{a})\\sin\\theta + \\mathbf{v}(\\mathbf{v}\\cdot\\mathbf{a})(1-\\cos\\theta)

where :math:`\\mathbf{v}=\\begin{bmatrix}v_x & v_y & v_z\\end{bmatrix}` is a
**unit vector** [#]_ describing the axis of rotation about which
:math:`\\mathbf{a}` rotates by an angle :math:`\\theta`.

The `Euler-Rodrigues Formula <https://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_formula>`_
can be further compacted to:

.. math::
    \\mathbf{a}' = \\mathbf{a} + 2\\alpha (\\vec{\\mathbf{v}}\\times\\mathbf{a}) + 2\\big(\\vec{\\mathbf{v}}\\times(\\vec{\\mathbf{v}}\\times\\mathbf{a})\\big)

where :math:`\\alpha = \\cos\\frac{\\theta}{2}` and :math:`\\vec{\\mathbf{v}}`
represents the vector of rotation as half angles:

.. math::
    \\vec{\\mathbf{v}} =
    \\mathbf{v}\\sin\\frac{\\theta}{2} =
    \\begin{bmatrix}
    v_x \\sin\\frac{\\theta}{2} \\\\ v_y \\sin\\frac{\\theta}{2} \\\\ v_z \\sin\\frac{\\theta}{2}
    \\end{bmatrix}

This rotation is defined by a *scalar* and *vector* pair. Now we can see that
the structure of a quaternion can be used to describe a rotation with an
`axis-angle representation <https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation>`_
such that:

.. math::
    \\mathbf{q} = \\begin{bmatrix}q_w \\\\ q_x \\\\ q_y \\\\ q_z\\end{bmatrix} =
    \\begin{bmatrix}
    \\cos\\frac{\\theta}{2} \\\\
    v_x \\sin\\frac{\\theta}{2} \\\\
    v_y \\sin\\frac{\\theta}{2} \\\\
    v_z \\sin\\frac{\\theta}{2}
    \\end{bmatrix} =
    \\begin{bmatrix}
    \\cos\\frac{\\theta}{2} \\\\
    \\mathbf{v} \\sin\\frac{\\theta}{2}
    \\end{bmatrix}

This is **only** valid if the quaternion is normalized, in which case it is
also known as a `versor <https://en.wikipedia.org/wiki/Versor>`_. To enforce
the versor, we normalize the quaternion:

.. math::
    \\mathbf{q} = \\frac{1}{\\sqrt{q_w^2+q_x^2+q_y^2+q_z^2}}\\begin{bmatrix}q_w \\\\ q_x \\\\ q_y \\\\ q_z\\end{bmatrix}

In this module the quaternions are considered rotation operators, so they will
**always** be normalized. From the example above:

.. code::

    >>> q = Quaternion([0.7071, 0.0, 0.7071, 0.0])
    >>> q
    Quaternion([0.70710678, 0.        , 0.70710678, 0.        ])
    >>> import numpy as np
    >>> np.linalg.norm(q)
    1.0

Very convenient conversion methods are reachable in this class. One of them is
the representation of `direction cosine matrices <https://en.wikipedia.org/wiki/Direction_cosine>`_
from the created quaternion.

.. code:: python

    >>> q.to_DCM()
    array([[-2.22044605e-16,  0.00000000e+00,  1.00000000e+00],
           [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
           [-1.00000000e+00,  0.00000000e+00, -2.22044605e-16]])

Some important observations here help us to clarify this further:

* When :math:`\\theta=0` (no rotation) the parametrization becomes
  :math:`\\mathbf{q} = \\begin{pmatrix}1 & 0 & 0 & 0\\end{pmatrix}`. This is
  identified as an **identity quaternion**.
* When :math:`\\theta=180` (half-circle), then :math:`q_w=0`, making
  :math:`\\mathbf{q}` a **pure quaternion**.
* The negative values of a versor, :math:`\\begin{pmatrix}-q_w & -q_x & -q_y & -q_z\\end{pmatrix}`
  represent the *same rotation* [Grosskatthoefer]_.

.. code:: python

    >>> q = Quaternion([1.0, 0.0, 0.0, 0.0])
    >>> q.is_identity()
    True
    >>> q = Quaternion([0.0, 1.0, 2.0, 3.0])
    >>> q
    Quaternion([0.        , 0.26726124, 0.53452248, 0.80178373])
    >>> q.is_pure()
    True
    >>> q1 = Quaternion([1.0, 2.0, 3.0, 4.0])
    >>> q2 = Quaternion([-1.0, -2.0, -3.0, -4.0])
    >>> np.all(q1.to_DCM()==q2.to_DCM())
    True

To summarize, unit quaternions can also be defined using Euler's rotation
theorem [#]_ in the form [Kuipers]_:

.. math::
    \\mathbf{q} = \\begin{bmatrix}\\cos\\theta \\\\ \\mathbf{v}\\sin\\theta\\end{bmatrix}

And they have similar algebraic characteristics as normal vectors [Sola]_ [Eberly]_.
For example, given two quaternions :math:`\\mathbf{p}` and :math:`\\mathbf{q}`:

.. math::
    \\mathbf{p} \\pm \\mathbf{q} = \\begin{bmatrix}p_w\\pm q_w \\\\ \\mathbf{p}_v \\pm \\mathbf{q}_v \\end{bmatrix}

.. code:: python

    >>> p = Quaternion([1., 2., 3., 4.])
    >>> p
    Quaternion([0.18257419, 0.36514837, 0.54772256, 0.73029674])
    >>> q = Quaternion([-5., 4., -3., 2.])
    >>> q
    Quaternion([-0.68041382,  0.54433105, -0.40824829,  0.27216553])
    >>> p+q
    Quaternion([-0.34359264,  0.62769298,  0.09626058,  0.6918667 ])
    >>> p-q
    Quaternion([ 0.62597531, -0.1299716 ,  0.69342116,  0.33230917])

The **quaternion product** uses the `Hamilton product <https://en.wikipedia.org/wiki/Quaternion#Hamilton_product>`_
to perform their multiplication [#]_, which can be represented in vector form,
as a known scalar-vector form, or even as a matrix multiplication:

.. math::
    \\begin{array}{rl}
    \\mathbf{pq} &=
    \\begin{bmatrix}
        p_wq_w-\\mathbf{p}_v^T\\mathbf{q}_v \\\\
        p_w\\mathbf{q}_v + q_w\\mathbf{p}_v + \\mathbf{p}_v\\times\\mathbf{q}_v
    \\end{bmatrix} \\\\ &=
    \\begin{bmatrix}
        p_w q_w - p_x q_x - p_y q_y - p_z q_z \\\\
        p_w q_x + p_x q_w + p_y q_z - p_z q_y \\\\
        p_w q_y - p_x q_z + p_y q_w + p_z q_x \\\\
        p_w q_z + p_x q_y - p_y q_x + p_z q_w
    \\end{bmatrix} \\\\ &=
    \\begin{bmatrix}
        p_w & -p_x & -p_y & -p_z \\\\
        p_x &  p_w & -p_z &  p_y \\\\
        p_y &  p_z &  p_w & -p_x \\\\
        p_z & -p_y &  p_x &  p_w
    \\end{bmatrix}
    \\begin{bmatrix} q_w \\\\ q_x \\\\ q_y \\\\ q_z \\end{bmatrix} \\\\ &=
    \\begin{bmatrix}
        p_w & -\\mathbf{p}_v^T \\\\ \\mathbf{p}_v & p_w \\mathbf{I}_3 + \\lfloor \\mathbf{p}_v \\rfloor_\\times
    \\end{bmatrix}
    \\begin{bmatrix} q_w \\\\ \\mathbf{q}_v \\end{bmatrix}
    \\end{array}

which is **not commutative**

.. math::
    \\mathbf{pq} \\neq \\mathbf{qp}

.. code:: python

    >>> p*q
    array([-0.2981424 ,  0.2981424 , -0.1490712 , -0.89442719])
    >>> q*p
    array([-2.98142397e-01, -5.96284794e-01, -7.45355992e-01,  4.16333634e-17])

The **conjugate** of the quaternion, defined as :math:`\\mathbf{q}^*=\\begin{pmatrix}q_w & -\\mathbf{q}_v\\end{pmatrix}`,
has the interesting property of:

.. math::
    \\mathbf{qq}^* = \\mathbf{q}^*\\mathbf{q} = \\begin{bmatrix}q_w^2+q_x^2+q_y^2+q_z^2 \\\\ \\mathbf{0}_v\\end{bmatrix}

But for the case, where the quaternion is a versor, like in this module, the
conjugate is actually the same as the inverse :math:`\\mathbf{q}^{-1}`, where:

.. math::
    \\mathbf{qq}^* = \\mathbf{qq}^{-1} = \\mathbf{q}^{-1}\\mathbf{q} = \\begin{pmatrix}1 & 0 & 0 & 0\\end{pmatrix}

.. code::

    >>> q*q.conj
    array([1., 0., 0., 0.])
    >>> q*q.inv
    array([1., 0., 0., 0.])

Rotation groups are normally represented with direction cosine matrices. It is
undeniable that they are the best way to express any rotation operation.
However, quaternions are also a good representation of it, and even a better
ally for numerical operations.

Footnotes
---------
.. [#] Some authors use different subscripts, but here we mainly deal with
    geometric transformations and the notation (w, x, y, z) is preferred.
.. [#] Any unit vector :math:`\\mathbf{v}\\in\\mathbb{R}^3` has, per definition,
    a magnitude equal to 1, which means :math:`\\|\\mathbf{v}\\|=1`.
.. [#] This is the reason why some early authors call the quaternions *Euler Parameters*.
.. [#] Many authors decide to use the symbol :math:`\\otimes` to indicate a
    quaternion product, but here is not used in order to avoid any confusions
    with the `outer product <https://en.wikipedia.org/wiki/Outer_product>`_.

References
----------
.. [Bar-Itzhack] Y. Bar-Itzhack. New method for Extracting the Quaternion from
    a Rotation Matrix. Journal of Guidance, Control, and Dynamics,
    23(6):1085–1087, 2000. (https://arc.aiaa.org/doi/abs/10.2514/2.4654)
.. [Chiaverini] S. Chiaverini & B. Siciliano. The Unit Quaternion: A Useful
    Tool for Inverse Kinematics of Robot Manipulators. Systems Analysis
    Modelling Simulation. May 1999.
    (https://www.researchgate.net/publication/262391661)
.. [Dantam] Dantam, N. (2014) Quaternion Computation. Institute for Robotics
    and Intelligent Machines. Georgia Tech. (http://www.neil.dantam.name/note/dantam-quaternion.pdf)
.. [Eberly] Eberly, D. (2010) Quaternion Algebra and Calculus. Geometric Tools.
    https://www.geometrictools.com/Documentation/Quaternions.pdf
.. [Grosskatthoefer] K. Grosskatthoefer. Introduction into quaternions from
    spacecraft attitude representation. TU Berlin. 2012.
    (http://www.tu-berlin.de/fileadmin/fg169/miscellaneous/Quaternions.pdf)
.. [Hughes] P. Hughes. Spacecraft Attitude Dynamics. 1986. p. 18
.. [Kuipers] Kuipers, Jack. Quaternions and Rotation Sequences. Princenton
    University Press. 1999.
.. [Markley2007] F. Landis Markley. Averaging Quaternions. Journal of Guidance,
    Control, and Dynamics. Vol 30, Num 4. 2007
    (https://arc.aiaa.org/doi/abs/10.2514/1.28949)
.. [Sarabandi] Sarabandi, S. et al. (2018) Accurate Computation of Quaternions
    from Rotation Matrices.
    (http://www.iri.upc.edu/files/scidoc/2068-Accurate-Computation-of-Quaternions-from-Rotation-Matrices.pdf)
.. [Sarkka] Särkkä, S. (2007) Notes on Quaternions (https://users.aalto.fi/~ssarkka/pub/quat.pdf)
.. [Shepperd] S.W. Shepperd. "Quaternion from rotation matrix." Journal of
    Guidance and Control, Vol. 1, No. 3, pp. 223-224, 1978.
    (https://arc.aiaa.org/doi/10.2514/3.55767b)
.. [Shoemake] K. Shoemake. Uniform random rotations. Graphics Gems III, pages
    124-132. Academic, New York, 1992.
.. [Sola] Solà, Joan. Quaternion kinematics for the error-state Kalman Filter.
    October 12, 2017. (http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf)
.. [WikiConversions] https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
.. [WikiQuaternion] https://en.wikipedia.org/wiki/Quaternion

"""

import numpy as np
from typing import Type, Union, NoReturn, Any, Tuple, List
# Functions to convert DCM to quaternion representation
from .orientation import shepperd
from .orientation import hughes
from .orientation import chiaverini
from .orientation import itzhack
from .orientation import sarabandi

def slerp(q0: np.ndarray, q1: np.ndarray, t_array: np.ndarray, threshold: float = 0.9995) -> np.ndarray:
    """Spherical Linear Interpolation between two quaternions.

    Return a valid quaternion rotation at a specified distance along the minor
    arc of a great circle passing through any two existing quaternion endpoints
    lying on the unit radius hypersphere.

    Based on the method detailed in [Wiki_SLERP]_

    Parameters
    ----------
    q0 : NumPy array
        First endpoint quaternion.
    q1 : NumPy array
        Second endpoint quaternion.
    t_array : NumPy array
        Array of times to interpolate to.
    threshold : float, default: 0.9995
        Threshold to closeness of interpolation.

    Returns
    -------
    q : array
        New quaternion representing the interpolated rotation.

    References
    ----------
    .. [Wiki_SLERP] https://en.wikipedia.org/wiki/Slerp

    """
    qdot = np.dot(q0, q1)
    # Ensure SLERP takes the shortest path
    if qdot < 0.0:
        q1 *= -1.0
        qdot *= -1.0
    # Interpolate linearly (LERP)
    if qdot > threshold:
        result = q0[np.newaxis, :] + t_array[:, np.newaxis]*(q1 - q0)[np.newaxis, :]
        return (result.T / np.linalg.norm(result, axis=1)).T
    # Angle between vectors
    theta_0 = np.arccos(qdot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0*t_array
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - qdot*sin_theta/sin_theta_0
    s1 = sin_theta/sin_theta_0
    return s0[:,np.newaxis]*q0[np.newaxis,:] + s1[:,np.newaxis]*q1[np.newaxis,:]

class Quaternion(np.ndarray):
    """
    Quaternion object

    Class to represent a quaternion. It can be built with 3- or 4-dimensional
    vectors. The quaternion objects are always normalized to represent
    rotations in 3D space, also known as versors.

    Parameters
    ----------
    q : array-like, default: None
        Vector to build the quaternion with. It can be either 3- or
        4-dimensional.

    Attributes
    ----------
    A : numpy.ndarray
        Array with the 4 elements of quaternion of the form [w, x, y, z]
    w : float
        Scalar part of the quaternion.
    x : float
        First element of the vector part of the quaternion.
    y : float
        Second element of the vector part of the quaternion.
    z : float
        Third element of the vector part of the quaternion.
    v : numpy.ndarray
        Vector part of the quaternion.

    Raises
    ------
    ValueError
        When length of input array is not equal to either 3 or 4.

    Examples
    --------
    >>> from ahrs import Quaternion
    >>> q = Quaternion([1., 2., 3., 4.])
    >>> str(q)
    '(0.1826 +0.3651i +0.5477j +0.7303k)'
    >>> x = [1., 2., 3.]
    >>> q.rot(x)
    [1.8 2.  2.6]
    >>> R = q.to_DCM()
    >>> R@x
    [1.8 2.  2.6]

    A call to method ``product()`` will return an array of a multiplied vector.

    >>> q1 = Quaternion([1., 2., 3., 4.])
    >>> q2 = Quaternion([5., 4., 3., 2.])
    >>> q1.product(q2)
    [-0.49690399  0.1987616   0.74535599  0.3975232 ]

    Multiplication operators are overriden to perform the expected hamilton
    product.

    >>> str(q1*q2)
    '(-0.4969 +0.1988i +0.7454j +0.3975k)'
    >>> str(q1@q2)
    '(-0.4969 +0.1988i +0.7454j +0.3975k)'

    Basic operators are also overriden.

    >>> str(q1+q2)
    '(0.4619 +0.4868i +0.5117j +0.5366k)'
    >>> str(q1-q2)
    '(-0.6976 -0.2511i +0.1954j +0.6420k)'

    Pure quaternions are built from arrays with three elements.

    >>> q = Quaternion([1., 2., 3.])
    >>> str(q)
    '(0.0000 +0.2673i +0.5345j +0.8018k)'
    >>> q.is_pure()
    True

    Conversions between representations are also possible

    >>> q.to_axang()
    (array([0.26726124, 0.53452248, 0.80178373]), 3.141592653589793)
    >>> q.to_angles()
    [ 1.24904577 -0.44291104  2.8198421 ]
    """
    def __new__(subtype, q: np.ndarray = None, **kwargs):
        if q is None:
            q = np.array([1.0, 0.0, 0.0, 0.0])
            if "angles" in kwargs:
                q = self.from_angles(kwargs["angles"])
            if "dcm" in kwargs:
                q = self.from_DCM(kwargs["dcm"])
            if "rpy" in kwargs:
                q = self.from_rpy(kwargs["rpy"])
        q = np.array(q, dtype=float)
        if q.ndim!=1 or q.shape[-1] not in [3, 4]:
            raise ValueError("Expected `q` to have shape (4,) or (3,), got {}.".format(q.shape))
        if q.shape[-1]==3:
            q = np.array([0.0, *q])
        q /= np.linalg.norm(q)
        # Create the ndarray instance of type Quaternion. This will call the
        # standard ndarray constructor, but return an object of type Quaternion.
        obj = super(Quaternion, subtype).__new__(subtype, q.shape, float, q)
        obj.A = q
        return obj

    @property
    def w(self) -> float:
        return self.A[0]

    @property
    def x(self) -> float:
        return self.A[1]

    @property
    def y(self) -> float:
        return self.A[2]

    @property
    def z(self) -> float:
        return self.A[3]

    @property
    def v(self) -> np.ndarray:
        return self.A[1:]

    @property
    def conjugate(self) -> np.ndarray:
        """
        Conjugate of quaternion

        A quaternion, whose form is :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`,
        has a conjugate of the form :math:`\\mathbf{q}^* = (q_w, -q_x, -q_y, -q_z)`.

        A product of the quaternion with its conjugate yields:

        .. math::
            \\mathbf{q}\\mathbf{q}^* =
            \\begin{bmatrix}q_w^2 + q_x^2 + q_y^2 + q_z^2\\\\ \\mathbf{0}_v \\end{bmatrix}

        A versor (normalized quaternion) multiplied with its own conjugate
        gives the identity quaternion back.

        .. math::
            \\mathbf{q}\\mathbf{q}^* =
            \\begin{bmatrix}1 & 0 & 0 & 0 \\end{bmatrix}

        Returns
        -------
        q* : numpy.array
            Conjugated quaternion.

        Examples
        --------
        >>> q = Quaternion([0.603297, 0.749259, 0.176548, 0.20850])
        >>> q.conjugate
        array([0.603297, -0.749259, -0.176548, -0.20850 ])

        """
        return self.A*np.array([1.0, -1.0, -1.0, -1.0])

    @property
    def conj(self) -> np.ndarray:
        """Synonym to property ``conjugate``

        Returns
        -------
        q* : numpy.ndarray
            Conjugated quaternion.

        Examples
        --------
        >>> q = Quaternion([0.603297, 0.749259, 0.176548, 0.20850])
        >>> q.conj
        array([0.603297, -0.749259, -0.176548, -0.20850 ])
        """
        return self.conjugate

    @property
    def inverse(self) -> np.ndarray:
        """
        Return the inverse Quaternion

        The inverse quaternion :math:`\\mathbf{q}^{-1}` is such that the
        quaternion times its inverse gives the identity quaternion
        :math:`\\mathbf{q}_I = (1, 0, 0, 0)`

        It is obtained as:

        .. math::

            \\mathbf{q}^{-1} = \\mathbf{q}^* / \\|\\mathbf{q}\\|^2

        If the quaternion is normalized (called 'versor') its inverse is the
        conjugate.

        .. math::

            \\mathbf{q}^{-1} = \\mathbf{q}^*

        Returns
        -------
        out : numpy.ndarray
            Inverse of quaternion

        Examples
        --------
        >>> q = Quaternion([1., -2., 3., -4.])
        >>> str(q)
        '(0.1826 -0.3651i +0.5477j -0.7303k)'
        >>> str(q.inverse)
        '[ 0.18257419  0.36514837 -0.54772256  0.73029674]'
        >>> q2 = q@q.inverse
        >>> str(q2)
        '(1.0000 +0.0000i +0.0000j +0.0000k)'
        """
        if self.is_versor():
            return self.conjugate
        return self.conjugate / np.linalg.norm(self.q)

    @property
    def inv(self) -> np.ndarray:
        """
        Synonym to property ``inverse``

        Returns
        -------
        out : numpy.ndarray
            Inverse of quaternion

        Examples
        --------
        >>> q = Quaternion([1., -2., 3., -4.])
        >>> str(q)
        '(0.1826 -0.3651i +0.5477j -0.7303k)'
        >>> q.inv
        '[ 0.18257419  0.36514837 -0.54772256  0.73029674]'
        >>> q2 = q@q.inv
        >>> str(q2)
        '(1.0000 +0.0000i +0.0000j +0.0000k)'
        """
        return self.inverse

    @property
    def exponential(self) -> np.ndarray:
        """
        Exponential of Quaternion

        The quaternion exponential works as in the ordinary case, defined with
        the absolute convergent power series:

        .. math::

            e^{\\mathbf{q}} = \\sum_{k=0}^{\\infty}\\frac{\\mathbf{q}^k}{k!}

        The exponential of **pure quaternions** is, with the help of Euler
        formula, redefined as:

        .. math::

            \\begin{eqnarray}
            e^{\\mathbf{q}_v} & = & \\sum_{k=0}^{\\infty}\\frac{\\mathbf{q}_v^k}{k!} \\\\
            & = &
            \\begin{bmatrix}
            \\cos \\theta \\\\
            \\mathbf{u} \\sin \\theta
            \\end{bmatrix}
            \\end{eqnarray}

        with :math:`\\mathbf{q}_v = \\mathbf{u}\\theta` and :math:`\\theta=\\|\mathbf{v}\\|`

        Since :math:`\\|e^{\\mathbf{q}_v}\\|^2=\\cos^2\\theta+\\sin^2\\theta=1`,
        the exponential of a pure quaternion is always a versor. Therefore, if
        the quaternion is real, its exponential is the identity.

        For **general quaternions** the exponential is defined using :math:`\\mathbf{u}\\theta=\\mathbf{q}_v`
        and the exponential of the pure quaternion:

        .. math::

            \\begin{eqnarray}
            e^{\\mathbf{q}} & = & e^{\\mathbf{q}_w+\\mathbf{q}_v} = e^{\\mathbf{q}_w}e^{\\mathbf{q}_v}\\\\
            & = & e^{\\mathbf{q}_w}
            \\begin{bmatrix}
            \\cos \\|\\mathbf{q}_v\\| \\\\
            \\frac{\\mathbf{q}}{\\|\\mathbf{q}_v\\|} \\sin \\|\\mathbf{q}_v\\|
            \\end{bmatrix}
            \\end{eqnarray}

        Returns
        -------
        exp : numpy.ndarray
            Exponential of quaternion

        Examples
        --------
        >>> q1 = Quaternion([0.0, -2.0, 3.0, -4.0])
        >>> str(q1)
        '(0.0000 -0.3714i +0.5571j -0.7428k)'
        >>> q1.exponential
        [ 0.54030231 -0.31251448  0.46877172 -0.62502896]
        >>> q2 = Quaternion([1.0, -2.0, 3.0, -4.0])
        >>> str(q2)
        '(0.1826 -0.3651i +0.5477j -0.7303k)'
        >>> q2.exponential
        [ 0.66541052 -0.37101103  0.55651655 -0.74202206]
        """
        if self.is_real():
            return np.array([1.0, 0.0, 0.0, 0.0])
        t = np.linalg.norm(self.v)
        u = self.v/t
        q_exp = np.array([np.cos(t), *u*np.sin(t)])
        if self.is_pure():
            return q_exp
        q_exp *= np.e**self.w
        return q_exp

    @property
    def exp(self) -> np.ndarray:
        """Synonym to method exponential()

        Returns
        -------
        exp : numpy.ndarray
            Exponential of quaternion

        Examples
        --------
        >>> q1 = Quaternion([0.0, -2.0, 3.0, -4.0])
        >>> str(q1)
        '(0.0000 -0.3714i +0.5571j -0.7428k)'
        >>> q1.exp
        [ 0.54030231 -0.31251448  0.46877172 -0.62502896]
        >>> q2 = Quaternion([1.0, -2.0, 3.0, -4.0])
        >>> str(q2)
        '(0.1826 -0.3651i +0.5477j -0.7303k)'
        >>> q2.exp
        [ 0.66541052 -0.37101103  0.55651655 -0.74202206]
        """
        return self.exponential

    @property
    def logarithm(self) -> np.ndarray:
        """
        Logarithm of Quaternion

        Return logarithm of normalized quaternion, which can be defined with
        the aid of the exponential of the quaternion.

        The logarithm of **pure quaternions** is obtained as:

        .. math::

            \\log \\mathbf{q} = \\log(e^{\\mathbf{u}\\theta}) = \\begin{bmatrix} 0 \\\\ \\mathbf{u}\\theta \\end{bmatrix}

        Similarly, for **general quaternions** the logarithm is:

        .. math::

            \\log \\mathbf{q} = \\log\\|\\mathbf{q}\\|+\\mathbf{u}\\theta
            = \\begin{bmatrix} \\log\\|\\mathbf{q}\\| \\\\ \\mathbf{u}\\theta \\end{bmatrix}

        Returns
        -------
        out : numpy.ndarray
            log(q)
        """
        v_norm = np.linalg.norm(self.v)
        u = self.v / v_norm
        if self.is_versor():
            return np.array([0.0, *u])
        t = np.arctan(v_norm/self.w)
        return np.array([np.log(np.linalg.norm(self.q)), *u*t])

    @property
    def log(self) -> np.ndarray:
        """Synonym to property ``logartihm``

        Returns
        -------
        out : numpy.ndarray
            log(q)
        """
        return self.logarithm

    def __str__(self) -> str:
        """
        Build a *printable* representation of quaternion

        Returns
        -------
        q : str
            Quaternion written as string.

        Examples
        --------
        >>> q = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])
        >>> str(q)
        '(0.5575 +0.1296i +0.5737j +0.5859k)'
        """
        return "({:-.4f} {:+.4f}i {:+.4f}j {:+.4f}k)".format(self.w, self.x, self.y, self.z)

    def __add__(self, p: Any):
        """
        Add quaternions

        Return the sum of two Quaternions. The given input must be of class
        Quaternion.

        Parameters
        ----------
        p : Quaternion
            Second Quaternion to sum. NOT an array.

        Returns
        -------
        q+p : Quaternion
            Normalized sum of quaternions

        Examples
        --------
        >>> q1 = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])
        >>> q2 = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709])
        >>> q3 = q1+q2
        >>> str(q3)
        '(0.5386 +0.3255i +0.5620j +0.5367k)'
        """
        return Quaternion(self.to_array() + p)

    def __sub__(self, p: Any):
        """
        Difference of quaternions

        Return the difference between two Quaternions. The given input must be
        of class Quaternion.

        Returns
        -------
        q-p : Quaternion
            Normalized difference of quaternions

        Examples
        --------
        >>> q1 = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])
        >>> q2 = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709])
        >>> q3 = q1-q2
        >>> str(q3)
        '(0.1482 -0.9358i +0.1152j +0.2983k)'
        """
        return Quaternion(self.to_array() - p)

    def __mul__(self, q: np.ndarray) -> Any:
        """
        Product between quaternions

        Given two unit quaternions :math:`\\mathbf{p}=(p_w, p_x, p_y, p_z)` and
        :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`, their product is obtained
        [Dantam]_ [MWQW]_ as:

        .. math::

            \\mathbf{pq} =
            \\begin{bmatrix}
            p_w q_w - p_x q_x - p_y q_y - p_z q_z \\\\
            p_x q_w + p_w q_x - p_z q_y + p_y q_z \\\\
            p_y q_w + p_z q_x + p_w q_y - p_x q_z \\\\
            p_z q_w - p_y q_x + p_x q_y + p_w q_z
            \\end{bmatrix}

        Parameters
        ----------
        q : numpy.ndarray, Quaternion
            Second quaternion to multiply with.

        Returns
        -------
        out : Quaternion
            Product of quaternions.

        Examples
        --------
        >>> q1 = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])
        >>> q2 = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709])
        >>> q1*q2
        array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])
        >>> q3 = q1*q2
        >>> q3
        <ahrs.common.quaternion.Quaternion object at 0x000001F379003748>
        >>> str(q3)
        '(-0.3635 +0.3896i +0.3419j +0.7740k)'
        """
        if not hasattr(q, 'A'):
            q = Quaternion(q)
        return self.product(q.A)

    def __matmul__(self, q: np.ndarray) -> np.ndarray:
        """
        Product between quaternions using @ operator

        Given two unit quaternions :math:`\\mathbf{p}=(p_w, p_x, p_y, p_z)` and
        :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`, their product is obtained
        [Dantam]_ [MWQW]_ as:

        .. math::
            \\mathbf{pq} =
            \\begin{bmatrix}
            p_w q_w - p_x q_x - p_y q_y - p_z q_z \\\\
            p_x q_w + p_w q_x - p_z q_y + p_y q_z \\\\
            p_y q_w + p_z q_x + p_w q_y - p_x q_z \\\\
            p_z q_w - p_y q_x + p_x q_y + p_w q_z
            \\end{bmatrix}

        Parameters
        ----------
        q : numpy.ndarray, Quaternion
            Second quaternion to multiply with.

        Returns
        -------
        out : Quaternion
            Product of quaternions.

        Examples
        --------
        >>> q1 = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])
        >>> q2 = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709])
        >>> q1@q2
        array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])
        >>> q3 = q1@q2
        >>> q3
        <ahrs.common.quaternion.Quaternion object at 0x000001F379003748>
        >>> str(q3)
        '(-0.3635 +0.3896i +0.3419j +0.7740k)'
        """
        if not hasattr(q, 'A'):
            q = Quaternion(q)
        return self.product(q.A)

    def __pow__(self, a: float) -> np.ndarray:
        """
        Returns array of quaternion to the power of ``a``

        Assuming the quaternion is a versor, its power can be defined using the
        exponential:

        .. math::
            \\begin{eqnarray}
            \\mathbf{q}^a & = & e^{\\log(\\mathbf{q}^a)} \\\\
            & = & e^{a \\log(\\mathbf{q})} \\\\
            & = & e^{a \\mathbf{u}\\theta} \\\\
            & = & \\begin{bmatrix}
            \\cos(a\\theta) \\\\
            \\mathbf{u} \\sin(a\\theta)
            \\end{bmatrix}
            \\end{eqnarray}

        Parameters
        ----------
        a : float
            Value to which to calculate quaternion power.

        Returns
        -------
        q^a : numpy.ndarray
            Quaternion :math:`\\mathbf{q}` to the power of ``a``
        """
        return np.e**(a*self.logarithm)

    def is_pure(self) -> bool:
        """
        Returns a bool value, where ``True`` if quaternion is pure.

        A pure quaternion has a scalar part equal to zero: :math:`\\mathbf{q} = 0 + xi + yj + zk`

        .. math::
            \\left\\{
            \\begin{array}{ll}
                \\mathrm{True} & \\: w = 0 \\\\
                \\mathrm{False} & \\: \\mathrm{otherwise}
            \\end{array}
            \\right.

        Returns
        -------
        out : bool
            Boolean equal to True if :math:`q_w = 0`.
        """
        return self.w==0.0

    def is_real(self) -> bool:
        """
        Returns a bool value, where ``True`` if quaternion is real.

        A real quaternion has all elements of its vector part equal to zero:
        :math:`\\mathbf{q} = w + 0i + 0j + 0k = (q_w, \\mathbf{0})`

        .. math::
            \\left\\{
            \\begin{array}{ll}
                \\mathrm{True} & \\: q_v = 0 \\\\
                \\mathrm{False} & \\: \\mathrm{otherwise}
            \\end{array}
            \\right.

        Returns
        -------
        out : bool
            Boolean equal to True if :math:`q_v = 0`.
        """
        return not any(self.v)

    def is_versor(self) -> bool:
        """
        Returns a bool value, where ``True`` if quaternion is a versor.

        A versor is a quaternion of norm equal to one:

        .. math::
            \\left\\{
            \\begin{array}{ll}
                \\mathrm{True} & \\: \\|\\mathbf{q}\\| = 1 \\\\
                \\mathrm{False} & \\: \\mathrm{otherwise}
            \\end{array}
            \\right.

        Returns
        -------
        out : bool
            Boolean equal to ``True`` if :math:`\\|\\mathbf{q}\\|=1`.
        """
        return np.isclose(np.linalg.norm(self.A), 1.0)

    def is_identity(self) -> bool:
        """
        Returns a bool value, where ``True`` if quaternion is identity quaternion.

        A quaternion is a quaternion if its scalar part is equal to 1, and the
        vector part is equal to 0:

        .. math::
            \\left\\{
            \\begin{array}{ll}
                \\mathrm{True} & \\: \\mathbf{q}\\ = \\begin{bmatrix} 1 & 0 & 0 & 0 \\end{bmatrix} \\\\
                \\mathrm{False} & \\: \\mathrm{otherwise}
            \\end{array}
            \\right.

        Returns
        -------
        out : bool
            Boolean equal to ``True`` if :math:`\\mathbf{q} = (1, \\mathbf{0})`.
        """
        return np.allclose(self.A, np.array([1.0, 0.0, 0.0, 0.0]))

    def normalize(self) -> NoReturn:
        """Normalize the quaternion
        """
        self.A /= np.linalg.norm(self.A)

    def product(self, q: np.ndarray) -> np.ndarray:
        """
        Product of two quaternions.

        Given two unit quaternions :math:`\\mathbf{p}=(p_w, \\mathbf{p}_v)` and
        :math:`\\mathbf{q} = (q_w, \\mathbf{q}_v)`, their product is defined
        [Sola]_ [Dantam]_ as:

        .. math::
            \\begin{eqnarray}
            \\mathbf{pq} & = & \\big( (q_w p_w - \\mathbf{q}_v \\cdot \\mathbf{p}_v) \\; ,
            \\; \\mathbf{q}_v \\times \\mathbf{p}_v + q_w \\mathbf{p}_v + p_w \\mathbf{q}_v \\big) \\\\
            & = &
            \\begin{bmatrix}
            p_w & -\\mathbf{p}_v^T \\\\ \\mathbf{p}_v & p_w \\mathbf{I}_3 + \\lfloor \\mathbf{p}_v \\rfloor_\\times
            \\end{bmatrix}
            \\begin{bmatrix} q_w \\\\ \\mathbf{q}_v \\end{bmatrix}
            \\\\
            & = &
            \\begin{bmatrix}
            p_w & -p_x & -p_y & -p_z \\\\
            p_x &  p_w & -p_z &  p_y \\\\
            p_y &  p_z &  p_w & -p_x \\\\
            p_z & -p_y &  p_x &  p_w
            \\end{bmatrix}
            \\begin{bmatrix} q_w \\\\ q_x \\\\ q_y \\\\ q_z \\end{bmatrix}
            \\\\
            & = &
            \\begin{bmatrix}
            p_w q_w - p_x q_x - p_y q_y - p_z q_z \\\\
            p_w q_x + p_x q_w + p_y q_z - p_z q_y \\\\
            p_w q_y - p_x q_z + p_y q_w + p_z q_x \\\\
            p_w q_z + p_x q_y - p_y q_x + p_z q_w
            \\end{bmatrix}
            \\end{eqnarray}

        where :math:`\\lfloor \\mathbf{a} \\rfloor_\\times` represents the
        `skew-symmetric matrix <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_
        of :math:`\\mathbf{a}`.

        Parameters
        ----------
        r : numpy.ndarray, Quaternion
            Quaternion to multiply with.

        Returns
        -------
        qr : numpy.ndarray
            Product of quaternions.

        Examples
        --------
        >>> q1 = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])

        Can multiply with a given quaternion in vector form...

        >>> q1.product([0.49753507, 0.50806522, 0.52711628, 0.4652709])
        array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])

        or with a Quaternion object...

        >>> q2 = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709 ])
        >>> q1.product(q2)
        array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])

        It holds with the result after the cross and dot product definition

        >>> qw = q1.w*q2.w - np.dot(q1.v, q2.v)
        >>> qv = q1.w*q2.v + q2.w*q1.v + np.cross(q1.v, q2.v)
        >>> qw, qv
        (-0.36348726, array([0.38962514,  0.34188103,  0.77407146]))

        """
        if isinstance(q, Quaternion):
            q = q.A.copy()
        q /= np.linalg.norm(q)
        return np.array([
            self.w*q[0] - self.x*q[1] - self.y*q[2] - self.z*q[3],
            self.w*q[1] + self.x*q[0] + self.y*q[3] - self.z*q[2],
            self.w*q[2] - self.x*q[3] + self.y*q[0] + self.z*q[1],
            self.w*q[3] + self.x*q[2] - self.y*q[1] + self.z*q[0]])

    def mult_L(self) -> np.ndarray:
        """
        Matrix form of a left-sided quaternion multiplication Q.

        Matrix representation of quaternion product with a left sided
        quaternion [Sarkka]_:

        .. math::
            \\mathbf{qp} = \\mathbf{L}(\\mathbf{q})\\mathbf{p} =
            \\begin{bmatrix}
            q_w & -q_x & -q_y & -q_z \\\\
            q_x &  q_w & -q_z &  q_y \\\\
            q_y &  q_z &  q_w & -q_x \\\\
            q_z & -q_y &  q_x &  q_w
            \\end{bmatrix}
            \\begin{bmatrix}p_w \\\\ p_x \\\\ p_y \\\\ p_z\\end{bmatrix}

        Returns
        -------
        Q : numpy.ndarray
            Matrix form of the left side quaternion multiplication.

        """
        return np.array([
            [self.w, -self.x, -self.y, -self.z],
            [self.x,  self.w, -self.z,  self.y],
            [self.y,  self.z,  self.w, -self.x],
            [self.z, -self.y,  self.x,  self.w]])

    def mult_R(self) -> np.ndarray:
        """
        Matrix form of a right-sided quaternion multiplication Q.

        Matrix representation of quaternion product with a right sided
        quaternion [Sarkka]_:

        .. math::
            \\mathbf{qp} = \\mathbf{R}(\\mathbf{p})\\mathbf{q} =
            \\begin{bmatrix}
            p_w & -p_x & -p_y & -p_z \\\\
            p_x &  p_w &  p_z & -p_y \\\\
            p_y & -p_z &  p_w &  p_x \\\\
            p_z &  p_y & -p_x &  p_w
            \\end{bmatrix}
            \\begin{bmatrix}q_w \\\\ q_x \\\\ q_y \\\\ q_z\\end{bmatrix}

        Returns
        -------
        Q : numpy.ndarray
            Matrix form of the right side quaternion multiplication.

        """
        return np.array([
            [self.w, -self.x, -self.y, -self.z],
            [self.x,  self.w,  self.z, -self.y],
            [self.y, -self.z,  self.w,  self.x],
            [self.z,  self.y, -self.x,  self.w]])

    def rotate(self, a: np.ndarray) -> np.ndarray:
        """Rotate array :math:`\\mathbf{a}` through quaternion :math:`\\mathbf{q}`.

        Parameters
        ----------
        a : numpy.ndarray
            3-by-N array to rotate in 3 dimensions, where N is the number of
            vectors to rotate.

        Returns
        -------
        a' : numpy.ndarray
            3-by-N rotated array by current quaternion.

        Examples
        --------
        >>> q = Quaternion([-0.00085769, -0.0404217, 0.29184193, -0.47288709])
        >>> v = [0.25557699 0.74814091 0.71491841]
        >>> q.rotate(v)
        array([-0.22481078 -0.99218916 -0.31806219])
        >>> A = [[0.18029565, 0.14234782], [0.47473686, 0.38233722], [0.90000689, 0.06117298]]
        >>> q.rotate(A)
        array([[-0.10633285 -0.16347163]
               [-1.02790041 -0.23738541]
               [-0.00284403 -0.29514739]])

        """
        a = np.array(a)
        if a.shape[0] != 3:
            raise ValueError("Expected `a` to have shape (3, N) or (3,), got {}.".format(a.shape))
        return self.to_DCM()@a

    def to_array(self) -> np.ndarray:
        """
        Return quaternion as a NumPy array

        Quaternion values are stored in attribute ``A``, which is a NumPy array.
        This method simply returns such attribute.

        Returns
        -------
        out : numpy.ndarray
            Quaternion

        Examples
        --------
        >>> q = Quaternion([0.0, -2.0, 3.0, -4.0])
        >>> q.to_array()
        array([ 0.         -0.37139068  0.55708601 -0.74278135])
        >>> type(q.to_array())
        <class 'numpy.ndarray'>
        """
        return self.A

    def to_list(self) -> list:
        """
        Return quaternion as list

        Quaternion values are stored in attribute ``A``, which is a NumPy array.
        This method reads that attribute and returns it as a list.

        Returns
        -------
        out : list
            Quaternion values in list

        Examples
        --------
        >>> q = Quaternion([0.0, -2.0, 3.0, -4.0])
        >>> q.to_list()
        [0.0, -0.3713906763541037, 0.5570860145311556, -0.7427813527082074]

        """
        return self.A.tolist()

    def to_axang(self) -> Tuple[np.ndarray, float]:
        """
        Return equivalent axis-angle representation of the quaternion.

        Returns
        -------
        axis : numpy.ndarray
            Three-dimensional axis to rotate about
        angle : float
            Amount of rotation, in radians, to rotate about.

        Examples
        --------
        >>> q = Quaternion([0.7071, 0.7071, 0.0, 0.0])
        >>> q.to_axang()
        (array([1., 0., 0.]), 1.5707963267948966)

        """
        denom = np.linalg.norm(self.v)
        angle = 2.0*np.arctan2(denom, self.w)
        axis = np.zeros(3) if angle==0.0 else self.v/denom
        return axis, angle

    def to_angles(self) -> np.ndarray:
        """
        Return corresponding Euler angles of quaternion.

        Given a unit quaternions :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`,
        its corresponding Euler angles [WikiConversions]_ are:

        .. math::

            \\begin{bmatrix}
            \\phi \\\\ \\theta \\\\ \\psi
            \\end{bmatrix} =
            \\begin{bmatrix}
            \\mathrm{atan2}\\big(2(q_wq_x + q_yq_z), 1-2(q_x^2+q_y^2)\\big) \\\\
            \\arcsin\\big(2(q_wq_y - q_zq_x)\\big) \\\\
            \\mathrm{atan2}\\big(2(q_wq_z + q_xq_y), 1-2(q_y^2+q_z^2)\\big)
            \\end{bmatrix}

        Returns
        -------
        angles : numpy.ndarray
            Euler angles of quaternion.

        """
        phi = np.arctan2(2.0*(self.w*self.x + self.y*self.z), 1.0 - 2.0*(self.x**2 + self.y**2))
        theta = np.arcsin(2.0*(self.w*self.y - self.z*self.x))
        psi = np.arctan2(2.0*(self.w*self.z + self.x*self.y), 1.0 - 2.0*(self.y**2 + self.z**2))
        return np.array([phi, theta, psi])

    def to_DCM(self) -> np.ndarray:
        """
        Return a Direction Cosine matrix :math:`\\mathbf{R} \\in SO(3)` from a
        given unit quaternion :math:`\\mathbf{q}`.

        The given unit quaternion :math:`\\mathbf{q}` must have the form
        :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`, where :math:`\\mathbf{q}_v = (q_x, q_y, q_z)`
        is the vector part, and :math:`q_w` is the scalar part.

        The resulting matrix :math:`\\mathbf{R}` [WikiConversions]_ has the
        form:

        .. math::

            \\mathbf{R}(\\mathbf{q}) =
            \\begin{bmatrix}
            1 - 2(q_y^2 + q_z^2) & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\\\
            2(q_xq_y + q_wq_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_yq_z - q_wq_x) \\\\
            2(q_xq_z - q_wq_y) & 2(q_wq_x + q_yq_z) & 1 - 2(q_x^2 + q_y^2)
            \\end{bmatrix}

        The identity Quaternion :math:`\\mathbf{q} = (1, 0, 0, 0)`, produces a
        a :math:`3 \\times 3` Identity matrix :math:`\\mathbf{I}_3`.

        Returns
        -------
        DCM : numpy.ndarray
            3-by-3 Direction Cosine Matrix

        """
        return np.array([
            [1.0-2.0*(self.y**2+self.z**2), 2.0*(self.x*self.y-self.w*self.z), 2.0*(self.x*self.z+self.w*self.y)],
            [2.0*(self.x*self.y+self.w*self.z), 1.0-2.0*(self.x**2+self.z**2), 2.0*(self.y*self.z-self.w*self.x)],
            [2.0*(self.x*self.z-self.w*self.y), 2.0*(self.w*self.x+self.y*self.z), 1.0-2.0*(self.x**2+self.y**2)]])

    def from_DCM(self, dcm: np.ndarray, method: str = 'chiaverini', **kw) -> NoReturn:
        """
        Quaternion from Direction Cosine Matrix.

        There are five methods available to obtain a quaternion from a
        Direction Cosine Matrix:

        * ``'chiaverini'`` as described in [Chiaverini]_
        * ``'hughes'`` as described in [Hughes]_
        * ``'itzhack'`` as described in [Bar-Itzhack]_
        * ``'sarabandi'`` as described in [Sarabandi]_
        * ``'shepperd'`` as described in [Shepperd]_

        Parameters
        ----------
        dcm : numpy.ndarray
            3-by-3 Direction Cosine Matrix.
        method : str, default: 'chiaverini'
            Method to use. Options are: 'chiaverini', 'hughes', 'itzhack',
            'sarabandi', and 'shepperd'.

        """
        if dcm.shape != (3, 3):
            raise TypeError("Expected matrix of size (3, 3). Got {}".format(dcm.shape))
        q = None
        if method.lower()=='hughes':
            q = hughes(dcm)
        if method.lower()=='chiaverini':
            q = chiaverini(dcm)
        if method.lower()=='shepperd':
            q = shepperd(dcm)
        if method.lower()=='itzhack':
            q = itzhack(dcm, version=kw.get('version', 3))
        if method.lower()=='sarabandi':
            q = sarabandi(dcm, eta=kw.get('threshold', 0.0))
        if q is None:
            raise KeyError("Given method '{}' is not implemented.".format(method))
        q /= np.linalg.norm(q)
        return q

    def from_rpy(self, angles: np.ndarray) -> NoReturn:
        """
        Quaternion from given RPY angles.

        Parameters
        ----------
        angles : numpy.ndarray
            3 cardanian angles in following order: roll -> pitch -> yaw.

        """
        angles = np.array(angles)
        if angles.ndim != 1 or angles.shape[0] != 3:
            raise ValueError("Expected `angles` to have shape (3,), got {}.".format(angles.shape))
        yaw, pitch, roll = angles
        cy = np.cos(0.5*yaw)
        sy = np.sin(0.5*yaw)
        cp = np.cos(0.5*pitch)
        sp = np.sin(0.5*pitch)
        cr = np.cos(0.5*roll)
        sr = np.sin(0.5*roll)
        q = np.zeros(4)
        q[0] = cy*cp*cr + sy*sp*sr
        q[1] = cy*cp*sr - sy*sp*cr
        q[2] = sy*cp*sr + cy*sp*cr
        q[3] = sy*cp*cr - cy*sp*sr
        return q

    def from_angles(self, angles: np.ndarray) -> NoReturn:
        """
        Synonym to method from_rpy()

        Parameters
        ----------
        angles : numpy.ndarray
            3 cardanian angles in following order: roll -> pitch -> yaw.
        """
        return self.from_rpy(angles)

    def ode(self, w: np.ndarray) -> np.ndarray:
        """
        Ordinary Differential Equation of the quaternion.

        Parameters
        ----------
        w : numpy.ndarray
            Angular velocity, in rad/s, about X-, Y- and Z-axis.

        Returns
        -------
        dq/dt : numpy.ndarray
            Derivative of quaternion

        """
        if w.ndim != 1 or w.shape[0] != 3:
            raise ValueError("Expected `w` to have shape (3,), got {}.".format(w.shape))
        F = np.array([
            [0.0, -w[0], -w[1], -w[2]],
            [w[0], 0.0, -w[2], w[1]],
            [w[1], w[2], 0.0, -w[0]],
            [w[2], -w[1], w[0], 0.0]])
        return 0.5*F@self.A

    def random(self) -> np.ndarray:
        """
        Generate a random quaternion

        To generate a random quaternion a mapping in SO(3) is first created and
        then transformed as explained originally by [Shoemake]_.

        Returns
        -------
        q : numpy.ndarray
            Random array corresponding to a valid quaternion
        """
        u = np.random.random(3)
        q = np.zeros(4)
        s2pi = np.sin(2.0*np.pi)
        c2pi = np.cos(2.0*np.pi)
        q[0] = np.sqrt(1.0-u[0])*s2pi*u[1]
        q[1] = np.sqrt(1.0-u[0])*c2pi*u[1]
        q[2] = np.sqrt(u[0])*s2pi*u[2]
        q[3] = np.sqrt(u[0])*c2pi*u[2]
        return q / np.linalg.norm(q)

class QuaternionArray:
    """Array of Quaternions

    Class to represent quaternion arrays. It can be built with N-by-3 or N-by-4
    NumPy arrays. The built objects are always normalized to represent
    rotations in 3D space.
    """
    def __init__(self, q: np.array = None, **kw):
        self.array = np.array([[1.0], [0.0], [0.0], [0.0]])
        self.num_qts = 1
        if q is not None:
            sz = q.shape
            if 2 < sz[-1] < 5:
                self.num_qts = sz[0]
                self.array = self._build_pure(q) if sz[-1] == 3 else q
                # Normalize Quaternions
                self.array /= np.linalg.norm(self.array, axis=1)[:, None]
            else:
                raise ValueError("Expected array to have shape (N, 3) or (N, 4), got {}.".format(q.shape))

    def _build_pure(self, X: np.ndarray) -> np.ndarray:
        """
        Build pure quaternions from 3-dimensional vectors

        Parameters
        ----------
        X : NumPy array
            N-by-3 array with values of vector part of pure quaternions.
        """
        return np.c_[np.zeros(X.shape[0]), X]

    def conjugate(self) -> np.ndarray:
        """
        Return the conjugate of all quaternions

        Returns
        -------
        q* : array
            Array of conjugated quaternions.
        """
        return self.array*np.array([1.0, -1.0, -1.0, -1.0])

    def conj(self) -> np.ndarray:
        """Synonym to method conjugate()
        """
        return self.conjugate()

    def to_DCM(self) -> np.ndarray:
        """
        Return N direction cosine matrices in SO(3) from a given Quaternion
        array, where N is the number of quaternions.

        The default values are identity quaternions, which produce N 3-by-3
        Identity matrices.

        Parameters
        ----------
        q : array
            N-by-4 array of Quaternions, where N is the number of quaternions.

        References
        ----------
        - https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        """
        R = np.zeros((self.num_qts, 3, 3))
        R[:, 0, 0] = 1.0 - 2.0*(self.array[:, 2]**2 + self.array[:, 3]**2)
        R[:, 1, 0] = 2.0*(self.array[:, 1]*self.array[:, 2]+self.array[:, 0]*self.array[:, 3])
        R[:, 2, 0] = 2.0*(self.array[:, 1]*self.array[:, 3]-self.array[:, 0]*self.array[:, 2])
        R[:, 0, 1] = 2.0*(self.array[:, 1]*self.array[:, 2]-self.array[:, 0]*self.array[:, 3])
        R[:, 1, 1] = 1.0 - 2.0*(self.array[:, 1]**2 + self.array[:, 3]**2)
        R[:, 2, 1] = 2.0*(self.array[:, 0]*self.array[:, 1]+self.array[:, 2]*self.array[:, 3])
        R[:, 0, 2] = 2.0*(self.array[:, 1]*self.array[:, 3]+self.array[:, 0]*self.array[:, 2])
        R[:, 1, 2] = 2.0*(self.array[:, 2]*self.array[:, 3]-self.array[:, 0]*self.array[:, 1])
        R[:, 2, 2] = 1.0 - 2.0*(self.array[:, 1]**2 + self.array[:, 2]**2)
        return R

    def average(self, span: Tuple[int, int] = None, w: np.ndarray = None) -> np.ndarray:
        """Average quaternion using Markley's method [Markley2007]_

        The average quaternion is the eigenvector of `M` corresponding to the
        maximum eigenvalue, where:

        .. math::
            \\mathbf{M} = \\mathbf{q}^T \\cdot \\mathbf{q}

        is a 4-by-4 matrix

        Parameters
        ----------
        span : tuple, default: None
            Span of data to average. If none given, it averages all.
        w : numpy.ndarray, default: None
            Weights of each quaternion. If none given, they are equal to 1.

        Returns
        -------
        q : numpy.ndarray
            Average quaternion
        """
        q = self.array.copy()
        if span is not None:
            if hasattr(span, '__iter__') and len(span)==2:
                q = q[span[0], span[1]]
            else:
                raise ValueError("span must be a pair of integers indicating the indices of the data.")
        if w is not None:
            q *= w
        eigvals, eigvecs = np.linalg.eig(q.T@q)
        return eigvecs[:, eigvals.argmax()]

    def remove_jumps(self) -> NoReturn:
        """
        Flip sign jumps on quaternions

        Some estimations and measurements of quaternions might have "jumps"
        produced when their values are multiplied by -1. They still represent
        the same rotation, but the continuity of the signal "flips", making it
        difficult to evaluate continuously.

        To revert this, the flipping instances are identified and the next
        samples are multiplied by -1, until it "flips back". This
        function does that correction over all values of the attribute 'array'.
        """
        q_diff = np.diff(self.array, axis=0)
        jumps = np.nonzero(np.where(np.linalg.norm(q_diff, axis=1)>1, 1, 0))[0]+1
        if len(jumps)%2:
            jumps = np.append(jumps, [len(q_diff)+1])
        jump_pairs = jumps.reshape((len(jumps)//2, 2))
        for j in jump_pairs:
            self.array[j[0]:j[1]] *= -1.0

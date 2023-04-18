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
rotation operator.

.. tip::

    You can have a *pretty formatting* of the quaternion if typecasted as a
    string, showing it with Hamilton's notation:

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
.. [#] This is the reason why some early authors call the quaternions *Euler
    Parameters*.
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
.. [Kuffner] James J. Kuffner. Effective Sampling and Distance Metrics for 3D
    Rigid Body Path Planning. Proc. 2004 IEEE International Conference on
    Robotics and Automation. 2004.
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
.. [Wiki_SLERP] https://en.wikipedia.org/wiki/Slerp
.. [MarioGC1] https://mariogc.com/post/angular-velocity-quaternions/

"""

import numpy as np
from typing import Union, Tuple

# Functions to convert DCM to quaternion representation
from .orientation import shepperd
from .orientation import hughes
from .orientation import chiaverini
from .orientation import itzhack
from .orientation import sarabandi

def _assert_iterables(item, item_name: str = 'iterable'):
    if not isinstance(item, (list, tuple, np.ndarray)):
        raise TypeError(f"{item_name} must be given as an array, got {type(item)}")

def slerp(q0: np.ndarray, q1: np.ndarray, t_array: np.ndarray, threshold: float = 0.9995) -> np.ndarray:
    """
    Spherical Linear Interpolation between two quaternions.

    Return a valid quaternion rotation at a specified distance along the minor
    arc of a great circle passing through any two existing quaternion endpoints
    lying on the unit radius hypersphere.

    Based on the method detailed in [Wiki_SLERP]_

    Parameters
    ----------
    q0 : numpy.ndarray
        First endpoint quaternion.
    q1 : numpy.ndarray
        Second endpoint quaternion.
    t_array : numpy.ndarray
        Array of times to interpolate to.
    threshold : float, default: 0.9995
        Threshold to closeness of interpolation.

    Returns
    -------
    q : numpy.ndarray
        New quaternion representing the interpolated rotation.

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
    return s0[:, np.newaxis]*q0[np.newaxis, :] + s1[:, np.newaxis]*q1[np.newaxis, :]

def random_attitudes(n: int = 1, representation: str = 'quaternion') -> np.ndarray:
    """
    Generate random attitudes

    To generate a random quaternion a mapping in SO(3) is first created and
    then transformed as explained originally by [Shoemake]_ and summarized in
    [Kuffner]_.

    Parameters
    ----------
    n : int, default: 1
        Number of random atitudes to generate. Default is 1.
    representation : str, default: ``'quaternion'``
        Attitude representation. Options are ``'quaternion'`` or ``'rotmat'``.

    Returns
    -------
    Q : numpy.ndarray
        Array of n random quaternions.
    """
    if not isinstance(n, int):
        raise TypeError(f"n must be an integer. Got {type(n)}")
    if n < 1:
        raise ValueError(f"n must be greater than 0. Got {n}")
    if not isinstance(representation, str):
        raise TypeError(f"representation must be a string. Got {type(representation)}")
    if representation.lower() not in ['rotmat', 'quaternion']:
        raise ValueError(f"Given representation '{representation}' is NOT valid. Try 'rotmat', or 'quaternion'")
    u = np.random.random((3, n))
    s1 = np.sqrt(1.0 - u[0])
    s2 = np.sqrt(u[0])
    t1 = 2.0 * np.pi * u[1]
    t2 = 2.0 * np.pi * u[2]
    Q = np.zeros((n, 4))
    Q[:, 0] = s2 * np.cos(t2)
    Q[:, 1] = s1 * np.sin(t1)
    Q[:, 2] = s1 * np.cos(t1)
    Q[:, 3] = s2 * np.sin(t2)
    if n < 2:
        q = Q.flatten()
        q /= np.linalg.norm(q)
        if representation.lower() == 'rotmat':
            return Quaternion(q).to_DCM()
        return q
    Q = Q / np.linalg.norm(Q, axis=1)[:, None]
    if representation.lower() == 'rotmat':
        return QuaternionArray(Q).to_DCM()
    return Q

class Quaternion(np.ndarray):
    """
    Representation of a quaternion. It can be built with 3- or 4-dimensional
    vectors. The quaternion object is always normalized to represent rotations
    in 3D space, also known as a **versor**.

    Parameters
    ----------
    q : array-like, default: None
        Vector to build the quaternion with. It can be either 3- or
        4-dimensional.
    versor : bool, default: True
        Treat the quaternion as versor. It will normalize it immediately.
    dcm : array-like
        Create quaternion object from a 3-by-3 rotation matrix. It is built
        only if no array was given to build.
    rpy : array-like
        Create quaternion object from roll-pitch-yaw angles. It is built only
        if no array was given to build.
    order : str, default: 'H'
        Specify the layout of the Quaternion, where the scalar part precedes
        the vector part by default. If the order is 'S' the vector part
        precedes the scalar part. The default is 'H' for a Hamiltonian notation
        with the scalar part first.

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
    >>> q
    Quaternion([0.18257419, 0.36514837, 0.54772256, 0.73029674])
    >>> x = [1., 2., 3.]
    >>> q.rot(x)
    [1.8 2.  2.6]
    >>> R = q.to_DCM()
    >>> R@x
    [1.8 2.  2.6]

    A call to method :meth:`product` will return an array of a multiplied
    vector.

    >>> q1 = Quaternion([1., 2., 3., 4.])
    >>> q2 = Quaternion([5., 4., 3., 2.])
    >>> q1.product(q2)
    array([-0.49690399,  0.1987616 ,  0.74535599,  0.3975232 ])

    Multiplication operators are overriden to perform the expected hamilton
    product.

    >>> q1*q2
    array([-0.49690399,  0.1987616 ,  0.74535599,  0.3975232 ])
    >>> q1@q2
    array([-0.49690399,  0.1987616 ,  0.74535599,  0.3975232 ])

    Basic operators are also overriden.

    >>> q1+q2
    Quaternion([0.46189977, 0.48678352, 0.51166727, 0.53655102])
    >>> q1-q2
    Quaternion([-0.69760203, -0.25108126,  0.19543951,  0.64196028])

    Pure quaternions are built from arrays with three elements.

    >>> q = Quaternion([1., 2., 3.])
    >>> q
    Quaternion([0.        , 0.26726124, 0.53452248, 0.80178373])
    >>> q.is_pure()
    True

    Conversions between representations are also possible.

    >>> q.to_axang()
    (array([0.26726124, 0.53452248, 0.80178373]), 3.141592653589793)
    >>> q.to_angles()
    array([ 1.24904577, -0.44291104,  2.8198421 ])

    And a nice representation as a string is also implemented to conform to
    Hamilton's notation.

    >>> str(q)
    '(0.0000 +0.2673i +0.5345j +0.8018k)'

    If the parameter ``order`` is set to 'S' the vector part will precede the
    scalar part.

    >>> q = Quaternion([2., -3., 4., -5.], order='S')
    >>> q.w
    -0.6804138174397717
    >>> q.v
    array([ 0.27216553, -0.40824829,  0.54433105])
    >>> print(q)
    (0.2722i -0.4082j +0.5443k -0.6804)

    """
    def __new__(subtype, q: Union[int, list, np.ndarray] = None, versor: bool = True, **kwargs):
        if q is None:
            q = np.array([1.0, 0.0, 0.0, 0.0])
            if kwargs.pop('random', False):
                q = random_attitudes()
            if 'dcm' in kwargs:
                q = Quaternion.from_DCM(Quaternion, kwargs.pop("dcm"), **kwargs)
            if 'rpy' in kwargs:
                q = Quaternion.from_rpy(Quaternion, kwargs.pop("rpy"))
            if 'angles' in kwargs:  # Older call to rpy
                q = Quaternion.from_angles(Quaternion, kwargs.pop("angles"))
        _assert_iterables(q, 'q')
        q = np.array(q, dtype=float)
        if q.ndim != 1 or q.shape[-1] not in [3, 4]:
            raise ValueError(f"Expected `q` to have shape (4,) or (3,), got {q.shape}.")
        if q.shape[-1] == 3:
            if not np.any(q):
                raise ValueError("Expected `q` to be a non-zero vector.")
            q = np.array([0.0, *q])
        if versor:
            q /= np.linalg.norm(q)
        # Create the ndarray instance of type Quaternion. This will call the
        # standard ndarray constructor, but return an object of type Quaternion.
        obj = super(Quaternion, subtype).__new__(subtype, q.shape, float, q)
        obj.A = q
        obj.scalar_vector = False if kwargs.get('order', 'H') == 'S' else True
        return obj

    @property
    def w(self) -> float:
        """
        Scalar part of the Quaternion.

        Given a quaternion :math:`\\mathbf{q}=\\begin{pmatrix}q_w & \\mathbf{q}_v\\end{pmatrix} = \\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}`,
        the scalar part, a.k.a. *real* part, is :math:`q_w`.

        Returns
        -------
        w : float
            Scalar part of the quaternion.

        Examples
        --------
        >>> q = Quaternion([2.0, -3.0, 4.0, -5.0])
        >>> q.view()
        Quaternion([ 0.27216553, -0.40824829,  0.54433105, -0.68041382])
        >>> q.w
        0.2721655269759087

        It can also be accessed directly, treating the Quaternion as an array:

        >>> q[0]
        0.2721655269759087
        """
        return self.A[0] if self.scalar_vector else self.A[3]

    @property
    def x(self) -> float:
        """
        First element of the vector part of the Quaternion.

        Given a quaternion :math:`\\mathbf{q}=\\begin{pmatrix}q_w & \\mathbf{q}_v\\end{pmatrix} = \\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}`,
        the first element of the vector part is :math:`q_x`.

        Returns
        -------
        x : float
            First element of vector part of the quaternion.

        Examples
        --------
        >>> q = Quaternion([2.0, -3.0, 4.0, -5.0])
        >>> q.view()
        Quaternion([ 0.27216553, -0.40824829,  0.54433105, -0.68041382])
        >>> q.x
        -0.408248290463863

        It can also be accessed directly, treating the Quaternion as an array:

        >>> q[1]
        -0.408248290463863
        """
        return self.A[1] if self.scalar_vector else self.A[0]

    @property
    def y(self) -> float:
        """
        Second element of the vector part of the Quaternion.

        Given a quaternion :math:`\\mathbf{q}=\\begin{pmatrix}q_w & \\mathbf{q}_v\\end{pmatrix} = \\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}`,
        the third element of the vector part is :math:`q_y`.

        Returns
        -------
        q_y : float
            Second element of vector part of the quaternion.

        Examples
        --------
        >>> q = Quaternion([2.0, -3.0, 4.0, -5.0])
        >>> q.view()
        Quaternion([ 0.27216553, -0.40824829,  0.54433105, -0.68041382])
        >>> q.y
        0.5443310539518174

        It can also be accessed directly, treating the Quaternion as an array:

        >>> q[2]
        0.5443310539518174
        """
        return self.A[2] if self.scalar_vector else self.A[1]

    @property
    def z(self) -> float:
        """
        Third element of the vector part of the Quaternion.

        Given a quaternion :math:`\\mathbf{q}=\\begin{pmatrix}q_w & \\mathbf{q}_v\\end{pmatrix} = \\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}`,
        the third element of the vector part is :math:`q_z`.

        Returns
        -------
        q_z : float
            Third element of vector part of the quaternion.

        Examples
        --------
        >>> q = Quaternion([2.0, -3.0, 4.0, -5.0])
        >>> q.view()
        Quaternion([ 0.27216553, -0.40824829,  0.54433105, -0.68041382])
        >>> q.z
        -0.6804138174397717

        It can also be accessed directly, treating the Quaternion as an array:

        >>> q[3]
        -0.6804138174397717
        """
        return self.A[3] if self.scalar_vector else self.A[2]

    @property
    def v(self) -> np.ndarray:
        """
        Vector part of the Quaternion.

        Given a quaternion :math:`\\mathbf{q}=\\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}`
        the vector part, a.k.a. *imaginary* part, is
        :math:`\\mathbf{q}_v=\\begin{bmatrix}q_x & q_y & q_z\\end{bmatrix}`.

        Returns
        -------
        q_v : numpy.ndarray
            Vector part of the quaternion.

        Examples
        --------
        >>> q = Quaternion([2.0, -3.0, 4.0, -5.0])
        >>> q.view()
        Quaternion([ 0.27216553, -0.40824829,  0.54433105, -0.68041382])
        >>> q.v
        array([-0.40824829,  0.54433105, -0.68041382])

        It can also be accessed directly, treating the Quaternion as an array,
        but is returned as a Quaternion object.

        >>> q[1:]
        Quaternion([-0.40824829,  0.54433105, -0.68041382])
        """
        return self.A[1:] if self.scalar_vector else self.A[:3]

    @property
    def conjugate(self) -> np.ndarray:
        """
        Conjugate of quaternion

        A quaternion, whose form is :math:`\\mathbf{q} = \\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}`,
        has a conjugate of the form :math:`\\mathbf{q}^* = \\begin{pmatrix}q_w & -q_x & -q_y & -q_z\\end{pmatrix}`.

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
        return self.A*np.array([1.0, -1.0, -1.0, -1.0]) if self.scalar_vector else self.A*np.array([-1.0, -1.0, -1.0, 1.0])

    @property
    def conj(self) -> np.ndarray:
        """
        Synonym of property :meth:`conjugate`

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
        :math:`\\mathbf{q}_I=\\begin{pmatrix}1 & 0 & 0 & 0\\end{pmatrix}`

        It is obtained as:

        .. math::
            \\mathbf{q}^{-1} = \\frac{\\mathbf{q}^*}{\\|\\mathbf{q}\\|^2}

        If the quaternion is normalized (called *versor*) its inverse is the
        conjugate.

        .. math::
            \\mathbf{q}^{-1} = \\mathbf{q}^*

        Returns
        -------
        out : numpy.ndarray
            Inverse of quaternion.

        Examples
        --------
        >>> q = Quaternion([1., -2., 3., -4.])
        >>> q
        Quaternion([ 0.18257419, -0.36514837,  0.54772256, -0.73029674])
        >>> q.inverse
        array([ 0.18257419,  0.36514837, -0.54772256,  0.73029674])
        >>> q@q.inverse
        array([1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.77555756e-17])
        """
        if self.is_versor():
            return self.conjugate
        return self.conjugate / np.linalg.norm(self.q)

    @property
    def inv(self) -> np.ndarray:
        """
        Synonym of property :meth:`inverse`

        Returns
        -------
        out : numpy.ndarray
            Inverse of quaternion.

        Examples
        --------
        >>> q = Quaternion([1., -2., 3., -4.])
        >>> q
        Quaternion([ 0.18257419, -0.36514837,  0.54772256, -0.73029674])
        >>> q.inv
        array([ 0.18257419,  0.36514837, -0.54772256,  0.73029674])
        >>> q@q.inv
        array([1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.77555756e-17])
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

        The exponential of a **pure quaternion** is, with the help of Euler
        formula and the series of :math:`\\cos\\theta` and :math:`\\sin\\theta`,
        redefined as:

        .. math::
            \\begin{array}{rcl}
            e^{\\mathbf{q}_v} &=& \\sum_{k=0}^{\\infty}\\frac{\\mathbf{q}_v^k}{k!} \\\\
            e^{\\mathbf{u}\\theta} &=&
            \\Big(1-\\frac{\\theta^2}{2!} + \\frac{\\theta^4}{4!}+\\cdots\\Big)+
            \\Big(\\mathbf{u}\\theta-\\frac{\\mathbf{u}\\theta^3}{3!} + \\frac{\\mathbf{u}\\theta^5}{5!}+\\cdots\\Big) \\\\
            &=& \\cos\\theta + \\mathbf{u}\\sin\\theta \\\\
            &=& \\begin{bmatrix}\\cos\\theta \\\\ \\mathbf{u}\\sin\\theta \\end{bmatrix}
            \\end{array}

        Letting :math:`\\mathbf{q}_v = \\mathbf{u}\\theta` with :math:`\\theta=\\|\mathbf{v}\\|`
        and :math:`\\|\\mathbf{u}\\|=1`.

        Since :math:`\\|e^{\\mathbf{q}_v}\\|^2=\\cos^2\\theta+\\sin^2\\theta=1`,
        the exponential of a pure quaternion is always unitary. Therefore, if
        the quaternion is real, its exponential is the identity.

        For **general quaternions** the exponential is defined using
        :math:`\\mathbf{u}\\theta=\\mathbf{q}_v` and the exponential of the pure
        quaternion:

        .. math::
            \\begin{array}{rcl}
            e^{\\mathbf{q}} &=& e^{q_w+\\mathbf{q}_v} = e^{q_w}e^{\\mathbf{q}_v}\\\\
            &=& e^{q_w}
            \\begin{bmatrix}
            \\cos\\|\\mathbf{q}_v\\| \\\\ \\frac{\\mathbf{q}}{\\|\\mathbf{q}_v\\|}\\sin\\|\\mathbf{q}_v\\|
            \\end{bmatrix}
            \\end{array}

        Returns
        -------
        exp : numpy.ndarray
            Exponential of quaternion.

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
        """
        Synonym of property :meth:`exponential`

        Returns
        -------
        exp : numpy.ndarray
            Exponential of quaternion.

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
        Logarithm of Quaternion.

        The logarithm of a **general quaternion**
        :math:`\\mathbf{q}=\\begin{pmatrix}q_w & \\mathbf{q_v}\\end{pmatrix}`
        is obtained from:

        .. math::
            \\log\\mathbf{q} = \\begin{bmatrix} \\log\\|\\mathbf{q}\\| \\\\ \\mathbf{u}\\theta \\end{bmatrix}

        with:

        .. math::
            \\begin{array}{rcl}
            \\mathbf{u} &=& \\frac{\\mathbf{q}_v}{\\|\\mathbf{q}_v\\|} \\\\
            \\theta &=& \\arccos\\Big(\\frac{q_w}{\\|\\mathbf{q}\\|}\\Big)
            \\end{array}

        It is easy to see, that for a **pure quaternion**
        :math:`\\mathbf{q}=\\begin{pmatrix}0 & \\mathbf{q_v}\\end{pmatrix}`, the
        logarithm simplifies the computation through :math:`\\theta=\\arccos(0)=\\frac{\\pi}{2}`:

        .. math::
            \\log\\mathbf{q} = \\begin{bmatrix}\\log\\|\\mathbf{q}\\| \\\\ \\mathbf{u}\\frac{\\pi}{2}\\end{bmatrix}

        Similarly, for **unitary quaternions** (:math:`\\|\\mathbf{q}\\|=1`)
        the logarithm is:

        .. math::
            \\log\\mathbf{q} = \\begin{bmatrix} 0 \\\\ \\mathbf{u}\\arccos(q_w) \\end{bmatrix}

        which further reduces for **pure unitary quaternions** (:math:`q_w=0` and :math:`\\|\\mathbf{q}\\|=1`)
        to:

        .. math::
            \\log\\mathbf{q} = \\begin{bmatrix} 0 \\\\ \\mathbf{u}\\frac{\\pi}{2} \\end{bmatrix}

        Returns
        -------
        log : numpy.ndarray
            Logarithm of quaternion.

        Examples
        --------
        >>> q = Quaternion([1.0, -2.0, 3.0, -4.0])
        >>> q.view()
        Quaternion([ 0.18257419, -0.36514837,  0.54772256, -0.73029674])
        >>> q.logarithm
        array([ 0.        , -0.51519029,  0.77278544, -1.03038059])
        >>> q = Quaternion([0.0, 1.0, -2.0, 3.0])
        >>> q.view()
        Quaternion([ 0.        ,  0.26726124, -0.53452248,  0.80178373])
        >>> q.logarithm
        array([ 0.        ,  0.41981298, -0.83962595,  1.25943893])

        """
        u = self.v/np.linalg.norm(self.v)
        if self.is_versor():
            if self.is_pure():
                return np.array([0.0, *(0.5*np.pi*u)])
            return np.array([0.0, *(u*np.arccos(self.w))])
        qn = np.linalg.norm(self.A)
        if self.is_pure():
            return np.array([np.log(qn), *(0.5*np.pi*u)])
        return np.array([np.log(qn), *(u*np.arccos(self.w/qn))])

    @property
    def log(self) -> np.ndarray:
        """
        Synonym of property :meth:`logarithm`

        Returns
        -------
        log : numpy.ndarray
            Logarithm of quaternion.

        Examples
        --------
        >>> q = Quaternion([1.0, -2.0, 3.0, -4.0])
        >>> q
        Quaternion([ 0.18257419, -0.36514837,  0.54772256, -0.73029674])
        >>> q.log
        array([ 0.        , -0.51519029,  0.77278544, -1.03038059])
        >>> q = Quaternion([0.0, 1.0, -2.0, 3.0])
        >>> q
        Quaternion([ 0.        ,  0.26726124, -0.53452248,  0.80178373])
        >>> q.log
        array([ 0.        ,  0.41981298, -0.83962595,  1.25943893])
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
        if self.scalar_vector:
            return f"({self.w:-.4f} {self.x:+.4f}i {self.y:+.4f}j {self.z:+.4f}k)"
        return f"({self.x:-.4f}i {self.y:+.4f}j {self.z:+.4f}k {self.w:+.4f})"

    def __add__(self, p: np.ndarray):
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

    def __sub__(self, p: np.ndarray):
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

    def __mul__(self, q: np.ndarray) -> np.ndarray:
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
        return self.product(q)

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
        q**a : numpy.ndarray
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

        Examples
        --------
        >>> q = Quaternion()
        >>> q
        Quaternion([1., 0., 0., 0.])
        >>> q.is_pure()
        False
        >>> q = Quaternion([0., 1., 2., 3.])
        >>> q
        Quaternion([0.        , 0.26726124, 0.53452248, 0.80178373])
        >>> q.is_pure()
        True

        """
        return self.w==0.0

    def is_real(self) -> bool:
        """
        Returns a bool value, where ``True`` if quaternion is real.

        A real quaternion has all elements of its vector part equal to zero:
        :math:`\\mathbf{q} = w + 0i + 0j + 0k = \\begin{pmatrix} q_w & \\mathbf{0}\\end{pmatrix}`

        .. math::
            \\left\\{
            \\begin{array}{ll}
                \\mathrm{True} & \\: \\mathbf{q}_v = \\begin{bmatrix} 0 & 0 & 0 \\end{bmatrix} \\\\
                \\mathrm{False} & \\: \\mathrm{otherwise}
            \\end{array}
            \\right.

        Returns
        -------
        out : bool
            Boolean equal to True if :math:`q_v = 0`.

        Examples
        --------
        >>> q = Quaternion()
        >>> q
        Quaternion([1., 0., 0., 0.])
        >>> q.is_real()
        True
        >>> q = Quaternion([0., 1., 2., 3.])
        >>> q
        Quaternion([0.        , 0.26726124, 0.53452248, 0.80178373])
        >>> q.is_real()
        False
        >>> q = Quaternion([1., 2., 3., 4.])
        >>> q
        Quaternion([0.18257419, 0.36514837, 0.54772256, 0.73029674])
        >>> q.is_real()
        False
        >>> q = Quaternion([5., 0., 0., 0.])    # All quaternions are normalized, by default
        >>> q
        Quaternion([1., 0., 0., 0.])
        >>> q.is_real()
        True
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

        An **identity quaternion** has its scalar part equal to 1, and its
        vector part equal to 0, such that :math:`\\mathbf{q} = 1 + 0i + 0j + 0k`.

        .. math::
            \\left\\{
            \\begin{array}{ll}
                \\mathrm{True} & \\: \\mathbf{q}\\ = \\begin{pmatrix} 1 & 0 & 0 & 0 \\end{pmatrix} \\\\
                \\mathrm{False} & \\: \\mathrm{otherwise}
            \\end{array}
            \\right.

        Returns
        -------
        out : bool
            Boolean equal to ``True`` if :math:`\\mathbf{q}=\\begin{pmatrix}1 & 0 & 0 & 0\\end{pmatrix}`.

        Examples
        --------
        >>> q = Quaternion()
        >>> q
        Quaternion([1., 0., 0., 0.])
        >>> q.is_identity()
        True
        >>> q = Quaternion([0., 1., 0., 0.])
        >>> q
        Quaternion([0., 1., 0., 0.])
        >>> q.is_identity()
        False

        """
        return np.allclose(self.A, np.array([1.0, 0.0, 0.0, 0.0]))

    def normalize(self) -> None:
        """Normalize the quaternion."""
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
            \\begin{bmatrix}q_w p_w - \\mathbf{q}_v \\cdot \\mathbf{p}_v \\\\
            \\mathbf{q}_v \\times \\mathbf{p}_v + q_w \\mathbf{p}_v + p_w \\mathbf{q}_v \\end{bmatrix} \\\\
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
        q : numpy.ndarray, Quaternion
            Quaternion to multiply with.

        Returns
        -------
        pq : numpy.ndarray
            Product of quaternions.

        Examples
        --------
        >>> p = Quaternion([0.55747131, 0.12956903, 0.5736954 , 0.58592763])

        Can multiply with a given quaternion in vector form...

        >>> p.product([0.49753507, 0.50806522, 0.52711628, 0.4652709])
        array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])

        or with a Quaternion object...

        >>> q = Quaternion([0.49753507, 0.50806522, 0.52711628, 0.4652709 ])
        >>> p.product(q)
        array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])

        It holds with the result after the cross and dot product definition

        >>> rw = p.w*q.w - np.dot(p.v, q.v)
        >>> rv = p.w*q.v + q.w*p.v + np.cross(p.v, q.v)
        >>> rw, rv
        (-0.36348726, array([0.38962514,  0.34188103,  0.77407146]))

        """
        if isinstance(q, Quaternion):
            qw, qx, qy, qz = q
        elif isinstance(q, (np.ndarray, list, tuple)):
            qw, qx, qy, qz = Quaternion(q)
        else:
            raise TypeError(f"q must be a Quaternion or an array, not {type(q)}")
        pq = np.array([
            self.w*qw - self.x*qx - self.y*qy - self.z*qz,
            self.w*qx + self.x*qw + self.y*qz - self.z*qy,
            self.w*qy - self.x*qz + self.y*qw + self.z*qx,
            self.w*qz + self.x*qy - self.y*qx + self.z*qw])
        return pq / np.linalg.norm(pq)

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
        """
        Rotate array :math:`\\mathbf{a}` through quaternion :math:`\\mathbf{q}`.

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
            raise ValueError(f"Expected `a` to have shape (3, N) or (3,), got {a.shape}.")
        return self.to_DCM()@a

    def to_array(self) -> np.ndarray:
        """
        Return quaternion as a NumPy array

        Quaternion values are stored in attribute ``A``, which is a NumPy array.
        This method simply returns that attribute.

        Returns
        -------
        out : numpy.ndarray
            Quaternion.

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

        Given a unit quaternion :math:`\\mathbf{q} = \\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}`,
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

        The given unit quaternion must have the form
        :math:`\\mathbf{q} = \\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}`,
        where :math:`\\mathbf{q}_v = \\begin{bmatrix}q_x & q_y & q_z\\end{bmatrix}`
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

        The identity Quaternion :math:`\\mathbf{q} = \\begin{pmatrix}1 & 0 & 0 & 0\\end{pmatrix}`,
        produces a :math:`3 \\times 3` Identity matrix :math:`\\mathbf{I}_3`.

        Returns
        -------
        DCM : numpy.ndarray
            3-by-3 Direction Cosine Matrix

        Examples
        --------
        >>> q = Quaternion()
        >>> q.view()
        Quaternion([1., 0., 0., 0.])
        >>> q.to_DCM()
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        >>> q = Quaternion([1., -2., 3., -4.])
        >>> q.view()
        Quaternion([ 0.18257419, -0.36514837,  0.54772256, -0.73029674])
        >>> q.to_DCM()
        array([[-0.66666667, -0.13333333,  0.73333333],
               [-0.66666667, -0.33333333, -0.66666667],
               [ 0.33333333, -0.93333333,  0.13333333]])
        >>> q = Quaternion([0., -4., 3., -2.])
        >>> q.view()
        Quaternion([ 0.        , -0.74278135,  0.55708601, -0.37139068])
        >>> q.to_DCM()
        array([[ 0.10344828, -0.82758621,  0.55172414],
               [-0.82758621, -0.37931034, -0.4137931 ],
               [ 0.55172414, -0.4137931 , -0.72413793]])

        """
        return np.array([
            [1.0-2.0*(self.y**2+self.z**2), 2.0*(self.x*self.y-self.w*self.z), 2.0*(self.x*self.z+self.w*self.y)],
            [2.0*(self.x*self.y+self.w*self.z), 1.0-2.0*(self.x**2+self.z**2), 2.0*(self.y*self.z-self.w*self.x)],
            [2.0*(self.x*self.z-self.w*self.y), 2.0*(self.w*self.x+self.y*self.z), 1.0-2.0*(self.x**2+self.y**2)]])

    def from_DCM(self, dcm: np.ndarray, method: str = 'chiaverini', **kw) -> np.ndarray:
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

        Notes
        -----
        The selection can be simplified with Structural Pattern Matching if
        using Python 3.10 or later.

        """
        _assert_iterables(dcm, 'Direction Cosine Matrix')
        in_SO3 = np.isclose(np.linalg.det(np.atleast_2d(dcm)), 1.0)
        in_SO3 &= np.allclose(dcm@dcm.T, np.identity(3))
        if not in_SO3:
            raise ValueError("Given Direction Cosine Matrix is not in SO(3).")
        dcm = np.copy(dcm)
        if dcm.shape != (3, 3):
            raise TypeError(f"Expected matrix of size (3, 3). Got {dcm.shape}")
        if method.lower() == 'hughes':
            q = hughes(dcm)
        elif method.lower() == 'chiaverini':
            q = chiaverini(dcm)
        elif method.lower() == 'shepperd':
            q = shepperd(dcm)
        elif method.lower() == 'itzhack':
            q = itzhack(dcm, version=kw.get('version', 3))
        elif method.lower() == 'sarabandi':
            q = sarabandi(dcm, eta=kw.get('threshold', 0.0))
        else:
            raise ValueError(f"Given method '{method}' is not implemented.")
        q /= np.linalg.norm(q)
        return q

    def from_rpy(self, angles: np.ndarray) -> np.ndarray:
        """
        Quaternion from given RPY angles.

        The quaternion can be constructed from the Aerospace cardanian angle
        sequence that follows the order :math:`\\phi\\to\\theta\\to\\psi`,
        where :math:`\\phi` is the **roll** (or *bank*) angle, :math:`\\theta`
        is the **pitch** (or *elevation*) angle, and :math:`\\psi` is the
        **yaw** (or *heading*) angle.

        The composing quaternions are:

        .. math::
            \\begin{array}{rcl}
            \\mathbf{q}_X &=& \\begin{pmatrix}\\cos\\frac{\\phi}{2} & \\sin\\frac{\\phi}{2} & 0 & 0\\end{pmatrix} \\\\ && \\\\
            \\mathbf{q}_Y &=& \\begin{pmatrix}\\cos\\frac{\\theta}{2} & 0 & \\sin\\frac{\\theta}{2} & 0\\end{pmatrix} \\\\ && \\\\
            \\mathbf{q}_Z &=& \\begin{pmatrix}\\cos\\frac{\\psi}{2} & 0 & 0 & \\sin\\frac{\\psi}{2}\\end{pmatrix}
            \\end{array}

        The elements of the final quaternion
        :math:`\\mathbf{q}=\\mathbf{q}_Z\\mathbf{q}_Y\\mathbf{q}_X = q_w+q_xi+q_yj+q_zk`
        are obtained as:

        .. math::
            \\begin{array}{rcl}
            q_w &=& \\cos\\frac{\\psi}{2}\\cos\\frac{\\theta}{2}\\cos\\frac{\\phi}{2} + \\sin\\frac{\\psi}{2}\\sin\\frac{\\theta}{2}\\sin\\frac{\\phi}{2} \\\\ && \\\\
            q_x &=& \\cos\\frac{\\psi}{2}\\cos\\frac{\\theta}{2}\\sin\\frac{\\phi}{2} - \\sin\\frac{\\psi}{2}\\sin\\frac{\\theta}{2}\\cos\\frac{\\phi}{2} \\\\ && \\\\
            q_y &=& \\cos\\frac{\\psi}{2}\\sin\\frac{\\theta}{2}\\cos\\frac{\\phi}{2} + \\sin\\frac{\\psi}{2}\\cos\\frac{\\theta}{2}\\sin\\frac{\\phi}{2} \\\\ && \\\\
            q_z &=& \\sin\\frac{\\psi}{2}\\cos\\frac{\\theta}{2}\\cos\\frac{\\phi}{2} - \\cos\\frac{\\psi}{2}\\sin\\frac{\\theta}{2}\\sin\\frac{\\phi}{2}
            \\end{array}

        .. warning::
            The Aerospace sequence :math:`\\phi\\to\\theta\\to\\psi` is only
            one of the `twelve possible rotation sequences
            <https://en.wikipedia.org/wiki/Euler_angles#Tait.E2.80.93Bryan_angles>`_
            around the main axes. Other sequences might be more suitable for
            other applications, but this one is the most common in practice.

        Parameters
        ----------
        angles : numpy.ndarray
            3 cardanian angles, in radians, following the order: roll -> pitch -> yaw.

        Returns
        -------
        q : numpy.ndarray
            Quaternion from roll-pitch-yaw angles.

        Examples
        --------
        >>> from ahrs import DEG2RAD    # Helper variable to convert angles to radians
        >>> q = Quaternion()
        >>> q.from_rpy(np.array([10.0, 20.0, 30.0])*DEG2RAD)    # Give roll-pitch-yaw angles as radians.
        array([0.95154852, 0.23929834, 0.18930786, 0.03813458])

        It can be corroborated with the class `DCM <./dcm.html>`_, which represents a Direction
        Cosine Matrix, and can also be built with roll-pitch-yaw angles.

        >>> from ahrs import DCM
        >>> R = DCM(rpy=[10.0, 20.0, 30.0])     # Here you give the angles as degrees
        >>> R
        DCM([[ 0.92541658,  0.01802831,  0.37852231],
             [ 0.16317591,  0.88256412, -0.44096961],
             [-0.34202014,  0.46984631,  0.81379768]])
        >>> q.from_DCM(R)
        array([0.95154852, 0.23929834, 0.18930786, 0.03813458])

        With both approaches the same quaternion is obtained.

        """
        _assert_iterables(angles, 'Roll-Pitch-Yaw angles')
        angles = np.array(angles)
        if angles.ndim != 1 or angles.shape[0] != 3:
            raise ValueError(f"Expected `angles` must have shape (3,), got {angles.shape}.")
        for angle in angles:
            if angle < -2.0* np.pi or angle > 2.0 * np.pi:
                raise ValueError(f"Expected `angles` must be in the range [-2pi, 2pi], got {angles}.")
        roll, pitch, yaw = angles
        cy = np.cos(0.5*yaw)
        sy = np.sin(0.5*yaw)
        cp = np.cos(0.5*pitch)
        sp = np.sin(0.5*pitch)
        cr = np.cos(0.5*roll)
        sr = np.sin(0.5*roll)
        q = np.zeros(4)
        q[0] = cy*cp*cr + sy*sp*sr
        q[1] = cy*cp*sr - sy*sp*cr
        q[2] = cy*sp*cr + sy*cp*sr
        q[3] = sy*cp*cr - cy*sp*sr
        return q

    def from_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        Synonym to method from_rpy()

        Parameters
        ----------
        angles : numpy.ndarray
            3 cardanian angles in following order: roll -> pitch -> yaw.

        Returns
        -------
        q : numpy.ndarray
            Quaternion from roll-pitch-yaw angles.

        Examples
        --------
        >>> from ahrs import DEG2RAD    # Helper variable to convert angles to radians
        >>> q = Quaternion()
        >>> q.from_angles(np.array([10.0, 20.0, 30.0])*DEG2RAD)    # Give roll-pitch-yaw angles as radians.
        array([0.95154852, 0.23929834, 0.18930786, 0.03813458])

        It can be corroborated with the class `DCM <./dcm.html>`_, which represents a Direction
        Cosine Matrix, and can also be built with roll-pitch-yaw angles.

        >>> from ahrs import DCM
        >>> R = DCM(rpy=[10.0, 20.0, 30.0])     # Here you give the angles as degrees
        >>> R
        DCM([[ 0.92541658,  0.01802831,  0.37852231],
             [ 0.16317591,  0.88256412, -0.44096961],
             [-0.34202014,  0.46984631,  0.81379768]])
        >>> q.from_DCM(R)
        array([0.95154852, 0.23929834, 0.18930786, 0.03813458])

        With both approaches the same quaternion is obtained.

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
        _assert_iterables(w, 'Angular velocity')
        if w.ndim != 1 or w.shape[0] != 3:
            raise ValueError(f"Expected `w` to have shape (3,), got {w.shape}")
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
        return random_attitudes(1)

class QuaternionArray(np.ndarray):
    """
    Array of Quaternions

    Class to represent quaternion arrays. It can be built from N-by-3 or N-by-4
    arrays. The objects are **always normalized** to represent rotations in 3D
    space (versors), unless explicitly specified setting the parameter
    ``versors`` to ``False``.

    If an N-by-3 array is given, it is assumed to represent pure quaternions,
    setting their scalar part equal to zero.

    Parameters
    ----------
    q : array-like or int, default: None
        N-by-4 or N-by-3 array containing the quaternion values to use. If an
        integer is given, it creates ``N`` random quaternions, where ``N`` is
        the given int. If None is given, a single identity quaternion is
        stored in a 2d array.
    versors : bool, default: True
        Treat quaternions as versors. It will normalize them immediately.
    order : str, default: 'H'
        Specify the layout of the Quaternions, where the default is 'H' for a
        Hamiltonian notation with the scalar parts preceding the vector parts.
        If order is 'S' the vector parts precede the scalar parts.

    Attributes
    ----------
    array : numpy.ndarray
        Array with all N quaternions.
    w : numpy.ndarray
        Scalar parts of all quaternions.
    x : numpy.ndarray
        First elements of the vector part of all quaternions.
    y : numpy.ndarray
        Second elements of the vector part of all quaternions.
    z : numpy.ndarray
        Third elements of the vector part of all quaternions.
    v : numpy.ndarray
        Vector part of all quaternions.

    Raises
    ------
    ValueError
        When length of input array is not equal to either 3 or 4.

    Examples
    --------
    >>> from ahrs import QuaternionArray
    >>> Q = QuaternionArray(np.random.random((3, 4))-0.5)
    >>> Q.view()
    QuaternionArray([[ 0.39338362, -0.29206111, -0.07445273,  0.86856573],
                     [ 0.65459935,  0.14192058, -0.69722158,  0.25542183],
                     [-0.42837174,  0.85451579, -0.02786928,  0.29244439]])

    If an N-by-3 array is given, it is used to build an array of pure
    quaternions:

    >>> Q = QuaternionArray(np.random.random((5, 3))-0.5)
    >>> Q.view()
    QuaternionArray([[ 0.        , -0.73961715,  0.23572589,  0.63039652],
                     [ 0.        , -0.54925142,  0.67303056,  0.49533093],
                     [ 0.        ,  0.46936253,  0.39912076,  0.78765566],
                     [ 0.        ,  0.52205066, -0.16510523, -0.83678155],
                     [ 0.        ,  0.11844943, -0.27839573, -0.95313459]])

    Transformations to other representations are possible:

    .. code-block:: python

        >>> Q = QuaternionArray(np.random.random((3, 4))-0.5)
        >>> Q.to_angles()
        array([[-0.41354414,  0.46539024,  2.191703  ],
               [-1.6441448 , -1.39912606,  2.21590455],
               [-2.12380045, -0.49600967, -0.34589322]])
        >>> Q.to_DCM()
        array([[[-0.51989927, -0.63986956, -0.56592552],
                [ 0.72684856, -0.67941224,  0.10044993],
                [-0.44877158, -0.3591183 ,  0.81831419]],

               [[-0.10271648, -0.53229811, -0.84030235],
                [ 0.13649774,  0.82923647, -0.54197346],
                [ 0.98530081, -0.17036898, -0.01251876]],

               [[ 0.82739916,  0.20292036,  0.52367352],
                [-0.2981793 , -0.63144191,  0.71580041],
                [ 0.47591988, -0.74840126, -0.46194785]]])

    Markley's method to obtain the average quaternion is implemented too:

    >>> qts = np.tile([1., -2., 3., -4], (5, 1))    # Five equal arrays
    >>> v = np.random.randn(5, 4)*0.1               # Gaussian noise
    >>> Q = QuaternionArray(qts + v)
    >>> Q.view()
    QuaternionArray([[ 0.17614144, -0.39173347,  0.56303067, -0.70605634],
                     [ 0.17607515, -0.3839024 ,  0.52673809, -0.73767437],
                     [ 0.16823806, -0.35898889,  0.53664261, -0.74487424],
                     [ 0.17094453, -0.3723117 ,  0.54109885, -0.73442086],
                     [ 0.1862619 , -0.38421818,  0.5260265 , -0.73551276]])
    >>> Q.average()
    array([-0.17557859,  0.37832975, -0.53884688,  0.73190355])

    If, for any reason, the signs of certain quaternions are flipped (they
    still represent the same rotation in 3D Euclidean space), we can use the
    method rempve jumps to flip them back.

    >>> Q.view()
    QuaternionArray([[ 0.17614144, -0.39173347,  0.56303067, -0.70605634],
                     [ 0.17607515, -0.3839024 ,  0.52673809, -0.73767437],
                     [ 0.16823806, -0.35898889,  0.53664261, -0.74487424],
                     [ 0.17094453, -0.3723117 ,  0.54109885, -0.73442086],
                     [ 0.1862619 , -0.38421818,  0.5260265 , -0.73551276]])
    >>> Q[1:3] *= -1
    >>> Q.view()
    QuaternionArray([[ 0.17614144, -0.39173347,  0.56303067, -0.70605634],
                     [-0.17607515,  0.3839024 , -0.52673809,  0.73767437],
                     [-0.16823806,  0.35898889, -0.53664261,  0.74487424],
                     [ 0.17094453, -0.3723117 ,  0.54109885, -0.73442086],
                     [ 0.1862619 , -0.38421818,  0.5260265 , -0.73551276]])
    >>> Q.remove_jumps()
    >>> Q.view()
    QuaternionArray([[ 0.17614144, -0.39173347,  0.56303067, -0.70605634],
                     [ 0.17607515, -0.3839024 ,  0.52673809, -0.73767437],
                     [ 0.16823806, -0.35898889,  0.53664261, -0.74487424],
                     [ 0.17094453, -0.3723117 ,  0.54109885, -0.73442086],
                     [ 0.1862619 , -0.38421818,  0.5260265 , -0.73551276]])
    """
    def __new__(subtype, q: np.ndarray = None, versors: bool = True, order: str = 'H'):
        if q is None:
            q = np.array([[1.0, 0.0, 0.0, 0.0]])
        if isinstance(q, int):
            q = np.atleast_2d(random_attitudes(q))
        _assert_iterables(q, 'Quaternion Array')
        q = np.array(q, dtype=float)
        if q.ndim != 2 or q.shape[-1] not in [3, 4]:
            raise ValueError(f"Expected array to have shape (N, 4) or (N, 3), got {q.shape}.")
        if q.shape[-1] == 3:
            q = np.c_[np.zeros(q.shape[0]), q]
        if versors:
            q /= np.linalg.norm(q, axis=1)[:, None]
        # Create the ndarray instance of type QuaternionArray. This will call
        # the standard ndarray constructor, but return an object of type
        # QuaternionArray.
        obj = super(QuaternionArray, subtype).__new__(subtype, q.shape, float, q)
        obj.array = q
        obj.scalar_vector = False if order == 'S' else True
        obj.num_qts = q.shape[0]
        return obj

    @property
    def w(self) -> np.ndarray:
        """
        Scalar parts of all Quaternions.

        Having the quaternion elements :math:`\\mathbf{q}_i=\\begin{pmatrix}w_i & \\mathbf{v}_i\\end{pmatrix}=\\begin{pmatrix}w_i & x_i & y_i & z_i\\end{pmatrix}\\in\\mathbb{R}^4`
        stacked vertically in an :math:`N\\times 4` matrix :math:`\\mathbf{Q}`:

        .. math::
            \\mathbf{Q} =
            \\begin{bmatrix} \\mathbf{q}_0 \\\\ \\mathbf{q}_1 \\\\ \\vdots \\\\ \\mathbf{q}_{N-1} \\end{bmatrix} =
            \\begin{bmatrix} w_0 & x_0 & y_0 & z_0 \\\\ w_1 & x_1 & y_1 & z_1 \\\\
            \\vdots & \\vdots & \\vdots & \\vdots \\\\ w_{N-1} & x_{N-1} & y_{N-1} & z_{N-1} \\end{bmatrix}

        The scalar elements of all quaternions are:

        .. math::
            \\mathbf{w} = \\begin{bmatrix}w_0 & w_1 & \\cdots & w_{N-1}\\end{bmatrix}

        Returns
        -------
        w : numpy.ndarray
            Scalar parts of all quaternions.

        Examples
        --------
        >>> Q = QuaternionArray(np.random.random((3, 4))-0.5)
        >>> Q.view()
        QuaternionArray([[ 0.39338362, -0.29206111, -0.07445273,  0.86856573],
                         [ 0.65459935,  0.14192058, -0.69722158,  0.25542183],
                         [-0.42837174,  0.85451579, -0.02786928,  0.29244439]])
        >>> Q.w
        array([ 0.39338362,  0.65459935, -0.42837174])

        They can also be accessed directly, returned as a QuaternionArray:

        >>> Q[:, 0]
        QuaternionArray([ 0.39338362,  0.65459935, -0.42837174])
        """
        return self.array[:, 0] if self.scalar_vector else self.array[:, 3]

    @property
    def x(self) -> np.ndarray:
        """
        First elements of the vector part of all Quaternions.

        Having the quaternion elements :math:`\\mathbf{q}_i=\\begin{pmatrix}w_i & \\mathbf{v}_i\\end{pmatrix}=\\begin{pmatrix}w_i & x_i & y_i & z_i\\end{pmatrix}\\in\\mathbb{R}^4`
        stacked vertically in an :math:`N\\times 4` matrix :math:`\\mathbf{Q}`:

        .. math::
            \\mathbf{Q} =
            \\begin{bmatrix} \\mathbf{q}_0 \\\\ \\mathbf{q}_1 \\\\ \\vdots \\\\ \\mathbf{q}_{N-1} \\end{bmatrix} =
            \\begin{bmatrix} w_0 & x_0 & y_0 & z_0 \\\\ w_1 & x_1 & y_1 & z_1 \\\\
            \\vdots & \\vdots & \\vdots & \\vdots \\\\ w_{N-1} & x_{N-1} & y_{N-1} & z_{N-1} \\end{bmatrix}

        The first elements of the vector parts of all quaternions are:

        .. math::
            \\mathbf{x} = \\begin{bmatrix}x_0 & x_1 & \\cdots & x_{N-1}\\end{bmatrix}

        Returns
        -------
        x : numpy.ndarray
            First elements of the vector part of all quaternions.

        Examples
        --------
        >>> Q = QuaternionArray(np.random.random((3, 4))-0.5)
        >>> Q.view()
        QuaternionArray([[ 0.39338362, -0.29206111, -0.07445273,  0.86856573],
                         [ 0.65459935,  0.14192058, -0.69722158,  0.25542183],
                         [-0.42837174,  0.85451579, -0.02786928,  0.29244439]])
        >>> Q.x
        array([-0.29206111,  0.14192058,  0.85451579])

        They can also be accessed directly, returned as a QuaternionArray:

        >>> Q[:, 1]
        QuaternionArray([-0.29206111,  0.14192058,  0.85451579])
        """
        return self.array[:, 1] if self.scalar_vector else self.array[:, 0]

    @property
    def y(self) -> np.ndarray:
        """
        Second elements of the vector part of all Quaternions.

        Having the quaternion elements :math:`\\mathbf{q}_i=\\begin{pmatrix}w_i & \\mathbf{v}_i\\end{pmatrix}=\\begin{pmatrix}w_i & x_i & y_i & z_i\\end{pmatrix}\\in\\mathbb{R}^4`
        stacked vertically in an :math:`N\\times 4` matrix :math:`\\mathbf{Q}`:

        .. math::
            \\mathbf{Q} =
            \\begin{bmatrix} \\mathbf{q}_0 \\\\ \\mathbf{q}_1 \\\\ \\vdots \\\\ \\mathbf{q}_{N-1} \\end{bmatrix} =
            \\begin{bmatrix} w_0 & x_0 & y_0 & z_0 \\\\ w_1 & x_1 & y_1 & z_1 \\\\
            \\vdots & \\vdots & \\vdots & \\vdots \\\\ w_{N-1} & x_{N-1} & y_{N-1} & z_{N-1} \\end{bmatrix}

        The second elements of the vector parts of all quaternions are:

        .. math::
            \\mathbf{y} = \\begin{bmatrix}y_0 & y_1 & \\cdots & y_{N-1}\\end{bmatrix}

        Returns
        -------
        y : numpy.ndarray
            Second elements of the vector part of all quaternions.

        Examples
        --------
        >>> Q = QuaternionArray(np.random.random((3, 4))-0.5)
        >>> Q.view()
        QuaternionArray([[ 0.39338362, -0.29206111, -0.07445273,  0.86856573],
                         [ 0.65459935,  0.14192058, -0.69722158,  0.25542183],
                         [-0.42837174,  0.85451579, -0.02786928,  0.29244439]])
        >>> Q.y
        array([-0.07445273, -0.69722158, -0.02786928])

        They can also be accessed directly, returned as a QuaternionArray:

        >>> Q[:, 2]
        QuaternionArray([-0.07445273, -0.69722158, -0.02786928])
        """
        return self.array[:, 2] if self.scalar_vector else self.array[:, 1]

    @property
    def z(self) -> np.ndarray:
        """
        Third elements of the vector part of all Quaternions.

        Having the quaternion elements :math:`\\mathbf{q}_i=\\begin{pmatrix}w_i & \\mathbf{v}_i\\end{pmatrix}=\\begin{pmatrix}w_i & x_i & y_i & z_i\\end{pmatrix}\\in\\mathbb{R}^4`
        stacked vertically in an :math:`N\\times 4` matrix :math:`\\mathbf{Q}`:

        .. math::
            \\mathbf{Q} =
            \\begin{bmatrix} \\mathbf{q}_0 \\\\ \\mathbf{q}_1 \\\\ \\vdots \\\\ \\mathbf{q}_{N-1} \\end{bmatrix} =
            \\begin{bmatrix} w_0 & x_0 & y_0 & z_0 \\\\ w_1 & x_1 & y_1 & z_1 \\\\
            \\vdots & \\vdots & \\vdots & \\vdots \\\\ w_{N-1} & x_{N-1} & y_{N-1} & z_{N-1} \\end{bmatrix}

        The third elements of the vector parts of all quaternions are:

        .. math::
            \\mathbf{z} = \\begin{bmatrix}z_0 & z_1 & \\cdots & z_{N-1}\\end{bmatrix}

        Returns
        -------
        z : numpy.ndarray
            Third elements of the vector part of all quaternions.

        Examples
        --------
        >>> Q = QuaternionArray(np.random.random((3, 4))-0.5)
        >>> Q.view()
        QuaternionArray([[ 0.39338362, -0.29206111, -0.07445273,  0.86856573],
                         [ 0.65459935,  0.14192058, -0.69722158,  0.25542183],
                         [-0.42837174,  0.85451579, -0.02786928,  0.29244439]])
        >>> Q.z
        array([0.86856573, 0.25542183, 0.29244439])

        They can also be accessed directly, returned as a QuaternionArray:

        >>> Q[:, 3]
        QuaternionArray([0.86856573, 0.25542183, 0.29244439])
        """
        return self.array[:, 3] if self.scalar_vector else self.array[:, 2]

    @property
    def v(self) -> np.ndarray:
        """
        Vector part of all Quaternions.

        Having the quaternion elements :math:`\\mathbf{q}_i=\\begin{pmatrix}w_i & \\mathbf{v}_i\\end{pmatrix}=\\begin{pmatrix}w_i & x_i & y_i & z_i\\end{pmatrix}\\in\\mathbb{R}^4`
        stacked vertically in an :math:`N\\times 4` matrix :math:`\\mathbf{Q}`:

        .. math::
            \\mathbf{Q} =
            \\begin{bmatrix} \\mathbf{q}_0 \\\\ \\mathbf{q}_1 \\\\ \\vdots \\\\ \\mathbf{q}_{N-1} \\end{bmatrix} =
            \\begin{bmatrix} w_0 & x_0 & y_0 & z_0 \\\\ w_1 & x_1 & y_1 & z_1 \\\\
            \\vdots & \\vdots & \\vdots & \\vdots \\\\ w_{N-1} & x_{N-1} & y_{N-1} & z_{N-1} \\end{bmatrix}

        The vector parts of all quaternions are:

        .. math::
            \\mathbf{V} = \\begin{bmatrix} x_0 & y_0 & z_0 \\\\ x_1 & y_1 & z_1 \\\\
            \\vdots & \\vdots & \\vdots \\\\ x_{N-1} & y_{N-1} & z_{N-1} \\end{bmatrix}

        Returns
        -------
        V : numpy.ndarray
            N-by-3 array with vector parts of all quaternions.

        Examples
        --------
        >>> Q = QuaternionArray(np.random.random((3, 4))-0.5)
        >>> Q.view()
        QuaternionArray([[ 0.39338362, -0.29206111, -0.07445273,  0.86856573],
                         [ 0.65459935,  0.14192058, -0.69722158,  0.25542183],
                         [-0.42837174,  0.85451579, -0.02786928,  0.29244439]])
        >>> Q.v
        array([[-0.29206111, -0.07445273,  0.86856573],
               [ 0.14192058, -0.69722158,  0.25542183],
               [ 0.85451579, -0.02786928,  0.29244439]])

        They can also be accessed directly, slicing the Quaternion like an
        array, but returned as a Quaternion object.

        >>> Q[:, 1:]
        QuaternionArray([[-0.29206111, -0.07445273,  0.86856573],
                         [ 0.14192058, -0.69722158,  0.25542183],
                         [ 0.85451579, -0.02786928,  0.29244439]])
        """
        return self.array[:, 1:] if self.scalar_vector else self.array[:, :3]

    def is_pure(self) -> np.ndarray:
        """
        Returns an array of boolean values, where a value is ``True`` if its
        corresponding quaternion is pure.

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
        out : np.ndarray
            Array of booleans.

        Example
        -------
        >>> Q = QuaternionArray(np.random.random((3, 4))-0.5)
        >>> Q[1, 0] = 0.0
        >>> Q.view()
        QuaternionArray([[ 0.32014817,  0.47060011,  0.78255824,  0.25227621],
                         [ 0.        , -0.79009137,  0.47021242, -0.26103598],
                         [-0.65182559, -0.3032904 ,  0.16078433, -0.67622979]])
        >>> Q.is_pure()
        array([False,  True, False])
        """
        return np.isclose(self.w, np.zeros_like(self.w.shape[0]))

    def is_real(self) -> np.ndarray:
        """
        Returns an array of boolean values, where a value is ``True`` if its
        corresponding quaternion is real.

        A real quaternion has all elements of its vector part equal to zero:
        :math:`\\mathbf{q} = w + 0i + 0j + 0k = \\begin{pmatrix} q_w & \\mathbf{0}\\end{pmatrix}`

        .. math::
            \\left\\{
            \\begin{array}{ll}
                \\mathrm{True} & \\: \\mathbf{q}_v = \\begin{bmatrix} 0 & 0 & 0 \\end{bmatrix} \\\\
                \\mathrm{False} & \\: \\mathrm{otherwise}
            \\end{array}
            \\right.

        Returns
        -------
        out : np.ndarray
            Array of booleans.

        Example
        -------
        >>> Q = QuaternionArray(np.random.random((3, 4))-0.5)
        >>> Q[1, 1:] = 0.0
        >>> Q.view()
        QuaternionArray([[-0.8061095 ,  0.42513151,  0.37790158, -0.16322091],
                         [ 0.04515362,  0.        ,  0.        ,  0.        ],
                         [ 0.29613776,  0.21692562, -0.16253866, -0.91587493]])
        >>> Q.is_real()
        array([False,  True, False])
        """
        return np.all(np.isclose(self.v, np.zeros_like(self.v)), axis=1)

    def is_versor(self) -> np.ndarray:
        """
        Returns an array of boolean values, where a value is ``True`` if its
        corresponding quaternion has a norm equal to one.

        A **versor** is a quaternion, whose `euclidean norm
        <https://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm>`_ is
        equal to one: :math:`\\|\\mathbf{q}\\| = \\sqrt{w^2+x^2+y^2+z^2} = 1`

        .. math::
            \\left\\{
            \\begin{array}{ll}
                \\mathrm{True} & \\: \\sqrt{w^2+x^2+y^2+z^2} = 1 \\\\
                \\mathrm{False} & \\: \\mathrm{otherwise}
            \\end{array}
            \\right.

        Returns
        -------
        out : np.ndarray
            Array of booleans.

        Example
        -------
        >>> Q = QuaternionArray(np.random.random((3, 4))-0.5)
        >>> Q[1] = [1.0, 2.0, 3.0, 4.0]
        >>> Q.view()
        QuaternionArray([[-0.8061095 ,  0.42513151,  0.37790158, -0.16322091],
                         [ 1.        ,  2.        ,  3.        ,  4.        ],
                         [ 0.29613776,  0.21692562, -0.16253866, -0.91587493]])
        >>> Q.is_versor()
        array([ True, False,  True])
        """
        return np.isclose(np.linalg.norm(self.array, axis=1), 1.0)

    def is_identity(self) -> np.ndarray:
        """
        Returns an array of boolean values, where a value is ``True`` if its
        quaternion is equal to the identity quaternion.

        An **identity quaternion** has its scalar part equal to 1, and its
        vector part equal to 0, such that :math:`\\mathbf{q} = 1 + 0i + 0j + 0k`.

        .. math::
            \\left\\{
            \\begin{array}{ll}
                \\mathrm{True} & \\: \\mathbf{q}\\ = \\begin{pmatrix} 1 & 0 & 0 & 0 \\end{pmatrix} \\\\
                \\mathrm{False} & \\: \\mathrm{otherwise}
            \\end{array}
            \\right.

        Returns
        -------
        out : np.ndarray
            Array of booleans.

        Example
        -------
        >>> Q = QuaternionArray(np.random.random((3, 4))-0.5)
        >>> Q[1] = [1.0, 0.0, 0.0, 0.0]
        >>> Q.view()
        QuaternionArray([[-0.8061095 ,  0.42513151,  0.37790158, -0.16322091],
                         [ 1.        ,  0.        ,  0.        ,  0.        ],
                         [ 0.29613776,  0.21692562, -0.16253866, -0.91587493]])
        >>> Q.is_identity()
        array([False,  True, False])

        """
        if self.scalar_vector:
            return np.all(np.isclose(self.array, np.tile([1., 0., 0., 0.], (self.array.shape[0], 1))), axis=1)
        return np.all(np.isclose(self.array, np.tile([0., 0., 0., 1.], (self.array.shape[0], 1))), axis=1)

    def conjugate(self) -> np.ndarray:
        """
        Return the conjugate of all quaternions.

        Returns
        -------
        q* : numpy.ndarray
            Array of conjugated quaternions.

        Examples
        --------
        >>> Q = QuaternionArray(np.random.random((5, 4))-0.5)
        >>> Q.view()
        QuaternionArray([[-0.68487217,  0.45395092, -0.53551826, -0.19518931],
                         [ 0.49389483,  0.28781475, -0.7085184 , -0.41380217],
                         [-0.39583397,  0.46873203, -0.21517704,  0.75980563],
                         [ 0.57515971,  0.33286283,  0.23442397,  0.70953439],
                         [-0.34067259, -0.24989624,  0.5950285 , -0.68369229]])
        >>> Q.conjugate()
        array([[-0.68487217, -0.45395092,  0.53551826,  0.19518931],
               [ 0.49389483, -0.28781475,  0.7085184 ,  0.41380217],
               [-0.39583397, -0.46873203,  0.21517704, -0.75980563],
               [ 0.57515971, -0.33286283, -0.23442397, -0.70953439],
               [-0.34067259,  0.24989624, -0.5950285 ,  0.68369229]])
        """
        if self.scalar_vector:
            return self.array*np.array([1.0, -1.0, -1.0, -1.0])
        return self.array*np.array([-1.0, -1.0, -1.0, 1.0])

    def conj(self) -> np.ndarray:
        """
        Synonym of :meth:`conjugate`

        Returns
        -------
        q* : numpy.ndarray
            Array of conjugated quaternions.

        Examples
        --------
        >>> Q = QuaternionArray(np.random.random((5, 4))-0.5)
        >>> Q.view()
        QuaternionArray([[-0.68487217,  0.45395092, -0.53551826, -0.19518931],
                         [ 0.49389483,  0.28781475, -0.7085184 , -0.41380217],
                         [-0.39583397,  0.46873203, -0.21517704,  0.75980563],
                         [ 0.57515971,  0.33286283,  0.23442397,  0.70953439],
                         [-0.34067259, -0.24989624,  0.5950285 , -0.68369229]])
        >>> Q.conj()
        array([[-0.68487217, -0.45395092,  0.53551826,  0.19518931],
               [ 0.49389483, -0.28781475,  0.7085184 ,  0.41380217],
               [-0.39583397, -0.46873203,  0.21517704, -0.75980563],
               [ 0.57515971, -0.33286283, -0.23442397, -0.70953439],
               [-0.34067259,  0.24989624, -0.5950285 ,  0.68369229]])
        """
        return self.conjugate()

    def from_rpy(self, Angles: np.ndarray) -> np.ndarray:
        """
        Quaternion Array from given RPY angles.

        The quaternion can be constructed from the Aerospace cardanian angle
        sequence that follows the order :math:`\\phi\\to\\theta\\to\\psi`,
        where :math:`\\phi` is the **roll** (or *bank*) angle, :math:`\\theta`
        is the **pitch** (or *elevation*) angle, and :math:`\\psi` is the
        **yaw** (or *heading*) angle.

        The composing quaternions are:

        .. math::
            \\begin{array}{rcl}
            \\mathbf{q}_X &=& \\begin{pmatrix}\\cos\\frac{\\phi}{2} & \\sin\\frac{\\phi}{2} & 0 & 0\\end{pmatrix} \\\\ && \\\\
            \\mathbf{q}_Y &=& \\begin{pmatrix}\\cos\\frac{\\theta}{2} & 0 & \\sin\\frac{\\theta}{2} & 0\\end{pmatrix} \\\\ && \\\\
            \\mathbf{q}_Z &=& \\begin{pmatrix}\\cos\\frac{\\psi}{2} & 0 & 0 & \\sin\\frac{\\psi}{2}\\end{pmatrix}
            \\end{array}

        The elements of the final quaternion
        :math:`\\mathbf{q}=\\mathbf{q}_Z\\mathbf{q}_Y\\mathbf{q}_X = q_w+q_xi+q_yj+q_zk`
        are obtained as:

        .. math::
            \\begin{array}{rcl}
            q_w &=& \\cos\\frac{\\psi}{2}\\cos\\frac{\\theta}{2}\\cos\\frac{\\phi}{2} + \\sin\\frac{\\psi}{2}\\sin\\frac{\\theta}{2}\\sin\\frac{\\phi}{2} \\\\ && \\\\
            q_x &=& \\cos\\frac{\\psi}{2}\\cos\\frac{\\theta}{2}\\sin\\frac{\\phi}{2} - \\sin\\frac{\\psi}{2}\\sin\\frac{\\theta}{2}\\cos\\frac{\\phi}{2} \\\\ && \\\\
            q_y &=& \\cos\\frac{\\psi}{2}\\sin\\frac{\\theta}{2}\\cos\\frac{\\phi}{2} + \\sin\\frac{\\psi}{2}\\cos\\frac{\\theta}{2}\\sin\\frac{\\phi}{2} \\\\ && \\\\
            q_z &=& \\sin\\frac{\\psi}{2}\\cos\\frac{\\theta}{2}\\cos\\frac{\\phi}{2} - \\cos\\frac{\\psi}{2}\\sin\\frac{\\theta}{2}\\sin\\frac{\\phi}{2}
            \\end{array}

        .. warning::
            The Aerospace sequence :math:`\\phi\\to\\theta\\to\\psi` is only
            one of the `twelve possible rotation sequences
            <https://en.wikipedia.org/wiki/Euler_angles#Tait.E2.80.93Bryan_angles>`_
            around the main axes. Other sequences might be more suitable for
            other applications, but this one is the most common in practice.

        Parameters
        ----------
        Angles : numpy.ndarray
            N-by-3 cardanian angles, in radians, following the order: roll -> pitch -> yaw.

        Returns
        -------
        Q : numpy.ndarray
            Quaternion Array from roll-pitch-yaw angles.

        """
        _assert_iterables(Angles, 'Roll-Pitch-Yaw angles')
        Angles = np.copy(Angles)
        if Angles.ndim != 2 or Angles.shape[-1] != 3:
            raise ValueError(f"Expected `angles` must have shape (N, 3), got {Angles.shape}.")
        # RPY to Quaternion
        cy = np.cos(0.5*Angles[:, 2])
        sy = np.sin(0.5*Angles[:, 2])
        cp = np.cos(0.5*Angles[:, 1])
        sp = np.sin(0.5*Angles[:, 1])
        cr = np.cos(0.5*Angles[:, 0])
        sr = np.sin(0.5*Angles[:, 0])
        Q = np.zeros((Angles.shape[0], 4))
        Q[:, 0] = cy*cp*cr + sy*sp*sr
        Q[:, 1] = cy*cp*sr - sy*sp*cr
        Q[:, 2] = sy*cp*sr + cy*sp*cr
        Q[:, 3] = sy*cp*cr - cy*sp*sr
        return Q/np.linalg.norm(Q, axis=1)[:, None]

    def to_angles(self) -> np.ndarray:
        """
        Return corresponding roll-pitch-yaw angles of quaternion.

        Having a unit quaternion :math:`\\mathbf{q} = \\begin{pmatrix}q_w & q_x & q_y & q_z\\end{pmatrix}`,
        its corresponding roll-pitch-yaw angles [WikiConversions]_ are:

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

        Examples
        --------
        >>> Q = QuaternionArray(np.random.random((5, 4))-0.5)   # Five random Quaternions
        >>> Q.view()
        QuaternionArray([[-0.5874517 , -0.2181631 , -0.25175194,  0.73751361],
                         [ 0.64812786,  0.18534342,  0.73606315, -0.06155591],
                         [-0.0014204 ,  0.8146498 ,  0.26040532,  0.51820146],
                         [ 0.55231315, -0.6287687 , -0.02216051,  0.5469086 ],
                         [ 0.08694828, -0.96884826,  0.05115712, -0.22617689]])
        >>> Q.to_angles()
        array([[-0.14676831,  0.66566299, -1.84716657],
               [ 2.36496457,  1.35564472,  2.01193563],
               [ 2.61751194, -1.00664968,  0.91202161],
               [-1.28870906,  0.72519173,  1.00562317],
               [-2.92779394, -0.4437908 , -0.15391635]])

        """
        phi = np.arctan2(2.0*(self.w*self.x + self.y*self.z), 1.0 - 2.0*(self.x**2 + self.y**2))
        theta = np.arcsin(2.0*(self.w*self.y - self.z*self.x))
        psi = np.arctan2(2.0*(self.w*self.z + self.x*self.y), 1.0 - 2.0*(self.y**2 + self.z**2))
        return np.c_[phi, theta, psi]

    def to_DCM(self) -> np.ndarray:
        """
        Having *N* quaternions return *N* `direction cosine matrices
        <https://en.wikipedia.org/wiki/Euclidean_vector#Conversion_between_multiple_Cartesian_bases>`_
        in `SO(3) <https://en.wikipedia.org/wiki/3D_rotation_group>`_.

        Any **unit quaternion** has the form
        :math:`\\mathbf{q} = \\begin{pmatrix}q_w & \\mathbf{q}_v\\end{pmatrix}`,
        where :math:`\\mathbf{q}_v = \\begin{bmatrix}q_x & q_y & q_z\\end{bmatrix}`
        is the vector part, :math:`q_w` is the scalar part, and :math:`\\|\\mathbf{q}\\|=1`.

        The `rotation matrix <https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions>`_
        :math:`\\mathbf{R}` [WikiConversions]_ built from :math:`\\mathbf{q}`
        has the form:

        .. math::
            \\mathbf{R}(\\mathbf{q}) =
            \\begin{bmatrix}
            1 - 2(q_y^2 + q_z^2) & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\\\
            2(q_xq_y + q_wq_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_yq_z - q_wq_x) \\\\
            2(q_xq_z - q_wq_y) & 2(q_wq_x + q_yq_z) & 1 - 2(q_x^2 + q_y^2)
            \\end{bmatrix}

        The identity quaternion :math:`\\mathbf{q}_\\mathbf{I} = \\begin{pmatrix}1 & 0 & 0 & 0\\end{pmatrix}`,
        produces a :math:`3 \\times 3` Identity matrix :math:`\\mathbf{I}_3`.

        Returns
        -------
        DCM : numpy.ndarray
            N-by-3-by-3 Direction Cosine Matrices.

        Examples
        --------

        .. code-block:: python

            >>> Q = QuaternionArray(3)  # Three random quaternions
            >>> Q.view()
            QuaternionArray([[-0.75641558,  0.42233104,  0.39637415,  0.30390704],
                             [-0.52953832, -0.7187872 , -0.44551683,  0.06669994],
                             [ 0.264412  ,  0.15784685, -0.80536887,  0.50650928]])
            >>> Q.to_DCM()
            array([[[ 0.50105608,  0.79456226, -0.34294842],
                    [-0.12495782,  0.45855401,  0.87983735],
                    [ 0.85634593, -0.39799377,  0.32904804]],

                   [[ 0.59413175,  0.71110393,  0.37595034],
                    [ 0.56982324, -0.04220784, -0.82068263],
                    [-0.56772259,  0.70181885, -0.43028056]],

                   [[-0.81034134, -0.52210413, -0.2659966 ],
                    [ 0.01360439,  0.43706545, -0.89932681],
                    [ 0.58580016, -0.73238041, -0.3470693 ]]])
        """
        if not all(self.is_versor()):
            raise AttributeError("All quaternions must be versors to be represented as Direction Cosine Matrices.")
        R = np.zeros((self.num_qts, 3, 3))
        R[:, 0, 0] = 1.0 - 2.0*(self.y**2 + self.z**2)
        R[:, 1, 0] = 2.0*(self.x*self.y+self.w*self.z)
        R[:, 2, 0] = 2.0*(self.x*self.z-self.w*self.y)
        R[:, 0, 1] = 2.0*(self.x*self.y-self.w*self.z)
        R[:, 1, 1] = 1.0 - 2.0*(self.x**2 + self.z**2)
        R[:, 2, 1] = 2.0*(self.w*self.x+self.y*self.z)
        R[:, 0, 2] = 2.0*(self.x*self.z+self.w*self.y)
        R[:, 1, 2] = 2.0*(self.y*self.z-self.w*self.x)
        R[:, 2, 2] = 1.0 - 2.0*(self.x**2 + self.y**2)
        return R

    def average(self, span: Tuple[int, int] = None, weights: np.ndarray = None) -> np.ndarray:
        """
        Average quaternion using Markley's method [Markley2007]_

        It has to be clear that we intend to average **attitudes** rather than
        quaternions. It just happens that we represent these attitudes with
        unit quaternions, that is :math:`\\|\\mathbf{q}\\|=1`.

        The average quaternion :math:`\\bar{\\mathbf{q}}` should minimize a
        weighted sum of the squared `Frobenius norms
        <https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>`_ of
        attitude matrix differences:

        .. math::
            \\bar{\\mathbf{q}} = \\mathrm{arg min}\\sum_{i=1}^nw_i\\|\\mathbf{A}(\\mathbf{q}) - \\mathbf{A}(\\mathbf{q}_i)\\|_F^2

        Taking advantage of the attitude's orthogonality in SO(3), this can be
        rewritten as a maximization problem:

        .. math::
            \\bar{\\mathbf{q}} = \\mathrm{arg max} \\big\\{\\mathrm{tr}(\\mathbf{A}(\\mathbf{q})\\mathbf{B}^T)\\big\\}

        with:

        .. math::
            \\mathbf{B} = \\sum_{i=1}^nw_i\\mathbf{A}(\\mathbf{q}_i)

        We can verify the identity:

        .. math::
            \\mathrm{tr}(\\mathbf{A}(\\mathbf{q})\\mathbf{B}^T) = \\mathbf{q}^T\\mathbf{Kq}

        using Davenport's symmetric traceless :math:`4\\times 4` matrix:

        .. math::
            \\mathbf{K}=4\\mathbf{M}-w_\\mathrm{tot}\\mathbf{I}_{4\\times 4}

        where :math:`w_\\mathrm{tot}=\\sum_{i=1}^nw_i`, and :math:`\\mathbf{M}` is the
        :math:`4\\times 4` matrix:

        .. math::
            \\mathbf{M} = \\sum_{i=1}^nw_i\\mathbf{q}_i\\mathbf{q}_i^T

        .. warning::
            In this case, the product :math:`\\mathbf{q}_i\\mathbf{q}_i^T` is a
            *normal matrix multiplication*, not the Hamilton product, of the
            elements of each quaternion.

        Finally, the average quaternion :math:`\\bar{\\mathbf{q}}` is the
        eigenvector corresponding to the maximum eigenvalue of :math:`\\mathbf{M}`,
        which in turns maximizes the procedure:

        .. math::
            \\bar{\\mathbf{q}} = \\mathrm{arg max} \\big\\{\\mathbf{q}^T\\mathbf{Mq}\\big\\}

        Changing the sign of any :math:`\\mathbf{q}_i` does not change the
        value of :math:`\\mathbf{M}`. Thus, the averaging procedure determines
        :math:`\\bar{\\mathbf{q}}` up to a sign, which is consistent with the
        nature of the attitude representation using unit quaternions.

        Parameters
        ----------
        span : tuple, default: None
            Span of data to average. If none given, it averages all.
        weights : numpy.ndarray, default: None
            Weights of each quaternion. If none given, they are all equal to 1.

        Returns
        -------
        q : numpy.ndarray
            Average quaternion.

        Example
        -------
        >>> qts = np.tile([1., -2., 3., -4], (5, 1))    # Five equal quaternions
        >>> v = np.random.standard_normal((5, 4))*0.1   # Zero-mean gaussian noise
        >>> Q = QuaternionArray(qts + v)
        >>> Q.view()
        QuaternionArray([[ 0.17614144, -0.39173347,  0.56303067, -0.70605634],
                         [ 0.17607515, -0.3839024 ,  0.52673809, -0.73767437],
                         [ 0.16823806, -0.35898889,  0.53664261, -0.74487424],
                         [ 0.17094453, -0.3723117 ,  0.54109885, -0.73442086],
                         [ 0.1862619 , -0.38421818,  0.5260265 , -0.73551276]])
        >>> Q.average()
        array([-0.17557859,  0.37832975, -0.53884688,  0.73190355])

        The result is as expected, remembering that a quaternion with opposite
        signs on each element represents the same orientation.
        """
        if not all(self.is_versor()):
            raise AttributeError("All quaternions must be versors to be averaged.")
        q = np.c_[self.w, self.v]
        if span is not None:
            if hasattr(span, '__iter__') and len(span) == 2:
                q = q[span[0]:span[1]]
            else:
                raise ValueError("span must be a pair of integers indicating the indices of the data.")
        if weights is not None:
            if weights.ndim > 1:
                raise ValueError("The weights must be in a one-dimensional array.")
            if weights.size != q.shape[0]:
                raise ValueError("The number of weights do not match the number of quaternions.")
            q *= weights[:, None]
        eigvals, eigvecs = np.linalg.eig(q.T@q)
        q_avg = eigvecs[:, eigvals.argmax()]
        if self.scalar_vector:
            return q_avg
        return np.roll(q_avg, -1)

    def remove_jumps(self) -> None:
        """
        Flip sign of opposite quaternions.

        Some estimations and measurements of quaternions might have "jumps"
        produced when their values are multiplied by -1. They still represent
        the same rotation, but the continuity of the signal "flips", making it
        difficult to evaluate continuously.

        To revert this, the flipping instances are identified and the next
        samples are multiplied by -1, until it "flips back". This
        function does that correction over all values of the attribute ``array``.

        Examples
        --------
        >>> qts = np.tile([1., -2., 3., -4], (5, 1))    # Five equal arrays
        >>> v = np.random.randn(5, 4)*0.1               # Gaussian noise
        >>> Q = QuaternionArray(qts + v)
        >>> Q.view()
        QuaternionArray([[ 0.17614144, -0.39173347,  0.56303067, -0.70605634],
                        [ 0.17607515, -0.3839024 ,  0.52673809, -0.73767437],
                        [ 0.16823806, -0.35898889,  0.53664261, -0.74487424],
                        [ 0.17094453, -0.3723117 ,  0.54109885, -0.73442086],
                        [ 0.1862619 , -0.38421818,  0.5260265 , -0.73551276]])
        >>> Q[1:3] *= -1    # 2nd and 3rd Quaternions "flip"
        >>> Q.view()
        QuaternionArray([[ 0.17614144, -0.39173347,  0.56303067, -0.70605634],
                        [-0.17607515,  0.3839024 , -0.52673809,  0.73767437],
                        [-0.16823806,  0.35898889, -0.53664261,  0.74487424],
                        [ 0.17094453, -0.3723117 ,  0.54109885, -0.73442086],
                        [ 0.1862619 , -0.38421818,  0.5260265 , -0.73551276]])
        >>> Q.remove_jumps()
        >>> Q.view()
        QuaternionArray([[ 0.17614144, -0.39173347,  0.56303067, -0.70605634],
                        [ 0.17607515, -0.3839024 ,  0.52673809, -0.73767437],
                        [ 0.16823806, -0.35898889,  0.53664261, -0.74487424],
                        [ 0.17094453, -0.3723117 ,  0.54109885, -0.73442086],
                        [ 0.1862619 , -0.38421818,  0.5260265 , -0.73551276]])
        """
        q_diff = np.diff(self.array, axis=0)
        jumps = np.nonzero(np.where(np.linalg.norm(q_diff, axis=1)>1, 1, 0))[0]+1
        if len(jumps) % 2:
            jumps = np.append(jumps, [len(q_diff)+1])
        jump_pairs = jumps.reshape((len(jumps)//2, 2))
        for j in jump_pairs:
            self.array[j[0]:j[1]] *= -1.0

    def rotate_by(self, q: np.ndarray, order: str = 'H') -> np.ndarray:
        """
        Rotate all Quaternions in the array around quaternion :math:`\\mathbf{q}`.

        Parameters
        ----------
        q : numpy.ndarray
            4 element array to rotate around.

        Returns
        -------
        Q' : numpy.ndarray
            4-by-N array with all Quaternions rotated around q.

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
        _assert_iterables(q, 'Quaternion')
        q = np.copy(q)
        if q.size != 4:
            raise ValueError("Given quaternion to rotate about must have 4 elements.")
        q /= np.linalg.norm(q)
        qQ = np.zeros_like(self.array)
        if order.upper() == 'S':
            q = np.roll(q, -1)
        qQ[:, 0] = q[0]*self.w - q[1]*self.x - q[2]*self.y - q[3]*self.z
        qQ[:, 1] = q[0]*self.x + q[1]*self.w + q[2]*self.z - q[3]*self.y
        qQ[:, 2] = q[0]*self.y - q[1]*self.z + q[2]*self.w + q[3]*self.x
        qQ[:, 3] = q[0]*self.z + q[1]*self.y - q[2]*self.x + q[3]*self.w
        qQ /= np.linalg.norm(qQ, axis=1)[:, None]
        return qQ

    def angular_velocities(self, dt: float) -> np.ndarray:
        """
        Compute the angular velocity between N Quaternions.

        It assumes a constant sampling rate of ``dt`` seconds, and returns the
        angular velocity around the X-, Y- and Z-axis (roll-pitch-yaw angles),
        in radians per second.

        The angular velocities :math:`\\omega_x`, :math:`\\omega_y`, and
        :math:`\\omega_z` are computed from quaternions :math:`\\mathbf{q}_t=\\Big(q_w(t), q_x(t), q_y(t), q_z(t)\\Big)`
        and :math:`\\mathbf{q}_{t+\\Delta t}=\\Big(q_w(t+\\Delta t), q_x(t+\\Delta t), q_y(t+\\Delta t), q_z(t+\\Delta t)\\Big)`
        as:

        .. math::
            \\begin{array}{rcl}
            \\omega_x &=& \\frac{2}{\\Delta t}\\Big(q_w(t) q_x(t+\\Delta t) - q_x(t) q_w(t+\\Delta t) - q_y(t) q_z(t+\\Delta t) + q_z(t) q_y(t+\\Delta t)\\Big) \\\\ \\\\
            \\omega_y &=& \\frac{2}{\\Delta t}\\Big(q_w(t) q_y(t+\\Delta t) + q_x(t) q_z(t+\\Delta t) - q_y(t) q_w(t+\\Delta t) - q_z(t) q_x(t+\\Delta t)\\Big) \\\\ \\\\
            \\omega_z &=& \\frac{2}{\\Delta t}\\Big(q_w(t) q_z(t+\\Delta t) - q_x(t) q_y(t+\\Delta t) + q_y(t) q_x(t+\\Delta t) - q_z(t) q_w(t+\\Delta t)\\Big)
            \\end{array}

        where :math:`\\Delta t` is the time step between consecutive
        quaternions [MarioGC1]_.

        Parameters
        ----------
        dt : float
            Time step, in seconds, between consecutive Quaternions.

        Returns
        -------
        w : numpy.ndarray
            (N-1)-by-3 array with angular velocities in rad/s.

        """
        if not isinstance(dt, float):
            raise TypeError(f"dt must be a float. Got {type(dt)}.")
        if dt <= 0:
            raise ValueError(f"dt must be greater than zero. Got {dt}.")
        w = np.c_[
            self.w[:-1]*self.x[1:] - self.x[:-1]*self.w[1:] - self.y[:-1]*self.z[1:] + self.z[:-1]*self.y[1:],
            self.w[:-1]*self.y[1:] + self.x[:-1]*self.z[1:] - self.y[:-1]*self.w[1:] - self.z[:-1]*self.x[1:],
            self.w[:-1]*self.z[1:] - self.x[:-1]*self.y[1:] + self.y[:-1]*self.x[1:] - self.z[:-1]*self.w[1:]]
        return 2.0 * w / dt

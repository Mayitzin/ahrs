# -*- coding: utf-8 -*-
"""
Direction Cosine Matrix
=======================

Rotations are linear operations preserving vector length. These rotation
operators are represented with matrices.

Let us assume there is a linear operator represented by a :math:`3\\times 3`
matrix:

.. math::
    \\mathbf{R} =
    \\begin{bmatrix}
    r_{11} & r_{12} & r_{13} \\\\ r_{21} & r_{22} & r_{23} \\\\ r_{31} & r_{32} & r_{33}
    \\end{bmatrix} \\in \\mathbb{R}^{3\\times 3}

where

.. math::
    \\begin{array}{lcr}
    \\mathbf{r}_1 = \\begin{bmatrix}r_{11}\\\\ r_{21} \\\\ r_{31} \\end{bmatrix} \\; , &
    \\mathbf{r}_2 = \\begin{bmatrix}r_{12}\\\\ r_{22} \\\\ r_{32} \\end{bmatrix} \\; , &
    \\mathbf{r}_3 = \\begin{bmatrix}r_{13}\\\\ r_{23} \\\\ r_{33} \\end{bmatrix}
    \\end{array}

are unit vectors `orthogonal <https://en.wikipedia.org/wiki/Orthogonality_(mathematics)>`_
to each other. All matrices satisfying this orthogonality are called
`orthogonal matrices <https://en.wikipedia.org/wiki/Orthogonal_matrix>`_.

The difference, in three dimensions, between any given orthogonal frame and a
base coordinate frame is the **orientation** or **attitude**.

A vector :math:`\\mathbf{v}\\in\\mathbb{R}^3` is used to represent a point in a
three-dimensional `euclidean space <https://en.wikipedia.org/wiki/Euclidean_space>`_.
When :math:`\\mathbf{v}` is multiplied with the matrix :math:`\\mathbf{R}`,
the result is a new vector :math:`\\mathbf{v}'\\in\\mathbb{R}^3`:

.. math::

    \\mathbf{v}' = \\mathbf{Rv}

We observe that :math:`\\mathbf{RR}^{-1}=\\mathbf{RR}^T=\\mathbf{R}^T\\mathbf{R}=\\mathbf{I}`,
indicating that the inverse of :math:`\\mathbf{R}` is its transpose. So,

.. math::
    \\mathbf{v} = \\mathbf{R}^T\\mathbf{v}'

The determinant of this matrix is always equal to :math:`+1`. This means,
its product with any vector will leave the vector's length unchanged.

:math:`3\\times 3` matrices conforming to these properties (orthogonality,
:math:`\\mathbf{R}^T=\\mathbf{R}^{-1}`, and :math:`\\mathrm{det}(\\mathbf{R})=
+1`) belong to the special orthogonal group :math:`SO(3)`, also known as the
`rotation group <https://en.wikipedia.org/wiki/3D_rotation_group>`_.

Even better, the product of two or more rotation matrices yields another
rotation matrix in :math:`SO(3)`.

`Direction cosines <https://en.wikipedia.org/wiki/Direction_cosine>`_ are
cosines of angles between a vector and a base coordinate frame
:cite:p:`Wiki_DirectionCosine`. In this case, the direction cosines describe
the differences between orthogonal vectors :math:`\\mathbf{r}_i` and the base
frame. The matrix containing these differences is commonly named the
**Direction Cosine Matrix**.

These matrices are used for two main purposes:

- To describe any frame's orientation relative to a base frame.
- To transform vectors (representing points in three dimensions) from one frame
  to another. This is a rotation operation.

Because of the latter, the DCM is also known as the **rotation matrix**.

DCMs are, therefore, the most common representation of rotations
:cite:p:`Wolfram_RotationMatrix`, especially in real applications of spacecraft
tracking and location.

Throughout this package they will be used to represent the attitudes with
respect to the global frame.

"""

from typing import Tuple
import numpy as np
from .mathfuncs import skew
from .constants import DEG2RAD
# Functions to convert DCM to quaternion representation
from .orientation import shepperd
from .orientation import hughes
from .orientation import chiaverini
from .orientation import itzhack
from .orientation import sarabandi

# Other useful functions
from ..utils.core import _assert_numerical_iterable

def _assert_SO3(array: np.ndarray, R_name: str = 'R'):
    if array.shape[-2:] != (3, 3) or array.ndim not in [2, 3]:
        raise ValueError(f"{R_name} must have shape (3, 3) or (N, 3, 3), got {array.shape}.")
    if array.ndim < 3:
        in_SO3 = np.isclose(np.linalg.det(array), 1.0)
        in_SO3 &= np.allclose(array@array.T, np.identity(3))
    else:
        in_SO3 = np.allclose(np.linalg.det(array), np.ones(array.shape[0]))
        in_SO3 &= np.allclose([x@x.T for x in array], np.identity(3))
    if not in_SO3:
        raise ValueError("Given attitude is not in SO(3)")

def rotation(ax: str | int = None, ang: float = 0.0, degrees: bool = False) -> np.ndarray:
    """
    Return a Direction Cosine Matrix

    The rotation matrix :math:`\\mathbf{R}` [1]_ is created for the given axis
    with the given angle :math:`\\theta`. Where the possible rotation axes are:

    .. math::

        \\mathbf{R}_X(\\theta) =
        \\begin{bmatrix}
        1 & 0 & 0 \\\\
        0 & \\cos \\theta & -\\sin \\theta \\\\
        0 & \\sin \\theta &  \\cos \\theta
        \\end{bmatrix}

        \\mathbf{R}_Y(\\theta) =
        \\begin{bmatrix}
        \\cos \\theta & 0 & \\sin \\theta \\\\
        0 & 1 & 0 \\\\
        -\\sin \\theta & 0 & \\cos \\theta
        \\end{bmatrix}

        \\mathbf{R}_Z(\\theta) =
        \\begin{bmatrix}
        \\cos \\theta & -\\sin \\theta & 0 \\\\
        \\sin \\theta &  \\cos \\theta & 0 \\\\
        0 & 0 & 1
        \\end{bmatrix}

    where :math:`\\theta` is a float number representing the angle of rotation
    in degrees.

    Parameters
    ----------
    ax : string or int
        Axis to rotate around. Possible are `X`, `Y` or `Z` (upper- or
        lowercase) or the corresponding axis index 0, 1 or 2. Defaults to 'z'.
    angle : float, default: 0.0
        Angle, in degrees, to rotate around.
    degrees : bool, default: False
        If True, the angle is given in degrees. Otherwise, it is given in
        radians.

    Returns
    -------
    R : numpy.ndarray
        3-by-3 Direction Cosine Matrix.

    Examples
    --------
    >>> from ahrs import rotation
    >>> rotation()
    array([[1. 0. 0.],
           [0. 1. 0.],
           [0. 0. 1.]])
    >>> rotation('z', 30.0)
    array([[ 0.8660254 -0.5        0.       ],
           [ 0.5        0.8660254  0.       ],
           [ 0.         0.         1.       ]])
    >>> # Accepts angle input as string
    ... rotation('x', '-30')
    array([[ 1.         0.         0.       ],
           [ 0.         0.8660254  0.5      ],
           [ 0.        -0.5        0.8660254]])

    Handles wrong inputs

    >>> rotation('false_axis', 'invalid_angle')
    array([[1. 0. 0.],
           [0. 1. 0.],
           [0. 0. 1.]])
    >>> rotation(None, None)
    array([[1. 0. 0.],
           [0. 1. 0.],
           [0. 0. 1.]])

    References
    ----------
    .. [1] http://mathworld.wolfram.com/RotationMatrix.html

    """
    # Default values
    valid_axes = list('xyzXYZ')
    I_3 = np.identity(3)
    if ax is None:
        ax = "z"
    if isinstance(ax, int):
        if ax < 0:
            ax = 2      # Negative axes default to 2 (Z-axis)
        ax = valid_axes[ax] if ax < 3 else "z"
    try:
        ang = float(ang)
    except ValueError:
        return I_3
    # Handle input
    if ang == 0.0:
        return I_3
    if np.isclose((ang*DEG2RAD) % (2*np.pi), 0.0):
        return I_3
    # Return 3-by-3 Identity matrix if invalid input
    if ax not in valid_axes:
        return I_3
    # Set sin and cos values
    if degrees:
        ang = ang*DEG2RAD
    ca, sa = np.cos(ang), np.sin(ang)
    # Compute rotation
    if ax.lower() == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    if ax.lower() == "y":
        return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    if ax.lower() == "z":
        return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])

def rot_seq(axes: list | str = None, angles: list | float = None, degrees: bool = False) -> np.ndarray:
    """
    Rotation matrix from a sequence of Euler angles

    A rotation matrix :math:`\\mathbf{R}` :cite:p:`Wolfram_RotationMatrix` is
    created from the given list of angles rotating around the given axes order.

    Parameters
    ----------
    axes : list of str
        List of rotation axes.
    angles : list of floats
        List of rotation angles.
    degrees : bool, default: False
        If True, the angle was given in degrees. Otherwise, it was given in
        radians.

    Returns
    -------
    R : numpy.ndarray
        3-by-3 Direction Cosine Matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import random
    >>> num_rotations = 5
    >>> axis_order = random.choices("XYZ", k=num_rotations)
    >>> axis_order
    ['Z', 'Z', 'X', 'Z', 'Y']
    >>> angles = np.random.uniform(low=-180.0, high=180.0, size=num_rotations)
    >>> angles
    array([-139.24498146,  99.8691407, -171.30712526, -60.57132043,
             17.4475838 ])
    >>> R = rot_seq(axis_order, angles)
    >>> R   # R = R_z(-139.24) R_z(99.87) R_x(-171.31) R_z(-60.57) R_y(17.45)
    array([[ 0.85465231  0.3651317   0.36911822]
           [ 0.3025091  -0.92798938  0.21754072]
           [ 0.4219688  -0.07426006 -0.90356393]])

    """
    valid_axes = list('xyzXYZ')
    if axes is None:
        axes = ['z']
    if isinstance(axes, int):
        axes = ['z'] if abs(axes) > 2 else valid_axes[axes]
    if isinstance(axes, str):
        axes = list(axes)
    if not isinstance(axes, list):
        raise TypeError(f"Axes must be a list of 'x', 'y' or 'z' characters. Got {type(axes)}")
    if not set(axes).issubset(set(valid_axes)):
        raise ValueError("Axes must be a list of 'x', 'y' or 'z' characters")
    R = np.identity(3)
    num_rotations = len(axes)
    if num_rotations < 1:
        return R
    if angles is None:
        # Creates random rotations around each given axis if None given in `angles`
        angles = np.random.uniform(low=-180.0, high=180.0, size=num_rotations)
    for x in angles:
        if not isinstance(x, (float, int)):
            raise TypeError(f"Angles must be float or int numbers. Got {type(x)}")
    # All good. Perform the matrix multiplications
    for i in range(num_rotations-1, -1, -1):
        R = rotation(axes[i], angles[i], degrees=degrees) @ R
    return R

class DCM(np.ndarray):
    """
    Direction Cosine Matrix in SO(3)

    Class to represent a Direction Cosine Matrix. It is built from a 3-by-3
    array, but it can also be built from 3-dimensional vectors representing the
    roll-pitch-yaw angles, a quaternion, or an axis-angle pair representation.

    Parameters
    ----------
    array : array-like, default: None
        Array to build the DCM with.
    q : array-like, default: None
        Quaternion to convert to DCM.
    rpy : array-like, default: None
        Array with roll->pitch->yaw angles.
    euler : tuple, default: None
        Dictionary with a set of angles as a pair of string and array.
    axang : tuple, default: None
        Tuple with an array and a float of the axis and the angle
        representation.

    Attributes
    ----------
    A : numpy.ndarray
        Array with the 3-by-3 direction cosine matrix.

    Examples
    --------
    All DCM are created as an identity matrix, which means no rotation.

    >>> from ahrs import DCM
    >>> from ahrs import DEG2RAD
    >>> DCM()
    DCM([[1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.]])

    A rotation around a single axis can be defined by giving the desired axis
    and its value, in degrees.

    >>> DCM(x=10.0*DEG2RAD)
    DCM([[ 1.        ,  0.        ,  0.        ],
         [ 0.        ,  0.98480775, -0.17364818],
         [ 0.        ,  0.17364818,  0.98480775]])
    >>> DCM(y=20.0*DEG2RAD)
    DCM([[ 0.93969262,  0.        ,  0.34202014],
         [ 0.        ,  1.        ,  0.        ],
         [-0.34202014,  0.        ,  0.93969262]])
    >>> DCM(z=30.0*DEG2RAD)
    DCM([[ 0.8660254, -0.5      ,  0.       ],
         [ 0.5      ,  0.8660254,  0.       ],
         [ 0.       ,  0.       ,  1.       ]])

    If we want a rotation conforming the roll-pitch-yaw sequence, we can give
    the corresponding angles.

    >>> DCM(rpy=np.array([30.0, 20.0, 10.0])*DEG2RAD)
    DCM([[ 0.81379768, -0.44096961,  0.37852231],
         [ 0.46984631,  0.88256412,  0.01802831],
         [-0.34202014,  0.16317591,  0.92541658]])

    .. note::
        Notice the angles are given in reverse order, as it is the way the
        matrices are multiplied.

    >>> DCM(z=30.0*DEG2RAD) @ DCM(y=20.0*DEG2RAD) @ DCM(x=10.0*DEG2RAD)
    DCM([[ 0.81379768, -0.44096961,  0.37852231],
         [ 0.46984631,  0.88256412,  0.01802831],
         [-0.34202014,  0.16317591,  0.92541658]])

    But also a different sequence can be defined, if given as a tuple with two
    elements: the order of the axis to rotate about, and the value of the
    rotation angles (again in reverse order)

    >>> DCM(euler=('zyz', np.array([40.0, 50.0, 60.0])*DEG2RAD))
    DCM([[-0.31046846, -0.74782807,  0.58682409],
         [ 0.8700019 ,  0.02520139,  0.49240388],
         [-0.38302222,  0.66341395,  0.64278761]])
    >>> DCM(z=40.0*DEG2RAD) @ DCM(y=50.0*DEG2RAD) @ DCM(z=60.0*DEG2RAD)
    DCM([[-0.31046846, -0.74782807,  0.58682409],
         [ 0.8700019 ,  0.02520139,  0.49240388],
         [-0.38302222,  0.66341395,  0.64278761]])

    Another option is to build the rotation matrix from a quaternion:

    >>> DCM(q=[1., 2., 3., 4.])
    DCM([[-0.66666667,  0.13333333,  0.73333333],
         [ 0.66666667, -0.33333333,  0.66666667],
         [ 0.33333333,  0.93333333,  0.13333333]])

    The quaternions are automatically normalized to make them versors and be
    used as rotation operators.

    Finally, we can also build the rotation matrix from an axis-angle
    representation:

    >>> DCM(axang=([1., 2., 3.], 60.0*DEG2RAD))
    DCM([[-0.81295491,  0.52330834,  0.25544608],
         [ 0.03452394, -0.3945807 ,  0.91821249],
         [ 0.58130234,  0.75528436,  0.30270965]])

    The axis of rotation is also normalized to be used as part of the rotation
    operator.

    """
    def __new__(subtype, array: np.ndarray = None, **kwargs):
        if array is None:
            array = np.identity(3)
            if 'q' in kwargs:
                array = DCM.from_q(DCM, kwargs.pop('q'))
            if any(x.lower() in ['x', 'y', 'z'] for x in kwargs):
                array = np.identity(3)
                array = array@rotation('x', kwargs.pop('x', 0.0), degrees=kwargs.get('degrees', False))
                array = array@rotation('y', kwargs.pop('y', 0.0), degrees=kwargs.get('degrees', False))
                array = array@rotation('z', kwargs.pop('z', 0.0), degrees=kwargs.get('degrees', False))
            if 'rpy' in kwargs:
                angles = kwargs.pop('rpy')
                _assert_numerical_iterable(angles, "Roll-Pitch-Yaw angles")
                if len(angles) != 3:
                    raise ValueError("roll-pitch-yaw angles must be an array with 3 rotations in degrees.")
                array = rot_seq('zyx', angles)
            if 'euler' in kwargs:
                seqangs = kwargs.pop('euler')
                if not isinstance(seqangs, tuple):
                    raise TypeError("Euler angles must be given as a tuple with a sequence, as string or list of strings, and its angles, as list of floats.")
                if len(seqangs) != 2:
                    raise ValueError("Euler angles must be given as a tuple with a sequence, as string or list of strings, and its angles, as list of floats.")
                seq, angs = seqangs
                # if not all(isinstance(i, str) for i in seq):
                #     raise TypeError("Euler sequence must be a string or list of strings.")
                _assert_numerical_iterable(angs, "Euler angles")
                array = rot_seq(seq, angs)
            if 'axang' in kwargs:
                ax, ang = kwargs.pop('axang')
                array = DCM.from_axisangle(DCM, np.array(ax), ang)
        _assert_numerical_iterable(array, "Direction Cosine Matrix")
        _assert_SO3(array, "Direction Cosine Matrix")
        # Create the ndarray instance of type DCM. This will call the standard
        # ndarray constructor, but return an object of type DCM.
        obj = super(DCM, subtype).__new__(subtype, array.shape, float, array)
        obj.A = array
        return obj

    @property
    def I(self) -> np.ndarray:
        """
        synonym of property :meth:`inv`.

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0])
        >>> R.view()
        DCM([[ 0.92541658, -0.31879578, -0.20487413],
             [ 0.16317591,  0.82317294, -0.54383814],
             [ 0.34202014,  0.46984631,  0.81379768]])
        >>> R.I
        array([[ 0.92541658,  0.16317591,  0.34202014],
               [-0.31879578,  0.82317294,  0.46984631],
               [-0.20487413, -0.54383814,  0.81379768]])

        Returns
        -------
        np.ndarray
            Inverse of the DCM.
        """
        return self.A.T

    @property
    def inv(self) -> np.ndarray:
        """
        Inverse of the DCM.

        The direction cosine matrix belongs to the Special Orthogonal group
        `SO(3) <https://en.wikipedia.org/wiki/SO(3)>`_, where its transpose is
        equal to its inverse:

        .. math::
            \\mathbf{R}^T\\mathbf{R} = \\mathbf{R}^{-1}\\mathbf{R} = \\mathbf{I}_3

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0])
        >>> R.view()
        DCM([[ 0.92541658, -0.31879578, -0.20487413],
             [ 0.16317591,  0.82317294, -0.54383814],
             [ 0.34202014,  0.46984631,  0.81379768]])
        >>> R.inv
        array([[ 0.92541658,  0.16317591,  0.34202014],
               [-0.31879578,  0.82317294,  0.46984631],
               [-0.20487413, -0.54383814,  0.81379768]])

        Returns
        -------
        np.ndarray
            Inverse of the DCM.
        """
        return self.A.T

    @property
    def det(self) -> float:
        """
        Synonym of property :meth:`determinant`.

        Returns
        -------
        float
            Determinant of the DCM.

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0])
        >>> R.view()
        DCM([[ 0.92541658, -0.31879578, -0.20487413],
             [ 0.16317591,  0.82317294, -0.54383814],
             [ 0.34202014,  0.46984631,  0.81379768]])
        >>> R.det
        1.0000000000000002
        """
        return np.linalg.det(self.A)

    @property
    def determinant(self) -> float:
        """
        Determinant of the DCM.

        Given a direction cosine matrix :math:`\\mathbf{R}`, its determinant
        :math:`|\\mathbf{R}|` is found as:

        .. math::
            |\\mathbf{R}| =
            \\begin{vmatrix}r_{11} & r_{12} & r_{13} \\\\ r_{21} & r_{22} & r_{23} \\\\ r_{31} & r_{32} & r_{33}\\end{vmatrix}=
            r_{11}\\begin{vmatrix}r_{22} & r_{23}\\\\r_{32} & r_{33}\\end{vmatrix} -
            r_{12}\\begin{vmatrix}r_{21} & r_{23}\\\\r_{31} & r_{33}\\end{vmatrix} +
            r_{13}\\begin{vmatrix}r_{21} & r_{22}\\\\r_{31} & r_{32}\\end{vmatrix}

        where the determinant of :math:`\\mathbf{B}\\in\\mathbb{R}^{2\\times 2}`
        is:

        .. math::
            |\\mathbf{B}|=\\begin{vmatrix}b_{11}&b_{12}\\\\b_{21}&b_{22}\\end{vmatrix}=b_{11}b_{22}-b_{12}b_{21}

        All matrices in SO(3), to which direction cosine matrices belong, have
        a determinant equal to :math:`+1`.

        Returns
        -------
        float
            Determinant of the DCM.

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0])
        >>> R.view()
        DCM([[ 0.92541658, -0.31879578, -0.20487413],
             [ 0.16317591,  0.82317294, -0.54383814],
             [ 0.34202014,  0.46984631,  0.81379768]])
        >>> R.determinant
        1.0000000000000002
        """
        return np.linalg.det(self.A)

    @property
    def fro(self) -> float:
        """
        Synonym of property :meth:`frobenius`.

        Returns
        -------
        float
            Frobenius norm of the DCM.

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0])
        >>> R.view()
        DCM([[ 0.92541658, -0.31879578, -0.20487413],
             [ 0.16317591,  0.82317294, -0.54383814],
             [ 0.34202014,  0.46984631,  0.81379768]])
        >>> R.fro
        1.7320508075688774
        """
        return np.linalg.norm(self.A, 'fro')

    @property
    def frobenius(self) -> float:
        """
        Frobenius norm of the DCM.

        The `Frobenius norm <https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm>`_
        of a matrix :math:`\\mathbf{A}` is defined as:

        .. math::
            \\|\\mathbf{A}\\|_F = \\sqrt{\\sum_{i=1}^m\\sum_{j=1}^n|a_{ij}|^2}

        Returns
        -------
        float
            Frobenius norm of the DCM.

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0])
        >>> R.view()
        DCM([[ 0.92541658, -0.31879578, -0.20487413],
             [ 0.16317591,  0.82317294, -0.54383814],
             [ 0.34202014,  0.46984631,  0.81379768]])
        >>> R.frobenius
        1.7320508075688774
        """
        return np.linalg.norm(self.A, 'fro')

    @property
    def log(self) -> np.ndarray:
        """
        Logarithm of DCM.

        The logarithmic map is defined as the inverse of the exponential map
        :cite:p:`cardoso2009`. It corresponds to the logarithm given by the Rodrigues
        rotation formula:

        .. math::
            \\log(\\mathbf{R}) = \\frac{\\theta(\\mathbf{R}^T-\\mathbf{R})}{2\\sin\\theta}

        with :math:`\\theta=\\arccos\\Big(\\frac{\\mathrm{tr}(\\mathbf{R}-1)}{2}\\Big)`.

        The angle of rotation :math:`-\\pi < \\theta < \\pi`, satisfies
        :math:`1+2\\cos\\theta = \\mathrm{tr}(\\mathbf{R})`.

        When :math:`\\theta=0`, we have the trivial case :math:`\\mathbf{R}=\\mathbf{I}`:

        .. math::

            \\log\\Bigg(\\begin{bmatrix}1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1\\end{bmatrix}\\Bigg) =
            \\begin{bmatrix}0 & 0 & 0 \\\\ 0 & 0 & 0 \\\\ 0 & 0 & 0\\end{bmatrix}

        Returns
        -------
        log : numpy.ndarray
            Logarithm of DCM

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0] * ahrs.DEG2RAD)
        >>> R.view()
        DCM([[ 0.92541658, -0.31879578, -0.20487413],
             [ 0.16317591,  0.82317294, -0.54383814],
             [ 0.34202014,  0.46984631,  0.81379768]])
        >>> R.log
        array([[ 0.        ,  0.26026043,  0.29531805],
               [-0.26026043,  0.        ,  0.5473806 ],
               [-0.29531805, -0.5473806 ,  0.        ]])

        """
        trace_R = self.A.trace()
        if np.isclose(trace_R, 3.0):
            return np.zeros((3, 3))
        theta = np.arccos((self.A.trace()-1)/2)
        nom = theta * (self.A.T - self.A)
        denom = 2*np.sin(theta)
        logR = nom / denom
        return logR

    @property
    def adjugate(self) -> np.ndarray:
        """
        Return the adjugate of the DCM.

        The adjugate, a.k.a. *classical adjoint*, of a matrix :math:`\\mathbf{A}`
        is the transpose of its *cofactor matrix*. For orthogonal matrices, it
        simplifies to:

        .. math::
            \\mathrm{adj}(\\mathbf{A}) = \\mathrm{det}(\\mathbf{A})\\mathbf{A}^T

        Returns
        -------
        np.ndarray
            Adjugate of the DCM.

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0])
        >>> R.view()
        DCM([[ 0.92541658, -0.31879578, -0.20487413],
             [ 0.16317591,  0.82317294, -0.54383814],
             [ 0.34202014,  0.46984631,  0.81379768]])
        >>> R.adjugate
        array([[ 0.92541658,  0.16317591,  0.34202014],
               [-0.31879578,  0.82317294,  0.46984631],
               [-0.20487413, -0.54383814,  0.81379768]])
        """
        return np.linalg.det(self.A)*self.A.T

    @property
    def adj(self) -> np.ndarray:
        """
        Synonym of property :meth:`adjugate`.

        Returns
        -------
        np.ndarray
            Adjugate of the DCM.

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0])
        >>> R.view()
        DCM([[ 0.92541658, -0.31879578, -0.20487413],
             [ 0.16317591,  0.82317294, -0.54383814],
             [ 0.34202014,  0.46984631,  0.81379768]])
        >>> R.adj
        array([[ 0.92541658,  0.16317591,  0.34202014],
               [-0.31879578,  0.82317294,  0.46984631],
               [-0.20487413, -0.54383814,  0.81379768]])
        """
        return np.linalg.det(self.A)*self.A.T

    def to_axisangle(self) -> Tuple[np.ndarray, float]:
        """
        Return axis-angle representation of the DCM.

        Defining a *rotation matrix* :math:`\\mathbf{R}`:

        .. math::
            \\mathbf{R} =
            \\begin{bmatrix}
            r_{11} & r_{12} & r_{13} \\\\
            r_{21} & r_{22} & r_{23} \\\\
            r_{31} & r_{32} & r_{33}
            \\end{bmatrix}

        The axis-angle representation of :math:`\\mathbf{R}` is obtained with:

        .. math::
            \\theta = \\arccos\\Big(\\frac{\\mathrm{tr}(\\mathbf{R})-1}{2}\\Big)

        for the **rotation angle**, and:

        .. math::
            \\mathbf{k} = \\frac{1}{2\\sin\\theta}
            \\begin{bmatrix}r_{32} - r_{23} \\\\ r_{13} - r_{31} \\\\ r_{21} - r_{12}\\end{bmatrix}

        for the **rotation vector**.

        .. note::
            The axis-angle representation is not unique since a rotation of
            :math:`-\\theta` about :math:`-\\mathbf{k}` is the same as a
            rotation of :math:`\\theta` about :math:`\\mathbf{k}`.

        Returns
        -------
        axis : numpy.ndarray
            Axis of rotation.
        angle : float
            Angle of rotation, in radians.

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0])
        >>> R.view()
        DCM([[ 0.92541658, -0.31879578, -0.20487413],
             [ 0.16317591,  0.82317294, -0.54383814],
             [ 0.34202014,  0.46984631,  0.81379768]])
        >>> R.to_axisangle()
        (array([ 0.81187135, -0.43801381,  0.38601658]), 0.6742208510527136)

        """
        angle = np.arccos((self.A.trace()-1)/2)
        axis = np.zeros(3)
        if angle!=0:
            S = np.array([self.A[2, 1]-self.A[1, 2], self.A[0, 2]-self.A[2, 0], self.A[1, 0]-self.A[0, 1]])
            axis = S/(2*np.sin(angle))
        return axis, angle

    def to_axang(self) -> Tuple[np.ndarray, float]:
        """
        Synonym of method :meth:`to_axisangle`.

        Returns
        -------
        axis : numpy.ndarray
            Axis of rotation.
        angle : float
            Angle of rotation, in radians.

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0])
        >>> R.view()
        DCM([[ 0.92541658, -0.31879578, -0.20487413],
             [ 0.16317591,  0.82317294, -0.54383814],
             [ 0.34202014,  0.46984631,  0.81379768]])
        >>> R.to_axang()
        (array([ 0.81187135, -0.43801381,  0.38601658]), 0.6742208510527136)

        """
        return self.to_axisangle()

    def from_axisangle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        DCM from axis-angle representation

        Use Rodrigue's formula to obtain the DCM from the axis-angle
        representation.

        .. math::
            \\mathbf{R} = \\mathbf{I}_3 - (\\sin\\theta)\\mathbf{K} + (1-\\cos\\theta)\\mathbf{K}^2

        where :math:`\\mathbf{R}` is the DCM, which rotates through an **angle**
        :math:`\\theta` counterclockwise about the **axis** :math:`\\mathbf{k}`,
        :math:`\\mathbf{I}_3` is the :math:`3\\times 3` identity matrix, and
        :math:`\\mathbf{K}` is the `skew-symmetric <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_
        matrix of :math:`\\mathbf{k}`.

        Parameters
        ----------
        axis : numpy.ndarray
            Axis of rotation.
        angle : float
            Angle of rotation, in radians.

        Returns
        -------
        R : numpy.ndarray
            3-by-3 direction cosine matrix

        Examples
        --------
        >>> R = DCM()
        >>> R.view()
        DCM([[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]])
        >>> R.from_axisangle([0.81187135, -0.43801381, 0.38601658], 0.6742208510527136)
        array([[ 0.92541658, -0.31879578, -0.20487413],
               [ 0.16317591,  0.82317294, -0.54383814],
               [ 0.34202014,  0.46984631,  0.81379768]])

        """
        _assert_numerical_iterable(axis, "Axis of rotation")
        if not isinstance(angle, (int, float)):
            raise ValueError(f"`angle` must be a float value. Got {type(angle)}")
        axis = axis / np.linalg.norm(axis)
        K = skew(axis)
        return np.identity(3) + np.sin(angle)*K + (1-np.cos(angle))*K@K

    def from_axang(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Synonym of method :meth:`from_axisangle`.

        Parameters
        ----------
        axis : numpy.ndarray
            Axis of rotation.
        angle : float
            Angle of rotation, in radians.

        Returns
        -------
        R : numpy.ndarray
            3-by-3 direction cosine matrix

        Examples
        --------
        >>> R = DCM()
        >>> R.view()
        DCM([[1., 0., 0.],
             [0., 1., 0.],
             [0., 0., 1.]])
        >>> R.from_axang([0.81187135, -0.43801381, 0.38601658], 0.6742208510527136)
        array([[ 0.92541658, -0.31879578, -0.20487413],
               [ 0.16317591,  0.82317294, -0.54383814],
               [ 0.34202014,  0.46984631,  0.81379768]])

        """
        return self.from_axisangle(axis, angle)

    @classmethod
    def from_quaternion(self, q: np.ndarray) -> np.ndarray:
        """
        DCM from given quaternion

        The quaternion :math:`\\mathbf{q}` has the form :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`,
        where :math:`\\mathbf{q}_v = (q_x, q_y, q_z)` is the vector part, and
        :math:`q_w` is the scalar part.

        The resulting matrix :math:`\\mathbf{R}` has the form:

        .. math::

            \\mathbf{R}(\\mathbf{q}) =
            \\begin{bmatrix}
            1 - 2(q_y^2 + q_z^2) & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\\\
            2(q_xq_y + q_wq_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_yq_z - q_wq_x) \\\\
            2(q_xq_z - q_wq_y) & 2(q_wq_x + q_yq_z) & 1 - 2(q_x^2 + q_y^2)
            \\end{bmatrix}

        The identity Quaternion :math:`\\mathbf{q} = \\begin{pmatrix}1 & 0 & 0 & 0\\end{pmatrix}`,
        produces a :math:`3 \\times 3` Identity matrix :math:`\\mathbf{I}_3`.

        Parameters
        ----------
        q : numpy.ndarray
            Quaternion

        Returns
        -------
        R : numpy.ndarray
            3-by-3 direction cosine matrix

        Examples
        --------
        >>> R = DCM()
        >>> R.from_quaternion([0.70710678, 0.0, 0.70710678, 0.0])
        array([[-2.22044605e-16,  0.00000000e+00,  1.00000000e+00],
               [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
               [-1.00000000e+00,  0.00000000e+00, -2.22044605e-16]])

        Non-normalized quaternions will be normalized and transformed too.

        >>> R.from_quaternion([1, 0.0, 1, 0.0])
        array([[ 2.22044605e-16,  0.00000000e+00,  1.00000000e+00],
               [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
               [-1.00000000e+00,  0.00000000e+00,  2.22044605e-16]])

        A list (or a Numpy array) with N quaternions will return an N-by-3-by-3
        array with the corresponding DCMs.

        .. code-block::

            >>> R.from_quaternion([[1, 0.0, 1, 0.0], [1.0, -1.0, 0.0, 1.0], [0.0, 0.0, -1.0, 1.0]])
            array([[[ 2.22044605e-16,  0.00000000e+00,  1.00000000e+00],
                    [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
                    [-1.00000000e+00,  0.00000000e+00,  2.22044605e-16]],

                   [[ 3.33333333e-01, -6.66666667e-01, -6.66666667e-01],
                    [ 6.66666667e-01, -3.33333333e-01,  6.66666667e-01],
                    [-6.66666667e-01, -6.66666667e-01,  3.33333333e-01]],

                   [[-1.00000000e+00, -0.00000000e+00,  0.00000000e+00],
                    [ 0.00000000e+00,  2.22044605e-16, -1.00000000e+00],
                    [ 0.00000000e+00, -1.00000000e+00,  2.22044605e-16]]])

        """
        if q is None:
            return np.identity(3)
        _assert_numerical_iterable(q, "Quaternion")
        q = np.copy(q)
        if q.shape[-1] != 4 or q.ndim > 2:
            raise ValueError(f"Quaternion must be of the form (4,) or (N, 4). Got {q.shape}")
        if q.ndim > 1:
            q /= np.linalg.norm(q, axis=1)[:, None]     # Normalize
            R = np.zeros((q.shape[0], 3, 3))
            R[:, 0, 0] = 1.0 - 2.0*(q[:, 2]**2 + q[:, 3]**2)
            R[:, 1, 0] = 2.0*(q[:, 1]*q[:, 2]+q[:, 0]*q[:, 3])
            R[:, 2, 0] = 2.0*(q[:, 1]*q[:, 3]-q[:, 0]*q[:, 2])
            R[:, 0, 1] = 2.0*(q[:, 1]*q[:, 2]-q[:, 0]*q[:, 3])
            R[:, 1, 1] = 1.0 - 2.0*(q[:, 1]**2 + q[:, 3]**2)
            R[:, 2, 1] = 2.0*(q[:, 0]*q[:, 1]+q[:, 2]*q[:, 3])
            R[:, 0, 2] = 2.0*(q[:, 1]*q[:, 3]+q[:, 0]*q[:, 2])
            R[:, 1, 2] = 2.0*(q[:, 2]*q[:, 3]-q[:, 0]*q[:, 1])
            R[:, 2, 2] = 1.0 - 2.0*(q[:, 1]**2 + q[:, 2]**2)
            return R
        q /= np.linalg.norm(q)
        return np.array([
            [1.0-2.0*(q[2]**2+q[3]**2), 2.0*(q[1]*q[2]-q[0]*q[3]), 2.0*(q[1]*q[3]+q[0]*q[2])],
            [2.0*(q[1]*q[2]+q[0]*q[3]), 1.0-2.0*(q[1]**2+q[3]**2), 2.0*(q[2]*q[3]-q[0]*q[1])],
            [2.0*(q[1]*q[3]-q[0]*q[2]), 2.0*(q[0]*q[1]+q[2]*q[3]), 1.0-2.0*(q[1]**2+q[2]**2)]])

    def from_q(self, q: np.ndarray) -> np.ndarray:
        """
        Synonym of method :meth:`from_quaternion`.

        Parameters
        ----------
        q : numpy.ndarray
            Quaternion

        Returns
        -------
        R : numpy.ndarray
            3-by-3 direction cosine matrix

        Examples
        --------
        >>> R = DCM()
        >>> R.from_q([0.70710678, 0.0, 0.70710678, 0.0])
        array([[-2.22044605e-16,  0.00000000e+00,  1.00000000e+00],
               [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
               [-1.00000000e+00,  0.00000000e+00, -2.22044605e-16]])

        Non-normalized quaternions will be normalized and transformed too.

        >>> R.from_q([1, 0.0, 1, 0.0])
        array([[ 2.22044605e-16,  0.00000000e+00,  1.00000000e+00],
               [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
               [-1.00000000e+00,  0.00000000e+00,  2.22044605e-16]])

        A list (or a Numpy array) with N quaternions will return an N-by-3-by-3
        array with the corresponding DCMs.

        .. code-block::

            >>> R.from_q([[1, 0.0, 1, 0.0], [1.0, -1.0, 0.0, 1.0], [0.0, 0.0, -1.0, 1.0]])
            array([[[ 2.22044605e-16,  0.00000000e+00,  1.00000000e+00],
                    [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
                    [-1.00000000e+00,  0.00000000e+00,  2.22044605e-16]],

                   [[ 3.33333333e-01, -6.66666667e-01, -6.66666667e-01],
                    [ 6.66666667e-01, -3.33333333e-01,  6.66666667e-01],
                    [-6.66666667e-01, -6.66666667e-01,  3.33333333e-01]],

                   [[-1.00000000e+00, -0.00000000e+00,  0.00000000e+00],
                    [ 0.00000000e+00,  2.22044605e-16, -1.00000000e+00],
                    [ 0.00000000e+00, -1.00000000e+00,  2.22044605e-16]]])
        """
        return self.from_quaternion(q)

    def to_quaternion(self, method: str='shepperd', **kw) -> np.ndarray:
        """
        Quaternion from Direction Cosine Matrix.

        There are five methods available to obtain a quaternion from a
        Direction Cosine Matrix:

        * ``'chiaverini'`` as described in :cite:p:`Chiaverini1999`.
        * ``'hughes'`` as described in :cite:p:`hughes1986spacecraft17`.
        * ``'itzhack'`` as described in :cite:p:`BarItzhack2000` using version ``3`` by
          default. Possible options are integers ``1``, ``2`` or ``3``.
        * ``'sarabandi'`` as described in :cite:p:`sarabandi2019` with a threshold equal
          to ``0.0`` by default. Possible threshold values are floats between
          ``-3.0`` and ``3.0``.
        * ``'shepperd'`` as described in :cite:p:`shepperd1978`.

        Parameters
        ----------
        method : str, default: ``'shepperd'``
            Method to use. Options are: ``'shepperd'``, ``'hughes'``,
            ``'itzhack'``, ``'sarabandi'``, and ``'chiaverini'``.

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0])
        >>> R.view()
        DCM([[ 0.92541658, -0.31879578, -0.20487413],
             [ 0.16317591,  0.82317294, -0.54383814],
             [ 0.34202014,  0.46984631,  0.81379768]])
        >>> R.to_quaternion()   # Uses method 'shepperd' by default
        array([ 0.15842345,  0.5510871 , -0.40599185,  0.71160076])
        >>> R.to_quaternion('shepperd')
        array([ 0.15842345,  0.5510871 , -0.40599185,  0.71160076])
        >>> R.to_quaternion('hughes')
        array([ 0.15842345,  0.5510871 , -0.40599185,  0.71160076])
        >>> R.to_quaternion('itzhack', version=2)
        array([ 0.15842345,  0.5510871 , -0.40599185,  0.71160076])
        >>> R.to_quaternion('sarabandi', threshold=0.5)
        array([0.15842345, 0.5510871 , 0.40599185, 0.71160076])

        """
        q = np.array([1., 0., 0., 0.])
        if method.lower() == 'hughes':
            q = hughes(self.A)
        elif method.lower() == 'chiaverini':
            q = chiaverini(self.A)
        elif method.lower() == 'shepperd':
            q = shepperd(self.A)
        elif method.lower() == 'itzhack':
            q = itzhack(self.A, version=kw.get('version', 3))
        elif method.lower() == 'sarabandi':
            q = sarabandi(self.A, eta=kw.get('threshold', 0.0))
        else:
            raise ValueError(f"Method {method} not available. Choose from 'chiaverini', 'hughes', 'itzhack', 'sarabandi', 'shepperd'")
        return q/np.linalg.norm(q)

    def to_q(self, method: str='shepperd', **kw) -> np.ndarray:
        """
        Synonym of method :meth:`to_quaternion`.

        Parameters
        ----------
        method : str, default: ``'shepperd'``
            Method to use. Options are: ``'chiaverini'``, ``'hughes'``,
            ``'itzhack'``, ``'sarabandi'``, and ``'shepperd'``.

        Examples
        --------
        >>> R = DCM(rpy=[10.0, -20.0, 30.0])
        >>> R.view()
        DCM([[-0.34241004, -0.67294223,  0.65567074],
             [-0.22200526, -0.62014526, -0.75241845],
             [ 0.91294525, -0.40319798,  0.06294725]])
        >>> R.to_q()   # Uses method 'shepperd' by default
        array([ 0.15842345,  0.5510871 , -0.40599185,  0.71160076])
        >>> R.to_q('shepperd')
        array([ 0.15842345,  0.5510871 , -0.40599185,  0.71160076])
        >>> R.to_q('hughes')
        array([ 0.15842345,  0.5510871 , -0.40599185,  0.71160076])
        >>> R.to_q('itzhack', version=2)
        array([ 0.15842345,  0.5510871 , -0.40599185,  0.71160076])
        >>> R.to_q('sarabandi', threshold=0.5)
        array([0.15842345, 0.5510871 , 0.40599185, 0.71160076])
        """
        return self.to_quaternion(method=method, **kw)

    def to_angles(self) -> np.ndarray:
        """
        Synonym of method :meth:`to_rpy`.

        Returns
        -------
        a : numpy.ndarray
            roll-pitch-yaw angles
        """
        return self.to_rpy()

    def to_rpy(self) -> np.ndarray:
        """
        Roll-Pitch-Yaw Angles from DCM

        A set of Roll-Pitch-Yaw angles may be written according to:

        .. math::
            \\mathbf{a} =
            \\begin{bmatrix}\\phi \\\\ \\theta \\\\ \\psi\\end{bmatrix} =
            \\begin{bmatrix}\\mathrm{arctan2}(r_{23}, r_{33}) \\\\ -\\arcsin(r_{13}) \\\\ \\mathrm{arctan2}(r_{12}, r_{11})\\end{bmatrix}

        Returns
        -------
        a : numpy.ndarray
            roll-pitch-yaw angles.
        """
        phi = np.arctan2(self.A[1, 2], self.A[2, 2])    # Roll Angle
        theta = -np.arcsin(self.A[0, 2])                # Pitch Angle
        psi = np.arctan2(self.A[0, 1], self.A[0, 0])    # Yaw Angle
        return np.array([phi, theta, psi])

    def ode(self, w: np.ndarray) -> np.ndarray:
        """
        Ordinary Differential Equation of the DCM.

        Parameters
        ----------
        w : numpy.ndarray
            Instantaneous angular velocity, in rad/s, about X-, Y- and Z-axis.

        Returns
        -------
        dR/dt : numpy.ndarray
            Derivative of DCM
        """
        return self.A@skew(w)

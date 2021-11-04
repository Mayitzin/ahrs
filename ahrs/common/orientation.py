# -*- coding: utf-8 -*-
"""
Routines for orientation estimation.

Further description will follow.

Notes
-----
- The functions involving quaternions are now better implemented, controlled
  and documented in the class Quaternion.

"""

from typing import Tuple, Union
import numpy as np
from .mathfuncs import cosd, sind
from .constants import *

def q_conj(q: np.ndarray) -> np.ndarray:
    """
    Conjugate of unit quaternion

    A unit quaternion, whose form is :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`,
    has a conjugate of the form :math:`\\mathbf{q}^* = (q_w, -q_x, -q_y, -q_z)`.

    Multiple quaternions in an N-by-4 array can also be conjugated.

    Parameters
    ----------
    q : numpy.ndarray
        Unit quaternion or 2D array of Quaternions.

    Returns
    -------
    q_conj : numpy.ndarray
        Conjugated quaternion or 2D array of conjugated Quaternions.

    Examples
    --------
    >>> from ahrs.common.orientation import q_conj
    >>> q = np.array([0.603297, 0.749259, 0.176548, 0.20850 ])
    >>> q_conj(q)
    array([0.603297, -0.749259, -0.176548, -0.20850 ])
    >>> Q = np.array([[0.039443, 0.307174, 0.915228, 0.257769],
        [0.085959, 0.708518, 0.039693, 0.699311],
        [0.555887, 0.489330, 0.590976, 0.319829],
        [0.578965, 0.202390, 0.280560, 0.738321],
        [0.848611, 0.442224, 0.112601, 0.267611]])
    >>> q_conj(Q)
    array([[ 0.039443, -0.307174, -0.915228, -0.257769],
           [ 0.085959, -0.708518, -0.039693, -0.699311],
           [ 0.555887, -0.489330, -0.590976, -0.319829],
           [ 0.578965, -0.202390, -0.280560, -0.738321],
           [ 0.848611, -0.442224, -0.112601, -0.267611]])

    References
    ----------
    .. [1] Dantam, N. (2014) Quaternion Computation. Institute for Robotics
           and Intelligent Machines. Georgia Tech.
           (http://www.neil.dantam.name/note/dantam-quaternion.pdf)
    .. [2] https://en.wikipedia.org/wiki/Quaternion#Conjugation,_the_norm,_and_reciprocal

    """
    if q.ndim>2 or q.shape[-1]!=4:
        raise ValueError("Quaternion must be of shape (4,) or (N, 4), but has shape {}".format(q.shape))
    return np.array([1., -1., -1., -1.])*np.array(q)

def q_random(size: int = 1) -> np.ndarray:
    """
    Generate random quaternions

    Parameters
    ----------
    size : int
        Number of Quaternions to generate. Default is 1 quaternion only.

    Returns
    -------
    q : numpy.ndarray
        M-by-4 array of generated random Quaternions, where M is the requested size.

    Examples
    --------
    >>> import ahrs
    >>> q = ahrs.common.orientation.q_random()
    array([0.65733485, 0.29442787, 0.55337745, 0.41832587])
    >>> q = ahrs.common.orientation.q_random(5)
    >>> q
    array([[-0.81543924, -0.06443342, -0.08727487, -0.56858621],
           [ 0.23124879,  0.55068024, -0.59577746, -0.53695855],
           [ 0.74998503, -0.38943692,  0.27506719,  0.45847506],
           [-0.43213176, -0.55350396, -0.54203589, -0.46161954],
           [-0.17662536,  0.55089287, -0.81357401,  0.05846234]])
    >>> np.linalg.norm(q, axis=1)   # Each quaternion is, naturally, normalized
    array([1., 1., 1., 1., 1.])

    """
    if size<1 or not isinstance(size, int):
        raise ValueError("size must be a positive non-zero integer value.")
    q = np.random.random((size, 4))-0.5
    q /= np.linalg.norm(q, axis=1)[:, np.newaxis]
    if size==1:
        return q[0]
    return q


def q_norm(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion [WQ1]_ :math:`\\mathbf{q}_u`, also known as a versor
    [WV1]_ :

    .. math::

        \\mathbf{q}_u = \\frac{1}{\\|\\mathbf{q}\\|} \\mathbf{q}

    where:

    .. math::

        \\|\\mathbf{q}_u\\| = 1.0

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion to normalize

    Returns
    -------
    q_u : numpy.ndarray
        Normalized Quaternion

    Examples
    --------
    >>> from ahrs.common.orientation import q_norm
    >>> q = np.random.random(4)
    >>> q
    array([0.94064704, 0.12645116, 0.80194097, 0.62633894])
    >>> q = q_norm(q)
    >>> q
    array([0.67600473, 0.0908753 , 0.57632232, 0.45012429])
    >>> np.linalg.norm(q)
    1.0

    References
    ----------
    .. [WQ1] https://en.wikipedia.org/wiki/Quaternion#Unit_quaternion
    .. [WV1] https://en.wikipedia.org/wiki/Versor

    """
    if q.ndim>2 or q.shape[-1]!=4:
        raise ValueError("Quaternion must be of shape (4,) or (N, 4), but has shape {}".format(q.shape))
    if q.ndim>1:
        return q/np.linalg.norm(q, axis=1)[:, np.newaxis]
    return q/np.linalg.norm(q)

def q_prod(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Product of two unit quaternions.

    Given two unit quaternions :math:`\\mathbf{p}=(p_w, \\mathbf{p}_v)` and
    :math:`\\mathbf{q} = (q_w, \\mathbf{q}_v)`, their product is defined [ND]_ [MWQW]_
    as:

    .. math::

        \\begin{eqnarray}
        \\mathbf{pq} & = & \\big( (q_w p_w - \\mathbf{q}_v \\cdot \\mathbf{p}_v) \\; ,
        \\; \\mathbf{q}_v \\times \\mathbf{p}_v + q_w \\mathbf{p}_v + p_w \\mathbf{q}_v \\big) \\\\
        & = &
        \\begin{bmatrix}
        p_w & -\\mathbf{p}_v^T \\\\ \\mathbf{p}_v & p_w \\mathbf{I}_3 + \\lfloor \\mathbf{p}_v \\rfloor
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
        p_x q_w + p_w q_x - p_z q_y + p_y q_z \\\\
        p_y q_w + p_z q_x + p_w q_y - p_x q_z \\\\
        p_z q_w - p_y q_x + p_x q_y + p_w q_z
        \\end{bmatrix}
        \\end{eqnarray}

    Parameters
    ----------
    p : numpy.ndarray
        First quaternion to multiply
    q : numpy.ndarray
        Second quaternion to multiply

    Returns
    -------
    pq : numpy.ndarray
        Product of both quaternions

    Examples
    --------
    >>> import numpy as np
    >>> from ahrs import quaternion
    >>> q = ahrs.common.orientation.q_random(2)
    >>> q[0]
    array([0.55747131, 0.12956903, 0.5736954 , 0.58592763])
    >>> q[1]
    array([0.49753507, 0.50806522, 0.52711628, 0.4652709 ])
    >>> quaternion.q_prod(q[0], q[1])
    array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])

    References
    ----------
    .. [ND] Dantam, N. (2014) Quaternion Computation. Institute for Robotics
            and Intelligent Machines. Georgia Tech.
            (http://www.neil.dantam.name/note/dantam-quaternion.pdf)
    .. [MWQM] Mathworks: Quaternion Multiplication.
           https://www.mathworks.com/help/aeroblks/quaternionmultiplication.html

    """
    pq = np.zeros(4)
    pq[0] = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    pq[1] = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    pq[2] = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
    pq[3] = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
    return pq

def q_mult_L(q: np.ndarray) -> np.ndarray:
    """
    Matrix form of a left-sided quaternion multiplication Q.

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion to multiply from the left side.

    Returns
    -------
    Q : numpy.ndarray
        Matrix form of the left side quaternion multiplication.

    """
    q /= np.linalg.norm(q)
    Q = np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1],  q[0], -q[3],  q[2]],
        [q[2],  q[3],  q[0], -q[1]],
        [q[3], -q[2],  q[1],  q[0]]])
    return Q

def q_mult_R(q: np.ndarray) -> np.ndarray:
    """
    Matrix form of a right-sided quaternion multiplication Q.

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion to multiply from the right side.

    Returns
    -------
    Q : numpy.ndarray
        Matrix form of the right side quaternion multiplication.

    """
    q /= np.linalg.norm(q)
    Q = np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1],  q[0],  q[3], -q[2]],
        [q[2], -q[3],  q[0],  q[1]],
        [q[3],  q[2], -q[1],  q[0]]])
    return Q

def q_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vector :math:`\\mathbf{v}` through quaternion :math:`\\mathbf{q}`.

    It should be equal to calling `q2R(q).T@v`

    Parameters
    ----------
    v : numpy.ndarray
        Vector to rotate in 3 dimensions.
    q : numpy.ndarray
        Quaternion to rotate through.

    Returns
    -------
    v' : numpy.ndarray
        Rotated vector `v` through quaternion `q`.

    """
    qw, qx, qy, qz = q
    return np.array([
        -2.0*v[0]*(qy**2 + qz**2 - 0.5) + 2.0*v[1]*(qw*qz + qx*qy)       - 2.0*v[2]*(qw*qy - qx*qz),
        -2.0*v[0]*(qw*qz - qx*qy)       - 2.0*v[1]*(qx**2 + qz**2 - 0.5) + 2.0*v[2]*(qw*qx + qy*qz),
         2.0*v[0]*(qw*qy + qx*qz)       - 2.0*v[1]*(qw*qx - qy*qz)       - 2.0*v[2]*(qx**2 + qy**2 - 0.5)])

def axang2quat(axis: np.ndarray, angle: Union[int, float], rad: bool = True) -> np.ndarray:
    """
    Quaternion from given Axis-Angle.

    Parameters
    ----------
    axis : numpy.ndarray
        Unit vector indicating the direction of an axis of rotation.
    angle : int or float
        Angle describing the magnitude of rotation about the axis.

    Returns
    -------
    q : numpy.ndarray
        Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> from ahrs.quaternion import axang2quat
    >>> q = axang2quat([1.0, 0.0, 0.0], np.pi/2.0)
    array([0.70710678 0.70710678 0.         0.        ])

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    .. [2] https://www.mathworks.com/help/robotics/ref/axang2quat.html

    """
    if axis is None:
        return [1.0, 0.0, 0.0, 0.0]
    if len(axis)!= 3:
        raise ValueError()
    axis /= np.linalg.norm(axis)
    qw = np.cos(angle/2.0) if rad else cosd(angle/2.0)
    s = np.sin(angle/2.0) if rad else sind(angle/2.0)
    q = np.array([qw] + list(s*axis))
    return q/np.linalg.norm(q)

def quat2axang(q: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Axis-Angle representation from Quaternion.

    Parameters
    ----------
    q : numpy.ndarray
        Unit quaternion

    Returns
    -------
    axis : numpy.ndarray
        Unit vector indicating the direction of an axis of rotation.
    angle : float
        Angle describing the magnitude of rotation about the axis.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Recovering_the_axis-angle_representation

    """
    if q is None:
        return [0.0, 0.0, 0.0], 1.0
    if len(q) != 4:
        return None
    # Normalize input quaternion
    q /= np.linalg.norm(q)
    axis = np.copy(q[1:])
    denom = np.linalg.norm(axis)
    angle = 2.0*np.arctan2(denom, q[0])
    axis = np.array([0.0, 0.0, 0.0]) if angle == 0.0 else axis/denom
    return axis, angle

def q_correct(q: np.ndarray) -> np.ndarray:
    """
    Correct quaternion flipping its sign

    If a quaternion flips its sign, it will be corrected and brought back to
    its original position.

    Parameters
    ----------
    q : numpy.ndarray
        N-by-4 array of quaternions, where N is the number of continuous
        quaternions.

    Returns
    -------
    new_q : numpy.ndarray
        Corrected array of quaternions.
    """
    if q.ndim<2 or q.shape[-1]!=4:
        raise ValueError("Input must be of shape (N, 4). Got {}".format(q.shape))
    q_diff = np.diff(q, axis=0)
    norms = np.linalg.norm(q_diff, axis=1)
    binaries = np.where(norms>1, 1, 0)
    nonzeros = np.nonzero(binaries)
    jumps = nonzeros[0]+1
    if len(jumps)%2:
        jumps = np.append(jumps, [len(q_diff)+1])
    jump_pairs = jumps.reshape((len(jumps)//2, 2))
    new_q = q.copy()
    for j in jump_pairs:
        new_q[j[0]:j[1]] *= -1.0
    return new_q

def q2R(q: np.ndarray) -> np.ndarray:
    """
    Direction Cosine Matrix from given quaternion.

    The given unit quaternion :math:`\\mathbf{q}` must have the form
    :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`, where :math:`\\mathbf{q}_v = (q_x, q_y, q_z)`
    is the vector part, and :math:`q_w` is the scalar part.

    The resulting DCM (a.k.a. rotation matrix) :math:`\\mathbf{R}` has the form:

    .. math::

        \\mathbf{R}(\\mathbf{q}) =
        \\begin{bmatrix}
        1 - 2(q_y^2 + q_z^2) & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\\\
        2(q_xq_y + q_wq_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_yq_z - q_wq_x) \\\\
        2(q_xq_z - q_wq_y) & 2(q_wq_x + q_yq_z) & 1 - 2(q_x^2 + q_y^2)
        \\end{bmatrix}

    The default value is the unit Quaternion :math:`\\mathbf{q} = (1, 0, 0, 0)`,
    which produces a :math:`3 \\times 3` Identity matrix :math:`\\mathbf{I}_3`.

    Parameters
    ----------
    q : numpy.ndarray
        Unit quaternion

    Returns
    -------
    R : numpy.ndarray
        3-by-3 Direction Cosine Matrix.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    .. [2] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix

    """
    if q is None:
        return np.identity(3)
    if q.shape[-1]!= 4:
        raise ValueError("Quaternion Array must be of the form (4,) or (N, 4)")
    if q.ndim>1:
        q /= np.linalg.norm(q, axis=1)[:, None]     # Normalize all quaternions
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

def q2euler(q: np.ndarray) -> np.ndarray:
    """
    Euler Angles from unit Quaternion.

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion

    Returns
    -------
    angles : numpy.ndarray
        Euler Angles around X-, Y- and Z-axis.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_Angles_Conversion

    """
    if sum(np.array([1., 0., 0., 0.])-q) == 0.0:
        return np.zeros(3)
    if len(q) != 4:
        return None
    R_00 = 2.0*q[0]**2 - 1.0 + 2.0*q[1]**2
    R_10 = 2.0*(q[1]*q[2] - q[0]*q[3])
    R_20 = 2.0*(q[1]*q[3] + q[0]*q[2])
    R_21 = 2.0*(q[2]*q[3] - q[0]*q[1])
    R_22 = 2.0*q[0]**2 - 1.0 + 2.0*q[3]**2
    phi = np.arctan2( R_21, R_22)
    theta = -np.arctan( R_20/np.sqrt(1.0-R_20**2))
    psi = np.arctan2( R_10, R_00)
    return np.array([phi, theta, psi])

def rotation(ax: Union[str, int] = None, ang: float = 0.0) -> np.ndarray:
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
    angle : float
        Angle, in degrees, to rotate around. Default is 0.

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
    # Handle input
    if ang==0.0:
        return I_3
    if ax is None:
        ax = "z"
    if isinstance(ax, int):
        if ax < 0:
            ax = 2      # Negative axes default to 2 (Z-axis)
        ax = valid_axes[ax] if ax < 3 else "z"
    try:
        ang = float(ang)
    except:
        return I_3
    # Return 3-by-3 Identity matrix if invalid input
    if ax not in valid_axes:
        return I_3
    # Compute rotation
    ca, sa = cosd(ang), sind(ang)
    if ax.lower()=="x":
        return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    if ax.lower()=="y":
        return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    if ax.lower()=="z":
        return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])

def rot_seq(axes: Union[list, str] = None, angles: Union[list, float] = None) -> np.ndarray:
    """
    Direciton Cosine Matrix from set of axes and angles.

    The rotation matrix :math:`\\mathbf{R}` is created from the given list of
    angles rotating around the given axes order.

    Parameters
    ----------
    axes : list of str
        List of rotation axes.
    angles : list of floats
        List of rotation angles.

    Returns
    -------
    R : numpy.ndarray
        3-by-3 Direction Cosine Matrix.

    Examples
    --------
    >>> import numpy as np
    >>> import random
    >>> from ahrs import quaternion
    >>> num_rotations = 5
    >>> axis_order = random.choices("XYZ", k=num_rotations)
    >>> axis_order
    ['Z', 'Z', 'X', 'Z', 'Y']
    >>> angles = np.random.uniform(low=-180.0, high=180.0, size=num_rotations)
    >>> angles
    array([-139.24498146,  99.8691407, -171.30712526, -60.57132043,
             17.4475838 ])
    >>> R = quaternion.rot_seq(axis_order, angles)
    >>> R   # R = R_z(-139.24) R_z(99.87) R_x(-171.31) R_z(-60.57) R_y(17.45)
    array([[ 0.85465231  0.3651317   0.36911822]
           [ 0.3025091  -0.92798938  0.21754072]
           [ 0.4219688  -0.07426006 -0.90356393]])

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
    .. [2] https://en.wikipedia.org/wiki/Euler_angles

    """
    accepted_axes = list('xyzXYZ')
    R = np.identity(3)
    if axes is None:
        axes = np.random.choice(accepted_axes, 3)
    if not isinstance(axes, list):
        axes = list(axes)
    num_rotations = len(axes)
    if num_rotations < 1:
        return R
    if angles is None:
        angles = np.random.uniform(low=-180.0, high=180.0, size=num_rotations)
    for x in angles:
        if not isinstance(x, (float, int)):
            raise TypeError(f"Angles must be float or int numbers. Got {type(x)}")
    if set(axes).issubset(set(accepted_axes)):
        # Perform the matrix multiplications
        for i in range(num_rotations-1, -1, -1):
            R = rotation(axes[i], angles[i])@R
    return R

def dcm2quat(R: np.ndarray) -> np.ndarray:
    """
    Quaternion from Direct Cosine Matrix.

    Parameters
    ----------
    R : numpy.ndarray
        Direction Cosine Matrix.

    Returns
    -------
    q : numpy.ndarray
        Unit Quaternion.

    References
    ----------
    .. [1] F. Landis Markley. Attitude Determination using two Vector
        Measurements.

    """
    if(R.shape[0] != R.shape[1]):
        raise ValueError('Input is not a square matrix')
    if(R.shape[0] != 3):
        raise ValueError('Input needs to be a 3x3 array or matrix')
    q = np.array([1., 0., 0., 0.])
    q[0] = 0.5*np.sqrt(1.0 + R.trace())
    q[1] = (R[1, 2] - R[2, 1]) / q[0]
    q[2] = (R[2, 0] - R[0, 2]) / q[0]
    q[3] = (R[0, 1] - R[1, 0]) / q[0]
    q[1:] /= 4.0
    return q / np.linalg.norm(q)

def rpy2q(angles: np.ndarray, in_deg: bool = False) -> np.ndarray:
    """
    Quaternion from roll-pitch-yaw angles
    
    Roll is the first rotation (about X-axis), pitch is the second rotation
    (about Y-axis), and yaw is the last rotation (about Z-axis.)

    Parameters
    ----------
    angles : numpy.ndarray
        roll-pitch-yaw angles.
    in_deg : bool, default: False
        Angles are given in degrees.

    Returns
    -------
    q : array
        Quaternion.

    """
    if angles.shape[-1] != 3:
        raise ValueError("Input angles must be an array with three elements.")
    if in_deg:
        angles *= DEG2RAD
    if angles.ndim<2:
        yaw, pitch, roll = angles
    else:
        yaw, pitch, roll = angles.T
    cr = np.cos(0.5*roll)
    sr = np.sin(0.5*roll)
    cp = np.cos(0.5*pitch)
    sp = np.sin(0.5*pitch)
    cy = np.cos(0.5*yaw)
    sy = np.sin(0.5*yaw)
    # To Quaternion
    q = np.array([
        cy*cp*cr + sy*sp*sr,
        cy*cp*sr - sy*sp*cr,
        sy*cp*sr + cy*sp*cr,
        sy*cp*cr - cy*sp*sr])
    q /= np.linalg.norm(q)
    return q

def cardan2q(angles: np.ndarray, in_deg: bool = False) -> np.ndarray:
    """synonym to function :func:`rpy2q`."""
    return rpy2q(angles, in_deg=in_deg)

def q2rpy(q: np.ndarray, in_deg: bool = False) -> np.ndarray:
    """
    Roll-pitch-yaw angles from quaternion.

    Roll is the first rotation (about X-axis), pitch is the second rotation
    (about Y-axis), and yaw is the last rotation (about Z-axis.)

    Parameters
    ----------
    q : numpy.ndarray
        Quaternion.
    in_deg : bool, default: False
        Return the angles in degrees.

    Returns
    -------
    angles : numpy.ndarray
        roll-pitch-yaw angles.
    """
    if q.shape[-1] != 4:
        return None
    roll = np.arctan2(2.0*(q[0]*q[1] + q[2]*q[3]), 1.0 - 2.0*(q[1]**2 + q[2]**2))
    pitch = np.arcsin(2.0*(q[0]*q[2] - q[3]*q[1]))
    yaw = np.arctan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]**2 + q[3]**2))
    angles = np.array([roll, pitch, yaw])
    if in_deg:
        return angles*RAD2DEG
    return angles

def q2cardan(q: np.ndarray, in_deg: bool = False) -> np.ndarray:
    """synonym to function :func:`q2rpy`."""
    return q2rpy(q, in_deg=in_deg)

def ecompass(a: np.ndarray, m: np.ndarray, frame: str = 'ENU', representation: str = 'rotmat') -> np.ndarray:
    """
    Orientation from accelerometer and magnetometer readings

    Parameters
    ----------
    a : numpy.ndarray
        Sample of tri-axial accelerometer, in m/s^2.
    m : numpy.ndarray
        Sample of tri-axial magnetometer, in uT.
    frame : str, default: ``'ENU'``
        Local tangent plane coordinate frame.
    representation : str, default: ``'rotmat'``
        Orientation representation.

    Returns
    -------
    np.ndarray
        Estimated orientation.

    Raises
    ------
    ValueError
        When wrong local tangent plane coordinates, or invalid representation,
        is given.
    """
    if frame.upper() not in ['ENU', 'NED']:
        raise ValueError("Wrong local tangent plane coordinate frame. Try 'ENU' or 'NED'")
    if representation.lower() not in ['rotmat', 'quaternion', 'rpy', 'axisangle']:
        raise ValueError("Wrong representation type. Try 'rotmat', 'quaternion', 'rpy', or 'axisangle'")
    a = np.copy(a)
    m = np.copy(m)
    if a.shape[-1] != 3 or m.shape[-1] != 3:
        raise ValueError("Input vectors must have exactly 3 elements.")
    m /= np.linalg.norm(m)
    Rz = a/np.linalg.norm(a)
    if frame.upper() == 'NED':
        Ry = np.cross(Rz, m)
        Rx = np.cross(Ry, Rz)
    else:
        Rx = np.cross(m, Rz)
        Ry = np.cross(Rz, Rx)
    Rx /= np.linalg.norm(Rx)
    Ry /= np.linalg.norm(Ry)
    R = np.c_[Rx, Ry, Rz].T
    if representation.lower() == 'quaternion':
        return chiaverini(R)
    if representation.lower() == 'rpy':
        phi = np.arctan2(R[1, 2], R[2, 2])    # Roll Angle
        theta = -np.arcsin(R[0, 2])           # Pitch Angle
        psi = np.arctan2(R[0, 1], R[0, 0])    # Yaw Angle
        return np.array([phi, theta, psi])
    if representation.lower() == 'axisangle':
        angle = np.arccos((R.trace()-1)/2)
        axis = np.zeros(3)
        if angle!=0:
            S = np.array([R[2, 1]-R[1, 2], R[0, 2]-R[2, 0], R[1, 0]-R[0, 1]])
            axis = S/(2*np.sin(angle))
        return (axis, angle)
    return R

def am2DCM(a: np.ndarray, m: np.ndarray, frame: str = 'ENU') -> np.ndarray:
    """
    Direction Cosine Matrix from acceleration and/or compass using TRIAD.

    Parameters
    ----------
    a : numpy.ndarray
        Array of single sample of 3 orthogonal accelerometers.
    m : numpy.ndarray
        Array of single sample of 3 orthogonal magnetometers.
    frame : str, default: ``'ENU'``
        Local Tangent Plane. Options are ``'ENU'`` or ``'NED'``. Defaults to
        ``'ENU'`` (East-North-Up) coordinates.

    Returns
    -------
    pose : numpy.ndarray
        Direction Cosine Matrix

    References
    ----------
    - Michel, T. et al. (2018) Attitude Estimation for Indoor Navigation and
      Augmented Reality with Smartphones.
      (http://tyrex.inria.fr/mobile/benchmarks-attitude/)
      (https://hal.inria.fr/hal-01650142v2/document)
    """
    if frame.upper() not in ['ENU', 'NED']:
        raise ValueError("Wrong coordinate frame. Try 'ENU' or 'NED'")
    a = np.array(a)
    m = np.array(m)
    H = np.cross(m, a)
    H /= np.linalg.norm(H)
    a /= np.linalg.norm(a)
    M = np.cross(a, H)
    if frame.upper()=='ENU':
        return np.array([[H[0], M[0], a[0]],
                         [H[1], M[1], a[1]],
                         [H[2], M[2], a[2]]])
    return np.array([[M[0], H[0], -a[0]],
                     [M[1], H[1], -a[1]],
                     [M[2], H[2], -a[2]]])

def am2q(a: np.ndarray, m: np.ndarray, frame: str = 'ENU') -> np.ndarray:
    """
    Quaternion from acceleration and/or compass using TRIAD method.

    Parameters
    ----------
    a : numpy.ndarray
        Array with sample of 3 orthogonal accelerometers.
    m : numpy.ndarray
        Array with sample of 3 orthogonal magnetometers.
    frame : str, default: 'ENU'
        Local Tangent Plane. Options are 'ENU' or 'NED'. Defaults to 'ENU'
        (East-North-Up) coordinates.

    Returns
    -------
    pose : numpy.ndarray
        Quaternion

    References
    ----------
    .. [1] Michel, T. et al. (2018) Attitude Estimation for Indoor
           Navigation and Augmented Reality with Smartphones.
           (http://tyrex.inria.fr/mobile/benchmarks-attitude/)
           (https://hal.inria.fr/hal-01650142v2/document)
    .. [2] Janota, A. Improving the Precision and Speed of Euler Angles
           Computation from Low-Cost Rotation Sensor Data.
           (https://www.mdpi.com/1424-8220/15/3/7016/pdf)

    """
    R = am2DCM(a, m, frame=frame)
    q = dcm2quat(R)
    return q

def acc2q(a: np.ndarray, return_euler: bool = False) -> np.ndarray:
    """
    Quaternion from given acceleration.

    Parameters
    ----------
    a : numpy.ndarray
        A sample of 3 orthogonal accelerometers.
    return_euler : bool, default: False
        Return pose as Euler angles

    Returns
    -------
    pose : numpy.ndarray
        Quaternion or Euler Angles.

    References
    ----------
    .. [1] Michel, T. et al. (2018) Attitude Estimation for Indoor
           Navigation and Augmented Reality with Smartphones.
           (http://tyrex.inria.fr/mobile/benchmarks-attitude/)
           (https://hal.inria.fr/hal-01650142v2/document)
    .. [2] Zhang, H. et al (2015) Axis-Exchanged Compensation and Gait
           Parameters Analysis for High Accuracy Indoor Pedestrian Dead Reckoning.
           (https://www.researchgate.net/publication/282535868_Axis-Exchanged_Compensation_and_Gait_Parameters_Analysis_for_High_Accuracy_Indoor_Pedestrian_Dead_Reckoning)
    .. [3] Yun, X. et al. (2008) A Simplified Quaternion-Based Algorithm for
           Orientation Estimation From Earth Gravity and Magnetic Field Measurements.
           (https://apps.dtic.mil/dtic/tr/fulltext/u2/a601113.pdf)
    .. [4] Jung, D. et al. Inertial Attitude and Position Reference System
           Development for a Small UAV.
           (https://pdfs.semanticscholar.org/fb62/903d8e6c051c8f4780c79b6b18fbd02a0ff9.pdf)
    .. [5] Bleything, T. How to convert Magnetometer data into Compass Heading.
           (https://blog.digilentinc.com/how-to-convert-magnetometer-data-into-compass-heading/)
    .. [6] RT IMU Library. (https://github.com/RTIMULib/RTIMULib2/blob/master/RTIMULib/RTFusion.cpp)
    .. [7] Janota, A. Improving the Precision and Speed of Euler Angles
           Computation from Low-Cost Rotation Sensor Data. (https://www.mdpi.com/1424-8220/15/3/7016/pdf)
    .. [8] Trimpe, S. Accelerometer -based Tilt Estimation of a Rigid Body
           with only Rotational Degrees of Freedom. 2010.
           (http://www.idsc.ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Research_DAndrea/Balancing%20Cube/ICRA10_1597_web.pdf)

    """
    q = np.array([1.0, 0.0, 0.0, 0.0])
    ex, ey, ez = 0.0, 0.0, 0.0
    if np.linalg.norm(a)>0 and len(a)==3:
        ax, ay, az = a
        # Normalize accelerometer measurements
        a_norm = np.linalg.norm(a)
        ax /= a_norm
        ay /= a_norm
        az /= a_norm
        # Euler Angles from Gravity vector
        ex = np.arctan2(ay, az)
        ey = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        ez = 0.0
        if return_euler:
            return np.array([ex, ey, ez])*RAD2DEG
        # Euler to Quaternion
        cx2 = np.cos(ex/2.0)
        sx2 = np.sin(ex/2.0)
        cy2 = np.cos(ey/2.0)
        sy2 = np.sin(ey/2.0)
        q = np.array([cx2*cy2, sx2*cy2, cx2*sy2, -sx2*sy2])
        q /= np.linalg.norm(q)
    return q

def am2angles(a: np.ndarray, m: np.ndarray, in_deg: bool = False) -> np.ndarray:
    """
    Roll-pitch-yaw angles from acceleration and compass.

    Parameters
    ----------
    a : numpy.ndarray
        N-by-3 array with N samples of 3 orthogonal accelerometers.
    m : numpy.ndarray
        N-by-3 array with N samples of 3 orthogonal magnetometers.
    in_deg : bool, default: False
        Return the angles in degrees.

    Returns
    -------
    pose : numpy.ndarray
        Roll-pitch-yaw angles

    References
    ----------
    .. [DT0058] A. Vitali. Computing tilt measurement and tilt-compensated
      e-compass. ST Technical Document DT0058. October 2018.
      (https://www.st.com/resource/en/design_tip/dm00269987.pdf)
    """
    if a.ndim<2:
        a = np.atleast_2d(a)
    if m.ndim<2:
        m = np.atleast_2d(m)
    # Normalization of 2D arrays
    a /= np.linalg.norm(a, axis=1)[:, None]
    m /= np.linalg.norm(m, axis=1)[:, None]
    angles = np.zeros((len(a), 3))   # Allocation of angles array
    # Estimate tilt angles
    angles[:, 0] = np.arctan2(a[:, 1], a[:, 2])
    angles[:, 1] = np.arctan2(-a[:, 0], np.sqrt(a[:, 1]**2 + a[:, 2]**2))
    # Estimate heading angle
    my2 = m[:, 2]*np.sin(angles[:, 0]) - m[:, 1]*np.cos(angles[:, 0])
    mz2 = m[:, 1]*np.sin(angles[:, 0]) + m[:, 2]*np.cos(angles[:, 0])
    mx3 = m[:, 0]*np.cos(angles[:, 1]) + mz2*np.sin(angles[:, 1])
    angles[:, 2] = np.arctan2(my2, mx3)
    # Return in degrees or in radians
    if in_deg:
        return angles*RAD2DEG
    return angles

def slerp(q0: np.ndarray, q1: np.ndarray, t_array: np.ndarray, threshold: float = 0.9995) -> np.ndarray:
    """
    Spherical Linear Interpolation between quaternions.

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

    Extra Parameters
    ----------------
    threshold : float
        Threshold to closeness of interpolation.

    Returns
    -------
    q : numpy.ndarray
        New quaternion representing the interpolated rotation.

    References
    ----------
    .. [Wiki_SLERP] https://en.wikipedia.org/wiki/Slerp

    """
    qdot = q0@q1
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

def logR(R: np.ndarray) -> np.ndarray:
    S = 0.5*(R-R.T)
    y = np.array([S[2, 1], -S[2, 0], S[1, 0]])
    if np.allclose(np.zeros(3), y):
        return np.zeros(3)
    y_norm = np.linalg.norm(y)
    return np.arcsin(y_norm)*y/y_norm

def chiaverini(dcm: np.ndarray) -> np.ndarray:
    """
    Quaternion from a Direction Cosine Matrix with Chiaverini's algebraic method [Chiaverini]_.

    Parameters
    ----------
    dcm : numpy.ndarray
        3-by-3 Direction Cosine Matrix.

    Returns
    -------
    q : numpy.ndarray
        Quaternion.
    """
    n = 0.5*np.sqrt(dcm.trace() + 1.0)
    e = np.array([0.5*np.sign(dcm[2, 1]-dcm[1, 2])*np.sqrt(dcm[0, 0]-dcm[1, 1]-dcm[2, 2]+1.0),
                  0.5*np.sign(dcm[0, 2]-dcm[2, 0])*np.sqrt(dcm[1, 1]-dcm[2, 2]-dcm[0, 0]+1.0),
                  0.5*np.sign(dcm[1, 0]-dcm[0, 1])*np.sqrt(dcm[2, 2]-dcm[0, 0]-dcm[1, 1]+1.0)])
    return np.array([n, *e])

def hughes(dcm: np.ndarray) -> np.ndarray:
    """
    Quaternion from a Direction Cosine Matrix with Hughe's method [Hughes]_.

    Parameters
    ----------
    dcm : numpy.ndarray
        3-by-3 Direction Cosine Matrix.

    Returns
    -------
    q : numpy.ndarray
        Quaternion.
    """
    tr = dcm.trace()
    if np.isclose(tr, 3.0):
        return np.array([1., 0., 0., 0.])   # No rotation. DCM is identity.
    n = 0.5*np.sqrt(1.0 + tr)
    if np.isclose(n, 0):    # trace = -1: q_w = 0 (Pure Quaternion)
        e = np.sqrt((1.0+np.diag(dcm))/2.0)
    else:
        e = 0.25*np.array([dcm[1, 2]-dcm[2, 1], dcm[2, 0]-dcm[0, 2], dcm[0, 1]-dcm[1, 0]])/n
    return np.array([n, *e])

def sarabandi(dcm: np.ndarray, eta: float = 0.0) -> np.ndarray:
    """
    Quaternion from a Direction Cosine Matrix with Sarabandi's method [Sarabandi]_.

    Parameters
    ----------
    dcm : numpy.ndarray
        3-by-3 Direction Cosine Matrix.
    eta : float,d efault: 0.0
        Threshold.

    Returns
    -------
    q : numpy.ndarray
        Quaternion.
    """
    # Get elements of R
    r11, r12, r13 = dcm[0, 0], dcm[0, 1], dcm[0, 2]
    r21, r22, r23 = dcm[1, 0], dcm[1, 1], dcm[1, 2]
    r31, r32, r33 = dcm[2, 0], dcm[2, 1], dcm[2, 2]
    # Compute qw
    dw = r11+r22+r33
    if dw > eta:
        qw = 0.5*np.sqrt(1.0+dw)
    else:
        nom = (r32-r23)**2+(r13-r31)**2+(r21-r12)**2
        denom = 3.0-dw
        qw = 0.5*np.sqrt(nom/denom)
    # Compute qx
    dx = r11-r22-r33
    if dx > eta:
        qx = 0.5*np.sqrt(1.0+dx)
    else:
        nom = (r32-r23)**2+(r12+r21)**2+(r31+r13)**2
        denom = 3.0-dx
        qx = 0.5*np.sqrt(nom/denom)
    # Compute qy
    dy = -r11+r22-r33
    if dy > eta:
        qy = 0.5*np.sqrt(1.0+dy)
    else:
        nom = (r13-r31)**2+(r12+r21)**2+(r23+r32)**2
        denom = 3.0-dy
        qy = 0.5*np.sqrt(nom/denom)
    # Compute qz
    dz = -r11-r22+r33
    if dz > eta:
        qz = 0.5*np.sqrt(1.0+dz)
    else:
        nom = (r21-r12)**2+(r31+r13)**2+(r23+r32)**2
        denom = 3.0-dz
        qz = 0.5*np.sqrt(nom/denom)
    return np.array([qw, qx, qy, qz])

def itzhack(dcm: np.ndarray, version: int = 3) -> np.ndarray:
    """
    Quaternion from a Direction Cosine Matrix with Bar-Itzhack's method [Itzhack]_.

    Versions 1 and 2 are used with orthogonal matrices (which all rotation
    matrices should be.)

    Parameters
    ----------
    dcm : numpy.ndarray
        3-by-3 Direction Cosine Matrix.
    version : int, default: 3
        Version used to compute the Quaternion. Options are 1, 2 or 3.

    Returns
    -------
    q : numpy.ndarray
        Quaternion.
    """
    is_orthogonal = np.isclose(np.linalg.det(dcm), 1.0) and np.allclose(dcm@dcm.T, np.eye(3))
    if is_orthogonal:
        if version == 1:
            K2 = np.array([
                [dcm[0, 0]-dcm[1, 1], dcm[1, 0]+dcm[0, 1], dcm[2, 0], -dcm[2, 1]],
                [dcm[1, 0]+dcm[0, 1], dcm[1, 1]-dcm[0, 0], dcm[2, 1], dcm[2, 0]],
                [dcm[2, 0], dcm[2, 1], -dcm[0, 0]-dcm[1, 1], dcm[0, 1]-dcm[1, 0]],
                [-dcm[2, 1], dcm[2, 0], dcm[0, 1]-dcm[1, 0], dcm[0, 0]+dcm[1, 1]]])/2.0
            eigval, eigvec = np.linalg.eig(K2)
            q = eigvec[:, np.where(np.isclose(eigval, 1.0))[0]].flatten().real
            return np.roll(q, 1)
        if version == 2:
            K3 = np.array([
                [dcm[0, 0]-dcm[1, 1]-dcm[2, 2], dcm[1, 0]+dcm[0, 1], dcm[2, 0]+dcm[0, 2], dcm[1, 2]-dcm[2, 1]],
                [dcm[1, 0]+dcm[0, 1], dcm[1, 1]-dcm[0, 0]-dcm[2, 2], dcm[2, 1]+dcm[1, 2], dcm[2, 0]-dcm[0, 2]],
                [dcm[2, 0]+dcm[0, 2], dcm[2, 1]+dcm[1, 2], dcm[2, 2]-dcm[0, 0]-dcm[1, 1], dcm[0, 1]-dcm[1, 0]],
                [dcm[1, 2]-dcm[2, 1], dcm[2, 0]-dcm[0, 2], dcm[0, 1]-dcm[1, 0], dcm[0, 0]+dcm[1, 1]+dcm[2, 2]]])/3.0
            eigval, eigvec = np.linalg.eig(K3)
            q = eigvec[:, np.where(np.isclose(eigval, 1.0))[0]].flatten().real
            return np.roll(q, 1)
    # Non-orthogonal DCM. Use version 3
    K3 = np.array([
        [dcm[0, 0]-dcm[1, 1]-dcm[2, 2], dcm[1, 0]+dcm[0, 1], dcm[2, 0]+dcm[0, 2], dcm[1, 2]-dcm[2, 1]],
        [dcm[1, 0]+dcm[0, 1], dcm[1, 1]-dcm[0, 0]-dcm[2, 2], dcm[2, 1]+dcm[1, 2], dcm[2, 0]-dcm[0, 2]],
        [dcm[2, 0]+dcm[0, 2], dcm[2, 1]+dcm[1, 2], dcm[2, 2]-dcm[0, 0]-dcm[1, 1], dcm[0, 1]-dcm[1, 0]],
        [dcm[1, 2]-dcm[2, 1], dcm[2, 0]-dcm[0, 2], dcm[0, 1]-dcm[1, 0], dcm[0, 0]+dcm[1, 1]+dcm[2, 2]]])/3.0
    eigval, eigvec = np.linalg.eig(K3)
    q = eigvec[:, eigval.argmax()]
    return np.roll(q, 1)

def shepperd(dcm: np.ndarray) -> np.ndarray:
    """
    Quaternion from a Direction Cosine Matrix with Shepperd's method [Shepperd]_.

    Parameters
    ----------
    dcm : numpy.ndarray
        3-by-3 Direction Cosine Matrix.

    Returns
    -------
    q : numpy.ndarray
        Quaternion.
    """
    d = np.diag(dcm)
    b = np.array([dcm.trace(), *d])
    i = b.argmax()
    if i==0:
        q = np.array([1.0+sum(d), dcm[1, 2]-dcm[2, 1], dcm[2, 0]-dcm[0, 2], dcm[0, 1]-dcm[1, 0]])
    elif i==1:
        q = np.array([dcm[1, 2]-dcm[2, 1], 1.0+d[0]-d[1]-d[2], dcm[1, 0]+dcm[0, 1], dcm[2, 0]+dcm[0, 2]])
    elif i==2:
        q = np.array([dcm[2, 0]-dcm[0, 2], dcm[1, 0]+dcm[0, 1], 1.0-d[0]+d[1]-d[2], dcm[2, 1]+dcm[1, 2]])
    else:
        q = np.array([dcm[0, 1]-dcm[1, 0], dcm[2, 0]+dcm[0, 2], dcm[2, 1]+dcm[1, 2], 1.0-d[0]-d[1]+d[2]])
    q /= 2.0*np.sqrt(q[i])
    return q


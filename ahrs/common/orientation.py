# -*- coding: utf-8 -*-
"""
Routines for orientation estimation.

Some functions involving quaternionds and DCM will be eventually removed, as
better implementations are available in their corresponding classes.
"""

from typing import Tuple, Union
import numpy as np
from .mathfuncs import cosd, sind
from .constants import RAD2DEG, DEG2RAD

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

    """
    q = np.copy(q)
    if q.ndim > 2 or q.shape[-1] != 4:
        raise ValueError(f"Quaternion must be of shape (4,) or (N, 4), but has shape {q.shape}")
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
    if size < 1 or not isinstance(size, int):
        raise ValueError("size must be a positive non-zero integer value.")
    q = np.random.random((size, 4))-0.5
    q /= np.linalg.norm(q, axis=1)[:, np.newaxis]
    if size == 1:
        return q[0]
    return q

def q_norm(q: np.ndarray) -> np.ndarray:
    """
    Normalize quaternion :cite:p:`Wiki_Quaternion` :math:`\\mathbf{q}_u`, also
    known as a versor :cite:p:`Wiki_Versor`:

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

    """
    if q.ndim > 2 or q.shape[-1] != 4:
        raise ValueError(f"Quaternion must be of shape (4,) or (N, 4), but has shape {q.shape}")
    if q.ndim > 1:
        return q/np.linalg.norm(q, axis=1)[:, np.newaxis]
    return q/np.linalg.norm(q)

def q_prod(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Product of two unit quaternions.

    Given two unit quaternions :math:`\\mathbf{p}=(p_w, \\mathbf{p}_v)` and
    :math:`\\mathbf{q} = (q_w, \\mathbf{q}_v)`, their product is defined
    :cite:p:`dantam2014` :cite:p:`MathWorks_QuaternionMultiplication`.
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
        return np.array([1.0, 0.0, 0.0, 0.0])
    if len(axis) != 3:
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
        return np.array([0.0, 0.0, 0.0]), 1.0
    if len(q) != 4:
        raise ValueError(f"The quaternion must be a 4-element array, not {len(q)}-element array.")
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
    if q.ndim < 2 or q.shape[-1] != 4:
        raise ValueError(f"Input must be of shape (N, 4). Got {q.shape}")
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

def q2R(q: np.ndarray, version: int = 1) -> np.ndarray:
    """
    Direction Cosine Matrix from given quaternion.

    The given unit quaternion :math:`\\mathbf{q}` must have the form
    :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`, where :math:`\\mathbf{q}_v = (q_x, q_y, q_z)`
    is the vector part, and :math:`q_w` is the scalar part.

    Two versions of the DCM can be obtained by setting the ``'version'``
    parameter. The default value is 1, which yields the DCM (a.k.a. rotation
    matrix) :math:`\\mathbf{R}` of the form:

    .. math::

        \\mathbf{R}(\\mathbf{q}) =
        \\begin{bmatrix}
        1 - 2(q_y^2 + q_z^2) & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\\\
        2(q_xq_y + q_wq_z) & 1 - 2(q_x^2 + q_z^2) & 2(q_yq_z - q_wq_x) \\\\
        2(q_xq_z - q_wq_y) & 2(q_wq_x + q_yq_z) & 1 - 2(q_x^2 + q_y^2)
        \\end{bmatrix}

    Version 2 yields the DCM of the form:

    .. math::

        \\mathbf{R}(\\mathbf{q}) =
        \\begin{bmatrix}
        q_w^2 + q_x^2 - q_y^2 - q_z^2 & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\\\
        2(q_xq_y + q_wq_z) & q_w^2 - q_x^2 + q_y^2 - q_z^2 & 2(q_yq_z - q_wq_x) \\\\
        2(q_xq_z - q_wq_y) & 2(q_wq_x + q_yq_z) & q_w^2 - q_x^2 - q_y^2 + q_z^2
        \\end{bmatrix}

    The default input is the unit Quaternion :math:`\\mathbf{q} = (1, 0, 0, 0)`,
    which produces a :math:`3 \\times 3` Identity matrix :math:`\\mathbf{I}_3`.

    .. warning::

        The input quaternion must be a unit quaternion, i.e.
        :math:`\\|\\mathbf{q}\\| = 1`, otherwise the resulting DCM will not be
        a rotation matrix. Thus, the given quaternion is normalized before the
        DCM is computed.

    Parameters
    ----------
    q : numpy.ndarray
        Unit quaternion
    version : int
        Version of the DCM. Default is 1.

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
    if q.shape[-1] != 4:
        raise ValueError("Quaternion Array must be of the form (4,) or (N, 4)")
    if version not in [1, 2]:
        raise ValueError("Version must be an int equal to 1 or 2.")
    if q.ndim > 1:
        # Convert multiple quaternions
        q /= np.linalg.norm(q, axis=1)[:, None]     # Normalize all quaternions
        R = np.zeros((q.shape[0], 3, 3))
        if version == 1:
            R[:, 0, 0] = 1.0 - 2.0*(q[:, 2]**2 + q[:, 3]**2)
            R[:, 1, 1] = 1.0 - 2.0*(q[:, 1]**2 + q[:, 3]**2)
            R[:, 2, 2] = 1.0 - 2.0*(q[:, 1]**2 + q[:, 2]**2)
        else:
            R[:, 0, 0] = q[:, 0]**2 + q[:, 1]**2 - q[:, 2]**2 - q[:, 3]**2
            R[:, 1, 1] = q[:, 0]**2 - q[:, 1]**2 + q[:, 2]**2 - q[:, 3]**2
            R[:, 2, 2] = q[:, 0]**2 - q[:, 1]**2 - q[:, 2]**2 + q[:, 3]**2
        R[:, 1, 0] = 2.0*(q[:, 1]*q[:, 2]+q[:, 0]*q[:, 3])
        R[:, 2, 0] = 2.0*(q[:, 1]*q[:, 3]-q[:, 0]*q[:, 2])
        R[:, 0, 1] = 2.0*(q[:, 1]*q[:, 2]-q[:, 0]*q[:, 3])
        R[:, 2, 1] = 2.0*(q[:, 0]*q[:, 1]+q[:, 2]*q[:, 3])
        R[:, 0, 2] = 2.0*(q[:, 1]*q[:, 3]+q[:, 0]*q[:, 2])
        R[:, 1, 2] = 2.0*(q[:, 2]*q[:, 3]-q[:, 0]*q[:, 1])
        return R
    # Convert single quaternion
    q /= np.linalg.norm(q)
    if version == 1:
        return np.array([
            [1.0-2.0*(q[2]**2+q[3]**2), 2.0*(q[1]*q[2]-q[0]*q[3]), 2.0*(q[1]*q[3]+q[0]*q[2])],
            [2.0*(q[1]*q[2]+q[0]*q[3]), 1.0-2.0*(q[1]**2+q[3]**2), 2.0*(q[2]*q[3]-q[0]*q[1])],
            [2.0*(q[1]*q[3]-q[0]*q[2]), 2.0*(q[0]*q[1]+q[2]*q[3]), 1.0-2.0*(q[1]**2+q[2]**2)]])
    return np.array([
        [q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2.0*(q[1]*q[2]-q[0]*q[3]), 2.0*(q[1]*q[3]+q[0]*q[2])],
        [2.0*(q[1]*q[2]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2.0*(q[2]*q[3]-q[0]*q[1])],
        [2.0*(q[1]*q[3]-q[0]*q[2]), 2.0*(q[0]*q[1]+q[2]*q[3]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]])

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

def dcm2quat(R: np.ndarray) -> np.ndarray:
    """
    Quaternion from Direction Cosine Matrix.

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
    if R.shape[0] != R.shape[1]:
        raise ValueError('Input is not a square matrix')
    if R.shape[0] != 3:
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
    if angles.ndim < 2:
        roll, pitch, yaw = angles
    else:
        roll, pitch, yaw = angles.T
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
    """Synonym to function :func:`rpy2q`."""
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
    """Synonym to function :func:`q2rpy`."""
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
        Orientation representation. Options are: ``'rotmat'``, ``'quaternion'``,
        ``'rpy'``, ``'axisangle'``.

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
    for item in [a, m]:
        if not isinstance(item, (np.ndarray, list, tuple)):
            raise TypeError("Both inputs a and m must be arrays.")
    a = np.copy(a)
    m = np.copy(m)
    if a.shape != m.shape:
        raise ValueError("Both vectors must have the same shape.")
    if len(a) != 3:
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
        if angle != 0:
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
    if frame.upper() == 'ENU':
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
    if np.linalg.norm(a) > 0 and len(a) == 3:
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
    if a.ndim < 2:
        a = np.atleast_2d(a)
    if m.ndim < 2:
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

    Based on the method detailed in :cite:p:`Wiki_SLERP`.

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

def chiaverini(dcm: np.ndarray) -> np.ndarray:
    """
    Quaternion from a Direction Cosine Matrix with Chiaverini's algebraic
    method :cite:p:`Chiaverini1999`.

    Defining the unit quaternion as:

    .. math::
        \\mathbf{q} = \\begin{bmatrix} q_w & \\mathbf{q}_v \\end{bmatrix}

    where

    .. math::
        \\begin{array}{rcl}
        q_w &=& \\cos \\big(\\frac{\\theta}{2}\\big) \\\\
        \\mathbf{q}_v &=& \\sin \\big(\\frac{\\theta}{2}\\big) \\mathbf{r}
        \\end{array}

    with :math:`q_w \\geq 0` for :math:`\\theta \\in [-\\pi, \\pi]`; :math:`q_w`
    is the scalar part, while :math:`\\mathbf{q}_v` is the vector part, and
    :math:`\\theta` is the rotation about the axis :math:`\\mathbf{r}`.

    We know the rotation matrix of a given unit quaternion is:

    .. math::
        \\mathbf{R}(\\mathbf{q}) = (q_w^2 - \\mathbf{q}_v^T\\mathbf{q}_v)\\mathbf{I}_3 + 2\\mathbf{q}_v\\mathbf{q}_v^T + 2 q_w\\lfloor\\mathbf{q}_v\\rfloor_\\times

    where :math:`\\mathbf{I}_3` is the :math:`3\\times 3` identity matrix, and
    :math:`\\lfloor\\mathbf{q}_v\\rfloor_\\times` is the `skew-symmetric matrix
    <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_ of the vector part.
    Solving the equation above for the scalar and vector parts we get the
    unitary quaternion:

    .. math::
        \\begin{array}{rcl}
        q_w &=& \\frac{1}{2} \\sqrt{R_{00} + R_{11} + R_{22} + 1} \\\\
        \\mathbf{q}_v &=& \\begin{bmatrix}
            \\frac{1}{2} \\mathrm{sgn}(R_{21} - R_{12}) \\sqrt{R_{00} - R_{11} - R_{22} + 1} \\\\
            \\frac{1}{2} \\mathrm{sgn}(R_{02} - R_{20}) \\sqrt{R_{11} - R_{22} - R_{00} + 1} \\\\
            \\frac{1}{2} \\mathrm{sgn}(R_{10} - R_{01}) \\sqrt{R_{22} - R_{00} - R_{11} + 1}
        \\end{bmatrix}
        \\end{array}

    where :math:`\\mathrm{sgn}` is the sign function:

    .. math::
        \\mathrm{sgn}(x) =
        \\left\\{
        \\begin{array}{ll}
            \\mathrm{-1} & \\: x < 0 \\\\
            \\mathrm{1} & \\: \\mathrm{otherwise}
        \\end{array}
        \\right.

    Parameters
    ----------
    dcm : numpy.ndarray
        3-by-3 or N-by-3-by-3 Direction Cosine Matrix (or Matrices).

    Returns
    -------
    q : numpy.ndarray
        4-dimensional or N-by-4 Quaternion array.
    """
    dcm = np.copy(dcm)
    if dcm.ndim not in [2, 3]:
        raise ValueError('dcm must be a 2- or 3-dimensional array.')
    if dcm.shape[-2:] != (3, 3):
        raise ValueError(f"dcm must be an array of shape 3-by-3 or N-by-3-by-3. Got {dcm.shape}")
    if dcm.ndim < 3:
        q = np.zeros(4)
        q[0] = 0.5*np.sqrt(np.clip(dcm.trace(), -1.0, 3.0) + 1.0)
        q[1] = 0.5*np.sign(dcm[2, 1]-dcm[1, 2])*np.sqrt(np.clip(dcm[0, 0]-dcm[1, 1]-dcm[2, 2], -1.0, 3.0)+1.0)
        q[2] = 0.5*np.sign(dcm[0, 2]-dcm[2, 0])*np.sqrt(np.clip(dcm[1, 1]-dcm[2, 2]-dcm[0, 0], -1.0, 3.0)+1.0)
        q[3] = 0.5*np.sign(dcm[1, 0]-dcm[0, 1])*np.sqrt(np.clip(dcm[2, 2]-dcm[0, 0]-dcm[1, 1], -1.0, 3.0)+1.0)
        if not any(q):
            q[0] = 1.0
        q /= np.linalg.norm(q)
        return q
    Q = np.zeros((dcm.shape[0], 4))
    Q[:, 0] = 0.5*np.sqrt(np.clip(dcm.trace(axis1=1, axis2=2), -1.0, 3.0) + 1.0)
    Q[:, 1] = 0.5*np.sign(dcm[:, 2, 1] - dcm[:, 1, 2])*np.sqrt(np.clip(dcm[:, 0, 0]-dcm[:, 1, 1]-dcm[:, 2, 2], -1.0, 3.0) + 1.0)
    Q[:, 2] = 0.5*np.sign(dcm[:, 0, 2] - dcm[:, 2, 0])*np.sqrt(np.clip(dcm[:, 1, 1]-dcm[:, 2, 2]-dcm[:, 0, 0], -1.0, 3.0) + 1.0)
    Q[:, 3] = 0.5*np.sign(dcm[:, 1, 0] - dcm[:, 0, 1])*np.sqrt(np.clip(dcm[:, 2, 2]-dcm[:, 0, 0]-dcm[:, 1, 1], -1.0, 3.0) + 1.0)
    Q /= np.linalg.norm(Q, axis=1)[:, None]
    return Q

def hughes(C: np.ndarray) -> np.ndarray:
    """
    Quaternion from a Direction Cosine Matrix with trigonometric Hughes' method
    :cite:p:`hughes1986spacecraft17`.

    Defining the quaternion (reluctantly called "Euler Parameters" in Hughes'
    book) as:

    .. math::
        \\mathbf{q} = \\begin{bmatrix} \\eta & \\boldsymbol{\\epsilon} \\end{bmatrix}

    where

    .. math::
        \\begin{array}{rcl}
        \\eta &=& \\cos \\big(\\frac{\\phi}{2}\\big) \\\\
        \\boldsymbol{\\epsilon} &=& \\sin \\big(\\frac{\\phi}{2}\\big) \\mathbf{a}
        \\end{array}

    The quaternion is subject to the constraint
    :math:`\\boldsymbol{\\epsilon}^T\\boldsymbol{\\epsilon}+\\eta^2=1`, and
    :math:`\\phi` is the rotation about the unitary axis :math:`\\mathbf{a}`.

    The rotation matrix associated to the quaternion is:

    .. math::
        \\mathbf{C} = (\\eta ^2 - \\boldsymbol{\\epsilon}^T\\boldsymbol{\\epsilon})\\mathbf{I} + 2\\boldsymbol{\\epsilon}\\boldsymbol{\\epsilon}^T - 2 \\eta\\boldsymbol{\\epsilon}^\\times

    where :math:`\\mathbf{I}` is the :math:`3\\times 3` identity matrix, and
    :math:`\\boldsymbol{\\epsilon}^\\times` is the `skew-symmetric matrix
    <https://en.wikipedia.org/wiki/Skew-symmetric_matrix>`_ of
    :math:`\\boldsymbol{\\epsilon}`.

    Given :math:`\\mathbf{C}`, the quaternion components are obtained as:

    .. math::
        \\begin{array}{rcl}
        \\eta &=& \\pm\\frac{1}{2} \\big(1 + c_{11} + c_{22} + c_{33}\\big)^{1/2} \\\\
        \\boldsymbol{\\epsilon} &=& \\frac{1}{4\\eta} \\begin{bmatrix}
            c_{23} - c_{32} \\\\ c_{31} - c_{13} \\\\ c_{12} - c_{21}
        \\end{bmatrix}
        \\end{array}

    The plus sign is chosen if it is advantageous to have a unique :math:`\\eta`;
    corresponding to :math:`\\phi \\in [0, \\pi]`. To ensure uniqueness, the
    vector part is multiplied by -1 if :math:`\\eta > 0`.

    If :math:`\\eta = 0`, then it is a pure quaternion, and its vector part is:

    .. math::

        \\boldsymbol{\\epsilon} = \\begin{bmatrix}
            \\sqrt{\\frac{1+c_{11}}{2}} \\\\
            \\sqrt{\\frac{1+c_{22}}{2}} \\\\
            \\sqrt{\\frac{1+c_{33}}{2}}
        \\end{bmatrix}

    Finally, we normalize the quaternion to have unitary norm.

    .. math::

        \\mathbf{q} = \\frac{\\mathbf{q}}{\\|\\mathbf{q}\\|}

    Parameters
    ----------
    C : numpy.ndarray
        3-by-3 or N-by-3-by-3 Direction Cosine Matrix (or Matrices).

    Returns
    -------
    q : numpy.ndarray
        4-dimensional or N-by-4 Quaternion array.
    """
    C = np.copy(C)
    if C.ndim not in [2, 3]:
        raise ValueError(f"C must be a 2- or 3-dimensional array. It is {C.ndim}-dimensional.")
    if C.shape[-2:] != (3, 3):
        raise ValueError(f"C must be an array of shape 3-by-3 or N-by-3-by-3. Got {C.shape}")
    if C.ndim < 3:
        tr = np.clip(C.trace(), -1.0, 3.0)      # Clip trace to [-1, 3]
        if np.isclose(tr, 3.0):
            return np.array([1., 0., 0., 0.])   # No rotation. DCM is identity.
        n = 0.5*np.sqrt(1.0 + tr)               # (eq. 15)
        if np.isclose(n, 0):                    # trace = -1: q_w = 0 (Pure Quaternion)
            e = np.sqrt((1.0+np.diag(C))/2.0)
        else:
            e = np.array([C[1, 2]-C[2, 1], C[2, 0]-C[0, 2], C[0, 1]-C[1, 0]])/(4*n)    # (eq. 16)
        if n > 0:
            e *= -1
        q = np.array([n, *e])
        return q / np.linalg.norm(q)
    # Handle three-dimensional array
    tr = np.clip(np.trace(C, axis1=1, axis2=2), -1.0, 3.0)
    Q = np.zeros((C.shape[0], 4))
    Q[:, 0] = 0.5*np.sqrt(1.0 + tr)             # (eq. 15)
    Q_w = np.where(np.isclose(Q[:, 0], 0.0), 1.0, Q[:, 0])  # Vector parts divided by one, when pure quaternion
    Q[:, 1] = np.array(C[:, 1, 2]-C[:, 2, 1])   # (eq. 16)
    Q[:, 2] = np.array(C[:, 2, 0]-C[:, 0, 2])
    Q[:, 3] = np.array(C[:, 0, 1]-C[:, 1, 0])
    Q[:, 1:] /= 4.0*Q_w[:, None]
    return Q

def sarabandi(dcm: np.ndarray, eta: float = 0.0) -> np.ndarray:
    """
    Quaternion from a Direction Cosine Matrix using Sarabandi's method
    :cite:p:`sarabandi2019`.

    A rotation matrix :math:`\\mathbf{R}` can be expressed as:

    .. math::

        \\mathbf{R} = \\begin{bmatrix}
            r_{11} & r_{12} & r_{13} \\\\
            r_{21} & r_{22} & r_{23} \\\\
            r_{31} & r_{32} & r_{33}
        \\end{bmatrix}

    A quaternion :math:`\\mathbf{q}` describing the same rotation can be
    expressed as:

    .. math::

        \\mathbf{q} = \\begin{bmatrix}
            q_w \\ q_x \\ q_y \\ q_z
        \\end{bmatrix} = \\begin{bmatrix}
            \\cos (\\frac{\\theta}{2}) \\\\
            n_x \\sin (\\frac{\\theta}{2}) \\\\
            n_y \\sin (\\frac{\\theta}{2}) \\\\
            n_z \\sin (\\frac{\\theta}{2})
        \\end{bmatrix}

    where :math:`\\theta` is the rotation angle, and :math:`\\mathbf{n}` is the
    unitary rotation axis. The quaternion is also unitary, i.e.,
    :math:`\\|\\mathbf{q}\\| = 1`.

    The rotation matrix :math:`\\mathbf{R}` can be expressed in terms of the
    quaternion :math:`\\mathbf{q}` as:

    .. math::

        \\mathbf{R} = \\begin{bmatrix}
            1 - 2q_y^2 - 2q_z^2 & 2(q_xq_y - q_zq_w) & 2(q_xq_z + q_yq_w) \\\\
            2(q_xq_y + q_zq_w) & 1 - 2q_x^2 - 2q_z^2 & 2(q_yq_z - q_xq_w) \\\\
            2(q_xq_z - q_yq_w) & 2(q_yq_z + q_xq_w) & 1 - 2q_x^2 - 2q_y^2
        \\end{bmatrix}

    As with `Shepperd's method <./shepperd.html>`_, we build a system of linear
    equations with :math:`q_w`, :math:`q_x`, :math:`q_y` and :math:`q_z`:

    .. math::

        \\begin{array}{rcl}
        4q_w^2 &=& 1 + r_{11} + r_{22} + r_{33} \\\\
        4q_x^2 &=& 1 + r_{11} - r_{22} - r_{33} \\\\
        4q_y^2 &=& 1 - r_{11} + r_{22} - r_{33} \\\\
        4q_z^2 &=& 1 - r_{11} - r_{22} + r_{33} \\\\
        4q_yq_z &=& r_{23} + r_{32} \\\\
        4q_xq_z &=& r_{31} + r_{13} \\\\
        4q_xq_y &=& r_{12} + r_{21} \\\\
        4q_wq_x &=& r_{32} - r_{23} \\\\
        4q_wq_y &=& r_{13} - r_{31} \\\\
        4q_wq_z &=& r_{21} - r_{12}
        \\end{array}

    Clearing for :math:`q_w`, we get:

    .. math::

        q_w = \\frac{1}{2}\\sqrt{1 + r_{11} + r_{22} + r_{33}}

    We see that
    :math:`\\mathrm{trace}(\\mathbf{R}) = r_{11}+r_{22}+r_{33} = 2\\cos\\theta + 1`.

    This becomes ill-conditioned when :math:`\\theta \\rightarrow \\pi`, and
    :math:`q_w` can even become negative due to rounding errors, which is not
    allowed for our unit quaternion.

    To obtain a more robust solution, we involve the off-diagonal elements of
    the rotation matrix.

    Using the system of linear equations to substitute :math:`q_w`, :math:`q_x`,
    :math:`q_y` and :math:`q_z` in :math:`q_w^2 + q_x^2 + q_y^2 + q_z^2 = 1` we
    get:

    .. math::

        \\frac{1+r_{11}+r_{22}+r_{33}}{4} +
        \\Bigg(\\frac{r_{32}-r_{23}}{4q_w}\\Bigg)^2 +
        \\Bigg(\\frac{r_{13}-r_{31}}{4q_w}\\Bigg)^2 +
        \\Bigg(\\frac{r_{21}-r_{12}}{4q_w}\\Bigg)^2 = 1

    Solving for :math:`q_w`:

    .. math::

        q_w = \\frac{1}{2}\\sqrt{\\frac{(r_{32}-r_{23})^2 + (r_{13}-r_{31})^2 + (r_{21}-r_{12})^2}{3 - r_{11} - r_{22} - r_{33}}}

    This new definition of :math:`q_w` is now ill-conditioned when
    :math:`\\theta \\rightarrow 0`. Thus, both definitions can be seen as
    complementary.

    Now we simply establish a threshold for the trace of the rotation matrix.
    This threshold is easily defined from the terms inside the square root.

    .. math::

        q_w =
        \\left\\{
        \\begin{array}{lc}
            \\frac{1}{2}\\sqrt{1+r_{11}+r_{22}+r_{33}} & \\mathrm{if}\\; r_{11}+r_{22}+r_{33} > \\eta \\\\
            \\frac{1}{2}\\sqrt{\\frac{(r_{32}-r_{23})^2 + (r_{13}-r_{31})^2 + (r_{21}-r_{12})^2}{3-r_{11}-r_{22}-r_{33}}} & \\mathrm{otherwise}
        \\end{array}
        \\right.

    Repeating the same process for the other elements of the quaternion:

    .. math::

        \\begin{array}{rcl}
        q_x &=&
            \\left\\{
            \\begin{array}{lc}
                \\frac{1}{2}\\sqrt{1+r_{11}-r_{22}-r_{33}} & \\mathrm{if}\\; r_{11}-r_{22}-r_{33} > \\eta \\\\
                \\frac{1}{2}\\sqrt{\\frac{(r_{32}-r_{23})^2 + (r_{12}+r_{21})^2 + (r_{31}+r_{13})^2}{3-r_{11}+r_{22}+r_{33}}} & \\mathrm{otherwise}
            \\end{array}
            \\right.\\\\\\\\
        q_y &=&
            \\left\\{
            \\begin{array}{lc}
                \\frac{1}{2}\\sqrt{1-r_{11}+r_{22}-r_{33}} & \\mathrm{if}\\; -r_{11}+r_{22}-r_{33} > \\eta \\\\
                \\frac{1}{2}\\sqrt{\\frac{(r_{13}-r_{31})^2 + (r_{12}+r_{21})^2 + (r_{23}+r_{32})^2}{3+r_{11}-r_{22}+r_{33}}} & \\mathrm{otherwise}
            \\end{array}
            \\right.\\\\\\\\
        q_z &=&
            \\left\\{
            \\begin{array}{lc}
                \\frac{1}{2}\\sqrt{1-r_{11}-r_{22}+r_{33}} & \\mathrm{if}\\; -r_{11}-r_{22}+r_{33} > \\eta \\\\
                \\frac{1}{2}\\sqrt{\\frac{(r_{21}-r_{12})^2 + (r_{31}+r_{13})^2 + (r_{23}+r_{32})^2}{3+r_{11}+r_{22}-r_{33}}} & \\mathrm{otherwise}
            \\end{array}
            \\right.
        \\end{array}

    Finally, if :math:`q_w` is positive, we redefine the sign of the quaternion
    elements, with the signs of :math:`r_{32}-r_{23}`, :math:`r_{13}-r_{31}`,
    and :math:`r_{21}-r_{12}` assigned to :math:`q_x`, :math:`q_y`, and
    :math:`q_z`, respectively.

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
    # Compute qw (eq. 23)
    dw = r11+r22+r33
    if dw > eta:
        qw = 0.5*np.sqrt(1.0+dw)
    else:
        nom = (r32-r23)**2+(r13-r31)**2+(r21-r12)**2
        denom = 3.0-dw
        qw = 0.5*np.sqrt(nom/denom)
    # Compute qx (eq. 24)
    dx = r11-r22-r33
    if dx > eta:
        qx = 0.5*np.sqrt(1.0+dx)
    else:
        nom = (r32-r23)**2+(r12+r21)**2+(r31+r13)**2
        denom = 3.0-dx
        qx = 0.5*np.sqrt(nom/denom)
    # Compute qy (eq. 25)
    dy = -r11+r22-r33
    if dy > eta:
        qy = 0.5*np.sqrt(1.0+dy)
    else:
        nom = (r13-r31)**2+(r12+r21)**2+(r23+r32)**2
        denom = 3.0-dy
        qy = 0.5*np.sqrt(nom/denom)
    # Compute qz (eq. 26)
    dz = -r11-r22+r33
    if dz > eta:
        qz = 0.5*np.sqrt(1.0+dz)
    else:
        nom = (r21-r12)**2+(r31+r13)**2+(r23+r32)**2
        denom = 3.0-dz
        qz = 0.5*np.sqrt(nom/denom)
    q = np.array([qw, qx, qy, qz])
    # Re-define the sign of the quaternion if q_w is positive
    if q[0] > 0.0:
        q[1] *= np.sign(r32-r23)
        q[2] *= np.sign(r13-r31)
        q[3] *= np.sign(r21-r12)
    return q / np.linalg.norm(q)

def itzhack(dcm: np.ndarray, version: int = 3) -> np.ndarray:
    """
    Quaternion from a Direction Cosine Matrix with Bar-Itzhack's method
    :cite:p:`BarItzhack2000`.

    This method to compute the quaternion from a Direction Cosine Matrix (DCM)
    is based on the eigenvalue decomposition of the matrix :math:`\\mathbf{K}`,
    and does not require any voting scheme like other known methods.

    Moreover, this method is able to handle non-orthogonal matrices, while
    other methods require an orthogonal matrix to be used.

    As defined in `Wahba's problem <https://en.wikipedia.org/wiki/Wahba%27s_problem>`_,
    we are looking for the quaternion, :math:`\\mathbf{q} = [q_x, q_y, q_z, q_w]`,
    that minimizes the following cost function:

    .. math::

        L(\\mathbf{D}) = \\frac{1}{2}\\sum_{i=1}^{k}a_i|\\mathbf{b}_i - \\mathbf{D}\\mathbf{r}_i|^2

    where :math:`\\mathbf{D}` is the DCM obtained from the quaternion `\\mathbf{q}`:

    .. math::

        \\mathbf{D}(\\mathbf{q}) = \\begin{bmatrix}
            q_w^2 + q_x^2 - q_y^2 - q_z^2 & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\\\
            2(q_xq_y + q_wq_z) & q_w^2 - q_x^2 + q_y^2 - q_z^2 & 2(q_yq_z - q_wq_x) \\\\
            2(q_xq_z - q_wq_y) & 2(q_yq_z + q_wq_x) & q_w^2 - q_x^2 - q_y^2 + q_z^2
        \\end{bmatrix}

    .. warning::

        This method defines the quaternion as :math:`[q_x, q_y, q_z, q_w]` with
        a trailing scalar part, while other methods use
        :math:`[q_w, q_x, q_y, q_z]`, with a leading scalar part.

        The algebra in this documentation will be using Itzhack's convention.
        To cope with this, this package's implementation re-orders the
        quaternion at the end to match the most common definition.

    :math:`\\mathbf{r}` are unit vectors in the reference coordinate frame,
    :math:`\\mathbf{b}` are the same vectors but in the body coordinate frame,
    and :math:`\\mathbf{a}` are a set of nonnegative weights assign to each
    pair.

    Paul Davenport :cite:p:`davenport1968` finds the optimal quaternion,
    :math:`\\mathbf{q}^*`, that minimizes the cost function, through the
    eigenvalue decomposition of the matrix :math:`\\mathbf{K}`:

    .. math::

        \\mathbf{K} = \\begin{bmatrix}
            \\mathbf{S} - \\boldsymbol{\\sigma}\\mathbf{I}_3 & \\mathbf{z} \\\\
            \\mathbf{z}^T & \\boldsymbol{\\sigma}
        \\end{bmatrix}

    `Davenport's Method <../filters/davenport.html>`_ yields the quaternion
    describing a rotation from one frame to another when the components of at
    least two vectors in each frame are known in both frames.

    If we know the precise DCM that characterizes a certain rotation, we can
    use it to generate such pairs, and then apply Daveport's method back to
    these pairs, which yield a quaternion. Thus, we have computed the sought
    quaternion.

    Itzhack's algorithm has three versions depending on the given DCM. The
    first two algorithms are for a given **orthogonal** attitude matrix.

    **Version 1**

    Because only two vectors are necessary to determine attitude, we can
    simplify the computation choosing two unit vectors of the reference
    coordinates:

    .. math::

        \\begin{array}{rcl}
        \\mathbf{r}_1^T &=& \\begin{bmatrix} 1 & 0 & 0 \\end{bmatrix} \\\\
        \\mathbf{r}_2^T &=& \\begin{bmatrix} 0 & 1 & 0 \\end{bmatrix}
        \\end{array}

    From the relation :math:`\\mathbf{b}_i = \\mathbf{D}\\mathbf{r}_i`, it is
    evident that the vectors in the body system that correspond to
    :math:`\\mathbf{r}_1` and :math:`\\mathbf{r}_2` are :math:`\\mathbf{b}_1`
    and :math:`\\mathbf{b}_2`, respectively.

    .. note::

        The matrix :math:`\\mathbf{D}` is `one-based indexing
        <https://en.wikipedia.org/wiki/Zero-based_numbering>`_. Thus,

        .. math::

            \\mathbf{D} = \\begin{bmatrix}
                | & | & | \\\\ \\mathbf{d}_1 & \\mathbf{d}_2 & \\mathbf{d}_3 \\\\ | & | & |
            \\end{bmatrix}
            = \\begin{bmatrix}
                d_{11} & d_{12} & d_{13} \\\\
                d_{21} & d_{22} & d_{23} \\\\
                d_{31} & d_{32} & d_{33}
            \\end{bmatrix}

    Because :math:`\\mathbf{D}` and :math:`\\mathbf{r}_i` are available and
    have a simple form, we can compute :math:`\\mathbf{K}_2` directly using
    :math:`a_i = 0.5`:

    .. math::

        \\mathbf{K}_2 = \\frac{1}{2}\\begin{bmatrix}
            d_{11} - d_{22} & d_{21} + d_{12} & d_{31} & -d_{32} \\\\
            d_{21} + d_{12} & d_{22} - d_{11} & d_{32} & d_{31} \\\\
            d_{31} & d_{32} & -d_{11} - d_{22} & d_{12} - d_{21} \\\\
            -d_{32} & d_{31} & d_{12} - d_{21} & d_{11} + d_{22}
        \\end{bmatrix}

    The sought quaternion, :math:`\\mathbf{q}`, is obtained by computing the
    eigenvector of :math:`\\mathbf{K}_2` that belongs to the eigenvalue 1.

    **Version 2**

    If the given DCM is **imprecise but still orthogonal**, we can use either
    two or three pairs and obtain the same results.

    However, the quaternion obtained when using three pairs yields the DCM that
    is the closest orthogonal matrix.

    Re-defining our cost function as:

    .. math::

        L(\\mathbf{D}) = \\sum_{i=1}^{k}a_i - \\mathrm{tr}(\\mathbf{DB})^T

    where:

    .. math::

        \\mathbf{B} = \\sum_{i=1}^{k}a_i\\mathbf{b}_i\\mathbf{r}_i^T

    In this case, the matrix :math:`\\mathbf{D}_{\\mathrm{orth}}` that
    minimizes the cost function :math:`L(\\mathbf{D})` is the same matrix that
    maximizes :math:`\\mathrm{tr}(\\mathbf{DB})^T`, and is computable as:

    .. math::

        \\mathbf{D}_{\\mathrm{orth}} = \\mathbf{B}(\\mathbf{B}^T\\mathbf{B})^{-\\frac{1}{2}}

    Adding a third pair of vectors similar to the first two:

    .. math::

        \\mathbf{r}_3^T = \\begin{bmatrix} 0 & 0 & 1 \\end{bmatrix}

    we easily redefine the matrix :math:`\\mathbf{B}`:

    .. math::

        \\begin{array}{rcl}
        \\mathbf{B}
            &=& \\frac{1}{3} \\mathbf{b}_1\\mathbf{r}_1^T + \\frac{1}{3} \\mathbf{b}_2\\mathbf{r}_2^T + \\frac{1}{3} \\mathbf{b}_3\\mathbf{r}_3^T \\\\
            &=& \\frac{1}{3}\\begin{bmatrix} \\mathbf{d}_1 & \\mathbf{d}_2 & \\mathbf{d}_3 \\end{bmatrix} \\\\
            &=& \\frac{1}{3}\\mathbf{D}
        \\end{array}

    Therefore,

    .. math::

        \\mathbf{D}_{\\mathrm{orth}} = \\mathbf{D}(\\mathbf{D}^T\\mathbf{D})^{-\\frac{1}{2}}

    Using only two vectors would still yield an optimal quaternion, but it
    would not correspond to the closest orthogonal matrix of the given
    imprecise \\mathbf{D}.

    Thus, :math:`\\mathbf{D}_{\\mathrm{orth}}` is the closest orthogonal matrix
    of the given imprecise :math:`\\mathbf{D}` that solves Wahba's problem.

    From here, we define :math:`\\mathbf{K}_3` as:

    .. math::

        \\mathbf{K}_3 = \\frac{1}{3}\\begin{bmatrix}
            d_{11} - d_{22} - d_{33} & d_{21} + d_{12} & d_{31} + d_{13} & d_{23} - d_{32} \\\\
            d_{21} + d_{12} & d_{22} - d_{11} - d_{33} & d_{32} + d_{23} & d_{31} - d_{13} \\\\
            d_{31} + d_{13} & d_{32} + d_{23} & d_{33} - d_{11} - d_{22} & d_{12} - d_{21} \\\\
            d_{23} - d_{32} & d_{31} - d_{13} & d_{12} - d_{21} & d_{11} + d_{22} + d_{33}
        \\end{bmatrix}

    And, similarly to version 1, the sought quaternion, :math:`\\mathbf{q}`, is
    the eigenvector of :math:`\\mathbf{K}_3` that belongs to the eigenvalue 1.

    **Version 3**

    If a given DCM is not quite orthogonal, the results will not be correct.
    If two resulting quaternions are converted to DCM, the two DCMs will be
    orthogonal because this is an inherent quality of the expression of the DCM
    in terms of the corresponding quaternion.

    Therefore, for a given non-orthogonal DCM, we re-use the same matrix
    :math:`\\mathbf{K}_3`.

    The sought quaternion, :math:`\\mathbf{q}`, is the eigenvector of
    :math:`\\mathbf{K}_3` that belongs to its largest eigenvalue,
    :math:`\\lambda_{\\mathrm{max}}`.

    The main benefit of this version is that the computed quaternion yields the
    closest orthogonal matrix to the given DCM.

    Finally, the extraction of the quaternion from the :math:`\\mathbf{K}`
    matrix can be done either using the `QUEST <../filters/quest.html>`_ and
    similar algorithms or, preferably, using a known standard algorithm for
    computing the eigenvalues and eigenvectors of a real symmetric matrix.

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
    dcm = np.copy(dcm)
    if dcm.ndim != 2 or dcm.shape != (3, 3):
        raise ValueError('dcm must be a 3-by-3 array.')
    if version not in [1, 2, 3]:
        raise ValueError('version must be 1, 2 or 3.')
    if np.isnan(dcm).any():
        return np.array([np.nan]*4)
    # Get elements of DCM
    d11, d12, d13 = dcm[0, 0], dcm[0, 1], dcm[0, 2]
    d21, d22, d23 = dcm[1, 0], dcm[1, 1], dcm[1, 2]
    d31, d32, d33 = dcm[2, 0], dcm[2, 1], dcm[2, 2]
    is_orthogonal = np.isclose(np.linalg.det(dcm), 1.0) and np.allclose(dcm@dcm.T, np.eye(3))
    if version in [1, 2] and not is_orthogonal:
        raise ValueError('Given matrix is non-orthogonal. Versions 1 and 2 are for orthogonal matrices.')
    if version == 1:
        K2 = np.array([
            [ d11-d22, d21+d12,      d31,    -d32],
            [ d21+d12, d22-d11,      d32,     d31],
            [ d31,         d32, -d11-d22, d12-d21],
            [-d32,         d31,  d12-d21, d11+d22]]) / 2.0  # (eq. 1)
        eigval, eigvec = np.linalg.eig(K2)
        q = eigvec[:, np.where(np.isclose(eigval, 1.0))[0]].flatten().real
    else:
        K3 = np.array([
            [d11-d22-d33,     d21+d12,     d31+d13,     d23-d32],
            [d21+d12,     d22-d11-d33,     d32+d23,     d31-d13],
            [d31+d13,         d32+d23, d33-d11-d22,     d12-d21],
            [d23-d32,         d31-d13,     d12-d21, d11+d22+d33]]) / 3.0    # (eq. 2)
        eigval, eigvec = np.linalg.eig(K3)
        if version == 2:
            q = eigvec[:, np.where(np.isclose(eigval, 1.0))[0]].flatten().real
        else:
            q = eigvec[:, eigval.argmax()]
    q = np.roll(q, 1)       # Re-arrange quaternion to [qw, qx, qy, qz]
    q[0] *= -1              # Original implementation computes inverse rotation
    return q / np.linalg.norm(q)

def shepperd(dcm: np.ndarray) -> np.ndarray:
    """
    Quaternion from a Direction Cosine Matrix with Shepperd's method
    :cite:p:`shepperd1978`.

    Since it was proposed in 1978, the Shepperd method has been widely used
    in the aerospace industry.

    An arbitrary rotation in :math:`\\mathbf{R}\\in\\mathbb{R}^3` is
    represented by an orthogonal matrix of the form:

    .. math::
        \\mathbf{R} = \\begin{bmatrix}
            r_{11} & r_{12} & r_{13} \\\\
            r_{21} & r_{22} & r_{23} \\\\
            r_{31} & r_{32} & r_{33}
        \\end{bmatrix}

    The information about the axis and angle of rotation is usually organized
    as a quaternion :math:`\\mathbf{q} = [q_w, q_x, q_y, q_z]`, where the
    unit vector :math:`\\mathbf{n} = [n_x, n_y, n_z]` is the axis of rotation,
    and :math:`\\theta` is the angle of rotation.

    .. math::

        \\begin{array}{rcl}
        q_w &=& \\cos (\\frac{\\theta}{2}) \\\\
        q_x &=& n_x \\sin (\\frac{\\theta}{2}) \\\\
        q_y &=& n_y \\sin (\\frac{\\theta}{2}) \\\\
        q_z &=& n_z \\sin (\\frac{\\theta}{2})
        \\end{array}

    This quaternion is unitary, which means that its norm is equal to 1:

    .. math::

        \\|\\mathbf{q}\\| = \\sqrt{q_w^2 + q_x^2 + q_y^2 + q_z^2} = 1

    A :math:`3\\times 3` rotation matrix can be expressed in terms of the
    quaternion as:

    .. math::

        \\mathbf{R} = \\begin{bmatrix}
            q_w^2+q_x^2-q_y^2-q_z^2 & 2(q_xq_y-q_zq_w) & 2(q_xq_z+q_yq_w) \\\\
            2(q_xq_y+q_zq_w) & q_w^2-q_x^2+q_y^2-q_z^2 & 2(q_yq_z-q_xq_w) \\\\
            2(q_xq_z-q_yq_w) & 2(q_yq_z+q_xq_w) & q_w^2-q_x^2-q_y^2+q_z^2
        \\end{bmatrix}

    Clearing for the quaternion components, we have:

    .. math::

        \\begin{array}{rcl}
        4q_w^2 &=& 1 + r_{11} + r_{22} + r_{33} \\\\
        4q_x^2 &=& 1 + r_{11} - r_{22} - r_{33} \\\\
        4q_y^2 &=& 1 - r_{11} + r_{22} - r_{33} \\\\
        4q_z^2 &=& 1 - r_{11} - r_{22} + r_{33} \\\\
        4q_yq_z &=& r_{23} + r_{32} \\\\
        4q_xq_z &=& r_{31} + r_{13} \\\\
        4q_xq_y &=& r_{12} + r_{21} \\\\
        4q_wq_x &=& r_{32} - r_{23} \\\\
        4q_wq_y &=& r_{13} - r_{31} \\\\
        4q_wq_z &=& r_{21} - r_{12}
        \\end{array}

    From this system of equations, there are 4 possible solutions for the
    quaternion, which are:

    .. math::

        \\begin{array}{rcl}
            \\mathbf{q}_1 &=& \\frac{1}{2} \\begin{bmatrix}
                \\sqrt{1+r_{11}+r_{22}+r_{33}} \\\\
                \\frac{r_{32}-r_{23}}{\\sqrt{1+r_{11}+r_{22}+r_{33}}} \\\\
                \\frac{r_{13}-r_{31}}{\\sqrt{1+r_{11}+r_{22}+r_{33}}} \\\\
                \\frac{r_{21}-r_{12}}{\\sqrt{1+r_{11}+r_{22}+r_{33}}}
            \\end{bmatrix} \\\\ \\\\
            \\mathbf{q}_2 &=& \\frac{1}{2} \\begin{bmatrix}
                \\frac{r_{32}-r_{23}}{\\sqrt{1+r_{11}-r_{22}-r_{33}}} \\\\
                \\sqrt{1+r_{11}-r_{22}-r_{33}} \\\\
                \\frac{r_{12}+r_{21}}{\\sqrt{1+r_{11}-r_{22}-r_{33}}} \\\\
                \\frac{r_{31}+r_{13}}{\\sqrt{1+r_{11}-r_{22}-r_{33}}}
            \\end{bmatrix} \\\\ \\\\
            \\mathbf{q}_3 &=& \\frac{1}{2} \\begin{bmatrix}
                \\frac{r_{13}-r_{31}}{\\sqrt{1-r_{11}+r_{22}-r_{33}}} \\\\
                \\frac{r_{12}+r_{21}}{\\sqrt{1-r_{11}+r_{22}-r_{33}}} \\\\
                \\sqrt{1-r_{11}+r_{22}-r_{33}} \\\\
                \\frac{r_{23}+r_{32}}{\\sqrt{1-r_{11}+r_{22}-r_{33}}}
            \\end{bmatrix} \\\\ \\\\
            \\mathbf{q}_4 &=& \\frac{1}{2} \\begin{bmatrix}
                \\frac{r_{21}-r_{12}}{\\sqrt{1-r_{11}-r_{22}+r_{33}}} \\\\
                \\frac{r_{31}+r_{13}}{\\sqrt{1-r_{11}-r_{22}+r_{33}}} \\\\
                \\frac{r_{32}+r_{23}}{\\sqrt{1-r_{11}-r_{22}+r_{33}}} \\\\
                \\sqrt{1-r_{11}-r_{22}+r_{33}}
            \\end{bmatrix}
        \\end{array}

    Numerically, one of the solutions is better conditioned than the others
    due to the square root operation or when dividing by very small numbers.

    To obtain the better conditioned solution for each case, we get the ordinal
    number :math:`i` of the largest element in the following vector:

    .. math::

        \\mathbf{u} = \\begin{bmatrix}
            r_{11} + r_{22} + r_{33} \\\\ r_{11} \\\\ r_{22} \\\\ r_{33}
        \\end{bmatrix}

    The best solution is, then, the one corresponding to the largest element in
    :math:`\\mathbf{u}`.

    For example, if :math:`r_{11} + r_{22} + r_{33}` is the largest of
    :math:`\\mathbf{u}`, the best solution is :math:`\\mathbf{q}_1`.

    Parameters
    ----------
    dcm : numpy.ndarray
        3-by-3 Direction Cosine Matrix.

    Returns
    -------
    q : numpy.ndarray
        Quaternion.
    """
    # Get elements of rotation matrix
    r11, r12, r13 = dcm[0, 0], dcm[0, 1], dcm[0, 2]
    r21, r22, r23 = dcm[1, 0], dcm[1, 1], dcm[1, 2]
    r31, r32, r33 = dcm[2, 0], dcm[2, 1], dcm[2, 2]
    u = np.array([r11+r22+r33, r11, r22, r33])
    i = u.argmax()      # Index of the largest element in u
    if i == 0:
        d = np.sqrt(1.0+r11+r22+r33)
        q = 0.5 * np.array([d, (r32-r23)/d, (r13-r31)/d, (r21-r12)/d])
    elif i == 1:
        d = np.sqrt(1.0+r11-r22-r33)
        q = 0.5 * np.array([(r32-r23)/d, d, (r12+r21)/d, (r31+r13)/d])
    elif i == 2:
        d = np.sqrt(1.0-r11+r22-r33)
        q = 0.5 * np.array([(r13-r31)/d, (r12+r21)/d, d, (r23+r32)/d])
    else:
        d = np.sqrt(1.0-r11-r22+r33)
        q = 0.5 * np.array([(r21-r12)/d, (r31+r13)/d, (r32+r23)/d, d])
    q /= np.linalg.norm(q)
    return q

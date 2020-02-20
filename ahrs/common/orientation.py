# -*- coding: utf-8 -*-
"""
Routines for orientation estimation.

Further description will follow.

Notes
-----
- The functions involving quaternions are now better implemented, controlled
  and documented in the class Quaternion.

"""

import numpy as np
from ahrs.common.mathfuncs import cosd, sind, DEG2RAD, RAD2DEG

def q_conj(q):
    """
    Return the conjugate of a unit quaternion

    A unit quaternion, whose form is :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`,
    has a conjugate of the form :math:`\\mathbf{q}^* = (q_w, -q_x, -q_y, -q_z)`.

    Remember, unit quaternions must have a norm equal to 1:

    .. math::

        \\|\\mathbf{q}\\| = \\sqrt{q_w^2+q_x^2+q_y^2+q_z^2} = 1.0

    Parameters
    ----------
    q : array
        Unit quaternion or 2D array of Quaternions.

    Returns
    -------
    q_conj : array
        Conjugated quaternion or 2D array of conjugated Quaternions.

    Examples
    --------
    >>> q = np.array([0.603297, 0.749259, 0.176548, 0.20850 ])
    >>> ahrs.common.orientation.q_conj(q)
    array([0.603297, -0.749259, -0.176548, -0.20850 ])
    >>> Q = np.array([[0.039443, 0.307174, 0.915228, 0.257769],
        [0.085959, 0.708518, 0.039693, 0.699311],
        [0.555887, 0.489330, 0.590976, 0.319829],
        [0.578965, 0.202390, 0.280560, 0.738321],
        [0.848611, 0.442224, 0.112601, 0.267611]])
    >>> ahrs.common.orientation.q_conj(Q)
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
    if q.ndim > 2 or q.shape[-1] != 4:
        return None
    # I think that can be done better with some clever use of numpy.ndim
    return np.array([1., -1., -1., -1.])*np.array(q)

def q_random(size=1):
    """
    Generate random quaternions

    Parameters
    ----------
    size : int
        Number of Quaternions to generate. Default is 1 quaternion only.

    Returns
    -------
    q : array
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
    if not(size > 0 and isinstance(size, int)):
        raise ValueError("size must be a positive non-zero integer value.")
    q = np.random.random((size, 4))-0.5
    q /= np.linalg.norm(q, axis=1)[:, np.newaxis]
    if size == 1:
        return q[0]
    return q


def q_norm(q):
    """
    Return the normalized quaternion [WQ1]_ :math:`\\mathbf{q}_u`, also known as a
    versor [WV1]_ :

    .. math::

        \\mathbf{q}_u = \\frac{1}{\\|\\mathbf{q}\\|} \\mathbf{q}

    where:

    .. math::

        \\|\\mathbf{q}_u\\| = 1.0

    Parameters
    ----------
    q : array
        Quaternion to normalize

    Returns
    -------
    q_u : array
        Normalized Quaternion

    Examples
    --------
    >>> import numpy as np
    >>> import ahrs
    >>> q = np.random.random(4)
    >>> q
    array([0.94064704, 0.12645116, 0.80194097, 0.62633894])
    >>> q = ahrs.common.orientation.q_norm(q)
    >>> q
    array([0.67600473, 0.0908753 , 0.57632232, 0.45012429])
    >>> np.linalg.norm(q)
    1.0

    References
    ----------
    .. [WQ1] https://en.wikipedia.org/wiki/Quaternion#Unit_quaternion
    .. [WV1] https://en.wikipedia.org/wiki/Versor

    """
    if len(q)!=4:
        return None
    return q/np.linalg.norm(q)

def q_prod(p, q):
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
    p : array
        First quaternion to multiply
    q : array
        Second quaternion to multiply

    Returns
    -------
    pq : array
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

def q_mult_L(q):
    """
    Return the matrix form of a left-sided quaternion multiplication Q.

    Parameters
    ----------
    q : array
        Quaternion to multiply from the left side.

    Returns
    -------
    Q : array
        Matrix form of the left side quaternion multiplication.

    """
    q /= np.linalg.norm(q)
    Q = np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1],  q[0], -q[3],  q[2]],
        [q[2],  q[3],  q[0], -q[1]],
        [q[3], -q[2],  q[1],  q[0]]])
    return Q

def q_mult_R(q):
    """
    Return the matrix form of a right-sided quaternion multiplication Q.

    Parameters
    ----------
    q : array
        Quaternion to multiply from the right side.

    Returns
    -------
    Q : array
        Matrix form of the right side quaternion multiplication.

    """
    q /= np.linalg.norm(q)
    Q = np.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1],  q[0],  q[3], -q[2]],
        [q[2], -q[3],  q[0],  q[1]],
        [q[3],  q[2], -q[1],  q[0]]])
    return Q

def q_rot(v, q):
    """
    Rotate vector :math:`\\mathbf{v}` through quaternion :math:`\\mathbf{q}`.

    It should be equal to calling `q2R(q)@v`.

    Parameters
    ----------
    v : array
        Vector to rotate in 3 dimensions.
    q : array
        Quaternion to rotate through.

    """
    qw, qx, qy, qz = q
    return np.array([
        -2.0*v[0]*(qy**2 + qz**2 - 0.5) + 2.0*v[1]*(qw*qz + qx*qy)       - 2.0*v[2]*(qw*qy - qx*qz),
        -2.0*v[0]*(qw*qz - qx*qy)       - 2.0*v[1]*(qx**2 + qz**2 - 0.5) + 2.0*v[2]*(qw*qx + qy*qz),
         2.0*v[0]*(qw*qy + qx*qz)       - 2.0*v[1]*(qw*qx - qy*qz)       - 2.0*v[2]*(qx**2 + qy**2 - 0.5)])

def axang2quat(axis, angle, rad=True):
    """
    Return Quaternion from given Axis-Angle.

    Parameters
    ----------
    axis : array
        Unit vector indicating the direction of an axis of rotation.
    angle : float
        Angle describing the magnitude of rotation about the axis.

    Returns
    -------
    q : array
        Unit quaternion

    Examples
    --------
    >>> import numpy as np
    >>> from ahrs import quaternion
    >>> q = quaternion.axang2quat([1.0, 0.0, 0.0], np.pi/2.0)
    array([0.70710678 0.70710678 0.         0.        ])

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
    .. [2] https://www.mathworks.com/help/robotics/ref/axang2quat.html

    """
    if axis is None:
        return [1.0, 0.0, 0.0, 0.0]
    if len(axis) != 3:
        return None
    axis /= np.linalg.norm(axis)
    qw = np.cos(angle/2.0) if rad else cosd(angle/2.0)
    s = np.sin(angle/2.0) if rad else sind(angle/2.0)
    q = np.array([qw] + list(s*axis))
    return q/np.linalg.norm(q)

def quat2axang(q):
    """
    Return Axis-Angle representation from a given Quaternion.

    Parameters
    ----------
    q : array
        Unit quaternion

    Returns
    -------
    axis : array
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
    axis = np.asarray(q[1:])
    denom = np.linalg.norm(axis)
    angle = 2.0*np.arctan2(denom, q[0])
    axis = np.array([0.0, 0.0, 0.0]) if angle == 0.0 else axis/denom
    return axis, angle

def q_correct(q, full=True):
    """
    Correct quaternion from flipping to its conjugate.

    If a quaternion jumps to its conjugate, it will be corrected and brought
    back to its original position.

    Parameters
    ----------
    q : array
        N-by-4 array of quaternions, where N is the number of continuous
        quaternions.
    full : bool
        Indiciates whether the flip is full (entirely negative) or only at the
        conjugate (scalar part).

    Returns
    -------
    q : array
        Corrected array of quaternions.
    """
    q_v = -q.copy() if full else -1.0*q[:, 1:]
    q_diff = np.diff(q_v, axis=0)
    q_spikes = np.unique(np.where(abs(q_diff) > 1.0)[0]) + 1
    if len(q_spikes) < 1:
        return q
    if len(q_spikes)%2:
        q_spikes = np.concatenate((q_spikes, [len(q_v)]))
    spans = q_spikes.reshape((len(q_spikes)//2, 2))
    q_corrected = q.copy()
    for s in spans:
        if full:
            q_corrected[s[0]:s[1]] = q_v[s[0]:s[1]]
        else:
            q_corrected[s[0]:s[1], 1:] = q_v[s[0]:s[1]]
    return q_corrected

def q2R(q):
    """
    Return a rotation matrix :math:`\\mathbf{R} \\in SO(3)` from a given unit
    quaternion :math:`\\mathbf{q}`.

    The given unit quaternion :math:`\\mathbf{q}` must have the form
    :math:`\\mathbf{q} = (q_w, q_x, q_y, q_z)`, where :math:`\\mathbf{q}_v = (q_x, q_y, q_z)`
    is the vector part, and :math:`q_w` is the scalar part.

    The resulting rotation matrix :math:`\\mathbf{R}` has the form  [W1]_ [W2]_:

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
    q : array
        Unit quaternion

    Returns
    -------
    R : array
        3-by-3 rotation matrix R.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    .. [2] https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix

    """
    if q is None:
        return np.identity(3)
    if len(q) != 4:
        return None
    if not isinstance(q, np.ndarray):
        q = np.asarray(q)
    q /= np.linalg.norm(q)
    return np.array([
        [1.0-2.0*(q[2]**2+q[3]**2), 2.0*(q[1]*q[2]-q[0]*q[3]), 2.0*(q[1]*q[3]+q[0]*q[2])],
        [2.0*(q[1]*q[2]+q[0]*q[3]), 1.0-2.0*(q[1]**2+q[3]**2), 2.0*(q[2]*q[3]-q[0]*q[1])],
        [2.0*(q[1]*q[3]-q[0]*q[2]), 2.0*(q[0]*q[1]+q[2]*q[3]), 1.0-2.0*(q[1]**2+q[2]**2)]])

def q2euler(q):
    """
    Convert from a unit Quaternion to Euler Angles.

    Parameters
    ----------
    q : array
        Unit quaternion

    Returns
    -------
    angles : array
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
    phi   = np.arctan2( R_21, R_22)
    theta = -np.arctan( R_20/np.sqrt(1.0-R_20**2))
    psi   = np.arctan2( R_10, R_00)
    return np.array([phi, theta, psi])

def rotation(ax=None, ang=0.0):
    """
    Return a :math:`3 \\times 3` rotation matrix :math:`\\mathbf{R} \\in SO(3)`

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
    R : ndarray
        3-by-3 rotation matrix.

    Examples
    --------
    >>> from ahrs import quaternion
    >>> quaternion.rotation()
    array([[1. 0. 0.],
           [0. 1. 0.],
           [0. 0. 1.]])
    >>> ahrs.rotation('z', 30.0)
    array([[ 0.8660254 -0.5        0.       ],
           [ 0.5        0.8660254  0.       ],
           [ 0.         0.         1.       ]])
    >>> # Accepts angle input as string
    ... ahrs.rotation('x', '-30')
    array([[ 1.         0.         0.       ],
           [ 0.         0.8660254  0.5      ],
           [ 0.        -0.5        0.8660254]])

    Handles wrong inputs

    >>> ahrs.rotation('false_axis', 'invalid_angle')
    array([[1. 0. 0.],
           [0. 1. 0.],
           [0. 0. 1.]])
    >>> ahrs.rotation(None, None)
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
    if ax is None:
        if ang == 0.0:
            return I_3
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
    if ax.lower() == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    if ax.lower() == "y":
        return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    if ax.lower() == "z":
        return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])

def rot_seq(axes=None, angles=None):
    """
    Return a :math:`3 \\times 3` rotation matrix :math:`\\mathbf{R} \\in SO(3)`
    from given set of axes and angles.

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
    R : ndarray
        Rotation matrix.

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
    if not isinstance(axes, list):
        axes = list(axes)
    num_rotations = len(axes)
    if num_rotations < 1:
        return R
    if set(axes).issubset(set(accepted_axes)):
        # Perform the matrix multiplications
        for i in range(num_rotations-1, -1, -1):
            R = rotation(axes[i], angles[i])@R
    return R

def dcm2quat(R):
    """
    Return a unit quaternion from a given Direct Cosine Matrix.

    Parameters
    ----------
    R : ndarray
        Direction Cosine Matrix.

    Returns
    -------
    q : ndarray
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
    q[0] = 0.5*np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2])
    q[1] = (R[1, 2] - R[2, 1]) / q[0]
    q[2] = (R[2, 0] - R[0, 2]) / q[0]
    q[3] = (R[0, 1] - R[1, 0]) / q[0]
    q[1:] /= 4.0
    return q / np.linalg.norm(q)

def cardan2q(angles, in_deg=False):
    """
    Return a Quaternion from given cardan angles with order: roll, pitch, yaw.
    Where roll is the first rotation (about X-axis), pitch is the second
    rotation (about Y-axis), and yaw is the last rotation (about Z-axis.)

    Parameters
    ----------
    angles : array
        Cardan angles.

    Returns
    -------
    q : array
        Quaternion.

    """
    if angles.shape[-1] != 3:
        return None
    if in_deg:
        angles *= DEG2RAD
    cr = np.cos(0.5*angles[0])
    sr = np.sin(0.5*angles[0])
    cp = np.cos(0.5*angles[1])
    sp = np.sin(0.5*angles[1])
    cy = np.cos(0.5*angles[2])
    sy = np.sin(0.5*angles[2])
    # To Quaternion
    q = np.array([
        cy*cp*cr + sy*sp*sr,
        cy*cp*sr - sy*sp*cr,
        sy*cp*sr + cy*sp*cr,
        sy*cp*cr - cy*sp*sr])
    q /= np.linalg.norm(q)
    return q

def q2cardan(q):
    """
    Return the cardan angles from a given quaternion, where the angles have the
    order: roll, pitch, yaw.

    Roll is the first rotation (about X-axis), pitch is the second
    rotation (about Y-axis), and yaw is the last rotation (about Z-axis.)

    Parameters
    ----------
    q : array
        Quaternion.

    Returns
    -------
    angles : array
        Cardan angles.
    """
    if q.shape[-1] != 4:
        return None
    roll = np.arctan2(2.0*(q[0]*q[1] + q[2]*q[3]), 1.0 - 2.0*(q[1]**2 + q[2]**2))
    pitch = np.arcsin(2.0*(q[0]*q[2] - q[3]*q[1]))
    yaw = np.arctan2(2.0*(q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0*(q[2]**2 + q[3]**2))
    return np.array([roll, pitch, yaw])

def am2q(a, m):
    """
    Estimate pose from given acceleration and/or compass using the TRIAD
    method.

    Parameters
    ----------
    a : array
        Array of single sample of 3 orthogonal accelerometers.
    m : array
        Array of single sample of 3 orthogonal magnetometers.

    Returns
    -------
    pose : array
        Estimated Quaternion

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
    if m is None:
        m = np.array([0.0, 0.0, 0.0])
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    if not isinstance(m, np.ndarray):
        m = np.array(m)
    H = np.cross(m, a)
    H /= np.linalg.norm(H)
    a /= np.linalg.norm(a)
    M = np.cross(a, H)
    # ENU
    R = np.array([[H[0], M[0], a[0]],
                  [H[1], M[1], a[1]],
                  [H[2], M[2], a[2]]])
    q = dcm2quat(R)
    return q

def acc2q(a, return_euler=False):
    """
    Estimate pose from given acceleration.

    Parameters
    ----------
    a : array
        A sample of 3 orthogonal accelerometers.
    return_euler : bool, default: False
        Return pose as Euler angles

    Returns
    -------
    pose : array
        Estimated Quaternion or Euler Angles.

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
    if len(a) == 3:
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
        q[0] =  cx2*cy2
        q[1] =  sx2*cy2
        q[2] =  cx2*sy2
        q[3] = -sx2*sy2
        # Normalize reference Quaternion
        q /= np.linalg.norm(q)
    return q

def am2angles(a, m, in_deg=False):
    """
    Estimate pose from given acceleration and compass.

    Parameters
    ----------
    a : array
        N-by-3 array with N samples of 3 orthogonal accelerometers.
    m : array
        N-by-3 array with N samples of 3 orthogonal magnetometers.

    Returns
    -------
    pose : array
        Estimated Angles

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

def triad(a, m, V1=None, V2=None, **kw):
    """
    Estimate pose from given acceleration and compass using TRIAD method.

    Parameters
    ----------
    a : array
        First 3-by-1 observation vector in body frame. Usually is normalized
        acceleration vector a = [ax ay az]^T
    m : array
        Second 3-by-1 observation vector in body frame. Usually is normalized
        magnetic field vector m = [mx my mz]^T
    V1 : array
        3-by-1 Reference vector 1. Defaults to gravity in navigation frame
        g = [0 0 1]^T
    V2 : array
        3-by-1 Reference vector 2. Defaults to magnetic field in navigation
        frame m = [cos(dip) 0 sin(dip)]^T, where dip is the magnetic dip in
        local latitude

    Extra Parameters
    ----------------
    dip : float
        Magnetic dip in local latitude. Defaults to 66.47° corresponding to
        Germany.

    Returns
    -------
    pose : array
        Estimated Direct Cosine Matrix

    References
    ----------
    .. [TRIAD] M.D. Shuster et al. Three-Axis Attitude Determination from
      Vector Observations. Journal of Guidance and Control. Volume 4. Number 1.
      1981. Page 70 (http://www.malcolmdshuster.com/Pub_1981a_J_TRIAD-QUEST_scan.pdf)
    .. [Shuster] M.D. Shuster. Deterministic Three-Axis Attitude Determination.
      The Journal of the Astronautical Sciences. Vol 52. Number 3. September
      2004. Pages 405-419 (http://www.malcolmdshuster.com/Pub_2004c_J_dirangs_AAS.pdf)
    .. [WikiTRIAD] Triad method in Wikipedia. (https://en.wikipedia.org/wiki/Triad_method)
    .. [Garcia] H. Garcia de Marina et al. UAV attitude estimation using
      Unscented Kalman Filter and TRIAD. IEE 2016. (https://arxiv.org/pdf/1609.07436.pdf)
    .. [CHall4] Chris Hall. Spacecraft Attitude Dynamics and Control.
      Chapter 4: Attitude Determination. 2003.
      (http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf)
    .. [iitbTRIAD] IIT Bombay Student Satellite Team. Triad Algorithm.
      (https://www.aero.iitb.ac.in/satelliteWiki/index.php/Triad_Algorithm)
    .. [MarkleyTRIAD] F.L. Makley et al. Fundamentals of Spacecraft Attitude
      Determination and Control. 2014. Pages 184-186.
    """
    if V1 is None:
        V1 = np.array([[0.], [0.], [1.]])
    if V2 is None:
        dip = kw.get('dip', 66.47)
        V2 = np.array([[cosd(dip)], [0.0], [sind(dip)]])
    # Normalized Observations
    W1 = np.array(a / np.linalg.norm(a)).reshape((3, 1))
    W2 = np.array(m / np.linalg.norm(m)).reshape((3, 1))
    # First Triad
    W1xW2 = np.cross(W1, W2, axis=0)
    s2 = W1xW2 / np.linalg.norm(W1xW2)
    s3 = np.cross(W1, W1xW2, axis=0) / np.linalg.norm(W1xW2)
    # Second Triad
    V1xV2 = np.cross(V1, V2, axis=0)
    r2 = V1xV2 / np.linalg.norm(V1xV2)
    r3 = np.cross(V1, V1xV2, axis=0) / np.linalg.norm(V1xV2)
    # Solve TRIAD
    Mobs = np.hstack((W1, s2, s3))
    Mref = np.hstack((V1, r2, r3))
    return Mobs@Mref.T

def quest(fb, mb, fn, mn, wf=1.0, wm=1.0):
    """
    Estimate pose from given acceleration and compass using TRIAD method.

    Parameters
    ----------
    fb : array
        3-by-1 Observation vector 1 in body frame. Usually is gravity vector
        g = [0 0 1]^T
    mb : array
        3-by-1 Observation vector 2 in body frame. Usually is magnetic field
        m = [cos(dip) 0 sin(dip)]^T, where dip is the magnetic dip in local
        latitude
    fn : array
        3-by-1 Reference vector 1. Defaults to gravity in navigation frame
    mn : array
        3-by-1 Reference vector 2. Defaults to magnetic field in navigation frame

    Extra Parameters
    ----------------
    dip : float
        Magnetic dip in local latitude. Defaults to 66.47° corresponding to
        Germany.

    Returns
    -------
    pose : array
        Estimated Quaternion

    References
    ----------
    .. [TRIAD] M.D. Shuster et al. Three-Axis Attitude Determination from
      Vector Observations. Journal of Guidance and Control. Volume 4. Number 1.
      1981. Page 70 (http://www.malcolmdshuster.com/Pub_1981a_J_TRIAD-QUEST_scan.pdf)
    .. [Shuster] M.D. Shuster. Deterministic Three-Axis Attitude Determination.
      The Journal of the Astronautical Sciences. Vol 52. Number 3. September
      2004. Pages 405-419 (http://www.malcolmdshuster.com/Pub_2004c_J_dirangs_AAS.pdf)
    .. [WikiTRIAD] Triad method in Wikipedia. (https://en.wikipedia.org/wiki/Triad_method)
    .. [Garcia] H. Garcia de Marina et al. UAV attitude estimation using
      Unscented Kalman Filter and TRIAD. IEE 2016. (https://arxiv.org/pdf/1609.07436.pdf)
    .. [CHall4] Chris Hall. Spacecraft Attitude Dynamics and Control.
      Chapter 4: Attitude Determination. 2003.
      (http://www.dept.aoe.vt.edu/~cdhall/courses/aoe4140/attde.pdf)
    .. [MarkleyQUEST] F.L. Makley et al. Fundamentals of Spacecraft Attitude
      Determination and Control. 2014. Pages 189-191.
    """
    fb = np.array(fb / np.linalg.norm(fb)).reshape((3, 1))
    mb = np.array(mb / np.linalg.norm(mb)).reshape((3, 1))
    fn = np.array(fn / np.linalg.norm(fn)).reshape((3, 1))
    mn = np.array(mn / np.linalg.norm(mn)).reshape((3, 1))
    # K matrix
    B = wf*(fb@fn.T) + wm*(mb@mn.T)
    K11 = B + B.T - np.eye(3)*np.trace(B)
    K22 = np.trace(B)
    K12 = wf*np.cross(fb, fn, axis=0) + wm*np.cross(mb, mn, axis=0)
    K21 = K12.T
    K = np.vstack((np.hstack((K11, K12)), np.hstack((K21, [[K22]]))))
    # Find eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(K)
    # Look for the largest eigenvalue
    index = np.argmax(eigvals)
    q = eigvecs[:, index]
    return q/np.linalg.norm(q)

def am2euler(a, m, in_deg=False):
    """
    Return Euler angles from gravity and geomagnetic field

    Parameters
    ----------
    a : array
        Gravitational acceleration
    m : array
        Geomagnetic field
    in_deg : bool
        Return angles in degrees.

    Returns
    -------
    angles : array
        Euler angles
    """
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

def slerp(q0, q1, t_array, **kwgars):
    """
    Spherical Linear Interpolation between quaternions.

    Return a valid quaternion rotation at a specified distance along the minor
    arc of a great circle passing through any two existing quaternion endpoints
    lying on the unit radius hypersphere.

    Based on the method detailed in [Wiki_SLERP]_

    Parameters
    ----------
    q0 : array
        First endpoint quaternion.
    q1 : array
        Second endpoint quaternion.
    t_array : array
        Array of times to interpolate to.
    threshold : float
        Threshold to closeness of interpolation.

    Returns
    -------
    q : array
        New quaternion representing the interpolated rotation.

    References
    ----------
    .. [Wiki_SLERP] https://en.wikipedia.org/wiki/Slerp

    """
    threshold = kwgars.get('threshold', 0.9995)
    t_array = np.array(t_array)
    q0 = np.array(q0)
    q1 = np.array(q1)
    qdot = np.sum(q0*q1)
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

if __name__ == '__main__':
    q = q_random(-1)

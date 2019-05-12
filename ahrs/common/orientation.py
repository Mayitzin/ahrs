# -*- coding: utf-8 -*-
"""
Routines for orientation estimation.

Further description will follow.

"""

import numpy as np
from .mathfuncs import *

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
        Unit quaternion

    Returns
    -------
    q_conj : array
        Conjugated quaternion

    References
    ----------
    .. [1] Dantam, N. (2014) Quaternion Computation. Institute for Robotics
           and Intelligent Machines. Georgia Tech.
           (http://www.neil.dantam.name/note/dantam-quaternion.pdf)
    .. [2] https://en.wikipedia.org/wiki/Quaternion#Conjugation,_the_norm,_and_reciprocal

    """
    if len(q) != 4:
        return None
    if type(q) is list:
        return [q[0], -q[1], -q[2], -q[3]]
    return -1.0*q[1:]

def q_norm(q):
    """
    Return the normalized quaternion [W1]_ :math:`\\mathbf{q}_u`, also known as a
    versor [W2]_ :

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
    >>> from ahrs import quaternion
    >>> q = np.random.random(4)
    >>> q
    array([0.94064704, 0.12645116, 0.80194097, 0.62633894])
    >>> q = quaternion.q_norm(q)
    >>> q
    array([0.67600473, 0.0908753 , 0.57632232, 0.45012429])
    >>> np.linalg.norm(q)
    1.0

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Quaternion#Unit_quaternion
    .. [2] https://en.wikipedia.org/wiki/Versor

    """
    if len(q)!=4:
        return q
    return q/np.linalg.norm(q)

def q_prod(p, q):
    """
    Product of two unit quaternions.

    Given two unit quaternions :math:`\\mathbf{p}=(p_w, \\mathbf{p}_v)` and
    :math:`\\mathbf{q} = (q_w, \\mathbf{q}_v)`, their product is defined [MW]_ [W1]_
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

    References
    ----------
    .. [1] Dantam, N. (2014) Quaternion Computation. Institute for Robotics
           and Intelligent Machines. Georgia Tech.
           (http://www.neil.dantam.name/note/dantam-quaternion.pdf)
    .. [2] https://www.mathworks.com/help/aeroblks/quaternionmultiplication.html

    Examples
    --------
    >>> import numpy as np
    >>> from ahrs import quaternion
    >>> p, q = np.random.random(4), np.random.random(4)
    >>> p /= np.linalg.norm(p)  # Quaternions must be normalized
    >>> q /= np.linalg.norm(q)
    >>> p
    array([0.55747131, 0.12956903, 0.5736954 , 0.58592763])
    >>> q
    array([0.49753507, 0.50806522, 0.52711628, 0.4652709 ])
    >>> quaternion.q_prod(p, q)
    array([-0.36348726,  0.38962514,  0.34188103,  0.77407146])

    """
    pq_w = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    pq_x = p[0]*q[1] + p[1]*q[0] - p[2]*q[3] + p[3]*q[2]
    pq_y = p[0]*q[2] + p[1]*q[3] + p[2]*q[0] - p[3]*q[1]
    pq_z = p[0]*q[3] - p[1]*q[2] + p[2]*q[1] + p[3]*q[0]
    pq = [pq_w, pq_x, pq_y, pq_z]
    if (type(p) is np.ndarray) or (type(q) is np.ndarray):
        return np.asarray(pq)
    else:
        return pq

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
    if type(q) is not np.ndarray:
        q = np.asarray(q)
    q /= np.linalg.norm(q)
    return np.array([
        [1.0-2.0*(q[2]**2+q[3]**2), 2.0*(q[1]*q[2]-q[0]*q[3]), 2.0*(q[1]*q[3]+q[0]*q[2])],
        [2.0*(q[1]*q[2]+q[0]*q[3]), 1.0-2.0*(q[1]**2+q[3]**2), 2.0*(q[2]*q[3]-q[0]*q[1])],
        [2.0*(q[1]*q[3]-q[0]*q[2]), 2.0*(q[0]*q[1]+q[2]*q[3]), 1.0-2.0*(q[1]**2+q[2]**2)]])

def q2eul(q=None, mode=0):
    """
    Convert from a unit Quaternion to Euler Angles.

    Parameters
    ----------
    q : array
        Unit quaternion

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_Angles_Conversion

    """
    if q is None:
        return [0.0, 0.0, 0.0]
    if len(q) != 4:
        return None
    q = q_norm(q)
    qw, qx, qy, qz = q
    if mode == 0:
        ex = np.arctan2(2.0*(qw*qx + qy*qz), 1.0 - 2.0*(qx*qx + qy*qy))
        ey = np.arcsin( 2.0*(qw*qy-qz*qx) )
        ez = np.arctan2(2.0*(qw*qz + qx*qy), 1.0 - 2.0*(qy*qy + qz*qz))
    elif mode == 1:
        ex = np.arctan2( (qw*qy+qx*qz), -(qx*qy-qw*qz))
        ey = np.arccos( -(qw*qw)-(qx*qx)+(qy*qy)+(qz*qz))
        ez = np.arctan2( (qw*qy-qx*qz),(qx*qy+qw*qz))
    elif mode == 2:
        ex = np.arctan((2.0*qy*qz-2.0*qw*qx)/(2.0*qw**2+2.0*qz**2-1.0))
        ey = -np.arcsin(2.0*qx*qz+2.0*qw*qy)
        ez = np.arctan((2.0*qx*qy-2.0*qw*qz)/(2*qw**2+2.0*qx**2-1.0))
    else:
        ex = 1.0 - 2.0*(qy*qy + qz*qz)
        ey = 1.0 - 2.0*(qx*qx + qz*qz)
        ez = 1.0 - 2.0*(qx*qx + qy*qy)
    return [ex*RAD2DEG, ey*RAD2DEG, ez*RAD2DEG]

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
    if type(ax) is int:
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

def rot_chain(axes=None, angles=None):
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
    >>> R = quaternion.rot_chain(axis_order, angles)
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
    if type(axes) is not list:
        axes = list(axes)
    num_rotations = len(axes)
    if num_rotations < 1:
        return R
    valid_given_axes = set(axes).issubset(set(accepted_axes))
    if valid_given_axes:
        # Perform the matrix multiplications
        for i in list(range(num_rotations-1, -1, -1)):
            R = rotation(axes[i], angles[i])@R
    return R

def R2q(R=None, eta=0.0):
    """
    Compute a Quaternion from a rotation matrix

    Use Shepperd's voting scheme to compute the corresponding Quaternion q from
    a given rotation matrix R. Optimized by Sarabandi et al.

    References
    ----------
    .. [1] Sarabandi, S. et al. (2018) Accurate Computation of Quaternions
           from Rotation Matrices.
           (http://www.iri.upc.edu/files/scidoc/2068-Accurate-Computation-of-Quaternions-from-Rotation-Matrices.pdf)

    """
    if R is None:
        R = np.identity(3)
    # Get elements of R
    r11, r12, r13 = R[0][0], R[0][1], R[0][2]
    r21, r22, r23 = R[1][0], R[1][1], R[1][2]
    r31, r32, r33 = R[2][0], R[2][1], R[2][2]
    # Compute qw
    d_w = r11+r22+r33
    if d_w > eta:
        q_w = 0.5*np.sqrt(1.0+d_w)
    else:
        nom = (r32-r23)**2+(r13-r31)**2+(r21-r12)**2
        q_w = 0.5*np.sqrt(nom/(3.0-d_w))
    # Compute qx
    d_x = r11-r22-r33
    if d_x > eta:
        q_x = 0.5*np.sqrt(1.0+d_x)
    else:
        nom = (r32-r23)**2+(r12+r21)**2+(r31+r13)**2
        q_x = 0.5*np.sqrt(nom/(3.0-d_x))
    # Compute qy
    d_y = -r11+r22-r33
    if d_y > eta:
        q_y = 0.5*np.sqrt(1.0+d_y)
    else:
        nom = (r13-r31)**2+(r12+r21)**2+(r23+r32)**2
        q_y = 0.5*np.sqrt(nom/(3.0-d_y))
    # Compute qz
    d_z = -r11-r22+r33
    if d_z > eta:
        q_z = 0.5*np.sqrt(1.0+d_z)
    else:
        nom = (r21-r12)**2+(r31+r13)**2+(r23+r32)**2
        q_z = 0.5*np.sqrt(nom/(3.0-d_z))
    # Assign signs
    if q_w >= 0.0:
        q_x *= np.sign(r32-r23)
        q_y *= np.sign(r13-r31)
        q_z *= np.sign(r21-r12)
    else:
        q_w *= -1.0
        q_x *= -np.sign(r32-r23)
        q_y *= -np.sign(r13-r31)
        q_z *= -np.sign(r21-r12)
    # Return values of quaternion
    return np.asarray([q_w, q_x, q_y, q_z])

def dcm2quat(R):
    """
    Return a unit quaternion from a given Direct Cosine Matrix.

    Parameters
    ----------
    R : array
        Direct Cosine Matrix.

    Returns
    -------
    q : array
        Unit Quaternion.

    References
    ----------
    .. [1] F. Landis Markley. Attitude Determination using two Vector
           Measurements.

    """
    if(R.shape[0]!=R.shape[1]):
        raise ValueError('Input is not a square matrix')
    if(R.shape[0]!=3):
        raise ValueError('Input needs to be a 3x3 array or matrix')
    qw = 0.5*np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2])
    qw4 = 4.0*qw
    qx = (R[1, 2] - R[2, 1]) / qw4
    qy = (R[2, 0] - R[0, 2]) / qw4
    qz = (R[0, 1] - R[1, 0]) / qw4
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)

def am2q(a, m):
    """
    Estimate pose from given acceleration and/or compass using Michel-method.

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
    if type(a) != np.ndarray:
        a = np.array(a)
    if type(m) != np.ndarray:
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
    Estimate pose from given acceleration and/or compass.

    Parameters
    ----------
    a : array
        A sample of 3 orthogonal accelerometers.
    m : array
        A sample of 3 orthogonal magnetometers.
    return_euler : bool
        Return pose as Euler angles

    Returns
    -------
    pose : array
        Estimated Quaternion or Euler Angles

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
    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    ex, ey, ez = 0.0, 0.0, 0.0
    if len(a) == 3:
        ax, ay, az = a[0], a[1], a[2]
        # Normalize accelerometer measurements
        a_norm = np.linalg.norm(a)
        ax /= a_norm
        ay /= a_norm
        az /= a_norm
        # Euler Angles from Gravity vector
        ex = np.arctan2(ay, az) - np.pi
        if ex*RAD2DEG < -180.0:
            ex += 2.0*np.pi
        ey = np.arctan2(-ax, np.sqrt(ay*ay + az*az))
        ez = 0.0
        if return_euler:
            return [ex*RAD2DEG, ey*RAD2DEG, ez*RAD2DEG]
        # Euler to Quaternion
        cx2 = np.cos(ex/2.0)
        sx2 = np.sin(ex/2.0)
        cy2 = np.cos(ey/2.0)
        sy2 = np.sin(ey/2.0)
        qrw =  cx2*cy2
        qrx =  sx2*cy2
        qry =  cx2*sy2
        qrz = -sx2*sy2
        # Normalize reference Quaternion
        q_norm = np.linalg.norm([qrw, qrx, qry, qrz])
        qw = qrw/q_norm
        qx = qrx/q_norm
        qy = qry/q_norm
        qz = qrz/q_norm
    return [qw, qx, qy, qz]

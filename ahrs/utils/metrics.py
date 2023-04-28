# -*- coding: utf-8 -*-
"""
Metrics
=======

Common metrics used in 3D Orientation representations.

References
----------
.. [Huynh] Huynh, D.Q. Metrics for 3D Rotations: Comparison and Analysis. J
    Math Imaging Vis 35, 155-164 (2009).
.. [Kuffner] Kuffner, J.J. Effective Sampling and Distance Metrics for 3D Rigid
    Body Path Planning. IEEE International Conference on Robotics and
    Automation (ICRA 2004)
.. [Hartley] R. Hartley, J. Trumpf, Y. Dai, H. Li. Rotation Averaging.
    International Journal of Computer Vision. Volume 101, Number 2. 2013.

"""

import numpy as np
from ..common.orientation import logR
from typing import Union

def _rotations_guard_clauses(R1: Union[list, np.ndarray], R2: Union[list, np.ndarray]) -> None:
    """
    Checks validity of rotation matrices.

    Raises
    ------
    TypeError
        If the rotation matrices are not valid types.
    ValueError
        If the rotation matrices are not valid shapes.

    """
    for label, rotation_matrix in zip(['R1', 'R2'], [R1, R2]):
        if not isinstance(rotation_matrix, (list, np.ndarray)):
            raise TypeError(f"{label} must be an array. Got {type(rotation_matrix)}")
    r1, r2 = np.copy(R1), np.copy(R2)
    for rotation_matrix in [r1, r2]:
        if rotation_matrix.shape[-2:] != (3, 3):
            raise ValueError(f"Rotation matrices must be of shape (N, 3, 3) or (3, 3). Got {rotation_matrix.shape}.")
    r1_shape, r2_shape = r1.shape, r2.shape
    if r1_shape != r2_shape:
        raise ValueError(f"Cannot compare R1 of shape {r1_shape} and R2 of shape {r2_shape}.")

def _quaternions_guard_clauses(q1: Union[list, np.ndarray], q2: Union[list, np.ndarray]) -> None:
    """
    Checks validity of quaternions.

    Raises
    ------
    TypeError
        If the quaternions are not valid types.
    ValueError
        If the quaternions are not valid shapes.

    """
    for label, quaternion in zip(['q1', 'q2'], [q1, q2]):
        if not isinstance(quaternion, (list, np.ndarray)):
            raise TypeError(f"{label} must be an array. Got {type(quaternion)}")
    q1, q2 = np.copy(q1), np.copy(q2)
    for quaternion in [q1, q2]:
        if quaternion.shape[-1] != 4:
            raise ValueError(f"Quaternions must be of shape (N, 4) or (4,). Got {quaternion.shape}.")
    if q1.shape != q2.shape:
        raise ValueError(f"Cannot compare q1 of shape {q1.shape} and q2 of shape {q2.shape}")

def euclidean(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    """
    Euclidean distance between two arrays as described in [Huynh]_:

    .. math::
        d(\\mathbf{x}, \\mathbf{y}) = \\sqrt{(x_0-y_0)^2 + \\dots + (x_n-y_n)^2}

    Accepts the same parameters as the function ``numpy.linalg.norm()``.

    This metric gives values in the range [0, :math:`\\pi\\sqrt{3}`]

    Parameters
    ----------
    x : array
        M-by-N array to compare. Usually a reference array.
    y : array
        M-by-N array to compare.
    mode : str
        Mode of distance computation.

    Return
    ------
    d : float
        Distance or difference between arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from ahrs.utils.metrics import euclidean
    >>> num_samples = 5
    >>> angles = np.random.uniform(low=-180.0, high=180.0, size=(num_samples, 3))
    >>> noisy = angles + np.random.randn(num_samples, 3)
    >>> euclidean(angles, noisy)
    2.585672169476804
    >>> euclidean(angles, noisy, axis=0)
    array([1.36319772, 1.78554071, 1.28032688])
    >>> euclidean(angles, noisy, axis=1)     # distance per sample
    array([0.88956871, 1.19727356, 1.5243858 , 0.68765523, 1.29007067])

    """
    x, y = np.copy(x), np.copy(y)
    if x.shape != y.shape:
        raise ValueError(f"Cannot compare x of shape {x.shape} and y of shape {y.shape}")
    return np.linalg.norm(x-y, **kwargs)

def chordal(R1: np.ndarray, R2: np.ndarray) -> Union[float, np.ndarray]:
    """
    Chordal Distance

    The chordal distance between two rotations :math:`\\mathbf{R}_1` and
    :math:`\\mathbf{R}_2` in SO(3) is the Euclidean distance between them in
    the embedding space :math:`\\mathbb{R}^{3\\times 3}=\\mathbb{R}^9`
    [Hartley]_:

    .. math::
        d(\\mathbf{R}_1, \\mathbf{R}_2) = \\|\\mathbf{R}_1-\\mathbf{R}_2\\|_F

    where :math:`\\|\\mathbf{X}\\|_F` represents the Frobenius norm of the
    matrix :math:`\\mathbf{X}`.

    The error lies within: [0, :math:`2\\sqrt{3}`]

    Parameters
    ----------
    R1 : numpy.ndarray
        3-by-3 rotation matrix.
    R2 : numpy.ndarray
        3-by-3 rotation matrix.

    Returns
    -------
    d : float or numpy.ndarray
        Chordal distance between matrices.

    Examples
    --------
    >>> import ahrs
    >>> R1 = ahrs.DCM(rpy=[0.0, 0.0, 0.0])
    >>> R2 = ahrs.DCM(rpy=[90.0, 90.0, 90.0])
    >>> ahrs.utils.chordal(R1, R2)
    2.0
    >>> R1 = ahrs.DCM(rpy=[10.0, -20.0, 30.0])
    >>> R2 = ahrs.DCM(rpy=[-10.0, 20.0, -30.0])
    >>> ahrs.utils.chordal(R1, R2)
    1.6916338074634352

    """
    _rotations_guard_clauses(R1, R2)
    R1, R2 = np.copy(R1), np.copy(R2)
    if R1.ndim < 3:
        return np.linalg.norm(R1-R2, 'fro')
    return np.array([np.linalg.norm(r1-r2, 'fro') for r1, r2 in zip(R1, R2)])

def identity_deviation(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Deviation from Identity Matrix as defined in [Huynh]_:

    .. math::
        d(\\mathbf{R}_1, \\mathbf{R}_2) = \\|\\mathbf{I}-\\mathbf{R}_1\\mathbf{R}_2^T\\|_F

    where :math:`\\|\\mathbf{X}\\|_F` represents the Frobenius norm of the
    matrix :math:`\\mathbf{X}`.

    The error lies within: [0, :math:`2\\sqrt{3}`]

    Parameters
    ----------
    R1 : numpy.ndarray
        3-by-3 rotation matrix.
    R2 : numpy.ndarray
        3-by-3 rotation matrix.

    Returns
    -------
    d : float
        Deviation from identity matrix.

    Examples
    --------
    >>> import ahrs
    >>> R1 = ahrs.DCM(rpy=[0.0, 0.0, 0.0])
    >>> R2 = ahrs.DCM(rpy=[90.0, 90.0, 90.0])
    >>> ahrs.utils.identity_deviation(R1, R2)
    2.0
    >>> R1 = ahrs.DCM(rpy=[10.0, -20.0, 30.0])
    >>> R2 = ahrs.DCM(rpy=[-10.0, 20.0, -30.0])
    >>> ahrs.utils.identity_deviation(R1, R2)
    1.6916338074634352

    """
    _rotations_guard_clauses(R1, R2)
    R1, R2 = np.copy(R1), np.copy(R2)
    return np.linalg.norm(np.eye(3)-R1@R2.T, 'fro')

def angular_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Angular distance between two rotations :math:`\\mathbf{R}_1` and
    :math:`\\mathbf{R}_2` in SO(3), as defined in [Hartley]_:

    .. math::
        d(\\mathbf{R}_1, \\mathbf{R}_2) = \\|\\log(\\mathbf{R}_1\\mathbf{R}_2^T)\\|

    where :math:`\\|\\mathbf{x}\\|` represents the usual euclidean norm of the
    vector :math:`\\mathbf{x}`.

    Parameters
    ----------
    R1 : numpy.ndarray
        3-by-3 rotation matrix.
    R2 : numpy.ndarray
        3-by-3 rotation matrix.

    Returns
    -------
    d : float
        Angular distance between rotation matrices

    Examples
    --------
    >>> import ahrs
    >>> R1 = ahrs.DCM(rpy=[0.0, 0.0, 0.0])
    >>> R2 = ahrs.DCM(rpy=[90.0, 90.0, 90.0])
    >>> ahrs.utils.angular_distance(R1, R2)
    1.5707963267948966
    >>> R1 = ahrs.DCM(rpy=[10.0, -20.0, 30.0])
    >>> R2 = ahrs.DCM(rpy=[-10.0, 20.0, -30.0])
    >>> ahrs.utils.angular_distance(R1, R2)
    1.282213683073497

    """
    _rotations_guard_clauses(R1, R2)
    R1, R2 = np.copy(R1), np.copy(R2)
    return np.linalg.norm(logR(R1@R2.T))

def qdist(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Euclidean distance between two unit quaternions as defined in [Huynh]_ and
    [Hartley]_:

    .. math::
        d(\\mathbf{q}_1, \\mathbf{q}_2) = \\mathrm{min} \\{ \\|\\mathbf{q}_1-\\mathbf{q}_2\\|, \\|\\mathbf{q}_1-\\mathbf{q}_2\\|\\}

    The error lies within [0, :math:`\\sqrt{2}`]

    Parameters
    ----------
    q1 : numpy.ndarray
        First quaternion, or set of quaternions, to compare.
    q2 : numpy.ndarray
        Second quaternion, or set of quaternions, to compare.

    Returns
    -------
    d : float
        Euclidean distance between given unit quaternions
    """
    _quaternions_guard_clauses(q1, q2)
    q1, q2 = np.copy(q1), np.copy(q2)
    if q1.ndim == 1:
        q1 /= np.linalg.norm(q1)
        q2 /= np.linalg.norm(q2)
        if np.allclose(q1, q2) or np.allclose(-q1, q2):
            return 0.0
        return min(np.linalg.norm(q1-q2), np.linalg.norm(q1+q2))
    q1 /= np.linalg.norm(q1, axis=1)[:, None]
    q2 /= np.linalg.norm(q2, axis=1)[:, None]
    return np.r_[[np.linalg.norm(q1-q2, axis=1)], [np.linalg.norm(q1+q2, axis=1)]].min(axis=0)

def qeip(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Euclidean distance of inner products as defined in [Huynh]_ and [Kuffner]_:

    .. math::
        d(\\mathbf{q}_1, \\mathbf{q}_2) = 1 - |\\mathbf{q}_1\\cdot\\mathbf{q}_2|

    The error lies within: [0, 1]

    Parameters
    ----------
    q1 : numpy.ndarray
        First quaternion, or set of quaternions, to compare.
    q2 : numpy.ndarray
        Second quaternion, or set of quaternions, to compare.

    Returns
    -------
    d : float
        Euclidean distance of inner products between given unit quaternions.
    """
    _quaternions_guard_clauses(q1, q2)
    q1, q2 = np.copy(q1), np.copy(q2)
    if q1.ndim == 1:
        q1 /= np.linalg.norm(q1)
        q2 /= np.linalg.norm(q2)
        if np.allclose(q1, q2) or np.allclose(-q1, q2):
            return 0.0
        return 1.0-abs(q1@q2)
    q1 /= np.linalg.norm(q1, axis=1)[:, None]
    q2 /= np.linalg.norm(q2, axis=1)[:, None]
    return 1.0-abs(np.nansum(q1*q2, axis=1))

def qcip(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Cosine of inner products as defined in [Huynh]_:

    .. math::
        d(\\mathbf{q}_1, \\mathbf{q}_2) = \\arccos(|\\mathbf{q}_1\\cdot\\mathbf{q}_2|)

    The error lies within: [0, :math:`\\frac{\\pi}{2}`]

    Parameters
    ----------
    q1 : numpy.ndarray
        First quaternion, or set of quaternions, to compare.
    q2 : numpy.ndarray
        Second quaternion, or set of quaternions, to compare.

    Returns
    -------
    d : float
        Cosine of inner products of quaternions.
    """
    _quaternions_guard_clauses(q1, q2)
    q1, q2 = np.copy(q1), np.copy(q2)
    if q1.ndim == 1:
        q1 /= np.linalg.norm(q1)
        q2 /= np.linalg.norm(q2)
        if np.allclose(q1, q2) or np.allclose(-q1, q2):
            return 0.0
        return np.arccos(abs(q1@q2))
    q1 /= np.linalg.norm(q1, axis=1)[:, None]
    q2 /= np.linalg.norm(q2, axis=1)[:, None]
    return np.arccos(abs(np.nansum(q1*q2, axis=1)))

def qad(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Quaternion Angle Difference

    Parameters
    ----------
    q1 : numpy.ndarray
        First quaternion, or set of quaternions, to compare.
    q2 : numpy.ndarray
        Second quaternion, or set of quaternions, to compare.

    The error lies within: [0, :math:`\\frac{\\pi}{2}`]

    Returns
    -------
    d : float
        Angle difference between given unit quaternions.
    """
    _quaternions_guard_clauses(q1, q2)
    q1, q2 = np.copy(q1), np.copy(q2)
    if q1.ndim == 1:
        q1 /= np.linalg.norm(q1)
        q2 /= np.linalg.norm(q2)
        if np.allclose(q1, q2) or np.allclose(-q1, q2):
            return 0.0
        return np.arccos(2.0*(q1@q2)**2-1.0)
    q1 /= np.linalg.norm(q1, axis=1)[:, None]
    q2 /= np.linalg.norm(q2, axis=1)[:, None]
    return np.arccos(np.clip(2.0*np.nansum(q1*q2, axis=1)**2-1.0, -1.0, 1.0))

def rmse(x: np.ndarray, y: np.ndarray):
    """
    Root Mean Squared Error

    Parameters
    ----------
    x : numpy.ndarray
        First set of values to compare.
    y : numpy.ndarray
        Second set of values to compare.

    Returns
    -------
    d : float
        Root mean squared error between given values.
    """
    return np.sqrt(np.nanmean((x-y)**2))

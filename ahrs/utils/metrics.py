# -*- coding: utf-8 -*-
"""
This submodule provides common metrics used in 3D orientation representations,
focusing on the validation and comparison of rotation matrices and quaternions.

They include guard clauses to raise appropriate errors if the inputs do not
meet the required criteria, ensuring robust and error-free computations.

Additionally, the submodule implements a euclidean error estimation function,
and functions to calculate the RMS error between two matrices or between an
array of matrices, either element-wise or as a whole.

"""

import numpy as np
from ..common.dcm import DCM
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
    Euclidean distance between two arrays as described in :cite:p:`huynh2009`:

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
    :cite:p:`hartley2013`:

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
    Deviation from Identity Matrix as defined in :cite:p:`huynh2009`:

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
    :math:`\\mathbf{R}_2` in SO(3), as defined in :cite:p:`hartley2013`:

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
    R1R2T = DCM(R1@R2.T)
    return np.linalg.norm(R1R2T.log)

def qdist(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Euclidean distance between two unit quaternions as defined in :cite:p:`huynh2009` and
    :cite:p:`hartley2013`:

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
        Euclidean distance between given unit quaternions.

    Examples
    --------
    >>> q1 = ahrs.Quaternion(random=True)
    >>> q1.view()
    Quaternion([ 0.94185064,  0.04451339, -0.00622856,  0.33301221])
    >>> q2 = ahrs.Quaternion(random=True)
    >>> q2.view()
    Quaternion([-0.51041283, -0.38336653,  0.76929238, -0.0264211 ])
    >>> ahrs.utils.qdist(q1, q2)
    0.9885466801358284
    >>> ahrs.utils.qdist(q1, -q1)
    0.0
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
    Euclidean distance of inner products as defined in :cite:p:`huynh2009` and
    :cite:p:`kuffner2004`:

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

    Examples
    --------
    >>> q1 = ahrs.Quaternion(random=True)
    >>> q1.view()
    Quaternion([ 0.94185064,  0.04451339, -0.00622856,  0.33301221])
    >>> q2 = ahrs.Quaternion(random=True)
    >>> q2.view()
    Quaternion([-0.51041283, -0.38336653,  0.76929238, -0.0264211 ])
    >>> ahrs.utils.qeip(q1, q2)
    0.48861226940378377
    >>> ahrs.utils.qeip(q1, -q1)
    0.0
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
    Cosine of inner products as defined in :cite:p:`huynh2009`:

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

    Examples
    --------
    >>> q1 = ahrs.Quaternion(random=True)
    >>> q1.view()
    Quaternion([ 0.94185064,  0.04451339, -0.00622856,  0.33301221])
    >>> q2 = ahrs.Quaternion(random=True)
    >>> q2.view()
    Quaternion([-0.51041283, -0.38336653,  0.76929238, -0.0264211 ])
    >>> ahrs.utils.qcip(q1, q2)
    1.0339974504196667
    >>> ahrs.utils.qcip(q1, -q1)
    0.0
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
    Quaternion Angle Difference as defined in :cite:p:`thibaud2017`:

    .. math::

        d(\\mathbf{q}_1, \\mathbf{q}_2) = \\arccos(2\\langle\\mathbf{q}_1,\\mathbf{q}_2\\rangle^2-1)

    The error lies within: [0, :math:`\\pi`]

    Parameters
    ----------
    q1 : numpy.ndarray
        First quaternion, or set of quaternions, to compare.
    q2 : numpy.ndarray
        Second quaternion, or set of quaternions, to compare.

    Returns
    -------
    d : float
        Angle difference between given unit quaternions.

    Examples
    --------
    >>> q1 = ahrs.Quaternion(random=True)
    >>> q1.view()
    Quaternion([ 0.94185064,  0.04451339, -0.00622856,  0.33301221])
    >>> q2 = ahrs.Quaternion(random=True)
    >>> q2.view()
    Quaternion([-0.51041283, -0.38336653,  0.76929238, -0.0264211 ])
    >>> ahrs.utils.qad(q1, q2)
    2.0679949008393335
    >>> ahrs.utils.qad(q1, -q1)
    0.0

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

    It is computed as:

    .. math::

        d(\\mathbf{x}, \\mathbf{y}) = \\sqrt{\\frac{1}{N}\\sum_{i=1}^N (x_i-y_i)^2}

    where :math:`N` is the number of elements in :math:`\\mathbf{x}` and
    :math:`\\mathbf{y}`.

    If :math:`\\mathbf{x}` and :math:`\\mathbf{y}` are :math:`M \\times N`
    matrices, then the RMSE is computed for each row, yielding a vector of
    length :math:`M`.

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
    x, y = np.copy(x), np.copy(y)
    if x.ndim > 1:
        return np.sqrt(np.nanmean((x-y)**2, axis=1))
    return np.sqrt(np.nanmean((x-y)**2))

def rmse_matrices(A: np.ndarray, B: np.ndarray, element_wise: bool = False) -> np.ndarray:
    """
    Root Mean Square Error between two arrays (matrices)

    Parameters
    ----------
    A : np.ndarray
        First M-by-N matrix or array of k M-by-N matrices.
    B : np.ndarray
        Second M-by-N matrix or array of k M-by-N matrices.
    element_wise : bool, default: False
        If True, calculate RMSE element-wise, and return an M-by-N array of
        RMSEs.

    Returns
    -------
    rmse : float or np.ndarray
        Root Mean Square Error between the two matrices, or array of k RMSEs
        between the two arrays of matrices.

    Raises
    ------
    ValueError
        If the comparing arrays do not have the same shape.

    Notes
    -----
    If the input arrays are 2-dimensional matrices, the RMSE is calculated as:

    .. math::
        RMSE = \\sqrt{\\frac{1}{MN}\\sum_{i=1}^{M}\\sum_{j=1}^{N}(A_{ij} - B_{ij})^2}

    If the input arrays are arrays of 2-dimensional matrices (3-dimensional
    array), the RMSE is calculated as:

    .. math::
        RMSE = \\sqrt{\\frac{1}{k}\\sum_{l=1}^{k}\\frac{1}{MN}\\sum_{i=1}^{M}\\sum_{j=1}^{N}(A_{ij}^{(l)} - B_{ij}^{(l)})^2}

    where :math:`k` is the number of matrices in the arrays.

    If the option ``element_wise`` is set to ``True``, the RMSE is calculated
    element-wise, and an M-by-N array of RMSEs is returned. The following calls
    are equivalent:

    .. code-block:: python

        rmse = rmse_matrices(A, B, element_wise=True)
        rmse = np.sqrt(np.nanmean((A-B)**2, axis=0))

    If the inputs are arrays of matrices (3-dimensional arrays), its call is
    also equivalent to:

    .. code-block:: python

        rmse = np.zeros_like(A[0])
        for i in range(A.shape[1]):
            for j in range(A.shape[2]):
                rmse[i, j] = np.sqrt(np.nanmean((A[:, i, j]-B[:, i, j])**2))

    If the inputs are 2-dimensional matrices, the following calls would return
    the same result:

    .. code-block:: python

        rmse_matrices(A, B)
        rmse_matrices(A, B, element_wise=False)
        rmse_matrices(A, B, element_wise=True)

    Examples
    --------
    .. code-block:: python

        >>> C = np.random.random((4, 3, 2))     # Array of four 3-by-2 matrices
        >>> C.view()
        array([[[0.2816407 , 0.30850589],
                [0.44618209, 0.33081522],
                [0.7994625 , 0.07377569]],

               [[0.35549399, 0.47050713],
                [0.94168683, 0.50388058],
                [0.70023837, 0.77216167]],

               [[0.79897129, 0.28555452],
                [0.892488  , 0.71476669],
                [0.19071524, 0.4123666 ]],

               [[0.86301978, 0.14686002],
                [0.98784823, 0.26129908],
                [0.46982206, 0.88037599]]])
        >>> D = np.random.random((4, 3, 2))     # Array of four 3-by-2 matrices
        >>> D.view()
        array([[[0.71560918, 0.34100321],
                [0.92518341, 0.50741267],
                [0.30730944, 0.19173378]],

               [[0.31846657, 0.08578454],
                [0.62643489, 0.84014104],
                [0.7111152 , 0.95428613]],

               [[0.8101591 , 0.9584096 ],
                [0.91118705, 0.71203119],
                [0.58217189, 0.45598271]],

               [[0.79837603, 0.09954558],
                [0.26532781, 0.55711476],
                [0.03909648, 0.10787888]]])
        >>> rmse_matrices(C[0], D[0])       # RMSE between first matrices
        0.3430603410873006
        >>> rmse_matrices(C, D)             # RMSE between each of the four matrices
        array([0.34306034, 0.25662067, 0.31842239, 0.48274156])
        >>> rmse_matrices(C, D, element_wise=True)  # RMSE element-wise along first dimension
        array([[0.22022923, 0.3886001 ],
               [0.46130561, 0.2407136 ],
               [0.38114819, 0.40178899]])
        >>> rmse_matrices(C[0], D[0], element_wise=True)
        0.3430603410873006
        >>> rmse_matrices(C[0], D[0])
        0.3430603410873006
    """
    A = np.copy(A)
    B = np.copy(B)
    if A.shape != B.shape:
        raise ValueError("Both arrays must have the same shape.")
    if A.ndim == 2:
        return np.sqrt(np.mean((A - B)**2))
    if element_wise:
        return np.sqrt(np.mean((A - B)**2, axis=0))
    return np.sqrt(np.mean(np.mean((A - B)**2, axis=2), axis=1))
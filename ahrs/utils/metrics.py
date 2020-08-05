# -*- coding: utf-8 -*-
"""
Metrics
=======

References
----------
.. [1] Huynh, D.Q. Metrics for 3D Rotations: Comparison and Analysis. J Math
    Imaging Vis 35, 155-164 (2009).
.. [2] Kuffner, J.J. Effective Sampling and Distance Metrics for 3D Rigid Body
    Path Planning. IEEE International Conference on Robotics and Automation
    (ICRA 2004)
.. [3] R. Hartley, J. Trumpf, Y. Dai, H. Li. Rotation Averaging. International
    Journal of Computer Vision. Volume 101, Number 2. 2013.

"""

import numpy as np
from ..common.orientation import logR

def euclidean(x, y, **kwargs):
    """Euclidean distance between two arrays.

    Accepts the same parameters as the function `numpy.linalg.norm()`.

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
    return np.linalg.norm(x-y, **kwargs)

def chordal(R1, R2):
    """Chordal Distance

    Euclidean distance `d` between two rotations `R1` and `R2` in SO(3):

    .. math::

        d(R_1, R_2) = \\|R_1-R_2\\|_F

    where :math:`\\|X\\|_F` represents the Frobenius norm of the matrix :math:`X`
    """
    return np.linalg.norm(R1-R2, 'fro')

def identity_deviation(R1, R2):
    """Deviation from Identity Matrix

    Error ranges: [0, 2*sqrt(2)]
    """
    return np.linalg.norm(np.eye(3)-R1@R2.T, 'fro')

def angular_distance(R1, R2):
    """Angular distance between two rotations `R1` and `R2` in SO(3)
    """
    if R1.shape!=R2.shape:
        raise ValueError("Cannot compare R1 of shape {} and R2 of shape {}".format(R1.shape, R2.shape))
    return np.linalg.norm(logR(R1@R2.T))

def qdist(q1, q2):
    """
    Error ranges: [0, sqrt(2)]
    """
    if q1.shape!=q2.shape:
        raise ValueError("Cannot compare q1 of shape {} and q2 of shape {}".format(q1.shape, q2.shape))
    if q1.ndim==1 and q2.ndim==1:
        if np.allclose(q1, q2) or np.allclose(-q1, q2):
            return 0.0
        return min(np.linalg.norm(q1-q2), np.linalg.norm(q1+q2))
    return np.r_[[np.linalg.norm(qa-qb, axis=1)], [np.linalg.norm(qa+qb, axis=1)]].min(axis=0)

def qcip(q1, q2):
    """
    Error ranges: [0, pi/2]
    """
    if q1.shape!=q2.shape:
        raise ValueError("Cannot compare q1 of shape {} and q2 of shape {}".format(q1.shape, q2.shape))
    if q1.ndim==1 and q2.ndim==1:
        if np.allclose(q1, q2) or np.allclose(-q1, q2):
            return 0.0
        return np.arccos(abs(q1@q2))
    return np.arccos(abs(np.nansum(q1*q2, axis=1)))

def qeip(q1, q2):
    """
    Error ranges: [0, 1]
    """
    if q1.shape!=q2.shape:
        raise ValueError("Cannot compare q1 of shape {} and q2 of shape {}".format(q1.shape, q2.shape))
    if q1.ndim==1 and q2.ndim==1:
        if np.allclose(q1, q2) or np.allclose(-q1, q2):
            return 0.0
        return 1.0-abs(q1@q2)
    return 1.0-abs(np.nansum(q1*q2, axis=1))

def qad(q1, q2):
    """Quaternion Angle Difference
    """
    if q1.ndim==1 and q2.ndim==1:
        if np.allclose(q1, q2) or np.allclose(-q1, q2):
            return 0.0
        return np.arccos(2.0*(q1@q2)**2-1.0)
    return np.arccos(2.0*np.nansum(q1*q2, axis=1)**2-1.0)

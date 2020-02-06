#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Metrics routines

"""

import numpy as np

def euclidean(x, y, **kwargs):
    """
    Computes a simple euclidean distance between two arrays.

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

    Simulating 5 samples of a Euler Angles compared against samples with noise.

    >>> num_samples = 5
    >>> angles = np.random.uniform(low=-180.0, high=180.0, size=(num_samples, 3))
    >>> noisy = angles + np.random.randn(num_samples, 3)
    >>> ahrs.utils.euclidean(angles, noisy)
    2.585672169476804
    >>> ahrs.utils.euclidean(angles, noisy, axis=0)
    array([1.36319772, 1.78554071, 1.28032688])
    >>> ahrs.utils.euclidean(angles, noisy, axis=1)     # distance per sample
    array([0.88956871, 1.19727356, 1.5243858 , 0.68765523, 1.29007067])

    """
    return np.linalg.norm(x-y, **kwargs)

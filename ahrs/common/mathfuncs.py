# -*- coding: utf-8 -*-
"""
Common mathematical routines
============================

These functions can be used at different scripts and submodules.

"""

import numpy as np
from .constants import DEG2RAD

def cosd(x):
    """
    Return the cosine of `x`, which is expressed in degrees.

    If `x` is a list, it will be converted first to a NumPy array, and then the
    cosine operation over each value will be carried out.

    Parameters
    ----------
    x : float or array-like
        Angle in Degrees.

    Returns
    -------
    y : float or numpy.ndarray
        Cosine of given angle.

    Examples
    --------
    >>> from ahrs.common.mathfuncs import cosd
    >>> cosd(0.0)
    1.0
    >>> cosd(90.0)
    0.0
    >>> cosd(-120.0)
    -0.5

    """
    if isinstance(x, (list, tuple, np.ndarray)):
        x = np.copy(x)
    return np.cos(x*DEG2RAD)

def sind(x):
    """
    Return the sine of `x`, which is expressed in degrees.

    If `x` is a list, it will be converted first to a NumPy array, and then the
    sine operation over each value will be carried out.

    Parameters
    ----------
    x : float or array-like
        Angle in Degrees.

    Returns
    -------
    y : float or numpy.ndarray
        Sine of given angle.

    Examples
    --------
    >>> from ahrs.common.mathfuncs import sind
    >>> sind(0.0)
    0.0
    >>> sind(90.0)
    1.0
    >>> sind(-120.0)
    -0.86602540378

    """
    if isinstance(x, (list, tuple, np.ndarray)):
        x = np.copy(x)
    return np.sin(x*DEG2RAD)

def skew(x):
    """
    Return the 3-by-3 skew-symmetric matrix :cite:p:`Wiki_skew` of a 3-element
    vector ``x``.

    Parameters
    ----------
    x : array-like
        3-element array with values to be ordered in a skew-symmetric matrix.

    Returns
    -------
    X : numpy.ndarray
        3-by-3 numpy array of the skew-symmetric matrix.

    Examples
    --------
    >>> from ahrs.common.mathfuncs import skew
    >>> a = [1, 2, 3]
    >>> skew(a)
    [[ 0. -3.  2.]
     [ 3.  0. -1.]
     [-2.  1.  0.]]
    >>> a = np.array([[4.0], [5.0], [6.0]])
    >>> skew(a)
    [[ 0. -6.  5.]
     [ 6.  0. -4.]
     [-5.  4.  0.]]

    """
    if isinstance(x, (list, tuple, np.ndarray)):
        x = np.copy(x)
    if len(x) != 3:
        raise ValueError("Input must be an array with three elements")
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0.0]])

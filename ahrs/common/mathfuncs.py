# -*- coding: utf-8 -*-
"""
Common mathematical routines.

"""

import numpy as np

# __all__ = ['M_PI', 'DEG2RAD', 'RAD2DEG', 'cosd', 'sind', 'skew']
__all__ = ['cosd', 'sind', 'skew']

M_PI = np.pi
DEG2RAD = M_PI / 180.0
RAD2DEG = 180.0 / M_PI

def cosd(x):
    """
    Return the cosine of `x`, which is expressed in degrees.

    If `x` is a list, it will be converted first to a NumPy array, and then the
    cosine operation over each value will be carried out.

    Parameters
    ----------
    x : float
        Angle in Degrees

    Returns
    -------
    y : float
        Cosine of given angle

    Examples
    --------
    >>> from ahrs.common.mathfuncs import *
    >>> cosd(0.0)
    1.0
    >>> cosd(90.0)
    0.0
    >>> cosd(-120.0)
    -0.5

    """
    if type(x) == list:
        x = np.asarray(x)
    return np.cos(x*DEG2RAD)

def sind(x):
    """
    Return the sine of `x`, which is expressed in degrees.

    If `x` is a list, it will be converted first to a NumPy array, and then the
    sine operation over each value will be carried out.

    Parameters
    ----------
    x : float
        Angle in Degrees

    Returns
    -------
    y : float
        Sine of given angle

    Examples
    --------
    >>> from ahrs.common.mathfuncs import *
    >>> sind(0.0)
    0.0
    >>> sind(90.0)
    1.0
    >>> sind(-120.0)
    -0.86602540378

    """
    if type(x) == list:
        x = np.asarray(x)
    return np.sin(x*DEG2RAD)

def skew(x):
    """
    Return the 3-by-3 skew-symmetric matrix [Wiki_skew]_ of a 3-element vector x.

    Parameters
    ----------
    x : array
        3-element array with values to be ordered in a skew-symmetric matrix.

    Returns
    -------
    X : ndarray
        3-by-3 numpy array of the skew-symmetric matrix.

    Examples
    --------
    >>> from ahrs.common.mathfuncs import skew_matrix
    >>> a = [1, 2, 3]
    >>> skew_matrix(a)
    [[ 0. -3.  2.]
     [ 3.  0. -1.]
     [-2.  1.  0.]]
    >>> a = np.array([[4.0], [5.0], [6.0]])
    >>> skew_matrix(a)
    [[ 0. -6.  5.]
     [ 6.  0. -4.]
     [-5.  4.  0.]]

    References
    ----------
    .. [Wiki_skew] https://en.wikipedia.org/wiki/Skew-symmetric_matrix

    """
    if len(x) != 3:
        return None
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0.0]])

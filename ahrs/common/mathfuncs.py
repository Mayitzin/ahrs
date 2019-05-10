# -*- coding: utf-8 -*-
"""
Common mathematical routines.

"""

import numpy as np

__all__ = ['M_PI', 'DEG2RAD', 'RAD2DEG', 'cosd', 'sind']

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
    >>> from protoboard.common.mathfuncs import *
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
    >>> from protoboard.common.mathfuncs import *
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

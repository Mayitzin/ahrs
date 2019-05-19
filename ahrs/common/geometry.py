# -*- coding: utf-8 -*-
"""
Geometrical routines

"""

import numpy as np

def circle(center, radius=1.0, num_points=20):
    """
    Build a circle with the given characteristics.

    Parameters
    ----------
    c : array
        2D Coordinates of center.
    r : float
        Radius of the circle.
    num_points : int
        Number of points.

    Returns
    -------
    points : array
        N-by-2 array with the coordinates of the circle.

    References
    ----------
    .. [WAC] Wolfram Alpha: Circle. (http://mathworld.wolfram.com/Circle.html)
    """
    R = np.linspace(0.0, 2.0*np.pi, num_points+1)
    x = center[0] + radius*np.cos(R)
    y = center[1] + radius*np.sin(R)
    return np.array([x, y]).transpose()

def ellipse(center, phi, axes, num_points=20):
    """
    Build an ellipse with the given characteristics.

    Parameters
    ----------
    center : array
        2D Coordinates of center.
    phi : float
        Angle, in radians, of the major axis w.r.t. the X-axis
    axes : array
        Lengths of major and minor axes, respectively.
    num_points : int
        Number of points. Defaults to 20.

    Returns
    -------
    points : array
        N-by-2 array with the coordinates of the ellipse.

    References
    ----------
    .. [W1] Wikipedia: https://de.wikipedia.org/wiki/Ellipse#Ellipsengleichung_(Parameterform)
    .. [WAE] Wolfram Alpha: Ellipse. (http://mathworld.wolfram.com/Ellipse.html)

    """
    R = np.linspace(0.0, 2.0*np.pi, num_points+1)
    a, b = axes
    x = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
    y = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
    return np.array([x, y]).transpose()

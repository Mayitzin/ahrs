# -*- coding: utf-8 -*-
"""
Reference Frames
================

Coordinate frames express the position of an object in relation to a reference.
There are 4 main frames:

- **Earth-Centered Inertial Frame** (ECI), also noted as **i-frame**, has its
  origin at the center of mass of the Earth. The X-axis points towards the
  `vernal equinox <https://en.wikipedia.org/wiki/March_equinox>`_ in the
  equatorial plane. The Z-axis is along the rotation axis of the Earth. The
  Y-axis completes with a right-hand system.
- **Earth-Centered Earth-Fixed Frame** (ECEF), also noted as **e-frame**, has
  its origin and Z-axis aligned to the i-frame, but rotates along with the
  Earth. Therefore, is Earth-Fixed.
- **Local-Level Frame** (LLF), also noted as **l-frame**, is the local
  navigation frame, whose origin coincides with the sensor frame.

References
----------
.. [Noureldin] Aboelmagd Noureldin, Tashfeen B. Karamat, Jacques Georgy.
    Fundamentals of Inertial Navigation, Satellite-based Position and their
    Integration. Springer-Verlag Berlin Heidelberg. 2013.
.. [WGS84] World Geodetic System 1984. Its Definition and Relationships with
    Local Geodetic Systems. National Geospatial-Intelligence Agency (NGA)
    Standarization Document. 2014.
    (ftp://ftp.nga.mil/pub2/gandg/website/wgs84/NGA.STND.0036_1.0.0_WGS84.pdf)

"""

import numpy as np
from .constants import *

def geo2rect(lon: float, lat: float, h: float, r: float, ecc: float = EARTH_SECOND_ECCENTRICITY_2) -> np.ndarray:
    """Geodetic to Rectangular Coordinates conversion in the e-frame.

    Parameters
    ----------
    lon : float
        Longitude
    lat : float
        Latitude
    h : float
        Height above ellipsoidal surface
    r : float
        Normal radius
    ecc : float, default: 6.739496742276486e-3
        Ellipsoid's second eccentricity squared. Defaults to Earth's.

    Returns
    -------
    X : numpy.ndarray
        ECEF rectangular coordinates
    """
    X = np.zeros(3)
    X[0] = (r+h)*np.cos(lat)*np.cos(lon)
    X[1] = (r+h)*np.cos(lat)*np.sin(lon)
    X[2] = (r*(1.0-ecc)+h)*np.sin(lat)
    return X

def rec2geo(X: np.ndarray, ecc: float = EARTH_SECOND_ECCENTRICITY_2) -> np.ndarray:
    """Rectangular to Geodetic Coordinates conversion in the e-frame.

    Parameters
    ----------
    X : numpy.ndarray
        Rectangular coordinates in the e-frame.
    ecc : float, default: 6.739496742276486e-3
        Ellipsoid's second eccentricity squared. Defaults to Earth's.
    """
    x, y, z = X
    p = np.linalg.norm([x, y])
    theta = np.arctan(z*a/(p*b))
    lon = 2*np.arctan(y/(x+p))
    lat = np.arctan((z+ecc*b*np.sin(theta)**3)/(p-e*a*np.cos(theta)**3))
    N = a**2/np.sqrt(a**2*np.cos(lat)**2 + b**2*np.sin(lat)**2)
    h = p/np.cos(lat) - N
    return np.array([lon, lat, h])

def llf2ecef(lat, lon):
    """Transform coordinates from LLF to ECEF

    Parameters
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.

    Returns
    -------
    R : np.ndarray
        Direction Cosine Matrix.
    """
    return np.array([
        [-np.sin(lat), -np.sin(lon)*np.cos(lat), np.cos(lon)*np.cos(lat)],
        [ np.cos(lat), -np.sin(lon)*np.sin(lat), np.cos(lon)*np.sin(lat)],
        [0.0, np.cos(lon), np.sin(lon)]])

def ecef2llf(lat, lon):
    """Transform coordinates from ECEF to LLF

    Parameters
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.

    Returns
    -------
    R : np.ndarray
        Direction Cosine Matrix.
    """
    return np.array([
        [-np.sin(lat), np.cos(lat), 0.0],
        [-np.sin(lon)*np.cos(lat), -np.sin(lon)*np.sin(lat), np.cos(lon)],
        [np.cos(lon)*np.cos(lat), np.cos(lon)*np.sin(lat), np.sin(lon)]])

def eci2ecef(w, t=0):
    """Transformation between ECI and ECEF

    Parameters
    ----------
    w : float
        Rotation rate in rad/s
    t : float, default: 0.0
        Time since reference epoch.
    """
    return np.array([
        [ np.cos(w)*t, np.sin(w)*t, 0.0],
        [-np.sin(w)*t, np.cos(w)*t, 0.0],
        [         0.0,         0.0, 1.0]])


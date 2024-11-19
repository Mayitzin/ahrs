# -*- coding: utf-8 -*-
"""
Reference Frames
================

A frame of reference specifies the position of an object in relation to a
reference within a coordinate system.

In the context of global navigation, the reference frame is used to define the
position of an object in relation to the Earth. The Earth is a non-inertial
frame, and the object is usually a sensor or a vehicle.

In this case, the sensor frame is usually attached to the object, and the
navigation frame is attached to the Earth.

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

"""

import numpy as np
from .constants import EARTH_FIRST_ECCENTRICITY
from .constants import EARTH_SECOND_ECCENTRICITY_2
from .constants import EARTH_EQUATOR_RADIUS
from .constants import EARTH_POLAR_RADIUS
from .constants import EARTH_FLATTENING
from .constants import RAD2DEG
from .constants import DEG2RAD

def geo2rect(lat: float, lon: float, h: float, a: float = EARTH_EQUATOR_RADIUS, ecc: float = EARTH_FIRST_ECCENTRICITY) -> np.ndarray:
    """
    Transform geodetic coordinates to Rectangular (Cartesian) Coordinates in
    the Earth-Centered Earth-Fixed frame.

    The cartesian coordinates of a point :math:`\\begin{pmatrix}x & y & z\\end{pmatrix}`
    can be calculated from the geodetic coordinates
    :math:`\\begin{pmatrix}\\phi & \\lambda & h\\end{pmatrix}` using the
    following equations :cite:p:`Wiki_GCC_geo2rect` :cite:p:`ESA_Coord_Conv`
    :cite:p:`noureldin2013`:

    .. math::

        \\begin{array}{rcl}
        x & = & (N + h) \\cos\\phi \\cos\\lambda \\\\
        y & = & (N + h) \\cos\\phi \\sin\\lambda \\\\
        z & = & \\big(\\left(1 - e^2\\right)N + h\\big) \\sin\\phi
        \\end{array}

    where :math:`\\phi` is the latitude, :math:`\\lambda` is the longitude,
    :math:`h` is the height, and :math:`N` is the radius of curvature in the
    prime vertical at the given latitude :math:`\\phi`:

    .. math::

        r = \\frac{a}{\\sqrt{1 - e^2 \\sin^2\\phi}}

    The first eccentricity of the ellipsoid **squared**, :math:`e^2`, is
    defined using the equatorial radius of the ellipsoid :math:`a`, and the
    polar radius of the ellipsoid :math:`b`:

    .. math::

        e^2 = \\frac{a^2-b^2}{a^2}

    These values default to Earth's ellipsoid.

    Parameters
    ----------
    lat : float
        Latitude, in degrees.
    lon : float
        Longitude, in degrees.
    h : float
        Height above ellipsoidal surface, in meters.
    a : float, default: 6378137.0
        Ellipsoid's equatorial radius (semi-major axis), in meters. Defaults to
        Earth's.
    ecc : float, default: 8.1819190842622e-2
        Ellipsoid's first eccentricity. Defaults to Earth's.

    Returns
    -------
    X : numpy.ndarray
        ECEF rectangular coordinates
    """
    if abs(lat) > 90.0:
        raise ValueError(f"Latitude must be between -90 and 90 degrees. Got {lat}")
    if abs(lon) > 180.0:
        raise ValueError(f"Longitude must be between -180 and 180 degrees. Got {lon}")
    lat *= DEG2RAD
    lon *= DEG2RAD
    N = a/np.sqrt(1 - ecc**2 *np.sin(lat)**2)
    X = np.zeros(3)
    X[0] = (N+h)*np.cos(lat)*np.cos(lon)
    X[1] = (N+h)*np.cos(lat)*np.sin(lon)
    X[2] = (N*(1.0-ecc**2)+h)*np.sin(lat)
    return X

def rec2geo(X: np.ndarray, a: float = EARTH_EQUATOR_RADIUS, b: float = EARTH_POLAR_RADIUS) -> np.ndarray:
    """
    Transform Rectangular (Cartesian) coordinates in the e-frame to Geodetic
    Coordinates :cite:p:`ESA_Coord_Conv`.

    First, the easiest is to compute the geodetic longitude :math:`\\lambda`:

    .. math::

        \\lambda = \\mathrm{arctan2}\\left(x, \\,y\\right)

    Then, we can iteratively compute the geodetic latitude :math:`\\phi`, and
    height :math:`h`. We start with an initial estimation of the latitude:

    .. math::

        \\phi_0 = \\mathrm{arctan}\\Bigg(\\frac{z}{(1-e^2)p}\\Bigg)

    with :math:`p = \\sqrt{x^2 + y^2}`.

    Then, we iterate until the change between two consecutive latitudes
    (:math:`\\phi_i` and :math:`\\phi_{i-1}`) is smaller than a given threshold
    :math:`\\delta`:

    .. math::

        \\begin{array}{rcl}
        N & = & \\frac{a}{\\sqrt{1 - e^2 \\sin^2(\\phi_{i-1})}} \\\\
        h & = & \\frac{p}{\\cos\\phi_{i-1}} - N \\\\
        \\phi_i & = & \\mathrm{arctan}\\Bigg(\\frac{z}{(1-e^2\\frac{N}{N+h})p}\\Bigg)
        \\end{array}

    where :math:`N` is the radius of curvature in the vertical prime, and
    :math:`e^2` is the square of the first eccentricity of the ellipsoid.

    The value of :math:`\\delta` is empirically found to perform well when set
    to :math:`10^{-8}` in this implementation.

    The final latitude and longitude are converted to degrees. The height is
    returned in meters.

    Parameters
    ----------
    X : numpy.ndarray
        Rectangular coordinates in the e-frame.
    a : float, default: 6378137.0
        Ellipsoid's equatorial radius, in meters. Defaults to Earth's.
    b : float, default: 6356752.3142
        Ellipsoid's polar radius, in meters. Defaults to Earth's.

    Returns
    -------
    lla : numpy.ndarray
        Geodetic coordinates [latitude, longitude, altitude].
    """
    x, y, z = X
    e2 = (a**2 - b**2)/a**2   # Square of the first eccentricity: f * (2 - f) = e^2
    p = np.sqrt(x**2 + y**2)
    lon = np.arctan2(y, x)
    # Iteratively compute latitude and height
    delta = 1e-8
    h = lat_old = 0
    lat = np.arctan(z / ((1-e2)*p))
    while abs(lat_old - lat) > delta:
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)    # Radius of curvature in the vertical prime
        h = p / np.cos(lat) - N
        lat_old = lat
        lat = np.arctan(z / ((1-e2*N/(N+h))*p))
    lat *= RAD2DEG
    lon *= RAD2DEG
    return np.array([lat, lon, h])

def llf2ecef(lat: float, lon: float) -> np.ndarray:
    """
    Transform coordinates from LLF to ECEF

    Parameters
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.

    Returns
    -------
    R : np.ndarray
        Rotation Matrix.
    """
    return np.array([
        [-np.sin(lat), -np.sin(lon)*np.cos(lat), np.cos(lon)*np.cos(lat)],
        [ np.cos(lat), -np.sin(lon)*np.sin(lat), np.cos(lon)*np.sin(lat)],
        [0.0, np.cos(lon), np.sin(lon)]])

def ecef2llf(lat: float, lon: float) -> np.ndarray:
    """
    Transform coordinates from ECEF to LLF

    Parameters
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.

    Returns
    -------
    R : np.ndarray
        Rotation Matrix.
    """
    return np.array([
        [-np.sin(lat), np.cos(lat), 0.0],
        [-np.sin(lon)*np.cos(lat), -np.sin(lon)*np.sin(lat), np.cos(lon)],
        [np.cos(lon)*np.cos(lat), np.cos(lon)*np.sin(lat), np.sin(lon)]])

def ecef2lla(ecef : np.ndarray, f: float = EARTH_FLATTENING, a: float = EARTH_EQUATOR_RADIUS) -> np.ndarray:
    """
    Calculate geodetic latitude, longitude, and altitude above planetary
    ellipsoid from a given Earth-centered Earth-fixed (ECEF) position, using
    Bowring's method.

    It defaults to `WGS84 <https://ahrs.readthedocs.io/en/latest/wgs84.html>`_
    ellipsoid parameters.

    Parameters
    ----------
    ecef : numpy.ndarray
        ECEF coordinates.
    f : float, default: 1/298.257223563
        Flattening of the ellipsoid.
    a : float, default: 6378137.0
        Equatorial radius of the ellipsoid.

    Returns
    -------
    lla : np.ndarray
        Geodetic coordinates [longitude, latitude, altitude].
    """
    e2 = f * (2 - f)  # Square of the first eccentricity: f * (2 - f) = e^2
    x, y, z = ecef

    # Compute longitude
    lon = np.arctan2(y, x)

    # Compute initial latitude approximation
    p = np.sqrt(x**2 + y**2)
    beta = np.arctan2(z, (1 - f) * p)
    lat = np.arctan2(z + e2 * (1 - f) * a * np.sin(beta)**3, p - e2 * a * np.cos(beta)**3)
    # Iteratively improve latitude
    for _ in range(5):  # Usually converges in a few iterations
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)    # Radius of curvature in the vertical prime
        lat = np.arctan2(z + e2 * N * sin_lat, p)

    # Compute altitude
    alt = p / np.cos(lat) - N

    # Convert radians to degrees for latitude and longitude
    lat *= RAD2DEG
    lon *= RAD2DEG

    return np.array([lat, lon, alt])

def eci2ecef(w: float, t: float = 0) -> np.ndarray:
    """
    Transformation between ECI and ECEF

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

def ecef2enu(lat: float, lon: float) -> np.ndarray:
    """
    Transform coordinates from ECEF to ENU

    Parameters
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.

    Returns
    -------
    R : np.ndarray
        Rotation Matrix.
    """
    return np.array([
        [-np.sin(lon), np.cos(lon), 0.0],
        [-np.cos(lat)*np.cos(lon), -np.cos(lat)*np.sin(lon), np.sin(lat)],
        [np.sin(lat)*np.cos(lon), np.sin(lat)*np.sin(lon), np.cos(lat)]])

def enu2ecef(lat: float, lon: float) -> np.ndarray:
    """
    Transform coordinates from ENU to ECEF

    Parameters
    ----------
    lat : float
        Latitude.
    lon : float
        Longitude.

    Returns
    -------
    R : np.ndarray
        Rotation Matrix.
    """
    return np.array([
        [-np.sin(lon), -np.cos(lat)*np.cos(lon), np.sin(lat)*np.cos(lon)],
        [np.cos(lon), -np.cos(lat)*np.sin(lon), np.sin(lat)*np.sin(lon)],
        [0.0, np.sin(lat), np.cos(lat)]])

def _ltp_transformation(x: np.ndarray) -> np.ndarray:
    """
    Transform coordinates between NED and ENU.
    """
    x = np.copy(x)
    if x.shape[-1] != 3 or x.ndim > 2:
        raise ValueError(f"Given coordinates must have form (3, ) or (N, 3). Got {x.shape}")
    A = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    if x.ndim > 1:
        return (A @ x.T).T
    return A @ x

def ned2enu(x: np.ndarray) -> np.ndarray:
    """
    Transform coordinates from NED to ENU.

    Parameters
    ----------
    x : np.ndarray
        3D coordinates of point(s) to project.

    Returns
    -------
    x' : np.ndarray
        Transformed coordinates.
    """
    return _ltp_transformation(x)

def enu2ned(x: np.ndarray) -> np.ndarray:
    """
    Transform coordinates from ENU to NED.

    Parameters
    ----------
    x : np.ndarray
        3D coordinates of point(s) to project.

    Returns
    -------
    x' : np.ndarray
        Transformed coordinates.
    """
    return _ltp_transformation(x)

# -*- coding: utf-8 -*-
"""
A frame of reference specifies the position of an object in relation to a
reference within a coordinate system.

In the context of global navigation, the reference frame is used to define the
position of an object in relation to the Earth. The Earth is a non-inertial
frame, and the object is usually a sensor or a vehicle.

In this case, the sensor frame is usually attached to the object, and the
navigation frame is attached to the Earth.

There main frames are:

- **Earth-Centered Inertial Frame** (ECI), also noted as **i-frame**, has its
  origin at the center of mass of the Earth. The X-axis points towards the
  `vernal equinox <https://en.wikipedia.org/wiki/March_equinox>`_ in the
  equatorial plane. The Z-axis is along the rotation axis of the Earth. The
  Y-axis completes with a right-hand system. This is also sometimes known as
  the `Celestial Reference System <https://gssc.esa.int/navipedia/index.php/Conventional_Celestial_Reference_System)>`_
  (CRS.)
- **Earth-Centered Earth-Fixed Frame** (ECEF), also noted as **e-frame**, has
  its origin and Z-axis aligned to the i-frame, but rotates along with the
  Earth. Therefore, is Earth-Fixed.
- The **North-East-Down** (NED) system has its origin fixed at the center of
  gravity of the aircraft. It is defined relative to a specific location on
  Earth, described by latitude, longitude and altitude.
- The local **East-North-Up** (ENU) coordinates are formed from a plane tangent
  to the Earth's surface fixed to a specific location and hence it is sometimes
  known as a `Local Tangent <https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates>`_
  or "Local Geodetic" plane.
- An **Azimuth-Elevation-Range** (AER) system uses the `spherical coordinates
  <https://en.wikipedia.org/wiki/Spherical_coordinate_system>`_ (az, elev,
  range) to represent position relative to a local origin. The local origin is
  described by the geodetic coordinates (latitude, longitude, height). Azimuth,
  elevation, and slant range are dependent on a local Cartesian system, for
  example, an ENU system.
"""

import numpy as np
from .constants import EARTH_FIRST_ECCENTRICITY
from .constants import EARTH_EQUATOR_RADIUS
from .constants import EARTH_POLAR_RADIUS
from .constants import RAD2DEG
from .constants import DEG2RAD

def geodetic2ecef(lat: float, lon: float, h: float, a: float = EARTH_EQUATOR_RADIUS, ecc: float = EARTH_FIRST_ECCENTRICITY) -> np.ndarray:
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
        ECEF cartesian coordinates.
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

def geodetic2enu(lat: float, lon: float, h: float, lat0: float, lon0: float, h0: float, a: float = EARTH_EQUATOR_RADIUS, ecc: float = EARTH_FIRST_ECCENTRICITY) -> np.ndarray:
    """
    Transform geodetic coordinates to east-north-up (ENU) coordinates
    :cite:p:`noureldin2013`.

    Transform local geodetic coordinates :math:`\\begin{pmatrix}\\phi &
    \\lambda & h\\end{pmatrix}` to local East-North-Up (ENU) coordinates
    :math:`\\begin{pmatrix}x & y & z\\end{pmatrix}`.

    The origin of the local ENU frame has to be defined by the geodetic
    coordinates :math:`\\begin{pmatrix}\\phi_0 & \\lambda_0 & h_0\\end{pmatrix}`.

    The transformation is performed in two steps:

    1. Convert both geodetic coordinates to ECEF coordinates.

    .. math::

        \\begin{array}{rcl}
        x & = & (N + h) \\cos\\phi \\cos\\lambda \\\\
        y & = & (N + h) \\cos\\phi \\sin\\lambda \\\\
        z & = & \\big(\\left(1 - e^2\\right)N + h\\big) \\sin\\phi
        \\end{array}

    .. math::

        \\begin{array}{rcl}
        x_0 & = & (N_0 + h_0) \\cos\\phi_0 \\cos\\lambda_0 \\\\
        y_0 & = & (N_0 + h_0) \\cos\\phi_0 \\sin\\lambda_0 \\\\
        z_0 & = & \\big(\\left(1 - e^2\\right)N_0 + h_0\\big) \\sin\\phi_0
        \\end{array}

    2. Convert the difference between the two ECEF coordinates to ENU.

    where :math:`N_1` and :math:`N_2` are the radius of curvature in the prime
    vertical at the given latitude :math:`\\phi_1` and :math:`\\phi_2`,
    respectively, and :math:`e^2` is the square of the first eccentricity of
    the ellipsoid.

    The ENU coordinates are computed as follows:

    .. math::

        \\begin{array}{rcl}
        x_{\\mathrm{ENU}} & = & -\\sin\\lambda_0 \\, (x_0 - x) + \\cos\\lambda_0 \\, (y_0 - y) \\\\
        y_{\\mathrm{ENU}} & = & -\\sin\\phi_0 \\, \\cos\\lambda_0 \\, (x_0 - x) - \\sin\\phi_0 \\, \\sin\\lambda_0 \\, (y_0 - y) + \\cos\\phi_0 \\, (z_0 - z) \\\\
        z_{\\mathrm{ENU}} & = & \\cos\\phi_0 \\, \\cos\\lambda_0 \\, (x_0 - x) + \\cos\\phi_0 \\, \\sin\\lambda_0 \\, (y_0 - y) + \\sin\\phi_0 \\, (z_0 - z)
        \\end{array}

    Parameters
    ----------
    lat : float
        Latitude of local origin, in degrees.
    lon : float
        Longitude of local origin, in degrees.
    h : float
        Height above ellipsoidal surface of local origin, in meters.
    lat0 : float
        Latitude of point of interesr, in degrees.
    lon0 : float
        Longitude of point of interest, in degrees.
    h0 : float
        Height above ellipsoidal surface of point of interest, in meters.
    a : float, default: 6378137.0
        Ellipsoid's equatorial radius (semi-major axis), in meters. Defaults to
        Earth's.
    ecc : float, default: 8.1819190842622e-2
        Ellipsoid's first eccentricity. Defaults to Earth's.

    Returns
    -------
    enu : numpy.ndarray
        ENU cartesian coordinates [east, north, up].

    The ENU coordinates are computed as follows:
    """
    x1, y1, z1 = geodetic2ecef(lat, lon, h, a, ecc)
    x2, y2, z2 = geodetic2ecef(lat0, lon0, h0, a, ecc)
    return ecef2enuv(x1, y1, z1, x2, y2, z2, lat0, lon0)

def ecef2geodetic(x: float, y: float, z: float, a: float = EARTH_EQUATOR_RADIUS, b: float = EARTH_POLAR_RADIUS) -> np.ndarray:
    """
    Transform cartesian coordinates in ECEF-frame to Geodetic Coordinates
    :cite:p:`ESA_Coord_Conv`.

    Given the Cartesian coordinates :math:`\\begin{pmatrix}x & y & z\\end{pmatrix}`
    in the Earth-Centered Earth-Fixed (ECEF) frame, we can start by computing
    the geodetic longitude :math:`\\lambda`:

    .. math::

        \\lambda = \\mathrm{arctan2}\\left(x, \\,y\\right)

    Then, we iteratively compute the geodetic latitude :math:`\\phi` using the
    initial estimation:

    .. math::

        \\phi_0 = \\mathrm{arctan2}\\big(z, \\,(1-e^2)p \\big)

    with :math:`p = \\sqrt{x^2 + y^2}`.

    Now we iterate until the difference between two consecutive latitudes
    (:math:`\\phi_i` and :math:`\\phi_{i-1}`) is smaller than a given threshold
    :math:`\\delta`. Each iteration updates the values as follows:

    .. math::

        \\begin{array}{rcl}
        N & \\leftarrow & \\frac{a}{\\sqrt{1 - e^2 \\sin^2(\\phi_{i-1})}} \\\\
        \\phi_i & \\leftarrow & \\mathrm{arctan2}\\big(z+e^2N\\sin(\\phi_{i-1}), \\,p\\big)
        \\end{array}

    where :math:`N` is the radius of curvature in the vertical prime, and
    :math:`e^2` is the square of the first eccentricity of the ellipsoid.

    .. math::

        e^2 = \\frac{a^2-b^2}{a^2}

    The value of :math:`\\delta` is empirically found to perform well when set
    to :math:`10^{-8}` in this implementation.

    The altitude (height) :math:`h` is computed as:

    .. math::

        h = \\frac{p}{\\cos\\phi} - N

    .. note::

        The altitude :math:`h` has an accuracy up to 0.1 m (10 cm), which is
        enough for most applications.

    The latitude and longitude are returned in degrees. The altitude is
    returned in meters.

    Parameters
    ----------
    x : float
        ECEF x-coordinate, in meters.
    y : float
        ECEF y-coordinate, in meters.
    z : float
        ECEF z-coordinate, in meters.
    a : float, default: 6378137.0
        Ellipsoid's equatorial radius, in meters. Defaults to Earth's.
    b : float, default: 6356752.3142
        Ellipsoid's polar radius, in meters. Defaults to Earth's.

    Returns
    -------
    lla : numpy.ndarray
        Geodetic coordinates [latitude, longitude, altitude].

    Examples
    --------
    >>> from ahrs.common.frames import ecef2geodetic
    >>> x = 4_201_000
    >>> y = 172_460
    >>> z = 4_780_100
    >>> ecef2geodetic(x, y, z)
    array([48.85616162,  2.35079383, 67.37006803])
    """
    e2 = (a**2 - b**2)/a**2  # Square of the first eccentricity: 2*f - f^2 = e^2 = (a^2 - b^2)/a^2
    p = np.sqrt(x**2 + y**2)
    lon = np.arctan2(y, x)
    # Iteratively compute latitude
    delta = 1e-8
    lat_old = 0
    lat = np.arctan2(z, (1-e2)*p)
    while abs(lat_old - lat) > delta:
        sin_lat = np.sin(lat)
        N = a / np.sqrt(1 - e2 * sin_lat**2)    # Radius of curvature in the vertical prime
        lat_old = lat
        lat = np.arctan2(z + e2 * N * sin_lat, p)
    h = p / np.cos(lat) - N
    # Convert to degrees
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
    R : numpy.ndarray
        Rotation Matrix.
    """
    return np.array([
        [-np.sin(lat), -np.sin(lon)*np.cos(lat), np.cos(lon)*np.cos(lat)],
        [ np.cos(lat), -np.sin(lon)*np.sin(lat), np.cos(lon)*np.sin(lat)],
        [         0.0,              np.cos(lon),             np.sin(lon)]])

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
    R : numpy.ndarray
        Rotation Matrix.
    """
    return np.array([
        [            -np.sin(lat),              np.cos(lat),         0.0],
        [-np.sin(lon)*np.cos(lat), -np.sin(lon)*np.sin(lat), np.cos(lon)],
        [ np.cos(lon)*np.cos(lat),  np.cos(lon)*np.sin(lat), np.sin(lon)]])

def ecef2lla(x: float, y: float, z: float, a: float = EARTH_EQUATOR_RADIUS, b: float = EARTH_POLAR_RADIUS) -> np.ndarray:
    """Synonym of :func:`ecef2geodetic`."""
    return ecef2geodetic(x, y, z, a, b)

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

def ecef2enu(x: float, y: float, z: float, lat: float, lon: float, h: float, a: float = EARTH_EQUATOR_RADIUS, ecc: float = EARTH_FIRST_ECCENTRICITY) -> np.ndarray:
    """
    Transform geocentric XYZ coordinates in ECEF-frame to Local East-North-Up
    (ENU) cartesian coordinates :cite:p:`noureldin2013`.

    Parameters
    ----------
    x : float
        ECEF x-coordinate, in meters.
    y : float
        ECEF y-coordinate, in meters.
    z : float
        ECEF z-coordinate, in meters.
    lat : float
        Latitude, in degrees.
    lon : float
        Longitude, in degrees.
    h : float
        Height above ellipsoidal surface, in kilometers.
    a : float, default: 6378137.0
        Ellipsoid's equatorial radius (semi-major axis), in meters. Defaults to
        Earth's.
    ecc : float, default: 8.1819190842622e-2
        Ellipsoid's first eccentricity. Defaults to Earth's.

    Returns
    -------
    enu : numpy.ndarray
        ENU cartesian coordinates [east, north, up].

    Examples
    --------
    >>> from ahrs.common.frames import ecef2enu
    >>> x = 660_930.19276
    >>> y = -4_701_424.22296
    >>> z = 4_246_579.60463
    >>> lat = 42.0
    >>> lon = -82.0
    >>> h = 200.0
    >>> ecef2enu(x, y, z, lat, lon, h)
    array([186.27751933, 286.84222383, 939.69262095])
    """
    ecef = geodetic2ecef(lat, lon, h, a, ecc)
    x0, y0, z0 = ecef
    return ecef2enuv(x, y, z, x0, y0, z0, lat, lon)

def ecef2enuv(x: float, y: float, z: float, x0: float, y0: float, z0: float, lat: float, lon: float) -> np.ndarray:
    """
    Transform coordinates from ECEF to ENU

    To carry a transformation from the Earth-Centered Earth-Fixed (ECEF) frame
    to local coordinates we define a reference point :math:`\\mathbf{x}_r=
    \\begin{pmatrix}x_r & y_r & z_r\\end{pmatrix}`, and a point of interest
    :math:`\\mathbf{x}_p=\\begin{pmatrix}x_p & y_p & z_p\\end{pmatrix}`.

    We then transform from ECEF coordinates to the local navigation frame (LLF)
    using the rotation matrix :math:`R_{LLF}` :cite:p:`noureldin2013`
    :cite:p:`Wiki_Geographic_Conversions`:

    .. math::

        R_{LLF} = \\begin{bmatrix}
        -\\sin\\lambda_r & \\cos\\lambda_r & 0 \\\\
        -\\sin\\phi_r\\cos\\lambda_r & -\\sin\\phi_r\\sin\\lambda_r & \\cos\\phi_r \\\\
        \\cos\\phi_r\\cos\\lambda_r & \\cos\\phi_r\\sin\\lambda_r & \\sin\\phi_r
        \\end{bmatrix}

    The LLF-frame is referred to as **ENU** since its axes are aligned with the
    East, North and Up directions.

    The vector pointing from the reference point to the point of interest in
    the ENU frame is given by:

    .. math::

        \\begin{array}{rcl}
        \\mathbf{x}_{\\mathrm{ENU}} & = & R_{LLF} \\, \\mathbf{x}_{\\mathrm{ECEF}} \\\\
        \\begin{bmatrix}x \\\\ y \\\\ z\\end{bmatrix}_{\\mathrm{ENU}}
        & = &
        \\begin{bmatrix} -\\sin\\lambda_r & \\cos\\lambda_r & 0 \\\\
        -\\sin\\phi_r\\cos\\lambda_r & -\\sin\\phi_r\\sin\\lambda_r & \\cos\\phi_r \\\\
        \\cos\\phi_r\\cos\\lambda_r & \\cos\\phi_r\\sin\\lambda_r & \\sin\\phi_r
        \\end{bmatrix}
        \\begin{bmatrix}x_p - x_r \\\\ y_p - y_r \\\\ z_p - z_r\\end{bmatrix}
        \\end{array}

    The final ENU coordinates are:

    .. math::

        \\begin{array}{rcl}
        x_{\\mathrm{ENU}} & = & -\\sin\\lambda_r \\, (x_p - x_r) + \\cos\\lambda_r \\, (y_p - y_r) \\\\
        y_{\\mathrm{ENU}} & = & -\\sin\\phi_r \\, \\cos\\lambda_r \\, (x_p - x_r) - \\sin\\phi_r \\, \\sin\\lambda_r \\, (y_p - y_r) + \\cos\\phi_r \\, (z_p - z_r) \\\\
        z_{\\mathrm{ENU}} & = & \\cos\\phi_r \\, \\cos\\lambda_r \\, (x_p - x_r) + \\cos\\phi_r \\, \\sin\\lambda_r \\, (y_p - y_r) + \\sin\\phi_r \\, (z_p - z_r)
        \\end{array}

    Parameters
    ----------
    x : float
        ECEF x-coordinate, in meters.
    y : float
        ECEF y-coordinate, in meters.
    z : float
        ECEF z-coordinate, in meters.
    x0 : float
        ECEF x-coordinate of reference point, in meters.
    y0 : float
        ECEF y-coordinate of reference point, in meters.
    z0 : float
        ECEF z-coordinate of reference point, in meters.
    lat : float
        Latitude, in degrees.
    lon : float
        Longitude, in degrees.

    Returns
    -------
    enu : numpy.ndarray
        ENU cartesian coordinates [east, north, up].
    """
    lat *= DEG2RAD
    lon *= DEG2RAD
    u = x - x0
    v = y - y0
    w = z - z0
    t     =  np.cos(lon)*u + np.sin(lon)*v
    east  = -np.sin(lon)*u + np.cos(lon)*v
    up    =  np.cos(lat)*t + np.sin(lat)*w
    north = -np.sin(lat)*t + np.cos(lat)*w
    return np.array([east, north, up])

def enu2uvw(east: float, north: float, up: float, lat: float, lon: float, angle_unit: str = 'deg') -> np.ndarray:
    """
    UVW Mapping for ENU coordinates

    Parameters
    ----------
    east : float
        East.
    north : float
        North.
    up : float
        Up.
    lat : float
        Latitude.
    lon : float
        Longitude.
    angle_unit : str, default: 'deg'
        Unit of angles. Can be 'rad' or 'deg'.

    Returns
    -------
    uvw : numpy.ndarray
        UVW cartesian coordinates.
    """
    if angle_unit == 'deg':
        lat *= DEG2RAD
        lon *= DEG2RAD
    t = np.cos(lat) * up - np.sin(lat) * north
    w = np.sin(lat) * up + np.cos(lat) * north
    u = np.cos(lon) * t - np.sin(lon) * east
    v = np.sin(lon) * t + np.cos(lon) * east
    return np.array([u, v, w])

def enu2ecef(east: float, north: float, up: float, lat: float, lon: float, h: float, a: float = EARTH_EQUATOR_RADIUS, ecc: float = EARTH_FIRST_ECCENTRICITY) -> np.ndarray:
    """
    Transforms the local east-north-up (ENU) Cartesian coordinates specified by
    east, north, and up to the geocentric Earth-centered Earth-fixed (ECEF)
    Cartesian coordinates.

    Parameters
    ----------
    east : float
        East.
    north : float
        North.
    up : float
        Up.
    lat : float
        Latitude.
    lon : float
        Longitude.
    h : float
        Height above ellipsoidal surface, in kilometers.
    a : float, default: 6378137.0
        Ellipsoid's equatorial radius (semi-major axis), in meters. Defaults to
        Earth's.
    ecc : float, default: 8.1819190842622e-2
        Ellipsoid's first eccentricity. Defaults to Earth's.

    Returns
    -------
    ecef : numpy.ndarray
        ECEF cartesian coordinates.
    """
    ecef = geodetic2ecef(lat, lon, h, a, ecc)
    ## Rotating ENU to ECEF
    uvw = enu2uvw(east, north, up, lat, lon, 'deg')
    ## Origin + offset from origin equals position in ECEF
    return ecef + uvw

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
    x : numpy.ndarray
        3D coordinates of point(s) to project.

    Returns
    -------
    x' : numpy.ndarray
        Transformed coordinates.
    """
    return _ltp_transformation(x)

def enu2ned(x: np.ndarray) -> np.ndarray:
    """
    Transform coordinates from ENU to NED.

    Parameters
    ----------
    x : numpy.ndarray
        3D coordinates of point(s) to project.

    Returns
    -------
    x' : numpy.ndarray
        Transformed coordinates.
    """
    return _ltp_transformation(x)

def aer2enu(az: float, elev: float, slant_range: float, deg: bool = True) -> np.ndarray:
    """
    Transform local azimuth-elevation-range (AER) spherical coordinates
    specified by ``az``, ``elev``, and ``slant_range`` to the local
    East-North-Up (ENU) Cartesian coordinates :cite:p:`strickland2020`.

    Parameters
    ----------
    az : float
        Azimuth measured clockwise from North.
    elev : float
        Elevation with respect to the local East-North plane.
    slant_range : float
        Distance from local origin.
    deg : bool, default: True
        If True, angles are given in degrees. Otherwise, they are in radians.

    Returns
    --------
    enu : numpy.ndarray
        ENU cartesian coordinates [east, north, up].
    """
    if deg:
        az *= DEG2RAD
        elev *= DEG2RAD
    r = slant_range*np.cos(elev)
    return np.array([r*np.sin(az), r*np.cos(az), slant_range*np.sin(elev)])

def enu2aer(east: float, north: float, up: float, deg: bool = True) -> np.ndarray:
    """
    Transform the local east-north-up (ENU) Cartesian coordinates specified by
    ``east``, ``north``, and ``up`` to the local azimuth-elevation-range (AER)
    Spherical coordinates :cite:p:`strickland2020`.

    Parameters
    ----------
    east : float
        X-coordinate of a point in the local ENU system.
    north : float
        Y-coordinate of a point in the local ENU system.
    up : float
        Z-coordinate of a point in the local ENU system.
    deg : bool, default: True
        If True, angles are returned in degrees. Otherwise, they are in radians.

    Returns
    -------
    aer : numpy.ndarray
        AER spherical coordinates [azimuth, elevation, slant_range].
    """
    r = np.linalg.norm([east, north])
    slant_range = np.linalg.norm([r, up])
    elev = np.arctan2(up, r)
    az = np.arctan2(east, north) % (2*np.pi)
    if deg:
        az *= RAD2DEG
        elev *= RAD2DEG
    return np.array([az, elev, slant_range])

def enu2dca(east: float, north: float, up: float, angle: float, deg: bool = True) -> np.ndarray:
    """
    Transform the local east-north-up (ENU) Cartesian coordinates specified by
    ``east``, ``north``, and ``up`` to the down-cross-above (DCA) reference
    frame :cite:p:`strickland2020`.

    The conversion is given by:

    .. math::

        \\begin{array}{rcl}
        \\begin{bmatrix}d \\\\ c \\\\ a\\end{bmatrix} & = &
        \\begin{bmatrix}
        \\sin\\theta & \\cos\\theta & 0 \\\\
        -\\cos\\theta & \\sin\\theta & 0 \\\\
        0 & 0 & 1
        \\end{bmatrix}
        \\begin{bmatrix}e \\\\ n \\\\ u\\end{bmatrix}

    Parameters
    ----------
    east : float
        X-coordinate of a point in the local ENU system.
    north : float
        Y-coordinate of a point in the local ENU system.
    up : float
        Z-coordinate of a point in the local ENU system.
    angle : float
        Angle measured clockwise from North.
    deg : bool, default: True
        If True, angles are given in degrees. Otherwise, they are in radians.

    Returns
    -------
    dca : numpy.ndarray
        DCA Cartesian coordinates [d, c, a].
    """
    if deg:
        angle *= DEG2RAD
    d = np.sin(angle)*east + np.cos(angle)*north
    c = -np.cos(angle)*east + np.sin(angle)*north
    return np.array([d, c, up])
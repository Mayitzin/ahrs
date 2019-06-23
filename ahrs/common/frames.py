# -*- coding: utf-8 -*-
"""
These are the most common coordinate frame operations.

Main definitions:

**Earth-Centered Inertial Frame** (ECI) has the origin at the center of mass of
the Earth. The X-axis points towards the vernal equinox in the equatorial
plane. The Z-axis is along the rotation axis of the Earth. The y-axis completes
with a right-hand system. Some literature names it **i-frame**.

**Earth-Centered Earth-Fixed Frame** (ECEF) has its origin and z-axis aligned
to the ECI frame, but rotates along with the Earth. Therefore is Earth-Fixed.
Some literature names it **e-frame**.

**Local-Level Frame** (LLF) is the local navigation frame, whose origin
coincides with the sensor frame. Some literature names it **l-frame**.

**Body Frame** matches the frame of the platform the sensors are mounted on.
The origin coincides with the center of gravity of the platform. The Y-axis
points forward of the moving platform, while the Z-axis points upwards. The
X-axis completes the right-hand system pointing in traverse direction. Some
literature names it **b-frame**.

**East-North-Up Frame** (ENU) is an LLF with the X-axis pointing East, Y-axis
pointing to the true North, and the Z-axis completes a right-hand system
pointing up (away from Earth.)

**North-East-Down Frame** (NED) is an LLF with the X-axis pointing to the true
North, Y-axis pointing East, and the Z-axis completing the right-hand system
pointing Down.

**Rectangular coordinates** in the ECEF represent position of a point with its
x, y, and z vector components aligned parallel to the corresponding e-frame
axes.

**Geodetic** (also ellipsoidal or curvilinear) **coordinates** in the ECEF are
defined for positioning elements on or near the Earth.

**Meridian** is the half of a creat circle on Earth's surface terminated by the
poles.

**Latitude** is the angle in the meridian plane from the equatorial plane to
the ellipsoidal normal at he point of interest.

**Longitude** is the angle in the equatorial plane from the prime meridian to
the projection of the point of interest onto the equatorial plane.

**Altitude** is the distance along the ellipsoidal normal between the surface
of the ellipsoid and the point of interest.

References
----------
.. [Noureldin] Fundamentals of Inertial Navigation, Satellite-based Position
    and their Integration.

"""

import numpy as np
from ahrs.common.constants import *

def geo2rect(r, h, long, lat, ecc):
    """
    Conversion from Geodetic to Rectangular Coordinates in the Earth-Centered
    Earth-Fixed Frame (ECEF.)

    Parameters
    ----------
    r : float
        Normal radius
    h : float
        Ellipsoidal height
    long : float
        Longitude
    lat : float
        Latitude
    ecc : float
        Eccentricity

    Returns
    -------
    X : array
        e-frame rectangular coordinates
    """
    X = np.zeros(3)
    X[0] = (r+h)*np.cos(lat)*np.cos(long)
    X[1] = (r+h)*np.cos(lat)*np.sin(long)
    X[2] = (r*(1.0-ecc**2)+h)*np.sin(lat)
    return X

def normal_radius(lat):
    """
    The normal radius, a.k.a the radius of curvature of the prime vertical,
    defined for the East-West direction.

    Parameters
    ----------
    lat : float
        Geodetic latitude

    Returns
    -------
    r : float
        Normal radius
    """
    return EQUATOR_RADIUS / (1.0-EARTH_ECCENTRICITY**2*np.sin(lat)**2)**0.5

def meridian_radius(lat):
    """
    The meridian radius is the Earth's radius of curvature in the (north-south)
    meridian at a given latitude `lat`.

    Parameters
    ----------
    lat : float
        Geodetic latitude

    Returns
    -------
    r : float
        Normal radius
    """
    return EQUATOR_RADIUS*(1.0-EARTH_ECCENTRICITY**2) / (1.0-EARTH_ECCENTRICITY**2*np.sin(lat)**2)**1.5

def gravity(lat, h=0.0, **kwargs):
    """
    Estimate acceleration due to gravity with Somigliana's formula.

    Parameters
    ----------
    lat : float
        Geodetic Latitude, in degrees.

    Extra Parameters
    ----------------
    h : float
        Height above Earth's surface, in meters.
    a : float
        Semi-major axis of Earth, in meters (Equatorial Radius)
    b : float
        Semi-minor axis of Earth, in meters (Polar Radius)
    ga : float
        Normal gravity at Equator, in m/s^2.
    gb : float
        Normal gravity at Poles, in m/s^2.
    e : float
        Spheroids' eccentricity.
    m : float
        Constant defined for Earth in [WGS84]_ as:

    .. math::

        m = \\frac{\\omega^2a^2b}{GM} = 0.00344978600313

    Returns
    -------
    g : float
        Ellipsoidal gravity 

    References
    ----------
    .. [WGS84] World Geodetic System 1984. Department of Defense. DMA Technical
        Report. Second Edition. September 1991.
    """
    # Set default parameters for Earth's properties
    a = kwargs.get('a', EQUATOR_RADIUS)
    b = kwargs.get('b', POLAR_RADIUS)
    ga = kwargs.get('ga', EQUATOR_GRAVITY)
    gb = kwargs.get('gb', POLAR_GRAVITY)
    e = kwargs.get('e', EARTH_ECCENTRICITY)
    f = kwargs.get('f', EARTH_FLATNESS)
    m = kwargs.get('m', EARTH_M)
    # Set variables
    lat *= np.pi/180.0
    k = (b*gb)/(a*ga)-1.0
    # Compute gravity at sea-level
    g = ga*(1.0+k*np.sin(lat)**2)/np.sqrt(1.0-e**2*np.sin(lat)**2)
    # Improve precision at a different height.
    if h != 0.0:
        k1 = 2.0*(1.0+f+m)/a
        k2 = 4.0*f/a
        k3 = 3/a**2
        g *= 1.0-(k1-k2*np.sin(lat)**2)*h + k3*h**2
    return g

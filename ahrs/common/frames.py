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
.. [1] Aboelmagd Noureldin, Tashfeen B. Karamat, Jacques Georgy. Fundamentals
       of Inertial Navigation, Satellite-based Position and their Integration.
       Springer-Verlag Berlin Heidelberg. 2013.
.. [2] World Geodetic System 1984. Its Definition and Relationships with Local
       Geodetic Systems. National Geospatial-Intelligence Agency (NGA)
       Standarization Document. 2014.
       ftp://ftp.nga.mil/pub2/gandg/website/wgs84/NGA.STND.0036_1.0.0_WGS84.pdf

"""

import numpy as np
from .constants import *

def geo2rect(lon, lat, h, r, ecc=EARTH_SECOND_ECCENTRICITY_2):
    """Geodetic to Rectangular Coordinates converstion in the Earth-Centered
    Earth-Fixed Frame (ECEF)

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
    ecc : float
        Second Eccentricity Squared

    Returns
    -------
    X : array
        ECEF rectangular coordinates
    """
    X = np.zeros(3)
    X[0] = (r+h)*np.cos(lat)*np.cos(lon)
    X[1] = (r+h)*np.cos(lat)*np.sin(lon)
    X[2] = (r*(1.0-ecc)+h)*np.sin(lat)
    return X

def rec2geo(X, ecc=EARTH_SECOND_ECCENTRICITY_2):
    x, y, z = X
    p = np.linalg.norm([x, y])
    theta = np.arctan(z*a/(p*b))
    lon = 2*np.arctan(y/(x+p))
    lat = np.arctan((z+ecc*b*np.sin(theta)**3)/(p-e*a*np.cos(theta)**3))
    N = a**2/np.sqrt(a**2*np.cos(lat)**2 + b**2*np.sin(lat)**2)
    h = p/np.cos(lat) - N

def eci2ecef(w, t=0):
    """Transformation between ECI and ECEF

    Parameters
    ----------
    w : float
        Rotation rate in rad/s
    t : float, default: 0.0
        Time since reference epoch.
    """
    return np.array([[np.cos(w)*t, np.sin(w)*t, 0.0],
                      [-np.sin(w)*t, np.cos(w)*t, 0.0],
                      [0.0, 0.0, 1.0]])



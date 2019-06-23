# -*- coding: utf-8 -*-
"""
Common constants used in AHRS.

References
----------
.. [WGS84] World Geodetic System 1984. Department of Defense. DMA Technical
    Report. Second Edition. September 1991.

"""

__all__ = [
    "EQUATOR_RADIUS",
    "POLAR_RADIUS",
    "EQUATOR_GRAVITY",
    "POLAR_GRAVITY",
    "EARTH_GM",
    "EARTH_FLATNESS",
    "EARTH_ECCENTRICITY",
    "EARTH_ROTATION",
    "EARTH_M"
]

EQUATOR_RADIUS = 6378137.0      # Semi-major axis of Earth, in meters (Equatorial Radius)
POLAR_RADIUS = 6356752.3142     # Semi-minor axis of Earth, in meters (Polar Radius)
EQUATOR_GRAVITY = 9.7803267714  # Normal gravity at Earth's Equator
POLAR_GRAVITY = 9.8321863685    # Normal gravity at Earth's Poles
EARTH_GM = 3.986005e14          # Earth's Gravitational Constant (Atmosphere included) [m^3/s^2]
EARTH_FLATNESS = (EQUATOR_RADIUS-POLAR_RADIUS)/EQUATOR_RADIUS
EARTH_ECCENTRICITY = (EARTH_FLATNESS*(2.0-EARTH_FLATNESS))**0.5
EARTH_ROTATION = 7.292115e-5    # Earth's Rotation rate (7.29 x 10^-5 rad/s)
EARTH_M = EARTH_ROTATION**2*EQUATOR_RADIUS**2*POLAR_RADIUS/EARTH_GM

# -*- coding: utf-8 -*-
"""
Common constants used in AHRS
=============================

References
----------
.. [1] World Geodetic System 1984. Its Definition and Relationships with Local
    Geodetic Systems. National Geospatial-Intelligence Agency (NGA)
    Standarization Document. 2014.
    ftp://ftp.nga.mil/pub2/gandg/website/wgs84/NGA.STND.0036_1.0.0_WGS84.pdf
.. [2] F. Chambat. Mean radius, mass, and inertia for reference Earth models.
    Physics of the Earth and Planetary Interiors Vol 124 (2001) p237–253.

"""
import numpy as np

# TRIGONOMETRY
M_PI = np.pi
DEG2RAD = np.pi/180.0
RAD2DEG = 180.0/np.pi

# WORLD GEODETIC SYSTEM 1984 (rev. 2014)
# Defining parameters
EARTH_EQUATOR_RADIUS = 6_378_137.0              # Semi-major axis of Earth (Equatorial Radius) [m]
EARTH_FLATTENING = 298.257223563                # Flattening Factor of the Earth
EARTH_GM = 3.986004418e14                       # Earth's Gravitational Constant (Atmosphere included) [m^3/s^2]
EARTH_ROTATION = 7.292115e-5                    # Earth's Rotation rate [rad/s]
# Fundamental constants
LIGHT_SPEED = 2.99792458e8                      # Velocity of light in vacuum [m/s]
UNIVERSAL_GRAVITATION = 6.67428e-11             # Universal Constant of Gravitation [m^3/(kg*s^2)]
EARTH_ATMOSPHERE_MASS = 5.148e18                # Total mean mass of the Atmosphere (with water vapor) [kg]
DYNAMIC_ELLIPTICITY = 3.2737949e-3               # Dynamic Ellipticity (H)
EARTH_GM_GPSNAV = 3.9860050e14                  # Earth's Gravitational Constant for GPS Navigation Message [m^3/s^2]
# Derived geometric constants
EARTH_FLATTENING_INV = 1/EARTH_FLATTENING       # Earth's Flattening (reduced)
EARTH_POLAR_RADIUS = 6_356_752.3142             # Semi-minor axis of Earth (Polar Radius) [m]
EARTH_FIRST_ECCENTRICITY = 8.1819190842622e-2
EARTH_FIRST_ECCENTRICITY_2 = EARTH_FIRST_ECCENTRICITY**2
EARTH_SECOND_ECCENTRICITY = 8.2094437949696e-2
EARTH_SECOND_ECCENTRICITY_2 = EARTH_SECOND_ECCENTRICITY**2
EARTH_LINEAR_ECCENTRICITY = 5.2185400842339e5
EARTH_POLAR_CURVATURE_RADIUS = 6_399_593.6258   # Polar radius of Curvature [m]
EARTH_AXIS_RATIO = 9.96647189335e-1             # Axis ratio: EARTH_POLAR_RADIUS / EARTH_EQUATOR_RADIUS
EARTH_MEAN_RADIUS = 6_371_200.0                 # Earth's Arithmetic Mean radius [m] ((2*EQUATOR_RADIUS + POLAR_RADIUS) / 3)
EARTH_MEAN_AXIAL_RADIUS = 6_371_008.7714        # Mean Radius of the Three Semi-axes [m]
EARTH_AUTHALIC_RADIUS = 6_371_007.1810          # Radius of equal area sphere [m]
EARTH_EQUIVOLUMETRIC_RADIUS = 6_371_000.79      # Tadius of equal volume sphere [m] ((EQUATOR_RADIUS^2 * POLAR_RADIUS)^(1/3))
EARTH_C20_DYN = -4.84165143790815e-4            # Earth's Dynamic Second Degree Zonal Harmonic (C_2,0 dyn)
EARTH_C22_DYN = 2.43938357328313e-6             # Earth's Dynamic Second Degree Sectorial Harmonic (C_2,2 dyn)
EARTH_C20_GEO = -4.84166774985e-4               # Earth's Geographic Second Degree Zonal Harmonic
EARTH_J2 = 1.08263e-3                           # Earth's Dynamic Form Factor
# Derived physical constants
NORMAL_GRAVITY_POTENTIAL = 6.26368517146        # Normal Gravity Potential on the Ellipsoid [m^2/s^2]
EQUATORIAL_NORMAL_GRAVITY = 9.7803253359        # Normal Gravity at the Equator (on the ellipsoid) [m/s^2]
POLAR_NORMAL_GRAVITY = 9.8321849379             # Normal Gravity at the Pole (on the ellipsoid) [m/s^2]
MEAN_NORMAL_GRAVITY = 9.7976432223              # Mean Normal Gravity [m/s^2]
SOMIGLIANA_GRAVITY = 1.931852652458e-3          # Somigliana's Formula Normal Gravity constant
NORMAL_GRAVITY_FORMULA = 3.449786506841e-3      # Normal Gravity Formula constant (EARTH_ROTATION^2 * EQUATOR_RADIUS^2 * POLAR_RADIUS / EARTH_GM)
EARTH_MASS = 5.9721864e24                       # Earth's Mass (Atmosphere inclulded) [kg]
EARTH_GM_1 = 3.986000982e14                     # Geocentric Gravitational Constant (Atmosphere excluded) [m^3/s^2]
EARTH_GM_2 = 3.4359e8                           # Gravitational Constant of the Earth’s Atmosphere [m^3/s^2]

G_UNIVERSAL = 6.67428e-11
# Standard Gravitational Constants (gravitational constant x Mass)
GM_SUN = 1.32712440018e20
GM_MERCURY = 2.2032e13
GM_VENUS = 3.24859e14
GM_EARTH = EARTH_GM
GM_MOON = 4.9048695e12
GM_MARS = 4.282837e13
GM_JUPITER = 1.26686534e17
GM_SATURN = 3.7931187e16
GM_URANUS = 5.793939e15
GM_NEPTUNE = 6.836529e15
GM_PLUTO = 8.71e11
GM_CERES = 6.26325e10
GM_ERIS = 1.108e12

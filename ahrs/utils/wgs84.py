"""
World Geodetic System (1984)
============================

The World Geodetic System 1984 (WGS 84) represents the best geodetic model of
the Earth using data, techniques and technology available through 2013 by the
United States of America's National Geospatial-Intelligence Agency (NGA).

It describes the best global geodetic reference system for the Earth available
for practical applications of mapping, charting, geopositioning, and navigation.

The current standard has been possible due to improved scientific models and
additional global data. Using these data, an improved Earth Gravitational Model
2008 (EGM2008) and its associated geoid were developed.

The WGS 84 Coordinate System is a Conventional Terrestrial Reference System
(CTRS), that follows the criteria outlined in the International Earth Rotation
and Reference Systems Service (IERS):

- It is geocentric, the center of mass being defined for the whole Earth
  including oceans and atmosphere.
- Its scale is that of the local Earth frame, in the meaning of a relativistic
  theory of gravitation.
- Its orientation was initially given by the Bureau International de l'Heure
  (BIH) orientation of 1984.0
- Its time evolution in orientation will create no residual global rotation
  with regards to the crust.
- It is defined as a right-handed, orthogonal, and Earth-fixed coordinate
  system, where the Z-axis serves as the rotational axis of the ellipsoid
  revolution.

WGS 84 (G1762) is the sixth and latest update (16 Oct 2013) to the realization
of the WGS 84 Reference Frame, and is updated to incorporate international
conventions and alignements to the International Terrestrial Reference Frame
2008 (ITRF2008).

A mathematically manageable reference surface (an ellipsoid) approximating the
broad features of the figure and of the gravity field of the Earth is used to
describe the Earth surface and the forces acting on and above it.

Additionally, an equipotential surface of the gravity field of the Earth,
called the geoid is used to define the EGM2008. Its implementation falls out of
the scope of this submodule.

The WGS 84 ellipsoid is defined by the semi-major axis (a), the reciprocal
flattening (1/f) of an oblate ellipsoid of revolution, the Geocentric
Gravitational Constant (GM) and the angular velocity (w).

+--------+---------------------------------------+---------------------+---------+
| Symbol | Definition                            | Value               | Unit    |
+========+=======================================+=====================+=========+
| a      | Semi-major Axis                       | 6378137.0           | m       |
| 1/f    | Flattening Factor of the Earth        | 298.257223563       |         |
| GM     | Geocentric Gravitational Constant     | 3.986004418 x 10^14 | m^3/s^2 |
| w      | Earth's Nominal Mean Angular Velocity | 7.292115 x 10^-5    | rad/s   |
+--------+---------------------------------------+---------------------+---------+

The first two parameters (a, 1/f) define the geometry of the rotational
ellipsoid, while the other two parameters (GM, w) permit the unique
determination of its associated normal gravity field.

Having these 4 elements defined, it is possible to estimate most of the WGS84
parameters directly.

Furthermore, there are three unitless values used in the computation of the
Moments of Inertia:

+-----------------+-------------------------------------------+---------------------------+
| Symbol          | Definition                                | Value                     |
+=================+===========================================+===========================+
| C_2,0 dyn[2008] | Dynamic Second Degree Zonal Harmonics     | -4.84165143790815 x 10^-4 |
| C_2,2 dyn[2008] | Dynamic Second Degree Sectorial Harmonics | 2.43938357328313 x 10^-6  |
| H               | Dynamic Ellipticity                       | 3.2737949 x 10^-3         |
+-----------------+-------------------------------------------+---------------------------+

The dynamic harmonics are obtained from the data obtained with EGM2008, and are
NOT derived from the ellipsoid parameters. The subscript `dyn[2008]` denote
their origin. The dynamic ellipticity is a factor in the theoretical value of
the rate of precession of the equinoxes, which is known from observation.

The WGS 84 Ellipsoid is identified as a geocentric, equipotential ellipsoid of
revolution, i.e., an ellipsoid with a surface on which the value of the gravity
potential is the same everywhere.

This gravity potential on the surface of the ellipsoid can be determined by the
closed formula of Somigliana given four unique parameters:

.. math::

    g = g_e \\frac{1+k\\sin^2(\\phi)}{\\sqrt{1-e^2\\sin^2(\\phi)}}

using the helper variable :math:`k`:

.. math::

    k = \\frac{bg_p}{ag_e} - 1

where :math:`a` is the semi-major axis of the ellipsoid, :math:`b` is the
semi-minor axis of the ellipsoid, :math:`g_e` is the normal gravity at the
Equator, :math:`g_p` is the normal gravity at the poles, :math:`e^2` is the
ellipsoid's first eccentricity squared, and :math:`\\phi` is the geodetic
latitude.

At small heights above the surface, the normal gravity can be approximated with
a truncated Taylor Series with a positive direction downward along the geodetic
normal to the ellipsoid:

.. math::

    g_h = g \\Big(1 - \\frac{2}{a}(1+f+m-2f\\sin^2(\\phi))h \\Big)

where

.. math::

    m = \\frac{w^2a^2b}{GM}

and :math:`h` is the height, in meters, above the ellipsoid's surface.

As a bonus, the normal gravity estimation of the European Cooperation on Legal
Metrology (WELMEC) is also implemented.

References
----------
.. [1] World Geodetic System 1984. Its Definition and Relationships with Local
       Geodetic Systems. National Geospatial-Intelligence Agency (NGA)
       Standarization Document. 2014.
       ftp://ftp.nga.mil/pub2/gandg/website/wgs84/NGA.STND.0036_1.0.0_WGS84.pdf
.. [2] Heiskanen, W. A. and Moritz, H. Physical Geodesy. W. H. Freeman and
       Company. 1967.
.. [3] WELMEC DIRECTIVE 2009/23/EC: Common application non-automatic weighing
       instruments.
       https://www.welmec.org/documents/guides/2/
"""

import unittest
import numpy as np
from ..common.constants import *

def normal_gravity(lat: float, h: float = 0.0, a:float = EARTH_EQUATOR_RADIUS, b: float = EARTH_POLAR_RADIUS, ge: float = EQUATORIAL_NORMAL_GRAVITY, gp: float = POLAR_NORMAL_GRAVITY, **kw) -> float:
    """Normal Gravity on (or above) Ellipsoidal Surface

    Somigliana's formula to estimate the normal gravity on or above the surface
    of an ellipsoidal body.

    Parameters
    ----------
    lat: float
        Latitude, in decimal degrees, in geodetic coordinates
    h : float, default: 0.0
        Mean sea level height, in meters.
    a : float, default: 6378137.0
        Semi-major axis of ellipsoid.
    b : float, default: 6356752.3142
        Semi-minor axis of ellipsoid.
    ge : float, default: 9.7803253359
        Equatorial normal gravity.
    gp : float, default: 9.8321849379
        Polar normal gravity.

    Return
    ------
    g : float
        Normal gravity at given point in space, in m/s^2.
    """
    lat *= DEG2RAD
    k = (b*gp)/(a*ge)-1
    e2 = (a**2-b**2)/a**2
    sin2 = np.sin(lat)**2
    g = ge*(1+k*sin2)/np.sqrt(1-e2*sin2)            # Normal Gravity on Ellipsoid Surface (eq. 4-1)
    if h==0.0:
        return g
    # Normal gravity above surface
    f = (a-b)/a                                     # Flattening Factor of Earth
    m = kw.get('m', NORMAL_GRAVITY_FORMULA)
    if not np.isclose(m, NORMAL_GRAVITY_FORMULA):
        w = kw.get('w', EARTH_ROTATION)
        gm = kw.get('GM', EARTH_GM)
        m = w**2*a**2*b/gm                          # (w^2 a^2 b)/GM    (eq. B-20)
    g *= 1 - 2*h*(1+f+m-2*sin2)/a + 3*h**2/a**2     # Normal Gravity Above Ellipsoid (eq. 4-3)
    return g

def welmec_gravity(lat: float, h: float = 0.0):
    """Reference value of WELMEC's gravity zone

    Gravity zones are implemented by European States on their territories for
    weighing instruments that are sensitive to gravity variations.

    Manufacturers may adjust their instruments using the reference gravity
    formula:

    .. math::

        g = 9.780318(1 + 0.0053024\\sin^2(\\phi) - 0.0000058\\sin^2(2\\phi)) - 0.000003085h

    where :math:`\\phi` is the geographical latitude and :math:`h` is the
    altitude in meters.

    Parameters
    ----------
    lat: float
        Geographical Latitude, in decimal degrees.
    h : float, default: 0.0
        Mean sea level height, in meters.

    Return
    ------
    g : float
        Normal gravity at given point in space, in m/s^2.

    Examples
    --------
    >>> welmec_gravity(52.3, 80.0)      # 52.3°, 80 m
    9.818628439187075
    """
    lat *= DEG2RAD
    return 9.780318*(1 + 0.0053024*np.sin(lat)**2 - 0.0000058*np.sin(2*lat)**2) - 0.000003085*h

class WGS:
    """World Geodetic System 1984

    Parameters
    ----------
    a : float, default: 6378137.0
        Earth's Semi-major axis (Equatorial Radius), in meters
    f : float, default: 0.0033528106647474805
        Earth's flattening factor (Equatorial Radius), in meters

    Extra Parameters
    ----------------
    GM : float
        Body's Standard Gravitational Constant in m^3/s^2
    w : float
        Body's rotation rate in rad/s

    Attributes
    ----------
    a : float
        Earth's Semi-major axis (Equatorial Radius), in meters.
    f : float
        Earth's flattening factor (Equatorial Radius), in meters.
    geodetic : bool
        Wheter the object describes Earth's characteristics.
    gm : float
        Body's Standard Gravitational Constant in m^3/s^2.
    w : float
        Body's rotation rate in rad/s.
    """
    def __init__(self, a: float = EARTH_EQUATOR_RADIUS, f: float = EARTH_FLATTENING_INV, **kw):
        self.a = a
        self.f = f
        self.is_geodetic = np.isclose(self.a, EARTH_EQUATOR_RADIUS) and np.isclose(self.f, EARTH_FLATTENING_INV)
        self.gm = kw.get('GM', EARTH_GM if self.is_geodetic else None)
        self.w = kw.get('w', EARTH_ROTATION if self.is_geodetic else None)

    @property
    def semi_minor_axis(self):
        return self.a*(1-self.f)

    @property
    def first_eccentricity_squared(self):
        return 2*self.f - self.f**2

    @property
    def second_eccentricity_squared(self):
        b = self.semi_minor_axis
        return (self.a**2-b**2)/b**2

    @property
    def linear_eccentricity(self):
        b = self.semi_minor_axis
        return np.sqrt(self.a**2-b**2)

    @property
    def aspect_ratio(self):
        b = self.semi_minor_axis
        return b/self.a

    @property
    def curvature_polar_radius(self):
        return self.a/(1-self.f)

    @property
    def arithmetic_mean_radius(self):
        return self.a*(1-self.f/3)

    @property
    def authalic_sphere_radius(self):
        r = self.curvature_polar_radius
        es = np.sqrt(self.second_eccentricity_squared)
        return r*(1 - 2*es**2/3 + 26*es**4/45 - 100*es**6/189 + 7034*es**8/14175 - 220652*es**10/467775)

    @property
    def equivolumetric_sphere_radius(self):
        return self.a*np.cbrt(1-self.f)

    @property
    def normal_gravity_constant(self):
        b = self.semi_minor_axis
        return self.w**2*self.a**2*b/self.gm

    @property
    def dynamical_form_factor(self):
        m = self.normal_gravity_constant
        e2 = self.first_eccentricity_squared
        es = np.sqrt(self.second_eccentricity_squared)
        q0 = 0.5*((1+3/es**2)*np.arctan(es) - 3/es)
        return e2*(1-2*m*es/(15*q0))/3

    @property
    def second_degree_zonal_harmonic(self):
        return -self.dynamical_form_factor/np.sqrt(5.0)

    @property
    def normal_gravity_potential(self):
        E = self.linear_eccentricity
        es = np.sqrt(self.second_eccentricity_squared)
        return self.gm*np.arctan(es)/E + self.w**2*self.a**2/3

    @property
    def equatorial_normal_gravity(self):
        b = self.semi_minor_axis
        m = self.normal_gravity_constant
        es = np.sqrt(self.second_eccentricity_squared)
        q0 = 0.5*((1 + 3/es**2)*np.arctan(es) - 3/es)
        q0s = 3*((1 + 1/es**2)*(1 - np.arctan(es)/es)) - 1
        return self.gm * (1 - m - m*es*q0s/(6*q0))/(self.a*b)

    @property
    def polar_normal_gravity(self):
        b = self.semi_minor_axis
        m = self.normal_gravity_constant
        es = np.sqrt(self.second_eccentricity_squared)
        q0 = 0.5*((1 + 3/es**2)*np.arctan(es) - 3/es)
        q0s = 3*((1 + 1/es**2)*(1 - np.arctan(es)/es)) - 1
        return self.gm * (1 + m*es*q0s/(3*q0))/self.a**2

    @property
    def mean_normal_gravity(self):
        b = self.semi_minor_axis
        e = np.sqrt(self.first_eccentricity_squared)
        gp = self.polar_normal_gravity
        ge = self.equatorial_normal_gravity
        k = (b*gp)/(self.a*ge)-1
        g = self.equatorial_normal_gravity.copy()
        g *= 1 + e**2/6 + k/3 + 59*e**4/360 + 5*e**2*k/18 + 2371*e**6/15120 + 259*e**4*k/1080 + 270229*e**8/1814400 + 9623*e**6*k/45360
        return g

    @property
    def mass(self):
        return self.gm/UNIVERSAL_GRAVITATION

    @property
    def geometric_inertial_moment_about_Z(self):
        return 2*self.mass*self.a**2*(1-0.4*np.sqrt(2.5*self.normal_gravity_constant/self.f - 1))/3

    @property
    def geometric_inertial_moment(self):
        return self.geometric_inertial_moment_about_Z + np.sqrt(5)*self.mass*self.a**2*EARTH_C20_GEO

    @property
    def geometric_dynamic_ellipticity(self):
        return (self.geometric_inertial_moment_about_Z - self.geometric_inertial_moment)/self.geometric_inertial_moment_about_Z

    # Geodetic Properties
    @property
    def atmosphere_gravitational_constant(self):
        return UNIVERSAL_GRAVITATION*EARTH_ATMOSPHERE_MASS

    @property
    def gravitational_constant_without_atmosphere(self):
        return self.gm - self.atmosphere_gravitational_constant

    @property
    def dynamic_inertial_moment_about_Z(self):
        return -np.sqrt(5)*self.mass*self.a**2*EARTH_C20_DYN/DYNAMIC_ELLIPTICITY

    @property
    def dynamic_inertial_moment_about_X(self):
        return np.sqrt(5)*self.mass*self.a**2*((1-1/DYNAMIC_ELLIPTICITY)*EARTH_C20_DYN - EARTH_C22_DYN/np.sqrt(3))

    @property
    def dynamic_inertial_moment_about_Y(self):
        return np.sqrt(5)*self.mass*self.a**2*((1-1/DYNAMIC_ELLIPTICITY)*EARTH_C20_DYN + EARTH_C22_DYN/np.sqrt(3))

class WGSTest(unittest.TestCase):
    def test_wgs84(self):
        """Test WGS84 with Earth's properties"""
        wgs = WGS()
        self.assertAlmostEqual(wgs.a, 6_378_137.0, 1)
        self.assertAlmostEqual(wgs.f, 1/298.257223563, 7)
        self.assertAlmostEqual(wgs.gm, 3.986004418e14, 7)
        self.assertAlmostEqual(wgs.w, 7.292115e-5, 7)
        self.assertAlmostEqual(wgs.semi_minor_axis, 6_356_752.3142, 4)
        self.assertAlmostEqual(wgs.first_eccentricity_squared, 8.1819190842622e-2**2, 7)
        self.assertAlmostEqual(wgs.second_eccentricity_squared, 8.2094437949696e-2**2, 7)
        self.assertAlmostEqual(wgs.linear_eccentricity, 5.2185400842339e5, 7)
        self.assertAlmostEqual(wgs.aspect_ratio, 9.966471893352525e-1, 10)
        self.assertAlmostEqual(wgs.curvature_polar_radius, 6399593.6258, 4)
        self.assertAlmostEqual(wgs.arithmetic_mean_radius, 6371008.7714, 4)
        self.assertAlmostEqual(wgs.authalic_sphere_radius, 6371007.181, 3)
        self.assertAlmostEqual(wgs.equivolumetric_sphere_radius, 6371000.79, 2)
        self.assertAlmostEqual(wgs.normal_gravity_constant, 3.44978650684084e-3, 10)
        self.assertAlmostEqual(wgs.dynamical_form_factor, 1.082629821313e-3, 10)
        self.assertAlmostEqual(wgs.second_degree_zonal_harmonic, -4.84166774985e-4, 10)
        self.assertAlmostEqual(wgs.normal_gravity_potential, 6.26368517146e7, 4)
        self.assertAlmostEqual(wgs.equatorial_normal_gravity, 9.7803253359, 10)
        self.assertAlmostEqual(wgs.polar_normal_gravity, 9.8321849379, 10)
        self.assertAlmostEqual(wgs.mean_normal_gravity, 9.7976432223, 10)
        self.assertAlmostEqual(wgs.mass, 5.9721864e24, delta=1e17)
        self.assertAlmostEqual(wgs.atmosphere_gravitational_constant, 3.43592e8, delta=1e3)
        self.assertAlmostEqual(wgs.gravitational_constant_without_atmosphere, 3.986000982e14, delta=1e4)
        self.assertAlmostEqual(wgs.dynamic_inertial_moment_about_Z, 8.0340094e37, delta=1e30)   # FAILS
        self.assertAlmostEqual(wgs.dynamic_inertial_moment_about_X, 8.00792178e37, delta=1e29)
        self.assertAlmostEqual(wgs.dynamic_inertial_moment_about_Y, 8.0080748e37, delta=1e30)
        self.assertAlmostEqual(wgs.geometric_inertial_moment_about_Z, 8.07302937e37, delta=1e29)
        self.assertAlmostEqual(wgs.geometric_inertial_moment, 8.04672663e37, delta=1e29)
        self.assertAlmostEqual(wgs.geometric_dynamic_ellipticity, 3.2581004e-3, 9)
        del wgs

    def test_welmec(self):
        """Test WELMEC reference gravity
        """
        self.assertAlmostEqual(welmec_gravity(52.3, 80.0), 9.812484, 6)     # Braunschweig, Germany
        self.assertAlmostEqual(welmec_gravity(60.0, 250.0), 9.818399, 6)    # Uppsala, Sweden

if __name__ == '__main__':
    unittest.main()


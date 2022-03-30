"""
World Geodetic System (1984)
============================

The World Geodetic System 1984 (WGS 84) [WGS84]_ describes the best geodetic
reference system for the Earth available for the practical applications of
mapping, charting, geopositioning, and navigation, using data, techniques and
technology available through 2013 by the United States of America's National
Geospatial-Intelligence Agency (NGA.)

The WGS 84 Coordinate System is a Conventional Terrestrial Reference System
(`CTRS <https://gssc.esa.int/navipedia/index.php/Conventional_Terrestrial_Reference_System>`_),
that follows the criteria outlined in the International Earth Rotation and
Reference Systems Service (`IERS <https://www.iers.org/IERS/EN/Science/ITRS/ITRS.html>`_):

- It is geocentric, meaning the center of mass is defined for the whole Earth
  including oceans and atmosphere.
- Its scale is that of the local Earth frame, in the meaning of a relativistic
  theory of gravitation.
- Its orientation was initially given by the `Bureau International de l'Heure
  <https://en.wikipedia.org/wiki/International_Time_Bureau>`_ (BIH) at the
  epoch 1984.0
- It is defined as a right-handed, orthogonal, and Earth-fixed coordinate
  system, where the Z-axis serves as the rotational axis of the ellipsoid
  revolution.

WGS 84 (G1762) is the sixth and latest update (16 Oct 2013) to the realization
of the WGS 84 Reference Frame, and is updated to incorporate international
conventions and alignements to the International Terrestrial Reference Frame
2008 (`ITRF2008 <https://www.iers.org/IERS/EN/DataProducts/ITRF/itrf.html>`_.)

A simple and practical approach is a mathematically manageable reference
surface (an ellipsoid) approximating the broad features of the figure and of
the gravity field of the Earth, which is used to describe the Earth surface and
the forces acting on and above it.

A second, and more precise, solution considers an equipotential surface of the
gravity field of the Earth, called the geoid, which is used to define the Earth
Gravitational Model (`EGM <https://en.wikipedia.org/wiki/Earth_Gravitational_Model>`_.)
The implementation of this second model falls out of the scope of this module.

The WGS 84 ellipsoid is defined by the semi-major axis (:math:`a`), the
reciprocal flattening (:math:`1/f`) of an oblate ellipsoid of revolution, the
Geocentric Gravitational Constant (:math:`GM`) and the angular velocity
(:math:`\\omega`)

+-----------------------+---------------------------------------+---------------------+---------+
| Symbol                | Definition                            | Value               | Unit    |
+=======================+=======================================+=====================+=========+
| :math:`a`             | Semi-major Axis                       | 6378137.0           | m       |
+-----------------------+---------------------------------------+---------------------+---------+
| :math:`1/f`           | Flattening Factor of the Earth        | 298.257223563       |         |
+-----------------------+---------------------------------------+---------------------+---------+
| :math:`GM`            | Geocentric Gravitational Constant     | 3.986004418 x 10^14 | m^3/s^2 |
+-----------------------+---------------------------------------+---------------------+---------+
| :math:`\\omega`        | Earth's Nominal Mean Angular Velocity | 7.292115 x 10^-5    | rad/s   |
+-----------------------+---------------------------------------+---------------------+---------+

The first two parameters (:math:`a`, :math:`1/f`) define the geometry of the
rotational ellipsoid, while the other two parameters (:math:`GM`, :math:`\\omega`)
permit the unique determination of its associated normal gravity field.

Having these 4 elements defined, it is possible to estimate most of the WGS84
parameters directly. The class ``WGS`` already sets these values by default
(and estimates the semi-minor axis ``b``), but with slightly different notation:

.. code:: python

    >>> wgs = ahrs.utils.WGS()
    >>> wgs.a       # Semi-major Axis
    6378137.0
    >>> 1/wgs.f     # Flattening Factor of the Earth
    298.257223563
    >>> wgs.gm      # Geocentric Gravitational Constant
    398600441800000.0
    >>> wgs.w       # Earth's Nominal Mean Angular Velocity
    7.292115e-05
    >>> wgs.b       # Semi-minor Axis
    6356752.314245179

Furthermore, there are three unitless values used in the computation of the
Moments of Inertia:

+--------------------------+-------------------------------------------+---------------------------+
| Symbol                   | Definition                                | Value                     |
+==========================+===========================================+===========================+
| C\ :sub:`2,0 dyn[2008]`  | Dynamic Second Degree Zonal Harmonics     | -4.84165143790815 x 10^-4 |
+--------------------------+-------------------------------------------+---------------------------+
| C\ :sub:`2,2 dyn[2008]`  | Dynamic Second Degree Sectorial Harmonics | 2.43938357328313 x 10^-6  |
+--------------------------+-------------------------------------------+---------------------------+
| H                        | Dynamic Ellipticity                       | 3.2737949 x 10^-3         |
+--------------------------+-------------------------------------------+---------------------------+

The dynamic harmonics are recovered from the data obtained empirically in
EGM2008, and are **NOT** derived from the ellipsoid parameters. The subscript
``dyn[2008]`` denote their origin.

The dynamic ellipticity is a factor in the theoretical value of the rate of
precession of the equinoxes, which is known from observation too.

The WGS 84 Ellipsoid is identified as a geocentric, equipotential ellipsoid of
revolution, i.e., an ellipsoid with a surface on which the value of the gravity
potential is the same everywhere.

Earth's Gravity Field
---------------------

In a rectangular system a point :math:`\\mathbf{p}` is located with the
coordinates :math:`\\begin{pmatrix}x & y & z\\end{pmatrix}`, but in
**ellipsoidal coordinates**, this point is found with
:math:`\\begin{pmatrix}u & \\phi & \\lambda\\end{pmatrix}`, where :math:`u` is
the semi-minor axis of the ellipsoid, :math:`\\phi` is the angle between the
plumb line and the equatorial plane called the **geographical latitude**, and
:math:`\\lambda` is the angle between :math:`\\mathbf{p}` and the meridian
plane of Greenwich called the **geographical longitude**.

We assume Earth is an ellipsoid of revolution which is an equipotential surface
of a **normal gravity field**. Although the Earth is not a perfect ellipsoid,
its gravity field is easier to handle assuming it is one. The deviations of the
field are so small that they are considered linear for this model.

The definition of the **potential of the normal gravity field** :math:`U`
[Heiskanen]_ is:

.. math::
    U = V + \\Phi

where :math:`V` is the **potential of gravitational force** defined as [#]_:

.. math::
    V = \\frac{GM}{E}\\arctan\\Big(\\frac{E}{u}\\Big) + \\frac{1}{2}\\omega^2a^2\\frac{q}{q_0}\\Big(\\sin^2\\beta-\\frac{1}{3}\\Big)

and :math:`\\Phi` is the **potential of centrifugal force**:

.. math::
    \\Phi = \\frac{1}{2} \\omega^2(u^2+E^2)\\cos^2\\beta

Here, the ellipsoid's **linear eccentricity** is simply :math:`E = \\sqrt{a^2-b^2}`,
and we set the helper constants:

.. math::
    \\begin{array}{ll}
    q &= \\frac{1}{2} \\Big[\\Big(1+\\frac{3u^2}{E^2}\\Big)\\arctan\\frac{E}{u}-\\frac{3u}{E}\\Big] \\\\
    q_0 &= \\frac{1}{2} \\Big[\\Big(1+\\frac{3b^2}{E^2}\\Big)\\arctan\\frac{E}{b}-\\frac{3b}{E}\\Big] \\\\
    q_0' &= 3\\Big[\\Big(1+\\frac{1}{e'^2}\\Big)\\Big(1-\\frac{1}{e'}\\arctan e'\\Big)\\Big] - 1
    \\end{array}

The **normal gravity vector** :math:`\\vec{\\mathbf{g}}` is obtained from its
potential :math:`U` :

.. math::
    \\vec{\\mathbf{g}} = \\nabla U =
    \\begin{bmatrix}g_u \\\\ g_\\beta \\\\ g_\\lambda \\end{bmatrix} =
    \\begin{bmatrix}
    \\frac{1}{w}\\frac{\\partial U}{\\partial u} \\\\
    \\frac{1}{w\\sqrt{u^2+E^2}}\\frac{\\partial U}{\\partial \\beta} \\\\ 0
    \\end{bmatrix}

We see that :math:`g_\\lambda=0`, because the effects of the potential along
the longitudinal direction are neglected.

Normal Gravity on the Surface
-----------------------------

The **magnitude of the normal gravity vector**, simply called **normal
gravity**, is:

.. math::
    g = \\|\\vec{\\mathbf{g}}\\| = \\sqrt{g_u^2+g_\\beta^2}

At the surface of the ellipsoid :math:`u=b` making :math:`g_{\\beta,0}=0`. So,
the total gravity on the surface of the ellipsoid  :math:`g_0` is just:

.. math::
    g_0 = \\|\\vec{\\mathbf{g}_0}\\| =
    \\frac{GM}{a\\sqrt{a^2\\sin^2\\beta+b^2\\cos^2\\beta}}
    \\Big[\\Big(1+\\frac{me'q_0'}{3q_0}\\Big)\\sin^2\\beta + \\Big(1-m-\\frac{me'q_0'}{6q_0}\\Big)\\cos^2\\beta\\Big]

where:

.. math::
    m = \\frac{\\omega^2a^2b}{GM}

and :math:`e'=\\frac{E}{b}` is the **second eccentrictiy**.

From here we can find two basic variables that will help us to linearly
estimate the normal gravity at any point on the ellipsoid.

Keeping it on the surface, we can estimate the **normal gravity at the equator**:

.. math::
    g_e = g_0(\\beta=0°) = \\frac{GM}{ab}\\Big(1-m-\\frac{me'q_0'}{6q_0}\\Big)

Similarly, we estimate the **normal gravity at the poles**:

.. math::
    g_p = g_0(\\beta=90°) = \\frac{GM}{a^2}\\Big(1+\\frac{me'q_0'}{3q_0}\\Big)

With these two basic values, we can linearly approximate the normal at any
latitude on the surface of the ellipsoid, but we need to do it in geographical
coordinates.

Using the property :math:`\\tan\\beta=\\frac{b}{a}\\tan\\phi`, we define a
closed form formula to find the normal gravity :math:`g` at any given latitude
:math:`\\phi` [Somigliana1929]_:

.. math::
    g(\\phi) = \\frac{ag_e \\cos^2\\phi + bg_p\\sin^2\\phi}{\\sqrt{a^2\\cos^2\\phi + b^2\\sin^2\\phi}}

For numerical computation, a more convenient form is:

.. math::
    g(\\phi) = g_e\\frac{1+k\\sin^2\\phi}{\\sqrt{1-e^2\\sin^2\\phi}}

using the helper variable :math:`k = \\frac{bg_p}{ag_e} - 1`.

The estimation of the normal gravity is already simplified with the class ``WGS``
requiring the latitude only.

.. code:: python

    >>> wgs.normal_gravity(0.0)     # Normal gravity at Equator (latitude = 0.0 °)
    9.78032533590406
    >>> wgs.normal_gravity(50.0)    # Normal gravity at latitude = 50.0 °
    9.810702135603085

Normal Gravity above the Surface
--------------------------------

At small heights above the surface, the normal gravity can be approximated with
a truncated Taylor Series with a positive direction downward along the geodetic
normal to the ellipsoid:

.. math::
    g(\\phi, h) = g(\\phi) \\Big(1 - \\frac{2}{a}\\big(1+f+m-2f\\sin^2\\phi\\big)h + \\frac{3}{a^2}h^2\\Big)

where :math:`h` is the height, in meters, above the ellipsoid's surface.

.. code:: python

    >>> wgs.normal_gravity(50.0, 1000.0)    # Gravity at latitude = 50.0 °, 1000 m above surface
    9.807617683884756

Other Gravitational Methods
---------------------------

The well known **International Gravity Formula** [Lambert]_ as described by
Helmut Moritz in [Tscherning]_ for the `Geodetic Reference System 1980
<https://en.wikipedia.org/wiki/Geodetic_Reference_System_1980>`_
is also implemented here:

.. math::
    g(\\phi) = 9.780327 (1 + 0.0053024 \\sin^2\\phi - 0.0000058 \\sin^2(2\\phi))

.. code:: python

    >>> ahrs.utils.international_gravity(10.0)
    9.781884110728155

As a bonus, the **normal gravity estimation** of the European Cooperation on
Legal Metrology (`WELMEC <https://en.wikipedia.org/wiki/WELMEC>`_) is also
implemented here:

.. math::
    g(\\phi, h) = 9.780318(1 + 0.0053024\\sin^2(\\phi) - 0.0000058\\sin^2(2\\phi)) - 0.000003085h

.. code:: python

    >>> ahrs.utils.welmec_gravity(50.0, 1000.0)     # 50.0° N, 1000 m above sea level
    9.807610187885896

Although this is thought and mainly used for European latitudes.

All methods above can be used for cartography and basic gravitational
references. For more advanced and precise estimations it is recommended to use
the `EGM2008 <https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/>`_,
which is defined, like the WMM, with spherical harmonics, but of degree 2190
and order 2159.

Because of its high complexity and demanding computation, the implementation of
the EGM2008 is left out of this module in favour of the more convenient
applications developed in `Fortran by the NGA <https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/egm08_wgs84.html>`_.

Footnotes
---------
.. [#] The most precise EGM2008 defines the potential of gravitational force in
    terms of spherical harmonics. But that definition is out of the scope of
    this package.

References
----------
.. [WGS84] World Geodetic System 1984. Its Definition and Relationships with
    Local Geodetic Systems. National Geospatial-Intelligence Agency (NGA)
    Standarization Document. 2014.
    (ftp://ftp.nga.mil/pub2/gandg/website/wgs84/NGA.STND.0036_1.0.0_WGS84.pdf)
.. [Heiskanen] Heiskanen, W. A. and Moritz, H. Physical Geodesy. W. H. Freeman
    and Company. 1967.
.. [WELMEC2021] WELMEC Directives 2014/31/EU and 2014/32/EU: Common Application.
    Non-Automatic Weighing Instruments (NAWI); Automatic Weighing Instruments
    (AWI); Multi-dimensional Measuring Instruments (MDMI.)
    (https://www.welmec.org/welmec/documents/guides/2/2021/WELMEC_Guide_2_v2021.pdf)
.. [Lambert] Walter D. Lambert. The International Gravity Formula. U.S. Coast
    and Geodetic Survey. 1945. (http://earth.geology.yale.edu/~ajs/1945A/360.pdf)
.. [Somigliana1929] Carlo Somigliana. Teoria generale del campo gravitazionale
    dell'ellissoide di rotazione. Memorie della Società Astronomia Italiana,
    Vol. 4, p.425. 1929
    (http://articles.adsabs.harvard.edu/pdf/1929MmSAI...4..425S)
.. [Tscherning] C. Tscherning. The Geodesist's Handbook 1984. Association
    Internationale de Géodésie. 1984.
    (https://office.iag-aig.org/doc/5d7f91ee333a3.pdf)
"""

import unittest
import numpy as np
from ..common.constants import *

def international_gravity(lat: float, epoch: str = '1980') -> float:
    """
    International Gravity Formula

    Estimate the normal gravity, :math:`g`, using the International Gravity
    Formula [Lambert]_, adapted from Stokes' formula, and adopted by the
    `International Association of Geodesy <https://www.iag-aig.org/>`_ at its
    Stockholm Assembly in 1930.

    The expression for gravity on a spheroid, which combines gravitational
    attraction and centrifugal acceleration, at a certain latitude,
    :math:`\\phi`, can be written in the form of a series:

    .. math::
        g = g_e\\big(1 + \\beta\\sin^2(\\phi) - \\beta_1\\sin^2(2\\phi)
        - \\beta_2\\sin^2(\\phi)\\sin^2(2\\phi)
        - \\beta_3\\sin^4(\\phi)\\sin^2(2\\phi) - \\dots\\big)

    where the values of the :math:`\\beta`'s are:

    .. math::
        \\begin{array}{ll}
        \\beta &= \\frac{5}{2}m\\Big(1-\\frac{17}{35}f - \\frac{1}{245}f^2
        - \\frac{13}{18865}f^3 - \\dots\\Big) - f \\\\
        \\beta_1 &= \\frac{1}{8}f(f+2\\beta) \\\\
        \\beta_2 &= \\frac{1}{8}f^2(2f+3\\beta)
        - \\frac{1}{32}f^3(3f+4\\beta) \\\\
        & \\vdots \\\\
        & \\mathrm{etc.}
        \\end{array}

    and :math:`g_e` is the measured normal gravitaty on the Equator. For the
    case of the International Ellipsoid, the third-order terms are negligible.
    So, in practice, the term :math:`\\beta_2` and all following terms are
    dropped to yield the form:

    .. math::
        g = g_e \\big(1 + \\beta \\sin^2\\phi - \\beta_1 \\sin^2(2\\phi)\\big)

    In the original definition the values of :math:`\\beta` and :math:`\\beta_1`
    are rounded off to seven decimal places to simply get the working formula:

    .. math::
        g = 9.78049 \\big(1 + 0.0052884 \\sin^2\\phi - 0.0000059 \\sin^2(2\\phi)\\big)

    Originally, the definitions of the elementary properties (:math:`a`,
    :math:`g_e`, etc.) weren't as accurate as now. At different moments in
    history, the values were updated to improve the accuracy of the formula.
    Those different moments are named **epochs** and are labeled according to
    the year they were updated:

    =====  ===========  ===============  ===========
    epoch  :math:`g_e`  :math:`\\beta`    :math:`\\beta_1`
    =====  ===========  ===============  ===========
    1930   9.78049      5.2884 x 10^-3   5.9 x 10^-6
    1948   9.780373     5.2891 x 10^-3   5.9 x 10^-6
    1967   9.780318     5.3024 x 10^-3   5.9 x 10^-6
    1980   9.780327     5.3024 x 10^-3   5.8 x 10^-6
    =====  ===========  ===============  ===========

    The latest epoch, 1980, is used here by default.

    Parameters
    ----------
    lat : float
        Geographical Latitude, in decimal degrees.
    epoch : str, default: '1980'
        Epoch of the Geodetic Reference System. Options are ``'1930'``,
        ``'1948'``, ``'1967'`` and ``'1980'``.

    Return
    ------
    g : float
        Normal gravity, in m/s^2, at given latitude.

    Examples
    --------
    >>> ahrs.utils.international_gravity(10.0)
    9.781884110728155
    >>> ahrs.utils.international_gravity(10.0, epoch='1930')
    9.7820428934191

    """
    if abs(lat) > 90.0:
        raise ValueError("Latitude must be between -90.0 and 90.0 degrees.")
    if epoch not in ['1930', '1948', '1967', '1980']:
        raise ValueError("Invalid epoch. Try '1930', '1948', '1967' or '1980'.")
    # Note: From Python 3.10 it is possible to use Structural Pattern Matching.
    g_e, b1, b2 = 9.780327, 5.3024e-3, 5.8e-6
    if epoch == '1930':
        g_e, b1, b2 = 9.78049, 5.2884e-3, 5.9e-6
    if epoch == '1948':
        g_e, b1, b2 = 9.780373, 5.2891e-3, 5.9e-6
    if epoch == '1967':
        g_e, b1, b2 = 9.780318, 5.3024e-3, 5.9e-6
    lat *= DEG2RAD
    return g_e*(1.0 + b1*np.sin(lat)**2 - b2*np.sin(2.0*lat)**2)

def welmec_gravity(lat: float, h: float = 0.0) -> float:
    """
    Reference normal gravity of WELMEC's gravity zone

    Gravity zones are implemented by European States on their territories for
    weighing instruments that are sensitive to variations of gravity [WELMEC2021]_.

    Manufacturers may adjust their instruments using the reference gravity
    formula:

    .. math::
        g = 9.780318(1 + 0.0053024\\sin^2(\\phi) - 0.0000058\\sin^2(2\\phi)) - 0.000003085h

    where :math:`\\phi` is the geographical latitude and :math:`h` is the
    height above sea level in meters.

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
    >>> ahrs.utils.welmec_gravity(52.3, 80.0)      # latitude = 52.3°, height = 80 m
    9.812483709897048

    """
    if abs(lat) > 90.0:
        raise ValueError("Latitude must be between -90.0 and 90.0 degrees.")
    lat *= DEG2RAD
    return 9.780318*(1.0 + 0.0053024*np.sin(lat)**2 - 0.0000058*np.sin(2.0*lat)**2) - 0.000003085*h

class WGS:
    """
    World Geodetic System 1984

    Parameters
    ----------
    a : float, default: 6378137.0
        Ellipsoid's Semi-major axis (Equatorial Radius), in meters. Defaults to
        Earth's semi-major axis.
    f : float, default: 0.0033528106647474805
        Ellipsoid's flattening factor. Defaults to Earth's flattening.
    GM : float, default: 3.986004418e14
        Ellipsoid's Standard Gravitational Constant in m^3/s^2
    w : float, default: 0.00007292115
        Ellipsoid's rotation rate in rad/s

    Attributes
    ----------
    a : float
        Ellipsoid's semi-major axis (Equatorial Radius), in meters.
    f : float
        Ellipsoid's flattening factor.
    gm : float
        Ellipsoid's Standard Gravitational Constant in m^3/s^2.
    w : float
        Ellipsoid's rotation rate in rad/s.
    b : float
        Ellipsoid's semi-minor axis (Polar Radius), in meters.
    is_geodetic : bool
        Whether the Ellipsoid describes Earth.

    """
    def __init__(self, a: float = EARTH_EQUATOR_RADIUS, f: float = EARTH_FLATTENING, GM: float = EARTH_GM, w: float = EARTH_ROTATION):
        self.a = a
        self.f = f
        self.b = self.a*(1-self.f)
        self.gm = GM
        self.w = w
        self.is_geodetic = np.isclose(self.a, EARTH_EQUATOR_RADIUS)
        self.is_geodetic &= np.isclose(self.f, EARTH_FLATTENING)
        self.is_geodetic &= np.isclose(self.gm, EARTH_GM)
        self.is_geodetic &= np.isclose(self.w, EARTH_ROTATION)

    def normal_gravity(self, lat: float, h: float = 0.0) -> float:
        """
        Normal Gravity on (or above) Ellipsoidal Surface

        Estimate the normal gravity on or above the surface of an ellipsoidal
        body using Somigliana's formula (on surface) and a series expansion
        (above surface).

        Somigliana's closed formula as desribed by H. Moritz in [Tscherning]_ is:

        .. math::
            g = \\frac{ag_e \\cos^2\\phi + bg_p\\sin^2\\phi}{\\sqrt{a^2cos^2\\phi + b^2\\sin^2\\phi}}

        For numerical computations, a more convenient form is:

        .. math::
            g = g_e\\frac{1+k\\sin^2\\phi}{\\sqrt{1-e^2\\sin^2\\phi}}

        with the helper constant :math:`k`:

        .. math::
            k = \\frac{bg_p}{ag_e}-1

        Parameters
        ----------
        lat: float
            Geographical latitude, in decimal degrees.
        h : float, default: 0.0
            Mean sea level height, in meters.

        Return
        ------
        g : float
            Normal gravity at given point in space, in m/s^2.

        Examples
        --------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.normal_gravity(50.0)
        9.810702135603085
        >>> wgs.normal_gravity(50.0, 100.0)
        9.810393625316983

        """
        ge = self.equatorial_normal_gravity
        gp = self.polar_normal_gravity
        if ge is None or gp is None:
            raise ValueError("No valid normal gravity values.")
        lat *= DEG2RAD
        e2 = self.first_eccentricity_squared
        k = (self.b*gp)/(self.a*ge)-1
        sin2 = np.sin(lat)**2
        g = ge*(1+k*sin2)/np.sqrt(1-e2*sin2)                        # Gravity on Ellipsoid Surface (eq. 4-1)
        if h==0.0:
            return g
        # Normal gravity above surface
        m = self.w**2*self.a**2*self.b/self.gm                      # Gravity constant (eq. B-20)
        g *= 1-2*h*(1+self.f+m-2*self.f*sin2)/self.a + 3.0*h**2/self.a**2   # Gravity Above Ellipsoid (eq. 4-3)
        return g

    def vertical_curvature_radius(self, lat: float) -> float:
        """
        Radius of the curvature in the prime vertical, estimated at a given
        latitude, :math:`\\phi`, as:

        .. math::
            R_N = \\frac{a}{\\sqrt{1-e^2\\sin^2\\phi}}

        Parameters
        ----------
        lat : float
            Geographical latitude, in decimal degrees.
        """
        e = np.sqrt(self.first_eccentricity_squared)
        return self.a/np.sqrt(1-e**2*np.sin(lat)**2)

    def meridian_curvature_radius(self, lat: float) -> float:
        """
        Radius of the curvature in the prime meridian, estimated at a given
        latitude, :math:`\\phi`, as:

        .. math::
            R_M = \\frac{a(1-e^2)}{\\sqrt[3]{1-e^2\\sin^2\\phi}}

        Parameters
        ----------
        lat : float
            Geographical latitude, in decimal degrees.
        """
        e = np.sqrt(self.first_eccentricity_squared)
        return self.a*(1-e**2)/np.cbrt(1-e**2*np.sin(lat)**2)

    @property
    def first_eccentricity_squared(self):
        """
        First Eccentricity Squared :math:`e^2` of the ellipsoid, computed as:

        .. math::
            e^2 = 2f - f^2

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.first_eccentricity_squared
        0.0066943799901413165
        """
        return 2*self.f - self.f**2

    @property
    def second_eccentricity_squared(self):
        """
        Second Eccentricity Squared :math:`e'^2`, computed as:

        .. math::
            e'^2 = \\frac{a^2-b^2}{b^2}

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.second_eccentricity_squared
        0.006739496742276434
        """
        return (self.a**2-self.b**2)/self.b**2

    @property
    def linear_eccentricity(self):
        """
        Linear Eccentricity :math:`E`, computed as:

        .. math::
            E = \\sqrt{a^2-b^2}

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.linear_eccentricity
        521854.00842338527
        """
        return np.sqrt(self.a**2-self.b**2)

    @property
    def aspect_ratio(self):
        """
        Aspect Ratio :math:`AR`, computed as:

        .. math::
            AR = \\frac{b}{a}

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.aspect_ratio
        0.9966471893352525
        """
        return self.b/self.a

    @property
    def curvature_polar_radius(self):
        """
        Polar Radius of Curvature :math:`R_P`, computed as:

        .. math::
            \\begin{array}{ll}
            R_P &= \\frac{a^2}{b} = \\frac{a}{\\sqrt{1-e^2}} \\\\
            &= \\frac{a}{1-f}
            \\end{array}

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.curvature_polar_radius
        6399593.625758493
        """
        return self.a/(1-self.f)

    @property
    def arithmetic_mean_radius(self):
        """
        Mean Radius :math:`R_1` of the Three Semi-Axes, computed as:

        .. math::
            R_1 = a\\Big(1-\\frac{f}{3}\\Big)

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.arithmetic_mean_radius
        6371008.771415059
        """
        return self.a*(1-self.f/3)

    @property
    def authalic_sphere_radius(self):
        """
        Radius :math:`R_2` of a Sphere of Equal Area, computed as:

        .. math::
            R_2 = R_P \\Big(1-\\frac{2}{3}e'^2 + \\frac{26}{45}e'^4 - \\frac{100}{189}e'^6 + \\frac{7034}{14175}e'^8 - \\frac{220652}{467775}e'^{10} \\Big)

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.authalic_sphere_radius
        6371007.1809182055
        """
        r = self.curvature_polar_radius
        es = np.sqrt(self.second_eccentricity_squared)
        return r*(1 - 2*es**2/3 + 26*es**4/45 - 100*es**6/189 + 7034*es**8/14175 - 220652*es**10/467775)

    @property
    def equivolumetric_sphere_radius(self):
        """
        Radius :math:`R_3` of a Sphere of Equal Volume, computed as:

        .. math::
            \\begin{array}{ll}
            R_3 &= \\sqrt[3]{a^2b} \\\\
            &= a\\sqrt[3]{1-f}
            \\end{array}

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.equivolumetric_sphere_radius
        6371000.790009159
        """
        return self.a*np.cbrt(1-self.f)

    @property
    def normal_gravity_constant(self):
        """
        Normal Gravity Formula Constant :math:`m`, computed as:

        .. math::
            m = \\frac{\\omega^2a^2b}{GM}

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.normal_gravity_constant
        0.0034497865068408447
        """
        return self.w**2*self.a**2*self.b/self.gm

    @property
    def dynamical_form_factor(self):
        """
        WGS 84 Dynamical Form Factor :math:`J_2`, computed as:

        .. math::
            J_2 = \\frac{e^2}{3} \\Big(1-\\frac{2me'}{15q_0}\\Big)

        where:

        .. math::
            q_0 = \\frac{1}{2}\\Big[\\Big(1+\\frac{3}{e'^2}\\Big)\\arctan e' - \\frac{3}{e'}\\Big]

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.dynamical_form_factor
        0.0010826298213129219
        """
        m = self.normal_gravity_constant
        e2 = self.first_eccentricity_squared
        es = np.sqrt(self.second_eccentricity_squared)
        q0 = 0.5*((1+3/es**2)*np.arctan(es) - 3/es)
        return e2*(1-2*m*es/(15*q0))/3

    @property
    def second_degree_zonal_harmonic(self):
        """
        WGS 84 Second Degree Zonal Harmonic :math:`C_{2,0}`, computed as:

        .. math::
            C_{2,0} = -\\frac{J_2}{\\sqrt{5}}

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.second_degree_zonal_harmonic
        -0.00048416677498482876
        """
        return -self.dynamical_form_factor/np.sqrt(5.0)

    @property
    def normal_gravity_potential(self):
        """
        Normal Gravity Potential :math:`U_0` of the WGS 84 Ellipsoid, computed
        as:

        .. math::
            U_0 = \\frac{GM}{E} \\arctan e' + \\frac{\\omega^2a^2}{3}

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.normal_gravity_potential
        62636851.71456948
        """
        es = np.sqrt(self.second_eccentricity_squared)
        return self.gm*np.arctan(es)/self.linear_eccentricity + self.w**2*self.a**2/3

    @property
    def equatorial_normal_gravity(self):
        """
        Normal Gravity :math:`g_e` at the Equator, in
        :math:`\\frac{\\mathrm{m}}{\\mathrm{s}^2}`, computed as:

        .. math::
            g_e = \\frac{GM}{ab}\\Big(1-m-\\frac{me'q_0'}{6q_0}\\Big)

        where:

        .. math::
            \\begin{array}{ll}
            q_0 &= \\frac{1}{2}\\Big[\\Big(1+\\frac{3}{e'^2}\\Big)\\arctan e' - \\frac{3}{e'}\\Big] \\\\
            q_0' &= 3\\Big[\\Big(1+\\frac{1}{e'^2}\\Big)\\Big(1-\\frac{1}{e'}\\arctan e'\\Big)\\Big] - 1
            \\end{array}

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.equatorial_normal_gravity
        9.78032533590406
        """
        m = self.normal_gravity_constant
        es = np.sqrt(self.second_eccentricity_squared)
        q0 = 0.5*((1 + 3/es**2)*np.arctan(es) - 3/es)
        q0s = 3*((1 + 1/es**2)*(1 - np.arctan(es)/es)) - 1
        return self.gm * (1 - m - m*es*q0s/(6*q0))/(self.a*self.b)

    @property
    def polar_normal_gravity(self):
        """
        Normal Gravity :math:`g_p` at the Pole, in
        :math:`\\frac{\\mathrm{m}}{\\mathrm{s}^2}`, computed as:

        .. math::
            g_p = \\frac{GM}{a^2}\\Big(1+\\frac{me'q_0'}{3q_0}\\Big)

        where:

        .. math::
            \\begin{array}{ll}
            q_0 &= \\frac{1}{2}\\Big[\\Big(1+\\frac{3}{e'^2}\\Big)\\arctan e' - \\frac{3}{e'}\\Big] \\\\
            q_0' &= 3\\Big[\\Big(1+\\frac{1}{e'^2}\\Big)\\Big(1-\\frac{1}{e'}\\arctan e'\\Big)\\Big] - 1
            \\end{array}

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.polar_normal_gravity
        9.832184937863065
        """
        m = self.normal_gravity_constant
        es = np.sqrt(self.second_eccentricity_squared)
        q0 = 0.5*((1 + 3/es**2)*np.arctan(es) - 3/es)
        q0s = 3*((1 + 1/es**2)*(1 - np.arctan(es)/es)) - 1
        return self.gm * (1 + m*es*q0s/(3*q0))/self.a**2

    @property
    def mean_normal_gravity(self):
        """
        Mean Value :math:`\\bar{g}` of Normal Gravity, in
        :math:`\\frac{\\mathrm{m}}{\\mathrm{s}^2}`, computed as:

        .. math::
            \\bar{g} = g_e\\Big(1 + \\frac{1}{6}e^2 + \\frac{1}{3}k + \\frac{59}{360}e^4 + \\frac{5}{18}e^2k + \\frac{2371}{15120}e^6 + \\frac{259}{1080}e^4k + \\frac{270229}{1814400}e^8 + \\frac{9623}{45360}e^6k \\Big)

        where:

        .. math::
            k = \\frac{bg_p - ag_e}{ag_e} = \\frac{bg_p}{ag_e}-1

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.mean_normal_gravity
        9.797643222256516
        """
        e = np.sqrt(self.first_eccentricity_squared)
        gp = self.polar_normal_gravity
        ge = self.equatorial_normal_gravity
        k = (self.b*gp)/(self.a*ge)-1
        g = ge * (1 + e**2/6 + k/3 + 59*e**4/360 + 5*e**2*k/18 + 2371*e**6/15120 + 259*e**4*k/1080 + 270229*e**8/1814400 + 9623*e**6*k/45360)
        return g

    @property
    def mass(self):
        """
        The Mass :math:`M` of the Earth, in kg, computed as:

        .. math::
            M = \\frac{GM}{G}

        where :math:`G` is the universal constant of gravitation equal to
        :math:`6.67428\\times 10^{-11} \\frac{\\mathrm{m}^3}{\\mathrm{kg}\ \\mathrm{s}^2}`

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.mass
        5.972186390142457e+24
        """
        return self.gm/UNIVERSAL_GRAVITATION_WGS84

    @property
    def geometric_inertial_moment_about_Z(self):
        """
        Geometric Moment of Inertia (:math:`C`), with respect to the Z-Axis of
        Rotation, computed as:

        .. math::
            C_{geo} = \\frac{2}{3}Ma^2\\Big(1-\\frac{2}{5}\\sqrt{\\frac{5m}{2f}-1}\\Big)

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.geometric_inertial_moment_about_Z
        8.073029370114392e+37
        """
        return 2*self.mass*self.a**2*(1-0.4*np.sqrt(2.5*self.normal_gravity_constant/self.f - 1))/3

    @property
    def geometric_inertial_moment(self):
        """
        Geometric Moment of Inertia (:math:`A`), with respect to Any Axis in
        the Equatorial Plane, computed as:

        .. math::
            A_{geo} = C_{geo} + \\sqrt{5}Ma^2 C_{2,0geo}

        where :math:`C_{2,0geo} = -4.84166774985\\times 10^{-4}` is Earth's
        Geographic Second Degree Zonal Harmonic.

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.geometric_inertial_moment
        8.046726628049449e+37
        """
        if not self.is_geodetic:
            raise NotImplementedError("The model must be Geodetic.")
        return self.geometric_inertial_moment_about_Z + np.sqrt(5)*self.mass*self.a**2*EARTH_C20_GEO

    @property
    def geometric_dynamic_ellipticity(self):
        """
        Geometric Solution for Dynamic Ellipticity :math:`H`, computed as:

        .. math::
            H_{geo} = \\frac{C_{geo}-A_{geo}}{C_{geo}}

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.geometric_dynamic_ellipticity
        0.003258100628533992
        """
        return (self.geometric_inertial_moment_about_Z - self.geometric_inertial_moment)/self.geometric_inertial_moment_about_Z

    # Geodetic Properties
    @property
    def atmosphere_gravitational_constant(self):
        """
        Gravitational Constant of the Atmosphere :math:`GM_A`, computed as:

        .. math::
            GM_A = G\\ M_A

        where :math:`M_A` is the total mean mass of the atmosphere (with water
        vapor) equal to :math:`5.148\\times 10^{18} \\mathrm{kg}`.

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.atmosphere_gravitational_constant
        343591934.4
        """
        if not self.is_geodetic:
            raise AttributeError("The model is not Geodetic.")
        return UNIVERSAL_GRAVITATION_WGS84*EARTH_ATMOSPHERE_MASS

    @property
    def gravitational_constant_without_atmosphere(self):
        """
        Geocentric Gravitational Constant with Earth's Atmosphere Excluded
        :math:`GM'`, computed as:

        .. math::
            GM' = GM - GM_A

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.gravitational_constant_without_atmosphere
        398600098208065.6
        """
        if not self.is_geodetic:
            raise NotImplementedError("The model must be Geodetic.")
        return self.gm - self.atmosphere_gravitational_constant

    @property
    def dynamic_inertial_moment_about_Z(self):
        """
        Dynamic Moment of Inertia (:math:`C`), with respect to the Z-Axis of
        Rotation, computed as:

        .. math::
            C_{dyn} = -\\sqrt{5}Ma^2\\frac{C_{2,0dyn}}{H}

        where :math:`H=3.2737949 \\times 10^{-3}` is the Dynamic Ellipticity.

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.dynamic_inertial_moment_about_Z
        8.03430094201443e+37
        """
        if not self.is_geodetic:
            raise NotImplementedError("The model must be Geodetic.")
        return -np.sqrt(5)*self.mass*self.a**2*EARTH_C20_DYN/DYNAMIC_ELLIPTICITY

    @property
    def dynamic_inertial_moment_about_X(self):
        """
        Dynamic Moment of Inertia (:math:`A`), with respect to the X-Axis of
        Rotation, computed as:

        .. math::
            A_{dyn} = -\\sqrt{5}Ma^2\\Big[\\Big(1-\\frac{1}{H}\\Big)C_{2,0dyn} - \\frac{C_{2,2dyn}}{\\sqrt{3}}\\Big]

        where :math:`H=3.2737949 \\times 10^{-3}` is the Dynamic Ellipticity.

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.dynamic_inertial_moment_about_X
        8.007921777277886e+37
        """
        if not self.is_geodetic:
            raise NotImplementedError("The model must be Geodetic.")
        return np.sqrt(5)*self.mass*self.a**2*((1-1/DYNAMIC_ELLIPTICITY)*EARTH_C20_DYN - EARTH_C22_DYN/np.sqrt(3))

    @property
    def dynamic_inertial_moment_about_Y(self):
        """
        Dynamic Moment of Inertia (:math:`B`), with respect to the X-Axis of
        Rotation, computed as:

        .. math::
            B_{dyn} = -\\sqrt{5}Ma^2\\Big[\\Big(1-\\frac{1}{H}\\Big)C_{2,0dyn} + \\frac{C_{2,2dyn}}{\\sqrt{3}}\\Big]

        where :math:`H=3.2737949 \\times 10^{-3}` is the Dynamic Ellipticity.

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.dynamic_inertial_moment_about_Y
        8.008074799852911e+37
        """
        if not self.is_geodetic:
            raise NotImplementedError("The model must be Geodetic.")
        return np.sqrt(5)*self.mass*self.a**2*((1-1/DYNAMIC_ELLIPTICITY)*EARTH_C20_DYN + EARTH_C22_DYN/np.sqrt(3))

if __name__ == '__main__':
    unittest.main()

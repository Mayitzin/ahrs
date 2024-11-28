"""
The World Geodetic System 1984 (WGS 84) :cite:p:`wgs84-2014` describes the best
geodetic reference system for the Earth available for the practical
applications of mapping, charting, geopositioning, and navigation, using data,
techniques and technology available through 2013 by the United States of
America's National Geospatial-Intelligence Agency (NGA.)

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
parameters directly.

The class :class:`WGS` already sets these values by default (and estimates the
semi-minor axis ``b``), but with slightly different notation:

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
| C :sub:`2,0 dyn[2008]`   | Dynamic Second Degree Zonal Harmonics     | -4.84165143790815 x 10^-4 |
+--------------------------+-------------------------------------------+---------------------------+
| C :sub:`2,2 dyn[2008]`   | Dynamic Second Degree Sectorial Harmonics | 2.43938357328313 x 10^-6  |
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
:cite:p:`hofmann2006` is:

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
    \\begin{bmatrix}\\mathrm{g}_u \\\\ \\mathrm{g}_\\beta \\\\ \\mathrm{g}_\\lambda \\end{bmatrix} =
    \\begin{bmatrix}
    \\frac{1}{w}\\frac{\\partial U}{\\partial u} \\\\
    \\frac{1}{w\\sqrt{u^2+E^2}}\\frac{\\partial U}{\\partial \\beta} \\\\ 0
    \\end{bmatrix}

We see that :math:`\\mathrm{g}_\\lambda=0`, because the effects of the
potential along the longitudinal direction are neglected.

Normal Gravity on the Surface
-----------------------------

The **magnitude of the normal gravity vector**, simply called **normal
gravity**, is:

.. math::
    \\mathrm{g} = \\|\\vec{\\mathbf{g}}\\| = \\sqrt{\\mathrm{g}_u^2+\\mathrm{g}_\\beta^2}

where :math:`u=b` at the surface of the ellipsoid making :math:`\\mathrm{g}_{\\beta,0}=0`.
So, the total gravity on its surface, :math:`\\mathrm{g}_0`, is just:

.. math::
    \\mathrm{g}_0 = \\|\\vec{\\mathbf{g}_0}\\| =
    \\frac{GM}{a\\sqrt{a^2\\sin^2\\beta+b^2\\cos^2\\beta}}
    \\Big[\\Big(1+\\frac{me'q_0'}{3q_0}\\Big)\\sin^2\\beta + \\Big(1-m-\\frac{me'q_0'}{6q_0}\\Big)\\cos^2\\beta\\Big]

where:

.. math::
    m = \\frac{\\omega^2a^2b}{GM}

and :math:`e'=\\frac{E}{b}` is the **second eccentricity**.

From here we can find two basic variables that will help us to linearly
estimate the normal gravity at any point on the ellipsoid.

Keeping it on the surface, we can estimate the **normal gravity at the equator**:

.. math::
    \\mathrm{g}_e = \\mathrm{g}_0(\\beta=0°) = \\frac{GM}{ab}\\Big(1-m-\\frac{me'q_0'}{6q_0}\\Big)

Similarly, we estimate the **normal gravity at the poles**:

.. math::
    \\mathrm{g}_p = \\mathrm{g}_0(\\beta=90°) = \\frac{GM}{a^2}\\Big(1+\\frac{me'q_0'}{3q_0}\\Big)

With these two basic values, we can linearly approximate the normal at any
latitude on the surface of the ellipsoid, but we need to do it in geographical
coordinates.

Using the property :math:`\\tan\\beta=\\frac{b}{a}\\tan\\phi`, we define a
closed form formula to find the normal gravity :math:`\\mathrm{g}` at any given latitude
:math:`\\phi` :cite:p:`somigliana1929`:

.. math::
    \\mathrm{g}(\\phi) = \\frac{ag_e \\cos^2\\phi + bg_p\\sin^2\\phi}{\\sqrt{a^2\\cos^2\\phi + b^2\\sin^2\\phi}}

For numerical computation, a more convenient form is:

.. math::
    \\mathrm{g}(\\phi) = \\mathrm{g}_e\\frac{1+k\\sin^2\\phi}{\\sqrt{1-e^2\\sin^2\\phi}}

using the helper variable :math:`k = \\frac{b\\mathrm{g}_p}{a\\mathrm{g}_e} - 1`.

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
    \\mathrm{g}(\\phi, h) = \\mathrm{g}(\\phi) \\Big(1 - \\frac{2}{a}\\big(1+f+m-2f\\sin^2\\phi\\big)h + \\frac{3}{a^2}h^2\\Big)

where :math:`h` is the height, in meters, above the ellipsoid's surface.

.. code:: python

    >>> wgs.normal_gravity(50.0, 1000.0)    # Gravity at latitude = 50.0 °, 1000 m above surface
    9.807617683884756

Other Gravitational Methods
---------------------------

The well known **International Gravity Formula** :cite:p:`lambert1945` as
described by Helmut Moritz in 1984 :cite:p:`moritz1984` for the `Geodetic
Reference System 1980 <https://en.wikipedia.org/wiki/Geodetic_Reference_System_1980>`_
is implemented in ``ahrs``:

.. math::
    \\mathrm{g}(\\phi) = 9.780327 (1 + 0.0053024 \\sin^2\\phi - 0.0000058 \\sin^2(2\\phi))

.. code:: python

    >>> ahrs.utils.international_gravity(10.0)
    9.781884110728155

Additionally, the **normal gravity estimation** of the European Cooperation on
Legal Metrology (`WELMEC <https://en.wikipedia.org/wiki/WELMEC>`_) is also
included here:

.. math::
    \\mathrm{g}(\\phi, h) = 9.780318(1 + 0.0053024\\sin^2(\\phi) - 0.0000058\\sin^2(2\\phi)) - 0.000003085h

.. code:: python

    >>> ahrs.utils.welmec_gravity(50.0, 1000.0)     # 50.0° N, 1000 m above sea level
    9.807610187885896

Although this is thought and mainly used for European latitudes.

Advanced Gravitational Methods
------------------------------

All methods above can be used for cartography and basic gravitational
references. For more advanced and precise estimations it is recommended to use
the `EGM2008 <https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/>`_,
which is defined, like the WMM, with spherical harmonics, but of degree 2190
and order 2159.

Because of its high complexity and demanding computation, the implementation of
the EGM2008 is left out of this module in favour of the more convenient
applications developed in `Fortran by the NGA
<https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/egm08_wgs84.html>`_.

Footnotes
---------
.. [#] The most precise EGM2008 defines the potential of gravitational force in
    terms of spherical harmonics. But that definition is out of the scope of
    this package.

"""

import numpy as np
from ..common import DEG2RAD
from .geodesy import ReferenceEllipsoid
from ..common.constants import EARTH_EQUATOR_RADIUS
from ..common.constants import EARTH_FLATTENING
from ..common.constants import EARTH_GM
from ..common.constants import EARTH_ROTATION
from ..common.constants import UNIVERSAL_GRAVITATION_WGS84
from ..common.constants import EARTH_C20_GEO
from ..common.constants import EARTH_ATMOSPHERE_MASS
from ..common.constants import DYNAMIC_ELLIPTICITY
from ..common.constants import EARTH_C20_DYN
from ..common.constants import EARTH_C22_DYN

def international_gravity(lat: float, epoch: str = '1980') -> float:
    """
    International Gravity Formula

    Compute the normal gravity, :math:`\\mathrm{g}`, using the International
    Gravity Formula :cite:p:`lambert1945`, adapted from Stokes' formula, and
    adopted by the `International Association of Geodesy <https://www.iag-aig.org/>`_
    in 1930.

    The expression for gravity on a spheroid, which combines gravitational
    attraction and centrifugal acceleration, at a certain latitude,
    :math:`\\phi`, can be written in the form of a series:

    .. math::
        \\mathrm{g} = \\mathrm{g}_e\\big(1 + \\beta\\sin^2(\\phi) - \\beta_1\\sin^2(2\\phi)
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

    and :math:`\\mathrm{g}_e` is the measured normal gravity on the Equator.
    For the case of the International Ellipsoid, the third-order terms are
    negligible. So, in practice, the term :math:`\\beta_2` and all following
    terms are dropped to yield the form:

    .. math::
        \\mathrm{g} = \\mathrm{g}_e \\big(1 + \\beta \\sin^2\\phi - \\beta_1 \\sin^2(2\\phi)\\big)

    In the original definition the values of :math:`\\beta` and :math:`\\beta_1`
    are rounded off to seven decimal places to simply get the working formula:

    .. math::
        \\mathrm{g} = 9.78049 \\big(1 + 0.0052884 \\sin^2\\phi - 0.0000059 \\sin^2(2\\phi)\\big)

    Originally, the definitions of the elementary properties (:math:`a`,
    :math:`\\mathrm{g}_e`, etc.) weren't as accurate as now. At different
    moments in history, the values were updated to improve the accuracy of the
    formula. Those different moments are named **epochs** and are labeled
    according to the year they were updated:

    =====  ====================  ===============  ===========
    epoch  :math:`\\mathrm{g}_e`  :math:`\\beta`    :math:`\\beta_1`
    =====  ====================  ===============  ===========
    1930   9.78049               5.2884 x 10^-3   5.9 x 10^-6
    1948   9.780373              5.2891 x 10^-3   5.9 x 10^-6
    1967   9.780318              5.3024 x 10^-3   5.9 x 10^-6
    1980   9.780327              5.3024 x 10^-3   5.8 x 10^-6
    =====  ====================  ===============  ===========

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
    weighing instruments that are sensitive to variations of gravity
    :cite:p:`welmec2023`.

    Manufacturers may adjust their instruments using the reference gravity
    formula:

    .. math::
        \\mathrm{g} = 9.780318(1 + 0.0053024\\sin^2(\\phi) - 0.0000058\\sin^2(2\\phi)) - 0.000003085h

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

class WGS(ReferenceEllipsoid):
    """
    World Geodetic System 1984

    This class is a subclass of the :class:`ReferenceEllipsoid` and sets the
    default values for the WGS 84 model:

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

    Examples
    --------
    >>> wgs = ahrs.utils.WGS()
    >>> wgs.a
    6378137.0
    >>> wgs.f
    0.0033528106647474805
    >>> wgs.gm
    398600441800000.0
    >>> wgs.w
    7.292115e-05

    It inherits methods and properties from the :class:`ReferenceEllipsoid`
    class, and adds some specific methods for the WGS 84 model.

    >>> wgs.mass
    5.972186390142457e+24
    >>> wgs.sidereal_day
    86164.09053083288
    >>> wgs.normal_gravity(0.0)     # Normal gravity at Equator (latitude = 0.0 °)
    9.78032533590406
    >>> wgs.normal_gravity(50.0)    # Normal gravity at latitude = 50.0 °
    9.810702135603085
    """
    def __init__(self, a: float = EARTH_EQUATOR_RADIUS, f: float = EARTH_FLATTENING, GM: float = EARTH_GM, w: float = EARTH_ROTATION):
        super().__init__(a, f, GM, w)

    @property
    def geometric_inertial_moment_about_Z(self) -> float:
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
    def geometric_inertial_moment(self) -> float:
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
    def geometric_dynamic_ellipticity(self) -> float:
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
    def mass(self) -> float:
        """
        The Mass :math:`M` of the Earth, in kg, computed as:

        .. math::
            M = \\frac{GM}{G}

        where :math:`G` is the universal constant of gravitation equal to
        :math:`6.67428\\times 10^{-11} \\frac{\\mathrm{m}^3}{\\mathrm{kg} \\mathrm{s}^2}`

        .. note::

            The universal constant of gravitation for WGS84 is not the same as
            the one defined by CODATA :cite:p:`codata2018`
            (:math:`6.67430\\times 10^{-11}`), which is the default value in
            the :class:`ReferenceEllipsoid` class.

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.mass
        5.972186390142457e+24
        """
        return self.gm/UNIVERSAL_GRAVITATION_WGS84

    @property
    def atmosphere_gravitational_constant(self) -> float:
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
    def gravitational_constant_without_atmosphere(self) -> float:
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
    def dynamic_inertial_moment_about_Z(self) -> float:
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
    def dynamic_inertial_moment_about_X(self) -> float:
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
    def dynamic_inertial_moment_about_Y(self) -> float:
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

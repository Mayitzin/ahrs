"""
Geodesy-related functions
"""

import numpy as np
from ..common import DEG2RAD
from ..common.constants import EARTH_EQUATOR_RADIUS
from ..common.constants import EARTH_FLATTENING
from ..common.constants import EARTH_GM
from ..common.constants import EARTH_ROTATION
from ..common.constants import UNIVERSAL_GRAVITATION_CODATA2018

# Default values for a basic reference ellipsoid
DEFAULT_MASS = 1.0  # 1 kg
DEFAULT_STANDARD_GRAVITATIONAL_PARAMETER = UNIVERSAL_GRAVITATION_CODATA2018 * DEFAULT_MASS  # GM

class referenceEllipsoid:
    def __init__(self, a: float = 1.0, f: float = 1.0, GM: float = DEFAULT_STANDARD_GRAVITATIONAL_PARAMETER, w: float = EARTH_ROTATION):
        self.a = a
        self.f = f
        self.gm = GM
        self.w = w
        self.b = self.a*(1-self.f)

    def normal_gravity(self, lat: float, h: float = 0.0) -> float:
        """
        Normal Gravity on (or above) Ellipsoidal Surface

        Estimate the normal gravity on or above the surface of an ellipsoidal
        body using Somigliana's formula (on surface) and a series expansion
        (above surface).

        Somigliana's closed formula as desribed by H. Moritz in
        :cite:p:`moritz1984` is:

        .. math::
            \\mathrm{g} = \\frac{a\\mathrm{g}_e \\cos^2\\phi + b\\mathrm{g}_p\\sin^2\\phi}{\\sqrt{a^2cos^2\\phi + b^2\\sin^2\\phi}}

        For numerical computations, a more convenient form is:

        .. math::
            \\mathrm{g} = \\mathrm{g}_e\\frac{1+k\\sin^2\\phi}{\\sqrt{1-e^2\\sin^2\\phi}}

        with the helper constant :math:`k`:

        .. math::
            k = \\frac{b\\mathrm{g}_p}{a\\mathrm{g}_e}-1

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
        latitude, :math:`\\phi`, as per :cite:p:`jekeli2016`:

        .. math::
            R_M = \\frac{a(1-e^2)}{(1-e^2\\sin^2\\phi)^{\\frac{3}{2}}}

        Parameters
        ----------
        lat : float
            Geographical latitude, in decimal degrees.
        """
        e = np.sqrt(self.first_eccentricity_squared)
        return self.a*(1-e**2)/((1-e**2*np.sin(lat)**2)**(3/2))

    @property
    def first_eccentricity_squared(self) -> float:
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
    def second_eccentricity_squared(self) -> float:
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
    def linear_eccentricity(self) -> float:
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
    def aspect_ratio(self) -> float:
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
    def curvature_polar_radius(self) -> float:
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
    def arithmetic_mean_radius(self) -> float:
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
    def authalic_sphere_radius(self) -> float:
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
    def equivolumetric_sphere_radius(self) -> float:
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
    def normal_gravity_constant(self) -> float:
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
    def dynamical_form_factor(self) -> float:
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
    def second_degree_zonal_harmonic(self) -> float:
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
    def normal_gravity_potential(self) -> float:
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
    def equatorial_normal_gravity(self) -> float:
        """
        Normal Gravity :math:`\\mathrm{g}_e` at the Equator, in
        :math:`\\frac{\\mathrm{m}}{\\mathrm{s}^2}`, computed as:

        .. math::
            \\mathrm{g}_e = \\frac{GM}{ab}\\Big(1-m-\\frac{me'q_0'}{6q_0}\\Big)

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
    def polar_normal_gravity(self) -> float:
        """
        Normal Gravity :math:`\\mathrm{g}_p` at the Pole, in
        :math:`\\frac{\\mathrm{m}}{\\mathrm{s}^2}`, computed as:

        .. math::
            \\mathrm{g}_p = \\frac{GM}{a^2}\\Big(1+\\frac{me'q_0'}{3q_0}\\Big)

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
    def mean_normal_gravity(self) -> float:
        """
        Mean Value :math:`\\bar{\\mathrm{g}}` of Normal Gravity, in
        :math:`\\frac{\\mathrm{m}}{\\mathrm{s}^2}`, computed as:

        .. math::
            \\bar{\\mathrm{g}} = \\mathrm{g}_e\\Big(1 + \\frac{1}{6}e^2 + \\frac{1}{3}k + \\frac{59}{360}e^4 + \\frac{5}{18}e^2k + \\frac{2371}{15120}e^6 + \\frac{259}{1080}e^4k + \\frac{270229}{1814400}e^8 + \\frac{9623}{45360}e^6k \\Big)

        where:

        .. math::
            k = \\frac{b\\mathrm{g}_p - a\\mathrm{g}_e}{a\\mathrm{g}_e} = \\frac{b\\mathrm{g}_p}{a\\mathrm{g}_e}-1

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
    def mass(self) -> float:
        """
        The Mass :math:`M` of the Earth, in kg, computed as:

        .. math::
            M = \\frac{GM}{G}

        where :math:`G` is the universal constant of gravitation equal to
        :math:`6.67428\\times 10^{-11} \\frac{\\mathrm{m}^3}{\\mathrm{kg} \\mathrm{s}^2}`

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.mass
        5.972186390142457e+24
        """
        return self.gm/UNIVERSAL_GRAVITATION_CODATA2018

    @property
    def is_geodetic(self) -> bool:
        """
        Check whether the ellipsoid model is geodetic (describes planet Earth.)

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.is_geodetic
        True
        >>> wgs = ahrs.utils.WGS(a=6_500_000)
        >>> wgs.is_geodetic
        False
        """
        is_geodetic = np.isclose(self.a, EARTH_EQUATOR_RADIUS)
        is_geodetic &= np.isclose(self.f, EARTH_FLATTENING)
        is_geodetic &= np.isclose(self.gm, EARTH_GM)
        is_geodetic &= np.isclose(self.w, EARTH_ROTATION)
        return is_geodetic

    @property
    def sidereal_day(self) -> float:
        """
        Sidereal Day, :math:`T_{sid}`, in seconds, computed as:

        .. math::
            T_{sid} = \\frac{2\\pi}{\\omega}

        Example
        -------
        >>> wgs = ahrs.utils.WGS()
        >>> wgs.sidereal_day
        86164.090530833
        """
        return 2*np.pi/self.w

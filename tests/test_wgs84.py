#!/usr/bin/env python3
import unittest
import ahrs

class TestWELMEC(unittest.TestCase):
    def test_correct_values(self):
        self.assertAlmostEqual(ahrs.utils.welmec_gravity(52.3, 80.0), 9.812484, places=6)     # Braunschweig, Germany
        self.assertAlmostEqual(ahrs.utils.welmec_gravity(60.0, 250.0), 9.818399, places=6)    # Uppsala, Sweden

class TestInternationalGravity(unittest.TestCase):
    def test_correct_values(self):
        self.assertEqual(ahrs.utils.international_gravity(10.0), 9.781884110728155)
        self.assertEqual(ahrs.utils.international_gravity(10.0, epoch='1930'), 9.7820428934191)

class TestWGS84(unittest.TestCase):
    def setUp(self) -> None:
        self.wgs = ahrs.utils.WGS()

    def test_normal_gravity(self):
        self.assertEqual(self.wgs.normal_gravity(50.0), 9.810702135603085)
        self.assertEqual(self.wgs.normal_gravity(50.0, 100.0), 9.810393625316983)
        self.assertAlmostEqual(self.wgs.normal_gravity(0.0), 9.7803253359, places=10)
        self.assertAlmostEqual(self.wgs.normal_gravity(90.0), 9.8321849379, places=10)

    def test_values(self):
        self.assertAlmostEqual(self.wgs.a, 6_378_137.0, 1)
        self.assertAlmostEqual(self.wgs.f, 1/298.257223563, 7)
        self.assertAlmostEqual(self.wgs.gm, 3.986004418e14, 7)
        self.assertAlmostEqual(self.wgs.w, 7.292115e-5, 7)
        self.assertAlmostEqual(self.wgs.b, 6_356_752.3142, 4)
        self.assertAlmostEqual(self.wgs.first_eccentricity_squared, 8.1819190842622e-2**2, 7)
        self.assertAlmostEqual(self.wgs.second_eccentricity_squared, 8.2094437949696e-2**2, 7)
        self.assertAlmostEqual(self.wgs.linear_eccentricity, 5.2185400842339e5, 7)
        self.assertAlmostEqual(self.wgs.aspect_ratio, 9.966471893352525e-1, 10)
        self.assertAlmostEqual(self.wgs.curvature_polar_radius, 6399593.6258, 4)
        self.assertAlmostEqual(self.wgs.arithmetic_mean_radius, 6371008.7714, 4)
        self.assertAlmostEqual(self.wgs.authalic_sphere_radius, 6371007.181, 3)
        self.assertAlmostEqual(self.wgs.equivolumetric_sphere_radius, 6371000.79, 2)
        self.assertAlmostEqual(self.wgs.normal_gravity_constant, 3.44978650684084e-3, 10)
        self.assertAlmostEqual(self.wgs.dynamical_form_factor, 1.082629821313e-3, 10)
        self.assertAlmostEqual(self.wgs.second_degree_zonal_harmonic, -4.84166774985e-4, 10)
        self.assertAlmostEqual(self.wgs.normal_gravity_potential, 6.26368517146e7, 4)
        self.assertAlmostEqual(self.wgs.equatorial_normal_gravity, 9.7803253359, 10)
        self.assertAlmostEqual(self.wgs.polar_normal_gravity, 9.8321849379, 10)
        self.assertAlmostEqual(self.wgs.mean_normal_gravity, 9.7976432223, 10)
        self.assertAlmostEqual(self.wgs.mass, 5.9721864e24, delta=1e17)
        self.assertAlmostEqual(self.wgs.atmosphere_gravitational_constant, 3.43592e8, delta=1e3)
        self.assertAlmostEqual(self.wgs.gravitational_constant_without_atmosphere, 3.986000982e14, delta=1e4)
        self.assertAlmostEqual(self.wgs.dynamic_inertial_moment_about_Z, 8.03430094201443e37, delta=1e30)   # FAILS with report's reference
        self.assertAlmostEqual(self.wgs.dynamic_inertial_moment_about_X, 8.00792178e37, delta=1e29)
        self.assertAlmostEqual(self.wgs.dynamic_inertial_moment_about_Y, 8.0080748e37, delta=1e30)
        self.assertAlmostEqual(self.wgs.geometric_inertial_moment_about_Z, 8.07302937e37, delta=1e29)
        self.assertAlmostEqual(self.wgs.geometric_inertial_moment, 8.04672663e37, delta=1e29)
        self.assertAlmostEqual(self.wgs.geometric_dynamic_ellipticity, 3.2581004e-3, 9)

if __name__ == '__main__':
    unittest.main()

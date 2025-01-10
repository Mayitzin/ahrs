#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

# Often used constants
SQ22 = np.sqrt(2)/2

class TestGeometry(unittest.TestCase):
    def test_geometry_circle_default(self):
        circle = ahrs.common.geometry.circle([0, 0])
        self.assertEqual(circle.max(), 1.0)
        self.assertEqual(circle.min(), -1.0)
        self.assertEqual(np.ptp(circle), 2.0)
        self.assertEqual(circle.shape, (21, 2))
        self.assertAlmostEqual(circle[:-1, 0].mean(), 0.0)
        self.assertAlmostEqual(circle[:-1, 1].mean(), 0.0)

    def test_geometry_circle_custom(self):
        circle = ahrs.common.geometry.circle([1, 1], 2.0, 10)
        self.assertEqual(circle.max(), 3.0)
        self.assertEqual(circle.min(), -1.0)
        self.assertEqual(np.ptp(circle), 4.0)
        self.assertEqual(circle.shape, (11, 2))
        self.assertAlmostEqual(circle[:-1, 0].mean(), 1.0)
        self.assertAlmostEqual(circle[:-1, 1].mean(), 1.0)

    def test_geometry_ellipse_default(self):
        ellipse = ahrs.common.geometry.ellipse([0, 0], 0.0, [1.0, 0.5])
        self.assertEqual(ellipse.max(), 1.0)
        self.assertEqual(ellipse.min(), -1.0)
        self.assertEqual(np.ptp(ellipse), 2.0)
        self.assertEqual(ellipse.shape, (21, 2))
        self.assertAlmostEqual(ellipse[:-1, 0].mean(), 0.0)
        self.assertAlmostEqual(ellipse[:-1, 1].mean(), 0.0)

    def test_geometry_ellipse_custom(self):
        ellipse = ahrs.common.geometry.ellipse([1, 1], 0.0, [2.0, 1.0], 10)
        self.assertEqual(ellipse.max(), 3.0)
        self.assertEqual(ellipse.min(), -1.0)
        self.assertEqual(np.ptp(ellipse), 4.0)
        self.assertEqual(ellipse.shape, (11, 2))
        self.assertAlmostEqual(ellipse[:-1, 0].mean(), 1.0)
        self.assertAlmostEqual(ellipse[:-1, 1].mean(), 1.0)

class TestFrames(unittest.TestCase):
    def setUp(self):
        self.lla_coords = [48.8562, 2.3508, 67.4]
        self.ecef_coords = [4201000, 172460, 4780100]
        self.ecef_decimal_tol = 1   # 1 decimal place = tolerance of 10 cm

    def test_geodetic2ecef(self):
        np.testing.assert_array_almost_equal(ahrs.common.frames.geodetic2ecef(0.0, 0.0, 0.0), [ahrs.EARTH_EQUATOR_RADIUS, 0.0, 0.0])
        np.testing.assert_array_almost_equal(ahrs.common.frames.geodetic2ecef(90.0, 0.0, 0.0), [0.0, 0.0, ahrs.EARTH_POLAR_RADIUS], decimal=4)
        np.testing.assert_array_almost_equal(ahrs.common.frames.geodetic2ecef(57.02929569, 9.950248114, 56.95), [3426949.397, 601195.852, 5327723.994], decimal=3)  # Example from GNU Octave mapping package

    def test_geodetic2ecef_wrong_inputs(self):
        self.assertRaises(ValueError, ahrs.common.frames.geodetic2ecef, 91.0, 0.0, 0.0)
        self.assertRaises(ValueError, ahrs.common.frames.geodetic2ecef, 0.0, 181.0, 0.0)

    def test_ecef2geodetic(self):
        np.testing.assert_array_almost_equal(ahrs.common.frames.ecef2geodetic(*self.ecef_coords), self.lla_coords, decimal=self.ecef_decimal_tol)

    def test_ecef2lla(self):
        np.testing.assert_array_almost_equal(ahrs.common.frames.ecef2lla(*self.ecef_coords), self.lla_coords, decimal=self.ecef_decimal_tol)

    def test_ecef2enu(self):
        np.testing.assert_array_almost_equal(ahrs.common.frames.ecef2enu(660930.192761082, -4701424.222957011, 4246579.604632881, 42, -82, 200),
                                             [186.27752, 286.84222, 939.69262],
                                             decimal=5)
        np.testing.assert_array_almost_equal(ahrs.common.frames.ecef2enu(5507528.9, 4556224.1, 6012820.8, 45.9132, 36.7484, 1877753.2),
                                             [355601.2616, -923083.1558, 1041016.4238],
                                             decimal=4)

    def test_enu2ecef(self):
        np.testing.assert_array_almost_equal(ahrs.common.frames.enu2ecef(186.27752, 286.84222, 939.69262, 42, -82, 200),
                                            [660930.192761082, -4701424.222957011, 4246579.604632881],
                                             decimal=5)
        np.testing.assert_array_almost_equal(ahrs.common.frames.enu2ecef(355601.2616, -923083.1558, 1041016.4238, 45.9132, 36.7484, 1877753.2),
                                             [5507528.9, 4556224.1, 6012820.8],
                                             decimal=4)

    def test_geodetic2enu(self):
        np.testing.assert_array_almost_equal(ahrs.common.frames.geodetic2enu(45.976, 7.658, 4531, 46.017, 7.750, 1673),
                                             [-7134.8, -4556.3, 2852.4],
                                             decimal=1)

    def test_aer2enu(self):
        np.testing.assert_array_almost_equal(ahrs.common.frames.aer2enu(34.1160, 4.1931, 15.1070),
                                             [8.4504, 12.4737, 1.1046],
                                             decimal=4)

    def test_enu2aer(self):
        np.testing.assert_array_almost_equal(ahrs.common.frames.enu2aer(8.4504, 12.4737, 1.1046),
                                             [34.1160, 4.1931, 15.1070],
                                             decimal=4)

class TestMathFuncs(unittest.TestCase):
    def test_sind(self):
        self.assertEqual(ahrs.common.mathfuncs.sind(0.0), 0.0)
        self.assertEqual(ahrs.common.mathfuncs.sind(90.0), 1.0)
        self.assertAlmostEqual(ahrs.common.mathfuncs.sind(-120.0), -0.86602540378)
        np.testing.assert_array_almost_equal(ahrs.common.mathfuncs.sind([0, 45, 90, 180, 270, 360]), [0.0, SQ22, 1.0, 0.0, -1.0, 0.0])

    def test_cosd(self):
        self.assertEqual(ahrs.common.mathfuncs.cosd(0.0), 1.0)
        self.assertAlmostEqual(ahrs.common.mathfuncs.cosd(90.0), 0.0)
        self.assertAlmostEqual(ahrs.common.mathfuncs.cosd(-120.0), -0.5)
        np.testing.assert_array_almost_equal(ahrs.common.mathfuncs.cosd([0, 45, 90, 180, 270, 360]), [1.0, SQ22, 0.0, -1.0, 0.0, 1.0])

    def test_skew(self):
        a = [1, 2, 3]
        skew = ahrs.common.mathfuncs.skew(a)
        self.assertEqual(skew.shape, (3, 3))
        np.testing.assert_allclose(skew, [[0., -3., 2.], [3., 0., -1.], [-2., 1., 0.]])

if __name__ == "__main__":
    unittest.main()

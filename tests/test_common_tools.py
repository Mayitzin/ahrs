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
    def test_geo2rect(self):
        np.testing.assert_allclose(ahrs.common.frames.geo2rect(0.0, 0.0, 0.0), [ahrs.EARTH_EQUATOR_RADIUS, 0.0, 0.0])
        np.testing.assert_array_almost_equal(ahrs.common.frames.geo2rect(90.0, 0.0, 0.0), [0.0, 0.0, ahrs.EARTH_POLAR_RADIUS], decimal=4)

    def test_geo2rect_wrong_inputs(self):
        self.assertRaises(ValueError, ahrs.common.frames.geo2rect, 91.0, 0.0, 0.0, 0.0)
        self.assertRaises(ValueError, ahrs.common.frames.geo2rect, 0.0, 181.0, 0.0, 0.0)

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

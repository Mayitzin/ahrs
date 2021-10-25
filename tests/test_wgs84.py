#!/usr/bin/env python3
import unittest
import ahrs

class TestWELMEC(unittest.TestCase):
    def test_correct_values(self):
        self.assertAlmostEqual(ahrs.utils.welmec_gravity(52.3, 80.0), 9.812484, places=6)

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

if __name__ == '__main__':
    unittest.main()

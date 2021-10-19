#!/usr/bin/env python3
import unittest
import ahrs

class TestWELMEC(unittest.TestCase):
    def test_correct_values(self):
        self.assertAlmostEqual(ahrs.utils.welmec_gravity(52.3, 80.0), 9.812484, places=6)

if __name__ == '__main__':
    unittest.main()

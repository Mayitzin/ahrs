#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

class TestEuclidean(unittest.TestCase):
    def test_correct_values(self):
        self.assertEqual(ahrs.utils.euclidean(np.zeros(3), np.zeros(3)), 0.0)
        self.assertAlmostEqual(ahrs.utils.euclidean(np.zeros(3), np.ones(3)), np.sqrt(3))
        self.assertAlmostEqual(ahrs.utils.euclidean(np.ones(3), -np.ones(3)), 2.0*np.sqrt(3))
        self.assertEqual(ahrs.utils.euclidean(np.array([1, 2, 3]), np.array([4, 5, 6])), 5.196152422706632)
        self.assertGreaterEqual(ahrs.utils.euclidean(np.random.random(3)-0.5, np.random.random(3)-0.5), 0.0)

    def test_invalid_values(self):
        self.assertRaises(ValueError, ahrs.utils.euclidean, np.zeros(3), np.zeros(2))

if __name__ == "__main__":
    unittest.main()

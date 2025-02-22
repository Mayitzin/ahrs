#!/usr/bin/env python3
import unittest
import numpy as np
from ahrs.utils import core as ahrs_core

class TestCoreFunctions(unittest.TestCase):
    def test_assert_same_shapes(self):
        self.assertRaises(ValueError, ahrs_core._assert_same_shapes, [1, 2, 3], [1, 2])

    def test_assert_numerical_iterable(self):
        self.assertRaises(TypeError, ahrs_core._assert_numerical_iterable, [1, 2, 'a'], 'test')

    def test_assert_acc_mag_inputs(self):
        self.assertRaises(ValueError, ahrs_core._assert_acc_mag_inputs, [1, 2, 3], [1, 2])

    def test_get_nan_intervals(self):
        A = np.random.random((10, 3))
        A[[1, 3, 4, 5, 8, 9]] = np.nan
        self.assertEqual(ahrs_core.get_nan_intervals(A), [(1, 1), (3, 5), (8, 9)])

    def test_get_nan_intervals_wrong_input(self):
        A = np.random.random((10, 3))
        A[[1, 3, 4, 5, 8, 9]] = np.nan
        np.testing.assert_array_equal(ahrs_core.get_nan_intervals(A), [(1, 1), (3, 5), (8, 9)])

if __name__ == "__main__":
    unittest.main()

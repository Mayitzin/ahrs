#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

class TestDCM(unittest.TestCase):
    def setUp(self) -> None:
        self.R0 = ahrs.DCM()
        self.vector = np.random.random(3)*180-90
        self.R = ahrs.DCM(rpy=self.vector)

    def test_rotation_matrix_in_SO3(self):
        self.assertAlmostEqual(np.linalg.det(self.R), 1.0)
        np.testing.assert_almost_equal(self.R@self.R.T, np.identity(3))

    def test_identity_rotation_matrix(self):
        np.testing.assert_equal(self.R0, np.identity(3))

    def test_wrong_rpy(self):
        self.assertRaises(TypeError, ahrs.DCM, rpy=45.0)
        self.assertRaises(TypeError, ahrs.DCM, rpy=45)
        self.assertRaises(TypeError, ahrs.DCM, rpy=True)
        self.assertRaises(TypeError, ahrs.DCM, rpy="25.0")
        self.assertRaises(TypeError, ahrs.DCM, rpy=["10.0", "20.0", "30.0"])
        self.assertRaises(ValueError, ahrs.DCM, rpy=np.random.random(4))

if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

GENERATOR = np.random.default_rng(42)
THRESHOLD = 0.1

class TestDCM(unittest.TestCase):
    def setUp(self) -> None:
        self.R0 = ahrs.DCM()
        self.vector = GENERATOR.random(3)*180-90
        self.R = ahrs.DCM(rpy=self.vector)
        self.q = ahrs.Quaternion(GENERATOR.random(4)-0.5)
        self.R1 = ahrs.DCM(q=self.q)

    def test_rotation_matrix_in_SO3(self):
        self.assertAlmostEqual(np.linalg.det(self.R), 1.0)
        np.testing.assert_almost_equal(self.R@self.R.T, np.identity(3))

    def test_identity_rotation_matrix(self):
        np.testing.assert_equal(self.R0, np.identity(3))

    def test_wrong_input_matrix(self):
        self.assertRaises(TypeError, ahrs.DCM, 3)
        self.assertRaises(TypeError, ahrs.DCM, 3.0)
        self.assertRaises(TypeError, ahrs.DCM, "np.eye(3)")
        self.assertRaises(ValueError, ahrs.DCM, np.random.random((3, 3)))
        self.assertRaises(ValueError, ahrs.DCM, -np.identity(3))
        self.assertRaises(ValueError, ahrs.DCM, np.identity(4))

    def test_wrong_rpy(self):
        self.assertRaises(TypeError, ahrs.DCM, rpy=45.0)
        self.assertRaises(TypeError, ahrs.DCM, rpy=45)
        self.assertRaises(TypeError, ahrs.DCM, rpy=True)
        self.assertRaises(TypeError, ahrs.DCM, rpy="25.0")
        self.assertRaises(TypeError, ahrs.DCM, rpy=["10.0", "20.0", "30.0"])
        self.assertRaises(ValueError, ahrs.DCM, rpy=np.random.random(4))

    def test_wrong_euler(self):
        self.assertRaises(TypeError, ahrs.DCM, euler=1)
        self.assertRaises(TypeError, ahrs.DCM, euler=1.0)
        self.assertRaises(TypeError, ahrs.DCM, euler=True)
        self.assertRaises(TypeError, ahrs.DCM, euler="x")
        self.assertRaises(TypeError, ahrs.DCM, euler=["x", "y", "z"])
        self.assertRaises(TypeError, ahrs.DCM, euler=np.random.random(4))
        self.assertRaises(TypeError, ahrs.DCM, euler=([], 3))
        self.assertRaises(ValueError, ahrs.DCM, euler=(3,))

    def test_wrong_quaternion(self):
        self.assertRaises(TypeError, ahrs.DCM, q=3)
        self.assertRaises(TypeError, ahrs.DCM, q=3.0)
        self.assertRaises(TypeError, ahrs.DCM, q="np.eye(3)")
        self.assertRaises(ValueError, ahrs.DCM, q=np.random.random(3))
        self.assertRaises(ValueError, ahrs.DCM, q=np.random.random(5))
        self.assertRaises(ValueError, ahrs.DCM, q=np.random.random((5, 3)))

    def test_default_method_dcm_to_q(self):
        self.assertLessEqual(ahrs.utils.metrics.qad(self.q, self.R1.to_quaternion()), THRESHOLD)

    def test_chiavereini_method_dcm_to_q(self):
        self.assertLessEqual(ahrs.utils.metrics.qad(self.q, self.R1.to_quaternion(method='chiaverini')), THRESHOLD)

if __name__ == "__main__":
    unittest.main()

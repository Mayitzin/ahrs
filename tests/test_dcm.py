#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

GENERATOR = np.random.default_rng(42)
THRESHOLD = 1e-6
SQRT2_2 = np.sqrt(2)/2

class TestDCM(unittest.TestCase):
    def setUp(self) -> None:
        self.R0 = ahrs.DCM()
        self.vector = GENERATOR.random(3)*180-90
        self.R = ahrs.DCM(rpy=self.vector)
        self.q = ahrs.Quaternion(GENERATOR.random(4)-0.5)
        self.R1 = ahrs.DCM(q=self.q)
        self.multi_R = ahrs.DCM(np.atleast_2d([self.R0, self.R, self.R1]))

    def test_rotation_matrix_in_SO3(self):
        self.assertAlmostEqual(np.linalg.det(self.R), 1.0)
        np.testing.assert_almost_equal(self.R@self.R.T, np.identity(3))
        self.assertRaises(ValueError, ahrs.DCM, -self.R)
        self.assertRaises(ValueError, ahrs.DCM, self.R[:2, :2])
        # Test construction of array with many rotation matrices
        self.assertTrue(all(np.allclose(self.multi_R[i].T, rt) for i, rt, in enumerate(np.transpose(self.multi_R, (0, 2, 1)))))
        self.assertRaises(ValueError, ahrs.DCM, GENERATOR.random((4, 3, 3)) - 0.5)
        self.assertRaises(ValueError, ahrs.DCM, GENERATOR.random((4, 2, 2)) - 0.5)
        self.assertRaises(ValueError, ahrs.DCM, np.zeros((4, 3, 3)))
        self.assertRaises(ValueError, ahrs.DCM, np.ones((4, 3, 3)))

    def test_rotation_matrix_from_euler_angles(self):
        np.testing.assert_almost_equal(ahrs.DCM(x=0.0), np.identity(3))
        np.testing.assert_almost_equal(ahrs.DCM(y=0.0), np.identity(3))
        np.testing.assert_almost_equal(ahrs.DCM(z=0.0), np.identity(3))
        rmx45 = np.array([[1.0, 0.0, 0.0], [0.0, SQRT2_2, -SQRT2_2], [0.0, SQRT2_2, SQRT2_2]])
        rmy45 = np.array([[SQRT2_2, 0.0, SQRT2_2], [0.0, 1.0, 0.0], [-SQRT2_2, 0.0, SQRT2_2]])
        rmz45 = np.array([[SQRT2_2, -SQRT2_2, 0.0], [SQRT2_2, SQRT2_2, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_almost_equal(ahrs.DCM(x=45*ahrs.DEG2RAD), rmx45)
        np.testing.assert_almost_equal(ahrs.DCM(y=45*ahrs.DEG2RAD), rmy45)
        np.testing.assert_almost_equal(ahrs.DCM(z=45*ahrs.DEG2RAD), rmz45)
        np.testing.assert_almost_equal(ahrs.DCM(euler=('x', [45*ahrs.DEG2RAD])), rmx45)
        np.testing.assert_almost_equal(ahrs.DCM(euler=('y', [45*ahrs.DEG2RAD])), rmy45)
        np.testing.assert_almost_equal(ahrs.DCM(euler=('z', [45*ahrs.DEG2RAD])), rmz45)
        np.testing.assert_almost_equal(ahrs.DCM(euler=(None, [45*ahrs.DEG2RAD])), rmz45)
        np.testing.assert_almost_equal(ahrs.DCM(euler=(0, [45*ahrs.DEG2RAD])), rmx45)
        np.testing.assert_almost_equal(ahrs.DCM(euler=(1, [45*ahrs.DEG2RAD])), rmy45)
        np.testing.assert_almost_equal(ahrs.DCM(euler=(2, [45*ahrs.DEG2RAD])), rmz45)

    def test_identity_rotation_matrix(self):
        np.testing.assert_equal(self.R0, np.identity(3))
        np.testing.assert_equal(ahrs.DCM(), np.identity(3))
        np.testing.assert_equal(ahrs.DCM(x=0.0), np.identity(3))
        np.testing.assert_equal(ahrs.DCM(y=0.0), np.identity(3))
        np.testing.assert_equal(ahrs.DCM(z=0.0), np.identity(3))
        np.testing.assert_almost_equal(ahrs.DCM(x=360.0*ahrs.DEG2RAD), np.identity(3))
        np.testing.assert_almost_equal(ahrs.DCM(y=360.0*ahrs.DEG2RAD), np.identity(3))
        np.testing.assert_almost_equal(ahrs.DCM(z=360.0*ahrs.DEG2RAD), np.identity(3))

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

    def test_dcm_to_q_default(self):
        self.assertLessEqual(ahrs.utils.metrics.qad(self.q, self.R1.to_quaternion()), THRESHOLD)

    def test_dcm_to_q_chiavereini(self):
        self.assertLessEqual(ahrs.utils.metrics.qad(self.q, self.R1.to_quaternion(method='chiaverini')), THRESHOLD)

    def test_dcm_to_q_hughes(self):
        self.assertLessEqual(ahrs.utils.metrics.qad(self.q, self.R1.to_quaternion(method='hughes')), THRESHOLD)

    def test_dcm_to_q_itzhack(self):
        self.assertLessEqual(ahrs.utils.metrics.qad(self.q, self.R1.to_quaternion(method='itzhack')), THRESHOLD)

    def test_dcm_to_q_itzhack_v1(self):
        self.assertLessEqual(ahrs.utils.metrics.qad(self.q, self.R1.to_quaternion(method='itzhack', version=1)), THRESHOLD)

    def test_dcm_to_q_itzhack_v2(self):
        self.assertLessEqual(ahrs.utils.metrics.qad(self.q, self.R1.to_quaternion(method='itzhack', version=2)), THRESHOLD)

    def test_dcm_to_q_itzhack_v3(self):
        self.assertLessEqual(ahrs.utils.metrics.qad(self.q, self.R1.to_quaternion(method='itzhack', version=3)), THRESHOLD)

    def test_dcm_to_q_sarabandi(self):
        self.assertLessEqual(ahrs.utils.metrics.qad(self.q, self.R1.to_quaternion(method='sarabandi')), THRESHOLD)

    def test_dcm_to_q_shepperd(self):
        self.assertLessEqual(ahrs.utils.metrics.qad(self.q, self.R1.to_quaternion(method='shepperd')), THRESHOLD)

if __name__ == "__main__":
    unittest.main()

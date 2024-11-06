#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

# Often used constants
SQ22 = np.sqrt(2)/2

class TestQuaternion(unittest.TestCase):
    def setUp(self) -> None:
        # Identity Quaternion
        self.q0 = ahrs.Quaternion()
        # Versor from 3D array
        self.vector3 = np.random.random(3)-0.5
        self.q3 = ahrs.Quaternion(self.vector3)
        self.vector3 /= np.linalg.norm(self.vector3)
        # Versor from 4D array
        self.vector4 = np.random.random(4)-0.5
        self.q4 = ahrs.Quaternion(self.vector4)
        self.vector4 /= np.linalg.norm(self.vector4)
        self.decimal_precision = 15

    def test_identity_quaternion(self):
        self.assertEqual(self.q0.w, 1.0)
        self.assertEqual(self.q0.x, 0.0)
        self.assertEqual(self.q0.y, 0.0)
        self.assertEqual(self.q0.z, 0.0)
        np.testing.assert_equal(self.q0.v, np.zeros(3))

    def test_conjugate(self):
        qc = self.q4.conjugate
        self.assertAlmostEqual(qc[0], self.q4.w, places=self.decimal_precision)
        self.assertAlmostEqual(qc[1], -self.q4.x, places=self.decimal_precision)
        self.assertAlmostEqual(qc[2], -self.q4.y, places=self.decimal_precision)
        self.assertAlmostEqual(qc[3], -self.q4.z, places=self.decimal_precision)
        np.testing.assert_almost_equal(qc[1:], -self.q4.v, decimal=self.decimal_precision)

    def test_conj(self):
        qc = self.q4.conj
        self.assertAlmostEqual(qc[0], self.q4.w, places=self.decimal_precision)
        self.assertAlmostEqual(qc[1], -self.q4.x, places=self.decimal_precision)
        self.assertAlmostEqual(qc[2], -self.q4.y, places=self.decimal_precision)
        self.assertAlmostEqual(qc[3], -self.q4.z, places=self.decimal_precision)
        np.testing.assert_almost_equal(qc[1:], -self.q4.v, decimal=self.decimal_precision)

    def test_inverse(self):
        qi = self.q4.inverse
        self.assertAlmostEqual(qi[0], self.q4.w, places=self.decimal_precision)
        self.assertAlmostEqual(qi[1], -self.q4.x, places=self.decimal_precision)
        self.assertAlmostEqual(qi[2], -self.q4.y, places=self.decimal_precision)
        self.assertAlmostEqual(qi[3], -self.q4.z, places=self.decimal_precision)
        np.testing.assert_almost_equal(qi[1:], -self.q4.v, decimal=self.decimal_precision)
        # Not versor
        qn = ahrs.Quaternion([1.0, 2.0, 3.0, 4.0], versor=False)
        expected_inverse = np.array([1.0, -2.0, -3.0, -4.0])/np.linalg.norm([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_almost_equal(qn.inverse, expected_inverse, decimal=self.decimal_precision)

    def test_inv(self):
        qi = self.q4.inv
        self.assertAlmostEqual(qi[0], self.q4.w, places=self.decimal_precision)
        self.assertAlmostEqual(qi[1], -self.q4.x, places=self.decimal_precision)
        self.assertAlmostEqual(qi[2], -self.q4.y, places=self.decimal_precision)
        self.assertAlmostEqual(qi[3], -self.q4.z, places=self.decimal_precision)
        np.testing.assert_almost_equal(qi[1:], -self.q4.v, decimal=self.decimal_precision)
        # Not versor
        qn = ahrs.Quaternion([1.0, 2.0, 3.0, 4.0], versor=False)
        expected_inverse = np.array([1.0, -2.0, -3.0, -4.0])/np.linalg.norm([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_almost_equal(qn.inv, expected_inverse, decimal=self.decimal_precision)

    def test_is_pure(self):
        self.assertFalse(self.q0.is_pure())
        self.assertTrue(self.q3.is_pure())
        self.assertFalse(self.q4.is_pure())

    def test_is_real(self):
        self.assertTrue(self.q0.is_real())
        self.assertFalse(self.q3.is_real())
        self.assertFalse(self.q4.is_real())

    def test_is_versor(self):
        self.assertTrue(self.q0.is_versor())
        self.assertTrue(self.q3.is_versor())
        self.assertTrue(self.q4.is_versor())

    def test_is_identity(self):
        self.assertTrue(self.q0.is_identity())
        self.assertFalse(self.q3.is_identity())
        self.assertFalse(self.q4.is_identity())

    def test_versor_from_3d_array(self):
        self.assertAlmostEqual(self.q3.w, 0.0, places=self.decimal_precision)
        self.assertAlmostEqual(self.q3.x, self.vector3[0], places=self.decimal_precision)
        self.assertAlmostEqual(self.q3.y, self.vector3[1], places=self.decimal_precision)
        self.assertAlmostEqual(self.q3.z, self.vector3[2], places=self.decimal_precision)
        np.testing.assert_almost_equal(self.q3.v, self.vector3, decimal=self.decimal_precision)

    def test_versor_from_4d_array(self):
        self.assertAlmostEqual(self.q4.w, self.vector4[0], places=self.decimal_precision)
        self.assertAlmostEqual(self.q4.x, self.vector4[1], places=self.decimal_precision)
        self.assertAlmostEqual(self.q4.y, self.vector4[2], places=self.decimal_precision)
        self.assertAlmostEqual(self.q4.z, self.vector4[3], places=self.decimal_precision)
        np.testing.assert_almost_equal(self.q4.v, self.vector4[1:], decimal=self.decimal_precision)

    def test_wrong_input_array(self):
        self.assertRaises(TypeError, ahrs.Quaternion, True)
        self.assertRaises(TypeError, ahrs.Quaternion, 3.0)
        self.assertRaises(TypeError, ahrs.Quaternion, "[1.0, 2.0, 3.0, 4.0]")
        self.assertRaises(ValueError, ahrs.Quaternion, np.random.random((6, 3)))
        self.assertRaises(ValueError, ahrs.Quaternion, np.random.random((6, 4)))
        self.assertRaises(ValueError, ahrs.Quaternion, np.random.random(2))
        self.assertRaises(ValueError, ahrs.Quaternion, np.random.random(2).tolist())
        self.assertRaises(ValueError, ahrs.Quaternion, np.zeros(3))

    def test_wrong_input_dcm(self):
        self.assertRaises(TypeError, ahrs.Quaternion, dcm=3)
        self.assertRaises(TypeError, ahrs.Quaternion, dcm=3.0)
        self.assertRaises(TypeError, ahrs.Quaternion, dcm="np.identity(3)")
        self.assertRaises(ValueError, ahrs.Quaternion, dcm=np.random.random((3, 3)))
        self.assertRaises(ValueError, ahrs.Quaternion, dcm=-np.identity(3))

    def test_random_attitudes(self):
        q = ahrs.Quaternion(random=True)
        self.assertEqual(q.shape, (4,))
        self.assertTrue(np.all(np.abs(np.linalg.norm(q)-1) < 1e-15))

    def test_rpy(self):
        np.testing.assert_almost_equal(ahrs.Quaternion(rpy=[np.pi, 0, 0]).to_array(), [0.0, 1.0, 0.0, 0.0], decimal=self.decimal_precision)
        np.testing.assert_almost_equal(ahrs.Quaternion(rpy=[0, np.pi, 0]).to_array(), [0.0, 0.0, 1.0, 0.0], decimal=self.decimal_precision)
        np.testing.assert_almost_equal(ahrs.Quaternion(rpy=[0, 0, np.pi]).to_array(), [0.0, 0.0, 0.0, 1.0], decimal=self.decimal_precision)
        np.testing.assert_almost_equal(ahrs.Quaternion(rpy=[np.pi/2, 0, 0]).to_array(), [SQ22, SQ22, 0.0, 0.0], decimal=self.decimal_precision)
        np.testing.assert_almost_equal(ahrs.Quaternion(rpy=[0, np.pi/2, 0]).to_array(), [SQ22, 0.0, SQ22, 0.0], decimal=self.decimal_precision)
        np.testing.assert_almost_equal(ahrs.Quaternion(rpy=[0, 0, np.pi/2]).to_array(), [SQ22, 0.0, 0.0, SQ22], decimal=self.decimal_precision)
        # Older calls using 'angles' instead of 'rpy'. Just for compatibility
        np.testing.assert_almost_equal(ahrs.Quaternion(angles=[np.pi/2, 0, 0]).to_array(), [SQ22, SQ22, 0.0, 0.0], decimal=self.decimal_precision)
        np.testing.assert_almost_equal(ahrs.Quaternion(angles=[0, np.pi/2, 0]).to_array(), [SQ22, 0.0, SQ22, 0.0], decimal=self.decimal_precision)
        np.testing.assert_almost_equal(ahrs.Quaternion(angles=[0, 0, np.pi/2]).to_array(), [SQ22, 0.0, 0.0, SQ22], decimal=self.decimal_precision)

class TestQuaternionArray(unittest.TestCase):
    def setUp(self) -> None:
        self.Q0 = ahrs.QuaternionArray()
        self.Q1 = ahrs.QuaternionArray(np.identity(4))
        self.decimal_precision = 15

    def test_identity_quaternion(self):
        self.assertEqual(self.Q0.w, [1.0])
        self.assertEqual(self.Q0.x, [0.0])
        self.assertEqual(self.Q0.y, [0.0])
        self.assertEqual(self.Q0.z, [0.0])
        np.testing.assert_equal(self.Q0.v, np.atleast_2d(np.zeros(3)))

    def test_4_by_4_array(self):
        np.testing.assert_equal(self.Q1.w, [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_equal(self.Q1.x, [0.0, 1.0, 0.0, 0.0])
        np.testing.assert_equal(self.Q1.y, [0.0, 0.0, 1.0, 0.0])
        np.testing.assert_equal(self.Q1.z, [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_equal(self.Q1.v, np.eye(4, 3, k=-1))

    def test_quaternion_average(self):
        # Create 10 continuous quaternions representing rotations around the
        # Z-axis between 0 and 90 degrees and average them. The mean quaternion
        # must be equal to the quaternion representing a rotation of 45 degrees
        # around the Z-axis.
        Q = ahrs.QuaternionArray([ahrs.Quaternion(rpy=np.array([0.0, 0.0, x])*ahrs.DEG2RAD) for x in range(0, 100, 10)])
        np.testing.assert_almost_equal(Q.average(), ahrs.Quaternion(rpy=np.array([0.0, 0.0, 45.0])*ahrs.DEG2RAD), decimal=self.decimal_precision)

    def test_wrong_input_array(self):
        self.assertRaises(TypeError, ahrs.QuaternionArray, True)
        self.assertRaises(TypeError, ahrs.QuaternionArray, 3.0)
        self.assertRaises(TypeError, ahrs.QuaternionArray, "[[1.0, 2.0, 3.0, 4.0]]")
        self.assertRaises(ValueError, ahrs.QuaternionArray, np.random.random(4))
        self.assertRaises(ValueError, ahrs.QuaternionArray, [[1., 2., 3., 4.], [1., 2., 3.]])

    def test_is_pure(self):
        Q = ahrs.QuaternionArray([[0., 1., 2., 3.], [1., 2., 3., 4.], [0., 0., 0., 1.]])
        self.assertListEqual(Q.is_pure().tolist(), [True, False, True])

    def test_is_real(self):
        Q = ahrs.QuaternionArray([[1., 0., 0., 0.], [1., 2., 3., 4.], [-1., 0., 0., 0.]])
        self.assertListEqual(Q.is_real().tolist(), [True, False, True])

    def test_is_versor(self):
        Q = ahrs.QuaternionArray([[1., 0., 0., 0.], [1., 2., 3., 4.], [0., 0., 0., 1.]], versors=False)
        self.assertListEqual(Q.is_versor().tolist(), [True, False, True])
        Q = ahrs.QuaternionArray([[1., 0., 0., 0.], [1., 2., 3., 4.], [0., 0., 0., 1.]])
        self.assertListEqual(Q.is_versor().tolist(), [True, True, True])

    def test_is_identity(self):
        Q = ahrs.QuaternionArray([[1., 0., 0., 0.], [1., 2., 3., 4.], [0., 0., 0., 1.]])
        self.assertListEqual(Q.is_identity().tolist(), [True, False, False])
        Q = ahrs.QuaternionArray([[1., 0., 0., 0.], [-1., 0., 0., 0.], [2., 0., 0., 0.]])
        self.assertListEqual(Q.is_identity().tolist(), [True, False, True])

    def test_conjugate(self):
        Q = ahrs.QuaternionArray([[1., 0., 0., 0.], [1., 2., 3., 4.], [0., 0., 0., 1.]])
        Q_conj = ahrs.QuaternionArray([[1., 0., 0., 0.], [1., -2., -3., -4.], [0., 0., 0., -1.]])
        np.testing.assert_almost_equal(Q.conjugate(), Q_conj, decimal=self.decimal_precision)

    def test_conj(self):
        Q = ahrs.QuaternionArray([[1., 0., 0., 0.], [1., 2., 3., 4.], [0., 0., 0., 1.]])
        Q_conj = ahrs.QuaternionArray([[1., 0., 0., 0.], [1., -2., -3., -4.], [0., 0., 0., -1.]])
        np.testing.assert_almost_equal(Q.conj(), Q_conj, decimal=self.decimal_precision)

    def test_to_DCM(self):
        R = self.Q1.to_DCM()
        np.testing.assert_equal(R[0], np.identity(3))
        np.testing.assert_equal(R[1], np.diag([1., -1., -1.]))
        np.testing.assert_equal(R[2], np.diag([-1., 1., -1.]))
        np.testing.assert_equal(R[3], np.diag([-1., -1., 1.]))

    def test_slerp_nan(self):
        Q = ahrs.QuaternionArray([[1., 0., 0., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
        Q[1] = np.nan
        Q_slerp = Q.slerp_nan(inplace=False)
        expected_interpolation = np.array([[1, 0, 0, 0],
                                           [SQ22, 0, 0, SQ22],
                                           [0, 0, 0, 1]])
        np.testing.assert_almost_equal(Q_slerp, expected_interpolation)

class TestSLERP(unittest.TestCase):
    def setUp(self) -> None:
        self.q1 = ahrs.Quaternion([1.0, 0.0, 0.0, 0.0])
        self.q2 = ahrs.Quaternion([0.0, 0.0, 0.0, 1.0])
        self.q3 = ahrs.Quaternion([1.0, 0.0, 0.0, 1.0])
        self.q4 = ahrs.Quaternion([1.0, 0.0, 1.0, 0.0])
        self.q5 = ahrs.Quaternion([1.0, 0.0, 0.0, 0.95])

    def test_slerp_interpolations(self):
        q_slerp = ahrs.common.quaternion.slerp(self.q1, self.q2, [0, 0.5, 1])
        expected_interpolation = np.array([[1, 0, 0, 0],
                                           [SQ22, 0, 0, SQ22],
                                           [0, 0, 0, 1]])
        np.testing.assert_almost_equal(q_slerp, expected_interpolation)
        q_slerp = ahrs.common.quaternion.slerp(self.q3, -self.q4, [0, 0.5, 1])
        expected_interpolation = np.array([[SQ22, 0, 0, SQ22],
                                           [1/np.sqrt(1.5), 0, 0.5/np.sqrt(1.5), 0.5/np.sqrt(1.5)],
                                           [SQ22, 0, SQ22, 0]])
        np.testing.assert_almost_equal(q_slerp, expected_interpolation)

    def test_lerp_interpolations(self):
        q_slerp = ahrs.common.quaternion.slerp(self.q3, self.q5, [0, 0.5, 1])
        midq = np.array([(self.q3[0]+self.q5[0])/2, 0., 0., (self.q3[3]+self.q5[3])/2])
        midq /= np.linalg.norm(midq)
        expected_interpolation = np.vstack((self.q3, midq, self.q5))
        np.testing.assert_almost_equal(q_slerp, expected_interpolation)

class TestRandomAttitudes(unittest.TestCase):
    def setUp(self) -> None:
        self.num_attitudes = 10

    def test_one_random_attitude(self):
        q = ahrs.common.quaternion.random_attitudes(1)
        self.assertEqual(q.shape, (4,))
        rotmat = ahrs.common.quaternion.random_attitudes(1, representation="rotmat")
        self.assertEqual(rotmat.shape, (3, 3))
        self.assertTrue(np.allclose(np.linalg.det(rotmat), 1))
        self.assertTrue(np.allclose(rotmat@rotmat.T, np.identity(3)))

    def test_many_random_attitudes(self):
        q = ahrs.common.quaternion.random_attitudes(self.num_attitudes)
        self.assertEqual(q.shape, (self.num_attitudes, 4))
        self.assertTrue(np.all(np.abs(np.linalg.norm(q, axis=1)-1) < 1e-15))
        rotmats = ahrs.common.quaternion.random_attitudes(self.num_attitudes, representation="rotmat")
        self.assertEqual(rotmats.shape, (self.num_attitudes, 3, 3))
        self.assertTrue(np.allclose(np.linalg.det(rotmats), np.ones(self.num_attitudes)))
        self.assertTrue(np.allclose([x@x.T for x in rotmats], np.identity(3)))

    def test_wrong_input(self):
        self.assertRaises(ValueError, ahrs.common.quaternion.random_attitudes, 0)
        self.assertRaises(ValueError, ahrs.common.quaternion.random_attitudes, -1)
        self.assertRaises(TypeError, ahrs.common.quaternion.random_attitudes, [10])
        self.assertRaises(TypeError, ahrs.common.quaternion.random_attitudes, (10,))
        self.assertRaises(TypeError, ahrs.common.quaternion.random_attitudes, np.array([10]))
        self.assertRaises(TypeError, ahrs.common.quaternion.random_attitudes, 1.0)
        self.assertRaises(TypeError, ahrs.common.quaternion.random_attitudes, "10")
        self.assertRaises(ValueError, ahrs.common.quaternion.random_attitudes, 10, representation="euler")
        self.assertRaises(ValueError, ahrs.common.quaternion.random_attitudes, 10, representation="dcm")
        self.assertRaises(TypeError, ahrs.common.quaternion.random_attitudes, 10, representation=0)
        self.assertRaises(TypeError, ahrs.common.quaternion.random_attitudes, 10, representation=1.0)
        self.assertRaises(TypeError, ahrs.common.quaternion.random_attitudes, 10, representation=["quaternion"])
        self.assertRaises(TypeError, ahrs.common.quaternion.random_attitudes, 10, representation=("quaternion",))

if __name__ == "__main__":
    unittest.main()

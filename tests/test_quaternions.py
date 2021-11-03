#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

class TestQuaternion(unittest.TestCase):
    def setUp(self) -> None:
        self.vector = np.random.random(3)-0.5
        self.vector /= np.linalg.norm(self.vector)
        self.q3 = ahrs.Quaternion(self.vector)

    def test_identity_quaternion(self):
        q = ahrs.Quaternion()
        self.assertEqual(q.w, 1.0)
        self.assertEqual(q.x, 0.0)
        self.assertEqual(q.y, 0.0)
        self.assertEqual(q.z, 0.0)
        np.testing.assert_equal(q.v, np.zeros(3))

    def test_versor_from_3d_array(self):
        self.assertEqual(self.q3.w, 0.0)
        self.assertEqual(self.q3.x, self.vector[0])
        self.assertEqual(self.q3.y, self.vector[1])
        self.assertEqual(self.q3.z, self.vector[2])
        np.testing.assert_equal(self.q3.v, self.vector)

if __name__ == "__main__":
    unittest.main()

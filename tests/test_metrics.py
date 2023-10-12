#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

class TestEuclidean(unittest.TestCase):
    def test_correct_values(self):
        self.assertEqual(ahrs.utils.euclidean(np.zeros(3), np.zeros(3)), 0.0)
        self.assertEqual(ahrs.utils.euclidean(np.zeros(3), np.ones(3)), np.sqrt(3))
        self.assertEqual(ahrs.utils.euclidean(np.ones(3), -np.ones(3)), 2.0*np.sqrt(3))
        self.assertEqual(ahrs.utils.euclidean(np.array([1, 2, 3]), np.array([4, 5, 6])), 5.196152422706632)
        self.assertGreaterEqual(ahrs.utils.euclidean(np.random.random(3)-0.5, np.random.random(3)-0.5), 0.0)

    def test_guard_clauses(self):
        self.assertRaises(ValueError, ahrs.utils.euclidean, np.zeros(3), np.zeros(2))

class TestChordal(unittest.TestCase):
    def setUp(self):
        self.R1 = ahrs.DCM(rpy=np.array([10.0, -20.0, 30.0])*ahrs.DEG2RAD)
        self.R2 = ahrs.DCM(rpy=np.array([-10.0, 20.0, -30.0])*ahrs.DEG2RAD)

    def test_correct_values(self):
        self.assertEqual(ahrs.utils.chordal(np.identity(3), np.identity(3)), 0.0)
        self.assertEqual(ahrs.utils.chordal(np.identity(3), -np.identity(3)), 2.0*np.sqrt(3))
        self.assertGreaterEqual(ahrs.utils.euclidean(np.random.random((3, 3))-0.5, np.random.random((3, 3))-0.5), 0.0)
        self.assertEqual(ahrs.utils.chordal(self.R1, self.R2), 1.6916338074634352)
        self.assertEqual(ahrs.utils.chordal(self.R1.tolist(), self.R2.tolist()), 1.6916338074634352)

    def test_guard_clauses(self):
        self.assertRaises(TypeError, ahrs.utils.chordal, np.identity(3), 3.0)
        self.assertRaises(TypeError, ahrs.utils.chordal, 3.0, np.identity(3))
        self.assertRaises(TypeError, ahrs.utils.chordal, "np.identity(3)", np.identity(3))
        self.assertRaises(TypeError, ahrs.utils.chordal, np.identity(3), "np.identity(3)")
        self.assertRaises(ValueError, ahrs.utils.chordal, np.identity(3), np.identity(2))
        self.assertRaises(ValueError, ahrs.utils.chordal, np.tile(np.identity(3), (2, 1, 1)), np.tile(np.identity(3), (3, 1, 1)))

class TestIdentityDeviation(unittest.TestCase):
    def setUp(self):
        self.R1 = ahrs.DCM(rpy=np.array([10.0, -20.0, 30.0])*ahrs.DEG2RAD)
        self.R2 = ahrs.DCM(rpy=np.array([-10.0, 20.0, -30.0])*ahrs.DEG2RAD)

    def test_correct_values(self):
        self.assertEqual(ahrs.utils.identity_deviation(np.identity(3), np.identity(3)), 0.0)
        self.assertEqual(ahrs.utils.identity_deviation(np.identity(3), -np.identity(3)), 2.0*np.sqrt(3))
        self.assertGreaterEqual(ahrs.utils.identity_deviation(np.random.random((3, 3))-0.5, np.random.random((3, 3))-0.5), 0.0)
        self.assertEqual(ahrs.utils.identity_deviation(self.R1, self.R2), 1.6916338074634352)
        self.assertEqual(ahrs.utils.identity_deviation(self.R1.tolist(), self.R2.tolist()), 1.6916338074634352)

    def test_guard_clauses(self):
        self.assertRaises(TypeError, ahrs.utils.identity_deviation, np.identity(3), 3.0)
        self.assertRaises(TypeError, ahrs.utils.identity_deviation, 3.0, np.identity(3))
        self.assertRaises(TypeError, ahrs.utils.identity_deviation, "np.identity(3)", np.identity(3))
        self.assertRaises(TypeError, ahrs.utils.identity_deviation, np.identity(3), "np.identity(3)")
        self.assertRaises(ValueError, ahrs.utils.identity_deviation, np.identity(3), np.identity(2))
        self.assertRaises(ValueError, ahrs.utils.identity_deviation, np.zeros((3, 3)), np.zeros((2, 2)))

class TestAngularDistance(unittest.TestCase):
    def setUp(self):
        self.R1 = ahrs.DCM(rpy=np.array([10.0, -20.0, 30.0])*ahrs.DEG2RAD)
        self.R2 = ahrs.DCM(rpy=np.array([-10.0, 20.0, -30.0])*ahrs.DEG2RAD)

    def test_correct_values(self):
        self.assertEqual(ahrs.utils.angular_distance(np.identity(3), np.identity(3)), 0.0)
        self.assertGreaterEqual(ahrs.utils.angular_distance(np.random.random((3, 3))-0.5, np.random.random((3, 3))-0.5), 0.0)
        self.assertAlmostEqual(ahrs.utils.angular_distance(self.R1, self.R2), 1.282213683073497)
        self.assertAlmostEqual(ahrs.utils.angular_distance(self.R1.tolist(), self.R2.tolist()), 1.282213683073497)

    def test_guard_clauses(self):
        self.assertRaises(TypeError, ahrs.utils.angular_distance, np.identity(3), 3.0)
        self.assertRaises(TypeError, ahrs.utils.angular_distance, 3.0, np.identity(3))
        self.assertRaises(TypeError, ahrs.utils.angular_distance, "np.identity(3)", np.identity(3))
        self.assertRaises(TypeError, ahrs.utils.angular_distance, np.identity(3), "np.identity(3)")
        self.assertRaises(ValueError, ahrs.utils.angular_distance, np.identity(3), np.identity(2))
        self.assertRaises(ValueError, ahrs.utils.angular_distance, np.zeros((3, 3)), np.zeros((2, 2)))

if __name__ == "__main__":
    unittest.main()

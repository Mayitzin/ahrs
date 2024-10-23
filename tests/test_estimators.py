#!/usr/bin/env python3
import unittest
import numpy as np
import ahrs

THRESHOLD = 0.5

# Generate random attitudes
SENSOR_DATA = ahrs.Sensors(num_samples=1000, in_degrees=False)
REFERENCE_QUATERNIONS = SENSOR_DATA.quaternions
REFERENCE_ROTATIONS = SENSOR_DATA.rotations

class TestTRIAD(unittest.TestCase):
    def setUp(self) -> None:
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_multiple_values(self):
        triad = ahrs.filters.TRIAD(self.accelerometers, self.magnetometers, v1=SENSOR_DATA.reference_gravitational_vector, v2=SENSOR_DATA.reference_magnetic_vector)
        triad_rotations = np.transpose(triad.A, (0, 2, 1))
        self.assertLess(np.nanmean(ahrs.utils.metrics.chordal(REFERENCE_ROTATIONS, triad_rotations)), THRESHOLD)

    def test_wrong_frame(self):
        self.assertRaises(TypeError, ahrs.filters.TRIAD, frame=1.0)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, frame=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, frame='Up')

    def test_wrong_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.TRIAD, w1=1.0)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, w2=1.0)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, w1=1.0, w2=2.0)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, w1="[1., 2., 3.]", w2="[2., 3., 4.]")
        self.assertRaises(TypeError, ahrs.filters.TRIAD, w1=True)
        self.assertRaises(TypeError, ahrs.filters.TRIAD, w2=True)
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1.0, 2.0], w2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=np.copy(1.0), w2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[2.0, 3.0, 4.0], w2=np.copy(1.0))
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=np.zeros(3), w2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[2.0, 3.0, 4.0], w2=np.zeros(3))
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1., 2., 3.], w2=[2., 3., 4.], v1=[1.0, 2.0], v2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1., 2., 3.], w2=[2., 3., 4.], v1=np.copy(1.0), v2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1., 2., 3.], w2=[2., 3., 4.], v1=[2.0, 3.0, 4.0], v2=np.copy(1.0))
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1., 2., 3.], w2=[2., 3., 4.], v1=np.zeros(3), v2=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.TRIAD, w1=[1., 2., 3.], w2=[2., 3., 4.], v1=[2.0, 3.0, 4.0], v2=np.zeros(3))

class TestSAAM(unittest.TestCase):
    def setUp(self) -> None:
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_single_values(self):
        saam_quaternion = ahrs.Quaternion(ahrs.filters.SAAM(self.accelerometers[0], self.magnetometers[0]).Q)
        self.assertLess(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS[0], saam_quaternion.conjugate), THRESHOLD)

    def test_single_values_as_rotation(self):
        saam_rotation = ahrs.filters.SAAM(self.accelerometers[0], self.magnetometers[0], representation='rotmat').A
        self.assertLess(ahrs.utils.metrics.chordal(REFERENCE_ROTATIONS[0], saam_rotation.T), THRESHOLD)

    def test_multiple_values(self):
        saam_quaternions = ahrs.QuaternionArray(ahrs.filters.SAAM(self.accelerometers, self.magnetometers).Q)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, saam_quaternions.conjugate())), THRESHOLD)

    def test_multiple_values_as_rotations(self):
        saam_rotations = ahrs.filters.SAAM(self.accelerometers, self.magnetometers, representation='rotmat').A
        saam_rotations = np.transpose(saam_rotations, (0, 2, 1))
        self.assertLess(np.nanmean(ahrs.utils.metrics.chordal(REFERENCE_ROTATIONS, saam_rotations)), THRESHOLD)

    def test_wrong_input_vectors_in_method_estimate(self):
        saam = ahrs.filters.SAAM()
        self.assertIsNone(saam.estimate(acc=self.accelerometers[0], mag=[0., 0., 0.]))
        self.assertIsNone(saam.estimate(acc=[0., 0., 0.], mag=self.magnetometers[0]))
        self.assertIsNone(saam.estimate(acc=[0., 0., 0.], mag=[0., 0., 0.]))

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_representation(self):
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.accelerometers, mag=self.magnetometers, representation=1.0)
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.accelerometers, mag=self.magnetometers, representation=['quaternion'])
        self.assertRaises(TypeError, ahrs.filters.SAAM, acc=self.accelerometers, mag=self.magnetometers, representation=None)
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=self.accelerometers, mag=self.magnetometers, representation='axisangle')
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=self.accelerometers, mag=self.magnetometers, representation='rpy')
        self.assertRaises(ValueError, ahrs.filters.SAAM, acc=self.accelerometers, mag=self.magnetometers, representation='DCM')
        self.assertRaises(AttributeError, getattr, ahrs.filters.SAAM(self.accelerometers, self.magnetometers, representation='quaternion'), 'A')

class TestFAMC(unittest.TestCase):
    def setUp(self) -> None:
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_single_values(self):
        quaternion_famc = ahrs.Quaternion(ahrs.filters.FAMC(self.accelerometers[0], self.magnetometers[0]).Q)
        self.assertLess(ahrs.utils.metrics.qad(quaternion_famc, REFERENCE_QUATERNIONS[0]), THRESHOLD)

    def test_multiple_values(self):
        quaternions_famc = ahrs.QuaternionArray(ahrs.filters.FAMC(self.accelerometers, self.magnetometers).Q)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(quaternions_famc, REFERENCE_QUATERNIONS)), THRESHOLD)

    def test_wrong_input_vectors_in_method_estimate(self):
        famc = ahrs.filters.FAMC()
        self.assertIsNone(famc.estimate(acc=self.accelerometers[0], mag=[0., 0., 0.]))
        self.assertIsNone(famc.estimate(acc=[0., 0., 0.], mag=self.magnetometers[0]))
        self.assertIsNone(famc.estimate(acc=[0., 0., 0.], mag=[0., 0., 0.]))

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.FAMC, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.FAMC, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FAMC, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_attribute_access(self):
        self.assertRaises(AttributeError, getattr, ahrs.filters.FAMC(self.accelerometers[0], self.magnetometers[0]), 'A')

class TestFLAE(unittest.TestCase):
    def setUp(self) -> None:
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_multiple_values(self):
        orientation = ahrs.filters.FLAE(self.accelerometers, self.magnetometers)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, REFERENCE_QUATERNIONS)), THRESHOLD)

    def test_multiple_values_method_eig(self):
        orientation = ahrs.filters.FLAE(self.accelerometers, self.magnetometers, method='eig')
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, REFERENCE_QUATERNIONS)), THRESHOLD)

    def test_multiple_values_method_newton(self):
        orientation = ahrs.filters.FLAE(self.accelerometers, self.magnetometers, method='newton')
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, REFERENCE_QUATERNIONS)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0], mag=[2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[0.0, 0.0, 0.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0], mag=[0.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[0.0, 0.0, 0.0], mag=[0.0, 0.0, 0.0])

    def test_wrong_input_vectors_in_method_estimate(self):
        flae = ahrs.filters.FLAE()
        self.assertRaises(TypeError, flae.estimate, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, flae.estimate, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, flae.estimate, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, flae.estimate, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, flae.estimate, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, flae.estimate, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, flae.estimate, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, flae.estimate, acc=[1.0, 2.0, 3.0], mag=[2.0, 3.0])
        self.assertRaises(ValueError, flae.estimate, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, flae.estimate, acc=[1.0, 2.0, 3.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(TypeError, flae.estimate, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, flae.estimate, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, flae.estimate, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, flae.estimate, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])
        self.assertRaises(ValueError, flae.estimate, acc=[0.0, 0.0, 0.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, flae.estimate, acc=[1.0, 2.0, 3.0], mag=[0.0, 0.0, 0.0])
        self.assertRaises(ValueError, flae.estimate, acc=[0.0, 0.0, 0.0], mag=[0.0, 0.0, 0.0])

    def test_wrong_method(self):
        self.assertRaises(TypeError, ahrs.filters.FLAE, self.accelerometers, self.magnetometers, method=1)
        self.assertRaises(TypeError, ahrs.filters.FLAE, self.accelerometers, self.magnetometers, method=3.14159)
        self.assertRaises(TypeError, ahrs.filters.FLAE, self.accelerometers, self.magnetometers, method=False)
        self.assertRaises(TypeError, ahrs.filters.FLAE, self.accelerometers, self.magnetometers, method=None)
        self.assertRaises(TypeError, ahrs.filters.FLAE, self.accelerometers, self.magnetometers, method=['symbolic'])
        self.assertRaises(TypeError, ahrs.filters.FLAE, self.accelerometers, self.magnetometers, method=('symbolic',))
        self.assertRaises(ValueError, ahrs.filters.FLAE, self.accelerometers, self.magnetometers, method='some_method')

class TestQUEST(unittest.TestCase):
    def setUp(self) -> None:
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_multiple_values(self):
        quest = ahrs.filters.QUEST(self.accelerometers, self.magnetometers)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, quest.Q)), THRESHOLD)

    def test_different_magnetic_references(self):
        sq22 = np.sqrt(2)/2
        quest = ahrs.filters.QUEST(None, None, magnetic_dip=-45)
        np.testing.assert_almost_equal(quest.m_q, np.array([sq22, 0, -sq22]))
        quest = ahrs.filters.QUEST(None, None, magnetic_dip=-45.0)
        np.testing.assert_almost_equal(quest.m_q, np.array([sq22, 0, -sq22]))
        quest = ahrs.filters.QUEST(None, None, magnetic_dip=[1, 0, -1])
        np.testing.assert_almost_equal(quest.m_q, np.array([sq22, 0, -sq22]))
        quest = ahrs.filters.QUEST(None, None, magnetic_dip=None)
        wmm = ahrs.utils.WMM(latitude=ahrs.MUNICH_LATITUDE, longitude=ahrs.MUNICH_LONGITUDE, height=ahrs.MUNICH_HEIGHT)
        np.testing.assert_almost_equal(quest.m_q, np.array([wmm.X, wmm.Y, wmm.Z])/np.linalg.norm([wmm.X, wmm.Y, wmm.Z]))

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.QUEST, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.QUEST, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.QUEST, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_magnetic_dip(self):
        self.assertRaises(TypeError, ahrs.filters.QUEST, self.accelerometers, self.magnetometers, magnetic_dip='34.5')
        self.assertRaises(TypeError, ahrs.filters.QUEST, self.accelerometers, self.magnetometers, magnetic_dip=False)
        self.assertRaises(TypeError, ahrs.filters.QUEST, self.accelerometers, self.magnetometers, magnetic_dip=['34.5'])
        self.assertRaises(TypeError, ahrs.filters.QUEST, self.accelerometers, self.magnetometers, magnetic_dip=('34.5',))
        self.assertRaises(TypeError, ahrs.filters.QUEST, self.accelerometers, self.magnetometers, magnetic_dip=[None, None, None])
        self.assertRaises(ValueError, ahrs.filters.QUEST, self.accelerometers, self.magnetometers, magnetic_dip=[34.5])

class TestDavenport(unittest.TestCase):
    def setUp(self) -> None:
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_single_values(self):
        orientation = ahrs.filters.Davenport(self.accelerometers[0], self.magnetometers[0])
        self.assertLess(ahrs.utils.metrics.qad(orientation.Q, REFERENCE_QUATERNIONS[0]), THRESHOLD)

    def test_multiple_values(self):
        orientation = ahrs.filters.Davenport(self.accelerometers, self.magnetometers)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation.Q, REFERENCE_QUATERNIONS)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Davenport, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Davenport, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Davenport, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

class TestAQUA(unittest.TestCase):
    def setUp(self) -> None:
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_acc_mag(self):
        aqua_quaternions = ahrs.QuaternionArray(ahrs.filters.AQUA(acc=self.accelerometers, mag=self.magnetometers).Q)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, aqua_quaternions.conjugate())), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.FLAE, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.FLAE, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, frequency=-100.0)

    def test_wrong_input_frame(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, frame=[100.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, frame=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, frame=True)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, frame='NWU')

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, Dt=-0.01)

    def test_wrong_input_alpha(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, alpha="1.0")
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, alpha=[1.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, alpha=(1.0,))
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, alpha=True)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, alpha=0.0)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, alpha=-1.0)

    def test_wrong_input_beta(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, beta="1.0")
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, beta=[1.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, beta=(1.0,))
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, beta=True)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, beta=0.0)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, beta=-1.0)

    def test_wrong_input_threshold(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, threshold="1.0")
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, threshold=[1.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, threshold=(1.0,))
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, threshold=True)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, threshold=0.0)
        self.assertRaises(ValueError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, threshold=-1.0)

    def test_wrong_input_adaptive(self):
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, adaptive=1.0)
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, adaptive="1.0")
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, adaptive=[1.0])
        self.assertRaises(TypeError, ahrs.filters.AQUA, acc=self.accelerometers, mag=self.magnetometers, adaptive=(1.0,))

class TestFQA(unittest.TestCase):
    def setUp(self) -> None:
        self.accelerometers = -np.copy(SENSOR_DATA.accelerometers)  # FQA's reference frame is NWU: g = [0, 0, -1]
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_acc_mag(self):
        fqa = ahrs.filters.FQA(acc=self.accelerometers, mag=self.magnetometers)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, fqa.Q)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.FQA, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=[1.0, 2.0, 3.0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=self.accelerometers, mag=self.magnetometers, mag_ref=1.0)
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=self.accelerometers, mag=self.magnetometers, mag_ref="1.0")
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=self.accelerometers, mag=self.magnetometers, mag_ref=['1.0', '2.0', '3.0'])
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=self.accelerometers, mag=self.magnetometers, mag_ref=('1.0', '2.0', '3.0'))
        self.assertRaises(ValueError, ahrs.filters.FQA, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.FQA, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ahrs.filters.FQA, acc=self.accelerometers, mag=self.magnetometers, mag_ref=[1.0])
        self.assertRaises(ValueError, ahrs.filters.FQA, acc=self.accelerometers, mag=self.magnetometers, mag_ref=(1.0,))
        self.assertRaises(ValueError, ahrs.filters.FQA, acc=self.accelerometers, mag=self.magnetometers, mag_ref=np.zeros(3))

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.FQA, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

class TestMadgwick(unittest.TestCase):
    def setUp(self) -> None:
        # Synthetic sensor data
        self.gyroscopes = np.copy(SENSOR_DATA.gyroscopes)
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_imu(self):
        madgwick_quaternions = ahrs.QuaternionArray(ahrs.filters.Madgwick(gyr=self.gyroscopes, acc=self.accelerometers).Q)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, madgwick_quaternions)), THRESHOLD)

    def test_marg(self):
        madgwick_quaternions = ahrs.QuaternionArray(ahrs.filters.Madgwick(gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers).Q)
        madgwick_quaternions.rotate_by(madgwick_quaternions[0]*np.array([1.0, -1.0, -1.0, -1.0]), inplace=True)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, madgwick_quaternions)), THRESHOLD)

    def test_zeroed_magnetometer(self):
        # It returns estimation with Gyro and Acc only
        madgwick_quaternions_imu = ahrs.QuaternionArray(ahrs.filters.Madgwick(gyr=self.gyroscopes, acc=self.accelerometers).Q)
        zeroed_magnetometers = np.zeros_like(self.magnetometers)
        zeroed_magnetometers[0] = self.magnetometers[0]     # Initial mag value used for initial quaternion estimation
        madgwick_quaternions_marg = ahrs.QuaternionArray(ahrs.filters.Madgwick(gyr=self.gyroscopes, acc=self.accelerometers, mag=zeroed_magnetometers).Q)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(madgwick_quaternions_imu, madgwick_quaternions_marg)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=1.0, acc=self.accelerometers)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr="self.gyroscopes", acc=self.accelerometers)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=True, acc=self.accelerometers)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc="self.accelerometers")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=True)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes[0], acc=self.accelerometers[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes[:2], acc=self.accelerometers, mag=self.magnetometers)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=[1.0, 2.0, 3.0], mag=[2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes[:10], acc=self.accelerometers, mag=self.magnetometers)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers[:10], mag=self.magnetometers)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers[:10])

    def test_wrong_input_in_updateIMU(self):
        madgwick = ahrs.filters.Madgwick()
        np.testing.assert_almost_equal(madgwick.updateIMU(q=[1., 0., 0., 0.], gyr=[0., 0., 0.], acc=self.accelerometers[0]), [1., 0., 0., 0.])
        self.assertRaises(TypeError, madgwick.updateIMU, q=None, gyr=self.gyroscopes[0], acc=self.accelerometers[0])
        self.assertRaises(TypeError, madgwick.updateIMU, q=[1., 0., 0., 0.], gyr=None, acc=self.accelerometers[0])
        self.assertRaises(TypeError, madgwick.updateIMU, q=[1., 0., 0., 0.], gyr=self.gyroscopes[0], acc=None)

    def test_wrong_input_in_updateMARG(self):
        madgwick = ahrs.filters.Madgwick()
        np.testing.assert_almost_equal(madgwick.updateMARG(q=[1., 0., 0., 0.], gyr=[0., 0., 0.], acc=self.accelerometers[0], mag=self.magnetometers[0]), [1., 0., 0., 0.])
        self.assertRaises(TypeError, madgwick.updateMARG, q=None, gyr=self.gyroscopes[0], acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertRaises(TypeError, madgwick.updateMARG, q=[1., 0., 0., 0.], gyr=None, acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertRaises(TypeError, madgwick.updateMARG, q=[1., 0., 0., 0.], gyr=self.gyroscopes[0], acc=None, mag=self.magnetometers[0])
        self.assertRaises(TypeError, madgwick.updateMARG, q=[1., 0., 0., 0.], gyr=self.gyroscopes[0], acc=self.accelerometers[0], mag=None)

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=['1.0', 2.0, 3.0], acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=['1.0', '2.0', '3.0'], acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes[0], acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes[0], acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes[0], acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes[0], acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=-100.0)

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=-0.01)

    def test_wrong_input_gain(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain="0.1")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain=True)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain="0.1")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain=True)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain=-0.1)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain=-0.1)

    def test_wrong_input_gain_imu(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain_imu="0.1")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain_imu=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain_imu=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain_imu=True)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain_imu="0.1")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain_imu=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain_imu=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain_imu=True)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain_imu=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain_imu=-0.1)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain_imu=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain_imu=-0.1)

    def test_wrong_input_gain_marg(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain_marg="0.1")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain_marg=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain_marg=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain_marg=True)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain_marg="0.1")
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain_marg=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain_marg=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain_marg=True)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain_marg=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, gain_marg=-0.1)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain_marg=0.0)
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain_marg=-0.1)

    def test_wrong_initial_quaternion(self):
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=1)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=1.0)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=True)
        self.assertRaises(TypeError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=[1.0])
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.Madgwick, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=np.zeros(4))

class TestMahony(unittest.TestCase):
    def setUp(self) -> None:
        # Synthetic sensor data
        self.gyroscopes = np.copy(SENSOR_DATA.gyroscopes)
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_imu(self):
        orientation = ahrs.QuaternionArray(ahrs.filters.Mahony(gyr=self.gyroscopes, acc=self.accelerometers).Q)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, orientation)), THRESHOLD)

    def test_marg(self):
        orientation = ahrs.QuaternionArray(ahrs.filters.Mahony(gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers).Q)
        orientation.rotate_by(orientation[0]*np.array([1.0, -1.0, -1.0, -1.0]), inplace=True)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, orientation)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=1.0, acc=self.accelerometers)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr="self.gyroscopes", acc=self.accelerometers)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=True, acc=self.accelerometers)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc="self.accelerometers")
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=True)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes[0], acc=self.accelerometers[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes[:2], acc=self.accelerometers, mag=self.magnetometers)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=['1.0', 2.0, 3.0], acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=['1.0', '2.0', '3.0'], acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes[0], acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes[0], acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes[0], acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes[0], acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=-100.0)

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=-0.01)

    def test_wrong_input_kP(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, k_P="0.01")
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, k_P=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, k_P=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, k_P=True)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, k_P=0.0)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, k_P=-0.01)

    def test_wrong_input_kI(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, k_I="0.01")
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, k_I=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, k_I=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, k_I=True)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, k_I=0.0)
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, k_I=-0.01)

    def test_wrong_initial_quaternion(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=1)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=1.0)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=True)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=[1.0])
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=np.zeros(4))

    def test_wrong_initial_bias(self):
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, b0=1)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, b0=1.0)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, b0=True)
        self.assertRaises(TypeError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, b0="[0.0, 0.0, 0.0]")
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, b0=[0.0])
        self.assertRaises(ValueError, ahrs.filters.Mahony, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, b0=np.zeros(4))

class TestFourati(unittest.TestCase):
    def setUp(self) -> None:
        # Synthetic sensor data
        self.gyroscopes = np.copy(SENSOR_DATA.gyroscopes)
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_estimation(self):
        orientation = ahrs.QuaternionArray(ahrs.filters.Fourati(gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers).Q)
        orientation.rotate_by(orientation[0]*np.array([1.0, -1.0, -1.0, -1.0]), inplace=True)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, orientation)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=1.0, acc=self.accelerometers, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr="self.gyroscopes", acc=self.accelerometers, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=True, acc=self.accelerometers, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes[0], acc=self.accelerometers[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes[:2], acc=self.accelerometers, mag=self.magnetometers)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=['1.0', 2.0, 3.0], acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=['1.0', '2.0', '3.0'], acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes[0], acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes[0], acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes[0], acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes[0], acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=-100.0)

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=-0.01)

    def test_wrong_input_gain(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, gain="0.1")
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, gain=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, gain=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, gain=True)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain="0.1")
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain=[0.1])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain=(0.1,))
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain=True)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, gain=0.0)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, gain=-0.1)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain=0.0)
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, gain=-0.1)

    def test_wrong_magnetic_dip(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_dip='34.5')
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_dip=False)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_dip=['34.5'])
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_dip=('34.5',))

    def test_wrong_initial_quaternion(self):
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=1)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=1.0)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=True)
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(TypeError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=['1.0', '0.0', '0.0', '0.0'])
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=[1.0])
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.Fourati, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=np.zeros(4))

class TestEKF(unittest.TestCase):
    def setUp(self) -> None:
        # Synthetic sensor data
        self.gyroscopes = np.copy(SENSOR_DATA.gyroscopes)
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_gyr_acc(self):
        orientation = ahrs.QuaternionArray(ahrs.filters.EKF(gyr=self.gyroscopes, acc=self.accelerometers).Q)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, orientation)), THRESHOLD)

    def test_gyr_acc_mag(self):
        orientation = ahrs.QuaternionArray(ahrs.filters.EKF(gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers).Q)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, orientation)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=1.0, acc=self.accelerometers)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr="self.gyroscopes", acc=self.accelerometers)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=True, acc=self.accelerometers)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc="self.accelerometers")
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=True)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes[0], acc=self.accelerometers[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes[:2], acc=self.accelerometers, mag=self.magnetometers)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=np.zeros(3), mag=self.magnetometers)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=np.zeros(3))

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=['1.0', 2.0, 3.0], acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=['1.0', '2.0', '3.0'], acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes[0], acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes[0], acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes[0], acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes[0], acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=-100.0)

    def test_wrong_input_frame(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frame=1)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frame=1.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frame=True)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frame=['NED'])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frame=('NED',))
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frame='NWU')

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=-0.01)

    def test_wrong_initial_quaternion(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=1)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=1.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=True)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=[1.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=np.zeros(4))

    def test_wrong_magnetic_reference(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref='34.5')
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=False)
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=['34.5'])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=('34.5',))
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=[1.0, 2.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=[[1.0], [2.0], [3.0]])

    def test_wrong_initial_state_covariance_matrix(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, P=1)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, P=1.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, P=True)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, P="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, P=[1.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, P=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, P=np.eye(5, 4))

    def test_wrong_spectral_noises_array(self):
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, noises=1)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, noises=1.0)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, noises=True)
        self.assertRaises(TypeError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, noises="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, noises=[1.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, noises=[1.0, 2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.EKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, noises=np.eye(5, 4))

class TestTilt(unittest.TestCase):
    def setUp(self) -> None:
        # Synthetic sensor data
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_acc_mag(self):
        orientation = ahrs.QuaternionArray(ahrs.filters.Tilt(acc=self.accelerometers, mag=self.magnetometers).Q)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, orientation)), THRESHOLD)

    def test_acc_mag_return_rotmat(self):
        tilt = ahrs.filters.Tilt(acc=SENSOR_DATA.accelerometers, mag=SENSOR_DATA.magnetometers, representation='rotmat')
        self.assertLess(np.nanmean(ahrs.utils.metrics.chordal(REFERENCE_ROTATIONS, tilt.Q)), THRESHOLD)

    def test_acc_only(self):
        sensors = ahrs.Sensors(num_samples=1000, in_degrees=False, yaw=0.0, span=(-np.pi/2, np.pi/2))
        tilt = ahrs.filters.Tilt(acc=sensors.accelerometers)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(sensors.quaternions, tilt.Q)), THRESHOLD)

    def test_acc_only_return_angles(self):
        sensors = ahrs.Sensors(num_samples=1000, in_degrees=False, yaw=0.0, span=(-np.pi/2, np.pi/2))
        tilt = ahrs.filters.Tilt(acc=sensors.accelerometers, representation='angles')
        self.assertLess(np.nanmean(ahrs.utils.metrics.rmse(sensors.ang_pos, tilt.Q)), THRESHOLD)
        self.assertLess(np.nanmean(ahrs.utils.metrics.rmse(sensors.ang_pos, tilt.angles)), THRESHOLD)

    def test_acc_only_return_rotmat(self):
        sensors = ahrs.Sensors(num_samples=1000, in_degrees=False, yaw=0.0, span=(-np.pi/2, np.pi/2))
        tilt = ahrs.filters.Tilt(acc=sensors.accelerometers, representation='rotmat')
        self.assertLess(np.nanmean(ahrs.utils.metrics.chordal(sensors.rotations, tilt.Q)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc="self.accelerometers")
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=True)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=self.accelerometers[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Tilt, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Tilt, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ahrs.filters.Tilt, acc=np.zeros(3), mag=self.magnetometers[0])
        self.assertRaises(ValueError, ahrs.filters.Tilt, acc=self.accelerometers[0], mag=np.zeros(3))

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_zero_magnetic_field(self):
        self.assertRaises(ValueError, ahrs.filters.Tilt, acc=self.accelerometers[0], mag=np.zeros(3))
        tilt = ahrs.filters.Tilt()
        self.assertRaises(ValueError, tilt.estimate, acc=self.accelerometers[0], mag=np.zeros(3))

    def test_method_estimate(self):
        orientation = ahrs.QuaternionArray(ahrs.filters.Tilt(acc=self.accelerometers, mag=self.magnetometers).Q)
        orientation_as_angles = ahrs.filters.Tilt(acc=self.accelerometers, mag=self.magnetometers, representation='angles').Q
        orientation_as_rotmat = ahrs.filters.Tilt(acc=self.accelerometers, mag=self.magnetometers, representation='rotmat').Q
        tilt = ahrs.filters.Tilt()
        first_estimation_quaternion = tilt.estimate(acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(orientation[0], first_estimation_quaternion)), THRESHOLD)
        first_estimation_angles = tilt.estimate(acc=self.accelerometers[0], mag=self.magnetometers[0], representation='angles')
        self.assertLess(np.nanmean(ahrs.utils.metrics.rmse(orientation_as_angles[0], first_estimation_angles)), THRESHOLD)
        first_estimation_rotmat = tilt.estimate(acc=self.accelerometers[0], mag=self.magnetometers[0], representation='rotmat')
        self.assertLess(np.nanmean(ahrs.utils.metrics.chordal(orientation_as_rotmat[0], first_estimation_rotmat)), THRESHOLD)

    def test_wrong_representation(self):
        self.assertRaises(TypeError, ahrs.filters.Tilt, representation=1)
        self.assertRaises(TypeError, ahrs.filters.Tilt, acc=self.accelerometers, mag=self.magnetometers, representation=1)
        self.assertRaises(ValueError, ahrs.filters.Tilt, representation="some_representation")
        self.assertRaises(ValueError, ahrs.filters.Tilt, acc=self.accelerometers, mag=self.magnetometers, representation="some_representation")

class TestComplementary(unittest.TestCase):
    def setUp(self) -> None:
        # Synthetic sensor data
        self.sensors_no_yaw = ahrs.Sensors(num_samples=1000, in_degrees=False, yaw=0.0, span=(-np.pi/2, np.pi/2))
        self.sensors = ahrs.Sensors(num_samples=1000, in_degrees=False, span=(-np.pi/2, np.pi/2))

    def test_imu(self):
        orientation = ahrs.filters.Complementary(gyr=self.sensors_no_yaw.gyroscopes, acc=self.sensors_no_yaw.accelerometers)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.sensors_no_yaw.quaternions, orientation.Q)), THRESHOLD)

    def test_marg(self):
        orientation = ahrs.filters.Complementary(gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(self.sensors.quaternions, orientation.Q)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=1.0, acc=self.sensors.accelerometers)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr="self.sensors.gyroscopes", acc=self.sensors.accelerometers)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=True, acc=self.sensors.accelerometers)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc="self.sensors.accelerometers")
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=True)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=1.0, mag=self.sensors.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc="self.sensors.accelerometers", mag="self.sensors.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes[0], acc=self.sensors.accelerometers[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes[:2], acc=self.sensors.accelerometers, mag=self.sensors.magnetometers)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes[0], acc=np.zeros(3), mag=self.sensors.magnetometers[0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes[0], acc=self.sensors.accelerometers[0], mag=np.zeros(3))

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=['1.0', 2.0, 3.0], acc=self.sensors.accelerometers[0], mag=self.sensors.magnetometers[0])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=['1.0', '2.0', '3.0'], acc=self.sensors.accelerometers[0], mag=self.sensors.magnetometers[0])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes[0], acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes[0], acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes[0], acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes[0], acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes[0], acc=self.sensors.accelerometers[0], mag=self.sensors.magnetometers[0], q0=['1.0', '0.0', '0.0', '0.0'])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes[0], acc=self.sensors.accelerometers[0], mag=self.sensors.magnetometers[0], q0=['1.0', 0.0, 0.0, 0.0])

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, frequency=-100.0)

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, Dt=-0.01)

    def test_wrong_input_gain(self):
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, gain="0.01")
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, gain=[0.01])
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, gain=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, gain=True)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, gain=-0.01)
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, gain=1.01)

    def test_wrong_initial_quaternion(self):
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, q0=1)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, q0=1.0)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, q0=True)
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, q0="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(TypeError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, q0=['1.0', '0.0', '0.0', '0.0'])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, q0=[1.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, q0=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, q0=[1.0, 2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, q0=np.zeros(4))
        self.assertRaises(ValueError, ahrs.filters.Complementary, gyr=self.sensors.gyroscopes, acc=self.sensors.accelerometers, mag=self.sensors.magnetometers, q0=np.identity(4))

class TestOLEQ(unittest.TestCase):
    def setUp(self) -> None:
        # Synthetic sensor data
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_estimation(self):
        orientation = ahrs.QuaternionArray(ahrs.filters.OLEQ(acc=self.accelerometers, mag=self.magnetometers).Q)
        orientation.rotate_by(orientation[0]*np.array([1.0, -1.0, -1.0, -1.0]), inplace=True)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, orientation)), THRESHOLD)

    def test_setting_reference_frames(self):
        oleq = ahrs.filters.OLEQ(magnetic_ref=[0.0, 0.0, 1.0])
        np.testing.assert_array_equal(oleq.m_ref, [0.0, 0.0, 1.0])
        oleq = ahrs.filters.OLEQ(magnetic_ref=np.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(oleq.m_ref, [0.0, 0.0, 1.0])
        oleq = ahrs.filters.OLEQ(magnetic_ref=[0.0, 0.0, 1.0], frame='ENU')
        np.testing.assert_array_equal(oleq.m_ref, [0.0, 0.0, 1.0])
        oleq = ahrs.filters.OLEQ(magnetic_ref=[0.0, 0.0, 1.0], frame='NED')
        np.testing.assert_array_equal(oleq.m_ref, [0.0, 0.0, 1.0])
        sq2 = np.sqrt(2)/2
        oleq = ahrs.filters.OLEQ(magnetic_ref=-45.0)
        np.testing.assert_almost_equal(oleq.m_ref, [-sq2, 0.0, sq2])
        oleq = ahrs.filters.OLEQ(magnetic_ref=-45.0, frame='ENU')
        np.testing.assert_almost_equal(oleq.m_ref, [0.0, sq2, sq2])

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc="self.accelerometers")
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=True)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=np.zeros(3), mag=self.magnetometers)
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=np.zeros(3))

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])

    def test_wrong_magnetic_reference(self):
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref='34.5')
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=False)
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=['34.5'])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=('34.5',))
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=[1.0, 2.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=[0.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=[[1.0], [2.0], [3.0]])

    def test_wrong_input_frame(self):
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, frame=1)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, frame=1.0)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, frame=True)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, frame=['NED'])
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, frame=('NED',))
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, frame='NWU')

    def test_wrong_weights(self):
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights=1)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights=1.0)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights=True)
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights="[1.0, 1.0]")
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights=['1.0', '1.0'])
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights=['1.0', 1.0])
        self.assertRaises(TypeError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights=[1.0, '1.0'])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights=[[1.0], [1.0]])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights=[[1.0, 1.0]])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights=[1.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights=[0.5, -0.5])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights=[0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.OLEQ, acc=self.accelerometers, mag=self.magnetometers, weights=np.zeros(4))

class TestROLEQ(unittest.TestCase):
    def setUp(self) -> None:
        # Synthetic sensor data
        self.gyroscopes = np.copy(SENSOR_DATA.gyroscopes)
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_estimation(self):
        orientation = ahrs.QuaternionArray(ahrs.filters.ROLEQ(gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers).Q)
        orientation.rotate_by(orientation[0]*np.array([1.0, -1.0, -1.0, -1.0]), inplace=True)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, orientation)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=1.0)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc="self.accelerometers")
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=True)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=[1.0, 2.0, 3.0, 4.0], mag=[2.0, 3.0, 4.0, 5.0])

    def test_wrong_input_vector_types(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=['1.0', 2.0, 3.0], acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=['1.0', '2.0', '3.0'], acc=self.accelerometers[0], mag=self.magnetometers[0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes[0], acc=['1.0', 2.0, 3.0], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes[0], acc=[1.0, 2.0, 3.0], mag=['2.0', 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes[0], acc=['1.0', '2.0', '3.0'], mag=[2.0, 3.0, 4.0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes[0], acc=[1.0, 2.0, 3.0], mag=['2.0', '3.0', '4.0'])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes[0], acc=self.accelerometers[0], mag=self.magnetometers[0], q0=['1.0', '0.0', '0.0', '0.0'])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes[0], acc=self.accelerometers[0], mag=self.magnetometers[0], q0=['1.0', 0.0, 0.0, 0.0])

    def test_wrong_magnetic_reference(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref='34.5')
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=False)
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=['34.5'])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=('34.5',))
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=[1.0, 2.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=[0.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, magnetic_ref=[[1.0], [2.0], [3.0]])

    def test_wrong_input_frame(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frame=1)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frame=1.0)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frame=True)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frame=['NED'])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frame=('NED',))
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frame='NWU')

    def test_wrong_weights(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights=1)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights=1.0)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights=True)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights="[1.0, 1.0]")
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights=['1.0', '1.0'])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights=['1.0', 1.0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights=[1.0, '1.0'])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights=[[1.0], [1.0]])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights=[[1.0, 1.0]])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights=[1.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights=[0.5, -0.5])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights=[0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, weights=np.zeros(4))

    def test_wrong_input_frequency(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency="100.0")
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=[100.0])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=(100.0,))
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=True)
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=0.0)
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, frequency=-100.0)

    def test_wrong_input_Dt(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt="0.01")
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=[0.01])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=(0.01,))
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=True)
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=0.0)
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, Dt=-0.01)

    def test_wrong_initial_quaternion(self):
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=1)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=1.0)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=True)
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0="[1.0, 0.0, 0.0, 0.0]")
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=['1.0', '0.0', '0.0', '0.0'])
        self.assertRaises(TypeError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=['1.0', 0.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=[1.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=[1.0, 0.0, 0.0])
        self.assertRaises(ValueError, ahrs.filters.ROLEQ, gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers, q0=np.zeros(4))

class TestFKF(unittest.TestCase):
    def setUp(self) -> None:
        # Synthetic sensor data
        self.gyroscopes = np.copy(SENSOR_DATA.gyroscopes)
        self.accelerometers = np.copy(SENSOR_DATA.accelerometers)
        self.magnetometers = np.copy(SENSOR_DATA.magnetometers)

    def test_estimation(self):
        orientation = ahrs.QuaternionArray(ahrs.filters.FKF(gyr=self.gyroscopes, acc=self.accelerometers, mag=self.magnetometers).Q)
        orientation.rotate_by(orientation[0]*np.array([1.0, -1.0, -1.0, -1.0]), inplace=True)
        self.assertLess(np.nanmean(ahrs.utils.metrics.qad(REFERENCE_QUATERNIONS, orientation)), THRESHOLD)

    def test_wrong_input_vectors(self):
        self.assertRaises(TypeError, ahrs.filters.FKF, gyr=self.gyroscopes, acc=1.0, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.FKF, gyr=self.gyroscopes, acc="self.accelerometers", mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.FKF, gyr=self.gyroscopes, acc=True, mag=self.magnetometers)
        self.assertRaises(TypeError, ahrs.filters.FKF, gyr=self.gyroscopes, acc=1.0, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=2.0)
        self.assertRaises(TypeError, ahrs.filters.FKF, gyr=self.gyroscopes, acc="self.accelerometers", mag="self.magnetometers")
        self.assertRaises(TypeError, ahrs.filters.FKF, gyr=self.gyroscopes, acc=self.accelerometers[0], mag=True)
        self.assertRaises(TypeError, ahrs.filters.FKF, gyr=self.gyroscopes, acc=True, mag=[1.0, 2.0, 3.0])
        self.assertRaises(ValueError, ahrs.filters.FKF, gyr=self.gyroscopes, acc=[1.0, 2.0, 3.0], mag=self.magnetometers)
        self.assertRaises(ValueError, ahrs.filters.FKF, gyr=self.gyroscopes, acc=self.accelerometers, mag=[2.0, 3.0, 4.0])
        self.assertRaises(ValueError, ahrs.filters.FKF, gyr=self.gyroscopes, acc=[1.0, 2.0], mag=[2.0, 3.0, 4.0])

if __name__ == '__main__':
    unittest.main()

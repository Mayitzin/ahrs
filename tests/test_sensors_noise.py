import unittest

import numpy as np

from ahrs import QuaternionArray, Sensors


class TestSensorsMagneticNoise(unittest.TestCase):
    def test_zero_magnetic_noise(self):
        half = np.sqrt(0.5)
        quaternions = QuaternionArray([[1., 0., 0., 0.], [half, 0., 0., half]])
        reference = np.array([2., 3., 4.])
        for normalized in (False, True):
            with self.subTest(normalized=normalized):
                sensors = Sensors(quaternions=quaternions, mag_noise=0.0,
                                  normalized_mag=normalized,
                                  reference_magnetic_vector=reference,
                                  rng=np.random.default_rng(10))
                self.assertEqual(sensors.mag_noise, 0.0)
                for name in ('magnetometers', 'magnetometers_nd', 'magnetometers_enu'):
                    suffix = name.replace('magnetometers', '')
                    vector = getattr(sensors, 'reference_magnetic_vector' + suffix)
                    expected = np.array([vector, [vector[1], -vector[0], vector[2]]])
                    if normalized:
                        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
                    np.testing.assert_allclose(getattr(sensors, name), expected, atol=1e-14)

    def test_requested_noise_scale(self):
        quaternions = QuaternionArray(rpy=np.array([[0., 0., 0.], [0.2, 0.4, 0.6]]))
        simulations = [Sensors(quaternions=quaternions, mag_noise=noise,
                               rng=np.random.default_rng(20)) for noise in (0., 0.5, 1.)]
        for sensors, noise in zip(simulations, (0., 0.5, 1.)):
            self.assertEqual(sensors.mag_noise, noise)
        for name in ('magnetometers', 'magnetometers_nd', 'magnetometers_enu'):
            zero, half, full = [getattr(sensors, name) for sensors in simulations]
            np.testing.assert_allclose(half - zero, 0.5 * (full - zero), atol=1e-11)

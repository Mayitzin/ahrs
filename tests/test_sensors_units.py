import unittest

import numpy as np

from ahrs import QuaternionArray, Sensors


class TestSensorsGyroscopeUnits(unittest.TestCase):
    def test_bias_and_measurement_units(self):
        quaternions = QuaternionArray(rpy=np.array([[0., 0., 0.], [0.1, 0.2, 0.3], [0.4, 0.2, 0.1]]))
        for noise in (0.0, 1.0):
            with self.subTest(noise=noise):
                radians = Sensors(quaternions=quaternions, gyr_noise=noise,
                                  rng=np.random.default_rng(10))
                degrees = Sensors(quaternions=quaternions, gyr_noise=noise,
                                  in_degrees=True, rng=np.random.default_rng(10))
                np.testing.assert_allclose(radians.biases_gyroscopes,
                                           np.deg2rad(degrees.biases_gyroscopes), atol=1e-14)
                np.testing.assert_allclose(radians.gyroscopes,
                                           np.deg2rad(degrees.gyroscopes), atol=1e-14)
                np.testing.assert_array_equal(radians.accelerometers, degrees.accelerometers)
                np.testing.assert_array_equal(radians.magnetometers, degrees.magnetometers)

# -*- coding: utf-8 -*-
"""
Quaternion from gravity acceleration

Attitude quaternion obtained via gravity acceleration measurements.

References
----------
.. [Trimpe] Sebastian Trimpe and Raffaello D'Andrea. The Balancing cube. A
    dynamic sculpture as test bed for distributed estimation and control. IEEE
    Control Systems Magazine. December 2012.
    http://trimpe.is.tuebingen.mpg.de/publications/trimpe-CSM12.pdf
.. [NXP-AN3461] Mark Pedley. Tilt Sensing Using a Three-Axis Accelerometer.
    Freescale Semoconductor Application Note. Document Number: AN3461. 2013.
    https://cache.freescale.com/files/sensors/doc/app_note/AN3461.pdf
.. [NXP-AN4248] Talat Ozyagcilar. Implementing a Tilt-Compensated eCompass using
    Accelerometer and Magnetometer sensors. Freescale Semoconductor Application
    Note. Document Number: AN4248. 2015.
    https://cache.freescale.com/files/sensors/doc/app_note/AN4248.pdf
.. [AD-AN1057] Christopher J. Fisher. Using an Accelerometer for Inclination
    Sensing. Analog Devices. Application Note. AN-1057.
    https://www.analog.com/media/en/technical-documentation/application-notes/AN-1057.pdf
.. [ST-AN4509] Tilt measurement using a low-g 3-axis accelerometer.
    STMicroelectronics. Application note. AN4509. 2014.
    https://www.st.com/resource/en/application_note/dm00119046.pdf
.. [WKDCM2Q] Wikipedia Conversion between quaternions and Euler angles.
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

"""

import numpy as np
from ahrs.common.orientation import *
from ahrs.common import DEG2RAD

class GravityQuaternion:
    """
    Class of Gravity-based estimation of quaternion.
    """
    def __init__(self, *args, **kwargs):
        self.input = args[0] if args else None
        # Data is given
        if self.input:
            self.Q = self.estimate_all()

    def estimate_all(self):
        data = self.input
        Q = np.zeros((data.num_samples, 4))
        for t in range(data.num_samples):
            Q[t] = self.estimate(data.acc[t].copy())
        return Q

    def estimate(self, a):
        """
        Estimate the quaternion from the tilting read by an orthogonal
        tri-axial array of accelerometers.

        The orientation of the roll and pitch angles is estimated using the
        measurements of the accelerometers, and finally converted to a
        quaternion representation according to [WKDCM2Q]_

        Parameters
        ----------
        a : array
            Sample of tri-axial Accelerometer in m/s^2.

        Returns
        -------
        q : array
            Estimated quaternion.

        """
        a_norm = np.linalg.norm(a)
        if a_norm == 0:
            return np.array([1.0, 0.0, 0.0, 0.0])
        a /= a_norm
        ax, ay, az = a
        # Euler Angles from Tilt
        ex = np.arctan2( ay, az)
        ey = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        ez = 0.0
        # Euler to Quaternion
        q = np.array([1.0, 0.0, 0.0, 0.0])
        # Roll
        cx = np.cos(ex/2.0)
        sx = np.sin(ex/2.0)
        # Pitch
        cy = np.cos(ey/2.0)
        sy = np.sin(ey/2.0)
        # Yaw
        cz = np.cos(ez/2.0)
        sz = np.sin(ez/2.0)
        q = np.array([
            cz*cy*cx + sz*sy*sx,
            cz*cy*sx - sz*sy*cx,
            sz*cy*sx + cz*sy*cx,
            sz*cy*cx - cz*sy*sx])
        return q/np.linalg.norm(q)

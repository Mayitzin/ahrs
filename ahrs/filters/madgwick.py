# -*- coding: utf-8 -*-
"""
Filtering routines

"""

import numpy as np
from ahrs.common.orientation import *

def updateIMU(gyr, acc, q, beta=0.1, freq=1.0/256.0):
    """Non-optimized Madgwick's AHRS algorithm with a IMU architecture.

    Adapted to Python from original implementation by Sebastian Madgwick.

    See: http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/
    @author: Sebastian Madgwick (2011)
    See: http://www.olliw.eu/2013/imu-data-fusing/
         https://motsai.com/omid-vs-madgwick-low-power-orientation-filters/
    """
    acc /= np.linalg.norm(acc)
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    # Gradient decent algorithm corrective step
    F = np.asarray([[2.0*(qx*qz - qw*qy)   - acc[0]],
                    [2.0*(qw*qx + qy*qz)   - acc[1]],
                    [2.0*(0.5-qx**2-qy**2) - acc[2]]])
    J = np.asarray([[-2.0*qy, 2.0*qz, -2.0*qw, 2.0*qx],
                    [ 2.0*qx, 2.0*qw,  2.0*qz, 2.0*qy],
                    [ 0.0,   -4.0*qx, -4.0*qy, 0.0   ]])
    step = np.dot(J.transpose(), F)
    step /= np.linalg.norm(step)
    # Compute rate of change of quaternion
    qDot = 0.5 * q_prod(q, [0, gyr[0], gyr[1], gyr[2]]) - beta * step.transpose()
    # Integrate to yield Quaternion
    q = q + qDot/freq
    q /= np.linalg.norm(q)
    return q[0]


def updateMARG(gyr, a, m, q, beta=0.1, samplePeriod=1.0/256.0):
    """Non-optimized Madgwick's AHRS algorithm with a MARG architecture.

    Adapted to Python from original implementation by Sebastian Madgwick.

    See: http://www.x-io.co.uk/open-source-imu-and-ahrs-algorithms/
    @author: Sebastian Madgwick (2011)
    See: http://www.olliw.eu/2013/imu-data-fusing/
         https://motsai.com/omid-vs-madgwick-low-power-orientation-filters/
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    # Normalise accelerometer measurement
    a_norm = np.linalg.norm(a)
    if a_norm == 0:     # handle NaN
        return q
    a /= a_norm
    # Normalise magnetometer measurement
    m_norm = np.linalg.norm(m)
    if m_norm == 0:     # handle NaN
        return q
    m /= m_norm
    # Reference direction of Earth's magnetic field
    h = q_prod(q, q_prod([0, m[0], m[1], m[2]], q_conj(q)))
    b = [0.0, np.linalg.norm([h[1], h[2]]), 0.0, h[3]]
    # Gradient decent algorithm corrective step    
    F = np.array([2.0*(qx * qz - qw * qy)   - a[0],
                  2.0*(qw * qx + qy * qz)   - a[1],
                  2.0*(0.5 - qx**2 - qy**2) - a[2],
                  2.0*b[1]*(0.5 - qy**2 - qz**2) + 2.0*b[3]*(qx*qz - qw*qy)       - m[0],
                  2.0*b[1]*(qx*qy - qw*qz)       + 2.0*b[3]*(qw*qx + qy*qz)       - m[1],
                  2.0*b[1]*(qw*qy + qx*qz)       + 2.0*b[3]*(0.5 - qx**2 - qy**2) - m[2]])
    J = np.array([[-2.0*qy,                 2.0*qz,                  -2.0*qw,                    2.0*qx],
                  [ 2.0*qx,                 2.0*qw,                   2.0*qz,                    2.0*qy],
                  [ 0.0,                   -4.0*qx,                  -4.0*qy,                    0.0],
                  [-2.0*b[3]*qy,            2.0*b[3]*qz,             -4.0*b[1]*qy-2.0*b[3]*qw,  -4.0*b[1]*qz+2.0*b[3]*qx],
                  [-2.0*b[1]*qz+2*b[3]*qx,  2.0*b[1]*qy+2.0*b[3]*qw,  2.0*b[1]*qx+2.0*b[3]*qz,  -2.0*b[1]*qw+2.0*b[3]*qy],
                  [ 2.0*b[1]*qy,            2.0*b[1]*qz-4.0*b[3]*qx,  2.0*b[1]*qw-4.0*b[3]*qy,   2.0*b[1]*qx]])
    step = J.T@F
    step /= np.linalg.norm(step)    # normalise step magnitude
    # Compute rate of change of quaternion
    qDot = 0.5 * q_prod(q, [0, gyr[0], gyr[1], gyr[2]]) - beta * step.T
    # Integrate to yield quaternion
    q += qDot*samplePeriod
    q /= np.linalg.norm(q) # normalise quaternion
    return q


#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Test Mahony Filter

"""

import numpy as np
import ahrs
RAD2DEG = ahrs.common.mathfuncs.RAD2DEG
DEG2RAD = ahrs.common.mathfuncs.DEG2RAD

def test_mahony(**kwargs):
    """
    Test Mahony Filter
    """
    test_file = kwargs.get('file', "ExampleData.mat")
    plot = kwargs.get('plot', False)
    # Load data
    data = ahrs.utils.io.load(test_file)
    # Allocate arrays
    Q = np.tile([1., 0., 0., 0.], (data.num_samples, 1))
    euler_angles = np.zeros((data.num_samples, 3))
    # Mahony object
    mahony = ahrs.filters.Mahony()
    for t in range(1, data.num_samples):
        Q[t] = mahony.updateMARG(DEG2RAD*data.gyr[t], data.acc[t], data.mag[t], Q[t-1])
        # Q[t] = mahony.updateIMU(DEG2RAD*data.gyr[t], data.acc[t], Q[t-1])
        euler_angles[t] = ahrs.common.orientation.q2euler(ahrs.common.orientation.q_conj(Q[t]))*RAD2DEG

    if plot:
        # Plot Signals
        import matplotlib.pyplot as plt
        ahrs.utils.plot_sensors(data.gyr, data.acc, data.mag, x_axis=data.time, title="Sensors: Mahony")
        ahrs.utils.plot_euler(euler_angles, x_axis=data.time, title="Euler Angles: Mahony")
        ahrs.utils.plot_quaternions(Q, x_axis=data.time, title="Quaternions: Mahony")
        plt.show()
    # Test computed data
    qts_ok = not(np.allclose(np.sum(Q, axis=0), data.num_samples*np.array([1., 0., 0., 0.])))
    qnm_ok = np.allclose(np.linalg.norm(Q, axis=1).mean(), 1.0)
    return qts_ok and qnm_ok

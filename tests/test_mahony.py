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
    data = ahrs.utils.io.load(test_file)
    time = data['time']
    gyrs = data['Gyroscope']
    accs = data['Accelerometer']
    mags = data['Magnetometer']

    num_samples = len(time)
    Q = np.tile([1., 0., 0., 0.], (num_samples, 1))
    euler_angles = np.zeros((num_samples, 3))
    # Mahony Object
    mahony = ahrs.filters.Mahony(beta=0.1)
    for t in range(1, num_samples):
        Q[t] = mahony.updateMARG(DEG2RAD*gyrs[t].copy(), accs[t].copy(), mags[t].copy(), Q[t-1].copy())
        # Q[t] = mahony.updateIMU(DEG2RAD*gyrs[t].copy(), accs[t].copy(), Q[t-1].copy())
        euler_angles[t] = ahrs.common.orientation.q2euler(ahrs.common.orientation.q_conj(Q[t]))*RAD2DEG

    if plot:
        # Plot Signals
        import matplotlib.pyplot as plt
        ahrs.utils.plot_sensors(gyrs, accs, mags, x_axis=time, title="Sensors: Mahony")
        ahrs.utils.plot_euler(euler_angles, x_axis=time, title="Euler Angles: Mahony")
        plt.show()

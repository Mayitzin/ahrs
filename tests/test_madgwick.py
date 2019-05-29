# -*- coding: utf-8 -*-
"""
Test Magdwick Filter
====================

"""

import sys, os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import ahrs
RAD2DEG = ahrs.common.mathfuncs.RAD2DEG
DEG2RAD = ahrs.common.mathfuncs.DEG2RAD

def test_madgwick(**kwargs):
    """
    Test
    """
    test_file = kwargs["file"] if "file" in kwargs else "ExampleData.mat"
    if not os.path.isfile(test_file):
        sys.exit("[ERROR] The file {} does not exist.".format(test_file))

    data = sio.loadmat(test_file)
    time = data['time']
    gyrs = data['Gyroscope']
    accs = data['Accelerometer']
    mags = data['Magnetometer']

    num_samples = len(time)
    Q = np.tile([1., 0., 0., 0.], (num_samples, 1))
    euler_angles = np.zeros((num_samples, 3))
    madgwick = ahrs.filters.Madgwick()
    for t in range(1, num_samples):
        Q[t] = madgwick.updateMARG(DEG2RAD*gyrs[t].copy(), accs[t].copy(), mags[t].copy(), Q[t-1])
        # Q[t] = madgwick.updateIMU(DEG2RAD*gyrs[t].copy(), accs[t].copy(), Q[t-1])
        euler_angles[t] = ahrs.common.orientation.q2euler(ahrs.common.orientation.q_conj(Q[t]))*RAD2DEG
    # Plot Signals
    ahrs.utils.plot_sensors(gyrs, accs, mags, time=time, title="Sensors: Madgwick")
    ahrs.utils.plot_euler(euler_angles, time=time, title="Euler Angles: Madgwick")
    plt.show()

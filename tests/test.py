# -*- coding: utf-8 -*-
"""
Test File
=========

"""

import sys
import os

def test_module_imports():
    try:
        import ahrs
    except:
        sys.exit("[ERROR] Package AHRS not found. Go to root directory of package and type:\n\n\tpip install .\n")
    try:
        import numpy as np
        import scipy.io as sio
        import matplotlib.pyplot as plt
    except:
        sys.exit("[ERROR] You don't have the required packages. Try reinstalling the package.")

def test_madgwick(**kwargs):
    import numpy as np
    import scipy.io as sio
    import matplotlib.pyplot as plt
    import ahrs
    RAD2DEG = ahrs.common.mathfuncs.RAD2DEG
    DEG2RAD = ahrs.common.mathfuncs.DEG2RAD

    test_file = kwargs["file"] if "file" in kwargs.keys() else "ExampleData.mat"
    if not os.path.isfile(test_file):
        sys.exit("[ERROR] The file {} does not exist.".format(test_file))

    data = sio.loadmat(test_file)
    time = data['time']
    accs = data['Accelerometer']
    gyrs = data['Gyroscope']
    mags = data['Magnetometer']
    sensors = ['Gyroscope', 'Accelerometer', 'Magnetometer']

    num_samples = len(time)
    Q = np.tile([1., 0., 0., 0.], (num_samples, 1))
    euler_angles = np.zeros((num_samples, 3))
    q = Q[0].copy()
    madgwick = ahrs.filters.Madgwick(beta=0.05)
    mahony = ahrs.filters.Mahony(Kp=1.0)
    for t,_ in enumerate(time):
        # q = madgwick.updateMARG(DEG2RAD*gyrs[t].copy(), accs[t].copy(), mags[t].copy(), q)
        q = madgwick.updateIMU(DEG2RAD*gyrs[t].copy(), accs[t].copy(), q, beta=0.5)
        # q = mahony.updateIMU(DEG2RAD*gyrs[t].copy(), accs[t].copy(), q)
        # q = mahony.updateMARG(DEG2RAD*gyrs[t].copy(), accs[t].copy(), mags[t].copy(), q)
        Q[t] = q.copy()
        euler_angles[t] = ahrs.common.orientation.q2euler(ahrs.common.orientation.q_conj(q))*RAD2DEG
    # Figure 1
    fig1 = plt.figure()
    for n, p in enumerate(sensors):
        fig1.add_subplot(len(sensors), 1, n+1, title=p, xlabel='Time')
        for sensor, sensor_data in data.items():
            plt.plot(time, data[p][:, 0], 'r-', time, data[p][:, 1], 'g-', time, data[p][:, 2], 'b-', linewidth=0.2)
    # Figure 2
    fig2 = plt.figure()
    plt.plot(time, euler_angles[:, 0], 'r-', time, euler_angles[:, 1], 'g-', time, euler_angles[:, 2], 'b-', linewidth=0.2)
    plt.show()

# test_module_imports()
test_madgwick()

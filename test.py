# -*- coding: utf-8 -*-
"""
Test File
=========

"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import ahrs

data = sio.loadmat('../Madgwick/ExampleData.mat')

time = data['time']
accs = data['Accelerometer']
gyrs = data['Gyroscope']
mags = data['Magnetometer']

num_samples = len(time)
angles = np.zeros((num_samples, 3))
q = np.zeros((num_samples, 4))
q[:, 0] = 1.0
for i in range(num_samples):
    q[i] = ahrs.filters.madgwick.updateMARG(gyrs[i], accs[i], mags[i], q[i])
    angles[i] = ahrs.common.orientation.q2euler(q[i])

sensors = ['Gyroscope', 'Accelerometer', 'Magnetometer']

fig = plt.figure()
for i, s in enumerate(sensors):
    plt.subplot(3, 1, i+1, title=s)
    plt.plot(time, data[s][:, 0], 'r-', time, data[s][:, 1], 'g-', time, data[s][:, 2], 'b-', linewidth=0.2)

fig2 = plt.figure()
plt.plot(time, angles[:, 0], 'r-', time, angles[:, 1], 'g-', time, angles[:, 2], 'b-', linewidth=0.2)
plt.show()

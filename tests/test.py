# -*- coding: utf-8 -*-
"""
Test File
=========

"""

import sys
import os

from test_madgwick import test_madgwick
from test_mahony import test_mahony

def test_module_imports():
    try:
        import ahrs
    except:
        sys.exit("[ERROR] Package AHRS not found. Go to root directory of package and type:\n\n\tpip install .\n")
    try:
        import numpy, scipy, matplotlib
    except ModuleNotFoundError:
        sys.exit("[ERROR] You don't have the required packages. Try reinstalling the package.")


def plot_sensors(*sensors, **kwargs):
    import matplotlib.pyplot as plt
    num_axes = kwargs['num_axes'] if 'num_axes' in kwargs else 3
    colors = ['#ff0000', '#00aa00', '#0000ff', '#aaaa00', '#000000']
    fig = plt.figure()
    for n, s in enumerate(sensors):
        fig.add_subplot(len(sensors), 1, n+1)
        time = kwargs['time'] if 'time' in kwargs else range(s.shape[0])
        for i in range(num_axes):
            plt.plot(time, s[:, i], c=colors[i], ls='-', lw=0.3)
    plt.plot()


def plot_euler(angles, **kwargs):
    import matplotlib.pyplot as plt
    colors = ['#ff0000', '#00aa00', '#0000ff', '#aaaa00', '#000000']
    sz = angles.shape
    time = kwargs['time'] if 'time' in kwargs else range(sz[0])
    fig = plt.figure()
    for col in range(sz[1]):
        plt.plot(time, angles[:, col], c=colors[col], ls='-', lw=0.3)
    plt.plot()

if __name__ == "__main__":
    # test_module_imports()
    # test_filters()
    test_madgwick()
    test_mahony()

# -*- coding: utf-8 -*-
"""
Plotting tools
==============

TODO:
- Find a work-around to use plt.show() without blocking execution.

"""

import matplotlib.pyplot as plt
COLORS = ['#ff0000', '#00aa00', '#0000ff', '#aaaa00', '#000000']

__all__ = ['plot_sensors', 'plot_euler']

def plot_sensors(*sensors, **kwargs):
    num_axes = kwargs['num_axes'] if 'num_axes' in kwargs else 3
    title = kwargs['title'] if 'title' in kwargs else "Sensors"
    fig = plt.figure(title)
    for n, s in enumerate(sensors):
        fig.add_subplot(len(sensors), 1, n+1)
        time = kwargs['time'] if 'time' in kwargs else range(s.shape[0])
        for i in range(num_axes):
            plt.plot(time, s[:, i], c=COLORS[i], ls='-', lw=0.3)
    plt.plot()


def plot_euler(angles, **kwargs):
    sz = angles.shape
    time = kwargs['time'] if 'time' in kwargs else range(sz[0])
    title = kwargs['title'] if 'title' in kwargs else "Euler Angles"
    fig = plt.figure(title)
    for i in range(sz[1]):
        plt.plot(time, angles[:, i], c=COLORS[i], ls='-', lw=0.3)
    plt.plot()

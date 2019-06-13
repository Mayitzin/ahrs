# -*- coding: utf-8 -*-
"""
Plotting tools

.. warning::

   Using plt.show() pauses the execution of a running-script.

"""

import matplotlib.pyplot as plt
COLORS = ['#aaaa00', '#ff0000', '#00aa00', '#0000ff', '#000000']

__all__ = ['plot_sensors', 'plot_euler', 'plot_quaternions']

def plot_sensors(*sensors, **kwargs):
    """
    Plot data of sensor arrays.

    Opens a window and plots each sensor array in a different row. The window
    builds a subplot for each sensor array.

    Parameters
    ----------
    sensors : arrays
        Arrays of sensors to plot. Each array is of size M-by-N, where M is the
        number of samples, and N is the number of axes.
    num_axes : int
        Optional. Number of axes per sensor. Default is 3.
    x_axis : array
        Optional. X-axis data array of the plots. Default is `range(M)`.
    title : str
        Optional. Title of window. Default is 'Sensors'

    Examples
    --------
    >>> data = ahrs.utils.io.load("data.mat")
    >>> ahrs.utils.plot_sensors(data.gyrs)   # Plot Gyroscopes

    Each call will open a new window with the requested plots and pause any
    further computation, until the window is closed.

    >>> ahrs.utils.plot_sensors(gyrs, accs) # Plot Gyroscopes and Accelerometers in same window
    >>> time = data['time']
    >>> ahrs.utils.plot_sensors(data.gyr, data.acc, data.mag, x_axis=data.time, title="Sensors")

    """
    num_axes = kwargs.get('num_axes', 3)
    title = kwargs.get('title', "Sensors")
    fig = plt.figure(title)
    for n, s in enumerate(sensors):
        fig.add_subplot(len(sensors), 1, n+1)
        x_axis = kwargs.get('x_axis', range(s.shape[0]))
        for i in range(num_axes):
            plt.plot(x_axis, s[:, i], c=COLORS[i+1], ls='-', lw=0.3)
    plt.show()


def plot_euler(angles, **kwargs):
    """
    Plot Euler Angles.

    Opens a window and plots the three Euler Angles in a centered plot.

    Parameters
    ----------
    sensors : arrays
        Array of Euler Angles to plot. Each array is of size M-by-3.
    x_axis : array
        Optional. X-axis data array of the plot. Default is `range(M)`.
    title : str
        Optional. Title of window. Default is 'Euler Angles'.

    Examples
    --------
    >>> data = ahrs.utils.io.load("data.mat")
    >>> ahrs.utils.plot_euler(data.euler_angles)

    Each call will open a new window with the requested plots and pause any
    further computation, until the window is closed.

    >>> time = data['time']
    >>> ahrs.utils.plot_euler(data.euler_angles, x_axis=data.time, title="My Angles")

    """
    sz = angles.shape
    if sz[1] != 3:
        return None
    x_axis = kwargs.get('x_axis', range(sz[0]))
    title = kwargs.get('title', "Euler Angles")
    fig = plt.figure(title)
    for i in range(sz[1]):
        plt.plot(x_axis, angles[:, i], c=COLORS[i+1], ls='-', lw=0.3)
    plt.show()


def plot_quaternions(*quaternions, **kwargs):
    """
    Plot Quaternions.

    Opens a window and plots the Quaternions in a centered plot.

    Parameters
    ----------
    sensors : arrays
        Array of Quaternions to plot. Each array is of size M-by-4.
    x_axis : array
        Optional. X-axis data array of the plot. Default is `range(M)`.
    title : str
        Optional. Title of window. Default is 'Quaternions'.

    Examples
    --------
    >>> data = ahrs.utils.io.load("data.mat")
    >>> ahrs.utils.plot_quaternions(data.qts)

    Each call will open a new window with the requested plots and pause any
    further computation, until the window is closed.

    >>> time = data['time']
    >>> ahrs.utils.plot_quaternions(data.qts, x_axis=time, title="My Quaternions")

    Two or more quaternions can also be plotted, like in the sensor plotting
    function.

    >>> ahrs.utils.plot_quaternions(data.qts, ref_quaternions)

    """
    title = kwargs.get('title', "Quaternions")
    fig = plt.figure(title)
    for n, q in enumerate(quaternions):
        fig.add_subplot(len(quaternions), 1, n+1)
        x_axis = kwargs.get('x_axis', range(q.shape[0]))
        for i in range(4):
            plt.plot(x_axis, q[:, i], c=COLORS[i], ls='-', lw=0.3)
    plt.show()

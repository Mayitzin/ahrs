# -*- coding: utf-8 -*-
"""
Plotting tools

.. warning::

   Using plt.show() pauses the execution of a running-script.

"""

import matplotlib.pyplot as plt
COLORS = ['#ff0000', '#00aa00', '#0000ff', '#aaaa00', '#000000']

__all__ = ['plot_sensors', 'plot_euler']

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

    Returns
    -------
    None

    Examples
    --------
    >>> import scipy.io as sio
    >>> import ahrs
    >>> data = data = sio.loadmat("data.mat")
    >>> gyrs = data['Gyroscope']
    >>> accs = data['Accelerometer']
    >>> ahrs.utils.plot_sensors(gyrs)   # Plot Gyroscopes

    Each call will open a new window with the requested plots and pause any
    further computation, until the window is closed.

    >>> ahrs.utils.plot_sensors(gyrs, accs) # Plot Gyroscopes and Accelerometers in same window
    >>> time = data['time']
    >>> ahrs.utils.plot_sensors(gyrs, accs, mags, x_axis=time, title="Sensors")

    """
    num_axes = kwargs['num_axes'] if 'num_axes' in kwargs else 3
    title = kwargs['title'] if 'title' in kwargs else "Sensors"
    fig = plt.figure(title)
    for n, s in enumerate(sensors):
        fig.add_subplot(len(sensors), 1, n+1)
        x_axis = kwargs['x_axis'] if 'x_axis' in kwargs else range(s.shape[0])
        for i in range(num_axes):
            plt.plot(x_axis, s[:, i], c=COLORS[i], ls='-', lw=0.3)
    plt.plot()


def plot_euler(angles, **kwargs):
    """
    Plot Euler Angles.

    Opens a window and plots each sensor array in a different row. The window
    builds a subplot for each sensor array.

    Parameters
    ----------
    sensors : arrays
        Array of Euler Angles to plot. Each array is of size M-by-3.
    x_axis : array
        Optional. X-axis data array of the plot. Default is `range(M)`.
    title : str
        Optional. Title of window. Default is 'Euler Angles'.

    Returns
    -------
    None

    Examples
    --------
    >>> import scipy.io as sio
    >>> import ahrs
    >>> data = data = sio.loadmat("data.mat")
    >>> euler_angles = data['Euler']
    >>> ahrs.utils.plot_euler(euler_angles)

    Each call will open a new window with the requested plots and pause any
    further computation, until the window is closed.

    >>> time = data['time']
    >>> ahrs.utils.plot_euler(euler_angles, x_axis=time, title="My Angles")

    """
    sz = angles.shape
    x_axis = kwargs['x_axis'] if 'x_axis' in kwargs else range(sz[0])
    title = kwargs['title'] if 'title' in kwargs else "Euler Angles"
    fig = plt.figure(title)
    for i in range(sz[1]):
        plt.plot(x_axis, angles[:, i], c=COLORS[i], ls='-', lw=0.3)
    plt.plot()

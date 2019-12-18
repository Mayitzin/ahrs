# -*- coding: utf-8 -*-
"""
Plotting tools

.. warning::

   Using plt.show() pauses the execution of a running-script.

"""

import numpy as np
import matplotlib.pyplot as plt

__all__ = ['plot', 'plot_sensors', 'plot_euler', 'plot_quaternions']

def hex_to_int(color):
    """Convert hex value to tuple of type int with values between 0 and 255
    """
    a = color.lstrip('#')
    return tuple(int(a[i:i+2], 16) for i in (0, 2, 4, 6))

def hex_to_float(color):
    """Convert hex value to tuple of type float with values between 0.0 and 1.0
    """
    a = color.lstrip('#')
    return tuple(int(a[i:i+2], 16)/255.0 for i in (0, 2, 4, 6))

COLORS = [
    "#FF0000FF", "#00AA00FF", "#0000FFFF", "#999933FF",
    "#FF8888FF", "#88AA88FF", "#8888FFFF", "#999955FF",
    "#660000FF", "#005500FF", "#000088FF", "#666600FF"]
COLORS_INTS = [hex_to_int(c) for c in COLORS]
COLORS_FLOATS = [hex_to_float(c) for c in COLORS]

def plot(*data, **kw):
    """
    Plot data with custom formatting.

    Given data is plotted in time domain. It locks any current process until
    plotting window is closed.

    Parameters
    ----------
    data : array
        Arrays with the contents of data to plot.

    Examples
    --------
    >>> from ahrs.utils import plot
    >>> data = np.array([2., 3., 4., 5.])
    >>> plot(data)
    >>> data_2 = np.array([4., 5., 6., 7.])
    >>> plot(data, data_2)
    >>> plot(data, data_2, subtitles=["data", "data 2"])
    """
    title = kw.get("title", 0)
    subtitles = kw.get("subtitles", None)
    xlabels = kw.get("xlabels", None)
    ylabels = kw.get("ylabels", None)
    yscales = kw.get("yscales", None)
    num_subplots = len(data)
    fig, axs = plt.subplots(num_subplots, 1, num=title, squeeze=False)
    for i, d in enumerate(data):
        d = np.array(d)
        # if isinstance(d, list):
        if d.ndim < 2:
            axs[i, 0].plot(d, color=COLORS[0], lw=0.5, ls='-') # Plot a single red line in subplot
        else:
            d_sz = d.shape
            if d_sz[0] > d_sz[1]:
                d = d.T
            for j, row in enumerate(d):
                axs[i, 0].plot(row, color=COLORS[j], lw=0.5, ls='-')
        if subtitles:
            axs[i, 0].set_title(subtitles[i])
        if xlabels:
            axs[i, 0].set_xlabel(xlabels[i])
        if ylabels:
            axs[i, 0].set_ylabel(ylabels[i])
        if yscales:
            axs[i, 0].set_yscale(yscales[i])
    fig.tight_layout()
    plt.show()

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
    num_axes : int, optional
        Number of axes per sensor. Default is 3.
    x_axis : array, optional
        X-axis data array of the plots. Default is `range(M)`.
    title : str, optional
        Title of window. Default is 'Sensors'
    subtitles : list of strings, optional
        List of titles for each subplot.

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
    subtitles = kwargs.get('subtitles', None)
    fig = plt.figure(title)
    for n, s in enumerate(sensors):
        fig.add_subplot(len(sensors), 1, n+1)
        if subtitles:
            plt.subplot(len(sensors), 1, n+1, title=subtitles[n])
        x_axis = kwargs.get('x_axis', range(s.shape[0]))
        if s.ndim < 2:
            plt.plot(s, 'k-', lw=0.3)
        else:
            for i in range(num_axes):
                plt.plot(x_axis, s[:, i], c=COLORS[i+1], ls='-', lw=0.3)
    plt.show()


def plot_euler(*angles, **kwargs):
    """
    Plot Euler Angles.

    Opens a window and plots the three Euler Angles in a centered plot.

    Parameters
    ----------
    angles : arrays
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
    # x_axis = kwargs.get('x_axis', range(sz[0]))
    title = kwargs.get('title', "Euler Angles")
    subtitles = kwargs.get('subtitles', None)
    fig = plt.figure(title)
    for n, a in enumerate(angles):
        fig.add_subplot(len(angles), 1, n+1)
        if subtitles:
            plt.subplot(len(angles), 1, n+1, title=subtitles[n])
        x_axis = kwargs.get('x_axis', range(a.shape[0]))
        for i in range(3):
            plt.plot(x_axis, a[:, i], c=COLORS[i+1], ls='-', lw=0.3)
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
    subtitles = kwargs.get('subtitles', None)
    fig = plt.figure(title)
    for n, q in enumerate(quaternions):
        fig.add_subplot(len(quaternions), 1, n+1)
        if subtitles:
            plt.subplot(len(quaternions), 1, n+1, title=subtitles[n])
        x_axis = kwargs.get('x_axis', range(q.shape[0]))
        for i in range(4):
            plt.plot(x_axis, q[:, i], c=COLORS[i], ls='-', lw=0.3)
    plt.show()

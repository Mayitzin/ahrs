# -*- coding: utf-8 -*-
"""
Tools for the Juypter notebooks used to showcase the functionality of the
package.

Many funcitons are based on the ones included in the Python package `ezview`
available in the repository https://https://github.com/Mayitzin/ezview
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)  # Light grey grid

def hex_to_int(color: str) -> tuple:
    """Convert a hex color to a tuple of integers."""
    a = color.lstrip('#')
    return tuple(int(a[i:i+2], 16) for i in (0, 2, 4, 6))

def hex_to_float(color: str) -> tuple:
    """Convert a hex color to a tuple of floats."""
    a = color.lstrip('#')
    return tuple(int(a[i:i+2], 16)/255.0 for i in (0, 2, 4, 6))

COLORS = [
    "#FF0000FF", "#00AA00FF", "#0000FFFF", "#999933FF",
    "#FF8888FF", "#88AA88FF", "#8888FFFF", "#999955FF",
    "#660000FF", "#005500FF", "#000088FF", "#666600FF"]
COLORS_INTS = [hex_to_int(c) for c in COLORS]
COLORS_FLOATS = [hex_to_float(c) for c in COLORS]

def describe_methods(obj):
    """Return the description of the methods of an object."""
    for method_name in [x for x in dir(obj) if hasattr(obj.__getattribute__(x), '__call__') and not x.startswith('__')]:
        method_description = '<No description found>'
        method_docstring = getattr(obj, method_name).__doc__
        if isinstance(method_docstring, str):
            method_description = method_docstring.split('\n')[1].strip()
            if len(method_description) < 1:
                method_description = [x.strip() for x in method_docstring.split('\n') if len(x) > 0][1]
        print(f"DCM.{method_name+'()': <20} {method_description.replace(':meth:', '')}")

def plot(*data, **kw):
    """
    Plot time-series data.

    Parameters
    ----------
    data : array
        Arrays with the contents of data to plot. They could be 1- (single line)
        or 2-dimensional.
    title : int or str
        Window title as number or label.
    subtitles : list
        List of strings of the titles of each subplot.
    labels : list
        List of labels that will be displayed in each subplot's legend.
    xlabels : list
        List of strings of the labels of each subplot's X-axis.
    ylabels : list
        List of strings of the labels of each subplot's Y-axis.
    yscales : str
        List of strings of the scales of each subplot's Y-axis. It supports
        matlabs defaults values: "linear", "log", "symlog" and "logit"

    """
    title = kw.get("title")
    subtitles = kw.get("subtitles")
    labels = kw.get("labels")
    xlabels = kw.get("xlabels")
    ylabels = kw.get("ylabels")
    yscales = kw.get("yscales")
    index = kw.get("index")
    indices = kw.get("indices")
    shades_spans = kw.get("shaded")
    num_subplots = len(data)        # Number of given arrays
    # Create figure with vertically stacked subplots
    fig, axs = plt.subplots(
        num_subplots,
        1,
        num=title,
        squeeze=False,
        sharex=kw.get('sharex', "indices" not in kw),
        sharey=kw.get('sharey', False)
        )
    for i, array in enumerate(data):
        array = np.copy(array)
        if array.ndim > 2:
            raise ValueError(f"Data array {i} has more than 2 dimensions.")
        if array.ndim < 2:
            # Plot a single line in the subplot (1-dimensional array)
            label = labels[i][0] if labels else None
            index = index if index is not None else np.arange(array.shape[0])
            axs[i, 0].plot(index, array, color=COLORS[0], lw=0.5, ls='-', label=label)
        else:
            # Plot multiple lines in the subplot (2-dimensional array)
            array_sz = array.shape
            if array_sz[0] > array_sz[1]:
                # Transpose array if it has more rows than columns
                array = array.T
            index = indices[i] if indices is not None else np.arange(array_sz[0])
            for j, row in enumerate(array):
                label = None
                if labels:
                    if len(labels[i]) == len(array):
                        label = labels[i][j]
                axs[i, 0].plot(index, row, color=COLORS[j], lw=0.5, ls='-', label=label)
        axs[i, 0].grid(axis='y')
        if subtitles:
            axs[i, 0].set_title(subtitles[i])
        if xlabels:
            axs[i, 0].set_xlabel(xlabels[i])
        if ylabels:
            axs[i, 0].set_ylabel(ylabels[i])
        if yscales:
            axs[i, 0].set_yscale(yscales[i])
        if shades_spans is not None:
            # Add shaded areas
            try:
                if isinstance(shades_spans, (list, np.ndarray)):
                    current_spans = shades_spans[i] if np.copy(shades_spans).ndim > 2 else shades_spans
                    for s in current_spans:
                        axs[i, 0].axvspan(s[0], s[1], color='gray', alpha=0.1)
                elif isinstance(shades_spans, dict):
                    # Add shades AND their corresponding labels
                    for k, v in shades_spans.items():
                        span = [v['start'], v['stop']]
                        axs[i, 0].axvspan(span[0], span[1], color='gray', alpha=0.1)
                        axs[i, 0].text(int(np.mean(span)), max(array), k, ha='center')
            except:
                print("No spans were given")
        if labels:
            if len(labels[i]) > 0:
                axs[i, 0].legend(loc='lower right')
    fig.tight_layout()
    plt.show()

def ellipsoid(center: np.ndarray = None, axes: np.ndarray = None, num_points: int = 20) -> tuple:
    """
    Return the mesh of an ellipsoid.

    Parameters
    ----------
    center : numpy.ndarray, optional
        3-element array with the ellipsoid's center. Default is [0, 0, 0].
    axes : numpy.ndarray, optional
        3-element array with the ellipsoid's main axes lengths. Default is
        [1, 1, 1].
    num_points : int, optional
        Number of points to use in the mesh. Default is 20.

    Returns
    -------
    tuple
        Tuple with the mesh of the ellipsoid in the form (x, y, z).

    """
    if center is None:
        center = np.zeros(3)
    if axes is None:
        axes = np.ones(3)
    if not isinstance(center, (np.ndarray, list, tuple)):
        raise TypeError("Center must be a 3-element array.")
    if not isinstance(axes, (np.ndarray, list, tuple)):
        raise TypeError("Axes must be a 3-element array.")
    # Create ellipsoid mesh
    cx, cy, cz = center
    sx, sy, sz = axes
    u, v = np.mgrid[0:2*np.pi:complex(num_points), 0:np.pi:complex(num_points)]
    x = sx * np.cos(u)*np.sin(v) + cx
    y = sy * np.sin(u)*np.sin(v) + cy
    z = sz * np.cos(v) + cz
    return x, y, z

def frame(dcm: np.ndarray = None, position: np.ndarray = None, scale: float = 1.0) -> list:
    """
    Return the coordinates of an orthogonal frame.

    Parameters
    ----------
    dcm : numpy.ndarray, optional
        3-by-3 array with the frame's axes. Default is the identity matrix.
    position : numpy.ndarray, optional
        3-element array with the frame's origin. Default is [0, 0, 0].
    scale : float, optional
        Scale factor for the frame's axes. Default is 1.0.

    """
    if dcm is None:
        dcm = np.identity(3)
    if position is None:
        position = np.zeros(3)
    f_coords = []
    for column_index in range(3):
        axis_end = dcm[:, column_index]*scale + position
        f_coords.append(np.c_[position, axis_end])
    return f_coords


def add_ellipsoid(ax, params: list | dict, num_points: int = 20, color = 'k', lw = 0.5, **kwargs) -> None:
    """
    Add a ellipsoid to an existing 3D plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes3D
        3D axis where the ellipsoid will be added.
    params : list or dict.
        List or dictionary with the parameters to draw an ellipsoid. If a list
        is given, it must be of the form [[a, b, c], [x, y, z]], where a, b, c
        are the coordinates of the ellipsoid's center, and x, y, z are the
        ellipsoid's main axes lengths. If a dictionary is given, it must be of
        the form {'center': [a, b, c], 'axes': [x, y, z]}
    num_points : int, optional
        Number of points, per axis, to use in the mesh. Default is 20.
    color : str, optional
        Color of the ellipsoid. Default is 'k'.
    lw : float, optional
        Line width of the ellipsoid. Default is 0.5.

    """
    if isinstance(params, (list, tuple, np.ndarray)):
        center, axes = params
    elif isinstance(params, dict):
        center = params.get("center", np.zeros(3))
        axes = params.get("axes", np.ones(3))
    else:
        raise TypeError("Unknown type for 'sphere'. Try a list or a dict.")
    # Extract only the expected parameters from kwargs
    expected_params = {'num_points': num_points, 'color': color, 'lw': lw, 'alpha': 0.5}
    for key in expected_params:
        if key in kwargs:
            expected_params[key] = kwargs[key]
    x, y, z = ellipsoid(center=center, axes=axes, num_points=expected_params['num_points'])   # Ellipsoid mesh
    ax.plot_wireframe(x, y, z, color=expected_params['color'], lw=expected_params['lw'], alpha=expected_params['alpha'])

def add_frame(ax, dcm: np.ndarray, position = None, color: str | list = None, scale: float = 1.0, lw: float = 1.5, **kwargs) -> None:
    """
    Add a frame to an existing 3D plot.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D axis where the frame will be added.
    frame : numpy.ndarray
        3-by-3 array with the frame's axes. Each row is a vector.
    position : numpy.ndarray, optional
        3-element array with the frame's position. Default is [0, 0, 0].
    color : str or list of strings, optional
        Color of the frame. Default is None, which iterates over RGB.
    scale : float, optional
        Scale factor of the frame. Default is 1.0
    lw : float, optional
        Line width of the frame. Default is 1.5

    """
    if not hasattr(ax, 'plot'):
        raise TypeError("The given axis is not a 3D plot item.")
    colors = ([color]*3 if isinstance(color, str) else color) if color is not None else COLORS[:3]
    # Extract only the expected parameters from kwargs
    expected_params = {'scale': scale, 'lw': lw}
    for key in expected_params:
        if key in kwargs:
            expected_params[key] = kwargs[key]
    frame_coords = frame(dcm, position, scale)
    for axis in frame_coords:
        ax.plot(*axis, color=colors.pop(0), lw=lw)

def add_items(ax, **kwargs) -> None:
    """
    Add items to an existing 3D plot.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D axis where the items will be added.
    kwargs : dict
        Dictionary with the items to be added. The keys are the items' types,
        and the values are the items' data and parameters.

    """
    if 'scatter' in kwargs:
        data = kwargs['scatter']
        if isinstance(data, (list, tuple, np.ndarray)):
            if isinstance(data, np.ndarray):
                data = data.T
            ax.scatter(*data)
        elif isinstance(data, dict):
            scatter_dict = copy.deepcopy(data)
            for k, v in scatter_dict.items():
                points = v.pop('data')    # Get N-by-3 numpy array from dict
                ax.scatter(*points.T, **v)
        else:
            raise TypeError(f"Unknown type for 'scatter': {type(data)}. Try an array, list or dict.")
    if 'lines' in kwargs:
        data = kwargs['lines']
        if isinstance(data, (list, tuple, np.ndarray)):
            if isinstance(data, np.ndarray):
                data = data.T
            ax.plot(*data)
        elif isinstance(data, dict):
            lines_dict = copy.deepcopy(data)
            for k, v in lines_dict.items():
                lines = v.pop('data')   # Get N-by-3 numpy array from dict
                ax.plot(*lines.T, **v)
        else:
            raise TypeError(f"Unknown type for 'lines': {type(data)}. Try an array, list or dict.")
    if 'frames' in kwargs:
        if isinstance(kwargs['frames'], dict):
            # Frames given as dictionary of dictionaries with each frame's attitude and position
            for k, v in kwargs['frames'].items():
                data = v.copy()
                add_frame(ax, dcm=data['attitude'], position=data.pop('position'), **data)
        elif isinstance(kwargs['frames'], list):
            # Frames given as list of frames.
            for frame_idx, frame_item in enumerate(kwargs['frames']):
                np_frame = np.copy(frame_item)
                if np_frame.shape == (3, 3):
                    add_frame(ax, dcm=np_frame)
                elif np_frame.shape == (3, 4):
                    add_frame(ax, dcm=np_frame[:, :3], position=np_frame[:, 3])
                else:
                    raise ValueError(f"Unknown shape for frame {frame_idx}: {np_frame.shape}")
        elif isinstance(kwargs['frames'], np.ndarray):
            # 3-dimensional array with the frames' data
            if kwargs['frames'].ndim == 2:
                np_frame = np.copy(kwargs['frames'])
                if np_frame.shape == (3, 3):
                    add_frame(ax, dcm=np_frame)
                elif np_frame.shape == (3, 4):
                    add_frame(ax, dcm=np_frame[:, :3], position=np_frame[:, 3])
                else:
                    raise ValueError(f"Unknown shape for frame: {np_frame.shape}")
            elif kwargs['frames'].ndim == 3:
                for frame_idx, frame_item in enumerate(kwargs['frames']):
                    np_frame = np.copy(frame_item)
                    if np_frame.shape == (3, 3):
                        add_frame(ax, dcm=np_frame)
                    elif np_frame.shape == (3, 4):
                        add_frame(ax, dcm=np_frame[:, :3], position=np_frame[:, 3])
                    else:
                        raise ValueError(f"Unknown shape for frame {frame_idx}: {np_frame.shape}")
            else:
                raise ValueError(f"Frames must be a 2D or 3D numpy array. Got shape {kwargs['frames'].shape}")
        else:
            raise TypeError(f"Unknown type for 'frames': {type(kwargs['frames'])}. Try a list or dict.")
    if 'ellipsoids' in kwargs:
        for k, v in kwargs['ellipsoids'].items():
            add_ellipsoid(ax, v, **v)

def plot3(**kwargs) -> None | tuple:
    """
    Plot 3-dimensional data in a cartesian coordinate system.

    Parameters
    ----------
    show : bool, optional
        Show the plot after creating it. Default is True. Otherwise, the plot
        Figure and Axes objects are returned.

    """
    # Build the plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Add items
    add_items(ax, **kwargs)

    # Set properties of plot
    plt.tight_layout()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')      # Added in matplotlib 3.6

    # Show or return the plot
    if not kwargs.get('show', True):
        return fig, ax
    plt.show()

# AHRS: Attitude and Heading Reference Systems

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Mayitzin/ahrs/Build%20Python%20Package)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ahrs)
![PyPI - License](https://img.shields.io/pypi/l/ahrs)
![PyPI](https://img.shields.io/pypi/v/ahrs)
![PyPI Downloads](https://pepy.tech/badge/ahrs)
![Codacy Badge](https://api.codacy.com/project/badge/Grade/bc366c601ed44e12b233218dd37cd32c)

AHRS is a zoo of functions and objects written in Python helping you to estimate the orientation and position of robotic systems.

Orginally, an [AHRS](https://en.wikipedia.org/wiki/Attitude_and_heading_reference_system) is defined as a set of orthogonal sensors providing attitude information about an aircraft. This field is now expanding to smaller devices, like wearables, automated transportation and all kinds of robots in motion.

The module __AHRS__ is developed with a focus on fast prototyping and easy modularity.

AHRS is compatible with __Python 3.6__ and above.

## Installation

AHRS may be installed using [pip](https://pip.pypa.io):

```shell
pip install ahrs
```

Or directly from the repository:

```shell
git clone https://github.com/Mayitzin/ahrs.git
cd ahrs
python setup.py install
```

AHRS depends on the most distributed packages of scientifc Python environments ([NumPy](https://numpy.org/), [SciPy](https://www.scipy.org/) and [matplotlib](https://matplotlib.org/)). If you don't have them, they will be automatically downloaded and installed.

## Using AHRS

To play with orientations, for example, we can use the `orientation` module.

```py
>>> from ahrs.common import orientation
>>> # Rotation product: R_y(10.0) @ R_x(20.0) @ R_y(30.0)
... Rx = orientation.rotation('x', 10.0)
>>> Ry = orientation.rotation('y', 20.0)
>>> Rz = orientation.rotation('z', 30.0)
>>> Rx@Ry@Rz
array([[ 0.81379768 -0.46984631  0.34202014]
       [ 0.54383814  0.82317294 -0.16317591]
       [-0.20487413  0.31879578  0.92541658]])
>>> # Same rotation sequence but with single call to rot_seq()
... orientation.rot_seq('xyz', [10.0, 20.0, 30.0])
array([[ 0.81379768 -0.46984631  0.34202014]
       [ 0.54383814  0.82317294 -0.16317591]
       [-0.20487413  0.31879578  0.92541658]])
```

### New in version 0.2.0

It now includes the class `Quaternion` to easily handle the orientation estimation with quaternions.

```py
>>> from ahrs import Quaternion
>>> q1 = Quaternion()
>>> str(q1)          # Empty quaternions default to identity quaternion
'(1.0000 +0.0000i +0.0000j +0.0000k)'
>>> q2 = Quaternion([1.0, 2.0, 3.0])
>>> str(q2)          # 3-element vectors build pure quaternions
'(0.0000 +0.2673i +0.5345j +0.8018k)'
>>> q3 = Quaternion([1., 2., 3., 4.])
>>> str(q3)          # All quaternions are normalized
'(0.1826 +0.3651i +0.5477j +0.7303k)'
>>> str(q2+q3)       # Use normal arithmetic operators
'(0.0918 +0.3181i +0.5444j +0.7707k)'
>>> q2.product(q3)   # Quaternion products are supported
array([-0.97590007,  0.        ,  0.19518001,  0.09759001])
>>> str(q2*q3)
'(-0.9759 +0.0000i +0.1952j +0.0976k)'
>>> q2.to_DCM()      # Conversions between representations are also implemented
array([[-0.85714286,  0.28571429,  0.42857143],
       [ 0.28571429, -0.42857143,  0.85714286],
       [ 0.42857143,  0.85714286,  0.28571429]])
```

And many other quaternion operations, properties and methods are also available.

To use the sensor data to estimate the attitude, the `filters` module includes various (more coming) algorithms for it.

```py
>>> madgwick = ahrs.filters.Madgwick()    # Madgwick's attitude estimation using default values
>>> Q = np.tile([1., 0., 0., 0.], (data.num_samples, 1)) # Allocate an array for all quaternions
>>> d2g = ahrs.common.DEG2RAD   # Constant to convert degrees to radians
>>> for t in range(1, data.num_samples):
...     Q[t] = madgwick.updateMARG(Q[t-1], d2g*data.gyr[t], data.acc[t], data.mag[t])
...
>>> Q.shape
(6959, 4)
```

Also works by simply passing the data to a desired filter, and it will automatically try to load the sensor information and estimate the quaternions with the given parameters.

```py
>>> orientation = ahrs.filters.Madgwick(data, beta=0.1, frequency=100.0)
>>> orientation.Q.shape
(6959, 4)
```

## Notes for future versions

`ahrs` will start to move away from plotting and data parsing submodules to better focus in the algorithmic parts. This means, the submodule `io` and `plot`-related functions will not be further developed and become obsolete.

This way you can also choose your favorite libraries for data loading and visualization.

## Documentation

A comprehensive documentation, with examples, will soon come to [Read the Docs](https://docs.readthedocs.io/).

# AHRS: Attitude and Heading Reference Systems

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/Mayitzin/ahrs/Build%20Python%20Package)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ahrs)
![PyPI - License](https://img.shields.io/pypi/l/ahrs)
![PyPI](https://img.shields.io/pypi/v/ahrs)
![PyPI Downloads](https://pepy.tech/badge/ahrs)
![Codacy Badge](https://api.codacy.com/project/badge/Grade/bc366c601ed44e12b233218dd37cd32c)

AHRS is a zoo of functions and algorithms in pure Python helping to estimate the orientation and position of robotic systems.

Orginally, an [AHRS](https://en.wikipedia.org/wiki/Attitude_and_heading_reference_system) is defined as a set of orthogonal sensors providing attitude information about an aircraft. This field is now expanding to smaller devices, like wearables, automated transportation and all kinds of systems in motion.

This package's focus is **fast prototyping**, **education**, **testing** and **easy modularity**. Performance is _NOT_ the main goal. For optimized implementations there are endless resources in C/C++.

AHRS is compatible with __Python 3.6__ and newer.

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

AHRS depends on common packages [NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/). Further packages are avoided, to reduce its third-party dependency.

## Using AHRS

To play with orientations, for example, we can use the `orientation` module.

```python
>>> from ahrs.common.orientation import rotation
>>> Rx = rotation('x', 10.0)
>>> Ry = rotation('y', 20.0)
>>> Rz = rotation('z', 30.0)
>>> Rx@Ry@Rz                # Rotation product: R_y(10.0) @ R_x(20.0) @ R_y(30.0)
array([[ 0.81379768 -0.46984631  0.34202014],
       [ 0.54383814  0.82317294 -0.16317591],
       [-0.20487413  0.31879578  0.92541658]])
>>> from ahrs.common.orientation import rot_seq
>>> rot_seq('xyz', [10.0, 20.0, 30.0])      # Same rotation calling rot_seq()
array([[ 0.81379768 -0.46984631  0.34202014],
       [ 0.54383814  0.82317294 -0.16317591],
       [-0.20487413  0.31879578  0.92541658]])
```

Or the newly implemented class `DCM`

```python
>>> from ahrs import DCM
>>> DCM(x=10.0, y=20.0, z=30.0)
DCM([[ 0.81379768 -0.46984631  0.34202014],
     [ 0.54383814  0.82317294 -0.16317591],
     [-0.20487413  0.31879578  0.92541658]])
```

## New in 0.3

- [Type hints](https://www.python.org/dev/peps/pep-0484/) are added.
- The **World Magnetic Model** ([WMM](https://www.ngdc.noaa.gov/geomag/WMM/DoDWMM.shtml)) is fully implemented. It can be used to estimate all magnetic field elements on any given place of Earth for dates between 2015 and 2025.

```python
>>> from ahrs.utils import WMM
>>> wmm = WMM(latitude=10.0, longitude=-20.0, height=10.5)
>>> wmm.magnetic_elements
{'X': 30499.640469609083, 'Y': -5230.267158472566, 'Z': -1716.633311360368,
'H': 30944.850352270452, 'F': 30992.427998627096, 'I': -3.1751692563622993,
'D': -9.73078560629778, 'GV': -9.73078560629778}
```

- The ellipsoid model of the **World Geodetic System** ([WGS84](https://earth-info.nga.mil/GandG/update/index.php?dir=wgs84&action=wgs84)) is included. A full implementation of the **Earth Gravitational Model** ([EGM2008](https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/egm08_wgs84.html)) is _NOT_ available here, but the estimation of the main and derived parameters of the WGS84 using the ellipsoid model are implemented:

```python
>>> from ahrs.utils import WGS
>>> wgs = WGS()      # Creates an ellipsoid model, using Earth's characteristics by default
>>> [x for x in dir(wgs) if not x.startswith('__')]
['a', 'arithmetic_mean_radius', 'aspect_ratio', 'atmosphere_gravitational_constant', 'authalic_sphere_radius', 'curvature_polar_radius', 'dynamic_inertial_moment_about_X', 'dynamic_inertial_moment_about_Y', 'dynamic_inertial_moment_about_Z', 'dynamical_form_factor', 'equatorial_normal_gravity', 'equivolumetric_sphere_radius', 'f', 'first_eccentricity_squared', 'geometric_dynamic_ellipticity', 'geometric_inertial_moment', 'geometric_inertial_moment_about_Z', 'gm', 'gravitational_constant_without_atmosphere', 'is_geodetic', 'linear_eccentricity', 'mass', 'mean_normal_gravity', 'normal_gravity', 'normal_gravity_constant', 'normal_gravity_potential', 'polar_normal_gravity', 'second_degree_zonal_harmonic', 'second_eccentricity_squared', 'semi_minor_axis', 'w']
```

It can be used, for example, to estimate the normal gravity acceleration (in $m/s^2$) at any location on Earth.

```python
>>> g = wgs.normal_gravity(50.0, h=500.0)        # Latitude = 50° N,   height above sea = 500 m
9.810958954471268
```

Setting the fundamental parameters (`a`, `f`, `gm`, `w`) yields a different ellipsoid. For the moon, for example, we build a new model:

```python
>>> moon_flattening = (ahrs.MOON_EQUATOR_RADIUS-ahrs.MOON_POLAR_RADIUS)/ahrs.MOON_EQUATOR_RADIUS
>>> mgs = WGS(a=ahrs.MOON_EQUATOR_RADIUS, f=moon_flattening, gm=ahrs.MOON_GM, w=ahrs.MOON_ROTATION)
>>> g = mgs.normal_gravity(10.0, h=500.0)
1.6239820345657434
```

- The [International Gravity Formula](http://earth.geology.yale.edu/~ajs/1945A/360.pdf) and the EU's [WELMEC](https://www.welmec.org/documents/guides/2/) normal gravity reference system are also implemented.

```python
>>> ahrs.utils.international_gravity(50.0)       # Works only on surface
9.810786421572386
>>> ahrs.utils.welmec_gravity(50.0, 500.0)
9.809152687885897
```

- New class `DCM` (derived from `numpy.ndarray` to use its advantages) for orientation/rotation representations as $3\times 3$ Direction Cosine Matrices.

```python
>>> from ahrs import DCM
>>> R = DCM(x=10.0, y=20.0, z=30.0)
>>> type(R)
<class 'ahrs.common.dcm.DCM'>
>>> R.view()
DCM([[ 0.81379768 -0.46984631  0.34202014],
     [ 0.54383814  0.82317294 -0.16317591],
     [-0.20487413  0.31879578  0.92541658]])
>>> R.inv
array([[ 0.81379768  0.54383814 -0.20487413]
       [-0.46984631  0.82317294  0.31879578]
       [ 0.34202014 -0.16317591  0.92541658]])
>>> R.log
array([0.26026043, 0.29531805, 0.5473806 ])
>>> R.to_quaternion()
array([0.94371436, 0.12767944, 0.14487813, 0.26853582])
>>> R.to_axisangle()
(array([0.38601658, 0.43801381, 0.81187135]), 0.6742208510527136)
```

- A whole bunch of new constant values (mainly for Geodesy) accessed from the top level of the package.
- New operations, properties and methods for class `Quaternion`.
- Docstrings are improved with further explanations, references and equations whenever possible.
- New Attitude Estimation Algorithms! All estimators are refactored to be consistent to the original articles describing them. They have in-code reference to the original equations.

To use the sensor data to estimate the attitude simply pass the data to a desired filter, and it will automatically estimate the quaternions with the given parameters.

```python
>>> attitude = ahrs.filters.Madgwick(acc=acc_data, gyr=gyro_data)
>>> attitude.Q.shape
(6959, 4)
```

Some algorithms allow a finer tuning of its estimation with different parameters. Check their documentation to see what can you change.

```python
>>> attitude = ahrs.filters.Madgwick(acc=acc_data, gyr=gyro_data, mag=mag_data, gain=0.1, frequency=100.0)
```

Implemented attitude estimators are checked:

- [x] Angular velocity integration
- [x] AQUA (Algebraic Quaternion Algorithm)
- [x] Complementary Filter with Quaternions
- [x] Davenport's q-method
- [ ] ESOQ (Estimator of the Optimal Quaternion)
- [ ] ESOQ-2 (New Estimator of the Optimal Quaternion)
- [ ] FCF (Fast Complementary Filter)
- [ ] FKF (Fast Kalman Filter)
- [x] FLAE (Fast Linear Attitude Estimator)
- [ ] FOAM (Fast Optimal Attitude Matrix)
- [x] Fourati (Nonlinear Dynamic Attitude Estimator)
- [x] FQA (Factored Quaternion Algorithm)
- [ ] GDA-LKF (Gradient-Descent Linear Kalman Filter)
- [x] Madgwick (Gradient-Descent Estimator)
- [ ] MAGYQ (Magnetic, Acceleration Fields and Gyroscope Quaternion)
- [x] Mahony (Nonlinear Explicit Complementary Filter)
- [ ] OLAE (Optimal Linear Attitude Estimator)
- [ ] OLEQ (Optimal Linear Estimator of Quaternion)
- [x] QUEST (QUaternion ESTimator)
- [ ] REQUEST (Recursive QUEST)
- [ ] ROLEQ (Recursive Optimal Linear Estimator of Quaternion)
- [ ] Sabatini
- [ ] SOLEQ (Sub-Optimal Linear Estimator of Quaternion)
- [x] TRIAD

## Notes for future versions

`ahrs` is still moving away from plotting and data parsing submodules to better focus in the algorithmic parts. Submodules `io` and `plot` will not be further developed and, eventually, will be removed.

This way you can also choose your favorite libraries for data loading and visualization. This also means, we get rid of its dependency on `matplotlib`.

## Documentation

A comprehensive documentation, with examples, will soon come to [Read the Docs](https://docs.readthedocs.io/).

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2024-06-18
### Added
- Added pyproject.toml file to replace the deprecated setup.py
- Added __version__ to main package __init__.py file

### Changed
- Moved from the setuptools to hatchling build backend
- Docutils version will now be installed by Sphinx
- Changed the docs/conf.py to get its version from the package import 
- Updated documentation for the new hatch workflow

### Removed
- Removed old requirements.txt
- Removed requirements.txt from docs
- Removed tools folder for version management. Check pyproject.toml

## [0.3.1] - 2021-09-29
### Added
- Error raise if geomagnetic field is invalid in class `EKF`.
- New method `Omega` in class `AQUA` to simplify the product between angular rate and quaternion.
- New method `rotate_by` in class `QuaternionArray`.
- More Acronyms in page `Nomeclature` of documentation.
- Individual pages for each method and attribute of classes `Quaternion` and `DCM` into documentation.
- Merge pull request of basic automation tests.

### Changed
- Fix undefined matrix `R` in method `update` of class `EKF`.
- Fix shape of converted points in function `ned2enu`.
- Fix parameters in function `rec2geo` of submodule `frames`.
- Fix method `from_quaternion` of class `DCM`.
- Fix Munich height (in km) in global constants.
- Improve detection of empty arrays when building `TRIAD` object.
- Improve description of estimator `AQUA` in its docstring.
- Improve imports in submodules `frames` and `dcm`.
- Improve style and descriptions in docstrings of functions in submodule `orientation`.
- Method `init_q` is now synonym of the more convenient method `estimate` in class `AQUA`.
- Parameter `as_quaternion` in method `estimate` of class `TRIAD` renamed to `representation`, and its value is now of type `str`.
- Versioning is defined using f-strings.

### Removed
- Redundant normalization of magnetic measurement vector in class `FQA`.

## [0.3.0] - 2021-02-03
### Added
- World Magnetic Model as class `WMM`.
- World Geodetic System as class `WGS`.
- Geodetic and planetary constants.
- New class `DCM` to represent Direction Cosine Matrices.
- Attitude Estimator `EKF`.
- Attitude Estimator `TRIAD`.
- Attitude Estimator `Davenport`.
- Attitude Estimator `QUEST`.
- Attitude Estimator `SAAM`.
- Attitude Estimator `OLEQ`.
- Attitude Estimator `ROLEQ`.
- Implementation of modes `newton`, `eig` and `symbolic` in estimator `FLAE`.
- New function `ecompass` to estimate Orientation from an observation of an accelerometer and a magnetometer.
- New method `row_reduction` in estimator `FLAE`.
- New methods `to_angles`, `is_pure`, `is_real`, `is_versor`, and `is_identity` in class `QuaternionArray`.
- New properties `logarithm` and `log` in class `Quaternion`.
- New parameter `versor` defaulting to `True` in class `Quaternion` to initialize quaternions as versors.
- New frame transformations `ned2enu`, `enu2ned`, `ecef2enu`, `enu2ecef`, `ll2ecef`, and `ecef2llf`.
- Requirements file.
- Manifest file.
- Documentation with Sphinx.
- Type hints.

### Changed
- Fix `AQUA`.
- Fix missing imports in several submodules.
- Fix versioning that prevented install from a fresh setup.
- `Tilt` can perform vectorized estimations over 2D arrays.
- `QuaternionArray` is now subclassed from a `numpy.array`.
- Simplify implementation of class `Complementary`.
- Create copies of quaternions and normalize them when given in `metrics` functions.
- Complete attitude estimator `FAMC`.
- Complete docstrings of all functions, methods and classes.
- Improve examples and information in `README`.

### Removed
- Submodules `plot` and `io` to remove dependencies on `scipy` and `matplotlib`.

## [0.2.2-dev1] - 2020-02-06
### Added
- Support for other characters as separators in function `load` of submodule `io`.
- Function `am2euler` to obtain Euler angles from gravity and geomagnetic measurements.
- Notice of discontinuation of `io` and `plotting` submodules.
- Add badge of code quality.

### Changed
- Using `isinstance(x, y)` instead of `type(x)==y` to confirm types of variables.
- Update setup information.

## [0.2.2] - 2020-01-06
### Added
- This changelog.
- Script `triad.py` for future submodule implementation of TRIAD method.

### Changed
- Action `Build Python Package` builds **only** when new commit is pushed to `master` branch.
- Simplify building triads in function `triad` of submodule `orientation`.
- Fix documentation of `am2angles` and `quest` of submodule `orientation`.

## [0.2.1] - 2020-12-28
### Added
- Class `QuaternionArray` to handle several quaternions at once.
- Add methods `log()`, `exp()`, `inv()`, `to_array()` and `to_list()` to class `Quaternion`.

## [0.2.0] - 2019-12-27
### Added
- Class `Quaternion` to handle quaternion operations.
- Add methods `log()`, `exp()`, `inv()`, `to_array()` and `to_list()` to class `Quaternion`.
- More definitions of colors with formats ints and floats.
- Functions `hex_to_int()` and `hex_to_float()` convert any color defined as hex into `int` and `float`.
- Badges in README to indicate basic information.

### Changed
- Improve versioning.

### Removed
- Function `R2q()` to get quaternion from rotation matrix. Method `from_DCM()` of class `Quaternion` is preferred.

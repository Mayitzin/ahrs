# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2020-01-06
### Added
- This changelog.
- Script `triad.py` for future submodule implementation of TRIAD method.

### Changed
- Action `Build Python Package` builds **only** when new commit is pushed to `master` branch.
- Simplify building triads in function `triad` of submodule `orientation`.
- Fix documentation of `am2angles` and `quest` of submodule `orientation`.

## [0.2.1] - 2020-01-06
### Added
- Class `QuaternionArray` to handle several quaternions at once.
- Add methods `log()`, `exp()`, `inv()`, `to_array()` and `to_list()` to class `Quaternion`.

### Changed
- Fix and/or improve 

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

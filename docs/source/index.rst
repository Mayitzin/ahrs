AHRS: Attitude and Heading Reference Systems
============================================

Welcome! ``ahrs`` is an open source Python toolbox for attitude estimation
using the most known algorithms, methods and resources.

It is designed to be flexible and simple to use, making it a great option for
fast prototyping, testing and integration with your own Python projects.

This package collects functions and utilities to help you understand and use
the most common techniques for attitude estimation, and in no way it is
recommended to be used commercially.

All algorithms and implementations have their proper documentation and
references, in case you need further clarification of their usage.

New in version 0.3
------------------

- The **World Magnetic Model** (`WMM <https://www.ngdc.noaa.gov/geomag/WMM/DoDWMM.shtml>`_)
  is fully implemented. It can be used to estimate all magnetic field elements
  on any given place of Earth for dates between 2015 and 2025.
- The *ellipsoid model* of the **World Geodetic System** (`WGS84
  <https://earth-info.nga.mil/GandG/update/index.php?dir=wgs84&action=wgs84>`_)
  is included. A full implementation of the **Earth Gravitational Model**
  (`EGM2008 <https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/egm08_wgs84.html>`_)
  is *NOT* available here, but the estimations of the main and derived
  parameters of the WGS84 using the ellipsoid model are carried out.
- The `International Gravity Formula <http://earth.geology.yale.edu/~ajs/1945A/360.pdf>`_
  and the EU's `WELMEC <https://www.welmec.org/documents/guides/2/>`_ normal
  gravity reference system are also implemented.
- New class ``DCM`` (derived from ``numpy.ndarray``) for orientation/rotation
  representations as 3x3 Direction Cosine Matrices.
- New class ``QuaternionArray`` to simultaneously handle an array with more
  quaternions at once.
- New submodule ``frames`` to represent the position of an object in different
  reference frames.
- `Metrics <https://ahrs.readthedocs.io/en/latest/metrics.html>`_ for rotations
  in 3D spaces using quaternions and direction cosine matrices.
- New operations, properties and methods for class ``Quaternion``, now also
  derived from ``numpy.ndarray``.
- A whole bunch of `new constant values <https://ahrs.readthedocs.io/en/latest/constants.html>`_
  (mainly for Geodesy) accessed from the top level of the package.
- Docstrings are improved with further explanations, references and equations
  whenever possible.
- New and improved estimators include a short, but clear documentation with
  references. Many use different sensor arrays. The available algorithms are:

=============  =========  =============  ============
Algorithm      Gyroscope  Accelerometer  Magnetometer
=============  =========  =============  ============
AQUA           Optional   YES            Optional
Complementary  YES        YES            Optional
Davenport's    NO         YES            YES
EKF            YES        YES            YES
FAMC           NO         YES            YES
FLAE           NO         YES            YES
Fourati        YES        YES            YES
FQA            NO         YES            Optional
Integration    YES        NO             NO
Madgwick       YES        YES            Optional
Mahony         YES        YES            Optional
OLEQ           NO         YES            YES
QUEST          NO         YES            YES
ROLEQ          NO         YES            YES
SAAM           NO         YES            YES
Tilt           NO         YES            Optional
TRIAD          NO         YES            YES
=============  =========  =============  ============

Deprecations
------------

Submodules ``io`` and ``plot`` are dismissed, removing dependecies on Scipy and
matplotlib. This decision was made with the intent to better focus on the
algorithmic part of the package.

Loading and visualizing the data is left to the preference of the user.

.. toctree::
   :maxdepth: 1

   installation

.. toctree::
   :maxdepth: 1
   :caption: Attitude Estimators

   filters

.. toctree::
   :maxdepth: 1
   :caption: World Models

   world_models

.. toctree::
   :maxdepth: 1
   :caption: Tools

   attitude_representations
   constants
   metrics
   nomenclature

Indices
=======

* :ref:`genindex`
* :ref:`search`

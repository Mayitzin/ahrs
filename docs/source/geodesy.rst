Geodesy
=======

.. automodule:: ahrs.utils.geodesy

Classes and Functions
---------------------

.. toctree::
   :titlesonly:

   geodesy/refEllipsoid
   geodesy/wmm
   geodesy/wgs84

Additionally, the library provides a set of methods that have been used
historically in geodesy, but are not part of the above impllementations.

.. toctree::
   :maxdepth: 1

   geodesy/igf
   geodesy/welmec

Reference Frames
----------------

.. automodule:: ahrs.common.frames

The most common **reference frames transformations** used in geodesy are also
implemented in the library.

.. toctree::
   :maxdepth: 1

   frames/ecef2enu
   frames/enu2ecef
   frames/ecef2geodetic
   frames/geodetic2ecef
   frames/geodetic2enu
   frames/aer2enu
   frames/enu2aer

Geodesy
=======

Geodesy is the science of measuring and understanding Earth's shape, size,
orientation in space, and gravity field.

The two most important geospatial systems used in mapping, navigation, and
Earth science applications are the **WGS84 (World Geodetic System 1984)** and
the **WMM (World Magnetic Model)**.

These systems often work together in navigation devices to provide both
positional (WGS84) and directional (WMM) information.

The **WGS84 (World Geodetic System 1984)** is a standard reference coordinate
system for Earth, providing a framework for mapping and navigation. It defines
the shape of the Earth as an ellipsoid, with a reference frame, and a
coordinate system.

It is widely used for global positioning systems (GPS) and serves as the
underlying geodetic system for latitude, longitude, and altitude coordinates.

WGS84 approximates Earth's shape using a `Reference Ellipsoid
<https://en.wikipedia.org/wiki/Earth_ellipsoid#Reference_ellipsoid>`_
model, which mathematically describes a celestial body as an `oblate spheroid
<https://en.wikipedia.org/wiki/Spheroid#Oblate_spheroids>`_.

The reference ellipsoid simplifies the calculation of other geodetic parameters,
such as the Earth's radius, circumference, and surface area.

It can also be used to build other planet's or moon's own geodetic systems.

The **WMM (World Magnetic Model)** is a mathematical model that represents the
Earth's magnetic field. It is used to calculate magnetic declination (the angle
difference between true north and magnetic north).

It provides magnetic field data for navigation systems, including compasses,
aircraft, and smartphones.

This model is updated every 5 years (e.g., 2020-2025 is the current model) to
account for changes in the Earth's magnetic field.

These three systems (WMM, reference ellipsoid, and WGS84) are implemented in
this library.

.. toctree::
   :titlesonly:

   refEllipsoid
   wmm
   wgs84
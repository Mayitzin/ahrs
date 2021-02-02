Nomenclature
============

These are the most common definitions in AHRS:

**Altitude** is the distance along the ellipsoidal normal between the surface
of the `ellipsoid <https://en.wikipedia.org/wiki/Reference_ellipsoid>`_ and a
point of interest above it.

**Attitude** is the orientation of a vehicle or other object relative to a
known reference, which could be a frame, the horizon, a direction of motion, or
even another object. The heading of the object cn be omitted in the definition
of its attitude.

**Body Frame** matches the frame of the platform the sensors are mounted on.
The origin coincides with the center of gravity of the platform. The Y-axis
points forward of the moving platform, while the Z-axis points upwards. The
X-axis completes the right-hand system pointing in traverse direction. Some
literature names it **b-frame**.

**Earth-Centered Earth-Fixed Frame** (ECEF) has its origin and z-axis aligned
to the ECI frame, but rotates along with the Earth. Therefore is Earth-Fixed.
Some literature names it **e-frame**.

**Earth-Centered Inertial Frame** (ECI) has the origin at the center of mass of
the Earth. The X-axis points towards the `vernal equinox <https://en.wikipedia.org/wiki/March_equinox>`_
in the equatorial plane. The Z-axis is along the rotation axis of the Earth.
The y-axis completes with a right-hand system. Some literature names it **i-frame**.

**East-North-Up Frame** (ENU) is an LLF with the X-axis pointing East, Y-axis
pointing to the true North, and the Z-axis completes a right-hand system
pointing up (away from Earth.)

**Elevation** is the height above or below a fixed reference point. In
Geography, it is most commonly the height of a terrain above (or below) the sea
level.

**Geodetic** (also ellipsoidal or curvilinear) **coordinates** in the ECEF are
defined for positioning elements on or near the Earth.

**Inertial Measurement Unit** is a device fitted with inertial sensors
(accelerometers and gyroscopes) measuring a body's `specific force <https://en.wikipedia.org/wiki/Specific_force>`_
and `angular rate <https://en.wikipedia.org/wiki/Angular_frequency>`_.

**Inertial Navigation System** is a navigation architecture consisting of an
IMU, and a computer to continuously estimate a device's positoin and orientation using dead reckoning.

**Latitude** is the angle in the meridian plane from the equatorial plane to
the ellipsoidal normal at he point of interest.

**Local-Level Frame** (LLF) is the local navigation frame, whose origin
coincides with the sensor frame. Some literature names it **l-frame**.

**Longitude** is the angle in the equatorial plane from the prime meridian to
the projection of the point of interest onto the equatorial plane.

**Meridian** is the half of a creat circle on Earth's surface terminated by the
poles.

**North-East-Down Frame** (NED) is an LLF with the X-axis pointing to the true
North, Y-axis pointing East, and the Z-axis completing the right-hand system
pointing Down.

**Orientation** is the rotation needed to place an object from a reference
placement to its current placement, neglecting its position or location in
space.

**Rectangular coordinates** in the ECEF represent position of a point with its
x, y, and z vector components aligned parallel to the corresponding e-frame
axes.

**Strapdown** INS has accelerometers rigidly mounted parallel to the body axes
of the vehicle. In this application the gyroscopes do not provide a stable
platform; they are instead used to sense the turning rates of the craft.

Abbreviations
-------------

======  =========
AACGM   Altitude-Adjusted Corrected Geomagnetic Coordinates
ADC     Analog-to-Digital Converter
ADEV    Allan deviation
AHRS    Attitude and Heading Reference System
AoA     Angle of Arrival
AoD     Angle of Departure
AR      Angular Rate
atan    Arctangent
atan2   Arctangent (four quadrants)
AUV     Autonomous Underwater Vehicle
AVAR    Allan variance
BIH     Bureau International de l'Heure
BPF     Band-Pass Filter
CD      Centered Dipole
CGM     Corrected Geomagnetic Coordinates
CIRS    Conventional Inertial Reference System
CRS     Conventional Celestial Reference System
CTP     Conventional Terrestrial Pole
CTRS    Conventional Terrestrial Reference System
DCM     Direction Cosine Matrix
DoD     Department of Defense
DoF     Degrees of Freedom
DR      Dead Reckoning
ECEF    Earth-Centered Earth-Fixed
ECIF    Earth-Centered Inertial Frame
EGM     Earth Gravitational Model
EGNSS   Enhanced GNSS
EKF     Extended Kalman Filter
ENU     East-North-Up
ESA     European Space Agency
EU      European Union
GNSS    Global Navigation Satellite System
GPS     Global Positioning System
GTRF    Galileo Terrestrial Reference Frame
gyro    Gyroscope
HP      High Performance
HPF     High-Pass Filter
IAU     International Astronomical Union
IC      Integrated Circuit
IERS    International Earth Rotation and Reference Systems Service
IGRF    International Geomagnetic Reference Field
IMU     Inertial Measurement Unit
INS     Inertial Navigation System
INU     Inertial Navigation Unit
CIRAS   Coriolis Inertial Rate and Acceleration Sensor
ITRF    International Terrestrial Reference Frame
KF      Kalman Filter
lat     Latitude
LLF     Local-level Frame
lon     Longitude
LPF     Low-Pass Filter
MagCal  Magnetic Calibration
MARG    Magnetism, Angular Rate, and Gravity
MCU     Micro-Controller Unit
MEMS    Micro-electromechanical Systems
MLT     Magnetic Local Time
MOEMS   Micro-opto electromechanical Systems
MST     Microsystem Technology
μC      Microcontroller
NED     North-East-Down
PF      Particle Filtering
PIGA    Pendulous Integrating Gyroscopic Accelerometer
QD      Quasi-Dipole
QUEST   Quaternion Estimator
RHR     Right-Hand Rule
SCI     Serial Communications Interface
SI      Système International d'unités
SLERP   Spherical Linear Interpolation
SOA     Silicon Oscillating Accelerometer
SVD     Singular Value Decomposition
TFG     Turning Fork Gyroscope
UART    Universal Asynchronous Receiver / Transmitter
UAV     Unmanned Aerial Vehicle
UKF     Unscented Kalman Filter
UT      Universal Time
WGS     World Geodetic System
WMM     World Magnetic Model
ZUPT    Zero Velocity Update
======  =========


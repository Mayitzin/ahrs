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
ACARE   Advisory Council for Aeronautics Research in Europe
ACAS    Airborne Collision Avoidance System
ACC     Adapttive Cruise Control
Ack     Acknowledge
ADC     Analog-to-Digital Converter
ADCS    Active Directed Control System
ADI     Attitude Director Indicator
ADIRS   Air Data Inertial Reference System
ADIRU   Air Data Inertial Reference Unit
ADEV    Allan deviation
AGNSSS  Assisted Global Navigation Satellite System
AGPS    Assisted Global Positioning System
AHRS    Attitude and Heading Reference System
AoA     Angle of Arrival
AoD     Angle of Departure
AR      Angular Rate
asl     above sea level
atan    Arctangent
atan2   Arctangent (four quadrants)
AQUA    Algebraic Quaternion Algorithm
AU      Astronomical Unit
AUV     Autonomous Underwater Vehicle
AVAR    Allan variance
AVCS    Autonomous Vehicle Control System
AVG     Autonomous Vehicle Guidance
AVLS    Autonomous Vehicle Localization System
AWGN    Additive White Gaussian Noise
az      Azimuth
B2B     Business to Business
BIH     Bureau International de l'Heure
BIIT    Built-in Integrity Test
BLE     Bluetooth Low Energy
BLUE    Best Linear Unbiased Estimate
BPF     Band-Pass Filter
CAN     Controller Area Network
CAS     Collision Avoidance System
CD      Centered Dipole
CDI     Course Deviation Indicator
CGM     Corrected Geomagnetic Coordinates
CIRAS   Coriolis Inertial Rate and Acceleration Sensor
CIRS    Conventional Inertial Reference System
CLIK    Closed-Loop Inverse Kinematics
CRS     Conventional Celestial Reference System
CTP     Conventional Terrestrial Pole
CTRS    Conventional Terrestrial Reference System
DAAS    Data as a Service
DAC     Digital-to-Analog Converter
dB      Decibel
DCM     Direction Cosine Matrix
Dec     Declination
deg     Degrees
Dev     Deviation
DGPS    Differential Global Positioning System
DME     Distance Measuring Equipment
DoD     Department of Defense
DoF     Degrees of Freedom
dps     Degrees per Second
DR      Dead Reckoning
DRS     Dead Reckoning System
DSP     Digital Signal Processing
ECDIS   Electronic Chart Display and Information Systems
ECEF    Earth-Centered Earth-Fixed
ECIF    Earth-Centered Inertial Frame
ECU     Electronic Control Unit
EDA     European Defence Agency
EEA     European Economic Area
EFCS    Electronic Flight Control System
EGM     Earth Gravitational Model
EGNOS   European Geostationary Navigation Overlay Service
EGNSS   Enhanced GNSS
EGPS    Enhanced Global Positioning System
EGR     Embedded Global Positioning System Receiver
EIR     Earth Inertial Reference
EKF     Extended Kalman Filter
EMI     Electromagnetic Interference
ENU     East-North-Up
ENS     Electronic Navigation System
ESA     European Space Agency
ETRS    European Terrestrial Reference System
EU      European Union
FAA     Federal Aviation Administration
FADEC   Full Authority Digital Engine Controller
FMS     Flight Management System
GAGAN   GPS and Geo Augmented Navigation
GGF     Global Earth-fixed frame with one axis aligned with gravity
GIS     Geographic Information System
GNSS    Global Navigation Satellite System
GPS     Global Positioning System
GTRF    Galileo Terrestrial Reference Frame
gyro    Gyroscope
HIL     Hardware-in-the-Loop
HMI     Human-Machine Interface
HP      High Performance
HPF     High-Pass Filter
IAG     International Association of Geodesy
IATA    International Air Transport Association
IAU     International Astronomical Union
IC      Integrated Circuit
ICAO    International Civil Aviation Organization
ICRF    International Celestial Reference Frame
ICRS    International Celestial Reference System
IERS    International Earth Rotation and Reference Systems Service
IFR     Instrument Flight Rules
IGRF    International Geomagnetic Reference Field
ILS     Instrument landing system
IMU     Inertial Measurement Unit
INS     Inertial Navigation System
INU     Inertial Navigation Unit
ITRF    International Terrestrial Reference Frame
ITRS    International Terrestrial Reference System
ITS     Intelligent Transportation Systems
JPALS   Joint Precision Approach and Landing System
KF      Kalman Filter
LAAS    Local Area Augmentation System
lat     Latitude
LERP    Linear Interpolation
LLF     Local-level Frame
LNAV    Lateral Navigation
lon     Longitude
LORAN   Long-range radio navigation
LPF     Low-Pass Filter
LPV     Localizer Performance with Vertical Guidance
MagCal  Magnetic Calibration
MANET   Mobile ad hoc Network
MARG    Magnetism, Angular Rate, and Gravity
MCU     Micro-Controller Unit
MEMS    Micro-electromechanical Systems
MHIL    Micro Hardware-in-the-Loop
MHW     Mean High Water
MIMU    Magnetic and Inertial Measurement Unit
MIS     Management information system
MLF     Marker-cluster Local Frame
MLS     Microwave landing system
MLT     Magnetic Local Time
mocap   Motion Capture
MOEMS   Micro-opto Electromechanical Systems
MST     Microsystem Technology
μC      Microcontroller
NAD     North American Datum
NASA    National Aeronautics and Space Administration
NDB     Non-directional beacon
NED     North-East-Down
NSE     Navigation System Error
PBN     Performance-Based Navigation
PDR     Pedestrian Dead Reckoning
PF      Particle Filtering
PIGA    Pendulous Integrating Gyroscopic Accelerometer
PLC     Programmable Logic Controller
PUMA    Programmable Universal Manipulation Arm
QD      Quasi-Dipole
QUEST   Quaternion Estimator
RHR     Right-Hand Rule
RMS     Root Mean Square
RNAV    Area Navigation
RTU     Remote Terminal Unit
SBAS    Satellite-Based Augmentation System
SCADA   System Control and Data Acquisition
SCI     Serial Communications Interface
SI      Système International d'unités
SLERP   Spherical Linear Interpolation
SOA     Silicon Oscillating Accelerometer
SVD     Singular Value Decomposition
TACAN   Tactical Air Navigation System
TAWS    Terrain Awareness and Warning System
TFG     Turning Fork Gyroscope
TLS     Transponder landing system
TSE     Total System Error
UART    Universal Asynchronous Receiver / Transmitter
UAV     Unmanned Aerial Vehicle
UKF     Unscented Kalman Filter
UT      Universal Time
V2V     Vehicle-to-Vehicle
VANET   Vehicular ad hoc Network
VFR     Visual Flight Rules
VNAV    Vertical Navigation
VOR     Very High Frequency Omnidirectional Radio Range
WAAS    Wide Area Augmentation System
WGS     World Geodetic System
WMM     World Magnetic Model
ZUPT    Zero Velocity Update
======  =========

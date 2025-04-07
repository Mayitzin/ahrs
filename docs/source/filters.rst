
Attitude Estimators
===================

Several attitude estimators written in pure Python as classes are included with
``AHRS`` in the module ``filters``, and they can be accessed easily with a
simple import.

For example, importing the QUEST estimator is simply done with:

.. code-block:: python

   >>> from ahrs.filters import QUEST

Most estimators are built to be working with signals of low-cost strapdown
navigation systems. Three types of sensors are mainly used for this purpose:

- **Gyroscopes** measure the angular velocity.
- **Accelerometers** measure the acceleration (rate of change of velocity).
- **Magnetometers** measure the local magnetic field.

Gyroscopes, provide good short-term reliability and resistance to vibration,
accelerometers provide information that is reliable over time, and
magnetometers provide heading information in addition to the limited attitude
information (pitch and roll).

For an attitude estimation we encounter two common strategies:

- **Instantaneous estimation** calculates the attitude using vectors in two
  frames (a body frame and a known reference frame). It finds the attitude at a
  single point in time, without necessarily considering the kinematics of the
  objective. Ideally, this works with a system in a quasi-static state. Thus,
  this estimation is sometimes called **Static Attitude Determination**.
- **Recursive estimation** not only uses vectorial observations, but also takes
  the system dynamics into account to capture and predict the behaviour of the
  system. Because the system kinematics are considered, these type of strategy
  is also called **Dynamic Attitude Determination**.

The most accurate estimators are the dynamic ones, but they are, generally,
more computationally demanding, against the much simpler and faster static
estimators.

Dynamic estimators mainly use angular motions to compute the attitude. These
displacements are normally measured by gyroscopes in a
`strapdown <https://en.wikipedia.org/wiki/Inertial_navigation_system#Strapdown_systems)>`_
system, where the sensors are rigidly attached to the body's frame.

The gyroscopes capture the angular velocity of such body, which is then
integrated over time and added to the previous estimations to continuously
obtain a new estimation after every sample. For them to work, however, an
initial orientation has to be known, so that it can *grow* on top of it.

The following algorithms are implemented in this package:

=============  =========  =============  ============
Algorithm      Gyroscope  Accelerometer  Magnetometer
=============  =========  =============  ============
AQUA           YES        Optional       Optional
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
UKF            YES        YES            NO
=============  =========  =============  ============

.. toctree::
   :maxdepth: 1

   filters/angular
   filters/aqua
   filters/complementary
   filters/davenport
   filters/ekf
   filters/famc
   filters/fkf
   filters/flae
   filters/fourati
   filters/fqa
   filters/madgwick
   filters/mahony
   filters/oleq
   filters/quest
   filters/roleq
   filters/saam
   filters/tilt
   filters/triad
   filters/ukf

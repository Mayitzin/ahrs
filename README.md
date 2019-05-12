# AHRS: Attitude and Heading Reference Systems

AHRS is a zoo of functions and objects written in Python helping you to estimate the orientation and position of robotic systems.

Orginally, an [AHRS](https://en.wikipedia.org/wiki/Attitude_and_heading_reference_system) is defined as a set of orthogonal sensors providing attitude information about an aircraft. This field is now expanding to smaller devices, like wearables, automated transportation and all kinds of robots in motion.

The module __AHRS__ is developed with a focus on fast prototyping and easy modularity.

AHRS is compatible with __Python 3.6__ and above.

## Installation

First, clone AHRS using `git`:

```sh
git clone https://github.com/Mayitzin/ahrs.git
```

Then, `cd` to the AHRS folder and run the install command:
```sh
cd ahrs
sudo python setup.py install
```

Alternatively, you can install AHRS using `pip`
```sh
sudo pip install .
```

AHRS depends on the most distributed packages of Python. If you don't have them, they will be automatically downloaded and installed.

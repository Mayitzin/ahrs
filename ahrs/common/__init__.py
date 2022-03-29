# -*- coding: utf-8 -*-
"""
These are the most common routines used to estimate attitude and heading.

"""

from . import mathfuncs
from . import orientation
from . import geometry
from . import frames

from .constants import M_PI, DEG2RAD, RAD2DEG
from .quaternion import Quaternion
from .dcm import DCM

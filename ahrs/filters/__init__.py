# -*- coding: utf-8 -*-
"""
These are the most common attitude filters.

References
----------
.. [1] http://www.olliw.eu/2013/imu-data-fusing/
.. [2] https://motsai.com/omid-vs-madgwick-low-power-orientation-filters/

"""

from .madgwick import Madgwick
from .mahony import Mahony
from .ekf import EKF
from .fourati import Fourati
from .fqa import FQA
from .aqua import AQUA
from .angular import AngularRate
from .flae import FLAE
from .gravityquaternion import GravityQuaternion
from .complementary import Complementary

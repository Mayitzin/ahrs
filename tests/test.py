#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Test File
=========

"""

import sys

from test_madgwick import test_madgwick
from test_mahony import test_mahony
from test_imports import test_module_imports
from test_ekf import test_ekf

if __name__ == "__main__":
    # test_module_imports()
    # test_mahony()
    # test_madgwick()
    test_ekf()

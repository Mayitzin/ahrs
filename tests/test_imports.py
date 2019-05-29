# -*- coding: utf-8 -*-
"""
Test module imports
===================

"""

import sys

def test_module_imports():
    try:
        import ahrs
    except:
        sys.exit("[ERROR] Package AHRS not found. Go to root directory of package and type:\n\n\tpip install .\n")
    try:
        import numpy, scipy, matplotlib
    except ModuleNotFoundError:
        sys.exit("[ERROR] You don't have the required packages. Try reinstalling the package.")

#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Test Script
===========

"""

from test_madgwick import test_madgwick
from test_mahony import test_mahony
from test_ekf import test_ekf
from test_fourati import test_fourati
from test_metrics import *

def test_filters(**kwargs):
    """
    Test Attitude Estimation Filters

    For now each test is imported from a different test script.

    To Do:

    - Create a unique script with, possibly, a class holding all tests method
      for each filter. Then call a desired method to test a filter.

    Parameters
    ----------
    file : str
        Name of the file to be used as test.
    plot : bool
        Flag to indicate if results are to be plotted. Requires package matplotlib

    """
    results = True
    # results &= test_fourati(**kwargs)
    # results &= test_ekf(**kwargs)
    # results &= test_mahony(**kwargs)
    results &= test_madgwick(**kwargs)
    print("Filter testing results: {}".format("OK" if results else "ERROR"))

def test_metrics(**kwargs):
    result = test_dist()

if __name__ == "__main__":
    test_filters(plot=True)
    # test_metrics()

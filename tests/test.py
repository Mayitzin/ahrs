#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Test Script
=========

"""

from test_madgwick import test_madgwick
from test_mahony import test_mahony
from test_ekf import test_ekf
from test_fourati import test_fourati

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
    # test_mahony(**kwargs)
    # test_madgwick(**kwargs)
    # test_ekf(**kwargs)
    test_fourati(**kwargs)

if __name__ == "__main__":
    test_filters(plot=True)

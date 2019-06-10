#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Test Script
===========

"""

from test_metrics import *
from test_filters import Test_Filter

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
    file_name = kwargs.get('file', "ExampleData.mat")
    test = Test_Filter(file_name, **kwargs)
    results = True
    results &= test.ekf()
    results &= test.fourati()
    results &= test.mahony()
    results &= test.madgwick()
    print("Filter testing results: {}".format("OK" if results else "FAILED"))

def test_metrics(**kwargs):
    result = test_dist()

if __name__ == "__main__":
    test_filters()
    # test_metrics()

#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Test Script
===========

Note:
- The angular velocity of ExampleData.mat is in degrees/second with f=256.0
- The angular velocity of repoIMU.csv is in radians/second with f=100.0

"""

from test_metrics import *
from test_filters import Test_Filter
import ahrs

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

def test_plot(**kwargs):
    """
    Test plotting capabilities of the package
    """
    file_name = kwargs.get('file', "repoIMU.csv")
    freq = kwargs.get('freq', 100.0)
    data = ahrs.utils.io.load(file_name)

    # filtered = ahrs.filters.Fourati(data, k=0.001, ka=0.01, km=0.1)
    # filtered = ahrs.filters.FQA(data, frequency=freq)
    # filtered = ahrs.filters.AQUA(data, frequency=100.0)
    # filtered = ahrs.filters.EKF(data, frequency=freq, noises=[0.1, 0.1, 0.1])
    # filtered = ahrs.filters.Mahony(data, Kp=0.2, Ki=0.1, frequency=freq)
    filtered = ahrs.filters.Madgwick(data, beta=0.01, frequency=freq)

    if data.q_ref is None:
        ahrs.utils.plot_quaternions(filtered.Q, subtitles=["Estimated"])
    else:
        ahrs.utils.plot_quaternions(data.q_ref, filtered.Q, subtitles=["Reference", "Estimated"])

if __name__ == "__main__":
    test_filters()
    # test_metrics()
    # test_plot(file="ExampleData.mat", freq=256.0)
    # test_plot(file="repoIMU.csv", freq=100.0)

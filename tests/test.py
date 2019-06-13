#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Test Script
===========

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
    data = ahrs.utils.io.load(file_name)
    Q = np.tile([1., 0., 0., 0.], (data.num_samples, 1))
    fourati = ahrs.filters.Fourati(k=0.001, ka=0.1, km=0.1)
    for t in range(1, data.num_samples):
        Q[t] = fourati.update(data.gyr[t], data.acc[t], data.mag[t], Q[t-1])
    ahrs.utils.plot_quaternions(data.qts, Q)
    # madgwick = ahrs.filters.Madgwick(beta=0.01, frequency=100.0)
    # for t in range(1, data.num_samples):
    #     Q[t] = madgwick.updateIMU(data.gyr[t], data.acc[t], Q[t-1])
    #     # Q[t] = madgwick.updateMARG(data.gyr[t], data.acc[t], data.mag[t], Q[t-1])
    # ahrs.utils.plot_quaternions(data.qts, Q)

if __name__ == "__main__":
    # test_filters()
    # test_metrics()
    test_plot()

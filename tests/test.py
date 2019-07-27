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
    print(test_dist())

def test_plot(**kwargs):
    """
    Test plotting capabilities of the package
    """
    freq = kwargs.get('freq', 100.0)
    file_name = kwargs.get('file', "repoIMU.csv")
    data = ahrs.utils.io.load(file_name)

    orientation = ahrs.filters.AngularRate(data, frequency=freq)
    # orientation = ahrs.filters.FLAE(data)
    # orientation = ahrs.filters.Fourati(data, k=0.001, ka=0.01, km=0.1)
    # orientation = ahrs.filters.FQA(data, frequency=freq)
    orientation_2 = ahrs.filters.GravityQuaternion(data)
    # orientation = ahrs.filters.AQUA(data)
    # orientation = ahrs.filters.EKF(data, frequency=freq, noises=[0.1, 0.1, 0.1])
    # orientation = ahrs.filters.Mahony(data, Kp=0.2, Ki=0.1, frequency=freq)
    # orientation = ahrs.filters.Madgwick(data, beta=0.01, frequency=freq)

    # ahrs.utils.plot_sensors(data.acc)

    if data.q_ref is None:
        ahrs.utils.plot_quaternions(orientation.Q, orientation_2.Q, subtitles=["Angular", "Gravity"])
    else:
        ahrs.utils.plot_quaternions(data.q_ref, orientation.Q, orientation_2.Q, subtitles=["Reference", "Angular", "Gravity"])

def test_load(path):
    data = ahrs.utils.io.load_OxIOD(path, sequence=3)
    q_c = ahrs.common.orientation.q_correct(data.q_ref)
    freq = ahrs.utils.io.get_freq(data.imu_time)
    # orientation = ahrs.filters.Madgwick(data, frequency=freq, beta=0.0001)
    # # data = ahrs.utils.io.load_ETH_EC(path)
    ahrs.utils.plot_sensors(data.gyr, data.acc, x_axis=data.imu_time)
    # ahrs.utils.plot_quaternions(data.q_ref, q_c, orientation.Q)

def test_quat(path):
    data = ahrs.utils.io.load_ETH_EuRoC(path)
    q = data.q_ref
    q_c = ahrs.common.orientation.q_correct(q, full=False)
    freq = ahrs.utils.io.get_freq(data.time, units='ns')
    # Estimate Quaternion
    # orientation = ahrs.filters.AngularRate(data, frequency=freq)
    orientation = ahrs.filters.Madgwick(data, frequency=freq, beta=0.01)
    # Compute Euler Angles
    num_samples = q.shape[0]
    q_eul = np.zeros((num_samples, 3))
    q_c_eul = np.zeros((num_samples, 3))
    q_eul_est = np.zeros((num_samples, 3))
    for i in range(num_samples):
        q_eul[i] = ahrs.common.orientation.q2euler(q[i])
        q_c_eul[i] = ahrs.common.orientation.q2euler(q_c[i])
        q_eul_est[i] = ahrs.common.orientation.q2euler(orientation.Q[i])
    ahrs.utils.plot_euler(q_eul, q_c_eul, q_eul_est, subtitles=["Reference", "Corrected", "Estimated"])
    # ahrs.utils.plot_sensors(data.acc, data.gyr, subtitles=["Acceleration", "Angular Rate"])

if __name__ == "__main__":
    # test_filters()
    # test_plot(file="ExampleData.mat", freq=256.0)
    # test_plot(file="repoIMU.csv", freq=100.0)
    test_load("../../Datasets/OxIOD/pocket/data1/raw")
    # test_quat('../../Datasets/EuRoC/V1_01_easy')

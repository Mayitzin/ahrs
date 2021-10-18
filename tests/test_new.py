#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
Pytest-based testing
====================
This file performs automated tests with pytest. It does not generate charts
or output to be reviewed.
Run with: pytest-3 tests/test_new.py
Run with: pytest-3 tests/test_new.py -s -vv --cov=ahrs for coverage + verbose

Copyright 2021 Mario Garcia and Federico Ceratto <federico@debian.org>
Released under MIT License
Formatted with Black

References
----------
.. [Crassidis] John L. Crassidis (2007) A Survey of Nonlinear Attitude
    Estimation Methods.
.. [Teage] Harris Teage (2016) Comparison of Attitude Estimation Techniques for
    Low-cost Unmanned Aerial Vehicles.
    https://arxiv.org/pdf/1602.07733.pdf
    http://ancs.eng.buffalo.edu/pdf/ancs_papers/2007/att_survey07.pdf
.. [Cirillo] A. Cirillo et al. (2016) A comparison of multisensor attitude
    estimation algorithms.
    https://www.researchgate.net/profile/Pasquale_Cirillo/publication/303738116_A_comparison_of_multisensor_attitude_estimation_algorithms/links/5750181208aeb753e7b4a0c0/A-comparison-of-multisensor-attitude-estimation-algorithms.pdf
"""

import numpy as np
import pytest
import scipy.io as sio

import ahrs
import ahrs.utils.io

DEG2RAD = ahrs.common.DEG2RAD


class Data:
    acc = None
    gyr = None
    mag = None


@pytest.fixture()
def data():
    fn = "ExampleData.mat"
    mat = sio.loadmat(fn)
    d = Data()
    d.acc = mat["Accelerometer"]
    d.gyr = mat["Gyroscope"]
    d.mag = mat["Magnetometer"]
    d.num_samples = len(d.acc)
    assert d.num_samples
    assert len(d.acc[0]) == 3
    assert len(d.gyr[0]) == 3
    assert len(d.mag[0]) == 3
    return d


def check_integrity(Q):
    assert Q is not None
    sz = Q.shape
    qts_ok = not np.allclose(np.sum(Q, axis=0), sz[0] * np.array([1.0, 0.0, 0.0, 0.0]))
    qnm_ok = np.allclose(np.linalg.norm(Q, axis=1).mean(), 1.0)
    assert qts_ok and qnm_ok


@pytest.fixture()
def Q(data):
    q = np.zeros((data.num_samples, 4))
    q[:, 0] = 1.0
    return q


def test_fourati(data, Q):
    fourati = ahrs.filters.Fourati()
    for t in range(1, data.num_samples):
        Q[t] = fourati.update(Q[t - 1], DEG2RAD * data.gyr[t], data.acc[t], data.mag[t])
    # check_integrity(Q)
    assert tuple(Q[0]) == (
        0.9999984512506995,
        -7.923098356158542e-05,
        -0.00010998618261451432,
        7.783371117384885e-05,
    )
    assert tuple(Q[-1]) == (
        0.8321632262796078,
        0.17064875423856807,
        -0.27862737470349475,
        0.44805150772046,
    )


def test_ekf(data, Q):
    ekf = ahrs.filters.EKF()
    for t in range(1, data.num_samples):
        Q[t] = ekf.update(Q[t - 1], DEG2RAD * data.gyr[t], data.acc[t], data.mag[t])
    check_integrity(Q)
    assert tuple(Q[0]) == (1.0, 0.0, 0.0, 0.0)
    assert tuple(Q[1]) == (
        0.9948152433072915,
        0.030997430898554206,
        -0.09666743395232329,
        0.006099030596487108,
    )
    assert tuple(Q[-1]) == (
        0.08996443890695231,
        0.23991941374716044,
        -0.958073763949303,
        -0.1282175396402196,
    )


def test_mahony(data, Q):
    mahony = ahrs.filters.Mahony()
    for t in range(1, data.num_samples):
        Q[t] = mahony.updateMARG(
            Q[t - 1], DEG2RAD * data.gyr[t], data.acc[t], data.mag[t]
        )
    check_integrity(Q)
    assert tuple(Q[0]) == (
        0.9999883099133865,
        -0.0007983637760660701,
        0.004762298093153807,
        0.00025133388483027455,
    )
    assert tuple(Q[-1]) == (
        -0.10375763267292282,
        -0.007875376758085736,
        -0.05233084545763538,
        0.9931937448034588,
    )


def test_madgwick(data, Q):
    madgwick = ahrs.filters.Madgwick()
    for t in range(1, data.num_samples):
        Q[t] = madgwick.updateMARG(
            Q[t - 1], DEG2RAD * data.gyr[t], data.acc[t], data.mag[t]
        )
    check_integrity(Q)
    assert tuple(Q[0]) == (
        0.999999906169997,
        -0.00039564882735884275,
        -0.00017641407301677547,
        -2.78332338967451e-07,
    )
    assert tuple(Q[-1]) == (
        0.9524138044137933,
        -0.10311931931141746,
        0.0038985200624795592,
        0.28680856453062387,
    )


def test_distance():
    a = np.random.random((2, 3))
    d = ahrs.utils.metrics.euclidean(a[0], a[1])
    assert np.allclose(d, np.linalg.norm(a[0] - a[1]))

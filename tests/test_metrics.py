# -*- coding: utf-8 -*-
"""
Test Metrics

"""

import numpy as np
import ahrs

def test_dist(**kwargs):
    """
    Test Distance
    """
    a = np.random.random((2, 3))
    d = ahrs.utils.metrics.euclidean(a[0], a[1])
    result = np.allclose(d, np.linalg.norm(a[0] - a[1]))
    return result

# -*- coding: utf-8 -*-
"""
Quaternion Class


"""

import numpy as np

class Quaternion:
    def __init__(self, v, *args, **kwargs):
        q = np.concatenate(([0.0], v)) if v.shape[-1] == 3 else np.array(v)
        if q.ndim not in [1, 2] or q.shape[-1] != 4:
            raise ValueError("Expected `q` to have shape (4,) or (N x 4), "
                             "got {}.".format(q.shape))
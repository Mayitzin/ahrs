# -*- coding: utf-8 -*-
"""
Filtering routines

"""

import numpy as np

def madgwick(x):
    """
    Return the Madgwick routine.

    Parameters
    ----------
    x : float
        Data

    Returns
    -------
    y : float
        Output

    """
    return 2.0*x

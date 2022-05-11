"""
Contains the core utilities for the proper use of AHRS: assertions, data
handling, etc.

This module is private. All functions and objects are available in the main
``ahrs`` namespace, or its corresponding submodule - use that instead.
"""

import numpy as np

def _assert_iterable(item, item_name: str = 'iterable'):
    if not isinstance(item, (list, tuple, np.ndarray)):
        raise TypeError(f"{item_name} must be given as an array. Got {type(item)}")

def _assert_numerical_iterable(item, item_name: str = 'iterable'):
    _assert_iterable(item, item_name)
    for i in item:
        if not isinstance(i, (int, float)):
            raise TypeError(f"{item_name} must be given as an array of numeric values. Got {type(i)}")

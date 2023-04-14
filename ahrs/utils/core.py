"""
Contains the core utilities for the proper use of AHRS: assertions, data
handling, etc.

These functions have no other goal in this package than to be used by other
modules. They are not meant to be used by the user.

This module is private. All functions and objects are available in the main
``ahrs`` namespace, or its corresponding submodule - use that instead.
"""

import numpy as np

def _assert_valid_array_type(item, item_name: str = 'iterable'):
    """Assert it is an iterable"""
    # NOTE: This could be changed to a more pythonic solution looking for the
    # dunder method __iter__(), but that yields strings too.
    if not isinstance(item, (list, tuple, np.ndarray)):
        raise TypeError(f"{item_name} must be given as an array. Got {type(item)}")

def _assert_numerical_iterable(item, item_name: str = 'iterable'):
    _assert_valid_array_type(item, item_name)
    item_copy = np.copy(item)
    if not(item_copy.dtype == np.dtype(int) or item_copy.dtype == np.dtype(float)):
        raise TypeError(f"{item_name} must have numerical values. Got {item_copy.dtype.name}")

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
    """Assert it is a list, tuple, or numpy array"""
    # NOTE: This could be changed to a more pythonic solution looking for the
    # dunder method __iter__(), but that yields strings too.
    if not isinstance(item, (list, tuple, np.ndarray)):
        raise TypeError(f"{item_name} must be given as an array. Got {type(item)}")

def _assert_numerical_iterable(item, item_name: str = 'iterable'):
    """Assert it is a list, tuple, or numpy array, and that it has numerical values"""
    _assert_valid_array_type(item, item_name)
    item_copy = np.copy(item)
    if not(item_copy.dtype == np.dtype(int) or item_copy.dtype == np.dtype(float)):
        raise TypeError(f"{item_name} must have numerical values. Got {item_copy.dtype.name}")

def get_nan_intervals(data: np.ndarray) -> list:
    """
    Get indices of NaN samples in data array.

    Based on an answer by `JonSG
    <https://codereview.stackexchange.com/users/105468/jonsg>`_ in
    `StackExchange <https://codereview.stackexchange.com/a/262803>`_.

    Parameters
    ----------
    data : array
        Data array.

    Returns
    -------
    intervals : list
        List of intervals of data samples with NaN.

    Examples
    --------
    >>> A = np.random.random((10, 3))
    >>> A[[1, 3, 4, 5, 8, 9]] = np.nan
    >>> A
    array([[0.74178186, 0.28444646, 0.35219214],
           [       nan,        nan,        nan],
           [0.05090735, 0.04161279, 0.10590561],
           [       nan,        nan,        nan],
           [       nan,        nan,        nan],
           [       nan,        nan,        nan],
           [0.31920402, 0.1523402 , 0.18907205],
           [0.32899368, 0.20042986, 0.73725579],
           [       nan,        nan,        nan],
           [       nan,        nan,        nan]])
    >>> get_nan_intervals(A)
    [(1, 1), (3, 5), (8, 9)]
    """
    data = np.copy(data)
    if data.ndim not in [1, 2]:
        raise ValueError(f"data array must be 1- or 2-dimensional. It has {data.ndim} dimensions.")
    isnan_list = np.any(np.isnan(data), axis=1) if data.ndim > 1 else np.isnan(data)
    nan_indices = np.where(isnan_list == True)[0]
    intervals = np.split(nan_indices, np.where(np.diff(nan_indices) > 1)[0] + 1)
    if len(intervals) == 0:
        return []
    return [(interval[0], interval[-1]) for interval in intervals]

#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Input and Output routines

To Do:

- Add support for loading of CSV Files.

"""

import os
import sys
import scipy.io as sio

def load(file_name):
    """
    Load the contents of a file into a dictionary.

    Supported formats, so far, are MAT files. More to come.

    Parameters
    ----------
    file_name : string
        Name of the file
    """
    if not os.path.isfile(file_name):
        sys.exit("[ERROR] The file {} does not exist.".format(file_name))
    file_ext = file_name.strip().split('.')[-1]
    if file_ext == 'mat':
        return sio.loadmat(file_name)
    return None

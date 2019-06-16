#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Input and Output routines

"""

import os
import sys
import scipy.io as sio
import numpy as np

def find_index(header, s):
    for h in header:
        if s in h.lower():
            return header.index(h)
    return None

def load(file_name):
    """
    Load the contents of a file into a dictionary.

    Supported formats, so far, are MAT and CSV files. More to come.

    To Do:
    - Get a better way to find data from keys of dictionary. PLEASE.

    Parameters
    ----------
    file_name : string
        Name of the file
    """
    if not os.path.isfile(file_name):
        sys.exit("[ERROR] The file {} does not exist.".format(file_name))
    file_ext = file_name.strip().split('.')[-1]
    if file_ext == 'mat':
        d = sio.loadmat(file_name)
        d.update({'rads':False})
        return Data(d)
    if file_ext == 'csv':
        with open(file_name, 'r') as f:
            all_lines = f.readlines()
        split_header = all_lines[0].strip().split(';')
        a_idx = find_index(split_header, 'acc')
        g_idx = find_index(split_header, 'gyr')
        m_idx = find_index(split_header, 'mag')
        q_idx = find_index(split_header, 'orient')
        data = np.genfromtxt(all_lines[2:], delimiter=';')
        d = {'time' : data[:, 0],
        'acc' : data[:, a_idx:a_idx+3],
        'gyr' : data[:, g_idx:g_idx+3],
        'mag' : data[:, m_idx:m_idx+3],
        'qts' : data[:, q_idx:q_idx+4]}
        d.update({'rads':True})
        return Data(d)
    return None

class Data:
    """
    Data to store the arrays of the most common variables.
    """
    def __init__(self, data_dict, **kwargs):
        # Create empty data attributes
        self.qts = None
        # Find possible data from keys of dictionary
        time_label = list(s for s in data_dict.keys() if 'time' in s.lower())
        acc_label = list(s for s in data_dict.keys() if 'acc' in s.lower())
        gyr_label = list(s for s in data_dict.keys() if 'gyr' in s.lower())
        mag_label = list(s for s in data_dict.keys() if 'mag' in s.lower())
        qts_label = list(s for s in data_dict.keys() if 'qts' in s.lower())
        rad_label = list(s for s in data_dict.keys() if 'rad' in s.lower())
        self.in_rads = data_dict.get(rad_label[0], False)
        # Load data into each attribute
        self.time = data_dict.get(time_label[0], None)
        self.acc = data_dict.get(acc_label[0], None)
        self.gyr = data_dict.get(gyr_label[0], None)
        self.mag = data_dict.get(mag_label[0], None)
        self.q_ref = data_dict.get(qts_label[0], None) if qts_label else None
        self.num_samples, self.num_axes = self.acc.shape

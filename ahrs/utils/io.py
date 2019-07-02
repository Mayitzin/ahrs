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

def load_ETH_EC(path):
    """
    Loads data from a directory containing files of the Event-Camera Dataset
    from the ETH Zurich (http://rpg.ifi.uzh.ch/davis_data.html)

    The dataset includes 4 basic text files with recorded data, plus a file
    listing all images of the recording included in the subfolder 'images.'

    **events.txt**: One event per line (timestamp x y polarity)
    **images.txt**: One image reference per line (timestamp filename)
    **imu.txt**: One measurement per line (timestamp ax ay az gx gy gz)
    **groundtruth.txt**: One ground truth measurements per line (timestamp px py pz qx qy qz qw)
    **calib.txt**: Camera parameters (fx fy cx cy k1 k2 p1 p2 k3)

    Parameters
    ----------
    path : str
        Path of the folder containing the TXT files.

    Returns
    -------
    data : Data
        class Data with the contents of the dataset.

    """
    if not os.path.isdir(path):
        print("Invalid path")
        return None
    data = {}
    files = []
    [files.append(f) for f in os.listdir(path) if f.endswith('.txt')]
    missing = list(set(files).symmetric_difference([
        'events.txt',
        'images.txt',
        'imu.txt',
        'groundtruth.txt',
        'calib.txt']))
    if missing:
        sys.exit("Incomplete data. Missing files:\n{}".format('\n'.join(missing)))
    imu_data = np.loadtxt(os.path.join(path, 'imu.txt'), delimiter=' ')
    data.update({"time_sensors": imu_data[:, 0]})
    data.update({"accs": imu_data[:, 1:4]})
    data.update({"gyros": imu_data[:, 4:7]})
    data.update({"rads": False})
    truth_data = np.loadtxt(os.path.join(path, 'groundtruth.txt'), delimiter=' ')
    data.update({"time_truth": truth_data[:, 0]})
    data.update({"qts": truth_data[:, 4:]})
    return Data(data)

def load_ETH_EuRoC(path):
    """
    Load data from the EuRoC MAV dataset of the ETH Zurich
    (https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

    Parameters
    ----------
    path : str
        Path to the folder containing the dataset.

    References
    ----------
    .. [ETH-EuRoC] M. Burri, J. Nikolic, P. Gohl, T. Schneider, J. Rehder, S.
        Omari, M. Achtelik and R. Siegwart, The EuRoC micro aerial vehicle
        datasets, International Journal of Robotic Research,
        DOI: 10.1177/0278364915620033, early 2016.
    """
    if not os.path.isdir(path):
        print("Invalid path")
        return None
    valid_folders = ["imu", "groundtruth", "vicon"]
    # Find data.csv files in each folder
    folders = os.listdir(path)
    subfolders = {}
    for f in valid_folders:
        for s in folders:
            if f in s:
                subfolders.update({f: s})
    # Build data dictionary
    data = {}
    files = []
    for sf in subfolders.keys():
        full_path = os.path.join(path, subfolders[sf])
        contents = os.listdir(full_path)
        if "data.csv" not in contents:
            print("ERROR: File data.csv was not found in {}".format(subfolders[sf]))
        if sf == "imu":
            file_path = os.path.join(full_path, "data.csv")
            with open(file_path, 'r') as f:
                all_lines = f.readlines()
            time_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=0)
            gyrs_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=(1, 2, 3))
            accs_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=(4, 5, 6))
            data.update({"time_imu": time_array})
            data.update({"gyr": gyrs_array})
            data.update({"acc": accs_array})
        if sf == "vicon":
            file_path = os.path.join(full_path, "data.csv")
            with open(file_path, 'r') as f:
                all_lines = f.readlines()
            time_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=0)
            pos_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=(1, 2, 3))
            qts_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=(4, 5, 6, 7))
            data.update({"time_vicon": time_array})
            data.update({"position": pos_array})
            data.update({"quaternion": qts_array})
        if sf == "groundtruth":
            file_path = os.path.join(full_path, "data.csv")
            with open(file_path, 'r') as f:
                all_lines = f.readlines()
            time_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=0)
            pos_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=(1, 2, 3))
            qts_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=(4, 5, 6, 7))
            vel_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=(8, 9, 10))
            ang_vel_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=(11, 12, 13))
            acc_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=(14, 15, 16))
            data.update({"truth_time": time_array})
            data.update({"truth_position": pos_array})
            data.update({"truth_quaternion": qts_array})
            data.update({"truth_vel": vel_array})
            data.update({"truth_ang_vel": ang_vel_array})
            data.update({"truth_acc": acc_array})
    return data


class Data:
    """
    Data to store the arrays of the most common variables.
    """
    def __init__(self, data_dict, **kwargs):
        # Create empty data attributes
        self.qts = None
        data_keys = list(data_dict.keys())
        # Find possible data from keys of dictionary
        time_labels = list(s for s in data_keys if 'time' in s.lower())
        acc_labels = list(s for s in data_keys if 'acc' in s.lower())
        gyr_labels = list(s for s in data_keys if 'gyr' in s.lower())
        mag_labels = list(s for s in data_keys if 'mag' in s.lower())
        qts_labels = list(s for s in data_keys if 'qts' in s.lower())
        rad_labels = list(s for s in data_keys if 'rad' in s.lower())
        self.in_rads = data_dict.get(rad_labels[0], False) if rad_labels else False
        # Load data into each attribute
        self.time = data_dict.get(time_labels[0], None) if time_labels else None
        if len(time_labels) > 1:
            self.time_ref = data_dict.get(time_labels[1], None) if time_labels else None
        self.acc = data_dict.get(acc_labels[0], None) if acc_labels else None
        self.gyr = data_dict.get(gyr_labels[0], None) if gyr_labels else None
        self.mag = data_dict.get(mag_labels[0], None) if mag_labels else None
        self.q_ref = data_dict.get(qts_labels[0], None) if qts_labels else None
        self.num_samples = self.acc.shape[0] if np.ndim(self.acc) > 0 else 0
        self.num_axes = self.acc.shape[1] if np.ndim(self.acc) > 1 else 0

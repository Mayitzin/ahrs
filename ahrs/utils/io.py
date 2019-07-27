#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""
Input and Output routines

"""

import os
import sys
import scipy.io as sio
import numpy as np

def get_freq(times, units='s'):
    """
    Identify and return the frequency a dataset is sampled.

    Given an array with timestamps, the step between times is estimated, then
    a mean of its values is inverted to obtain the sampling frequency.

    Parameters
    ----------
    times : array
        1-D array with the timestamps of the file.
    units : str
        Time units of the array of timestamps. Default is 's' for seconds.
        Possible options are: 's', 'ms', 'us' and 'ns'.

    Returns
    -------
    frequency : float
        Estimated sampling frequency in Herz.

    Examples
    --------
    >>> t = np.arange(500) + np.random.randn(500)
    >>> ahrs.utils.io.id_frequency(t)
    0.9984740941199178
    >>> t = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> ahrs.utils.io.id_frequency(t, 'ms')
    10000.0

    """
    diffs = np.diff(times)
    mean = np.nanmean(diffs)
    if units == 'ms':
        mean *= 1e-3
    if units == 'us':
        mean *= 1e-6
    if units == 'ns':
        mean *= 1e-9
    return 1.0 / mean

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
        d.update({'in_rads':True})
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
    data.update({"in_rads": False})
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
            data.update({"imu_time": time_array})
            data.update({"imu_gyr": gyrs_array})
            data.update({"imu_acc": accs_array})
        if sf == "vicon":
            file_path = os.path.join(full_path, "data.csv")
            with open(file_path, 'r') as f:
                all_lines = f.readlines()
            time_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=0)
            pos_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=(1, 2, 3))
            qts_array = np.genfromtxt(all_lines[1:], dtype=float, comments='#', delimiter=',', usecols=(4, 5, 6, 7))
            data.update({"vicon_time": time_array})
            data.update({"vicon_position": pos_array})
            data.update({"vicon_quaternion": qts_array})
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
            data.update({"time": time_array})
            data.update({"position": pos_array})
            data.update({"qts": qts_array})
            data.update({"vel": vel_array})
            data.update({"ang_vel": ang_vel_array})
            data.update({"acc": acc_array})
            data.update({'in_rads':True})
    return Data(data)

def load_OxIOD(path, sequence=1):
    """
    Load data from the Oxford Inertial Odometry Dataset
    (http://deepio.cs.ox.ac.uk/)

    The OxIOD has several sequences stored in CSV, composed of sensors and
    vicon recordings, with the names and formats:

    imu:
        [0] Time
        [1] attitude_roll(radians)
        [2] attitude_pitch(radians)
        [3] attitude_yaw(radians)
        [4] rotation_rate_x(radians/s)
        [5] rotation_rate_y(radians/s)
        [6] rotation_rate_z(radians/s)
        [7] gravity_x(G)
        [8] gravity_y(G)
        [9] gravity_z(G)
        [10] user_acc_x(G)
        [11] user_acc_y(G)
        [12] user_acc_z(G)
        [13] magnetic_field_x(microteslas)
        [14] magnetic_field_y(microteslas)
        [15] magnetic_field_z(microteslas)

    vicon:
        [0] Time
        [1] Header
        [2] translation.x
        [3] translation.y
        [4] translation.z
        [5] rotation.x
        [6] rotation.y
        [7] rotation.z
        [8] rotation.w

    Parameters
    ----------
    path : str
        Path to the folder containing the dataset.
    sequence : int
        Sequence to load. Default is 1.

    References
    ----------
    .. [OxIOD] Changhao Chen, Peijun Zhao, Chris Xiaoxuan Lu, Wei Wang, Andrew
        Markham, Niki Trigoni. OxIOD: The Dataset for Deep Inertial Odometry.
        arXiv:1809.07491. September 2018.
        (https://arxiv.org/pdf/1809.07491.pdf)
    """
    if not os.path.isdir(path):
        print("Invalid path")
        return None
    imu_file = 'imu{}.csv'.format(sequence)
    vicon_file = 'vi{}.csv'.format(sequence)
    all_files = os.listdir(path)
    # Assert exitence of required files
    if imu_file not in all_files:
        print("IMU Sequence does NOT exists.")
        return None
    if vicon_file not in all_files:
        print("Vicon Sequence does NOT exists.")
        return None
    # Read files
    base_path = os.path.relpath(path)
    imu_file = os.path.join(base_path, imu_file)
    vicon_file = os.path.join(base_path, vicon_file)
    # Read Sensor information
    data = {}
    imu_data = np.genfromtxt(imu_file, dtype=np.float, delimiter=',', filling_values=np.nan)
    data.update({"imu_time": imu_data[:, 0]})
    data.update({"ang_pos": imu_data[:, 1:4]})
    data.update({"gyr": imu_data[:, 4:7]})
    data.update({"acc": imu_data[:, 7:10]})
    data.update({"usr_acc": imu_data[:, 10:13]})
    data.update({"mag": imu_data[:, 13:]})
    data.update({'in_rads':True})
    vicon_data = np.genfromtxt(vicon_file, dtype=np.float, delimiter=',', filling_values=np.nan)
    data.update({"vicon_time": vicon_data[:, 0]})
    data.update({"pos": vicon_data[:, 2:5]})
    data.update({"q_ref": np.roll(vicon_data[:, 5:], 1, axis=1)}) # Roll data to fit standard quaternion notation
    return Data(data)

class Data:
    """
    Data to store the arrays of the most common variables.
    """
    time = None
    acc = None
    gyr = None
    mag = None
    q_ref = None
    def __init__(self, *initial_data, **kwargs):
        # def_attributes = ['time', 'acc', 'gyr', 'mag', 'q_ref', 'pos']
        # for a in def_attributes:
        #     setattr(self, a, None)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.num_samples = len(self.acc) if self.acc is not None else 0

# Support functions for TS file manipulation
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Aims: Provide support for quantum time series analysis
# Date: 2024

import sys
import numpy as np
import math
import os
import json
import csv
import pandas as pd
import pickle

sys.path.append('.')
sys.path.append('..')
sys.path

def if_path_exists(fpath):
    return os.path.exists(fpath)

def create_folder_if_needed(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath, exist_ok=True)

def read_pickle_file(fpath):
    if not os.path.exists(fpath):
        print(f'*** ERROR: The log file does not exist or is corrupted: {fpath}')
        return {}
    else:
        try:
            f = open(fpath, 'rb')
        except OSError:
            print(f'*** ERROR: Could not open/read the PICKLE file: {fpath}')
            return {}
        with f:
            data = pickle.load(f)
            f.close()
            return data

def write_pickle_file(fpath, data):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        f.close()

def read_json_file(fpath):
    if not os.path.exists(fpath):
        print(f'*** ERROR: The log file does not exist or is corrupted: {fpath}')
        return {}
    else:
        try:
            f = open(fpath, 'r')
        except OSError:
            print(f'*** ERROR: Could not open/read the JSON file: {fpath}')
            return {}
        with f:
            data = json.load(f)
            f.close()
            return data

def write_json_file(fpath, json_data):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(f'{fpath}', 'w') as f:
        json.dump(json_data, f)

def read_ts_file(fpath):
    if not os.path.exists(fpath):
        print(f'*** ERROR: The log file does not exist or is corrupted: {fpath}')
        sys.exit()
    else:
        try:
            f = open(fpath, 'r')
        except OSError:
            print(f'*** ERROR: Could not open/read the TS file: {fpath}')
            sys.exit()
        with f:
            data = np.loadtxt(f, dtype=float)
            return data

def write_ts_file(fpath, ts_data):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        np.savetxt(f, ts_data, fmt='%.8f')

def read_csv_file(fpath, delimiter=',', quotechar='"'):
    if not os.path.exists(fpath):
        print(f'*** ERROR: The CSV file does not exist or is corrupted: {fpath}')
        sys.exit()
    else:
        try:
            f = open(fpath, 'r', newline='')
        except OSError:
                print(f'*** ERROR: Could not open/read the CSV file: {fpath}')
                sys.exit()
        with f:
            df = pd.read_csv(fpath, delimiter=delimiter, quotechar=quotechar)
            header = list(df.columns)
            data = np.array(df)
            return header, data

def write_csv_file(fpath, csv_data):
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(f'{fpath}', 'w') as f:
        f.write(csv_data)


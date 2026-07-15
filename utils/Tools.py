# Support functions for QML and QNN demos
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Date: September 2024

import os
import pylab
import math
import numpy as np
import pennylane as qml

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import set_loglevel
set_loglevel("error")

from IPython.display import clear_output

####### Useful functions

### Returns the current date and time
import datetime

def get_timestamp_now():
    now = datetime.datetime.now()
    now_date = (now.year, now.month, now.day)
    now_time = (now.hour, now.minute, now.second)
    time_stamp = f'%04d-%02d-%02d %02d:%02d:%02d' % (now_date + now_time)
    return time_stamp


### Bit to list translation for data entry and state interpretation
#   Note: PennyLane interprets qubits state in reverse order than Qiskit

### Transform int number to a list of bits, bit 0 comes first
def bin_int_to_list(a, n_bits):
    a = int(a)
    a_list = [int(i) for i in f'{a:0{n_bits}b}']
    # a_list.reverse()
    return np.array(a_list)

### Transform a list of bits to an int number, bit 0 comes first
def bin_list_to_int(bin_list):
    b = list(bin_list)
    # b.reverse()
    return int("".join(map(str, b)), base=2)

####### PennyLane extensions

### Draw this circuit beautifully as in Qiskit
#   Lots of styles apply, e.g. 'black_white', 'black_white_dark', 'sketch', 
#     'pennylane', 'pennylane_sketch', 'sketch_dark', 'solarized_light', 'solarized_dark', 
#     'default', we can even use 'rcParams' to redefine all attributes
#   level = None, 'user', 'top', 'device', 'gradient', 0, 1, ...
def draw_circuit(circuit, fontsize=20, style='pennylane', 
                 scale=None, title=None, decimals=2, level=None):
    def _draw_circuit(*args, **kwargs):
        nonlocal circuit, fontsize, style, scale, title, level
        qml.drawer.use_style(style)
        fig, ax = qml.draw_mpl(circuit, decimals=decimals, level=level)(*args, **kwargs)
        if scale is not None:
            dpi = fig.get_dpi()
            fig.set_dpi(dpi*scale)
        if title is not None:
            fig.suptitle(title, fontsize=fontsize)
        plt.show()
    return _draw_circuit

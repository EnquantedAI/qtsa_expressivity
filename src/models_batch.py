# Support functions for model creation
# Author: Jacob Cybulski, ironfrown[at]gmail.com
# Date: 2026

import pennylane as qml
from pennylane import numpy as np
import torch
import matplotlib.pyplot as plt

### Draw this circuit beautifully as in Qiskit
#   Lots of styles apply, e.g. 'black_white', 'black_white_dark', 'sketch', 
#     'pennylane', 'pennylane_sketch', 'sketch_dark', 'solarized_light', 'solarized_dark', 
#     'default', we can even use 'rcParams' to redefine all attributes
#   level = None, 'user', 'top', 'device', 'gradient', 0, 1, ...
def draw_circuit(circuit, fontsize=20, style='pennylane', 
                 scale=None, title=None, decimals=2, level='user'):
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

### Creates a ZZ feature map (Qiskit style)
def zz_feature_map(x, wires, repeats=1):
    """
    Implements a second-order Havlíček ZZ feature map in PennyLane.
    Safely fetches feature dimension regardless of 1D or 2D structure.
    
    Parameters:
    -----------
    x : array-like
        The input classical data vector. Length must match the number of wires.
        Alternatively it consists of a batch of input data.
    wires : list or iterable
        The quantum wires/qubits to apply the feature map on.
    repeats : int
        Number of times to repeat the feature map layer (depth).
    """
    num_features = qml.math.shape(x)[-1]
    
    for _ in range(repeats):
        # 1. Uniform Superposition Layer
        for i in range(num_features):
            qml.Hadamard(wires=wires[i])
            
        # 2. First-Order Terms: Single-qubit Pauli-Z rotations
        for i in range(num_features):
            # x[..., i] slices out the entire column if 2D, or element if 1D
            qml.RZ(-2.0 * x[..., i], wires=wires[i])
            
        # 3. Second-Order Terms: Pairwise ZZ interactions
        for i in range(num_features):
            for j in range(i + 1, num_features):
                phi_ij = (np.pi - x[..., i]) * (np.pi - x[..., j])
                
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.RZ(-2 * phi_ij, wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])


### Returns the shape of the model weights
def gqnn_shape(n_layers, n_qubits):
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return shape

### Creates a model generator
def gqnn(n_layers, n_qubits, dev,
    interface='autograd', diff_method='adjoint', fm_style='zzfm', meas=[0]):
    """
    Creates a generalised QNN model.
    
    Parameters:
    -----------
    n_layers: number of layers
    n_qubits: number of qubits
    fm_style: string indicating the type of the feature map, possible values
        xxfm - for Havlicek ZZ feature map (Qiskit style)
        iqp - for PennyLane style IQP
        X | Y | Z - angle encoding with indicated rotation
    """
    
    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def circuit(inputs, weights):
        nonlocal n_layers, n_qubits, dev
        nonlocal interface, diff_method, fm_style

        wires = dev.wires
        
        # Safe extraction of the feature count (works for 1D or 2D inputs)
        num_features = qml.math.shape(inputs)[-1]
        
        # --- ZZ / IQP encoding ---
        if fm_style=='iqp':
            qml.IQPEmbedding(inputs, wires)
        elif fm_style=='zzfm':
            zz_feature_map(inputs, wires)
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=fm_style)
        qml.Barrier()
                
        # --- layers: U rotation + Rx reuploading + circular entanglement ---
        # Uses feature size instead of batch size, keeping your architecture rule intact!
        qml.StronglyEntanglingLayers(weights, wires=range(num_features))
        
        qml.Barrier()
        
        return [qml.expval(qml.PauliZ(m)) for m in meas]
        
    return circuit

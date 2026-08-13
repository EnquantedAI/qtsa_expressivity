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
def zz_feature_map(x, wires=None, repeats=1):
    """
    Implements a second-order Havlíček ZZ feature map in PennyLane.
    
    Parameters:
    -----------
    x : array-like
        The input classical data vector. Length must match the number of wires.
    wires : list or iterable
        The quantum wires/qubits to apply the feature map on.
    repeats : int
        Number of times to repeat the feature map layer (depth).
    """
    n_features = qml.math.shape(x)[-1]
    if wires is None: wires = range(n_features)
    
    for _ in range(repeats):
        # 1. Uniform Superposition Layer
        for i in range(n_features):
            qml.Hadamard(wires=wires[i])
            
        # 2. First-Order Terms: Single-qubit Pauli-Z rotations
        for i in range(n_features):
            # PennyLane Rz(theta) applies exp(-i * theta * Z / 2)
            # To get exp(i * x_i * Z), we pass theta = -2 * x_i
            qml.RZ(-2.0 * x[i], wires=wires[i])
            
        # 3. Second-Order Terms: Pairwise ZZ interactions
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Calculate the Havlíček coupling coefficient
                phi_ij = (np.pi - x[i]) * (np.pi - x[j])
                
                # Implement exp(i * phi_ij * Z_i * Z_j) using CNOT-Rz-CNOT
                qml.CNOT(wires=[wires[i], wires[j]])
                qml.RZ(-2 * phi_ij, wires=wires[j])
                qml.CNOT(wires=[wires[i], wires[j]])

### Returns the shape of the model weights
def gqnn_shape(n_layers, n_qubits):
    shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    return shape

### Creates a model generator
def gqnn_v1(n_layers, n_qubits, dev,
    interface='autograd', diff_method='adjoint', fm_style='zzfm', 
    reupload=False, meas=[0]):
    """
    Creates a generalised QNN model.
    Note that the number of qubits may be greater than
      the number of features, in which case we can create
      ancilla qubits to add extra parameters and entanglements
    
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
        nonlocal reupload

        wires = dev.wires
        n_features = qml.math.shape(inputs)[-1]
        
        # --- ZZ / IQP encoding (entangles + phase-encodes the window)
        if fm_style=='iqp':
            qml.IQPEmbedding(inputs, wires=range(n_features))
        elif fm_style=='zzfm':
            zz_feature_map(inputs, wires=range(n_features))
        else:
            qml.AngleEmbedding(inputs, wires=range(n_features), rotation=fm_style)
        qml.Barrier()
                
        # --- layers: U rotation + Rx reuploading + circular entanglement ---
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        
        qml.Barrier()
        
        return [qml.expval(qml.PauliZ(m)) for m in meas]
        
    return circuit

### Creates a model generator
def gqnn(n_layers, n_qubits, dev,
    interface='autograd', diff_method='adjoint', 
    fm_style='zzfm', reup_style=None, meas=[0]):
    """
    Creates a generalised QNN model.
    Note that the number of qubits may be greater than
      the number of features, in which case we can create
      ancilla qubits to add extra parameters and entanglements
    
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
        nonlocal reup_style

        wires = dev.wires
        n_features = qml.math.shape(inputs)[-1]
        if n_features > n_qubits:
            n_features = n_qubits
            inputs = inputs[-n_qubits:]
        
        # --- ZZ / IQP encoding (entangles + phase-encodes the window)
        if fm_style=='iqp':
            qml.IQPEmbedding(inputs, wires=range(n_features))
        elif fm_style=='zzfm':
            zz_feature_map(inputs, wires=range(n_features))
        else:
            for i in range(n_features):
                if fm_style == 'X':
                    qml.RX(inputs[i], wires=wires[i])
                elif fm_style == 'Y':
                    qml.RY(inputs[i], wires=wires[i])
                elif fm_style == 'Z':
                    qml.RZ(inputs[i], wires=wires[i])
                
        # --- layers: U rotation + Rx reuploading + circular entanglement
        for l in range(n_layers):
            qml.Barrier()

            # Trainable weight block
            for i in range(n_qubits):
                qml.RZ(weights[l][i][0], wires=wires[i])
                qml.RY(weights[l][i][1], wires=wires[i])
                qml.RZ(weights[l][i][2], wires=wires[i])

            # Entangling block
            if n_qubits > 1:
                qml.Barrier()
                r = (l % (n_qubits - 1)) + 1
                if n_qubits > 1:
                    for i in range(n_qubits):
                        control = wires[i]
                        target = wires[(i + r) % n_qubits]
                        qml.CNOT(wires=[control, target])

            # Reuploading block
            if reup_style is not None:
                qml.Barrier()
                for i in range(n_features):
                    if reup_style == 'X':
                        qml.RX(inputs[i], wires=wires[i])
                    elif reup_style == 'Y':
                        qml.RY(inputs[i], wires=wires[i])
                    elif reup_style == 'Z':
                        qml.RZ(inputs[i], wires=wires[i])
        
        qml.Barrier()
        
        return [qml.expval(qml.PauliZ(m)) for m in meas]
        
    return circuit


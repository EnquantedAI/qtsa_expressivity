import os
import sys

project_root = os.path.abspath("../../..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Project root: {project_root}")

from src.models import draw_circuit, gqnn_shape, zz_feature_map
import torch
import pennylane as qml

def gqnn(n_layers, n_qubits, dev, interface='torch', diff_method='torch', fm_style='zzfm', meas=[0]):
    
    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def circuit(inputs, weights):
        nonlocal n_layers, n_qubits, dev
        nonlocal interface, diff_method, fm_style

        wires = dev.wires

        if fm_style == "iqp":
            qml.IQPEmbedding(inputs, wires)
        elif fm_style == "zzfm":
            zz_feature_map(inputs, wires)
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=fm_style)
        
        qml.Barrier()

        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

        qml.Barrier()

        return [qml.expval(qml.PauliZ(m)) for m in meas]

    weight_shapes = {"weights": gqnn_shape(n_layers, n_qubits)}
    qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
    
    return qlayer
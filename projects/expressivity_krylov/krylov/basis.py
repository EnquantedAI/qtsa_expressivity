import pennylane as qml
from pennylane import numpy as np

def get_subnetwork_unitary(n_layers_k, n_qubits, weights, inputs, fm_style="zzfm"):
    """
    Computes the full unitary matrix U_k of the QNN up to layer k.
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    
    # Extract weights only up to layer k
    weights_k = weights[:n_layers_k]
    
    @qml.qnode(dev)
    def circuit():
        # Feature map encoding
        if fm_style == 'iqp':
            qml.IQPEmbedding(inputs, wires=range(n_qubits))
        elif fm_style == 'zzfm':
            # Handle Havlíček ZZ feature map
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(-2.0 * inputs[i], wires=i)
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    phi = (np.pi - inputs[i]) * (np.pi - inputs[j])
                    qml.CNOT(wires=[i, j])
                    qml.RZ(-2 * phi, wires=j)
                    qml.CNOT(wires=[i, j])
        else:
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=fm_style)

        # Strongly entangling layers up to layer k
        qml.StronglyEntanglingLayers(weights_k, wires=range(n_qubits))
        return qml.state()

    # Get matrix representation of the circuit unitary U_k
    matrix_fn = qml.matrix(circuit)
    return matrix_fn()

def hs_inner_product(A, B):
    """Normalized Hilbert-Schmidt inner product: Tr(A^\dagger B) / 2^N."""
    dim = A.shape[0]
    return np.real(np.trace(A.conj().T @ B)) / dim

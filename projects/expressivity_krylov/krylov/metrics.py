import numpy as np
from .lanczos import run_operator_lanczos
from .basis import get_subnetwork_unitary

def compute_krylov_expressivity(n_layers, n_qubits, weights, inputs, O_0, fm_style="zzfm"):
    """
    Computes Krylov metrics for sub-networks of layer depth 1 <= k <= N.
    """
    expressivity_results = {
        "k_layers": [],
        "krylov_dims": [],
        "b_decay_rates": []
    }
    
    for k in range(1, n_layers + 1):
        # 1. Get Unitary for first k layers
        U_k = get_subnetwork_unitary(k, n_qubits, weights, inputs, fm_style=fm_style)
        
        # 2. Run Lanczos
        res = run_operator_lanczos(O_0, U_k, max_steps=2**n_qubits)
        
        # 3. Compute Expressivity Metric (Krylov Dimension / Capacity)
        k_dim = res["krylov_dim"]
        
        expressivity_results["k_layers"].append(k)
        expressivity_results["krylov_dims"].append(k_dim)
        
    return expressivity_results

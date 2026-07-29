import numpy as np
from .basis import hs_inner_product

def run_operator_lanczos(O_0, U_k, max_steps=50, tol=1e-6):
    """
    Applies the operator Lanczos algorithm to generate Krylov basis and coefficients.
    
    Evolution Operator L(O) = U_k^\dagger O U_k (Heisenberg picture step)
    """
    # Normalize initial observable
    norm_0 = np.sqrt(hs_inner_product(O_0, O_0))
    V = [O_0 / norm_0]
    
    a_coeffs = []
    b_coeffs = []
    
    for n in range(max_steps):
        # Apply Liouvillian/Unitary map: L(V_n) = U_k^\dagger V_n U_k
        V_next = U_k.conj().T @ V[n] @ U_k
        
        # Diagonal term a_n = <V_n, L(V_n)>
        a_n = hs_inner_product(V[n], V_next)
        a_coeffs.append(a_n)
        
        # Orthogonalize against current and previous vector
        V_next = V_next - a_n * V[n]
        if n > 0:
            V_next = V_next - b_coeffs[n-1] * V[n-1]
            
        # Compute norm b_{n+1}
        b_n = np.sqrt(max(0.0, hs_inner_product(V_next, V_next)))
        
        if b_n < tol:
            break
            
        b_coeffs.append(b_n)
        V.append(V_next / b_n)
        
    return {
        "b_coefficients": np.array(b_coeffs),
        "a_coefficients": np.array(a_coeffs),
        "krylov_dim": len(V),
        "basis": V
    }

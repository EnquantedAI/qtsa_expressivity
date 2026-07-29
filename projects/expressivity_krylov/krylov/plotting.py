import matplotlib.pyplot as plt

def plot_krylov_expressivity(results, title="Krylov Expressivity vs. Layer Depth"):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(results["k_layers"], results["krylov_dims"], marker='o', linewidth=2, color='indigo')
    ax.set_xlabel("Layer Depth ($k$)", fontsize=12)
    ax.set_ylabel("Krylov Dimension / Expressivity", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

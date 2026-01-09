import numpy as np
import matplotlib.pyplot as plt
from core.qld_2x2 import qld_log_odds

# Scaled matrix used in the brief
A = np.array([[2, 4],
              [0, 6]])

def plot_phase_portrait(T):
    U, V = np.meshgrid(
        np.linspace(-4, 4, 40),
        np.linspace(-4, 4, 40)
    )

    DU = np.zeros_like(U)
    DV = np.zeros_like(V)

    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            DU[i, j], DV[i, j] = qld_log_odds(U[i, j], V[i, j], T, A)

    plt.figure(figsize=(6, 6))
    plt.streamplot(U, V, DU, DV, density=1.2)
    plt.scatter([0], [0], color='red', s=60, zorder=5, label='Symmetric QRE')
    plt.legend()
    plt.title(f"2x2 QLD Phase Portrait (T={T})")
    plt.xlabel("u = log(x1/x2)")
    plt.ylabel("v = log(y1/y2)")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    plot_phase_portrait(T=1.5)  # Above critical
    plot_phase_portrait(T=0.5)  # Below critical

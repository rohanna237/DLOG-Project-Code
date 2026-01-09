import os
import numpy as np
import matplotlib.pyplot as plt
from core.qld_2x2 import qld_log_odds

A = np.array([[2.0, 4.0],
              [0.0, 6.0]])

def export_phase_portrait(T, Umax=4.0, grid_n=45):
    os.makedirs("figures", exist_ok=True)

    U, V = np.meshgrid(
        np.linspace(-Umax, Umax, grid_n),
        np.linspace(-Umax, Umax, grid_n)
    )

    DU = np.zeros_like(U)
    DV = np.zeros_like(V)

    for i in range(grid_n):
        for j in range(grid_n):
            du, dv = qld_log_odds(U[i, j], V[i, j], T, A)
            DU[i, j] = du
            DV[i, j] = dv

    plt.figure(figsize=(6, 6))
    plt.streamplot(U, V, DU, DV, density=1.2)
    plt.scatter([0], [0], s=60, label="Symmetric QRE")  # (u,v)=(0,0)
    plt.title(f"2×2 QLD phase portrait (T={T})")
    plt.xlabel(r"$u=\log(x_1/x_2)$")
    plt.ylabel(r"$v=\log(y_1/y_2)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    outpath = os.path.join("figures", f"phase_portrait_T_{T}.png")
    plt.savefig(outpath, dpi=300)
    plt.show()
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    T = 1.0
    export_phase_portrait(T)

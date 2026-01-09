import os
import numpy as np
import matplotlib.pyplot as plt
from core.qld_2x2 import qld_log_odds

A = np.array([[2.0, 4.0],
              [0.0, 6.0]])

def integrate(u0, v0, T, eta=0.15, t_max=40.0, dt=0.01):
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)
    u = np.zeros(n_steps)
    v = np.zeros(n_steps)
    u[0], v[0] = u0, v0

    for k in range(n_steps - 1):
        du, dv = qld_log_odds(u[k], v[k], T, A)
        u[k+1] = u[k] + eta * dt * du
        v[k+1] = v[k] + eta * dt * dv

    return t, u, v

def export_trajectories(T, eta=0.15):
    os.makedirs("figures", exist_ok=True)

    initials = [(-2.5, -2.5), (-1.5, 0.5), (0.5, -1.5), (2.0, 2.0)]

    plt.figure(figsize=(7, 4))
    for (u0, v0) in initials:
        t, u, v = integrate(u0, v0, T, eta=eta)
        x = 1 / (1 + np.exp(-u))
        plt.plot(t, x, lw=1.6)

    plt.xlabel("time")
    plt.ylabel("probability of tight spread")
    plt.title(f"2×2 QLD trajectories (T={T}, η={eta})")
    plt.grid(True)
    plt.tight_layout()

    outpath = os.path.join("figures", f"trajectories_T_{T}.png")
    plt.savefig(outpath, dpi=300)
    plt.show()
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    T = 1
    export_trajectories(T)

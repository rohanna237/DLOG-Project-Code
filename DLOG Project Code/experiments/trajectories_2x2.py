import numpy as np
import matplotlib.pyplot as plt
from core.qld_2x2 import qld_log_odds

# ============================================================
# Payoff matrix (scaled, asymmetric to avoid trivial collapse)
# ============================================================
A = np.array([
    [2.0, 4.0],
    [0.0, 6.0]
])

# ============================================================
# Integrate QLD in log-odds space
# ============================================================
def integrate_qld(u0, v0, T, eta=0.15, t_max=40.0, dt=0.01):
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)

    u = np.zeros(n_steps)
    v = np.zeros(n_steps)

    u[0], v[0] = u0, v0

    for k in range(n_steps - 1):
        du, dv = qld_log_odds(u[k], v[k], T, A)

        # slowed dynamics via explicit learning rate
        u[k + 1] = u[k] + eta * dt * du
        v[k + 1] = v[k] + eta * dt * dv

    return t, u, v

# ============================================================
# Plot trajectories in probability space
# ============================================================
def plot_trajectories(T, initials, eta=0.15):
    plt.figure(figsize=(7, 4))

    for (u0, v0) in initials:
        t, u, v = integrate_qld(u0, v0, T, eta=eta)

        # map log-odds → probability
        x = 1.0 / (1.0 + np.exp(-u))
        plt.plot(t, x, lw=1.6)

    # Logit QRE reference (only when meaningful)
    if T > 1.0:
        x_star = 0.21
        plt.axhline(
            y=x_star,
            linestyle="--",
            color="black",
            linewidth=1.5,
            label="Logit QRE"
        )
        plt.legend()

    plt.xlabel("time")
    plt.ylabel("probability of tight spread")
    plt.title(f"2×2 QLD trajectories (T={T}, η={eta})")
    plt.grid()
    plt.show()

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    # Diverse initial conditions (log-odds space)
    initials = [
        (-2.5, -2.5),
        (-1.5,  0.5),
        ( 0.5, -1.5),
        ( 2.0,  2.0)
    ]

    # Above critical temperature → interior QRE
    plot_trajectories(
        T=1.5,
        eta=0.15,
        initials=initials
    )

    # Below critical temperature → boundary attraction
    plot_trajectories(
        T=0.5,
        eta=0.15,
        initials=initials
    )

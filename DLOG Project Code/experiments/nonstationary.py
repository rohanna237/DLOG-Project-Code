import numpy as np
import matplotlib.pyplot as plt

from core.qld_m_actions import payoff_matrix, qld_rhs

# ============================================================
# Spread menu
# ============================================================

spreads = np.array([0.2, 0.3, 0.4])
n_strats = len(spreads)

# ============================================================
# Toxicity regime
# ============================================================

p_L = 0.10
p_H = 0.35
tau = 20.0   # now meaningful

# ============================================================
# Learning parameters
# ============================================================

T_vals = [1.5, 3.0]
dt = 0.01
t_max = 60.0

# ------------------------------------------------------------
# CRUCIAL: learning-rate scaling
# ------------------------------------------------------------

alpha = 0.1     # slows adaptation (THIS is the fix)
eps = 0.01      # mutation / exploration

# ============================================================
# Toxicity schedule
# ============================================================

def p_I(t):
    return p_L if t < tau else p_H


# ============================================================
# Simulation
# ============================================================

def simulate(T):
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)

    x = np.ones(n_strats) / n_strats
    y = x.copy()

    x_hist = np.zeros((n_steps, n_strats))
    p_hist = np.zeros(n_steps)

    for k in range(n_steps):
        p = p_I(t[k])
        A = payoff_matrix(spreads, p)

        dx, dy = qld_rhs(x, y, A, T)

        # ---- slowed QLD update ----
        x += alpha * dt * dx
        y += alpha * dt * dy

        # ---- mutation ----
        x = (1 - eps) * x + eps / n_strats
        y = (1 - eps) * y + eps / n_strats

        x = np.clip(x, 1e-10, 1.0)
        y = np.clip(y, 1e-10, 1.0)
        x /= np.sum(x)
        y /= np.sum(y)

        x_hist[k] = x
        p_hist[k] = p

    return t, x_hist, p_hist


# ============================================================
# Run experiment
# ============================================================

if __name__ == "__main__":

    plt.figure(figsize=(8, 5))

    for T in T_vals:
        t, x_hist, _ = simulate(T)

        for i, s in enumerate(spreads):
            plt.plot(
                t,
                x_hist[:, i],
                label=f"s={s}, T={T}"
            )

    plt.axvline(
        tau,
        color="k",
        linestyle="--",
        label="toxicity switch"
    )

    plt.xlabel("time")
    plt.ylabel("strategy probability")
    plt.title("Adaptation under nonstationary toxicity")
    plt.legend()
    plt.grid()
    plt.show()

    # --------------------------------------------------------

    plt.figure(figsize=(8, 3))
    t, _, p_hist = simulate(T_vals[0])
    plt.plot(t, p_hist)
    plt.axvline(tau, color="k", linestyle="--")
    plt.xlabel("time")
    plt.ylabel("p_I(t)")
    plt.title("Toxicity regime shift")
    plt.grid()
    plt.show()

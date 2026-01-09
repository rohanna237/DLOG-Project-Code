import os
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12


# ============================================================
# Two-player spread competition payoff (tightest wins, ties split)
# ============================================================

def build_payoff_matrix(spreads, p_I):
    """
    A_ij = E_i if i<j, E_i/2 if i=j, 0 if i>j, where E_i = s_i - p_I.
    """
    m = len(spreads)
    E = spreads - p_I
    A = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i < j:
                A[i, j] = E[i]
            elif i == j:
                A[i, j] = 0.5 * E[i]
            else:
                A[i, j] = 0.0
    return A, A.T


def qld_rhs(x, y, A, B, T):
    """
    QLD on simplex for two players with m actions.
    Uses the standard entropy regularisation term.
    """
    x = np.clip(x, EPS, 1.0)
    y = np.clip(y, EPS, 1.0)
    x /= x.sum()
    y /= y.sum()

    Ay = A @ y
    xAy = float(x @ Ay)

    Bx = B @ x
    yBx = float(y @ Bx)

    ex = np.log(x) - float(x @ np.log(x))
    ey = np.log(y) - float(y @ np.log(y))

    # sign convention: use + T * sum_j x_j log(x_j/x_i)
    # which is equivalent to -T*(log x_i - <log x>)
    xdot = x * ((Ay - xAy) - T * ex)
    ydot = y * ((Bx - yBx) - T * ey)

    # numerical mass correction
    xdot -= x * float(xdot.sum())
    ydot -= y * float(ydot.sum())

    return xdot, ydot


# ============================================================
# Stationary computation by time integration
# ============================================================

def stationary_average_spread(spreads, p_I, T, t_max=80.0, dt=0.01, n_inits=6, seed=0):
    """
    For fixed menu size m and toxicity p_I, run QLD from multiple initial conditions
    and return the mean final average spread x^T s (row player).
    """
    rng = np.random.default_rng(seed)
    m = len(spreads)
    A, B = build_payoff_matrix(spreads, p_I)

    steps = int(t_max / dt)
    avg_spreads = []

    for _ in range(n_inits):
        x = rng.dirichlet(np.ones(m))
        y = rng.dirichlet(np.ones(m))

        for _ in range(steps):
            xdot, ydot = qld_rhs(x, y, A, B, T)
            x += dt * xdot
            y += dt * ydot

            x = np.clip(x, EPS, None)
            y = np.clip(y, EPS, None)
            x /= x.sum()
            y /= y.sum()

        avg_spreads.append(float(x @ spreads))

    return float(np.mean(avg_spreads))


# ============================================================
# Main robustness experiment
# ============================================================

if __name__ == "__main__":

    # Make output folder
    os.makedirs("figures", exist_ok=True)

    # Fixed temperature for robustness check (choose moderate T)
    T = 0.10

    # Toxicity sweep
    p_grid = np.linspace(0.05, 0.45, 41)

    # Common spread range, evenly spaced menus
    s_min = 0.20
    s_max = 0.40
    menu_sizes = [2, 3, 5]

    curves = {}

    for m in menu_sizes:
        spreads = np.linspace(s_min, s_max, m)
        vals = []
        for p_I in p_grid:
            vals.append(stationary_average_spread(spreads, p_I, T))
        curves[m] = np.array(vals)

    # Plot
    plt.figure(figsize=(7, 4))
    for m in menu_sizes:
        plt.plot(p_grid, curves[m], label=f"m = {m}")

    plt.xlabel(r"toxicity $p_I$")
    plt.ylabel(r"stationary average quoted spread $\bar{s}(p_I)$")
    plt.title(rf"Robustness under menu size (fixed $T={T}$)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    outpath = "figures/m_menu_robustness.png"
    plt.savefig(outpath, dpi=300)
    plt.show()

    print("Saved:", outpath)

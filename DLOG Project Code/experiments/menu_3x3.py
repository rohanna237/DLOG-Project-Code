import os
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12

# ============================================================
# Two-player spread competition payoff (tightest wins, ties split)
# ============================================================

def build_payoff_matrix(spreads, p_I):
    """
    A_ij = E_i if i<j, E_i/2 if i=j, 0 if i>j
    where E_i = s_i - p_I
    spreads assumed strictly increasing.
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
    QLD on simplex:
      xdot_i = x_i[(Ay)_i - x^T Ay + T * sum_j x_j log(x_j/x_i)]
    This is equivalent to xdot = x * ((Ay - xAy) - T*(log x - <log x>))
    similarly for y.
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

    xdot = x * ((Ay - xAy) - T * ex)
    ydot = y * ((Bx - yBx) - T * ey)

    # enforce sum zero numerically
    xdot -= x * float(xdot.sum())
    ydot -= y * float(ydot.sum())

    return xdot, ydot


def integrate_qld(spreads, p_I, T, t_max=80.0, dt=0.01, n_inits=10, seed=0):
    """
    Run QLD for a fixed (spreads, p_I, T), average over multiple interior initial conditions.
    Returns:
      avg_spread = mean over inits of (x_final · spreads)
      diversity  = mean over inits of (1 - max_i x_final_i)
    """
    rng = np.random.default_rng(seed)
    m = len(spreads)
    A, B = build_payoff_matrix(spreads, p_I)

    steps = int(t_max / dt)
    avg_spreads = []
    diversities = []

    for _ in range(n_inits):
        x = rng.dirichlet(np.ones(m))
        y = rng.dirichlet(np.ones(m))

        for _ in range(steps):
            xdot, ydot = qld_rhs(x, y, A, B, T)
            x = x + dt * xdot
            y = y + dt * ydot

            x = np.clip(x, EPS, None)
            y = np.clip(y, EPS, None)
            x /= x.sum()
            y /= y.sum()

        avg_spreads.append(float(x @ spreads))
        diversities.append(1.0 - float(np.max(x)))

    return float(np.mean(avg_spreads)), float(np.mean(diversities))


# ============================================================
# Main sweep: p_I ↦ (avg spread, diversity)
# ============================================================

if __name__ == "__main__":

    os.makedirs("figures", exist_ok=True)

    # 3×3 menu used in your report
    spreads = np.array([0.20, 0.30, 0.40])

    # sweep toxicity
    p_grid = np.linspace(0.05, 0.45, 41)

    # temperature (choose moderate so diversity is visible)
    T = 0.10

    avg_spread_vals = []
    diversity_vals = []

    for p_I in p_grid:
        sbar, div = integrate_qld(
            spreads=spreads,
            p_I=p_I,
            T=T,
            t_max=80.0,
            dt=0.01,
            n_inits=10,
            seed=123
        )
        avg_spread_vals.append(sbar)
        diversity_vals.append(div)

    avg_spread_vals = np.array(avg_spread_vals)
    diversity_vals = np.array(diversity_vals)

    # ========================================================
    # Plot 1: average spread vs toxicity
    # ========================================================
    plt.figure(figsize=(7, 4))
    plt.plot(p_grid, avg_spread_vals, lw=2)
    plt.xlabel(r"toxicity $p_I$")
    plt.ylabel(r"mean final average spread $\bar{s}(p_I)$")
    plt.title(rf"$3\times 3$ QLD: average spread vs toxicity (T={T})")
    plt.grid(True)
    plt.tight_layout()
    out1 = "figures/menu3x3_avg_spread_vs_pI.png"
    plt.savefig(out1, dpi=300)
    plt.show()
    print("Saved:", out1)

    # ========================================================
    # Plot 2: diversity vs toxicity (mass outside best action)
    # ========================================================
    plt.figure(figsize=(7, 4))
    plt.plot(p_grid, diversity_vals, lw=2, color="darkred")
    plt.xlabel(r"toxicity $p_I$")
    plt.ylabel(r"diversity $1-\max_i x_i$")
    plt.title(rf"$3\times 3$ QLD: strategic diversity vs toxicity (T={T})")
    plt.grid(True)
    plt.tight_layout()
    out2 = "figures/menu3x3_diversity_vs_pI.png"
    plt.savefig(out2, dpi=300)
    plt.show()
    print("Saved:", out2)

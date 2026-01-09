import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# ============================================================
# Core primitives
# ============================================================

def expected_share(N, p):
    k = np.arange(N)
    return np.sum(binom.pmf(k, N-1, p) / (1 + k))


def payoff(x, spreads, p_I, N):
    U = np.zeros(len(spreads))
    tail = np.array([x[i:].sum() for i in range(len(spreads))])

    for i in range(len(spreads)):
        if tail[i] > 0:
            p_win = x[i] / tail[i]
            U[i] = (spreads[i] - p_I) * expected_share(N, p_win)

    return U


def qld_rhs(x, spreads, p_I, N, T):
    U = payoff(x, spreads, p_I, N)
    avg = np.dot(x, U)
    return x * (U - avg + T * (np.log(x + 1e-12) - np.dot(x, np.log(x + 1e-12))))


# ============================================================
# Competition experiment: selection strength
# ============================================================

def initial_selection_speed(spreads, p_I, N, T=0.15):
    """
    Instantaneous rate of decrease of expected spread.
    """
    x0 = np.ones(len(spreads)) / len(spreads)
    dx = qld_rhs(x0, spreads, p_I, N, T)
    return np.dot(dx, spreads)


# ============================================================
# Run experiment
# ============================================================

if __name__ == "__main__":

    spreads = np.array([0.10, 0.15, 0.20, 0.25])
    p_I = 0.15

    N_values = [2, 3, 5, 10, 20, 40]
    selection_rates = []

    for N in N_values:
        rate = initial_selection_speed(spreads, p_I, N)
        selection_rates.append(rate)

    # ========================================================
    # Plot
    # ========================================================

    plt.figure(figsize=(6,4))
    plt.plot(N_values, selection_rates, marker='o')
    plt.xlabel("number of market makers N")
    plt.ylabel("initial rate of spread reduction")
    plt.title("Competition strengthens price selection")
    plt.grid(True)
    plt.show()

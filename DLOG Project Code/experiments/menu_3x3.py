# experiments/menu_3x3.py

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Economic parameters
# ============================================================

p_I = 0.15          # toxicity
N = 10              # number of market makers
T = 0.05            # exploration strength (entropy pressure)

# ============================================================
# Menus (this is the whole point of the experiment)
# ============================================================

menus = {
    "coarse": np.array([0.20, 0.40]),
    "medium": np.array([0.15, 0.25, 0.35]),
    "fine": np.array([0.12, 0.18, 0.24, 0.30])
}

# ============================================================
# Simple, stable equilibrium proxy
# ============================================================

def equilibrium_distribution(spreads, p_I, N, T):
    """
    Softmax-style equilibrium proxy:
    - lower spreads are more competitive
    - entropy parameter T prevents collapse
    """
    payoffs = spreads - p_I
    logits = -N * payoffs / max(T, 1e-6)

    logits -= np.max(logits)  # numerical safety
    x = np.exp(logits)
    x /= np.sum(x)

    return x


def stationary_observables(spreads):
    x = equilibrium_distribution(spreads, p_I, N, T)

    avg_spread = np.dot(x, spreads)
    entropy = -np.sum(x * np.log(x + 1e-12))

    return avg_spread, entropy


# ============================================================
# Run experiment
# ============================================================

labels = []
avg_spreads = []
entropies = []

for name, spreads in menus.items():
    s, h = stationary_observables(spreads)
    labels.append(name)
    avg_spreads.append(s)
    entropies.append(h)

# ============================================================
# Plots
# ============================================================

plt.figure(figsize=(6, 4))
plt.bar(labels, avg_spreads)
plt.ylabel("stationary average spread")
plt.title("Menu richness effect on spreads")
plt.grid(True, axis="y")
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(labels, entropies, color="darkred")
plt.ylabel("strategy entropy")
plt.title("Menu richness effect on strategic diversity")
plt.grid(True, axis="y")
plt.show()

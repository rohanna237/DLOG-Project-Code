import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================

def softmax(q, T):
    z = q / max(T, 1e-9)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p))


# ============================================================
# Plot 1: single-agent baseline (stationary bandit)
# ============================================================

def plot_stationary_bandit_means(spreads, p_I, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)

    mu = spreads - p_I

    plt.figure(figsize=(7, 4))
    plt.plot(spreads, mu, marker="o")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("half-spread s")
    plt.ylabel("expected reward μ(s) = s - p_I")
    plt.title(f"Single-maker bandit: arm means (p_I={p_I})")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(outdir, "bandit_means.png")
    plt.savefig(path, dpi=300)
    plt.show()
    print("Saved:", path)


# ============================================================
# Plot 2: nonstationary bandit learning with softmax Q-updates
# ============================================================

def run_nonstationary_bandit(spreads, T, alpha, t_max, dt, p_L, p_H, tau, seed=0):
    """
    Single-maker bandit: always executed, reward mean = s - p_I(t)
    Q update: Q[a] <- Q[a] + alpha * (r - Q[a])
    Action selection: softmax(Q/T)
    """
    rng = np.random.default_rng(seed)
    n = len(spreads)
    steps = int(t_max / dt)
    t = np.linspace(0, t_max, steps)

    # time-varying toxicity
    p_path = np.where(t < tau, p_L, p_H)

    Q = np.zeros(n)
    probs_hist = np.zeros((steps, n))
    entropy_hist = np.zeros(steps)
    pnl_hist = np.zeros(steps)

    cum_pnl = 0.0

    for k in range(steps):
        p_I = p_path[k]
        probs = softmax(Q, T)
        probs_hist[k] = probs
        entropy_hist[k] = entropy(probs)

        # sample action
        a = rng.choice(n, p=probs)

        # realised reward: mean (s - p_I) + noise
        mean_r = spreads[a] - p_I
        r = mean_r + rng.normal(0.0, 0.05)  # noise level adjustable

        # Q update
        Q[a] += alpha * (r - Q[a])

        cum_pnl += r
        pnl_hist[k] = cum_pnl

    return t, p_path, probs_hist, entropy_hist, pnl_hist


def plot_nonstationary_bandit(spreads, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)

    # Experiment settings
    t_max = 100.0
    dt = 0.05
    p_L = 0.10
    p_H = 0.35
    tau = 40.0
    alpha = 0.15

    # Compare two temperatures
    configs = [
        {"T": 0.05, "label": "T=0.05 (low exploration)", "seed": 1},
        {"T": 0.20, "label": "T=0.20 (more exploration)", "seed": 2},
    ]

    # --- Toxicity path plot ---
    t = np.linspace(0, t_max, int(t_max / dt))
    p_path = np.where(t < tau, p_L, p_H)
    plt.figure(figsize=(7, 3))
    plt.plot(t, p_path, linewidth=2)
    plt.axvline(tau, linestyle="--", color="black", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("p_I(t)")
    plt.title("Nonstationary environment: toxicity regime shift")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(outdir, "bandit_toxicity_path.png")
    plt.savefig(path, dpi=300)
    plt.show()
    print("Saved:", path)

    # --- Probability on tightest arm over time (or pick an arm index) ---
    plt.figure(figsize=(7, 4))
    for cfg in configs:
        t, p_path, probs_hist, H, pnl = run_nonstationary_bandit(
            spreads, cfg["T"], alpha, t_max, dt, p_L, p_H, tau, seed=cfg["seed"]
        )
        plt.plot(t, probs_hist[:, 0], label=cfg["label"])  # arm 0 = smallest spread
    plt.axvline(tau, linestyle="--", color="black", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("P(select smallest spread)")
    plt.title("Bandit adaptation: policy probability over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "bandit_policy_prob.png")
    plt.savefig(path, dpi=300)
    plt.show()
    print("Saved:", path)

    # --- Entropy over time ---
    plt.figure(figsize=(7, 4))
    for cfg in configs:
        t, p_path, probs_hist, H, pnl = run_nonstationary_bandit(
            spreads, cfg["T"], alpha, t_max, dt, p_L, p_H, tau, seed=cfg["seed"]
        )
        plt.plot(t, H, label=cfg["label"])
    plt.axvline(tau, linestyle="--", color="black", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("entropy H(policy)")
    plt.title("Bandit exploration: entropy over time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "bandit_entropy.png")
    plt.savefig(path, dpi=300)
    plt.show()
    print("Saved:", path)

    # --- Cumulative P&L over time ---
    plt.figure(figsize=(7, 4))
    for cfg in configs:
        t, p_path, probs_hist, H, pnl = run_nonstationary_bandit(
            spreads, cfg["T"], alpha, t_max, dt, p_L, p_H, tau, seed=cfg["seed"]
        )
        plt.plot(t, pnl, label=cfg["label"])
    plt.axvline(tau, linestyle="--", color="black", linewidth=1)
    plt.xlabel("time")
    plt.ylabel("cumulative P&L")
    plt.title("Bandit performance under nonstationarity")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "bandit_cum_pnl.png")
    plt.savefig(path, dpi=300)
    plt.show()
    print("Saved:", path)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    spreads = np.array([0.20, 0.25, 0.30, 0.35, 0.40])

    # Plot stationary arm means (pick one p_I)
    plot_stationary_bandit_means(spreads, p_I=0.15)

    # Plot nonstationary bandit learning results
    plot_nonstationary_bandit(spreads)

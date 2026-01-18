import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Half-spreads
    s_L = 0.20
    s_M = 0.30
    s_H = 0.40

    # Toxicity grid
    p = np.linspace(0.0, 0.50, 501)

    # Win-all-flow payoffs
    E_L = s_L - p
    E_M = s_M - p
    E_H = s_H - p

    # Tie splits (n = 3 market makers)
    E_L_2 = 0.5 * E_L
    E_M_2 = 0.5 * E_M
    E_H_2 = 0.5 * E_H

    E_L_3 = (1/3) * E_L
    E_M_3 = (1/3) * E_M
    E_H_3 = (1/3) * E_H

    # Large-n (perfect competition limit): slightly above zero for visibility
    epsilon = 0.005
    E_pc = epsilon * np.ones_like(p)


    # Create output folder
    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(8, 4.5))

    # Win-all-flow
    plt.plot(p, E_L, label=r"$E_L = s_L - p_I$")
    plt.plot(p, E_M, label=r"$E_M = s_M - p_I$")
    plt.plot(p, E_H, label=r"$E_H = s_H - p_I$")

    # Two-way ties
    plt.plot(p, E_L_2, "--", label=r"$E_L/2$")
    plt.plot(p, E_M_2, "--", label=r"$E_M/2$")
    plt.plot(p, E_H_2, "--", label=r"$E_H/2$")

    # Three-way ties
    plt.plot(p, E_L_3, ":", label=r"$E_L/3$")
    plt.plot(p, E_M_3, ":", label=r"$E_M/3$")
    plt.plot(p, E_H_3, ":", label=r"$E_H/3$")

    # Perfect competition line
    plt.plot(p, E_pc, color="black", linewidth=2,
             label=r"large $n$ (perfect competition)")

    # Regime markers
    plt.axvline(s_L, color="black", linestyle=":", linewidth=1)
    plt.axvline(s_M, color="black", linestyle=":", linewidth=1)

    plt.xlabel(r"toxicity $p_I$")
    plt.ylabel("expected profit per executed trade")
    plt.title("Per-trade profit vs toxicity (3 spreads, increasing competition)")
    plt.grid(True)
    plt.legend(fontsize=9, ncol=3)
    plt.tight_layout()

    outpath = os.path.join("figures", "fig_payoff_regimes_3x3_n3_largeN.png")
    plt.savefig(outpath, dpi=300)
    plt.show()

    print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()

import os
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Half-spreads used in your project
    s_L = 0.20
    s_H = 0.30

    # Toxicity grid
    p = np.linspace(0.0, 0.50, 501)

    E_L = s_L - p
    E_H = s_H - p
    E_L_half = 0.5 * E_L
    E_H_half = 0.5 * E_H

    # Create output folder
    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(p, E_L, label=r"$E_L = s_L - p_I$ (win all flow)")
    plt.plot(p, E_H, label=r"$E_H = s_H - p_I$ (win all flow)")
    plt.plot(p, E_L_half, linestyle="--", label=r"$E_L/2$ (tie split)")
    plt.plot(p, E_H_half, linestyle="--", label=r"$E_H/2$ (tie split)")

    # Regime markers that matter in your analysis
    plt.axvline(0.10, color="black", linestyle=":", linewidth=1)
    plt.axvline(0.20, color="black", linestyle=":", linewidth=1)

    plt.xlabel(r"toxicity $p_I$")
    plt.ylabel(r"expected profit per executed trade")
    plt.title("Per-trade profit vs toxicity")
    plt.grid(True)
    plt.legend(fontsize=9)
    plt.tight_layout()

    outpath = os.path.join("figures", "fig_payoff_regimes.png")
    plt.savefig(outpath, dpi=300)
    plt.show()

    print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()

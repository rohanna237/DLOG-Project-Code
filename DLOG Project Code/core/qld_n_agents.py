import numpy as np
from scipy.stats import binom

def expected_share(N, p):
    """
    E[1 / (1 + K)] where K ~ Binomial(N-1, p)
    """
    ks = np.arange(N)
    probs = binom.pmf(ks, N-1, p)
    return np.sum(probs / (1 + ks))

def payoff_vector(x, spreads, p_I, N):
    """
    Compute U_i(x) for symmetric N-agent model.
    """
    m = len(spreads)
    U = np.zeros(m)

    S = np.array([np.sum(x[i:]) for i in range(m)])

    for i in range(m):
        if S[i] > 0:
            E_i = spreads[i] - p_I
            share = expected_share(N, x[i] / S[i])
            U[i] = E_i * (S[i] ** (N - 1)) * share
        else:
            U[i] = 0.0

    return U

def qld_mean_field_rhs(x, spreads, p_I, N, T):
    """
    Mean-field QLD for symmetric N-agent competition.
    """
    U = payoff_vector(x, spreads, p_I, N)
    avg_U = np.dot(x, U)

    dx = x * (U - avg_U + T * (np.log(x) - np.dot(x, np.log(x))))

    return dx
import numpy as np

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)

def qld_log_odds(u, v, T, A):
    """
    Log-odds dynamics for 2x2 QLD.
    u = log(x1 / x2)
    v = log(y1 / y2)
    """
    x = softmax(np.array([u, 0.0]))
    y = softmax(np.array([v, 0.0]))

    Ay = A @ y
    Bx = A.T @ x

    du = Ay[0] - Ay[1] - T * u
    dv = Bx[0] - Bx[1] - T * v

    return du, dv

import numpy as np

def payoff_matrix(spreads, p_I):
    """
    Construct A(p_I) for two makers with m spreads.
    Tightest quote wins, ties split.
    """
    m = len(spreads)
    A = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            if spreads[i] < spreads[j]:
                A[i, j] = spreads[i] - p_I
            elif spreads[i] == spreads[j]:
                A[i, j] = 0.5 * (spreads[i] - p_I)
            else:
                A[i, j] = 0.0

    return A

def qld_rhs(x, y, A, T):
    """
    RHS of QLD for two players with m actions.
    """
    Ay = A @ y
    avg_x = x @ Ay

    dx = x * (Ay - avg_x + T * (np.log(x) - np.dot(x, np.log(x))))

    B = A.T
    Bx = B @ x
    avg_y = y @ Bx

    dy = y * (Bx - avg_y + T * (np.log(y) - np.dot(y, np.log(y))))

    return dx, dy

import numpy as np
from numpy.random import default_rng


def cycle_pattern(pattern, length):
    """Repeat/trim a base list to desired length."""
    pattern = np.asarray(pattern)
    reps = (length + len(pattern) - 1) // len(pattern)
    return np.tile(pattern, reps)[:length]


def generate_data_set(data_size=30, m=6, n=20, seed=None):
    """
    Generate a dataset of data_size instances of the two-stage stochastic LP problem.
    Each instance has m constraints, n first-stage variables.
    Returns: list of dicts, each dict with keys: c, q, T, W, h, gamma, u
    """
    rng = default_rng(seed)

    # Shared coefficients
    c_base = [-12, -20, -18, -40]#, -20, -10, -24]
    c = cycle_pattern(c_base, n).astype(float)

    q_base = [6, 12]
    #q = q_base
    q = np.hstack([cycle_pattern(q_base, m), np.zeros(m)]).astype(float)

    # Deterministic matrix β used to build T
    beta_row0 = np.array([4, 9, 7, 10])
    beta_row1 = np.array([3, 1, 3, 6])
    # beta_row2 = np.array([3,7,9,10])
    # beta_row3 = np.array([2,7,3,6])
    # beta_row4 = np.array([6,9,3,5])
    # beta_row5 = np.array([2,3,4,5])
    beta = np.vstack([
        cycle_pattern(beta_row0, n), cycle_pattern(beta_row1, n), cycle_pattern(beta_row0, n), cycle_pattern(beta_row1, n),
        cycle_pattern(beta_row0, n), cycle_pattern(beta_row1, n), cycle_pattern(beta_row0, n), cycle_pattern(beta_row1, n),
        cycle_pattern(beta_row0, n), cycle_pattern(beta_row1, n)
    ])

    # Shared mean & covariances
    mu1 = cycle_pattern([12, 8], m)
    mu2 = cycle_pattern([2, 1], m)
    sig1 = np.diag(cycle_pattern([5.76, 2.56], m))
    sig2 = np.diag(cycle_pattern([0.16, 0.04], m))
    for k in range(0, m - 1, 2):
        sig1[k, k + 1] = sig1[k + 1, k] = 1.92

    data_list = []

    for i in range(data_size):
        # Draw mixture component
        mix_flag = rng.binomial(1, 0.7)
        gamma = (
            rng.multivariate_normal(mu1, sig1) if mix_flag
            else rng.multivariate_normal(mu2, sig2)
        )
        u = rng.uniform(0.6, 1.2, size=m)

        T = beta - gamma[:, None] / 4.0
        W = np.hstack([-np.diag(u), np.eye(m)])
        h = 500 * gamma

        data_list.append(dict(
            c=c,
            q=q,
            T=T,
            W=W,
            h=h,
            gamma=gamma,
            u=u
        ))

    return data_list


# quick demo -------------------------------------------------------
if __name__ == "__main__":
    dataset = generate_data_set(data_size=30, m=6, n=20, seed=1234)
    print(f"Generated {len(dataset)} instances.")
    # print(f"First‑stage dimension n = {dataset[0]['n']}")
    print(f"Second‑stage dimension |y| = {dataset[0]['q'].size}")
    print("T shape:", dataset[0]['T'].shape, "W shape:", dataset[0]['W'].shape,"h shape:", dataset[0]['h'].shape)
    print("Instance[0] c[:5]:", dataset[1]['c'][:5])
    print("Instance[0] h[:3]:", dataset[0]['h'][:3])

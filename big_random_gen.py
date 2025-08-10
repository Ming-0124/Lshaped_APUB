import numpy as np
from scipy.linalg import block_diag
import params


def generate_W():
    val1 = np.random.uniform(0.6, 1.0)
    val2 = np.random.uniform(0.6, 1.0)
    W = np.array([[val1, 0, -1, 0, 1, 0],
                  [0, val2, 0, -1, 0, 1],
                  [0, 0, 0, 0, 1, 1]])
    return W


def make_random_block_matrix(k):
    """
    生成包含 k 个 T 块的块对角矩阵
    :param k: T 矩阵的数量
    :return: 一个形状为 (2k,4k) 的 NumPy 数组
    """
    blocks = [generate_W() for _ in range(k)]
    W = block_diag(*blocks)
    return W


def generate_Th(k):
    T_list = []
    h_list = []
    # std1 = np.sqrt(params.var1)
    # std2 = np.sqrt(params.var2)
    if np.random.rand() < params.p:  # 以 p 的概率从第一个分布采样
        samples = np.random.normal(loc=params.mu1, scale=params.var1)
    else:
        samples = np.random.normal(loc=params.mu2, scale=params.var2)
    
    H = 5*samples

    T_row1 = [4, 9, 7, 10]
    T_row2 = [3, 1, 3, 6]
    T_row3 = [0, 0, 0, 0]
    # T_row1 = [4, 9, 7, 10] - (gamma1 / 4)
    # T_row2 = [3, 1, 3, 6] - (gamma2 / 4)
    T = np.vstack([T_row1, T_row2, T_row3])

    for i in range(k):
        T_list.append(T)
        h_list.append(np.array([0, 0, H]))
    h = np.concatenate(h_list)
    blocks = [T for T in T_list]
    M = block_diag(*blocks)
    return M, h


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
    # Shared coefficients
    c = params.c
    c = cycle_pattern(c, n).astype(float)
    q = params.q
    q = np.hstack([cycle_pattern(q, m), np.zeros(2*m)]).astype(float)

    data_list = []
    for i in range(data_size):
        T, h = generate_Th(m//2)
        W = make_random_block_matrix(m//2)

        data_list.append(dict(
            c=c,
            q=q,
            T=T,
            W=W,
            h=h,
        ))

    return data_list


# quick demo -------------------------------------------------------
if __name__ == "__main__":
    dataset = generate_data_set(data_size=30, m=2, n=4, seed=1234)
    # print(f"First‑stage dimension n = {dataset[0]['n']}")
    print(f"Second‑stage dimension |y| = {len(dataset[0]['q'])}")
    # print(f"c: {dataset[1]['c']}")
    print(f"q: {dataset[1]['q']}")
    # print(f"T: {dataset[0]['T']}")
    print("T shape:", dataset[0]['T'].shape, "W shape:",
          dataset[0]['W'].shape, "h shape:", dataset[0]['h'].shape)
    # print("Instance[0] c[:5]:", dataset[1]['c'][:5])
    # print("Instance[0] h[:3]:", dataset[0]['h'][:3],'\n')
    # print("W: ", dataset[0]['W'])
    np.savetxt("matrix1.txt", dataset[0]['h'], fmt="%.2f")

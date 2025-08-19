import numpy as np
import os
from typing import Any, Dict, Union
try:
    import yaml
except ImportError:  # Lightweight fallback if yaml is missing
    yaml = None

# --- 1) 采样正稳定分布 S(α) 的函数（Chambers–Mallows–Stuck 方法, α∈(0,1]) ---
def positive_stable(alpha, size=None, rng=None):
    """
    返回 V ~ Stable(alpha) 的非负样本，满足 Laplace: E[e^{-t V}] = exp(-t^alpha)
    参考: CMS 变换法在 α∈(0,1) 的正稳定情形
    """
    if rng is None:
        rng = np.random.default_rng()
    U = rng.random(size) * np.pi/2  # Uniform(0, pi/2)
    W = rng.exponential(scale=1.0, size=size)  # Exp(1)
    # Chambers-Mallows-Stuck formula for positive stable
    # V = (sin(alpha*U) / (cos(U))**(1/alpha)) * ( cos(U - alpha*U) / W )**((1-alpha)/alpha)
    # 为数值稳定，分步写：
    s1 = np.sin(alpha * U) / (np.cos(U))**(1/alpha)
    s2 = np.cos(U - alpha * U) / W
    V = (s1 * (s2)**((1-alpha)/alpha))
    return V

# --- 2) 从 Gumbel copula 采样 ---
def sample_gumbel_copula(n, d, lam, rng=None):
    """
    返回形状 (n, d) 的 U∈(0,1)^d,服从 Gumbel(λ) copula,λ>=1
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha = 1.0 / lam
    V = positive_stable(alpha, size=n, rng=rng)  # (n,)
    E = rng.exponential(scale=1.0, size=(n, d))  # (n,d)
    U = np.exp(- (E / V[:, None])**alpha)
    return U

# --- 3) 把 U 映射回各自的均匀边缘区间 ---
def map_to_uniform_intervals(U, lows, highs):
    """
    U: (n,d) in (0,1); lows, highs: (d,) 对应每一维的区间端点
    """
    lows = np.asarray(lows); highs = np.asarray(highs)
    return lows + (highs - lows) * U

# --- 4) 混合常态/极端两套参数抽样（论文的设定） ---
def sample_h_q_w(n, J, p, lam_r, lam_w,
                 h_int_r, q_ints_r, w_ints_r,
                 h_int_w, q_ints_w, w_ints_w,
                 rng=None):
    """
    n: 样本数; J: 部门个数; p: 常态期概率
    lam_r/lam_w: Gumbel 参数 λ^r / λ^w (>=1)
    h_int_*: (low, high)
    q_ints_*, w_ints_*: List of J, each element is (low, high)
    return: dict 含 h, q (n,J), w (n,J)
    """
    if rng is None:
        rng = np.random.default_rng()

    d = 1 + J + J
    regime = rng.random(n) < p  # True=常态，False=极端

    # 先各自生成 U
    n_r = int(np.sum(regime))
    n_w = n - n_r
    U_r = sample_gumbel_copula(n_r, d, lam_r, rng=rng) if n_r>0 else np.empty((0,d))
    U_w = sample_gumbel_copula(n_w, d, lam_w, rng=rng) if n_w>0 else np.empty((0,d))

    # 拼接（打乱顺序保持与 regime 对齐）
    U = np.empty((n, d))
    U[regime] = U_r
    U[~regime] = U_w

    # 构造每一维的区间（按论文顺序：h, q_1..q_J, w_1..w_J）
    def build_intervals(h_int, q_ints, w_ints):
        lows = [h_int[0]] + [a for a,b in q_ints] + [a for a,b in w_ints]
        highs= [h_int[1]] + [b for a,b in q_ints] + [b for a,b in w_ints]
        return np.array(lows), np.array(highs)
    lows_r, highs_r = build_intervals(h_int_r, q_ints_r, w_ints_r)
    lows_w, highs_w = build_intervals(h_int_w, q_ints_w, w_ints_w)

    # 对不同 regime 用不同区间映射
    X = np.empty((n, d))
    if n_r>0:
        X[regime] = map_to_uniform_intervals(U[regime], lows_r, highs_r)
    if n_w>0:
        X[~regime] = map_to_uniform_intervals(U[~regime], lows_w, highs_w)

    # 拆回 h, q, w
    h = X[:, 0]
    q = X[:, 1:1+J]
    w = X[:, 1+J:1+2*J]

    # T should be provided externally via configuration in typical usage.
    # Retain a sensible default for backward compatibility.
    T_row1 = [10, 6, 8, 4]
    T_row2 = [6, 2, 3, 2]
    T_row3 = [0, 0, 0, 0]
    T = np.vstack([T_row1, T_row2, T_row3])
    return dict(
            q=q,
            T=T,
            W=w,
            h=h,
        )


# ------------------------------
# Config-based helpers
# ------------------------------
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("PyYAML is required. Please install with: pip install pyyaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _ensure_list_of_pairs(x, expected_len: int, name: str):
    arr = list(x)
    if len(arr) != expected_len:
        raise ValueError(f"Config for {name} must have length {expected_len}, got {len(arr)}")
    for idx, pair in enumerate(arr):
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            raise ValueError(f"Each entry of {name} must be a pair [low, high]; bad entry at index {idx}: {pair}")
    return arr


def sample_h_q_w_from_config(cfg_or_path: Union[str, Dict[str, Any]], rng=None) -> Dict[str, Any]:
    """
    Load parameters from YAML config (or dict) and call the sampler.

    Expected YAML structure (keys):
      random_generator:
        n: int
        J: int
        p: float
        lam_r: float
        lam_w: float
        h_int_r: [low, high]
        q_ints_r: [[low, high], ...]  # length J
        w_ints_r: [[low, high], ...]  # length J
        h_int_w: [low, high]
        q_ints_w: [[low, high], ...]  # length J
        w_ints_w: [[low, high], ...]  # length J
        T: [[...], [...], ...]        # 2D matrix
    """
    if isinstance(cfg_or_path, str):
        cfg = load_config(cfg_or_path)
    else:
        cfg = cfg_or_path

    g = cfg.get("random_generator", cfg)

    n = int(g["n"]) 
    J = int(g["J"]) 
    p = float(g["p"]) 
    lam_r = float(g["lam_r"]) 
    lam_w = float(g["lam_w"]) 

    h_int_r = list(g["h_int_r"])  # [low, high]
    h_int_w = list(g["h_int_w"])  # [low, high]

    q_ints_r = _ensure_list_of_pairs(g["q_ints_r"], J, "q_ints_r")
    w_ints_r = _ensure_list_of_pairs(g["w_ints_r"], J, "w_ints_r")
    q_ints_w = _ensure_list_of_pairs(g["q_ints_w"], J, "q_ints_w")
    w_ints_w = _ensure_list_of_pairs(g["w_ints_w"], J, "w_ints_w")

    out = sample_h_q_w(
        n=n, J=J, p=p,
        lam_r=lam_r, lam_w=lam_w,
        h_int_r=h_int_r, q_ints_r=q_ints_r, w_ints_r=w_ints_r,
        h_int_w=h_int_w, q_ints_w=q_ints_w, w_ints_w=w_ints_w,
        rng=rng,
    )

    # Override T if provided in config
    if "T" in g and g["T"] is not None:
        T_cfg = np.array(g["T"], dtype=float)
        out["T"] = T_cfg
    return out


if __name__ == "__main__":
    n = 100000
    J = 2
    p = 0.9
    lam_r = 1.5
    lam_w = 1.5
    h_int_r = (10, 20)
    h_int_w = (10, 20)
    q_ints_r = [(6, 8)]*J
    q_ints_w = [(15, 20)]*J
    w_ints_r = [(10, 20)]*J
    w_ints_w = [(10, 20)]*J

    print(sample_h_q_w(n, J, p, lam_r, lam_w,
                 h_int_r, q_ints_r, w_ints_r, h_int_w, q_ints_w, w_ints_w, rng=None))

import numpy as np
from random_generator import sample_h_q_w

class ParametersGenerator:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
    

    def build_w(self, w: np.ndarray) -> np.ndarray:
        n, j = w.shape
        dtype = w.dtype
        result = np.zeros((n, j+1, 3 * j))

        I = np.eye(j)  
        bottom = np.concatenate([
            np.zeros(j, dtype=dtype),
            np.ones(j,  dtype=dtype),
            np.zeros(j, dtype=dtype)
        ])
    
        for i in range(n):
            diag_w = np.diag(w[i])       
            diag_1 = I                   
            diag_neg1 = -I               
            top = np.hstack([diag_w, diag_1, diag_neg1])
            result[i] = np.vstack([top, bottom])
        return result
    

    def build_q(self, q):
        n_rows, j = q.shape  
        extension = np.full((n_rows, 2 * j), 0)  
        extended_q = np.concatenate([q, extension], axis=1) 
        return extended_q


    def build_h(self, h, J):
        n = len(h)
        extended = np.zeros((n, J+1))  
        extended[:, -1] = h  
        return extended
    

    def build_t(self):
        T_row1 = [-10, -6, -8, -4]
        T_row2 = [-6, -2, -3, -2]
        T_row3 = [0, 0, 0, 0]
        return np.vstack([T_row1, T_row2, T_row3])
    

    def generate_parameters(self, n, J, p, lam_r, lam_w,
                          h_int_r, q_ints_r, w_ints_r,
                          h_int_w, q_ints_w, w_ints_w):
        samples = sample_h_q_w(
            n=n, J=J, p=p,
            lam_r=lam_r, lam_w=lam_w,
            h_int_r=h_int_r, q_ints_r=q_ints_r, w_ints_r=w_ints_r,
            h_int_w=h_int_w, q_ints_w=q_ints_w, w_ints_w=w_ints_w,
            rng=self.rng
        )
        
        h = self.build_h(samples['h'], J)
        q = self.build_q(samples['q'])
        w = self.build_w(samples['W'])
        T = self.build_t()
        
        return dict(
            q=q,
            T=T,
            W=w,
            h=h,
        )


if __name__ == "__main__":
    J = 2
    p = 0.8          
    lam_r, lam_w = 2.0, 4.0   # 例子参数（Kendall τ≈0.5 与 0.75）

    h_int_r = (1000, 1500)
    q_ints_r = [(6, 10)]*J
    w_ints_r = [(0.8, 1.0)]*J

    h_int_w = (600, 900)
    q_ints_w = [(10, 16)]*J
    w_ints_w = [(0.6, 0.8)]*J
    rd = ParametersGenerator()
    samples = rd.generate_parameters(
        n=1000, J=J, p=p,
        lam_r=lam_r, lam_w=lam_w,
        h_int_r=h_int_r, q_ints_r=q_ints_r, w_ints_r=w_ints_r,
        h_int_w=h_int_w, q_ints_w=q_ints_w, w_ints_w=w_ints_w
    )

    print(samples["q"].shape)

    
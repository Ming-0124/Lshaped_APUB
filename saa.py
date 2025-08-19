import gurobipy as gp
from gurobipy import GRB
import numpy as np
from typing import Dict


class SAA:
    """
    Solve the two-stage stochastic LP via Sample Average Approximation:

      min_x>=0 c^T x + (1/N) * sum_{n=1..N} Q_n(x)

    where each Q_n(x) is the optimal value of the recourse LP:
      min_{y,z >= 0} sum_j q_{n,j} y_{n,j}
      s.t.   sum_j z_{n,j} = h_n
             w_{n,j} y_{n,j} + z_{n,j} >= sum_i T_{j,i} x_i,  for all j

    Training scenarios are provided via a dict with keys:
      q: shape (N, J)
      W: shape (N, J)   (per j weight w_j)
      h: shape (N,)     (scalar per scenario)
      T: shape (J_T, I) (common across scenarios). We'll use first J rows if J_T >= J
    """

    def __init__(self, c: np.ndarray, n_items: int, x_upper_bound: float = 2000.0):
        self.c = np.asarray(c, dtype=float)
        self.n_items = int(n_items)
        self.x_upper_bound = float(x_upper_bound)

    def _infer_dimensions(self, train_params: Dict) -> Dict:
        q = np.asarray(train_params["q"])  # (N, J)
        W = np.asarray(train_params["W"])  # (N, J)
        h = np.asarray(train_params["h"])  # (N,)
        T = np.asarray(train_params["T"])  # (J_T, I)

        N, J_q = q.shape
        J_w = W.shape[1]
        J_T = T.shape[0]
        J_eff = int(min(J_q, J_w, J_T))
        I_eff = int(min(T.shape[1], self.n_items, len(self.c)))

        return {
            "N": N,
            "J": J_eff,
            "I": I_eff,
            "q": q[:, :J_eff],
            "W": W[:, :J_eff],
            "h": h.reshape(-1),
            "T": T[:J_eff, :I_eff],
        }

    def solve(self, train_params: Dict):
        dims = self._infer_dimensions(train_params)
        N, J, I = dims["N"], dims["J"], dims["I"]
        q, W, h, T = dims["q"], dims["W"], dims["h"], dims["T"]

        model = gp.Model("SAA")
        model.setParam('OutputFlag', 0)

        # First-stage variables x >= 0
        x = model.addVars(I, lb=0.0, ub=self.x_upper_bound, name="x")

        # Second-stage variables for each scenario
        y = {}
        z = {}
        for n in range(N):
            y[n] = model.addVars(J, lb=0.0, name=f"y_{n}")
            z[n] = model.addVars(J, lb=0.0, name=f"z_{n}")

            # sum_j z_{n,j} = h_n
            model.addConstr(gp.quicksum(z[n][j] for j in range(J)) == float(h[n]), name=f"sum_z_{n}")

            # w_{n,j} y_{n,j} + z_{n,j} >= T_j x
            for j in range(J):
                model.addConstr(W[n, j] * y[n][j] + z[n][j] \
                                 >= gp.quicksum(float(T[j, i]) * x[i] for i in range(I)),
                                 name=f"link_{n}_{j}")

        # Objective: c^T x + (1/N) * sum_n sum_j q_{n,j} y_{n,j}
        obj = gp.quicksum(float(self.c[i]) * x[i] for i in range(I)) \
              + (1.0 / N) * gp.quicksum(gp.quicksum(float(q[n, j]) * y[n][j] for j in range(J))
                                        for n in range(N))

        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"SAA solve failed with status {model.status}")

        x_opt = np.array([x[i].X for i in range(I)], dtype=float)
        obj_val = float(model.ObjVal)
        return x_opt, obj_val

    def evaluate_oos(self, x: np.ndarray, test_params: Dict) -> float:
        """Compute out-of-sample expected cost c^T x + E[Q(x, ξ)] via sample average on test set."""
        dims = self._infer_dimensions(test_params)
        N, J, I = dims["N"], dims["J"], dims["I"]
        q, W, h, T = dims["q"], dims["W"], dims["h"], dims["T"]

        x_use = np.asarray(x[:I], dtype=float)
        c_term = float(np.dot(self.c[:I], x_use))

        # For each scenario, solve the second-stage recourse LP with fixed x
        q_vals = []
        for n in range(N):
            m = gp.Model(f"Q_n_{n}")
            m.setParam('OutputFlag', 0)
            y = m.addVars(J, lb=0.0, name="y")
            z = m.addVars(J, lb=0.0, name="z")

            m.addConstr(gp.quicksum(z[j] for j in range(J)) == float(h[n]))
            for j in range(J):
                rhs = float(np.dot(T[j, :], x_use))
                m.addConstr(float(W[n, j]) * y[j] + z[j] >= rhs)

            m.setObjective(gp.quicksum(float(q[n, j]) * y[j] for j in range(J)), GRB.MINIMIZE)
            m.optimize()
            if m.status != GRB.OPTIMAL:
                raise RuntimeError(f"OOS second-stage failed at scenario {n} with status {m.status}")
            q_vals.append(float(m.objVal))

        recourse_mean = float(np.mean(q_vals))
        return c_term + recourse_mean



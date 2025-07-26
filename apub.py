import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.stats import multinomial
from big_random_gen import generate_data_set
import time


class APUB:
    def __init__(self, A, b, n_items, n_machines, data_set, model):
        self.A = A
        self.b = b
        self.c = data_set[0]['c']
        self.model = model
        self.n_items = n_items
        self.n_machines = n_machines
        return

    def initialize_master_problem(self):
        """初始化主问题模型和变量"""
        x = self.model.addVars(self.n_items, lb=0, ub=500, name="x")  # 决策变量x
        eta = self.model.addVar(lb=-3000, name="eta")

        # 第一阶段目标函数
        self.model.setObjective(gp.quicksum(self.c[i] * x[i] for i in range(self.n_items)) + eta, GRB.MINIMIZE)

        # 第一阶段约束 Ax = b
        for i in range(self.n_machines):
            self.model.addConstr(gp.quicksum(self.A[i, j] * x[j] for j in range(self.n_items)) == self.b[i],
                                 name=f"First_Stage_Constr_{i}")

        self.model.update()
        return self.model.getVars()

    def solve_master_problem(self):
        """求解主问题"""
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            return self.model.getVars()
        else:
            raise Exception(f"主问题求解失败，状态码: {self.model.status}")

    def check_feasibility(self, x_vals, params_list):
        for params in params_list:
            h_n, T_n, W_n = params['h'], params['T'], params['W']
            feas_model = gp.Model("Feasibility_Check")
            y = feas_model.addVars(self.n_machines*2, lb=0, name="y")
            v_p = feas_model.addVars(self.n_machines, lb=0, name="v_p")
            v_m = feas_model.addVars(self.n_machines, lb=0, name="v_m")

            # 约束：Wy + v = h - Tx
            for i in range(self.n_machines):
                feas_model.addConstr(
                    gp.quicksum(W_n[i, j] * y[j] for j in range(2*self.n_machines)) + v_p[i] - v_m[i] == h_n[i] - gp.quicksum(
                        T_n[i, j] * x_vals[j].X for j in range(2*self.n_machines)),
                    name=f"Feas_Constr_{i}")

            feas_model.setObjective(sum(v_p[i] + v_m[i] for i in range(self.n_machines)), GRB.MINIMIZE)
            feas_model.update()
            feas_model.setParam('OutputFlag', 0)
            feas_model.optimize()

            if feas_model.ObjVal > 1e-6:  # 不可行时生成切割
                phi = [constr.Pi for constr in feas_model.getConstrs()]  # 对偶变量
                D_new = np.dot(phi, T_n)  # 切割系数 D_j
                d_new = np.dot(phi, h_n)  # 切割常数 d_j
                self.model.addConstr(gp.quicksum(D_new[j] * x_vals[j].X for j in range(self.n_items)) >= d_new)
                self.model.update()
                print('feasile cut added')
                return True
        return False

    def generate_optimality_cuts(self, x_vals, params_list, alpha, M_bootstrap, eta_hat):
        N = len(params_list)
        Q_values = []
        E_list = []
        e_list = []
        duals = []
        T_list = []

        # 计算所有样本的第二阶段成本和对偶乘子
        for params in params_list:
            q_n, W_n, h_n, T_n = params['q'], params['W'], params['h'], params['T']
            T_list.append(T_n)
            model = gp.Model("Second_Stage")
            y = model.addVars(2 * self.n_machines, lb=0, name="y")  # y包含决策变量和松弛变量

            # 约束：Wy = h - Tx
            for i in range(self.n_machines):
                model.addConstr(
                    gp.quicksum(W_n[i, j] * y[j] for j in range(2 * self.n_machines)) == h_n[i] - gp.quicksum(
                        T_n[i, j] * x_vals[j].X for j in range(self.n_items)),
                    name=f"Sub_Constr_{i}")

            model.setObjective(gp.quicksum(q_n[j] * y[j] for j in range(2 * self.n_machines)), GRB.MINIMIZE)
            model.update()
            model.setParam('OutputFlag', 0)
            model.optimize()
            Q_values.append(model.objVal)
            duals.append([con.Pi for con in model.getConstrs()])

        # Bootstrap计算APUB
        r = []
        for m in range(M_bootstrap):
            bootstrap_indices = np.random.choice(N, size=N, replace=True)
            V_mn = np.bincount(bootstrap_indices, minlength=N)
            r_m = (Q_values @ V_mn) / N
            r.append(r_m)
            # E_m = np.zeros(self.n_items)
            # e_m = 0
            # for n in range(N):
            #     E_m += V_mn[n] * np.dot(duals[n], T_list[n])
            #     e_m += V_mn[n] * np.dot(duals[n], params_list[n]['h'])
            # E_m, e_m = E_m / N, e_m / N
            # E_list.append(E_m)
            # e_list.append(e_m)
            # # 向量化计算
            duals_arr = np.stack(duals)  # (N, 4)
            T_arr = np.stack(T_list)  # (N, 4, n_items)
            h_arr = np.array([p['h'] for p in params_list])  # (N, 4)

            dot_ET = np.einsum('ni, nij -> nj', duals_arr, T_arr)  # (N, n_items)
            dot_e = np.einsum('ni, ni -> n', duals_arr, h_arr)  # (N,)

            E_m = np.sum(V_mn[:, None] * dot_ET, axis=0) / N
            e_m = np.sum(V_mn * dot_e) / N 

            E_list.append(E_m)
            e_list.append(e_m)

        J = int(np.ceil((1 - alpha) * M_bootstrap))
        sorted_indices = np.argsort(r)
        e_arr = np.array(e_list)
        E_arr = np.array(E_list)
        r_arr = np.array(r)

        E_new = (1 - (M_bootstrap - J) / (alpha * M_bootstrap)) * E_list[sorted_indices[J]] + \
                (1 / (alpha * M_bootstrap)) * np.sum(E_arr[sorted_indices[J + 1 : M_bootstrap]], axis=0)

        e_new = (1 - (M_bootstrap - J) / (alpha * M_bootstrap)) * e_list[sorted_indices[J]] + \
                (1 / (alpha * M_bootstrap)) * np.sum(e_arr[sorted_indices[J + 1 : M_bootstrap]])

        w_est = (1 - (M_bootstrap - J) / (alpha * M_bootstrap)) * r[sorted_indices[J]] + \
                (1 / (alpha * M_bootstrap)) * np.sum(r_arr[sorted_indices[J + 1 : M_bootstrap]])

        if eta_hat.X >= w_est:
            return False
        self.model.addConstr(E_new @ x_vals + eta_hat >= e_new)
        self.model.update()
        return True

    def solve_two_stage_apub(self, random_params, alpha=0.1, M_bootstrap=1500):
        """
        求解两阶段APUB问题的L-Shaped算法主
        :param random_params: 列表，每个元素为字典 {'q': q, 'W': W, 'h': h, 'T': T}
        :param alpha: APUB的置信水平（默认0.1）
        :param M_bootstrap: Bootstrap样本量（默认1500）
        :return: 最优解 x, 最优值 objective_value
        """
        self.initialize_master_problem()

        num_feasibility_cut = 0
        num_optimal_cut = 0

        while True:
            # Step 1: 求解主问题（返回变量对象）
            *x_vars, eta_var = self.solve_master_problem()
            #x_opt = np.array([x_vars[i].X for i in range(self.n_items)])
            #print(f'x_opt: {x_opt}, eta_var: {eta_var.X}')

            #Step 2: 生成可行性切割（直接传递变量对象）
            cut_added = self.check_feasibility(x_vars, params_list=random_params)
            if cut_added:
                num_feasibility_cut += 1
                continue

            # Step 3: 生成最优性切割（直接传递变量对象）


            cut_added = self.generate_optimality_cuts(x_vars, params_list=random_params, M_bootstrap=M_bootstrap,
                                                      alpha=alpha, eta_hat=eta_var)
            if cut_added:
                num_optimal_cut += 1
                # constraints = self.model.getConstrs()
                # for con in constraints:
                #     row = self.model.getRow(con)  # 左侧表达式
                #     expr = f"{row} {con.Sense} {con.RHS}"  # 拼接完整公式
                #     print(f"{con.ConstrName}: {expr}")
                continue
            else:
                break

        self.model.write("model.lp")
        return self.model.getVars(), self.model.ObjVal

    def extensive_form(self, params_list, alpha=0.2, M_bootstrap=500):
        N = len(params_list)  # Number of original scenarios
        # Generate bootstrap samples (multinomial counts)
        bootstrap_counts = multinomial.rvs(N, [1 / N] * N, size=M_bootstrap)
        # print(bootstrap_counts.shape)

        try:
            # Create a new model
            model = gp.Model("TwoStage_APUB")

            # First-stage variables
            x = model.addVars(self.n_items, lb=0, ub=500, name="x")
            t = model.addVar(lb=-GRB.INFINITY, name="t")
            s = model.addVars(M_bootstrap, lb=0, name="s")

            # Second-stage variables for each original scenario
            y = {}
            for n in range(N):
                y[n] = model.addVars(2 * self.n_machines, lb=0, name=f"y_{n}")

            # Set objective: c'x + t + (1/(alpha*M)) * sum(s)
            model.setObjective(
                gp.quicksum(self.c[i] * x[i] for i in range(self.n_items)) + t + (1 / (alpha * M_bootstrap)) * gp.quicksum(
                    s[m] for m in range(M_bootstrap)),
                GRB.MINIMIZE
            )

            # First-stage constraints: Ax = b
            for i in range(self.A.shape[0]):
                model.addConstr(
                    gp.quicksum(self.A[i, j] * x[j] for j in range(self.n_items)) == self.b[i],
                    name=f"first_stage_{i}"
                )

            # Bootstrap constraints: s_m + t >= (1/N) * sum(V_mn * q_n' y_n) for each m
            for m in range(M_bootstrap):
                V_m = bootstrap_counts[m]
                model.addConstr(
                    s[m] + t >= (1 / N) * gp.quicksum(
                        V_m[n] * gp.quicksum(params_list[n]['q'][j] * y[n][j] for j in range(2 * self.n_machines))
                        for n in range(N)
                    ),
                    name=f"bootstrap_{m}"
                )

            # Second-stage constraints: W_n y_n = h_n - T_n x for each n
            for n in range(N):
                for i in range(self.n_machines):
                    model.addConstr(
                        gp.quicksum(params_list[n]['W'][i, j] * y[n][j] for j in range(2 * self.n_machines)) ==
                        params_list[n]['h'][i] - gp.quicksum(params_list[n]['T'][i, k] * x[k] for k in range(self.n_items)),
                        name=f"second_stage_{n}_{i}"
                    )

            # Optimize model
            model.setParam('OutputFlag', 0)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                x_opt = np.array([x[i].X for i in range(self.n_items)])
                obj_val = model.ObjVal
                #print(f'extensive form x_opt = {x_opt}, obj_val = {obj_val}')
                return x_opt, obj_val
            else:
                print(f"Optimization failed with status {model.status}")
                return None, None

        except gp.GurobiError as e:
            print(f"Gurobi error: {e}")
        except Exception as e:
            print(f"Other error: {e}")


if __name__ == "__main__":
    b = np.zeros(10)
    A = np.zeros((10, 20))
    xi_samples = generate_data_set(120, 10, 20)
    model = gp.Model('Master Problem')
    apub = APUB(A, b, n_items=20, n_machines=10, data_set=xi_samples, model=model)
    start1 = time.perf_counter()
    b,a = apub.extensive_form(xi_samples, M_bootstrap=1500)
    print(f'extensive form: {a}')
    end1 = time.perf_counter()
    print(f'extensive form time: {end1-start1}s')
    start2 = time.perf_counter()
    (*x_optimal, eta_optimal), a = apub.solve_two_stage_apub(
        xi_samples,
        alpha=0.2,
        M_bootstrap=1500,
    )
    end2 = time.perf_counter()
    print(f'ours: {a}')
    print(f'ours time: {end2 - start2}s')
import gurobipy as gp
import numpy as np
from gurobipy import GRB


class APUB:
    def __init__(self, A, b, c, model):
        self.A = A
        self.b = b
        self.c = c
        self.model = model
        return

    def initialize_master_problem(self):
        """初始化主问题模型和变量"""
        x = self.model.addVars(4, lb=0, ub=500, name="x")  # 决策变量x（4种产品）
        eta = self.model.addVar(lb=-5000, name="eta")

        # for i in range(4):
        #     x[i].start = 7
        # eta.start = 0.0

        # 第一阶段目标函数
        self.model.setObjective(gp.quicksum(self.c[i] * x[i] for i in range(4)) + eta, GRB.MINIMIZE)

        # 第一阶段约束 Ax = b
        for i in range(self.A.shape[0]):
            self.model.addConstr(gp.quicksum(self.A[i, j] * x[j] for j in range(4)) == self.b[i], name=f"First_Stage_Constr_{i}")

        self.model.update()

        return self.model.getVars()

    def solve_master_problem(self, ignore_eta=False):
        """求解主问题，可选择忽略eta的影响"""
        # if ignore_eta:
        #     # 临时保存原目标函数
        #     original_obj = self.model.getObjective()
        #     print('***************************')
        #
        #     *x, eta = self.model.getVars()
        #     # 设置新目标（仅优化c^T x）
        #     self.model.setObjective(gp.quicksum(self.c[i] * x[i] for i in range(4)), GRB.MINIMIZE)
        #
        #     # 移除可能存在的eta约束
        #     for constr in self.model.getConstrs():
        #         if "eta" in constr.ConstrName:
        #             self.model.remove(constr)
        #
        #     self.model.update()
        #     self.model.setParam('OutputFlag', 0)
        #     # 求解并恢复原目标
        #     self.model.optimize()
        #     self.model.setObjective(original_obj, GRB.MINIMIZE)
        # else:
        self.model.setParam('OutputFlag', 0)
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            *x, eta = self.model.getVars()
            print('解主函数')
            print('x0: ', x[0].X, 'x1: ', x[1].X, 'x2: ', x[2].X, 'x3: ', x[3].X)
            print('eta: ', eta.X)
            return self.model.getVars()
        else:
            raise Exception(f"主问题求解失败，状态码: {self.model.status}")


    def check_feasibility(self, x_vals, params_list):
        for params in params_list:
            h_n, T_n, W_n = params['h'], params['T'], params['W']
            feas_model = gp.Model("Feasibility_Check")
            y = feas_model.addVars(4, lb=0, name="y")
            v_p = feas_model.addVars(2, lb=0, name="v_p")
            v_m = feas_model.addVars(2, lb=0, name="v_m")

            # 约束：Wy + v = h - Tx
            for i in range(2):
                feas_model.addConstr(
                    gp.quicksum(W_n[i, j] * y[j] for j in range(4)) + v_p[i] - v_m[i] == h_n[i] - gp.quicksum(
                        T_n[i, j] * x_vals[j].X for j in range(4)),
                    name=f"Feas_Constr_{i}")

            feas_model.setObjective(sum(v_p[i] + v_m[i] for i in range(2)), GRB.MINIMIZE)
            feas_model.update()
            feas_model.optimize()

            if feas_model.ObjVal > 1e-6:  # 不可行时生成切割
                phi = [constr.Pi for constr in feas_model.getConstrs()]  # 对偶变量
                D_new = np.dot(phi, T_n)  # 切割系数 D_j
                d_new = np.dot(phi, h_n)  # 切割常数 d_j
                self.model.addConstr(gp.quicksum(D_new[j] * x_vals[j].X for j in range(4)) >= d_new)
                self.model.update()
                return True
        return False

    def generate_optimality_cuts(self, x_vals, params_list, alpha, M_bootstrap, eta_hat):
        N = len(params_list)
        Q_values = []
        E_list = []
        e_list = []
        duals = []
        E_m = np.zeros(4)
        e_m = 0
        T_list = []

        # 计算所有样本的第二阶段成本和对偶乘子
        for params in params_list:
            q_n, W_n, h_n, T_n = params['q'], params['W'], params['h'], params['T']
            T_list.append(T_n)
            model = gp.Model("Second_Stage")
            y = model.addVars(4, lb=0, name="y")  # y包含决策变量和松弛变量

            # 约束：Wy = h - Tx
            for i in range(2):
                model.addConstr(
                    gp.quicksum(W_n[i, j] * y[j] for j in range(4)) == h_n[i] - gp.quicksum(
                        T_n[i, j] * x_vals[j].X for j in range(4)),
                    name=f"Sub_Constr_{i}")

            model.setObjective(gp.quicksum(q_n[j] * y[j] for j in range(4)), GRB.MINIMIZE)
            model.update()
            model.setParam('OutputFlag', 0)
            model.optimize()
            Q_values.append(model.objVal)
            duals.append([con.Pi for con in model.getConstrs()])
        print('duals: ', duals)

        # Bootstrap计算APUB
        r = []
        for m in range(M_bootstrap):
            bootstrap_indices = np.random.choice(N, size=N, replace=True)
            V_mn = np.bincount(bootstrap_indices, minlength=N)

            r_m = (Q_values @ V_mn) / N
            r.append(r_m)
            for n in range(N):
                E_m += V_mn[n] * np.dot(duals[n], T_list[n])
                e_m += V_mn[n] * np.dot(duals[n], h_n)
            E_m, e_m = E_m / N, e_m / N
            E_list.append(E_m)
            e_list.append(e_m)
        # print('r: ', r)
        # print('Q_values: ', Q_values)

        J = int(np.ceil((1 - alpha) * M_bootstrap))
        sorted_indices = np.argsort(r)
        E_new = (1 - (M_bootstrap - J) / (alpha * M_bootstrap)) * E_list[sorted_indices[J]] + \
                (1 / (alpha * M_bootstrap)) * sum(E_list[sorted_indices[m]] for m in range(J + 1, M_bootstrap))
        e_new = (1 - (M_bootstrap - J) / (alpha * M_bootstrap)) * e_list[sorted_indices[J]] + \
                (1 / (alpha * M_bootstrap)) * sum(e_list[sorted_indices[m]] for m in range(J + 1, M_bootstrap))
        # print('E: ', E_new)
        # print('e', e_new)

        if eta_hat.X < (1 - (M_bootstrap - J) / (alpha * M_bootstrap)) * r[sorted_indices[J]] + \
                (1 / (alpha * M_bootstrap)) * sum(r[sorted_indices[m]] for m in range(J + 1, M_bootstrap)):
            self.model.addConstr(gp.quicksum(E_new[j] * x_vals[j] for j in range(4)) + eta_hat >= e_new)
            self.model.update()
            # print('*******add optimal cut to main problem*********')
            return True
        return False

    def solve_two_stage_apub(self, random_params, alpha=0.1, M_bootstrap=500):
        """
        求解两阶段APUB问题的L-Shaped算法主
        :param random_params: 列表，每个元素为字典 {'q': q, 'W': W, 'h': h, 'T': T}
        :param alpha: APUB的置信水平（默认0.1）
        :param M_bootstrap: Bootstrap样本量（默认500）
        :return: 最优解 x, 最优值 objective_value
        """
        self.initialize_master_problem()


        # first_iteration = True

        num_feasibility_cut = 0
        num_optimal_cut = 0

        while True:
            # Step 1: 求解主问题（返回变量对象）
            *x_vars, eta_var = self.solve_master_problem()
            # first_iteration = False

            # Step 2: 生成可行性切割（直接传递变量对象）
            # cut_added = self.check_feasibility(x_vars, params_list=random_params)
            # if cut_added:
            #     num_feasibility_cut += 1
            #     continue

            # Step 3: 生成最优性切割（直接传递变量对象）
            cut_added = self.generate_optimality_cuts(x_vars, params_list=random_params, M_bootstrap=M_bootstrap,
                                                      alpha=alpha, eta_hat=eta_var)
            if cut_added:
                num_optimal_cut += 1
                # self.model.computeIIS()
                constraints = self.model.getConstrs()
                for con in constraints:
                    row = self.model.getRow(con)  # 左侧表达式
                    expr = f"{row} {con.Sense} {con.RHS}"  # 拼接完整公式
                    print(f"{con.ConstrName}: {expr}")
                self.model.write("model.lp")
            else:
                break


        *x, eta = self.model.getVars()
        print("\n最优解:")
        print(f"x = {[round(val.X, 2) for val in x]}")
        print(f"eta = {round(eta.X, 2)}")
        print(f'optimal value: ', self.c[0]*x[0].X+self.c[1]*x[1].X+self.c[2]*x[2].X+self.c[3]*x[3].X+eta.X)


        print(num_feasibility_cut, '-------' ,num_optimal_cut)
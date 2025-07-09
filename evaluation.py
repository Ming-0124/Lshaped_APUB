import numpy as np
import gurobipy as gp
from gurobipy import GRB

def evaluate_oos(model, x_optimal, test_samples, c):
    """在测试集上评估解的性能"""
    costs = []
    for sample in test_samples:
        # 提取测试样本的W, h, T
        W_test = sample['W']
        h_test = sample['h']
        T_test = sample['T']
        q_test = sample['q']

        # 计算第二阶段成本
        sub_model = gp.Model("OOS_Evaluation")
        y = sub_model.addVars(4, lb=0)
        sub_model.setObjective(q_test @ y, GRB.MINIMIZE)
        sub_model.addConstrs(W_test[i] @ y == h_test[i] - T_test[i] @ x_optimal for i in range(2))
        sub_model.optimize()

        if sub_model.status == GRB.OPTIMAL:
            total_cost = c @ x_optimal + sub_model.ObjVal
            costs.append(total_cost)
        else:
            costs.append(np.inf)  # 标记不可行解

    return {
        'mean_cost': np.mean(costs),
        'std_cost': np.std(costs),
        'percentiles': np.percentile(costs, [5, 25, 50, 75, 95])
    }
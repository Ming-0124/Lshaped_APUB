import numpy as np
import gurobipy as gp
from gurobipy import GRB
from apub import APUB
import time
import matplotlib.pyplot as plt

def evaluate_oos(eta_optimal, x_optimal, test_samples, c):
    """在测试集上评估解的性能"""
    costs = []
    reliability = []
    certificate = 0
    for i in range(len(x_optimal)):
        certificate += c[i] * x_optimal[i].X
    certificate += eta_optimal.X

    for sample in test_samples:
        # 提取测试样本的W, h, T
        W_test = sample['W']
        h_test = sample['h']
        T_test = sample['T']
        q_test = sample['q']

        # 计算第二阶段成本
        sub_model = gp.Model("OOS_Evaluation")
        y = sub_model.addVars(4, lb=0)
        sub_model.setObjective(gp.quicksum(q_test[j] * y[j] for j in range(4)), GRB.MINIMIZE)
        # 约束：Wy = h - Tx
        for i in range(2):
            sub_model.addConstr(
                gp.quicksum(W_test[i, j] * y[j] for j in range(4)) == h_test[i] - gp.quicksum(
                    T_test[i, j] * x_optimal[j].X for j in range(4)),
                name=f"Sub_Constr_{i}")
        sub_model.update()
        sub_model.setParam('OutputFlag', 0)
        sub_model.optimize()

        temp = 0
        if sub_model.status == GRB.OPTIMAL:
            for i in range(len(x_optimal)):
                temp += c[i] * x_optimal[i].X
            total_cost = temp + sub_model.ObjVal
            costs.append(total_cost)
            reliability.append(certificate >= total_cost)
        else:
            costs.append(np.inf)  # 标记不可行解

    return {
        'mean_cost': np.mean(costs),
        'mean_reliability': np.mean(reliability)
    }


def evaluate_M_T_performance(A, b, c, random_params, M_list, alpha):
    """
    Plot time performance on different bootstrap size
    """
    time_list = []
    for M in M_list:
        model = gp.Model('Master Problem')
        apub = APUB(A, b, c, model=model)
        start = time.perf_counter()
        apub.solve_two_stage_apub(
            random_params,
            alpha=alpha,
            M_bootstrap=M,
        )
        end = time.perf_counter()
        time_list.append(end-start)
        print('bootstrap size: ', M, "   running time：", end - start, "s")
    plt.plot(M_list, time_list, label='M-T')
    plt.title('Function Plot')
    plt.xlabel('m')
    plt.ylabel('t')
    plt.legend()
    #plt.grid(True)
    plt.show()

def run_experiment(A, b, c, M, random_params, test_samples, K=30, alpha_list=None):
    if alpha_list is None:
        alpha_list = [0.01] + [0.05 * i for i in range(1, 20)]

    results = {alpha: {'costs': [], 'reliabilities': [], 'stds': []} for alpha in alpha_list}

    for trial in range(K):
        for alpha in alpha_list:
            apub = APUB(A, b, c, model=gp.Model())
            *x_optimal, eta_optimal = apub.solve_two_stage_apub(
                random_params,
                alpha=alpha,
                M_bootstrap=M,
            )

            eval_result = evaluate_oos(eta_optimal, x_optimal, test_samples,c=c)
            results[alpha]['costs'].append(eval_result['mean_cost'])
            results[alpha]['reliabilities'].append(eval_result['mean_reliability'])
            results[alpha]['stds'].append(eval_result['std_cost'])
    return results


def plot_apub_results2(results):
    alpha_list = sorted(results.keys())
    x_vals = [1 - a for a in alpha_list]
    means = [np.mean(results[a]['costs']) for a in alpha_list]
    coverage = [np.mean(results[a]['reliabilities']) for a in alpha_list]
    costs_array = [results[a]['costs'] for a in alpha_list]  # shape: (len_alpha, K)

    # 转置：每列对应一个 alpha 的 30组实验结果
    costs_array = np.array(costs_array).T  # shape: (K, len_alpha)

    lower = np.quantile(costs_array, 0.1, axis=0)
    upper = np.quantile(costs_array, 0.9, axis=0)

    best_idx = np.argmin(means)
    best_x = x_vals[best_idx]
    best_y = means[best_idx]

    fig, ax1 = plt.subplots(figsize=(8,5))
    ax1.plot(x_vals, means, 'b-', label='Out-of-sample cost')
    ax1.fill_between(x_vals, lower, upper, alpha=0.2, color='blue')
    ax1.plot(best_x, best_y, 'ko')  # 最优点标记
    ax1.annotate(f'Min: {best_y:.2f}', xy=(best_x, best_y), xytext=(best_x + 0.02, best_y + 2),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=10)
    ax1.set_xlabel('1 - alpha (confidence level)')
    ax1.set_ylabel('Expected Cost', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(x_vals, coverage, 'r--', label='Coverage Probability')
    ax2.set_ylabel('Coverage', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.title('APUB Performance vs Confidence')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return best_x, best_y

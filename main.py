from data_gennerator import DataGenerator
from apub import APUB
import numpy as np
import gurobipy as gb
from evaluation import *

if __name__ == '__main__':
    # np.random.seed(96438)
    # alpha = 0.05  # 置信水平
    c = [-12, -20, -18, -40]  # 第一阶段成本
    b = [0, 0]  # 等式约束 Ax = b
    A = np.zeros((2, 4))  # 约束矩阵
    M = 1000   # bootstrap size

    train_data_generator = DataGenerator('',30)
    xi_samples = train_data_generator.generate_product_mix_data()
    random_params = train_data_generator.generate_random_parameters(xi_samples)

    test_data_generator = DataGenerator('',500)
    test_samples = test_data_generator.generate_product_mix_data()
    test_params = test_data_generator.generate_random_parameters(test_samples)

    # evaluate_M_T_performance(A,b,c,random_params,[300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500],alpha=0.1)

    results = run_experiment(A, b, c, M, random_params, test_params)
    plot_apub_results2(results)

    # evaluate_varying_alpha_performance(random_params=random_params,test_samples=test_params,M=M,A=A,b=b,c=c)

    # apub = APUB(A, b, c, model=gb.Model('Master Problem'))
    # *x_optimal, eta_optimal = apub.solve_two_stage_apub(
    #     random_params,
    #     alpha=0.05,
    #     M_bootstrap=M,
    # )
    #
    # test_result = evaluate_oos(eta_optimal, x_optimal, test_params, c)
    # print(f"test mean cost: {test_result['mean_cost']:.4f}")
    # print(f"test mean reliability: {test_result['mean_reliability']:.4f}")



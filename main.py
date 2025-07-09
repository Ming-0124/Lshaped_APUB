from data_gennerator import DataGenerator
from apub import APUB
import numpy as np
import gurobipy as gb

if __name__ == '__main__':
    alpha = 0.1  # 置信水平
    c = [12, 20, 18, 40]  # 第一阶段成本
    b = [0, 0]  # 等式约束 Ax = b
    A = np.zeros((2, 4))  # 约束矩阵
    model = gb.Model('Master Problem')

    apub = APUB(A, b, c, model=model)
    data_generator = DataGenerator('',30)
    xi_samples = data_generator.generate_product_mix_data()
    random_params = data_generator.generate_random_parameters(xi_samples)
    apub.solve_two_stage_apub(
        random_params,
        alpha=alpha,
        M_bootstrap=500,
    )

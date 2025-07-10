from data_gennerator import DataGenerator
from apub import APUB
import numpy as np
import gurobipy as gb
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    alpha = 0.1  # 置信水平
    c = [-12, -20, -18, -40]  # 第一阶段成本
    b = [0, 0]  # 等式约束 Ax = b
    # A = np.array([
    #     [2, 3, 0, 0],
    #     [1, 0, 4, 0]
    # ])
    A = np.zeros((2, 4))  # 约束矩阵
    time_list = []
    M_list = [100,300,500,700,900,1100,1300,1500,1700,1900,2100,2300,2500,2700,2900]
    # model = gb.Model('Master Problem')
    #
    # apub = APUB(A, b, c, model=model)
    data_generator = DataGenerator('',30)
    xi_samples = data_generator.generate_product_mix_data()
    random_params = data_generator.generate_random_parameters(xi_samples)

    for M in M_list:
        model = gb.Model('Master Problem')

        apub = APUB(A, b, c, model=model)
        start = time.perf_counter()
        apub.solve_two_stage_apub(
            random_params,
            alpha=alpha,
            M_bootstrap=M,
        )
        end = time.perf_counter()
        time_list.append(end-start)
    print("运行时间为：", end - start, "秒")
    plt.plot(M_list, time_list, label='M-T')

    # 添加标题、标签和图例
    plt.title('Function Plot')
    plt.xlabel('m')
    plt.ylabel('t')
    plt.legend()
    plt.grid(True)

    # 显示图像
    plt.show()



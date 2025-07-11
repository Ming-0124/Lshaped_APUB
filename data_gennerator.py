import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

class DataGenerator:
    def __init__(self, data_path, data_size):
        self.data_path = data_path
        self.data_size = data_size  # N
        # self.num_groups = num_groups

    def generate_product_mix_data(self):
        """
        生成两阶段产品混合问题的随机样本
        :return: 样本数组 (N, 2)，每行为 [gamma1, gamma2]
        """
        # 定义混合正态分布的参数
        mu1 = np.array([12, 8])
        mu2 = np.array([2, 1])
        Sigma1 = np.array([[5.76, 1.92], [1.92, 2.56]])
        Sigma2 = np.array([[0.16, 0.04], [0.04, 0.04]])

        # 生成样本
        samples = np.zeros((self.data_size, 2))
        for i in range(self.data_size):
            if np.random.rand() < 0.7:  # 以70%概率从第一个分布采样
                samples[i] = np.random.multivariate_normal(mu1, Sigma1)
            else:  # 30%概率从第二个分布采样
                samples[i] = np.random.multivariate_normal(mu2, Sigma2)

        # 确保gamma非负（实际工时不可能为负）
        samples = np.maximum(samples, 0)
        return samples

    def generate_random_recourse_W(self):
        return np.array([
            [-np.random.uniform(0.6, 1.2), 0, 1, 0],
            [0, -np.random.uniform(0.6, 1.2), 0, 1]
        ])

    def generate_random_parameters(self, gamma_samples):
        """
        根据gamma样本生成每组样本对应的(q, W, h, T)
        :param gamma_samples: 二维数组 (N, 2)，每行为 [gamma1, gamma2]
        :return: 列表，每个元素为字典 {'q': q, 'W': W, 'h': h, 'T': T}
        """
        N = gamma_samples.shape[0]
        params_list = []

        # 固定参数
        q = np.array([6, 12, 0, 0])  # 第二阶段成本

        for i in range(N):
            gamma1, gamma2 = gamma_samples[i]

            # 随机参数h
            h = np.array([500 * gamma1, 500 * gamma2])

            # 随机参数T（生产时间矩阵）
            T = np.array([
                [4 - (gamma1 / 4), 9 - (gamma1 / 4), 7 - (gamma1 / 4), 10 - (gamma1 / 4)],
                [3 - (gamma2 / 4), 1 - (gamma2 / 4), 3 - (gamma2 / 4), 6 - (gamma2 / 4)]
            ])

            # 效率矩阵
            W = self.generate_random_recourse_W()

            params_list.append({'q': q, 'W': W, 'h': h, 'T': T})

        return params_list


    def split_data(self, samples, k_folds=5, test_ratio=0.2):
        """分割数据为训练集和测试集"""
        n = len(samples)
        if k_folds > 1:
            kf = KFold(n_splits=k_folds, shuffle=True)
            return [(samples[train_idx], samples[test_idx]) for train_idx, test_idx in kf.split(samples)]
        else:
            test_size = int(n * test_ratio)
            np.random.shuffle(samples)
            return [(samples[test_size:], samples[:test_size])]



if __name__ == '__main__':
    train_data_generator = DataGenerator('', 30)
    xi_samples = train_data_generator.generate_product_mix_data()
    random_params = train_data_generator.generate_random_parameters(xi_samples)

    test_data_generator = DataGenerator('', 500)
    test_samples = test_data_generator.generate_product_mix_data()
    test_params = test_data_generator.generate_random_parameters(test_samples)

    # print(xi_samples,'\n')
    # print(test_samples)
    print(random_params,'\n')
    print(test_params)


import numpy as np
import params
from evaluation import *

if __name__ == '__main__':
    # np.random.seed(params.seed)

    n_items = params.n_items
    n_machines = params.n_machines
    b = params.b  
    A = params.A  
    N = params.data_size

    #m_list = [250] + [250 * i for i in range(2, 12)]
    #evaluate_M_T_performance(A, b, m_list, n_items=n_items, n_machines=n_machines)

    results = run_experiment(A, b, M=params.bootstrap_size, n_items=n_items, n_machines=n_machines,
                             data_size=N, test_size=1000, K=params.epoch)
    plot_apub_results(results)

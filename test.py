import gurobipy as gp
import params
import numpy as np
from apub import APUB
import matplotlib.pyplot as plt
from big_random_gen import generate_data_set


if __name__ == "__main__":
    b = params.b
    A = params.A
    model = gp.Model('Master Problem')
    apub = APUB(A, b, n_items=4, n_machines=2, model=model)
    obj_values = []

    for r in range(30):
        params_list = generate_data_set(240, 2, 4)
        _, obj_val = apub.run_saa(params_list)
        if obj_val is not None:
            obj_values.append(obj_val)

    plt.figure(figsize=(6, 5))
    plt.boxplot(obj_values, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel("Objective Value")
    plt.title(f"SAA Objective Value Boxplot (N={30})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

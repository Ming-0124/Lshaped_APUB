import gurobipy as gp
import params
from apub import APUB
import matplotlib.pyplot as plt


if __name__ == "__main__":
    b = params.b
    A = params.A
    model = gp.Model('Master Problem')
    apub = APUB(A, b, n_items=4, n_machines=2, model=model)
    obj_values = []
    J = 2
    p = 0.995
    lam_r, lam_w = 2.0, 5.0 

    h_int_r = (5000, 6000)
    q_ints_r = [(6, 8)]*J
    w_ints_r = [(0.8, 1.0)]*J

    h_int_w = (500, 600)
    q_ints_w = [(20, 25)]*J
    w_ints_w = [(0.5, 0.6)]*J
    from params_generator import ParametersGenerator
    pg = ParametersGenerator()

    for r in range(100):
        params_list = samples = pg.generate_parameters(
                                n=60, J=J, p=p,
                                lam_r=lam_r, lam_w=lam_w,
                                h_int_r=h_int_r, q_ints_r=q_ints_r, w_ints_r=w_ints_r,
                                h_int_w=h_int_w, q_ints_w=q_ints_w, w_ints_w=w_ints_w)
        _, obj_val = apub.run_saa(params_list)
        if obj_val is not None:
            obj_values.append(obj_val)

    plt.figure(figsize=(6, 6))
    plt.boxplot(obj_values, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel("Objective Value")
    plt.title(f"SAA Objective Value Boxplot (N={240})")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from saa import SAA
from random_generator import sample_h_q_w_from_config, load_config
from apub import APUB
from params_generator import ParametersGenerator
from evaluation import evaluate_oos
import params


def main():
    cfg_path = "config.yaml"

    # Load shared sampling hyperparameters from config
    full_cfg = load_config(cfg_path)
    rg_cfg = full_cfg.get("random_generator", full_cfg)

    # Default train/test sizes if not present in config
    n_train = int(rg_cfg.get("train_n", 60))
    n_test = int(rg_cfg.get("test_n", 1000))

    # Training set from config (for SAA)
    cfg_train = {"random_generator": {**rg_cfg, "n": n_train}}
    train = sample_h_q_w_from_config(cfg_train)

    # Solve SAA
    saa = SAA(c=np.array(params.c), n_items=params.n_items)
    x_opt, train_obj = saa.solve(train)

    # Evaluate OOS performance (SAA) and APUB alpha=1 side-by-side
    saa_oos_costs = []
    apub_oos_costs = []
    num_trials = 30

    # APUB setup (hyperparameters also from config)
    A = params.A
    b = params.b
    n_items = params.n_items
    n_machines = params.n_machines
    J = int(rg_cfg.get("J", n_machines))
    p = float(rg_cfg.get("p", params.p))
    lam_r = float(rg_cfg.get("lam_r", 2.0))
    lam_w = float(rg_cfg.get("lam_w", 5.0))
    h_int_r = tuple(rg_cfg.get("h_int_r", [5000, 6000]))
    q_ints_r = [tuple(x) for x in rg_cfg.get("q_ints_r", [(6, 8)] * J)]
    w_ints_r = [tuple(x) for x in rg_cfg.get("w_ints_r", [(0.8, 1.0)] * J)]
    h_int_w = tuple(rg_cfg.get("h_int_w", [500, 600]))
    q_ints_w = [tuple(x) for x in rg_cfg.get("q_ints_w", [(15, 20)] * J)]
    w_ints_w = [tuple(x) for x in rg_cfg.get("w_ints_w", [(0.5, 0.6)] * J)]
    alpha = 1.0
    pg = ParametersGenerator()

    for _ in range(num_trials):
        # SAA OOS cost (resample test each trial)
        cfg_test = {"random_generator": {**rg_cfg, "n": n_test}}
        test_saa = sample_h_q_w_from_config(cfg_test)
        oos_cost = saa.evaluate_oos(x_opt, test_saa)
        saa_oos_costs.append(oos_cost)

        # APUB trial
        train_apub = pg.generate_parameters(
            n=n_train, J=J, p=p,
            lam_r=lam_r, lam_w=lam_w,
            h_int_r=h_int_r, q_ints_r=q_ints_r, w_ints_r=w_ints_r,
            h_int_w=h_int_w, q_ints_w=q_ints_w, w_ints_w=w_ints_w)
        test_apub = pg.generate_parameters(
            n=n_test, J=J, p=p,
            lam_r=lam_r, lam_w=lam_w,
            h_int_r=h_int_r, q_ints_r=q_ints_r, w_ints_r=w_ints_r,
            h_int_w=h_int_w, q_ints_w=q_ints_w, w_ints_w=w_ints_w)
        apub_model = gp.Model('APUB')
        apub_solver = APUB(A, b, n_items=n_items, n_machines=n_machines, model=apub_model)
        x_optimal, eta_optimal, certificate = apub_solver.solve_two_stage_apub(
            train_apub, alpha=alpha, M_bootstrap=1000
        )
        eval_result = evaluate_oos(certificate, x_optimal, test_apub, c=params.c,
                                   n_items=n_items, n_machines=n_machines)
        apub_oos_costs.append(eval_result['mean_cost'])

    # SAA-only boxplot
    plt.figure(figsize=(6, 6))
    plt.boxplot(saa_oos_costs, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel("Out-of-Sample Cost")
    plt.title("SAA Out-of-Sample Performance")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Side-by-side boxplot
    fig, ax = plt.subplots(figsize=(7, 6))
    box_data = [saa_oos_costs, apub_oos_costs]
    labels = ['SAA', f'APUB (α={alpha:.0f})']
    colors = ['lightblue', 'lightgreen']
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                    medianprops=dict(color='red', linewidth=2))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel("Out-of-Sample Cost")
    ax.set_title("SAA vs APUB Out-of-Sample Performance")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    print(f"SAA  Mean OOS cost: {np.mean(saa_oos_costs):.2f}  | Std: {np.std(saa_oos_costs):.2f}")
    print(f"APUB Mean OOS cost: {np.mean(apub_oos_costs):.2f} | Std: {np.std(apub_oos_costs):.2f}")


if __name__ == "__main__":
    main()



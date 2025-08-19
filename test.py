import gurobipy as gp
import params
from apub import APUB
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    b = params.b
    A = params.A
    model = gp.Model('Master Problem')
    apub = APUB(A, b, n_items=4, n_machines=2, model=model)
    obj_values = []
    J = 2
    p = 0.9
    alpha = 1
    lam_r, lam_w = 2.0, 5.0 

    h_int_r = (5000, 6000)
    q_ints_r = [(6, 8)]*J
    w_ints_r = [(0.8, 1.0)]*J

    h_int_w = (500, 600)
    q_ints_w = [(15, 20)]*J
    w_ints_w = [(0.5, 0.6)]*J
    from params_generator import ParametersGenerator
    pg = ParametersGenerator()

    # SAA results
    for r in range(100):
        params_list = samples = pg.generate_parameters(
                                n=30, J=J, p=p,
                                lam_r=lam_r, lam_w=lam_w,
                                h_int_r=h_int_r, q_ints_r=q_ints_r, w_ints_r=w_ints_r,
                                h_int_w=h_int_w, q_ints_w=q_ints_w, w_ints_w=w_ints_w)
        _, obj_val = apub.run_saa(params_list)
        if obj_val is not None:
            obj_values.append(obj_val)
    
    # APUB out-of-sample performance for alpha = 0.0
    print(f"Running APUB with alpha = {alpha} for out-of-sample evaluation...")
    apub_costs = []
    apub_reliabilities = []
    
    for r in range(30):  # Run 30 trials for APUB
        # Generate training and test samples
        train_samples = pg.generate_parameters(
            n=60, J=J, p=p,
            lam_r=lam_r, lam_w=lam_w,
            h_int_r=h_int_r, q_ints_r=q_ints_r, w_ints_r=w_ints_r,
            h_int_w=h_int_w, q_ints_w=q_ints_w, w_ints_w=w_ints_w)
        
        test_samples = pg.generate_parameters(
            n=1000, J=J, p=p,
            lam_r=lam_r, lam_w=lam_w,
            h_int_r=h_int_r, q_ints_r=q_ints_r, w_ints_r=w_ints_r,
            h_int_w=h_int_w, q_ints_w=q_ints_w, w_ints_w=w_ints_w)
        
        # Solve APUB
        apub_model = gp.Model('APUB')
        apub_solver = APUB(A, b, n_items=4, n_machines=2, model=apub_model)
        x_optimal, eta_optimal, certificate = apub_solver.solve_two_stage_apub(
            train_samples, alpha=alpha, M_bootstrap=1000
        )
        
        # Evaluate out-of-sample performance
        from evaluation import evaluate_oos
        eval_result = evaluate_oos(certificate, x_optimal, test_samples, c=params.c,
                                   n_items=4, n_machines=2)
        
        apub_costs.append(eval_result['mean_cost'])
        apub_reliabilities.append(eval_result['reliability'])
        
        print(f'APUB trial {r+1}/30: cost={eval_result["mean_cost"]:.2f}, reliability={eval_result["reliability"]:.3f}')

    # Create side-by-side boxplot with reliability information
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Side-by-side boxplots
    box_data = [obj_values, apub_costs]
    box_labels = [f'SAA', f'APUB (α={alpha})']
    box_colors = ['lightblue', 'lightgreen']
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True,
                     medianprops=dict(color='red', linewidth=2))
    
    # Set colors for each box individually
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
    ax1.set_ylabel("Cost Value")
    ax1.set_title("SAA vs APUB Performance Comparison")
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Add reliability information as text on the plot
    saa_reliability = "N/A"  # SAA doesn't have reliability concept
    apub_reliability = f"{np.mean(apub_reliabilities):.3f}"
    
    ax1.text(0.02, 0.98, f'SAA Reliability: {saa_reliability}', 
              transform=ax1.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax1.text(0.02, 0.92, f'APUB Reliability: {apub_reliability}', 
              transform=ax1.transAxes, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Reliability distribution plot
    ax2.hist(apub_reliabilities, bins=15, alpha=0.7, color='lightgreen', edgecolor='green')
    ax2.axvline(np.mean(apub_reliabilities), color='red', linestyle='--', linewidth=2, 
                 label=f'Mean: {np.mean(apub_reliabilities):.3f}')
    ax2.set_xlabel("Reliability")
    ax2.set_ylabel("Frequency")
    ax2.set_title("APUB Coverage Probability Distribution (α=0.3)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSAA Summary:")
    print(f"Mean cost: {np.mean(obj_values):.2f}")
    print(f"Std cost: {np.std(obj_values):.2f}")
    
    print(f"\nAPUB Summary (α={alpha}):")
    print(f"Mean out-of-sample cost: {np.mean(apub_costs):.2f}")
    print(f"Std out-of-sample cost: {np.std(apub_costs):.2f}")
    print(f"Mean reliability: {np.mean(apub_reliabilities):.3f}")
    print(f"Std reliability: {np.std(apub_reliabilities):.3f}")

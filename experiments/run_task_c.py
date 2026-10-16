"""
run_task_c.py - Executes Task C: Deep Dive for Student Roll Number R = 05
========================================================================
Assigned Algorithms:
1. Algorithm 5: Value Function
2. Algorithm 17: PESA-II
3. Algorithm 29: MOBA

Generates:
1. Exact 2-iteration hand trace verifications on tiny instances.
2. 3-level parameter sensitivity analysis with 10 independent random seeds.
3. Statistical plots (mean +/- std) and summary CSVs using standard csv module (NO pandas).
"""

import os
import sys
import csv
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('algorithms'))
import moo_utils as mu
import alg05_value_function as a05
import alg17_pesa2 as a17
import alg29_moba as a29


# ==============================================================================
# PART 1: EXACT 2-ITERATION HAND TRACES ON TINY INSTANCES
# ==============================================================================

def run_hand_trace_alg05():
    """
    Hand trace verification for Algorithm 5 (Value Function)
    Tiny instance: P4 Investment Portfolio with 2 risk aversion settings:
    Setting 1: lambda = 1.0 (Risk-tolerant / Aggressive)
    Setting 2: lambda = 10.0 (Risk-averse / Conservative)
    """
    p4 = mu.ProblemP4(use_repair=True)

    trace_log = []
    # Setting 1: lambda = 1.0
    cfg1 = {'utility_type': 'portfolio', 'lambda_risk': 1.0}
    x_opt1, f_opt1, cv1, u_opt1 = a05.step(p4, cfg1, x0=np.array([0.25, 0.25, 0.25, 0.25]))
    trace_log.append({
        'Setting': 'Iteration 1: lambda = 1.0',
        'Weights_w': np.round(x_opt1, 4).tolist(),
        'Risk_f1': round(f_opt1[0], 5),
        'Return_ret': round(-f_opt1[1], 5),
        'Utility_U': round(u_opt1, 5)
    })

    # Setting 2: lambda = 10.0
    cfg2 = {'utility_type': 'portfolio', 'lambda_risk': 10.0}
    x_opt2, f_opt2, cv2, u_opt2 = a05.step(p4, cfg2, x0=np.array([0.25, 0.25, 0.25, 0.25]))
    trace_log.append({
        'Setting': 'Iteration 2: lambda = 10.0',
        'Weights_w': np.round(x_opt2, 4).tolist(),
        'Risk_f1': round(f_opt2[0], 5),
        'Return_ret': round(-f_opt2[1], 5),
        'Utility_U': round(u_opt2, 5)
    })

    print("\n--- Algorithm 5 Hand Trace Verification ---")
    for item in trace_log:
        print(item)
    return trace_log


def run_hand_trace_alg17():
    """
    Hand trace verification for Algorithm 17 (PESA-II)
    Tiny instance: Problem P1 (Factory Production Speed), N = 4, Archive Size = 4, K = 4.
    """
    p1 = mu.ProblemP1()
    # Tiny initial population of 4 points: x = [2.0, 4.0, 6.0, 8.0]
    init_pop = np.array([[2.0], [4.0], [6.0], [8.0]])
    F_list = np.array([p1.evaluate(x)[0] for x in init_pop])
    CV_list = np.zeros(4)

    # Initial non-dominated sort
    fronts = mu.fast_non_dominated_sort(F_list, CV_list)
    arc_X = init_pop[fronts[0]].copy()
    arc_F = F_list[fronts[0]].copy()
    arc_CV = CV_list[fronts[0]].copy()

    _, boxes_dict, squeeze_factors = a17.assign_hyperboxes(arc_F, K=4)

    trace = {
        'Iteration_0': {
            'Initial_X': init_pop.flatten().tolist(),
            'Archive_F': arc_F.tolist(),
            'Squeeze_Factors': squeeze_factors
        }
    }

    # Iteration 1: candidate child at x = 5.0 -> f = [4.0, 4.0]
    cand1 = np.array([5.0])
    f_cand1, cv1 = p1.evaluate(cand1)
    arc_X, arc_F, arc_CV = a17.update_pesa_archive(cand1, f_cand1, cv1, arc_X, arc_F, arc_CV, max_archive_size=4, K=4)
    _, boxes_dict1, squeeze1 = a17.assign_hyperboxes(arc_F, K=4)
    trace['Iteration_1'] = {
        'Candidate_X': cand1.tolist(),
        'Candidate_F': f_cand1.tolist(),
        'Archive_F': arc_F.tolist(),
        'Squeeze_Factors': squeeze1
    }

    # Iteration 2: candidate child at x = 3.5 -> f = [0.25, 12.25]
    cand2 = np.array([3.5])
    f_cand2, cv2 = p1.evaluate(cand2)
    arc_X, arc_F, arc_CV = a17.update_pesa_archive(cand2, f_cand2, cv2, arc_X, arc_F, arc_CV, max_archive_size=4, K=4)
    _, boxes_dict2, squeeze2 = a17.assign_hyperboxes(arc_F, K=4)
    trace['Iteration_2'] = {
        'Candidate_X': cand2.tolist(),
        'Candidate_F': f_cand2.tolist(),
        'Archive_F': arc_F.tolist(),
        'Squeeze_Factors': squeeze2
    }

    print("\n--- Algorithm 17 (PESA-II) Hand Trace Verification ---")
    print("Iter 0 Archive F:\n", np.array(trace['Iteration_0']['Archive_F']))
    print("Iter 1 Added [4, 4] -> Updated Archive F:\n", np.array(trace['Iteration_1']['Archive_F']))
    print("Iter 2 Added [0.25, 12.25] -> Updated Archive F:\n", np.array(trace['Iteration_2']['Archive_F']))
    return trace


def run_hand_trace_alg29():
    """
    Hand trace verification for Algorithm 29 (MOBA)
    Tiny instance: Problem P1, N = 4 bats, 2 iterations with fixed random seed.
    """
    p1 = mu.ProblemP1()
    X = np.array([[2.0], [4.0], [6.0], [8.0]])
    V = np.zeros((4, 1))
    F = np.array([p1.evaluate(x)[0] for x in X])
    CV = np.zeros(4)
    A = np.full(4, 0.9)
    r = np.full(4, 0.1)
    freq = np.zeros(4)

    # Initial archive
    fronts = mu.fast_non_dominated_sort(F, CV)
    arc_X = X[fronts[0]].copy()
    arc_F = F[fronts[0]].copy()
    arc_CV = CV[fronts[0]].copy()

    trace = {'Iter_0': {'X': X.copy(), 'F': F.copy()}}

    # Step 1
    np.random.seed(42)
    X, V, F, CV, A, r, freq, arc_X, arc_F, arc_CV = a29.step(
        X, V, F, CV, A, r, freq, arc_X, arc_F, arc_CV, p1, current_iter=1,
        f_min=0.0, f_max=2.0, alpha=0.95, gamma=0.05, r0=0.1, archive_size=4
    )
    trace['Iter_1'] = {'X': np.round(X.copy(), 4), 'F': np.round(F.copy(), 4)}

    # Step 2
    X, V, F, CV, A, r, freq, arc_X, arc_F, arc_CV = a29.step(
        X, V, F, CV, A, r, freq, arc_X, arc_F, arc_CV, p1, current_iter=2,
        f_min=0.0, f_max=2.0, alpha=0.95, gamma=0.05, r0=0.1, archive_size=4
    )
    trace['Iter_2'] = {'X': np.round(X.copy(), 4), 'F': np.round(F.copy(), 4)}

    print("\n--- Algorithm 29 (MOBA) Hand Trace Verification ---")
    print("Iter 1 Bat Positions:\n", trace['Iter_1']['X'])
    print("Iter 2 Bat Positions:\n", trace['Iter_2']['X'])
    return trace


# ==============================================================================
# PART 2: SENSITIVITY STUDIES (10 SEEDS EACH ACROSS 3 PARAMETER LEVELS)
# ==============================================================================

def run_sensitivity_alg05() -> List[Dict[str, Any]]:
    """
    Sensitivity study for Alg 5 (Value Function) on P4 (Investment Portfolio):
    Parameter: Risk Aversion lambda in {0.5, 5.0, 50.0} across 10 random seeds.
    """
    print("\nRunning Sensitivity Study: Alg 5 (Value Function) on P4...")
    p4 = mu.ProblemP4(use_repair=True)
    lambdas = [0.5, 5.0, 50.0]
    records = []

    for lam in lambdas:
        for seed in range(10):
            cfg = {'utility_type': 'portfolio', 'lambda_risk': lam}
            rng = np.random.default_rng(seed * 10 + 7)
            x0 = rng.uniform(0.1, 1.0, 4)
            x0 = x0 / np.sum(x0)
            x_opt, f_opt, cv, u_opt = a05.step(p4, cfg, x0=x0)
            risk = round(float(f_opt[0]), 5)
            ret = round(float(-f_opt[1]), 5)

            records.append({
                'Algorithm': 'Value_Function',
                'Parameter_Name': 'Risk_Aversion_lambda',
                'Parameter_Value': lam,
                'Seed': seed,
                'Risk': risk,
                'Return': ret,
                'Utility': round(float(u_opt), 5),
                'HV': np.nan,
                'Spacing': np.nan,
                'IGD': np.nan
            })
    return records


def run_sensitivity_alg17() -> List[Dict[str, Any]]:
    """
    Sensitivity study for Alg 17 (PESA-II) on P3 (ZDT2 non-convex):
    Parameter: Grid divisions K in {5, 10, 20} across 10 random seeds.
    """
    print("Running Sensitivity Study: Alg 17 (PESA-II) on P3...")
    p3 = mu.ProblemP3()
    true_F = p3.true_pareto_front(n_points=100)
    K_vals = [5, 10, 20]
    records = []

    for k in K_vals:
        for seed in range(10):
            res = a17.run(p3, archive_size=50, n_gen=30, K=k, seed=seed * 10 + 7)
            front_F = res['F_front']
            hv_m, igd_m, sp_m = mu.compute_metrics(front_F, 'P3')
            hv_val = hv_m if hv_m is not None else np.nan
            igd_val = igd_m if igd_m is not None else np.nan
            spacing_val = sp_m if sp_m is not None else np.nan

            records.append({
                'Algorithm': 'PESA-II',
                'Parameter_Name': 'Grid_Divisions_K',
                'Parameter_Value': k,
                'Seed': seed,
                'Risk': np.nan,
                'Return': np.nan,
                'Utility': np.nan,
                'HV': round(float(hv_val), 4),
                'Spacing': round(float(spacing_val), 4),
                'IGD': round(float(igd_val), 4)
            })
    return records


def run_sensitivity_alg29() -> List[Dict[str, Any]]:
    """
    Sensitivity study for Alg 29 (MOBA) on P3 (ZDT2):
    Parameter: Loudness cooling alpha in {0.80, 0.95, 0.99} across 10 random seeds.
    """
    print("Running Sensitivity Study: Alg 29 (MOBA) on P3...")
    p3 = mu.ProblemP3()
    true_F = p3.true_pareto_front(n_points=100)
    alphas = [0.80, 0.95, 0.99]
    records = []

    for a_val in alphas:
        for seed in range(10):
            res = a29.run(p3, n_bats=50, max_iter=30, alpha=a_val, seed=seed * 10 + 7)
            front_F = res['F_front']
            hv_m, igd_m, sp_m = mu.compute_metrics(front_F, 'P3')
            hv_val = hv_m if hv_m is not None else np.nan
            igd_val = igd_m if igd_m is not None else np.nan
            spacing_val = sp_m if sp_m is not None else np.nan

            records.append({
                'Algorithm': 'MOBA',
                'Parameter_Name': 'Loudness_Cooling_alpha',
                'Parameter_Value': a_val,
                'Seed': seed,
                'Risk': np.nan,
                'Return': np.nan,
                'Utility': np.nan,
                'HV': round(float(hv_val), 4),
                'Spacing': round(float(spacing_val), 4),
                'IGD': round(float(igd_val), 4)
            })
    return records


def main():
    print("=" * 80)
    print("EXECUTING TASK C: DEEP DIVE FOR ROLL NUMBER R = 05")
    print("Assigned Algorithms: 5 (Value Function), 17 (PESA-II), 29 (MOBA)")
    print("=" * 80)

    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)

    # 1. Run hand traces
    run_hand_trace_alg05()
    run_hand_trace_alg17()
    run_hand_trace_alg29()

    # 2. Run sensitivity studies
    rec5 = run_sensitivity_alg05()
    rec17 = run_sensitivity_alg17()
    rec29 = run_sensitivity_alg29()

    all_records = rec5 + rec17 + rec29

    # Save CSV using standard csv module (NO pandas)
    csv_path = 'results/data/task_c_sensitivity.csv'
    fieldnames = ['Algorithm', 'Parameter_Name', 'Parameter_Value', 'Seed', 'Risk', 'Return', 'Utility', 'HV', 'Spacing', 'IGD']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[OK] Task C Sensitivity CSV saved to: {csv_path}")

    # 3. Generate Sensitivity Plots using Matplotlib
    # Plot Alg 5: Risk vs Return across lambda
    plt.figure(figsize=(6.5, 4.5))
    for lam, col in zip([0.5, 5.0, 50.0], ['crimson', 'royalblue', 'forestgreen']):
        sub_risks = [r['Risk'] for r in rec5 if r['Parameter_Value'] == lam]
        sub_returns = [r['Return'] for r in rec5 if r['Parameter_Value'] == lam]
        plt.scatter(sub_risks, sub_returns, s=65, color=col, alpha=0.85, edgecolors='k', label=rf"$\lambda = {lam}$")
    plt.title(r"Alg 05 (Value Function): Risk vs Return across $\lambda_{\mathrm{risk}}$ (10 seeds)")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Return")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/figures/task_c_sensitivity_alg05.png', dpi=200)
    plt.close()

    # Plot Alg 17: PESA-II IGD across K
    plt.figure(figsize=(6.5, 4.5))
    K_vals = [5, 10, 20]
    means_igd = []
    stds_igd = []
    for k in K_vals:
        igds = [r['IGD'] for r in rec17 if r['Parameter_Value'] == k]
        means_igd.append(np.mean(igds))
        stds_igd.append(np.std(igds))
    plt.errorbar(K_vals, means_igd, yerr=stds_igd, fmt='-o', color='royalblue', capsize=5, lw=2, markersize=7)
    plt.title(r"Alg 17 (PESA-II): IGD vs Grid Divisions $K$ on P3 (Mean $\pm$ Std, 10 seeds)")
    plt.xlabel(r"Grid Divisions per Objective ($K$)")
    plt.ylabel("Inverted Generational Distance (IGD)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/figures/task_c_sensitivity_alg17.png', dpi=200)
    plt.close()

    # Plot Alg 29: MOBA HV across alpha
    plt.figure(figsize=(6.5, 4.5))
    alpha_vals = [0.80, 0.95, 0.99]
    means_hv = []
    stds_hv = []
    for a in alpha_vals:
        hvs = [r['HV'] for r in rec29 if r['Parameter_Value'] == a]
        means_hv.append(np.mean(hvs))
        stds_hv.append(np.std(hvs))
    plt.errorbar(alpha_vals, means_hv, yerr=stds_hv, fmt='-s', color='forestgreen', capsize=5, lw=2, markersize=7)
    plt.title(r"Alg 29 (MOBA): Hypervolume vs Loudness Cooling $\alpha$ on P3 (10 seeds)")
    plt.xlabel(r"Loudness Cooling Factor $\alpha$")
    plt.ylabel("Hypervolume (HV)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('results/figures/task_c_sensitivity_alg29.png', dpi=200)
    plt.close()

    print("Task C figures and tables generated successfully!")


if __name__ == "__main__":
    main()

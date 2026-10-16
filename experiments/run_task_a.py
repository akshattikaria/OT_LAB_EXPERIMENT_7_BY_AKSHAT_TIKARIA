"""
run_task_a.py - Executes Task A: Runs all 37 algorithms on their test problems
==============================================================================
Generates:
1. Pareto front plots / visualizations for ALL 37 algorithms.
2. Performance metrics: HV, Spacing, IGD (where true front known), runtime.
3. Implementation Log: results/data/task_a_log.csv using standard csv module (NO pandas).
"""

import os
import sys
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('algorithms'))
import moo_utils as mu

# Import all 37 algorithms
import alg01_weighted_sum as a01
import alg02_epsilon_constraint as a02
import alg03_benson as a03
import alg04_weight_metric as a04
import alg05_value_function as a05
import alg06_goal_programming as a06
import alg07_lexicographic as a07
import alg08_topsis as a08
import alg09_vikor as a09
import alg10_vega as a10
import alg11_moga as a11
import alg12_nsga as a12
import alg13_npga as a13
import alg14_nsga2 as a14
import alg15_nsga3 as a15
import alg16_spea2 as a16
import alg17_pesa2 as a17
import alg18_moead as a18
import alg19_moead_de as a19
import alg20_moead_aco as a20
import alg21_constrained_nsga2 as a21
import alg22_mopso as a22
import alg23_mogwo as a23
import alg24_moabc as a24
import alg25_moaco as a25
import alg26_mosa as a26
import alg27_mode as a27
import alg28_motlbo as a28
import alg29_moba as a29
import alg30_moma as a30
import alg31_rvea as a31
import alg32_ibea as a32
import alg33_sms_emoa as a33
import alg34_hype as a34
import alg35_r_nsga2 as a35
import alg36_mgda as a36
import alg37_gradnorm as a37


def get_problem_instance(p_name: str):
    """Instantiates benchmark problem by name."""
    if p_name == "P1": return mu.ProblemP1()
    elif p_name == "P2": return mu.ProblemP2()
    elif p_name == "P3": return mu.ProblemP3()
    elif p_name == "P4": return mu.ProblemP4()
    elif p_name == "P5": return mu.ProblemP5()
    elif p_name == "P6": return mu.ProblemP6()
    elif p_name == "P7": return mu.ProblemP7()
    elif p_name == "P8": return mu.ProblemP8()
    elif p_name == "P9": return mu.ProblemP9()
    elif p_name == "P10": return mu.ProblemP10(M=3)
    elif p_name == "P10_M5": return mu.ProblemP10(M=5)
    elif p_name == "P11": return mu.ProblemP11()
    elif p_name == "P12": return mu.ProblemP12()
    elif p_name == "P13": return mu.ProblemP13()
    elif p_name == "P14": return mu.ProblemP14()
    else: raise ValueError(f"Unknown problem {p_name}")


# Mapping of each algorithm to its assigned test problems from Catalogue
CATALOGUE_PROBLEMS = {
    1: ("Weighted Sum", a01, ["P1", "P3"]),
    2: ("Epsilon-Constraint", a02, ["P2", "P3"]),
    3: ("Benson's Method", a03, ["P1", "P7"]),
    4: ("Weight Metric", a04, ["P1", "P3"]),
    5: ("Value Function", a05, ["P4", "P14"]),
    6: ("Goal Programming", a06, ["P11", "P7"]),
    7: ("Lexicographic", a07, ["P11", "P7"]),
    8: ("TOPSIS", a08, ["P12"]),
    9: ("VIKOR", a09, ["P12"]),
    10: ("VEGA", a10, ["P2", "P3"]),
    11: ("MOGA", a11, ["P3", "P5"]),
    12: ("NSGA", a12, ["P5", "P7"]),
    13: ("NPGA", a13, ["P3", "P5"]),
    14: ("NSGA-II", a14, ["P3", "P5"]),
    15: ("NSGA-III", a15, ["P10"]),
    16: ("SPEA2", a16, ["P4", "P5"]),
    17: ("PESA-II", a17, ["P3", "P5"]),
    18: ("MOEA/D", a18, ["P3", "P10"]),
    19: ("MOEA/D-DE", a19, ["P3", "P5"]),
    20: ("MOEA/D-ACO", a20, ["P8"]),
    21: ("Constrained NSGA-II", a21, ["P5", "P6"]),
    22: ("MOPSO", a22, ["P4", "P5"]),
    23: ("MOGWO", a23, ["P5", "P9"]),
    24: ("MOABC", a24, ["P3", "P6"]),
    25: ("MOACO", a25, ["P8"]),
    26: ("MOSA", a26, ["P3", "P5"]),
    27: ("MODE", a27, ["P3", "P5"]),
    28: ("MO-TLBO", a28, ["P3", "P5"]),
    29: ("MOBA", a29, ["P3", "P4"]),
    30: ("MOMA", a30, ["P5", "P9"]),
    31: ("RVEA", a31, ["P10"]),
    32: ("IBEA", a32, ["P3", "P14"]),
    33: ("SMS-EMOA", a33, ["P3", "P14"]),
    34: ("HypE", a34, ["P10"]),
    35: ("R-NSGA-II", a35, ["P5", "P14"]),
    36: ("MGDA", a36, ["P13"]),
    37: ("GradNorm", a37, ["P13"]),
}


def get_preference_input(alg_num: int, p_name: str) -> str:
    """Returns preference input specification for single-solution and ranking methods."""
    if alg_num == 5:
        return "lambda_risk=5.0" if p_name == 'P4' else "weights=[0.5, 0.5]"
    elif alg_num == 6:
        return "targets=[-2400, -600], weights=[1, 1]" if p_name == 'P11' else "targets=[0, 0], weights=[1, 1]"
    elif alg_num == 7:
        return "priority=[0, 1] (profit > units)" if p_name == 'P11' else "priority=[0, 1]"
    elif alg_num == 8:
        return "weights=[0.35, 0.25, 0.25, 0.15], criteria=[cost, ben, ben, ben]"
    elif alg_num == 9:
        return "weights=[0.35, 0.25, 0.25, 0.15], v=0.5"
    elif alg_num == 36:
        return "equal-weights gradient descent, Frank-Wolfe min-norm"
    elif alg_num == 37:
        return "alpha=1.5, task balancing"
    return "N/A"


def run_single_algorithm(alg_num: int, name: str, module: Any, p_name: str, seed: int = 42) -> Dict[str, Any]:
    """Executes one algorithm on one problem and calculates metrics."""
    prob = get_problem_instance(p_name)
    start_time = time.time()

    from experiments.budget import run_budgeted
    res = run_budgeted(alg_num, module, prob, seed=seed)
    runtime = time.time() - start_time
    front_F = res.get('F_front', np.array([]))
    budget = res.get('evaluations', 0)

    # Compute metrics: single solutions and rankings log N/A
    pref_input = get_preference_input(alg_num, p_name)
    hv_val = np.nan
    spacing_val = np.nan
    igd_val = np.nan

    true_F = prob.true_pareto_front(n_points=100) if hasattr(prob, 'true_pareto_front') else None

    if alg_num in [5, 6, 7, 8, 9, 36, 37]:
        # Single-solution / ranking methods: HV/IGD/Spacing are N/A
        hv_val = np.nan
        igd_val = np.nan
        spacing_val = np.nan
    elif front_F is not None and len(front_F) > 0:
        hv_m, igd_m, sp_m = mu.compute_metrics(front_F, p_name)
        hv_val = hv_m if hv_m is not None else np.nan
        igd_val = igd_m if igd_m is not None else np.nan
        spacing_val = sp_m if sp_m is not None else np.nan

    # Honest quantitative status classification via experiments/status.py
    from experiments.status import classify_status
    use_constraints = True
    if alg_num == 14 and p_name == 'P5':
        use_constraints = False  # intentionally unconstrained test
    status = classify_status(alg_num, p_name, res, igd_val=igd_val, use_constraints=use_constraints)

    return {
        'num': alg_num,
        'algorithm': name,
        'problem': p_name,
        'budget': budget,
        'runtime': runtime,
        'hv': hv_val,
        'spacing': spacing_val,
        'igd': igd_val,
        'status': status,
        'preference_input': pref_input,
        'front_F': front_F,
        'true_F': true_F,
        'raw_res': res
    }


def generate_front_plot(rec: Dict[str, Any]):
    """Generates informative front plots for all algorithms (2D, 3D, MCDM, and Gradient)."""
    alg_num = rec['num']
    name = rec['algorithm']
    p_name = rec['problem']
    fig_path = f"results/figures/front_alg{alg_num:02d}_{p_name}.png"

    # 1. MCDM Ranking Plots (Alg 08 TOPSIS, Alg 09 VIKOR)
    if alg_num == 8:
        raw = rec['raw_res']
        closeness = raw['closeness']
        suppliers = [f"S{i+1}" for i in range(len(closeness))]
        plt.figure(figsize=(6, 4))
        plt.bar(suppliers, closeness, color='steelblue', edgecolor='black', width=0.5)
        plt.title(f"Alg 08: TOPSIS Closeness Coefficient on {p_name}")
        plt.xlabel("Supplier Alternative")
        plt.ylabel("Relative Closeness $C_i$")
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        return

    if alg_num == 9:
        raw = rec['raw_res']
        Q = raw['Q']
        suppliers = [f"S{i+1}" for i in range(len(Q))]
        plt.figure(figsize=(6, 4))
        plt.bar(suppliers, Q, color='coral', edgecolor='black', width=0.5)
        plt.title(f"Alg 09: VIKOR Compromise Score $Q_i$ on {p_name} (Lowest = Best)")
        plt.xlabel("Supplier Alternative")
        plt.ylabel("VIKOR Metric $Q_i$")
        plt.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        return

    # 2. Gradient-based Multi-Task Learning (Alg 36 MGDA, Alg 37 GradNorm on P13)
    if alg_num == 36:
        raw = rec['raw_res']
        hist_x = raw.get('history_x', np.array([]))
        if len(hist_x) > 0:
            plt.figure(figsize=(6, 4.5))
            # Plot Pareto set line: x1 = x2 from -3 to 3
            line_pts = np.linspace(-3, 3, 50)
            plt.plot(line_pts, line_pts, 'k--', label=r'Pareto Set ($x_1 = x_2$)', alpha=0.7)
            plt.plot(hist_x[:, 0], hist_x[:, 1], 'bo-', markersize=4, label='MGDA Trajectory', alpha=0.8)
            plt.scatter(hist_x[0, 0], hist_x[0, 1], c='red', s=80, marker='*', zorder=10, label='Start (0, 5)')
            plt.scatter(hist_x[-1, 0], hist_x[-1, 1], c='green', s=80, marker='X', zorder=10, label=f'End ({hist_x[-1, 0]:.2f}, {hist_x[-1, 1]:.2f})')
            plt.title(f"Alg 36: MGDA Parameter Trajectory on {p_name}")
            plt.xlabel(r"$x_1$")
            plt.ylabel(r"$x_2$")
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close()
            return

    if alg_num == 37:
        raw = rec['raw_res']
        hist_f = raw.get('history_f', np.array([]))
        if len(hist_f) > 0:
            plt.figure(figsize=(6, 4.5))
            iters = np.arange(len(hist_f))
            plt.plot(iters, hist_f[:, 0], 'r-', lw=2, label=r'Task 1 Loss $L_1$')
            plt.plot(iters, hist_f[:, 1], 'b-', lw=2, label=r'Task 2 Loss $L_2$')
            plt.title(f"Alg 37: GradNorm Multi-Task Loss Convergence on {p_name}")
            plt.xlabel("Iteration")
            plt.ylabel("Loss Value")
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close()
            return

    # 3. 3D Many-Objective Scatter (Alg 15 NSGA-III, Alg 18 MOEA/D on P10, Alg 31 RVEA, Alg 34 HypE)
    front_F = rec['front_F']
    if front_F is not None and len(front_F) > 0 and front_F.shape[1] == 3:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(front_F[:, 0], front_F[:, 1], front_F[:, 2], c='crimson', edgecolors='k', s=35, alpha=0.85)
        ax.set_title(f"Alg {alg_num:02d}: {name} on {p_name} (M=3)")
        ax.set_xlabel(r"$f_1$")
        ax.set_ylabel(r"$f_2$")
        ax.set_zlabel(r"$f_3$")
        fig.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        return

    # 4. Standard 2D Pareto Front Plots
    if front_F is not None and len(front_F) > 0 and front_F.shape[1] == 2:
        true_F = rec['true_F']
        plt.figure(figsize=(6, 4.5))
        if true_F is not None and len(true_F) > 0:
            plt.plot(true_F[:, 0], true_F[:, 1], 'k--', label='True Pareto Front', alpha=0.7)
        plt.scatter(front_F[:, 0], front_F[:, 1], c='crimson', edgecolors='k', s=35,
                    label=f'Obtained Front ({name})', zorder=5)
        plt.title(f"Alg {alg_num:02d}: {name} on {p_name}")
        plt.xlabel(r"$f_1$")
        plt.ylabel(r"$f_2$")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        return


def main():
    print("=" * 80)
    print("STARTING TASK A: EXECUTING ALL 37 ALGORITHMS ON CATALOGUE PROBLEMS")
    print("=" * 80)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)

    records = []

    for alg_num, (name, module, prob_list) in CATALOGUE_PROBLEMS.items():
        print(f"\n[Alg {alg_num:02d}] {name} on {prob_list}...")
        for p_name in prob_list:
            rec = run_single_algorithm(alg_num, name, module, p_name, seed=42)
            records.append({
                'Algorithm_Number': rec['num'],
                'Algorithm': rec['algorithm'],
                'Problem': rec['problem'],
                'Evaluations': rec['budget'],
                'HV': round(rec['hv'], 4) if not np.isnan(rec['hv']) else 'N/A',
                'Spacing': round(rec['spacing'], 4) if not np.isnan(rec['spacing']) else 'N/A',
                'IGD': round(rec['igd'], 4) if not np.isnan(rec['igd']) else 'N/A',
                'Runtime_sec': round(rec['runtime'], 3),
                'Status': rec['status'],
                'Preference_Input': rec.get('preference_input', 'N/A')
            })

            # Generate front plot
            generate_front_plot(rec)
            hv_disp = rec['hv'] if not np.isnan(rec['hv']) else 0.0
            print(f"  -> Problem {p_name}: Evaluations={rec['budget']}, Runtime={rec['runtime']:.2f}s, HV={hv_disp:.4f}")

    # Write results/data/task_a_log.csv using standard csv library (NO PANDAS)
    csv_path = 'results/data/task_a_log.csv'
    fieldnames = ['Algorithm_Number', 'Algorithm', 'Problem', 'Evaluations', 'HV', 'Spacing', 'IGD', 'Runtime_sec', 'Status', 'Preference_Input']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print("\n" + "=" * 80)
    print(f"TASK A COMPLETE! Master Implementation Log saved to: {csv_path}")
    print(f"Total benchmark executions recorded: {len(records)}")
    print(f"All 37 algorithm plots generated in results/figures/")
    print("=" * 80)


if __name__ == "__main__":
    main()

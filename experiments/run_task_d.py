"""
run_task_d.py - Executes Task D: Head-to-Head Comparisons
==========================================================
Covers:
1. Continuous Bi-Objective Head-to-Head on P3 (unconstrained) and P5 (constrained).
   (10 seeds each, mean +/- std of HV, Spacing, IGD; combined Pareto front plots)
2. Many-Objective Head-to-Head on P10 (M = 5):
   Compares NSGA-II, NSGA-III, MOEA/D, RVEA, and HypE.
3. Discrete Combinatorial Head-to-Head on P8 (TSP):
   Compares MOEA/D-ACO, MOACO, NSGA-II, and MOMA with permutation operators.
4. Preference, Ranking & Gradient Special Comparisons:
   - TOPSIS and VIKOR on NSGA-II front from P4, plus 0.1 weight shift sensitivity
   - MGDA vs GradNorm vs Equal Weights on P13
All outputs written with standard csv module (ZERO PANDAS).
"""

import os
import sys
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('algorithms'))
import moo_utils as mu

import alg01_weighted_sum as a01
import alg02_epsilon_constraint as a02
import alg04_weight_metric as a04
import alg05_value_function as a05
import alg06_goal_programming as a06
import alg07_lexicographic as a07
import alg08_topsis as a08
import alg09_vikor as a09
import alg10_vega as a10
import alg11_moga as a11
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
from operators import order_crossover_ox, swap_mutation, inversion_mutation
from experiments.budget import run_budgeted


def save_csv(filepath: str, records: List[Dict[str, Any]], fieldnames: List[str]):
    """Helper to write CSV files without pandas."""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# ==============================================================================
# PART 1: CONTINUOUS BI-OBJECTIVE COMPARISON (P3 & P5)
# ==============================================================================

def run_biobjective_comparison(p_name: str = "P3", n_seeds: int = 10):
    """
    Runs continuous bi-objective algorithms on P3 or P5 across n_seeds with equal budget.
    """
    prob = mu.ProblemP3() if p_name == "P3" else mu.ProblemP5()
    true_F = prob.true_pareto_front(n_points=100)

    # Algorithms applicable to continuous bi-objective
    algs = [
        (14, "NSGA-II", a14),
        (16, "SPEA2", a16),
        (17, "PESA-II", a17),
        (18, "MOEA/D", a18),
        (19, "MOEA/D-DE", a19),
        (21, "Constrained NSGA-II", a21),
        (22, "MOPSO", a22),
        (23, "MOGWO", a23),
        (24, "MOABC", a24),
        (27, "MODE", a27),
        (28, "MO-TLBO", a28),
        (29, "MOBA", a29),
        (30, "MOMA", a30),
        (32, "IBEA", a32)
    ]

    records = []
    fronts_dict = {}

    for num, name, mod in algs:
        hv_list, sp_list, igd_list = [], [], []
        best_front = None
        best_hv = -1.0

        for s in range(n_seeds):
            seed = s * 10 + 13
            res = run_budgeted(num, mod, prob, seed=seed)
            front_F = res.get('F_front', np.array([]))

            if len(front_F) > 0:
                hv_m, igd_m, sp_m = mu.compute_metrics(front_F, prob.name)
                hv = hv_m if hv_m is not None else 0.0
                igd = igd_m if igd_m is not None else 999.0
                sp = sp_m if sp_m is not None else 0.0

                hv_list.append(hv)
                sp_list.append(sp)
                igd_list.append(igd)

                if hv > best_hv:
                    best_hv = hv
                    best_front = front_F

        fronts_dict[name] = best_front
        records.append({
            'Problem': p_name,
            'Algorithm': name,
            'HV_Mean': round(float(np.mean(hv_list)), 4) if len(hv_list) > 0 else 0.0,
            'HV_Std': round(float(np.std(hv_list)), 4) if len(hv_list) > 0 else 0.0,
            'Spacing_Mean': round(float(np.mean(sp_list)), 4) if len(sp_list) > 0 else 0.0,
            'Spacing_Std': round(float(np.std(sp_list)), 4) if len(sp_list) > 0 else 0.0,
            'IGD_Mean': round(float(np.mean(igd_list)), 4) if len(igd_list) > 0 else 999.0,
            'IGD_Std': round(float(np.std(igd_list)), 4) if len(igd_list) > 0 else 0.0,
        })

    # Sort descending by HV_Mean
    records.sort(key=lambda r: r['HV_Mean'], reverse=True)
    for idx, r in enumerate(records):
        r['Rank'] = idx + 1

    # Plot Pareto fronts together
    plt.figure(figsize=(8.5, 6))
    if true_F is not None and len(true_F) > 0:
        plt.plot(true_F[:, 0], true_F[:, 1], 'k--', lw=2, label='True Pareto Front', alpha=0.8)

    top_algs = [r['Algorithm'] for r in records[:5]]
    colors = ['crimson', 'blue', 'forestgreen', 'darkorange', 'purple']
    for idx, alg_name in enumerate(top_algs):
        f = fronts_dict.get(alg_name)
        if f is not None and len(f) > 0:
            plt.scatter(f[:, 0], f[:, 1], s=35, color=colors[idx], label=f"Rank {idx+1}: {alg_name}", alpha=0.85)

    plt.title(f"Task D1: Continuous Bi-Objective Head-to-Head Comparison on {p_name} (Top 5 Algorithms)")
    plt.xlabel(r"$f_1$")
    plt.ylabel(r"$f_2$")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(f"results/figures/task_d1_comparison_{p_name}.png", dpi=200)
    plt.close()

    return records


# ==============================================================================
# PART 2: MANY-OBJECTIVE COMPARISON ON P10 (M = 5)
# ==============================================================================

def run_manyobjective_comparison(n_seeds: int = 10):
    """
    On P10 (DTLZ2, M = 5), compares NSGA-II, NSGA-III, MOEA/D, RVEA, and HypE.
    """
    print("\nRunning Task D2: Many-Objective Comparison on P10 (M = 5)...")
    prob = mu.ProblemP10(M=5)
    true_F = prob.true_pareto_front(n_points=200)

    algs = [
        (14, "NSGA-II", a14),
        (15, "NSGA-III", a15),
        (18, "MOEA/D", a18),
        (31, "RVEA", a31),
        (34, "HypE", a34)
    ]

    records = []
    ref_norm = np.ones(5) * 1.5

    for num, name, mod in algs:
        hv_list, igd_list = [], []

        for s in range(n_seeds):
            seed = s * 10 + 17
            res = run_budgeted(num, mod, prob, seed=seed)
            front_F = res.get('F_front', np.array([]))

            if len(front_F) > 0:
                hv_m, igd_m, _ = mu.compute_metrics(front_F, 'P10_M5')
                hv = hv_m if hv_m is not None else 0.0
                igd = igd_m if igd_m is not None else 999.0
                hv_list.append(hv)
                igd_list.append(igd)

        records.append({
            'Algorithm': name,
            'HV_Mean': round(float(np.mean(hv_list)), 4) if len(hv_list) > 0 else 0.0,
            'HV_Std': round(float(np.std(hv_list)), 4) if len(hv_list) > 0 else 0.0,
            'IGD_Mean': round(float(np.mean(igd_list)), 4) if len(igd_list) > 0 else 999.0,
            'IGD_Std': round(float(np.std(igd_list)), 4) if len(igd_list) > 0 else 0.0
        })

    records.sort(key=lambda r: r['HV_Mean'], reverse=True)
    for idx, r in enumerate(records):
        r['Rank'] = idx + 1

    # Plot Many-Objective Bar Chart Comparison
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    alg_names = [r['Algorithm'] for r in records]
    hv_means = [r['HV_Mean'] for r in records]
    hv_stds = [r['HV_Std'] for r in records]
    igd_means = [r['IGD_Mean'] for r in records]
    igd_stds = [r['IGD_Std'] for r in records]

    ax[0].bar(alg_names, hv_means, yerr=hv_stds, capsize=5, color='royalblue', edgecolor='black')
    ax[0].set_title("Hypervolume on P10 (M=5, higher is better)")
    ax[0].set_ylabel("HV (Monte Carlo)")
    ax[0].grid(axis='y', linestyle=':', alpha=0.7)

    ax[1].bar(alg_names, igd_means, yerr=igd_stds, capsize=5, color='coral', edgecolor='black')
    ax[1].set_title("IGD on P10 (M=5, lower is better)")
    ax[1].set_ylabel("Inverted Generational Distance")
    ax[1].grid(axis='y', linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig('results/figures/task_d2_manyobjective_p10_m5.png', dpi=200)
    plt.close()

    return records


# ==============================================================================
# PART 3: DISCRETE COMBINATORIAL COMPARISON ON P8 (TSP)
# ==============================================================================

def run_discrete_comparison(n_seeds: int = 10):
    """
    On P8 (15-city bi-objective TSP), compares MOEA/D-ACO, MOACO, NSGA-II (permutation), and MOMA.
    """
    print("\nRunning Task D3: Discrete Combinatorial Comparison on P8 (TSP)...")
    prob = mu.ProblemP8(seed=7)

    records = []
    fronts_dict = {}

    # 1. MOEA/D-ACO
    hv_list, sp_list = [], []
    best_f = None
    for s in range(n_seeds):
        res = run_budgeted(20, a20, prob, seed=s * 10 + 7)
        front = res['F_front']
        hv_m, _, sp_m = mu.compute_metrics(front, 'P8')
        hv = hv_m if hv_m is not None else 0.0
        sp = sp_m if sp_m is not None else 0.0
        hv_list.append(hv); sp_list.append(sp)
        if best_f is None or hv > np.max(hv_list[:-1] or [0]): best_f = front
    fronts_dict['MOEA/D-ACO'] = best_f
    records.append({
        'Algorithm': 'MOEA/D-ACO',
        'HV_Mean': round(float(np.mean(hv_list)), 4),
        'HV_Std': round(float(np.std(hv_list)), 4),
        'Spacing_Mean': round(float(np.mean(sp_list)), 4),
        'Spacing_Std': round(float(np.std(sp_list)), 4)
    })

    # 2. MOACO
    hv_list, sp_list = [], []
    best_f = None
    for s in range(n_seeds):
        res = run_budgeted(25, a25, prob, seed=s * 10 + 7)
        front = res['F_front']
        hv_m, _, sp_m = mu.compute_metrics(front, 'P8')
        hv = hv_m if hv_m is not None else 0.0
        sp = sp_m if sp_m is not None else 0.0
        hv_list.append(hv); sp_list.append(sp)
        if best_f is None or hv > np.max(hv_list[:-1] or [0]): best_f = front
    fronts_dict['MOACO'] = best_f
    records.append({
        'Algorithm': 'MOACO',
        'HV_Mean': round(float(np.mean(hv_list)), 4),
        'HV_Std': round(float(np.std(hv_list)), 4),
        'Spacing_Mean': round(float(np.mean(sp_list)), 4),
        'Spacing_Std': round(float(np.std(sp_list)), 4)
    })

    # 3. Permutation NSGA-II (coded with OX and swap mutation)
    hv_list, sp_list = [], []
    best_f = None
    for s in range(n_seeds):
        rng = np.random.default_rng(s * 10 + 7)
        N = 20
        pop = [rng.permutation(15) for _ in range(N)]
        F = np.array([prob.evaluate(p)[0] for p in pop])
        CV = np.zeros(N)

        for gen in range(10):
            offspring = []
            for _ in range(0, N, 2):
                p1 = pop[np.random.randint(N)]
                p2 = pop[np.random.randint(N)]
                c1, c2 = order_crossover_ox(p1, p2)
                c1 = inversion_mutation(c1, pm=0.2)
                c2 = swap_mutation(c2, pm=0.2)
                offspring.extend([c1, c2])

            comb_pop = np.vstack([pop, offspring[:N]])
            comb_F = np.array([prob.evaluate(x)[0] for x in comb_pop])
            comb_CV = np.zeros(len(comb_pop))
            fronts = mu.fast_non_dominated_sort(comb_F, comb_CV)
            new_idx = []
            for f in fronts:
                if len(new_idx) + len(f) <= N: new_idx.extend(f)
                else:
                    cd = mu.calculate_crowding_distance(comb_F, f)
                    new_idx.extend([f[i] for i in np.argsort(-cd)[:(N - len(new_idx))]])
                    break
            pop = comb_pop[new_idx]
            F = comb_F[new_idx]

        fronts = mu.fast_non_dominated_sort(F, CV)
        front = F[fronts[0]]
        hv_m, _, sp_m = mu.compute_metrics(front, 'P8')
        hv = hv_m if hv_m is not None else 0.0
        sp = sp_m if sp_m is not None else 0.0
        hv_list.append(hv); sp_list.append(sp)
        if best_f is None or hv > np.max(hv_list[:-1] or [0]): best_f = front
    fronts_dict['Permutation NSGA-II'] = best_f
    records.append({
        'Algorithm': 'Permutation NSGA-II',
        'HV_Mean': round(float(np.mean(hv_list)), 4),
        'HV_Std': round(float(np.std(hv_list)), 4),
        'Spacing_Mean': round(float(np.mean(sp_list)), 4),
        'Spacing_Std': round(float(np.std(sp_list)), 4)
    })

    # 4. MOMA with 2-opt Local Search
    hv_list, sp_list = [], []
    best_f = None
    for s in range(n_seeds):
        rng = np.random.default_rng(s * 10 + 7)
        N = 20
        pop = [rng.permutation(15) for _ in range(N)]
        F = np.array([prob.evaluate(p)[0] for p in pop])

        for gen in range(10):
            offspring = []
            for _ in range(0, N, 2):
                p1 = pop[np.random.randint(N)]
                p2 = pop[np.random.randint(N)]
                c1, c2 = order_crossover_ox(p1, p2)
                # 2-opt local search hill climbing
                for _ in range(2):
                    i, j = sorted(np.random.choice(15, size=2, replace=False))
                    cand1 = c1.copy(); cand1[i:j+1] = cand1[i:j+1][::-1]
                    if prob.evaluate(cand1)[0][0] < prob.evaluate(c1)[0][0]: c1 = cand1
                offspring.extend([c1, c2])

            comb_pop = np.vstack([pop, offspring[:N]])
            comb_F = np.array([prob.evaluate(x)[0] for x in comb_pop])
            fronts = mu.fast_non_dominated_sort(comb_F, np.zeros(len(comb_pop)))
            new_idx = []
            for f in fronts:
                if len(new_idx) + len(f) <= N: new_idx.extend(f)
                else:
                    cd = mu.calculate_crowding_distance(comb_F, f)
                    new_idx.extend([f[i] for i in np.argsort(-cd)[:(N - len(new_idx))]])
                    break
            pop = comb_pop[new_idx]
            F = comb_F[new_idx]

        fronts = mu.fast_non_dominated_sort(F, np.zeros(N))
        front = F[fronts[0]]
        hv_m, _, sp_m = mu.compute_metrics(front, 'P8')
        hv = hv_m if hv_m is not None else 0.0
        sp = sp_m if sp_m is not None else 0.0
        hv_list.append(hv); sp_list.append(sp)
        if best_f is None or hv > np.max(hv_list[:-1] or [0]): best_f = front
    fronts_dict['MOMA (2-opt)'] = best_f
    records.append({
        'Algorithm': 'MOMA (2-opt)',
        'HV_Mean': round(float(np.mean(hv_list)), 4),
        'HV_Std': round(float(np.std(hv_list)), 4),
        'Spacing_Mean': round(float(np.mean(sp_list)), 4),
        'Spacing_Std': round(float(np.std(sp_list)), 4)
    })

    records.sort(key=lambda r: r['HV_Mean'], reverse=True)
    for idx, r in enumerate(records):
        r['Rank'] = idx + 1

    # Plot Discrete Fronts
    plt.figure(figsize=(7.5, 5))
    colors = ['crimson', 'royalblue', 'forestgreen', 'darkmagenta']
    for idx, (name, front) in enumerate(fronts_dict.items()):
        if front is not None and len(front) > 0:
            plt.scatter(front[:, 0], front[:, 1], s=45, color=colors[idx % len(colors)], label=name, edgecolors='k')

    plt.title("Task D3: Bi-Objective Delivery Routing (P8 TSP) Pareto Front Comparison")
    plt.xlabel("Total Tour Distance")
    plt.ylabel("Total Travel Time")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/figures/task_d3_discrete_tsp.png', dpi=200)
    plt.close()

    return records


# ==============================================================================
# PART 4: TOPSIS & VIKOR ON P4 NSGA-II FRONT & WEIGHT SENSITIVITY
# ==============================================================================

def run_mcdm_comparison():
    """
    Applies TOPSIS and VIKOR to the Pareto front obtained by NSGA-II on P4.
    Reports top 3 solutions under each, shifts weight by 0.1, checks if ranking flips.
    """
    print("\nRunning Task D4: TOPSIS & VIKOR on P4 Portfolio Front...")
    p4 = mu.ProblemP4()
    res_nsga2 = a14.run(p4, pop_size=30, n_gen=20, seed=42)
    front_X = res_nsga2['X_front']
    front_F = res_nsga2['F_front']

    # Criteria: Risk (min) and Return (max = -f2)
    risk = front_F[:, 0]
    ret = -front_F[:, 1]
    decision_matrix = np.column_stack([risk, ret])
    weights = np.array([0.5, 0.5])
    is_beneficial = np.array([False, True]) # Risk is Min, Return is Max

    # 1. Base TOPSIS
    res_topsis = a08.run(decision_matrix, weights, is_beneficial)
    top3_topsis = res_topsis['rankings'][:3]

    # Shift weight by +0.1 to risk
    sens_topsis = a08.step(a08.initialize(decision_matrix, weights, is_beneficial), weight_delta_idx=0, delta=0.1)
    top3_topsis_shifted = sens_topsis['ranking_perturbed'][:3]
    topsis_flip = bool(top3_topsis[0] != top3_topsis_shifted[0])

    # 2. Base VIKOR
    res_vikor = a09.run(decision_matrix, weights, is_beneficial, v_list=[0.25, 0.5, 0.75])
    top3_vikor = res_vikor['rankings'][:3]

    # Shift weight by +0.1 in VIKOR
    cfg_vikor = a09.initialize(decision_matrix, np.array([0.6, 0.4]), is_beneficial, v=0.5)
    _, _, _, vikor_shifted_ranks = a09.evaluate(cfg_vikor)
    vikor_flip = bool(top3_vikor[0] != vikor_shifted_ranks[0])

    mcdm_results = [{
        'TOPSIS_Top3': str(top3_topsis.tolist()),
        'TOPSIS_Shifted_Top3': str(top3_topsis_shifted.tolist()),
        'TOPSIS_Rank_Flipped': topsis_flip,
        'VIKOR_Top3': str(top3_vikor.tolist()),
        'VIKOR_Shifted_Top3': str(vikor_shifted_ranks[:3].tolist()),
        'VIKOR_Rank_Flipped': vikor_flip,
        'Condition_1_Advantage': res_vikor['condition_1_acceptable_advantage'],
        'Condition_2_Stability': res_vikor['condition_2_acceptable_stability']
    }]

    print("MCDM Results on P4 Front:")
    print("TOPSIS Top 3:", top3_topsis, "Shifted (+0.1):", top3_topsis_shifted, "Flip:", topsis_flip)
    print("VIKOR Top 3:", top3_vikor, "Shifted (+0.1):", vikor_shifted_ranks[:3], "Flip:", vikor_flip)
    return mcdm_results


# ==============================================================================
# PART 5: GRADIENT MULTI-TASK LEARNING COMPARISON ON P13
# ==============================================================================

def run_gradient_comparison():
    """
    Compares MGDA vs GradNorm vs Equal Fixed Weights on P13.
    """
    print("\nRunning Task D5: MGDA vs GradNorm vs Equal Weights on P13...")
    p13 = mu.ProblemP13()

    # 1. MGDA
    res_mgda = a36.run(p13, max_iter=80, lr=0.1)

    # 2. GradNorm
    res_gradnorm = a37.run(p13, max_iter=80, lr_model=0.1, lr_weights=0.03)

    # 3. Fixed Equal Weights Gradient Descent
    x_eq = np.array([0.0, 5.0])
    hist_f_eq = [p13.evaluate(x_eq)[0]]
    for _ in range(80):
        g1, g2 = p13.gradients(x_eq)
        g_comb = 0.5 * g1 + 0.5 * g2
        x_eq = x_eq - 0.1 * g_comb
        hist_f_eq.append(p13.evaluate(x_eq)[0])
    hist_f_eq = np.array(hist_f_eq)

    # Plot Loss Curves
    plt.figure(figsize=(7, 4.5))
    plt.plot(np.sum(res_mgda['history_f'], axis=1), 'b-', lw=2, label='MGDA Total Loss')
    plt.plot(np.sum(res_gradnorm['history_f'], axis=1), 'g--', lw=2, label='GradNorm Total Loss')
    plt.plot(np.sum(hist_f_eq, axis=1), 'r:', lw=2, label='Equal Weights Total Loss')
    plt.title("Task D5: Multi-Task Learning on P13 (Total Task Loss Convergence)")
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss ($f_1 + f_2$)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/figures/task_d5_mtl_loss.png', dpi=200)
    plt.close()

    return [{
        'MGDA_Final_Loss_Task1': round(float(res_mgda['history_f'][-1][0]), 4),
        'MGDA_Final_Loss_Task2': round(float(res_mgda['history_f'][-1][1]), 4),
        'GradNorm_Final_Loss_Task1': round(float(res_gradnorm['history_f'][-1][0]), 4),
        'GradNorm_Final_Loss_Task2': round(float(res_gradnorm['history_f'][-1][1]), 4),
        'EqualWeights_Final_Loss_Task1': round(float(hist_f_eq[-1][0]), 4),
        'EqualWeights_Final_Loss_Task2': round(float(hist_f_eq[-1][1]), 4)
    }]


def main():
    print("=" * 80)
    print("EXECUTING TASK D: HEAD-TO-HEAD COMPARISONS (ALL 5 PARTS)")
    print("=" * 80)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)

    # D1: Bi-objective on P3 and P5
    rec_p3 = run_biobjective_comparison("P3", n_seeds=10)
    save_csv('results/data/task_d1_p3_ranked.csv', rec_p3,
             ['Rank', 'Problem', 'Algorithm', 'HV_Mean', 'HV_Std', 'Spacing_Mean', 'Spacing_Std', 'IGD_Mean', 'IGD_Std'])

    rec_p5 = run_biobjective_comparison("P5", n_seeds=10)
    save_csv('results/data/task_d1_p5_ranked.csv', rec_p5,
             ['Rank', 'Problem', 'Algorithm', 'HV_Mean', 'HV_Std', 'Spacing_Mean', 'Spacing_Std', 'IGD_Mean', 'IGD_Std'])

    # D2: Many-objective on P10 M=5
    rec_m5 = run_manyobjective_comparison(n_seeds=10)
    save_csv('results/data/task_d2_p10_m5_ranked.csv', rec_m5,
             ['Rank', 'Algorithm', 'HV_Mean', 'HV_Std', 'IGD_Mean', 'IGD_Std'])

    # D3: Discrete on P8 TSP
    rec_p8 = run_discrete_comparison(n_seeds=10)
    save_csv('results/data/task_d3_p8_tsp_ranked.csv', rec_p8,
             ['Rank', 'Algorithm', 'HV_Mean', 'HV_Std', 'Spacing_Mean', 'Spacing_Std'])

    # D4: MCDM on P4 front
    mcdm_res = run_mcdm_comparison()
    save_csv('results/data/task_d4_mcdm.csv', mcdm_res, list(mcdm_res[0].keys()))

    # D5: Gradient MTL on P13
    mtl_res = run_gradient_comparison()
    save_csv('results/data/task_d5_mtl.csv', mtl_res, list(mtl_res[0].keys()))

    print("\n" + "=" * 80)
    print("ALL TASK D EXPERIMENTS COMPLETED AND RAW CSVs WRITTEN TO results/data/!")
    print("=" * 80)


if __name__ == "__main__":
    main()

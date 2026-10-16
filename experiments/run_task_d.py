"""
run_task_d.py - Executes Task D: Head-to-Head Comparisons
==========================================================
Covers:
1. Continuous Bi-Objective Head-to-Head on P3 (unconstrained) and P5 (constrained).
   (10 seeds each, 10,000 evals, mean +/- std of HV, Spacing, IGD; combined Pareto front plots).
   On P5: plain NSGA-II (use_constraints=False) and Constrained NSGA-II (alg21) tested separately.
2. Many-Objective Head-to-Head on P10 (M = 5):
   Compares NSGA-II, NSGA-III, MOEA/D, RVEA, and HypE (10 seeds, 10,000 evals).
3. Discrete Combinatorial Head-to-Head on P8 (TSP):
   Compares MOEA/D-ACO, MOACO, Permutation NSGA-II, and MOMA (2-opt) (10 seeds, 10,000 evals).
4. Preference, Ranking & Gradient Special Comparisons:
   - TOPSIS and VIKOR on NSGA-II front from P4, plus 0.1 weight shift sensitivity
   - MGDA vs GradNorm vs Equal Weights on P13

Multiprocessing: Windows-compatible multiprocessing.Pool(os.cpu_count()-1)
under if __name__ == "__main__":.
Jobs are (algorithm, problem, seed); each result is written to results/data/raw/
immediately, skipping existing completed seed files.
All outputs written with standard csv module (ZERO PANDAS).
"""

import os
import sys
import time
import json
import csv
import argparse
import multiprocessing
from typing import Dict, List, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('algorithms'))
import moo_utils as mu

import alg08_topsis as a08
import alg09_vikor as a09
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
import alg34_hype as a34
import alg36_mgda as a36
import alg37_gradnorm as a37
from operators import order_crossover_ox, swap_mutation, inversion_mutation
from experiments.budget import run_budgeted


ALGORITHM_MODULES = {
    14: a14, 15: a15, 16: a16, 17: a17, 18: a18, 19: a19,
    20: a20, 21: a21, 22: a22, 23: a23, 24: a24, 25: a25,
    26: a26, 27: a27, 28: a28, 29: a29, 30: a30, 31: a31,
    32: a32, 34: a34, 36: a36, 37: a37
}


def save_csv(filepath: str, records: List[Dict[str, Any]], fieldnames: List[str]):
    """Helper to write CSV files without pandas."""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_perm_nsga2(prob, seed: int = 42, budget: int = 10000) -> np.ndarray:
    """Permutation NSGA-II on P8 with Order Crossover (OX) and mutation."""
    rng = np.random.default_rng(seed)
    N = 100
    n_gen = max(1, budget // N)
    pop = [rng.permutation(15) for _ in range(N)]
    F = np.array([prob.evaluate(p)[0] for p in pop])
    CV = np.zeros(N)

    for gen in range(n_gen):
        offspring = []
        for _ in range(0, N, 2):
            p1 = pop[rng.integers(N)]
            p2 = pop[rng.integers(N)]
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
            if len(new_idx) + len(f) <= N:
                new_idx.extend(f)
            else:
                cd = mu.calculate_crowding_distance(comb_F, f)
                new_idx.extend([f[i] for i in np.argsort(-cd)[:(N - len(new_idx))]])
                break
        pop = comb_pop[new_idx]
        F = comb_F[new_idx]

    fronts = mu.fast_non_dominated_sort(F, CV)
    return F[fronts[0]]


def run_moma_2opt(prob, seed: int = 42, budget: int = 10000) -> np.ndarray:
    """Multi-Objective Memetic Algorithm with 2-opt local search on P8."""
    rng = np.random.default_rng(seed)
    N = 50
    n_gen = max(1, budget // (3 * N))
    pop = [rng.permutation(15) for _ in range(N)]
    F = np.array([prob.evaluate(p)[0] for p in pop])

    for gen in range(n_gen):
        offspring = []
        for _ in range(0, N, 2):
            p1 = pop[rng.integers(N)]
            p2 = pop[rng.integers(N)]
            c1, c2 = order_crossover_ox(p1, p2)
            for _ in range(2):
                i, j = sorted(rng.choice(15, size=2, replace=False))
                cand1 = c1.copy()
                cand1[i:j+1] = cand1[i:j+1][::-1]
                if prob.evaluate(cand1)[0][0] < prob.evaluate(c1)[0][0]:
                    c1 = cand1
            offspring.extend([c1, c2])

        comb_pop = np.vstack([pop, offspring[:N]])
        comb_F = np.array([prob.evaluate(x)[0] for x in comb_pop])
        fronts = mu.fast_non_dominated_sort(comb_F, np.zeros(len(comb_pop)))
        new_idx = []
        for f in fronts:
            if len(new_idx) + len(f) <= N:
                new_idx.extend(f)
            else:
                cd = mu.calculate_crowding_distance(comb_F, f)
                new_idx.extend([f[i] for i in np.argsort(-cd)[:(N - len(new_idx))]])
                break
        pop = comb_pop[new_idx]
        F = comb_F[new_idx]

    fronts = mu.fast_non_dominated_sort(F, np.zeros(N))
    return F[fronts[0]]


# ==============================================================================
# MULTIPROCESSING WORKERS (PICKLABLE TOP-LEVEL FUNCTIONS)
# ==============================================================================

def _worker_biobjective_job(args: Tuple[int, str, str, int, int, str]) -> Dict[str, Any]:
    """Worker executing a single bi-objective job on P3 or P5."""
    alg_num, alg_name, p_name, seed, budget, raw_dir = args
    cache_name = f"d1_{alg_name}_{p_name}_seed{seed}.json".replace(" ", "_").replace("/", "_")
    cache_path = os.path.join(raw_dir, cache_name)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    prob = mu.ProblemP3() if p_name == "P3" else mu.ProblemP5()
    mod = ALGORITHM_MODULES[alg_num]
    kwargs = {}
    if alg_num == 14 and p_name == "P5":
        kwargs['use_constraints'] = False

    res = run_budgeted(alg_num, mod, prob, seed=seed, budget=budget, **kwargs)
    front_F = res.get('F_front', np.array([]))

    hv, sp, igd = 0.0, 0.0, 999.0
    if front_F is not None and len(front_F) > 0:
        hv_m, igd_m, sp_m = mu.compute_metrics(front_F, prob.name)
        hv = hv_m if hv_m is not None else 0.0
        igd = igd_m if igd_m is not None else 999.0
        sp = sp_m if sp_m is not None else 0.0

    result = {
        'problem': p_name,
        'algorithm': alg_name,
        'seed': seed,
        'hv': float(hv),
        'spacing': float(sp),
        'igd': float(igd),
        'evaluations': int(res.get('evaluations', budget)),
        'front_F': front_F.tolist() if front_F is not None and len(front_F) > 0 else []
    }

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(result, f)

    return result


def _worker_manyobjective_job(args: Tuple[int, str, int, int, str]) -> Dict[str, Any]:
    """Worker executing a single many-objective job on P10 M=5."""
    alg_num, alg_name, seed, budget, raw_dir = args
    cache_name = f"d2_{alg_name}_P10_M5_seed{seed}.json".replace(" ", "_").replace("/", "_")
    cache_path = os.path.join(raw_dir, cache_name)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    prob = mu.ProblemP10(M=5)
    mod = ALGORITHM_MODULES[alg_num]
    res = run_budgeted(alg_num, mod, prob, seed=seed, budget=budget)
    front_F = res.get('F_front', np.array([]))

    hv, igd = 0.0, 999.0
    if front_F is not None and len(front_F) > 0:
        hv_m, igd_m, _ = mu.compute_metrics(front_F, 'P10_M5')
        hv = hv_m if hv_m is not None else 0.0
        igd = igd_m if igd_m is not None else 999.0

    result = {
        'problem': 'P10_M5',
        'algorithm': alg_name,
        'seed': seed,
        'hv': float(hv),
        'igd': float(igd),
        'evaluations': int(res.get('evaluations', budget)),
        'front_F': front_F.tolist() if front_F is not None and len(front_F) > 0 else []
    }

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(result, f)

    return result


def _worker_discrete_job(args: Tuple[Any, str, int, int, str]) -> Dict[str, Any]:
    """Worker executing a single discrete combinatorial job on P8."""
    alg_key, alg_name, seed, budget, raw_dir = args
    cache_name = f"d3_{alg_name}_P8_seed{seed}.json".replace(" ", "_").replace("/", "_")
    cache_path = os.path.join(raw_dir, cache_name)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass

    prob = mu.ProblemP8(seed=7)
    evals = budget

    if alg_key == 20:
        res = run_budgeted(20, a20, prob, seed=seed, budget=budget)
        front_F = res.get('F_front', np.array([]))
        evals = res.get('evaluations', budget)
    elif alg_key == 25:
        res = run_budgeted(25, a25, prob, seed=seed, budget=budget)
        front_F = res.get('F_front', np.array([]))
        evals = res.get('evaluations', budget)
    elif alg_key == 'perm_nsga2':
        front_F = run_perm_nsga2(prob, seed=seed, budget=budget)
    elif alg_key == 'moma_2opt':
        front_F = run_moma_2opt(prob, seed=seed, budget=budget)
    else:
        raise ValueError(f"Unknown discrete alg {alg_key}")

    hv, sp = 0.0, 0.0
    if front_F is not None and len(front_F) > 0:
        hv_m, _, sp_m = mu.compute_metrics(front_F, 'P8')
        hv = hv_m if hv_m is not None else 0.0
        sp = sp_m if sp_m is not None else 0.0

    result = {
        'problem': 'P8',
        'algorithm': alg_name,
        'seed': seed,
        'hv': float(hv),
        'spacing': float(sp),
        'evaluations': int(evals),
        'front_F': front_F.tolist() if front_F is not None and len(front_F) > 0 else []
    }

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(result, f)

    return result


# ==============================================================================
# PART 1: CONTINUOUS BI-OBJECTIVE COMPARISON (P3 & P5)
# ==============================================================================

def run_biobjective_comparison(p_name: str = "P3", n_seeds: int = 10, budget: int = 10000, pool=None):
    """
    Runs continuous bi-objective algorithms on P3 or P5 across n_seeds with equal budget.
    """
    prob = mu.ProblemP3() if p_name == "P3" else mu.ProblemP5()
    true_F = prob.true_pareto_front(n_points=100)
    raw_dir = 'results/data/raw'
    os.makedirs(raw_dir, exist_ok=True)

    algs = [
        (14, "NSGA-II"),
        (16, "SPEA2"),
        (17, "PESA-II"),
        (18, "MOEA/D"),
        (19, "MOEA/D-DE"),
        (21, "Constrained NSGA-II"),
        (22, "MOPSO"),
        (23, "MOGWO"),
        (24, "MOABC"),
        (27, "MODE"),
        (28, "MO-TLBO"),
        (29, "MOBA"),
        (30, "MOMA"),
        (32, "IBEA")
    ]

    job_args = []
    for num, name in algs:
        for s in range(n_seeds):
            seed = s * 10 + 13
            job_args.append((num, name, p_name, seed, budget, raw_dir))

    if pool is not None:
        raw_results = pool.map(_worker_biobjective_job, job_args)
    else:
        raw_results = [_worker_biobjective_job(arg) for arg in job_args]

    # Group by algorithm
    grouped = {}
    fronts_dict = {}
    for r in raw_results:
        alg_name = r['algorithm']
        if alg_name not in grouped:
            grouped[alg_name] = {'hv': [], 'sp': [], 'igd': [], 'best_hv': -1.0, 'best_front': None}
        grouped[alg_name]['hv'].append(r['hv'])
        grouped[alg_name]['sp'].append(r['spacing'])
        grouped[alg_name]['igd'].append(r['igd'])
        if r['hv'] > grouped[alg_name]['best_hv']:
            grouped[alg_name]['best_hv'] = r['hv']
            grouped[alg_name]['best_front'] = np.array(r['front_F'])

    records = []
    for num, name in algs:
        g = grouped.get(name, {'hv': [], 'sp': [], 'igd': [], 'best_front': None})
        fronts_dict[name] = g['best_front']
        hv_arr = g['hv']
        sp_arr = g['sp']
        igd_arr = g['igd']

        records.append({
            'Problem': p_name,
            'Algorithm': name,
            'HV_Mean': round(float(np.mean(hv_arr)), 4) if len(hv_arr) > 0 else 0.0,
            'HV_Std': round(float(np.std(hv_arr)), 4) if len(hv_arr) > 0 else 0.0,
            'Spacing_Mean': round(float(np.mean(sp_arr)), 4) if len(sp_arr) > 0 else 0.0,
            'Spacing_Std': round(float(np.std(sp_arr)), 4) if len(sp_arr) > 0 else 0.0,
            'IGD_Mean': round(float(np.mean(igd_arr)), 4) if len(igd_arr) > 0 else 999.0,
            'IGD_Std': round(float(np.std(igd_arr)), 4) if len(igd_arr) > 0 else 0.0,
        })

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

def run_manyobjective_comparison(n_seeds: int = 10, budget: int = 10000, pool=None):
    """
    On P10 (DTLZ2, M = 5), compares NSGA-II, NSGA-III, MOEA/D, RVEA, and HypE.
    """
    print("\nRunning Task D2: Many-Objective Comparison on P10 (M = 5)...")
    raw_dir = 'results/data/raw'
    os.makedirs(raw_dir, exist_ok=True)

    algs = [
        (14, "NSGA-II"),
        (15, "NSGA-III"),
        (18, "MOEA/D"),
        (31, "RVEA"),
        (34, "HypE")
    ]

    job_args = []
    for num, name in algs:
        for s in range(n_seeds):
            seed = s * 10 + 17
            job_args.append((num, name, seed, budget, raw_dir))

    if pool is not None:
        raw_results = pool.map(_worker_manyobjective_job, job_args)
    else:
        raw_results = [_worker_manyobjective_job(arg) for arg in job_args]

    grouped = {}
    for r in raw_results:
        alg_name = r['algorithm']
        if alg_name not in grouped:
            grouped[alg_name] = {'hv': [], 'igd': []}
        grouped[alg_name]['hv'].append(r['hv'])
        grouped[alg_name]['igd'].append(r['igd'])

    records = []
    for num, name in algs:
        g = grouped.get(name, {'hv': [], 'igd': []})
        hv_arr = g['hv']
        igd_arr = g['igd']

        records.append({
            'Algorithm': name,
            'HV_Mean': round(float(np.mean(hv_arr)), 4) if len(hv_arr) > 0 else 0.0,
            'HV_Std': round(float(np.std(hv_arr)), 4) if len(hv_arr) > 0 else 0.0,
            'IGD_Mean': round(float(np.mean(igd_arr)), 4) if len(igd_arr) > 0 else 999.0,
            'IGD_Std': round(float(np.std(igd_arr)), 4) if len(igd_arr) > 0 else 0.0
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

def run_discrete_comparison(n_seeds: int = 10, budget: int = 10000, pool=None):
    """
    On P8 (15-city bi-objective TSP), compares MOEA/D-ACO, MOACO, NSGA-II (permutation), and MOMA.
    """
    print("\nRunning Task D3: Discrete Combinatorial Comparison on P8 (TSP)...")
    raw_dir = 'results/data/raw'
    os.makedirs(raw_dir, exist_ok=True)

    algs = [
        (20, 'MOEA/D-ACO'),
        (25, 'MOACO'),
        ('perm_nsga2', 'Permutation NSGA-II'),
        ('moma_2opt', 'MOMA (2-opt)')
    ]

    job_args = []
    for alg_key, alg_name in algs:
        for s in range(n_seeds):
            seed = s * 10 + 7
            job_args.append((alg_key, alg_name, seed, budget, raw_dir))

    if pool is not None:
        raw_results = pool.map(_worker_discrete_job, job_args)
    else:
        raw_results = [_worker_discrete_job(arg) for arg in job_args]

    grouped = {}
    fronts_dict = {}
    for r in raw_results:
        alg_name = r['algorithm']
        if alg_name not in grouped:
            grouped[alg_name] = {'hv': [], 'sp': [], 'best_hv': -1.0, 'best_front': None}
        grouped[alg_name]['hv'].append(r['hv'])
        grouped[alg_name]['sp'].append(r['spacing'])
        if r['hv'] > grouped[alg_name]['best_hv']:
            grouped[alg_name]['best_hv'] = r['hv']
            grouped[alg_name]['best_front'] = np.array(r['front_F'])

    records = []
    for alg_key, alg_name in algs:
        g = grouped.get(alg_name, {'hv': [], 'sp': [], 'best_front': None})
        fronts_dict[alg_name] = g['best_front']
        hv_arr = g['hv']
        sp_arr = g['sp']

        records.append({
            'Algorithm': alg_name,
            'HV_Mean': round(float(np.mean(hv_arr)), 4) if len(hv_arr) > 0 else 0.0,
            'HV_Std': round(float(np.std(hv_arr)), 4) if len(hv_arr) > 0 else 0.0,
            'Spacing_Mean': round(float(np.mean(sp_arr)), 4) if len(sp_arr) > 0 else 0.0,
            'Spacing_Std': round(float(np.std(sp_arr)), 4) if len(sp_arr) > 0 else 0.0
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
    parser = argparse.ArgumentParser(description="Task D Head-to-Head Comparisons")
    parser.add_argument("--smoke", action="store_true", help="Run quick 1-seed smoke test")
    args = parser.parse_args()

    n_seeds = 1 if args.smoke else 10
    budget = 100 if args.smoke else 10000

    print("=" * 80)
    mode_str = "SMOKE TEST (1 seed, small budget)" if args.smoke else "FULL SWEEP (10 seeds, 10,000 evals)"
    print(f"EXECUTING TASK D: HEAD-TO-HEAD COMPARISONS — {mode_str}")
    print("=" * 80)

    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/data', exist_ok=True)
    os.makedirs('results/data/raw', exist_ok=True)

    num_workers = max(1, os.cpu_count() - 1)
    print(f"Initializing multiprocessing pool with {num_workers} workers...")

    with multiprocessing.Pool(processes=num_workers) as pool:
        # D1: Bi-objective on P3 and P5
        print("\n[D1] Running Bi-Objective Comparison on P3...")
        rec_p3 = run_biobjective_comparison("P3", n_seeds=n_seeds, budget=budget, pool=pool)
        save_csv('results/data/task_d1_p3_ranked.csv', rec_p3,
                 ['Rank', 'Problem', 'Algorithm', 'HV_Mean', 'HV_Std', 'Spacing_Mean', 'Spacing_Std', 'IGD_Mean', 'IGD_Std'])
        print(f"  -> Saved results/data/task_d1_p3_ranked.csv ({len(rec_p3)} algorithms)")

        print("\n[D1] Running Bi-Objective Comparison on P5...")
        rec_p5 = run_biobjective_comparison("P5", n_seeds=n_seeds, budget=budget, pool=pool)
        save_csv('results/data/task_d1_p5_ranked.csv', rec_p5,
                 ['Rank', 'Problem', 'Algorithm', 'HV_Mean', 'HV_Std', 'Spacing_Mean', 'Spacing_Std', 'IGD_Mean', 'IGD_Std'])
        print(f"  -> Saved results/data/task_d1_p5_ranked.csv ({len(rec_p5)} algorithms)")

        # D2: Many-objective on P10 M=5
        print("\n[D2] Running Many-Objective Comparison on P10 (M=5)...")
        rec_m5 = run_manyobjective_comparison(n_seeds=n_seeds, budget=budget, pool=pool)
        save_csv('results/data/task_d2_p10_m5_ranked.csv', rec_m5,
                 ['Rank', 'Algorithm', 'HV_Mean', 'HV_Std', 'IGD_Mean', 'IGD_Std'])
        print(f"  -> Saved results/data/task_d2_p10_m5_ranked.csv ({len(rec_m5)} algorithms)")

        # D3: Discrete on P8 TSP
        print("\n[D3] Running Discrete Combinatorial Comparison on P8 TSP...")
        rec_p8 = run_discrete_comparison(n_seeds=n_seeds, budget=budget, pool=pool)
        save_csv('results/data/task_d3_p8_tsp_ranked.csv', rec_p8,
                 ['Rank', 'Algorithm', 'HV_Mean', 'HV_Std', 'Spacing_Mean', 'Spacing_Std'])
        print(f"  -> Saved results/data/task_d3_p8_tsp_ranked.csv ({len(rec_p8)} algorithms)")

    # D4: MCDM on P4 front
    print("\n[D4] Running MCDM Comparison on P4...")
    mcdm_res = run_mcdm_comparison()
    save_csv('results/data/task_d4_mcdm.csv', mcdm_res, list(mcdm_res[0].keys()))
    print("  -> Saved results/data/task_d4_mcdm.csv")

    # D5: Gradient MTL on P13
    print("\n[D5] Running Gradient MTL Comparison on P13...")
    mtl_res = run_gradient_comparison()
    save_csv('results/data/task_d5_mtl.csv', mtl_res, list(mtl_res[0].keys()))
    print("  -> Saved results/data/task_d5_mtl.csv")

    print("\n" + "=" * 80)
    print("ALL TASK D EXPERIMENTS COMPLETED AND RAW CSVs WRITTEN TO results/data/!")
    print("=" * 80)


if __name__ == "__main__":
    main()

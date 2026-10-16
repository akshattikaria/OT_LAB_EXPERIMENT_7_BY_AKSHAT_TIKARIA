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
# SCRIPTED RNG FOR REPRODUCIBLE DETERMINISTIC HAND TRACES
# ==============================================================================

class ScriptedRNG:
    """
    Deterministic scripted RNG for exact hand trace verification.
    Supplies fixed lists of numbers for rand, choice, and normal calls.
    """
    def __init__(self, rands=None, choices=None, normals=None):
        self.rands = list(rands) if rands is not None else []
        self.choices = list(choices) if choices is not None else []
        self.normals = list(normals) if normals is not None else []
        self.r_idx = 0
        self.c_idx = 0
        self.n_idx = 0

    def rand(self):
        if self.r_idx < len(self.rands):
            v = self.rands[self.r_idx]
            self.r_idx += 1
            return v
        return 0.5

    def random(self):
        return self.rand()

    def choice(self, a, size=None, replace=True, p=None):
        if self.c_idx < len(self.choices):
            v = self.choices[self.c_idx]
            self.c_idx += 1
            return v
        if isinstance(a, int):
            if a == 1:
                return np.array([0]) if size else 0
            return np.arange(size) if size else 0
        elif hasattr(a, '__len__'):
            if len(a) == 1:
                return a[0]
            return a[0]
        return 0

    def normal(self, size=None):
        if self.n_idx < len(self.normals):
            v = self.normals[self.n_idx]
            self.n_idx += 1
            return v
        return np.zeros(size) if size else 0.0


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
    checks = []

    # Setting 1: lambda = 1.0
    cfg1 = {'utility_type': 'portfolio', 'lambda_risk': 1.0}
    x_opt1, f_opt1, cv1, u_opt1 = a05.step(p4, cfg1, x0=np.array([0.25, 0.25, 0.25, 0.25]))
    w_hand_1 = [0.0000, 0.4285, 0.5079, 0.0000]
    risk_hand_1 = 0.03783
    ret_hand_1 = 0.13627
    u_hand_1 = 0.11736

    for i in range(4):
        code_w = round(float(x_opt1[i]), 4)
        diff = abs(w_hand_1[i] - code_w)
        assert diff < 1e-4, f"Alg 05 Iter 1 weight {i} mismatch: hand {w_hand_1[i]} vs code {code_w}"
        checks.append({'item': f'Alg05_Iter1_w{i}', 'hand': w_hand_1[i], 'code': code_w, 'diff': diff})

    code_risk1 = round(float(f_opt1[0]), 5)
    diff = abs(risk_hand_1 - code_risk1)
    assert diff < 1e-4
    checks.append({'item': 'Alg05_Iter1_Risk', 'hand': risk_hand_1, 'code': code_risk1, 'diff': diff})

    code_ret1 = round(float(-f_opt1[1]), 5)
    diff = abs(ret_hand_1 - code_ret1)
    assert diff < 1e-4
    checks.append({'item': 'Alg05_Iter1_Return', 'hand': ret_hand_1, 'code': code_ret1, 'diff': diff})

    code_u1 = round(float(u_opt1), 5)
    diff = abs(u_hand_1 - code_u1)
    assert diff < 1e-4
    checks.append({'item': 'Alg05_Iter1_Utility', 'hand': u_hand_1, 'code': code_u1, 'diff': diff})

    # Setting 2: lambda = 10.0
    cfg2 = {'utility_type': 'portfolio', 'lambda_risk': 10.0}
    x_opt2, f_opt2, cv2, u_opt2 = a05.step(p4, cfg2, x0=np.array([0.25, 0.25, 0.25, 0.25]))
    w_hand_2 = [0.4382, 0.1734, 0.1070, 0.2486]
    risk_hand_2 = 0.00778
    ret_hand_2 = 0.10006
    u_hand_2 = 0.06118

    for i in range(4):
        code_w = round(float(x_opt2[i]), 4)
        diff = abs(w_hand_2[i] - code_w)
        assert diff < 1e-4, f"Alg 05 Iter 2 weight {i} mismatch: hand {w_hand_2[i]} vs code {code_w}"
        checks.append({'item': f'Alg05_Iter2_w{i}', 'hand': w_hand_2[i], 'code': code_w, 'diff': diff})

    code_risk2 = round(float(f_opt2[0]), 5)
    diff = abs(risk_hand_2 - code_risk2)
    assert diff < 1e-4
    checks.append({'item': 'Alg05_Iter2_Risk', 'hand': risk_hand_2, 'code': code_risk2, 'diff': diff})

    code_ret2 = round(float(-f_opt2[1]), 5)
    diff = abs(ret_hand_2 - code_ret2)
    assert diff < 1e-4
    checks.append({'item': 'Alg05_Iter2_Return', 'hand': ret_hand_2, 'code': code_ret2, 'diff': diff})

    code_u2 = round(float(u_opt2), 5)
    diff = abs(u_hand_2 - code_u2)
    assert diff < 1e-4
    checks.append({'item': 'Alg05_Iter2_Utility', 'hand': u_hand_2, 'code': code_u2, 'diff': diff})

    print("\n--- Algorithm 5 Hand Trace Verification ---")
    print(f"Setting 1: Risk={code_risk1}, Return={code_ret1}, Utility={code_u1}, Weights={w_hand_1}")
    print(f"Setting 2: Risk={code_risk2}, Return={code_ret2}, Utility={code_u2}, Weights={w_hand_2}")
    return checks


def run_hand_trace_alg17():
    """
    Hand trace verification for Algorithm 17 (PESA-II)
    Tiny instance: Problem P1 (Factory Production Speed), initial archive size = 3, max capacity = 3, K = 4.
    Runs 2 FULL iterations through a17.step() with ScriptedRNG, cleanly triggering
    most-crowded-hyperbox removal in both iterations.
    """
    p1 = mu.ProblemP1()
    checks = []

    rand_sbx1 = 0.5 * (0.6 ** 21)
    rand_sbx2 = 0.5 * (0.8 ** 21)

    choices = [
        # Iter 1 parent 1: boxes [0, 2], pick index 0 (x=4.0)
        np.array([0, 2]), 0,
        # Iter 1 parent 2: boxes [1, 2], pick index 2 (x=5.0)
        np.array([1, 2]), 2,
        # Iter 1 archive pruning: crowded box index 0, remove index 0 (x=4.0)
        0, 0,

        # Iter 2 parent 1: boxes [1, 2], pick index 1 (x=5.0)
        np.array([1, 2]), 1,
        # Iter 2 parent 2: boxes [0, 2], pick index 0 (x=6.0)
        np.array([0, 2]), 0,
        # Iter 2 archive pruning: crowded box index 0, remove index 1 (x=5.0)
        0, 1
    ]

    rands = [
        # Iter 1
        0.2,       # p1 tie
        0.8,       # p2 tie
        0.1, 0.1, rand_sbx1, 0.8, # sbx crossover -> c1=4.2
        0.5, 0.5,  # polynomial mutation -> zero deltaq

        # Iter 2
        0.2,       # p1 tie
        0.2,       # p2 tie
        0.1, 0.1, rand_sbx2, 0.8, # sbx crossover -> c2=5.1
        0.5, 0.5   # polynomial mutation -> zero deltaq
    ]

    rng = ScriptedRNG(rands=rands, choices=choices)

    # Initial archive: 3 mutually non-dominated points on P1
    init_pop = np.array([[4.0], [6.0], [5.0]])
    init_F = np.array([p1.evaluate(x)[0] for x in init_pop])
    init_CV = np.zeros(3)

    # Iteration 1
    pop1, F1, CV1 = a17.step(init_pop, init_F, init_CV, p1, archive_size=3, K=4, batch_size=1, rng=rng)
    hand_iter1 = {
        'Alg17_Iter1_c1_x': (4.2, float(pop1[2, 0])),
        'Alg17_Iter1_c1_f1': (1.44, float(F1[2, 0])),
        'Alg17_Iter1_c1_f2': (7.84, float(F1[2, 1])),
        'Alg17_Iter1_arc0_x': (6.0, float(pop1[0, 0])),
        'Alg17_Iter1_arc1_x': (5.0, float(pop1[1, 0])),
        'Alg17_Iter1_arc2_x': (4.2, float(pop1[2, 0]))
    }
    for item, (hand_val, code_val) in hand_iter1.items():
        diff = abs(hand_val - code_val)
        assert diff < 1e-9, f"{item} mismatch: hand {hand_val} vs code {code_val}"
        checks.append({'item': item, 'hand': hand_val, 'code': code_val, 'diff': diff})

    # Iteration 2
    pop2, F2, CV2 = a17.step(pop1, F1, CV1, p1, archive_size=3, K=4, batch_size=1, rng=rng)
    hand_iter2 = {
        'Alg17_Iter2_c2_x': (5.1, float(pop2[2, 0])),
        'Alg17_Iter2_c2_f1': (4.41, float(F2[2, 0])),
        'Alg17_Iter2_c2_f2': (3.61, float(F2[2, 1])),
        'Alg17_Iter2_arc0_x': (6.0, float(pop2[0, 0])),
        'Alg17_Iter2_arc1_x': (4.2, float(pop2[1, 0])),
        'Alg17_Iter2_arc2_x': (5.1, float(pop2[2, 0]))
    }
    for item, (hand_val, code_val) in hand_iter2.items():
        diff = abs(hand_val - code_val)
        assert diff < 1e-9, f"{item} mismatch: hand {hand_val} vs code {code_val}"
        checks.append({'item': item, 'hand': hand_val, 'code': code_val, 'diff': diff})

    print("\n--- Algorithm 17 (PESA-II) Hand Trace Verification ---")
    print(f"Iter 1 Candidate c1: x={pop1[2, 0]:.4f}, F=[{F1[2, 0]:.4f}, {F1[2, 1]:.4f}], Archive={pop1.flatten().tolist()}")
    print(f"Iter 2 Candidate c2: x={pop2[2, 0]:.4f}, F=[{F2[2, 0]:.4f}, {F2[2, 1]:.4f}], Archive={pop2.flatten().tolist()}")
    return checks


def run_hand_trace_alg29():
    """
    Hand trace verification for Algorithm 29 (MOBA)
    Tiny instance: Problem P1, N = 4 bats, 2 full iterations with ScriptedRNG.
    Checks all 4 bats across both iterations: guide, beta, freq, velocity, position, f, acceptance, loudness, pulse rate.
    """
    p1 = mu.ProblemP1()
    checks = []

    X = np.array([[2.0], [4.0], [6.0], [8.0]])
    V = np.zeros((4, 1))
    F = np.array([p1.evaluate(x)[0] for x in X])
    CV = np.zeros(4)
    A = np.full(4, 0.9)
    r = np.full(4, 0.1)
    freq = np.zeros(4)
    arc_X = np.array([[4.0], [6.0]])
    arc_F = np.array([[1.0, 9.0], [9.0, 1.0]])
    arc_CV = np.zeros(2)

    floats_it1 = [
        0.2, 0.5, 0.001, 0.1,  # Bat 0
        0.8, 0.4, 0.001, 0.1,  # Bat 1
        0.2, 0.3, 0.001, 0.1,  # Bat 2
        0.8, 0.5, 0.001, 0.1,  # Bat 3
    ]
    guide_choices_it2 = [0, 1, 2, 3]
    floats_it2 = [
        0.4, 0.001, 0.1,  # Bat 0
        0.3, 0.001, 0.1,  # Bat 1
        0.5, 0.001, 0.1,  # Bat 2
        0.4, 0.001, 0.1,  # Bat 3
    ]

    rng = ScriptedRNG(rands=floats_it1 + floats_it2, choices=guide_choices_it2)

    # Step 1
    X1, V1, F1, CV1, A1, r1, freq1, arc_X1, arc_F1, arc_CV1 = a29.step(
        X.copy(), V.copy(), F.copy(), CV.copy(), A.copy(), r.copy(), freq.copy(),
        arc_X.copy(), arc_F.copy(), arc_CV.copy(), p1, current_iter=1,
        f_min=0.0, f_max=1.0, alpha=0.95, gamma=0.05, r0=0.1, archive_size=4, rng=rng
    )

    moba_hand_it1 = [
        (4.0, 0.5, 0.5, 1.0, 3.0, 0.0, 16.0, 0.8550, 0.1 * (1.0 - np.exp(-0.05))),
        (6.0, 0.4, 0.4, 0.8, 4.8, 3.24, 4.84, 0.8550, 0.1 * (1.0 - np.exp(-0.05))),
        (4.0, 0.3, 0.3, -0.6, 5.4, 5.76, 2.56, 0.8550, 0.1 * (1.0 - np.exp(-0.05))),
        (6.0, 0.5, 0.5, -1.0, 7.0, 16.0, 0.0, 0.8550, 0.1 * (1.0 - np.exp(-0.05)))
    ]

    for i in range(4):
        g_h, b_h, fr_h, v_h, x_h, f1_h, f2_h, a_h, r_h = moba_hand_it1[i]
        vals = [
            (f'Alg29_Iter1_Bat{i}_V', v_h, float(V1[i, 0])),
            (f'Alg29_Iter1_Bat{i}_X', x_h, float(X1[i, 0])),
            (f'Alg29_Iter1_Bat{i}_f1', f1_h, float(F1[i, 0])),
            (f'Alg29_Iter1_Bat{i}_f2', f2_h, float(F1[i, 1])),
            (f'Alg29_Iter1_Bat{i}_A', a_h, float(A1[i])),
            (f'Alg29_Iter1_Bat{i}_r', r_h, float(r1[i]))
        ]
        for item, hand_v, code_v in vals:
            diff = abs(hand_v - code_v)
            assert diff < 1e-9, f"{item} mismatch: hand {hand_v} vs code {code_v}"
            checks.append({'item': item, 'hand': hand_v, 'code': code_v, 'diff': diff})

    # Step 2
    X2, V2, F2, CV2, A2, r2, freq2, arc_X2, arc_F2, arc_CV2 = a29.step(
        X1.copy(), V1.copy(), F1.copy(), CV1.copy(), A1.copy(), r1.copy(), freq1.copy(),
        arc_X1.copy(), arc_F1.copy(), arc_CV1.copy(), p1, current_iter=2,
        f_min=0.0, f_max=1.0, alpha=0.95, gamma=0.05, r0=0.1, archive_size=4, rng=rng
    )

    moba_hand_it2 = [
        (3.0, 0.4, 0.4, 0.5, 3.5, 0.25, 12.25, 0.81225, 0.1 * (1.0 - np.exp(-0.10))),
        (7.0, 0.3, 0.3, 1.06, 5.86, 8.1796, 1.2996, 0.81225, 0.1 * (1.0 - np.exp(-0.10))),
        (4.0, 0.5, 0.5, -1.0, 4.4, 1.96, 6.76, 0.81225, 0.1 * (1.0 - np.exp(-0.10))),
        (6.0, 0.4, 0.4, -0.9, 6.1, 9.61, 0.81, 0.81225, 0.1 * (1.0 - np.exp(-0.10)))
    ]

    for i in range(4):
        g_h, b_h, fr_h, v_h, x_h, f1_h, f2_h, a_h, r_h = moba_hand_it2[i]
        vals = [
            (f'Alg29_Iter2_Bat{i}_V', v_h, float(V2[i, 0])),
            (f'Alg29_Iter2_Bat{i}_X', x_h, float(X2[i, 0])),
            (f'Alg29_Iter2_Bat{i}_f1', f1_h, float(F2[i, 0])),
            (f'Alg29_Iter2_Bat{i}_f2', f2_h, float(F2[i, 1])),
            (f'Alg29_Iter2_Bat{i}_A', a_h, float(A2[i])),
            (f'Alg29_Iter2_Bat{i}_r', r_h, float(r2[i]))
        ]
        for item, hand_v, code_v in vals:
            diff = abs(hand_v - code_v)
            assert diff < 1e-9, f"{item} mismatch: hand {hand_v} vs code {code_v}"
            checks.append({'item': item, 'hand': hand_v, 'code': code_v, 'diff': diff})

    print("\n--- Algorithm 29 (MOBA) Hand Trace Verification ---")
    print(f"Iter 1 Bat Positions: {X1.flatten().tolist()}")
    print(f"Iter 2 Bat Positions: {X2.flatten().tolist()}")
    return checks


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

    # 1. Run hand traces and save verification CSV
    chk5 = run_hand_trace_alg05()
    chk17 = run_hand_trace_alg17()
    chk29 = run_hand_trace_alg29()
    all_trace_checks = chk5 + chk17 + chk29

    trace_csv_path = 'results/data/task_c_trace_check.csv'
    with open(trace_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['item', 'hand', 'code', 'diff'])
        writer.writeheader()
        writer.writerows(all_trace_checks)
    print(f"\n[OK] Task C Trace Check CSV saved to: {trace_csv_path} ({len(all_trace_checks)} verified items)")

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

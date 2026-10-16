"""
alg02_epsilon_constraint.py - Epsilon-Constraint Method
======================================================
Family: Classical (Part 1)

Core Mechanism:
Optimizes one primary objective while bounding all other (M-1) objectives
by upper thresholds (epsilon values):
    min f_1(x)
    subject to f_k(x) <= eps_k,  k = 2, ..., M
    and original problem constraints.
By systematically varying eps_k across the objective range, non-convex and
dented regions of the Pareto front can be reached.

Lecture Steps:
1. Determine the individual minimum and maximum for each objective via a Payoff Table:
   - Minimize f_2 alone (with multiple restarts) -> gives f_2 at f_2-minimizer (f2_min).
   - Lexicographic minimization of f_1 (min f_1, then min f_2 subject to f_1 <= f_1_min)
     -> gives f_2 at lexicographic f_1-minimizer (f2_max).
2. Designate f_1 as primary objective.
3. Discretize eps in [f2_min, f2_max] into n_eps values.
4. For each eps:
   a. Solve min f_1(x) subject to f_2(x) <= eps and constraints via SLSQP.
   b. Warm-start the next eps from the previous solution.
5. Filter non-dominated solutions.

Default Parameters:
- n_eps: 25
- primary_obj: 0
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def compute_payoff_bounds(problem: Any, n_starts: int = 5, seed: int = 42) -> Tuple[float, float, np.ndarray]:
    """
    Computes payoff table bounds for secondary objective f2:
    - f2_min: f2 at f2-minimizer
    - f2_max: f2 at lexicographic f1-minimizer
    Also returns the initial solution x_lex.
    """
    rng = np.random.default_rng(seed)
    bounds = list(zip(problem.xl, problem.xu))

    # 1. Minimize f2 (f2-minimizer)
    best_f2 = np.inf
    best_x_f2 = None
    for _ in range(n_starts):
        x0 = rng.uniform(problem.xl, problem.xu)
        res = minimize(
            lambda x: problem.evaluate(x)[0][1] + 1e4 * problem.evaluate(x)[1],
            x0, bounds=bounds, method='SLSQP', options={'maxiter': 300, 'ftol': 1e-6}
        )
        f, cv = problem.evaluate(res.x)
        if cv < 1e-4 and f[1] < best_f2:
            best_f2 = f[1]
            best_x_f2 = res.x

    # 2. Lexicographic f1-minimizer:
    # Step A: Find global f1_min
    best_f1 = np.inf
    for _ in range(n_starts):
        x0 = rng.uniform(problem.xl, problem.xu)
        res = minimize(
            lambda x: problem.evaluate(x)[0][0] + 1e4 * problem.evaluate(x)[1],
            x0, bounds=bounds, method='SLSQP', options={'maxiter': 300, 'ftol': 1e-6}
        )
        f, cv = problem.evaluate(res.x)
        if cv < 1e-4 and f[0] < best_f1:
            best_f1 = f[0]

    # Step B: Minimize f2 subject to f1 <= best_f1 + 1e-4
    best_f2_lex = np.inf
    best_x_lex = None
    for _ in range(n_starts):
        x0 = rng.uniform(problem.xl, problem.xu)
        res = minimize(
            lambda x: problem.evaluate(x)[0][1] + 1e4 * problem.evaluate(x)[1],
            x0,
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda x: best_f1 + 1e-4 - problem.evaluate(x)[0][0]},
            method='SLSQP',
            options={'maxiter': 300, 'ftol': 1e-6}
        )
        f, cv = problem.evaluate(res.x)
        if cv < 1e-4 and f[0] <= best_f1 + 1e-3 and f[1] < best_f2_lex:
            best_f2_lex = f[1]
            best_x_lex = res.x

    if best_x_lex is None:
        best_x_lex = 0.5 * (problem.xl + problem.xu)
        best_f2_lex = float(problem.evaluate(best_x_lex)[0][1])

    f2_min = float(best_f2) if np.isfinite(best_f2) else 0.0
    f2_max = float(best_f2_lex) if np.isfinite(best_f2_lex) else 1.0

    if f2_min > f2_max:
        f2_min, f2_max = f2_max, f2_min
    if abs(f2_max - f2_min) < 1e-5:
        f2_max = f2_min + 1.0

    return f2_min, f2_max, best_x_lex


def initialize(problem: Any, n_eps: int = 25, primary_obj: int = 0, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Computes epsilon grid from the payoff table."""
    f2_min, f2_max, x0 = compute_payoff_bounds(problem, n_starts=5, seed=seed)
    eps_grid = np.linspace(f2_min, f2_max, n_eps)
    return eps_grid, x0


def step(problem: Any, eps_val: float, primary_obj: int = 0,
         x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solves one epsilon-constrained subproblem via SLSQP."""
    if x0 is None:
        x0 = 0.5 * (problem.xl + problem.xu)

    bounds = list(zip(problem.xl, problem.xu))
    sec_obj = 1 if primary_obj == 0 else 0

    def objective(x):
        f, cv = problem.evaluate(x)
        return f[primary_obj] + 1e4 * cv

    def constraint_fun(x):
        f, _ = problem.evaluate(x)
        return eps_val - f[sec_obj]  # eps_val - f2 >= 0

    res = minimize(
        fun=objective,
        x0=x0,
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': constraint_fun},
        method='SLSQP',
        options={'maxiter': 500, 'ftol': 1e-7}
    )
    x_opt = res.x
    f_opt, cv_opt = problem.evaluate(x_opt)
    # Check if eps constraint is violated
    if f_opt[sec_obj] > eps_val + 1e-4:
        cv_opt += (f_opt[sec_obj] - eps_val)
    return x_opt, f_opt, cv_opt


def run(problem: Any, n_eps: int = 25, seed: int = 42) -> Dict[str, Any]:
    """Runs the full epsilon-constraint sweep using payoff bounds."""
    eps_vals, x_init = initialize(problem, n_eps=n_eps, primary_obj=0, seed=seed)
    X_res = []
    F_res = []
    CV_res = []

    x_curr = x_init.copy()

    for eps in eps_vals:
        x_opt, f_opt, cv_opt = step(problem, float(eps), primary_obj=0, x0=x_curr)
        X_res.append(x_opt)
        F_res.append(f_opt)
        CV_res.append(cv_opt)
        if cv_opt < 1e-3:
            x_curr = x_opt

    X_arr = np.array(X_res)
    F_arr = np.array(F_res)
    CV_arr = np.array(CV_res)

    fronts = mu.fast_non_dominated_sort(F_arr, CV_arr)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    evals = getattr(problem, 'evaluations', n_eps * 150)

    return {
        'algorithm': 'Epsilon-Constraint',
        'X_all': X_arr,
        'F_all': F_arr,
        'CV_all': CV_arr,
        'X_front': X_arr[nd_indices],
        'F_front': F_arr[nd_indices],
        'eps_grid': eps_vals,
        'evaluations': evals
    }

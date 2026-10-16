"""
alg03_benson.py - Benson's Method
=================================
Family: Classical (Part 1)

Core Mechanism:
Benson's method projects a dominated reference point u0 onto the Pareto
front by maximizing the distance/shortfall past the reference point:
    max sum_{m=1}^M l_m  (or min -sum l_m)
    subject to:
        f_m(x) + l_m <= u0_m,  for all m = 1, ..., M
        l_m >= 0,  x in Feasible Region.
This guarantees that the resulting point is strictly Pareto-optimal. By varying
the reference points u0 along the Pareto front boundary, different parts of
the front are reached.

Lecture Steps:
1. Determine Pareto extremes for each objective via single-objective minimization.
2. Construct reference points u0 as feasible points dominated by the front,
   spread across both objectives to cover the entire Pareto boundary.
3. For each reference point u0:
   a. Formulate Benson's NLP:
      min_{x} -sum(max(0, u0 - f(x))) + penalty * violations
   b. Solve using nonlinear optimizer (SLSQP).
4. Filter non-dominated solutions.

Default Parameters:
- n_points: 25
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, n_points: int = 25, seed: int = 42) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generates reference points u0 as feasible points dominated by the front,
    spread systematically across the objective trade-off.
    """
    bounds = list(zip(problem.xl, problem.xu))
    rng = np.random.default_rng(seed)

    # 1. Minimize f1
    best_f1 = np.inf
    x1_opt = None
    for _ in range(5):
        x0 = rng.uniform(problem.xl, problem.xu)
        res = minimize(
            lambda x: problem.evaluate(x)[0][0] + 1e4 * problem.evaluate(x)[1],
            x0, bounds=bounds, method='SLSQP', options={'maxiter': 300, 'ftol': 1e-6}
        )
        f, cv = problem.evaluate(res.x)
        if cv < 1e-4 and f[0] < best_f1:
            best_f1 = f[0]
            x1_opt = res.x

    # 2. Minimize f2
    best_f2 = np.inf
    x2_opt = None
    for _ in range(5):
        x0 = rng.uniform(problem.xl, problem.xu)
        res = minimize(
            lambda x: problem.evaluate(x)[0][1] + 1e4 * problem.evaluate(x)[1],
            x0, bounds=bounds, method='SLSQP', options={'maxiter': 300, 'ftol': 1e-6}
        )
        f, cv = problem.evaluate(res.x)
        if cv < 1e-4 and f[1] < best_f2:
            best_f2 = f[1]
            x2_opt = res.x

    if x1_opt is None:
        x1_opt = 0.5 * (problem.xl + problem.xu)
    if x2_opt is None:
        x2_opt = 0.5 * (problem.xl + problem.xu)

    f1_pt = problem.evaluate(x1_opt)[0]
    f2_pt = problem.evaluate(x2_opt)[0]

    f1_min, f1_max = min(f1_pt[0], f2_pt[0]), max(f1_pt[0], f2_pt[0])
    f2_min, f2_max = min(f1_pt[1], f2_pt[1]), max(f1_pt[1], f2_pt[1])
    r1 = max(1e-4, f1_max - f1_min)
    r2 = max(1e-4, f2_max - f2_min)

    x_mid = 0.5 * (x1_opt + x2_opt)
    n1 = n_points // 2
    n2 = n_points - n1

    u0_list = []
    x_inits = []

    # First half: bound f1, sweep towards midpoint
    for k in range(n1):
        t = k / max(1, n1 - 1) * 0.5
        f1_val = f1_min + (2.0 * t) * (f1_max - f1_min) * 0.5
        u0_list.append(np.array([f1_val, f2_max + 0.05 * r2]))
        x_inits.append((1.0 - 2.0 * t) * x1_opt + (2.0 * t) * x_mid)

    # Second half: bound f2, sweep towards midpoint
    for k in range(n2):
        t = k / max(1, n2 - 1) * 0.5
        f2_val = f2_min + (2.0 * t) * (f2_max - f2_min) * 0.5
        u0_list.append(np.array([f1_max + 0.05 * r1, f2_val]))
        x_inits.append((1.0 - 2.0 * t) * x2_opt + (2.0 * t) * x_mid)

    return u0_list, x_inits


def step(problem: Any, u0: np.ndarray,
         w: Optional[np.ndarray] = None,
         x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solves one Benson projection subproblem."""
    if x0 is None:
        x0 = 0.5 * (problem.xl + problem.xu)

    bounds = list(zip(problem.xl, problem.xu))
    weights = np.ones_like(u0) if w is None else np.asarray(w)

    def benson_obj(x):
        f, cv = problem.evaluate(x)
        shortfall = np.maximum(0.0, u0 - f) * weights
        violation = np.maximum(0.0, f - u0)
        return float(-np.sum(shortfall) + 1e4 * np.sum(violation) + 1e4 * cv)

    res = minimize(
        fun=benson_obj,
        x0=x0,
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 300, 'ftol': 1e-7}
    )
    x_opt = res.x
    f_opt, cv_opt = problem.evaluate(x_opt)
    return x_opt, f_opt, cv_opt


def run(problem: Any, n_points: int = 25, seed: int = 42) -> Dict[str, Any]:
    """Runs Benson's method across spread reference targets."""
    u0_list, x_inits = initialize(problem, n_points=n_points, seed=seed)
    X_res = []
    F_res = []
    CV_res = []

    for u0, x_init in zip(u0_list, x_inits):
        x_opt, f_opt, cv_opt = step(problem, u0, x0=x_init)
        X_res.append(x_opt)
        F_res.append(f_opt)
        CV_res.append(cv_opt)

    X_arr = np.array(X_res)
    F_arr = np.array(F_res)
    CV_arr = np.array(CV_res)

    fronts = mu.fast_non_dominated_sort(F_arr, CV_arr)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    evals = getattr(problem, 'evaluations', n_points * 150)

    return {
        'algorithm': "Benson's Method",
        'X_all': X_arr,
        'F_all': F_arr,
        'CV_all': CV_arr,
        'X_front': X_arr[nd_indices],
        'F_front': F_arr[nd_indices],
        'ref_points': np.array(u0_list),
        'evaluations': evals
    }

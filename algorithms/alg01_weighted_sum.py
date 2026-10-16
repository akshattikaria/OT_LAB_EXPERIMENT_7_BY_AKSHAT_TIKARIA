"""
alg01_weighted_sum.py - Weighted Sum Method
===========================================
Family: Classical (Part 1)

Core Mechanism:
Converts multi-objective problem into a single-objective scalar optimization
by assigning non-negative weights to each objective:
    min F(x) = sum_{m=1}^M w_m * f_m(x),  where sum w_m = 1, w_m >= 0.
The Pareto front is approximated by systematically sweeping weight vectors.

Lecture Steps:
1. Normalize objectives if scales differ.
2. Generate a systematic grid/simplex of weight vectors w in W.
3. For each weight vector w:
   a. Formulate scalar sub-problem: min sum_m w_m * f_m(x) s.t. constraints.
   b. Solve using single-objective numerical optimizer (e.g., SLSQP / L-BFGS-B / Nelder-Mead).
   c. Record optimal decision vector x* and objective vector F(x*).
4. Filter dominated solutions using fast non-dominated sorting.

Default Parameters:
- n_weights: 25 (for 2D problems)
- optimizer: 'SLSQP'
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, n_weights: int = 25) -> np.ndarray:
    """Generates uniform weight vectors for M objectives."""
    M = problem.n_objs
    if M == 2:
        w1 = np.linspace(0.0, 1.0, n_weights)
        w2 = 1.0 - w1
        return np.column_stack([w1, w2])
    else:
        # Dirichlet uniform sampling for M > 2
        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(M), size=n_weights)
        return weights


def evaluate_scalar(x: np.ndarray, weights: np.ndarray, problem: Any) -> float:
    """Computes scalar weighted sum objective with penalty for constraint violations."""
    f, cv = problem.evaluate(x)
    return float(np.dot(weights, f) + 1e4 * cv)


def step(problem: Any, weights: np.ndarray, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solves one scalarized subproblem for a given weight vector."""
    if x0 is None:
        if hasattr(problem, 'xl') and len(problem.xl) > 0:
            x0 = 0.5 * (problem.xl + problem.xu)
        else:
            x0 = np.zeros(problem.n_vars)

    bounds = None
    if hasattr(problem, 'xl') and len(problem.xl) > 0:
        bounds = list(zip(problem.xl, problem.xu))

    res = minimize(
        fun=lambda x: evaluate_scalar(x, weights, problem),
        x0=x0,
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 500, 'ftol': 1e-7}
    )
    x_opt = res.x
    f_opt, cv_opt = problem.evaluate(x_opt)
    return x_opt, f_opt, cv_opt


def run(problem: Any, n_weights: int = 25, seed: int = 42) -> Dict[str, Any]:
    """Runs the full Weighted Sum sweep."""
    weights_grid = initialize(problem, n_weights=n_weights)
    X_res = []
    F_res = []
    CV_res = []

    x0 = None
    if hasattr(problem, 'xl') and len(problem.xl) > 0:
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(problem.xl, problem.xu)

    for w in weights_grid:
        x_opt, f_opt, cv_opt = step(problem, w, x0=x0)
        X_res.append(x_opt)
        F_res.append(f_opt)
        CV_res.append(cv_opt)
        x0 = x_opt # Warm-start next optimization

    X_arr = np.array(X_res)
    F_arr = np.array(F_res)
    CV_arr = np.array(CV_res)

    # Filter non-dominated front
    fronts = mu.fast_non_dominated_sort(F_arr, CV_arr)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'Weighted Sum',
        'X_all': X_arr,
        'F_all': F_arr,
        'CV_all': CV_arr,
        'X_front': X_arr[nd_indices],
        'F_front': F_arr[nd_indices],
        'weights': weights_grid,
        'evaluations': n_weights * 150 # approx evaluations
    }

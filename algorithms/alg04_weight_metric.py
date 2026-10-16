"""
alg04_weight_metric.py - Weighted Metric Method (p-norm, Tchebycheff)
=====================================================================
Family: Classical (Part 1)

Core Mechanism:
Minimizes the weighted distance from an ideal reference point z*:
    min ||w * (f(x) - z*)||_p
Cases:
- p = 1: Weighted Manhattan metric: sum_m w_m * |f_m(x) - z*_m|
- p = 2: Weighted Euclidean metric: sqrt( sum_m w_m^2 * (f_m(x) - z*_m)^2 )
- p = inf (Tchebycheff): max_m w_m * |f_m(x) - z*_m|
The Tchebycheff metric (p = inf) can reach ANY Pareto-optimal solution,
including those situated in non-convex (dented) regions.

Lecture Steps:
1. Estimate or compute the ideal point z*_m = min_{x} f_m(x) for each objective.
2. Select distance norm p in {1, 2, np.inf}.
3. Generate uniform weight vectors w on the simplex.
4. For each weight vector:
   a. Formulate distance scalarization:
      if p < inf: min (sum_m w_m * (f_m(x) - z*_m)^p)^(1/p)
      if p == inf: min max_m w_m * (f_m(x) - z*_m) + rho * sum_m w_m * (f_m(x) - z*_m)
   b. Solve with nonlinear optimizer.
5. Filter non-dominated solutions.

Default Parameters:
- p: np.inf (Tchebycheff)
- n_weights: 25
- rho: 1e-4 (augmented factor)
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, n_weights: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Generates weights and computes/approximates ideal point z*."""
    M = problem.n_objs
    if M == 2:
        w1 = np.linspace(0.01, 0.99, n_weights)
        w2 = 1.0 - w1
        weights = np.column_stack([w1, w2])
    else:
        rng = np.random.default_rng(42)
        weights = rng.dirichlet(np.ones(M), size=n_weights)

    # Approximate ideal point z* by optimizing each objective individually
    z_star = np.zeros(M)
    bounds = list(zip(problem.xl, problem.xu))
    x0 = 0.5 * (problem.xl + problem.xu)
    for m in range(M):
        res = minimize(
            fun=lambda x, m=m: problem.evaluate(x)[0][m] + 1e4 * problem.evaluate(x)[1],
            x0=x0,
            bounds=bounds,
            method='SLSQP'
        )
        z_star[m] = problem.evaluate(res.x)[0][m]

    return weights, z_star


def evaluate_metric(x: np.ndarray, weights: np.ndarray, z_star: np.ndarray,
                    p_norm: float, rho: float, problem: Any) -> float:
    """Evaluates weighted distance metric from ideal point z*."""
    f, cv = problem.evaluate(x)
    diff = np.maximum(0.0, f - z_star) # non-negative difference

    if np.isposinf(p_norm):
        # Augmented Tchebycheff
        val = np.max(weights * diff) + rho * np.sum(weights * diff)
    elif p_norm == 1:
        val = np.sum(weights * diff)
    elif p_norm == 2:
        val = np.sqrt(np.sum((weights * diff) ** 2))
    else:
        val = (np.sum((weights * diff) ** p_norm)) ** (1.0 / p_norm)

    return float(val + 1e4 * cv)


def step(problem: Any, weights: np.ndarray, z_star: np.ndarray,
         p_norm: float = np.inf, rho: float = 1e-4,
         x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solves one weighted metric subproblem."""
    if x0 is None:
        x0 = 0.5 * (problem.xl + problem.xu)

    bounds = list(zip(problem.xl, problem.xu))
    res = minimize(
        fun=lambda x: evaluate_metric(x, weights, z_star, p_norm, rho, problem),
        x0=x0,
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 500, 'ftol': 1e-7}
    )
    x_opt = res.x
    f_opt, cv_opt = problem.evaluate(x_opt)
    return x_opt, f_opt, cv_opt


def run(problem: Any, p_norm: float = np.inf, n_weights: int = 25, seed: int = 42) -> Dict[str, Any]:
    """Runs the full Weighted Metric method for chosen p-norm."""
    weights_grid, z_star = initialize(problem, n_weights=n_weights)
    X_res = []
    F_res = []
    CV_res = []

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(problem.xl, problem.xu)

    for w in weights_grid:
        x_opt, f_opt, cv_opt = step(problem, w, z_star, p_norm=p_norm, x0=x0)
        X_res.append(x_opt)
        F_res.append(f_opt)
        CV_res.append(cv_opt)
        x0 = x_opt

    X_arr = np.array(X_res)
    F_arr = np.array(F_res)
    CV_arr = np.array(CV_res)

    fronts = mu.fast_non_dominated_sort(F_arr, CV_arr)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    p_str = "inf (Tchebycheff)" if np.isposinf(p_norm) else str(p_norm)
    return {
        'algorithm': f'Weight Metric (p={p_str})',
        'X_all': X_arr,
        'F_all': F_arr,
        'CV_all': CV_arr,
        'X_front': X_arr[nd_indices],
        'F_front': F_arr[nd_indices],
        'z_star': z_star,
        'weights': weights_grid,
        'evaluations': n_weights * 150
    }

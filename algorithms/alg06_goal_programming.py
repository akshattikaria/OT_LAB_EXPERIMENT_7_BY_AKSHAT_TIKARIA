"""
alg06_goal_programming.py - Goal Programming Method
===================================================
Family: Classical (Part 1, 5E)

Core Mechanism:
Solves multi-objective optimization by establishing aspiration targets T_m
for each objective and minimizing the weighted sum of unwanted deviations:
    f_m(x) - d_m^+ + d_m^- = T_m,  where d_m^+, d_m^- >= 0
    min sum_m [ w_m^+ * d_m^+ + w_m^- * d_m^- ]
For "at least" goals (e.g. profit >= 2400), underachievement d_m^- is penalized.
For "at most" goals (e.g. cost <= 100), overachievement d_m^+ is penalized.

Lecture Steps:
1. Specify target levels T_1, ..., T_M.
2. Determine deviation direction (underachievement or overachievement).
3. Assign penalty weights w_m to deviations (unit-weight or normalized by target).
4. Formulate LP / NLP with auxiliary deviation variables.
5. Solve and report achieved objective values and deviation shortfalls.

Default Parameters:
- weights_mode: 'normalized' (or 'unit')
- optimizer: 'SLSQP'
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, targets: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Initializes targets and deviation weights."""
    if targets is None:
        if "Production" in getattr(problem, 'name', ''):
            # P11: Profit >= 2400, Units >= 600. Remember in P11 objectives are MIN -profit, MIN -units.
            # So -profit <= -2400, -units <= -600.
            targets = np.array([-2400.0, -600.0])
        else:
            # Default midpoint targets
            targets = np.zeros(problem.n_objs)

    return {
        'targets': targets,
        'weights': np.ones(len(targets))
    }


def step(problem: Any, targets: np.ndarray, weights: np.ndarray,
         weights_mode: str = "unit",
         x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Solves one Goal Programming subproblem."""
    if x0 is None:
        if hasattr(problem, 'xl') and len(problem.xl) > 0:
            x0 = 0.5 * (problem.xl + problem.xu)
        else:
            x0 = np.zeros(problem.n_vars)

    bounds = None
    if hasattr(problem, 'xl') and len(problem.xl) > 0:
        bounds = list(zip(problem.xl, problem.xu))

    def gp_objective(x):
        f, cv = problem.evaluate(x)
        # Deviation: for minimization f <= T => unwanted deviation is max(0, f - T)
        deviations = np.maximum(0.0, f - targets)
        if weights_mode == "normalized":
            denom = np.abs(targets)
            denom[denom < 1e-6] = 1.0
            norm_dev = deviations / denom
            obj = np.dot(weights, norm_dev)
        else:
            obj = np.dot(weights, deviations)
        return float(obj + 1e4 * cv)

    res = minimize(
        fun=gp_objective,
        x0=x0,
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 500, 'ftol': 1e-7}
    )
    x_opt = res.x
    f_opt, cv_opt = problem.evaluate(x_opt)
    dev_opt = np.maximum(0.0, f_opt - targets)
    return x_opt, f_opt, cv_opt, dev_opt


def run(problem: Any, targets: Optional[np.ndarray] = None,
        weights_mode: str = "unit", seed: int = 42) -> Dict[str, Any]:
    """Runs Goal Programming and sweeps target weighting."""
    config = initialize(problem, targets=targets)
    targets = config['targets']

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(problem.xl, problem.xu) if hasattr(problem, 'xl') and len(problem.xl) > 0 else np.zeros(problem.n_vars)

    # Solve base goal programming
    x_opt, f_opt, cv_opt, dev_opt = step(problem, targets, config['weights'], weights_mode=weights_mode, x0=x0)

    # Sweep goal priority weights to map alternative compromise options
    sweep_w = np.linspace(0.05, 0.95, 20)
    X_list, F_list, CV_list = [x_opt], [f_opt], [cv_opt]
    for w1 in sweep_w:
        w_vec = np.array([w1, 1.0 - w1])
        xs, fs, cvs, _ = step(problem, targets, w_vec, weights_mode=weights_mode, x0=x_opt)
        X_list.append(xs)
        F_list.append(fs)
        CV_list.append(cvs)

    X_all = np.array(X_list)
    F_all = np.array(F_list)
    CV_all = np.array(CV_list)

    fronts = mu.fast_non_dominated_sort(F_all, CV_all)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'Goal Programming',
        'x_opt': x_opt,
        'f_opt': f_opt,
        'cv_opt': cv_opt,
        'deviations': dev_opt,
        'targets': targets,
        'X_all': X_all,
        'F_all': F_all,
        'CV_all': CV_all,
        'X_front': X_all[nd_indices],
        'F_front': F_all[nd_indices],
        'evaluations': len(X_list) * 150
    }

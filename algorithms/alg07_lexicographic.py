"""
alg07_lexicographic.py - Lexicographic Method
=============================================
Family: Classical (Part 1, 5E)

Core Mechanism:
Optimizes objectives sequentially in strict pre-defined priority order:
    Priority 1: min f_{(1)}(x) s.t. x in Feasible  => f_{(1)}*
    Priority 2: min f_{(2)}(x) s.t. x in Feasible and f_{(1)}(x) <= f_{(1)}* + delta_1
    ...
    Priority M: min f_{(M)}(x) s.t. all previous objectives within allowed tolerances.
Higher-priority objectives are never compromised for lower-priority gains.
When ties or flat plateaus exist for the primary objective, secondary objectives
break the tie.

Lecture Steps:
1. Arrange objectives according to strict hierarchy of importance.
2. Step 1: Solve single-objective optimization for Priority 1 objective to find f_{(1)}*.
3. Formulate constraint bounding Priority 1 to within tolerance delta_1:
   f_{(1)}(x) <= f_{(1)}* + delta_1.
4. Step 2: Solve for Priority 2 objective subject to the new constraint.
5. Repeat through all M objectives until a unique Pareto-optimal solution is found.

Default Parameters:
- priority_order: [0, 1] (or custom list of objective indices)
- tolerances: [1e-4, 1e-4]
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, priority_order: Optional[List[int]] = None,
               tolerances: Optional[List[float]] = None) -> Dict[str, Any]:
    """Sets objective priority order and tolerances."""
    M = problem.n_objs
    if priority_order is None:
        priority_order = list(range(M))
    if tolerances is None:
        tolerances = [1e-4] * M
    return {
        'priority_order': priority_order,
        'tolerances': tolerances
    }


def step(problem: Any, target_obj: int, active_bounds: List[Tuple[int, float]],
         x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Solves one priority level optimization subject to bounds on higher priorities."""
    bounds = None
    if hasattr(problem, 'xl') and len(problem.xl) > 0:
        bounds = list(zip(problem.xl, problem.xu))

    constraints = []
    for obj_idx, limit_val in active_bounds:
        def c_fun(x, idx=obj_idx, lim=limit_val):
            f, _ = problem.evaluate(x)
            return lim - f[idx] # lim - f >= 0
        constraints.append({'type': 'ineq', 'fun': c_fun})

    def obj_fun(x):
        f, cv = problem.evaluate(x)
        # Add penalty for higher priority violations
        viol = 0.0
        for obj_idx, limit_val in active_bounds:
            if f[obj_idx] > limit_val:
                viol += (f[obj_idx] - limit_val)
        return float(f[target_obj] + 1e4 * cv + 1e5 * viol)

    # Multi-start to escape flat plateaus and ties
    candidates = []
    if x0 is not None:
        candidates.append(x0)
    if hasattr(problem, 'xl') and len(problem.xl) > 0:
        candidates.append(0.5 * (problem.xl + problem.xu))
        rng = np.random.default_rng(42)
        for _ in range(8):
            candidates.append(rng.uniform(problem.xl, problem.xu))
    else:
        candidates.append(np.zeros(problem.n_vars))

    best_x = None
    best_score = float('inf')

    for cand_x in candidates:
        res = minimize(
            fun=obj_fun,
            x0=cand_x,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-7}
        )
        score = obj_fun(res.x)
        if score < best_score:
            best_score = score
            best_x = res.x

    x_opt = best_x
    f_opt, cv_opt = problem.evaluate(x_opt)
    return x_opt, f_opt, cv_opt


def run(problem: Any, priority_order: Optional[List[int]] = None,
        tolerances: Optional[List[float]] = None,
        seed: int = 42) -> Dict[str, Any]:
    """Executes the full sequential lexicographic optimization."""
    config = initialize(problem, priority_order=priority_order, tolerances=tolerances)
    p_order = config['priority_order']
    tols = config['tolerances']

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(problem.xl, problem.xu) if hasattr(problem, 'xl') and len(problem.xl) > 0 else np.zeros(problem.n_vars)

    active_bounds: List[Tuple[int, float]] = []
    level_history = []

    current_x = x0
    for level, obj_idx in enumerate(p_order):
        x_opt, f_opt, cv_opt = step(problem, obj_idx, active_bounds, x0=current_x)
        current_x = x_opt
        best_val = f_opt[obj_idx]
        limit = best_val + tols[level]
        active_bounds.append((obj_idx, limit))
        level_history.append({
            'level': level,
            'obj_idx': obj_idx,
            'best_val': best_val,
            'x': x_opt,
            'f': f_opt
        })

    # Also generate the reverse priority order for comparison
    rev_order = list(reversed(p_order))
    rev_bounds: List[Tuple[int, float]] = []
    current_x_rev = x0
    for level, obj_idx in enumerate(rev_order):
        x_opt_rev, f_opt_rev, cv_opt_rev = step(problem, obj_idx, rev_bounds, x0=current_x_rev)
        current_x_rev = x_opt_rev
        rev_bounds.append((obj_idx, f_opt_rev[obj_idx] + tols[level]))

    X_all = np.array([current_x, current_x_rev])
    F_all = np.array([f_opt, f_opt_rev])
    CV_all = np.array([cv_opt, cv_opt_rev])

    fronts = mu.fast_non_dominated_sort(F_all, CV_all)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'Lexicographic',
        'x_opt': current_x,
        'f_opt': f_opt,
        'cv_opt': cv_opt,
        'level_history': level_history,
        'X_all': X_all,
        'F_all': F_all,
        'CV_all': CV_all,
        'X_front': X_all[nd_indices],
        'F_front': F_all[nd_indices],
        'evaluations': len(p_order) * 150
    }

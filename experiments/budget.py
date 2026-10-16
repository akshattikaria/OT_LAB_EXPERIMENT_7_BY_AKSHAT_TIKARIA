"""
experiments/budget.py - Calibration and execution for budgeted MOO evaluations
==============================================================================
Calibrates iteration parameters to ensure stochastic algorithms run within
the target evaluation budget [9000, 10100] (default target = 10,000).
"""

import sys
import os
from typing import Any, Dict, Optional
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('algorithms'))
import moo_utils as mu


def get_iteration_param_name(alg_num: int) -> Optional[str]:
    """Returns the name of the iteration parameter for algorithm number."""
    if alg_num in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 27, 28, 30, 31, 32, 34, 35]:
        return 'n_gen'
    elif alg_num in [22, 23, 29]:
        return 'max_iter'
    elif alg_num == 24:
        return 'max_cycles'
    elif alg_num == 25:
        return 'n_cycles'
    elif alg_num in [26, 33]:
        return 'n_steps'
    else:
        return None  # Unbudgeted (algs 1-9, 36, 37)


def get_base_kwargs(alg_num: int, problem: Any) -> Dict[str, Any]:
    """Returns default population and archive kwargs (100 for 2-obj problems)."""
    kwargs = {}
    n_objs = getattr(problem, 'n_objs', 2)

    if alg_num in [10, 11, 12, 13, 14, 18, 19, 21, 27, 28, 30, 32, 33, 34, 35]:
        kwargs['pop_size'] = 100
    elif alg_num == 15:  # NSGA-III
        # NSGA-III reference points determine pop_size if not 2-obj
        if n_objs == 2:
            kwargs['pop_size'] = 100
    elif alg_num == 16:  # SPEA2
        kwargs['pop_size'] = 100
        kwargs['archive_size'] = 100
    elif alg_num == 17:  # PESA-II
        kwargs['archive_size'] = 100
    elif alg_num in [20, 25]:  # MOEA/D-ACO, MOACO (discrete TSP)
        kwargs['n_ants'] = 100
    elif alg_num == 22:  # MOPSO
        kwargs['n_particles'] = 100
        kwargs['archive_size'] = 100
    elif alg_num == 23:  # MOGWO
        kwargs['n_wolves'] = 100
        kwargs['archive_size'] = 100
    elif alg_num == 24:  # MOABC
        kwargs['n_sources'] = 50
        kwargs['archive_size'] = 100
    elif alg_num == 26:  # MOSA
        kwargs['chain_length'] = 50
        kwargs['archive_size'] = 100
    elif alg_num == 29:  # MOBA
        kwargs['n_bats'] = 100
        kwargs['archive_size'] = 100

    return kwargs


def calibrate_iterations(alg_num: int, module: Any, problem: Any,
                         param_name: str, base_kwargs: Dict[str, Any],
                         budget: int = 10000, seed: int = 42) -> int:
    """
    Calibrates the iteration parameter so total counted evaluations
    land inside [9000, 10100]. Runs with iter=1 and iter=2 to find slope and intercept.
    """
    # Underlying problem without outer counter
    raw_prob = problem._problem if isinstance(problem, mu.CountingProblem) else problem

    # Run with iter = 1
    cp1 = mu.CountingProblem(raw_prob)
    kw1 = dict(base_kwargs)
    kw1[param_name] = 1
    kw1['seed'] = seed
    try:
        module.run(cp1, **kw1)
        e1 = cp1.evaluations
    except Exception:
        return 50

    # Run with iter = 2
    cp2 = mu.CountingProblem(raw_prob)
    kw2 = dict(base_kwargs)
    kw2[param_name] = 2
    kw2['seed'] = seed
    try:
        module.run(cp2, **kw2)
        e2 = cp2.evaluations
    except Exception:
        return 50

    m = e2 - e1
    b = e1 - m

    if m > 0:
        target_iter = int(round((budget - b) / m))
        projected = b + m * target_iter
        if projected < 9000 and (projected + m) <= 10100:
            target_iter += 1
        elif projected > 10100 and (projected - m) >= 9000:
            target_iter -= 1
        target_iter = max(1, target_iter)
    else:
        target_iter = max(1, budget // max(1, e1))

    # Refine algorithms with variable evaluations per step (e.g. MOMA local search)
    if alg_num == 30 and m > 0:
        kw_t = dict(base_kwargs)
        kw_t['seed'] = seed
        cp_t = mu.CountingProblem(raw_prob, max_evals=budget)
        kw_t[param_name] = target_iter
        try:
            module.run(cp_t, **kw_t)
            if cp_t.evaluations < 9000:
                target_iter += 1
        except Exception:
            pass

    return target_iter


def run_budgeted(alg_num: int, module: Any, problem: Any, seed: int = 42,
                 budget: int = 10000, target_iter: Optional[int] = None) -> Dict[str, Any]:
    """
    Executes an algorithm with an exact calibrated evaluation count or unbudgeted run.

    Returns the algorithm result dictionary with 'evaluations' set to the actual
    counted number of evaluate() calls.
    """
    raw_prob = problem._problem if isinstance(problem, mu.CountingProblem) else problem
    param_name = get_iteration_param_name(alg_num)

    # 1. Unbudgeted algorithms (1-9, 36, 37)
    if param_name is None:
        cp = mu.CountingProblem(raw_prob, max_evals=None)
        if alg_num in [8, 9]:
            # MCDM ranking
            X, w, is_ben = cp.get_decision_matrix()
            res = module.run(X, w, is_ben)
            res['evaluations'] = len(X)
            res['counted_evals'] = len(X)
            return res
        elif alg_num in [36, 37]:
            res = module.run(cp, max_iter=100, seed=seed)
        else:
            res = module.run(cp, seed=seed)

        res['evaluations'] = cp.evaluations
        res['counted_evals'] = cp.evaluations
        return res

    # 2. Budgeted stochastic algorithms (10-35)
    base_kwargs = get_base_kwargs(alg_num, problem)

    if target_iter is None:
        target_iter = calibrate_iterations(
            alg_num, module, raw_prob, param_name, base_kwargs, budget=budget, seed=seed
        )

    run_kwargs = dict(base_kwargs)
    run_kwargs[param_name] = target_iter
    run_kwargs['seed'] = seed

    cp = mu.CountingProblem(raw_prob, max_evals=budget)
    res = module.run(cp, **run_kwargs)
    res['evaluations'] = cp.evaluations
    res['counted_evals'] = cp.evaluations
    res['calibrated_iter'] = target_iter
    return res

"""
experiments/status.py - Quantitative status classification for MOO algorithms
=============================================================================
Rules:
- 'works':
    * Multi-objective: >=10 distinct non-dominated points AND (if true front known)
      normalized IGD <= 0.05 AND no infeasible points unless intentionally unconstrained.
    * Single-solution / Ranking: matches brief sanity check (e.g., P11 profit 2400 / units 500).
- 'partly works':
    * Valid front or solution, but poor front, failed threshold, or documented mathematical
      limitation (e.g. Weighted Sum on non-convex P3).
    * Every non-'works' row MUST provide a measured, factual reason.
- 'not done':
    * No valid front (empty, all NaN, or fatal crash).
"""

import numpy as np
from typing import Dict, Any, Optional
import moo_utils as mu


def classify_status(alg_num: int, p_name: str, res: Dict[str, Any],
                    igd_val: Optional[float] = None,
                    use_constraints: bool = True) -> str:
    """Classifies algorithm run status according to rigorous quantitative criteria."""
    
    # --------------------------------------------------------------------------
    # 1. Single-Solution & MCDM Ranking Methods (Algs 5, 6, 7, 8, 9, 36, 37)
    # --------------------------------------------------------------------------
    if alg_num in [5, 6, 7, 8, 9, 36, 37]:
        # Alg 07: Lexicographic Method
        if alg_num == 7:
            if p_name == 'P11':
                f_opt = res.get('f_opt')
                if f_opt is not None:
                    profit = -f_opt[0]
                    units = -f_opt[1]
                    cv = res.get('cv_opt', 0.0)
                    if cv <= 1e-4 and abs(profit - 2400.0) < 5.0 and abs(units - 500.0) < 5.0:
                        return 'works'
                    else:
                        return f'partly works: P11 profit={profit:.1f} (target 2400), units={units:.1f} (target 500)'
            cv_opt = res.get('cv_opt', 0.0)
            return 'works' if cv_opt <= 1e-4 else f'partly works: constraint violation CV={cv_opt:.4f}'

        # Alg 06: Goal Programming
        elif alg_num == 6:
            if p_name == 'P11':
                f_opt = res.get('f_opt')
                if f_opt is not None:
                    profit = -f_opt[0]
                    cv = res.get('cv_opt', 0.0)
                    if cv <= 1e-4 and abs(profit - 2400.0) < 5.0:
                        return 'works'
                    else:
                        return f'partly works: P11 profit={profit:.1f} (target 2400), CV={cv:.4f}'
            cv_opt = res.get('cv_opt', 0.0)
            return 'works' if cv_opt <= 1e-4 else f'partly works: constraint violation CV={cv_opt:.4f}'

        # Alg 05: Value Function
        elif alg_num == 5:
            cv_opt = res.get('cv_opt', 0.0)
            return 'works' if cv_opt <= 1e-4 else f'partly works: constraint violation CV={cv_opt:.4f}'

        # Alg 08: TOPSIS
        elif alg_num == 8:
            closeness = res.get('closeness')
            if closeness is not None and len(closeness) > 0 and np.all(np.isfinite(closeness)):
                return 'works'
            return 'not done: invalid closeness coefficients'

        # Alg 09: VIKOR
        elif alg_num == 9:
            Q = res.get('Q')
            if Q is not None and len(Q) > 0 and np.all(np.isfinite(Q)):
                return 'works'
            return 'not done: invalid VIKOR compromise scores'

        # Alg 36: MGDA
        elif alg_num == 36:
            hx = res.get('history_x', np.array([]))
            if len(hx) > 0 and np.all(np.isfinite(hx)):
                return 'works'
            return 'not done: gradient optimization failed'

        # Alg 37: GradNorm
        elif alg_num == 37:
            hx = res.get('history_x', np.array([]))
            if len(hx) > 0 and np.all(np.isfinite(hx)):
                return 'works'
            return 'not done: GradNorm optimization failed'

    # --------------------------------------------------------------------------
    # 2. Multi-Objective Generating Algorithms (Algs 1-4, 10-35)
    # --------------------------------------------------------------------------
    front_F = res.get('F_front')
    if front_F is None or len(front_F) == 0:
        return 'not done: empty Pareto front'

    # Drop dominated and duplicate points
    F_nd = mu._drop_dominated_duplicates(front_F)
    n_distinct = len(np.unique(np.round(F_nd, 4), axis=0))
    if n_distinct == 0:
        return 'not done: 0 non-dominated points'

    # Documented mathematical limitations
    if alg_num == 1 and p_name == 'P3':
        return 'partly works: Weighted Sum cannot access non-convex Pareto front interior'
    if alg_num == 10 and p_name == 'P3':
        return 'partly works: VEGA suffers speciation & misses non-convex interior'

    # Check constraint feasibility (unless intentionally unconstrained)
    if use_constraints:
        cv_front = res.get('CV_front')
        if cv_front is None and 'CV' in res:
            cv_front = res['CV']
        if cv_front is not None and len(cv_front) > 0:
            n_infeas = int(np.sum(cv_front > 1e-5))
            if n_infeas > 0:
                return f'partly works: front contains {n_infeas}/{len(cv_front)} infeasible solutions'

    # Quantitative acceptance checks: distinct >= 10, normalized IGD <= 0.05
    distinct_ok = (n_distinct >= 10)
    
    # Check IGD if true Pareto front is analytically known
    igd_known = (igd_val is not None and not np.isnan(igd_val))
    igd_ok = (not igd_known) or (igd_val <= 0.05)

    if distinct_ok and igd_ok:
        return 'works'
    
    # Non-'works': provide exact measured reason
    reasons = []
    if not distinct_ok:
        reasons.append(f'{n_distinct} distinct points (<10)')
    if igd_known and not igd_ok:
        reasons.append(f'normalized IGD={igd_val:.4f} (>0.05)')

    return f"partly works: {', '.join(reasons)}"

"""
alg09_vikor.py - VIKOR Compromise Ranking Method
=================================================
Family: Ranking / MCDM (Part 1, 5E)

Core Mechanism:
VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje) determines
a compromise ranking list based on measuring closeness to the ideal solution.
It balances majority benefit (group utility S) against maximum individual regret (R):
    S_i = sum_j w_j * |f_j* - f_{ij}| / |f_j* - f_j^-|   (Manhattan L1 distance)
    R_i = max_j w_j * |f_j* - f_{ij}| / |f_j* - f_j^-|   (Chebyshev L_inf distance)
    Q_i = v * (S_i - S*) / (S^- - S*) + (1 - v) * (R_i - R*) / (R^- - R*)
where v in [0, 1] is the weight of the strategy of "maximum group utility".

Lecture Steps:
1. Determine best f_j* and worst f_j^- values for all criteria.
2. Calculate normalized distance ratios and compute S_i and R_i for each alternative.
3. Compute Q_i values for parameter v (e.g., v = 0.25, 0.5, 0.75).
4. Rank alternatives by S, R, and Q in ascending order (lowest value is best).
5. Propose compromise solution checking Condition 1 (Acceptable advantage)
   and Condition 2 (Acceptable stability).

Default Parameters:
- v: 0.5 (compromise weight)
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(decision_matrix: np.ndarray, weights: np.ndarray,
               is_beneficial: np.ndarray, v: float = 0.5) -> Dict[str, Any]:
    """Sets up criteria matrices and normalizes weights."""
    w = np.array(weights, dtype=float)
    w = w / np.sum(w)
    return {
        'X': np.array(decision_matrix, dtype=float),
        'weights': w,
        'is_beneficial': np.array(is_beneficial, dtype=bool),
        'v': float(v)
    }


def evaluate(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes S, R, Q values and ranks alternatives."""
    X = config['X']
    w = config['weights']
    is_beneficial = config['is_beneficial']
    v = config['v']
    N, M = X.shape

    # 1. Best f* and worst f-
    f_star = np.zeros(M)
    f_minus = np.zeros(M)
    for j in range(M):
        if is_beneficial[j]:
            f_star[j] = np.max(X[:, j])
            f_minus[j] = np.min(X[:, j])
        else:
            f_star[j] = np.min(X[:, j])
            f_minus[j] = np.max(X[:, j])

    # 2. Compute S and R
    denom = np.abs(f_star - f_minus)
    denom[denom == 0] = 1e-12

    norm_dist = np.abs(f_star - X) / denom
    weighted_dist = norm_dist * w

    S = np.sum(weighted_dist, axis=1)
    R = np.max(weighted_dist, axis=1)

    # 3. Compute Q
    S_star, S_minus = np.min(S), np.max(S)
    R_star, R_minus = np.min(R), np.max(R)

    denom_S = (S_minus - S_star) if (S_minus - S_star) > 1e-12 else 1.0
    denom_R = (R_minus - R_star) if (R_minus - R_star) > 1e-12 else 1.0

    Q = v * ((S - S_star) / denom_S) + (1.0 - v) * ((R - R_star) / denom_R)

    # Ascending sort: lowest Q is rank 1
    rank_Q = np.argsort(Q)
    return S, R, Q, rank_Q


def step(config: Dict[str, Any], v_val: float) -> Dict[str, Any]:
    """Evaluates VIKOR with a specific value of v parameter."""
    cfg = config.copy()
    cfg['v'] = v_val
    S, R, Q, rank_Q = evaluate(cfg)
    return {
        'v': v_val,
        'S': S,
        'R': R,
        'Q': Q,
        'rank_Q': rank_Q
    }


def run(decision_matrix: np.ndarray, weights: np.ndarray,
        is_beneficial: np.ndarray, v_list: Optional[List[float]] = None,
        v: Optional[float] = None) -> Dict[str, Any]:
    """Runs full VIKOR analysis across multiple consensus parameters v."""
    if v_list is None:
        v_list = [0.25, 0.50, 0.75]
    v_val = 0.5 if v is None else float(v)

    config = initialize(decision_matrix, weights, is_beneficial, v=v_val)
    S, R, Q, rank_Q = evaluate(config)

    # Condition checks on top rank
    N = len(decision_matrix)
    DQ = 1.0 / (N - 1) if N > 1 else 1.0
    best_idx = rank_Q[0]
    second_idx = rank_Q[1] if N > 1 else rank_Q[0]

    cond1_adv = bool((Q[second_idx] - Q[best_idx]) >= DQ - 1e-7)
    cond2_stab = bool(best_idx == np.argmin(S) or best_idx == np.argmin(R))

    v_runs = {v: step(config, v) for v in v_list}

    return {
        'algorithm': 'VIKOR',
        'S': S,
        'R': R,
        'Q': Q,
        'rankings': rank_Q,
        'v_runs': v_runs,
        'condition_1_acceptable_advantage': cond1_adv,
        'condition_2_acceptable_stability': cond2_stab,
        'evaluations': len(decision_matrix)
    }

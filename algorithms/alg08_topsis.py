"""
alg08_topsis.py - TOPSIS Method (Technique for Order of Preference by Similarity to Ideal Solution)
====================================================================================================
Family: Ranking / MCDM (Part 1, 5E)

Core Mechanism:
Ranks a finite set of discrete alternatives by geometric distance to the
Ideal and Anti-Ideal solutions:
    Best alternative has the shortest Euclidean distance to the Positive-Ideal
    Solution (PIS) and the farthest distance from the Negative-Ideal Solution (NIS).

Lecture Steps:
1. Formulate Decision Matrix X of size (N, M) across criteria.
2. Vector normalization: r_{ij} = x_{ij} / sqrt( sum_{k=1}^N x_{kj}^2 ).
3. Calculate weighted normalized decision matrix: v_{ij} = w_j * r_{ij}.
4. Determine Ideal (A+) and Anti-Ideal (A-) reference points:
   - For beneficial criteria: v_j+ = max_i v_{ij}, v_j- = min_i v_{ij}
   - For cost criteria:       v_j+ = min_i v_{ij}, v_j- = max_i v_{ij}
5. Calculate Euclidean separation distances:
   S_i+ = sqrt( sum_j (v_{ij} - v_j+)^2 )
   S_i- = sqrt( sum_j (v_{ij} - v_j-)^2 )
6. Calculate relative closeness to ideal: C_i = S_i- / (S_i+ + S_i-).
7. Rank alternatives in descending order of C_i (C_i in [0, 1]).

Default Parameters:
- weights: problem criteria weights
- is_beneficial: boolean mask for criteria types
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(decision_matrix: np.ndarray, weights: np.ndarray,
               is_beneficial: np.ndarray) -> Dict[str, Any]:
    """Validates inputs and normalizes weights."""
    w = np.array(weights, dtype=float)
    w = w / np.sum(w)
    return {
        'X': np.array(decision_matrix, dtype=float),
        'weights': w,
        'is_beneficial': np.array(is_beneficial, dtype=bool)
    }


def evaluate(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes TOPSIS relative closeness scores and rankings."""
    X = config['X']
    w = config['weights']
    is_beneficial = config['is_beneficial']

    N, M = X.shape

    # 1. Vector normalization
    norm_factors = np.sqrt(np.sum(X ** 2, axis=0))
    norm_factors[norm_factors == 0] = 1.0
    R = X / norm_factors

    # 2. Weighted normalized matrix
    V = R * w

    # 3. Determine positive ideal (A+) and negative ideal (A-)
    A_pos = np.zeros(M)
    A_neg = np.zeros(M)
    for j in range(M):
        if is_beneficial[j]:
            A_pos[j] = np.max(V[:, j])
            A_neg[j] = np.min(V[:, j])
        else:
            A_pos[j] = np.min(V[:, j])
            A_neg[j] = np.max(V[:, j])

    # 4. Separation measures
    S_pos = np.sqrt(np.sum((V - A_pos) ** 2, axis=1))
    S_neg = np.sqrt(np.sum((V - A_neg) ** 2, axis=1))

    # 5. Relative closeness
    denom = S_pos + S_neg
    denom[denom == 0] = 1e-12
    C = S_neg / denom

    # Rank descending (highest C is rank 1)
    rankings = np.argsort(-C)
    return C, rankings, S_pos, S_neg


def step(config: Dict[str, Any], weight_delta_idx: int = 0, delta: float = 0.1) -> Dict[str, Any]:
    """
    Performs sensitivity analysis by perturbing one criterion weight by delta.
    Tests if the ranking flips.
    """
    w_perturbed = np.copy(config['weights'])
    w_perturbed[weight_delta_idx] += delta
    w_perturbed = np.maximum(0.0, w_perturbed)
    w_perturbed = w_perturbed / np.sum(w_perturbed)

    perturbed_config = config.copy()
    perturbed_config['weights'] = w_perturbed
    C_pert, rank_pert, _, _ = evaluate(perturbed_config)

    return {
        'perturbed_weights': w_perturbed,
        'C_perturbed': C_pert,
        'ranking_perturbed': rank_pert
    }


def run(decision_matrix: np.ndarray, weights: np.ndarray,
        is_beneficial: np.ndarray) -> Dict[str, Any]:
    """Executes full TOPSIS evaluation and sensitivity analysis."""
    config = initialize(decision_matrix, weights, is_beneficial)
    C, rankings, S_pos, S_neg = evaluate(config)
    sensitivity = step(config, weight_delta_idx=0, delta=0.1)

    return {
        'algorithm': 'TOPSIS',
        'closeness': C,
        'rankings': rankings,
        'S_pos': S_pos,
        'S_neg': S_neg,
        'sensitivity': sensitivity,
        'evaluations': len(decision_matrix)
    }

"""
alg26_mosa.py - Multi-Objective Simulated Annealing (MOSA)
==========================================================
Family: Physics-inspired Metaheuristic (Part 4)

Core Mechanism:
Extends classical Metropolis simulated annealing to multi-objective optimization:
1. Candidate Neighbor Generation:
   Perturbs the current solution x using Gaussian random moves scaled to decision bounds.
2. Multi-Objective Acceptance Criterion:
   - If candidate x' dominates current x: ACCEPT unconditionally.
   - If current x dominates candidate x': ACCEPT with Boltzmann probability:
         P_accept = exp( - delta_E / T ),  where delta_E = max_m [ (f_m(x') - f_m(x)) / range_m ]
   - If x' and x are mutually non-dominated: ACCEPT to promote trade-off exploration.
3. External Pareto Archive:
   Every accepted solution updates a non-dominated archive.
4. Annealing / Cooling Schedule:
   Temperature decreases geometrically: T_{k+1} = alpha * T_k (e.g. alpha = 0.95).

Lecture Steps:
1. Initialize temperature T_0 and starting solution x_0. Evaluate objectives.
2. Initialize empty external archive and seed with x_0.
3. For each temperature step:
   a. Perform L Markov chain neighborhood moves.
   b. Generate neighbor x' and evaluate F(x') and CV(x').
   c. Apply multi-objective Boltzmann dominance acceptance rule.
   d. If accepted, update current state and external archive.
   e. Cool temperature T = alpha * T.
4. Return non-dominated archive front.

Default Parameters:
- T0: 10.0, T_min: 1e-3, alpha: 0.95
- chain_length (L): 40
- archive_size: 100
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float, float]:
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(problem.xl, problem.xu)
    f0, cv0 = problem.evaluate(x0)
    T0 = 10.0
    return x0, f0, cv0, T0


def update_archive(candidate_x: np.ndarray, candidate_f: np.ndarray, candidate_cv: float,
                   arc_X: np.ndarray, arc_F: np.ndarray, arc_CV: np.ndarray,
                   max_size: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(arc_X) == 0:
        return np.array([candidate_x]), np.array([candidate_f]), np.array([candidate_cv])

    comb_X = np.vstack([arc_X, candidate_x])
    comb_F = np.vstack([arc_F, candidate_f])
    comb_CV = np.append(arc_CV, candidate_cv)

    fronts = mu.fast_non_dominated_sort(comb_F, comb_CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    nd_X = comb_X[nd_indices]
    nd_F = comb_F[nd_indices]
    nd_CV = comb_CV[nd_indices]

    if len(nd_X) > max_size:
        cd = mu.calculate_crowding_distance(nd_F, list(range(len(nd_X))))
        sorted_idx = np.argsort(-cd)
        chosen = sorted_idx[:max_size]
        return nd_X[chosen], nd_F[chosen], nd_CV[chosen]

    return nd_X, nd_F, nd_CV


def step(curr_x: np.ndarray, curr_f: np.ndarray, curr_cv: float,
         arc_X: np.ndarray, arc_F: np.ndarray, arc_CV: np.ndarray,
         T: float, problem: Any, chain_length: int = 40,
         obj_ranges: Optional[np.ndarray] = None,
         archive_size: int = 100) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Executes one temperature cycle (Markov chain) of MOSA."""
    if obj_ranges is None:
        obj_ranges = np.ones(problem.n_objs)

    step_sigma = 0.05 * (problem.xu - problem.xl)

    for _ in range(chain_length):
        # Generate Gaussian neighbor
        cand_x = curr_x + np.random.normal(0.0, step_sigma)
        cand_x = np.clip(cand_x, problem.xl, problem.xu)
        cand_f, cand_cv = problem.evaluate(cand_x)

        # Acceptance check
        accept = False
        if mu.deb_constrained_dominates(cand_f, cand_cv, curr_f, curr_cv):
            accept = True
        elif mu.deb_constrained_dominates(curr_f, curr_cv, cand_f, cand_cv):
            # Infeasible or dominated -> evaluate energy penalty
            if curr_cv <= 1e-6 and cand_cv > 1e-6:
                delta_E = 1e3 * cand_cv
            elif curr_cv > 1e-6 and cand_cv > 1e-6:
                delta_E = cand_cv - curr_cv
            else:
                diff = (cand_f - curr_f) / obj_ranges
                delta_E = np.max(diff)

            prob = np.exp(-max(0.0, delta_E) / max(1e-6, T))
            accept = bool(np.random.rand() < prob)
        else:
            # Mutually non-dominated
            accept = True

        if accept:
            curr_x = cand_x
            curr_f = cand_f
            curr_cv = cand_cv
            arc_X, arc_F, arc_CV = update_archive(cand_x, cand_f, cand_cv,
                                                  arc_X, arc_F, arc_CV, max_size=archive_size)

    return curr_x, curr_f, curr_cv, arc_X, arc_F, arc_CV


def run(problem: Any, T0: float = 10.0, alpha: float = 0.95,
        chain_length: int = 40, n_steps: int = 50,
        archive_size: int = 100, seed: int = 42) -> Dict[str, Any]:
    """Runs MOSA."""
    np.random.seed(seed)
    curr_x, curr_f, curr_cv, T = initialize(problem, seed=seed)
    arc_X, arc_F, arc_CV = np.array([curr_x]), np.array([curr_f]), np.array([curr_cv])

    # Sample to estimate objective ranges
    samples = np.random.uniform(problem.xl, problem.xu, size=(30, problem.n_vars))
    F_samples = np.array([problem.evaluate(s)[0] for s in samples])
    obj_ranges = np.ptp(F_samples, axis=0)
    obj_ranges[obj_ranges < 1e-6] = 1.0

    T_curr = T0
    for step_idx in range(n_steps):
        curr_x, curr_f, curr_cv, arc_X, arc_F, arc_CV = step(
            curr_x, curr_f, curr_cv, arc_X, arc_F, arc_CV,
            T_curr, problem, chain_length=chain_length,
            obj_ranges=obj_ranges, archive_size=archive_size
        )
        T_curr *= alpha

    fronts = mu.fast_non_dominated_sort(arc_F, arc_CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MOSA',
        'pop': arc_X,
        'F': arc_F,
        'CV': arc_CV,
        'X_front': arc_X[nd_indices],
        'F_front': arc_F[nd_indices],
        'evaluations': 1 + chain_length * n_steps
    }

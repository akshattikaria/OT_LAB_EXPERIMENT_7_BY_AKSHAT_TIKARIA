"""
alg35_r_nsga2.py - Reference Point based NSGA-II (R-NSGA-II)
============================================================
Family: Preference-based Evolutionary (Module 5E)

Core Mechanism:
Proposed by Deb & Sundar (IEEE TEVC, 2006).
Incorporates Decision-Maker (DM) preference through one or more target Reference Points z_ref:
1. Preference Distance Metric:
   Calculates normalized Euclidean distance to the reference point:
       d(x, z_ref) = sqrt( sum_m [ (f_m(x) - z_ref_m) / (f_max_m - f_min_m) ]^2 )
   Solutions closer to z_ref are preferred within the same non-dominated front.
2. Epsilon-Clearing Mechanism:
   To prevent crowding into duplicate copies at the single nearest point,
   an epsilon-clearing parameter (eps_clear) enforces a minimum spread:
   Any candidate within distance eps_clear from an already selected individual
   is demoted / cleared.
This focuses computational resources exclusively on the Region of Interest (ROI)
specified by the decision maker.

Lecture Steps:
1. Elicit DM reference point z_ref and epsilon-clearing threshold eps_clear.
2. Initialize population P_0 of size N.
3. In each generation:
   a. Create offspring Q_t via crossover and mutation.
   b. Combine R_t = P_t U Q_t and perform non-dominated sorting.
   c. For each front, rank individuals by ascending preference distance d(x, z_ref).
   d. Apply epsilon-clearing to select non-redundant solutions closest to z_ref.
4. Return preferred Pareto front region.

Default Parameters:
- pop_size: 100
- n_gen: 50
- eps_clear: 0.05 (in normalized objective space)
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu
from operators import sbx_crossover, polynomial_mutation


def initialize(problem: Any, pop_size: int = 100,
               z_ref: Optional[np.ndarray] = None, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pop = rng.uniform(problem.xl, problem.xu, size=(pop_size, problem.n_vars))
    F_list, CV_list = [], []
    for x in pop:
        f, cv = problem.evaluate(x)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list)
    CV = np.array(CV_list)

    if z_ref is None:
        if hasattr(problem, 'dm_reference'):
            # Convert dm_reference to minimization if needed
            # In P14 dm_reference is [1.8, 12.0], f is [weight, -runtime] => [1.8, -12.0]
            z_ref = np.array([problem.dm_reference[0], -problem.dm_reference[1]])
        else:
            z_ref = np.min(F, axis=0)

    return pop, F, CV, z_ref


def preference_distance(F: np.ndarray, z_ref: np.ndarray) -> np.ndarray:
    """Computes normalized Euclidean distance to reference point z_ref."""
    F_norm, ideal, nadir = mu.normalize_objectives(F)
    denom = nadir - ideal
    denom[denom == 0] = 1.0
    z_ref_norm = (z_ref - ideal) / denom

    dists = np.sqrt(np.sum((F_norm - z_ref_norm) ** 2, axis=1))
    return dists


def epsilon_clearing(front_indices: List[int], F_norm: np.ndarray,
                     dists: np.ndarray, eps_clear: float = 0.05) -> List[int]:
    """Sorts front by distance to z_ref and applies epsilon-clearing."""
    sorted_idx = [front_indices[i] for i in np.argsort(dists[front_indices])]
    selected = []

    for cand in sorted_idx:
        # Check distance to already selected individuals
        if len(selected) == 0:
            selected.append(cand)
            continue

        pair_dists = np.sqrt(np.sum((F_norm[selected] - F_norm[cand]) ** 2, axis=1))
        if np.all(pair_dists >= eps_clear):
            selected.append(cand)

    # If any remain that were cleared, append at end as fallback
    for cand in sorted_idx:
        if cand not in selected:
            selected.append(cand)

    return selected


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         z_ref: np.ndarray, problem: Any, eps_clear: float = 0.05,
         pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generational step of R-NSGA-II."""
    N = len(pop)

    # Generate offspring
    offspring = []
    for _ in range(0, N, 2):
        i1, i2 = np.random.choice(N, size=2, replace=False)
        c1, c2 = sbx_crossover(pop[i1], pop[i2], problem.xl, problem.xu, eta_c=eta_c, pc=pc)
        c1 = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=eta_m)
        c2 = polynomial_mutation(c2, problem.xl, problem.xu, eta_m=eta_m)
        offspring.append(c1)
        if len(offspring) < N:
            offspring.append(c2)

    offspring_pop = np.array(offspring[:N])
    off_evals = [problem.evaluate(x) for x in offspring_pop]
    off_F = np.array([e[0] for e in off_evals])
    off_CV = np.array([e[1] for e in off_evals])

    R_pop = np.vstack([pop, offspring_pop])
    R_F = np.vstack([F, off_F])
    R_CV = np.concatenate([CV, off_CV])

    fronts = mu.fast_non_dominated_sort(R_F, R_CV)
    dists = preference_distance(R_F, z_ref)
    F_norm, _, _ = mu.normalize_objectives(R_F)

    new_indices = []
    for front in fronts:
        if len(new_indices) + len(front) <= N:
            ordered_front = epsilon_clearing(front, F_norm, dists, eps_clear=eps_clear)
            new_indices.extend(ordered_front)
        else:
            needed = N - len(new_indices)
            ordered_front = epsilon_clearing(front, F_norm, dists, eps_clear=eps_clear)
            new_indices.extend(ordered_front[:needed])
            break

    return R_pop[new_indices[:N]], R_F[new_indices[:N]], R_CV[new_indices[:N]]


def run(problem: Any, pop_size: int = 100, n_gen: int = 50,
        z_ref: Optional[np.ndarray] = None, eps_clear: float = 0.05,
        seed: int = 42) -> Dict[str, Any]:
    """Runs R-NSGA-II."""
    np.random.seed(seed)
    pop, F, CV, z_ref = initialize(problem, pop_size=pop_size, z_ref=z_ref, seed=seed)

    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, z_ref, problem, eps_clear=eps_clear)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'R-NSGA-II',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'z_ref': z_ref,
        'evaluations': pop_size * (n_gen + 1)
    }

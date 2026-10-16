r"""
alg33_sms_emoa.py - S-Metric Selection EMOA (SMS-EMOA)
=====================================================
Family: Indicator-based Evolutionary (Module 5E)

Core Mechanism:
Proposed by Beume, Naujoks, & Emmerich (EJOR 2007).
Directly maximizes the Hypervolume (S-metric) of the population using a steady-state
(mu + 1) evolutionary framework:
1. Steady-State Variation:
   In each step, exactly ONE offspring solution is generated via tournament selection,
   SBX crossover, and polynomial mutation, and appended to the population (size N + 1).
2. Fast Non-Dominated Sorting:
   Partitions the (N + 1) solutions into layered fronts F_1, ..., F_K.
3. Hypervolume Contribution Reduction:
   Examines the worst front F_K:
   - If |F_K| > 1: computes the exclusive Hypervolume Contribution of each individual:
         Delta S(x_i) = S(F_K) - S(F_K \ {x_i})
     and discards the individual with the MINIMAL hypervolume contribution.
   - If |F_K| == 1: discards that single dominated individual.
Guarantees strict monotonic improvement in population hypervolume.

Lecture Steps:
1. Initialize population P_0 of size N. Evaluate objectives and constraints.
2. In each steady-state step:
   a. Select two parents via tournament selection.
   b. Produce one offspring y via crossover and mutation.
   c. Evaluate y and add to population (size N + 1).
   d. Perform fast non-dominated sorting.
   e. Identify worst front F_K.
   f. Discard the solution in F_K having the smallest exclusive hypervolume contribution.
3. Repeat for the allocated evaluation budget.
4. Return non-dominated Pareto front.

Default Parameters:
- pop_size: 50
- n_evals: 2000
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu
from operators import sbx_crossover, polynomial_mutation


def initialize(problem: Any, pop_size: int = 50, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pop = rng.uniform(problem.xl, problem.xu, size=(pop_size, problem.n_vars))
    F_list, CV_list = [], []
    for x in pop:
        f, cv = problem.evaluate(x)
        F_list.append(f)
        CV_list.append(cv)
    return pop, np.array(F_list), np.array(CV_list)


def compute_hv_contributions_2d(front_F: np.ndarray, ref_point: np.ndarray) -> np.ndarray:
    """Computes exclusive 2D hypervolume contributions for all points in a non-dominated front."""
    l = len(front_F)
    if l <= 1:
        return np.ones(l)

    # Sort ascending by f1 (so f2 descends)
    sorted_idx = np.argsort(front_F[:, 0])
    sorted_F = front_F[sorted_idx]

    contributions = np.zeros(l)
    for i in range(l):
        # Left boundary
        left_f1 = sorted_F[i - 1, 0] if i > 0 else 0.0 # or ideal
        # Right boundary
        right_f1 = sorted_F[i + 1, 0] if i < (l - 1) else ref_point[0]
        # Below boundary in f2
        below_f2 = sorted_F[i + 1, 1] if i < (l - 1) else ref_point[1]

        w = right_f1 - sorted_F[i, 0]
        h = below_f2 - sorted_F[i, 1]
        contributions[sorted_idx[i]] = max(0.0, w) * max(0.0, h)

    return contributions


def reduce_one(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
               ref_point: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discards exactly one individual from population of size N+1 using SMS-EMOA rule."""
    fronts = mu.fast_non_dominated_sort(F, CV)
    worst_front = fronts[-1]

    if len(worst_front) == 1:
        discard_idx = worst_front[0]
    else:
        # Calculate HV contribution in normalized space
        F_norm, _, _ = mu.normalize_objectives(F)
        sub_F = F_norm[worst_front]
        ref_norm = np.ones(F.shape[1]) * 1.1

        if F.shape[1] == 2:
            hv_contribs = compute_hv_contributions_2d(sub_F, ref_norm)
        else:
            # Monte Carlo / box approximation for M >= 3
            hv_contribs = np.zeros(len(worst_front))
            total_hv = mu.hypervolume_monte_carlo(sub_F, ref_norm, n_samples=2000)
            for i in range(len(worst_front)):
                rem_F = np.delete(sub_F, i, axis=0)
                hv_rem = mu.hypervolume_monte_carlo(rem_F, ref_norm, n_samples=2000)
                hv_contribs[i] = total_hv - hv_rem

        min_pos = np.argmin(hv_contribs)
        discard_idx = worst_front[min_pos]

    new_pop = np.delete(pop, discard_idx, axis=0)
    new_F = np.delete(F, discard_idx, axis=0)
    new_CV = np.delete(CV, discard_idx)
    return new_pop, new_F, new_CV


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any, ref_point: np.ndarray,
         pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One steady-state (mu + 1) step of SMS-EMOA."""
    N = len(pop)
    i1, i2 = np.random.choice(N, size=2, replace=False)

    c1, _ = sbx_crossover(pop[i1], pop[i2], problem.xl, problem.xu, eta_c=eta_c, pc=pc)
    c1 = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=eta_m)

    f_c1, cv_c1 = problem.evaluate(c1)

    comb_pop = np.vstack([pop, c1])
    comb_F = np.vstack([F, f_c1])
    comb_CV = np.append(CV, cv_c1)

    return reduce_one(comb_pop, comb_F, comb_CV, ref_point)


def run(problem: Any, pop_size: int = 50, n_steps: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Runs SMS-EMOA."""
    np.random.seed(seed)
    pop, F, CV = initialize(problem, pop_size=pop_size, seed=seed)
    ref_point = getattr(problem, 'ref_point', np.ones(problem.n_objs) * 1.5)

    for st in range(n_steps):
        pop, F, CV = step(pop, F, CV, problem, ref_point)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'SMS-EMOA',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'evaluations': pop_size + n_steps
    }

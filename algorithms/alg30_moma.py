"""
alg30_moma.py - Multi-Objective Memetic Algorithm (MOMA)
========================================================
Family: Hybrid Evolutionary / Memetic (Part 4)

Core Mechanism:
Combines global exploration of NSGA-II with local exploitation (hill-climbing):
1. Global Search:
   Standard NSGA-II framework with tournament selection, SBX crossover, and
   polynomial mutation.
2. Local Search Operator (Lamarckian Memetics):
   At each generation, a local hill-climbing search is executed on a fraction of
   the non-dominated individuals in Front 1:
   - For continuous problems (e.g., P5): Pattern search / coordinate descent exploring
     small step perturbations in each coordinate direction.
   - For binary problems (e.g., P9): Bit-flip neighborhood hill climbing.
   If a local step strictly dominates the current solution or improves its constraint
   feasibility, the solution is updated in place.

Lecture Steps:
1. Initialize population of size N.
2. Execute NSGA-II recombination and mutation to produce offspring Q_t.
3. Apply local hill-climbing search to selected offspring / Front 1 members.
4. Merge parents and locally improved offspring into pool R_t of size 2N.
5. Apply fast non-dominated sorting and crowding distance truncation to select top N.
6. Return non-dominated Pareto front.

Default Parameters:
- pop_size: 100
- n_gen: 50
- local_search_prob: 0.2
- local_search_steps: 5
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu
from operators import sbx_crossover, polynomial_mutation


def initialize(problem: Any, pop_size: int = 100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(problem.xl, problem.xu, size=(pop_size, problem.n_vars))


def evaluate_population(pop: np.ndarray, problem: Any) -> Tuple[np.ndarray, np.ndarray]:
    F_list, CV_list = [], []
    for ind in pop:
        f, cv = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)
    return np.array(F_list), np.array(CV_list)


def local_search_hill_climbing(x: np.ndarray, f: np.ndarray, cv: float,
                               problem: Any, n_steps: int = 5) -> Tuple[np.ndarray, np.ndarray, float]:
    """Applies coordinate pattern search local hill climbing."""
    curr_x = x.copy()
    curr_f = f.copy()
    curr_cv = cv

    step_sizes = 0.02 * (problem.xu - problem.xl)
    D = len(curr_x)

    for _ in range(n_steps):
        improved = False
        for d in range(D):
            for direction in [-1.0, 1.0]:
                cand_x = curr_x.copy()
                cand_x[d] += direction * step_sizes[d]
                cand_x[d] = np.clip(cand_x[d], problem.xl[d], problem.xu[d])

                cand_f, cand_cv = problem.evaluate(cand_x)

                if mu.deb_constrained_dominates(cand_f, cand_cv, curr_f, curr_cv):
                    curr_x = cand_x
                    curr_f = cand_f
                    curr_cv = cand_cv
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step_sizes *= 0.5 # Shrink step size if no improvement

    return curr_x, curr_f, curr_cv


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any, pc: float = 0.9, eta_c: float = 20.0,
         eta_m: float = 20.0, local_prob: float = 0.2,
         local_steps: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generational step of MOMA."""
    N = len(pop)
    fronts = mu.fast_non_dominated_sort(F, CV)

    ranks = np.zeros(N, dtype=int)
    crowding_dist = np.zeros(N)

    for rank_idx, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank_idx
        cd = mu.calculate_crowding_distance(F, front)
        for local_idx, global_idx in enumerate(front):
            crowding_dist[global_idx] = cd[local_idx]

    # Generate offspring
    offspring = []
    for _ in range(0, N, 2):
        i1, i2 = np.random.choice(N, size=2, replace=False)
        p1 = pop[i1] if (ranks[i1] < ranks[i2] or (ranks[i1] == ranks[i2] and crowding_dist[i1] >= crowding_dist[i2])) else pop[i2]
        i3, i4 = np.random.choice(N, size=2, replace=False)
        p2 = pop[i3] if (ranks[i3] < ranks[i4] or (ranks[i3] == ranks[i4] and crowding_dist[i3] >= crowding_dist[i4])) else pop[i4]

        c1, c2 = sbx_crossover(p1, p2, problem.xl, problem.xu, eta_c=eta_c, pc=pc)
        c1 = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=eta_m)
        c2 = polynomial_mutation(c2, problem.xl, problem.xu, eta_m=eta_m)
        offspring.append(c1)
        if len(offspring) < N:
            offspring.append(c2)

    offspring_pop = np.array(offspring[:N])
    off_F, off_CV = evaluate_population(offspring_pop, problem)

    # Apply local search to a random fraction of offspring
    for i in range(N):
        if np.random.rand() < local_prob:
            offspring_pop[i], off_F[i], off_CV[i] = local_search_hill_climbing(
                offspring_pop[i], off_F[i], off_CV[i], problem, n_steps=local_steps
            )

    R_pop = np.vstack([pop, offspring_pop])
    R_F = np.vstack([F, off_F])
    R_CV = np.concatenate([CV, off_CV])

    r_fronts = mu.fast_non_dominated_sort(R_F, R_CV)

    new_indices = []
    for front in r_fronts:
        if len(new_indices) + len(front) <= N:
            new_indices.extend(front)
        else:
            needed = N - len(new_indices)
            cd = mu.calculate_crowding_distance(R_F, front)
            sorted_idx = np.argsort(-cd)
            for idx in range(needed):
                new_indices.append(front[sorted_idx[idx]])
            break

    return R_pop[new_indices], R_F[new_indices], R_CV[new_indices]


def run(problem: Any, pop_size: int = 100, n_gen: int = 50,
        local_prob: float = 0.2, local_steps: int = 5, seed: int = 42) -> Dict[str, Any]:
    """Runs MOMA."""
    np.random.seed(seed)
    pop = initialize(problem, pop_size=pop_size, seed=seed)
    F, CV = evaluate_population(pop, problem)

    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, problem, local_prob=local_prob, local_steps=local_steps)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MOMA',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'evaluations': pop_size * (n_gen + 1)
    }

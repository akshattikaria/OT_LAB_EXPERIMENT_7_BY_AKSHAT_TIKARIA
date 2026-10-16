"""
alg14_nsga2.py - Non-dominated Sorting Genetic Algorithm II (NSGA-II)
=====================================================================
Family: Elitist Evolutionary (Part 3)

Core Mechanism:
Proposed by Deb, Pratap, Agarwal, & Meyarivan (2002).
Key innovations over original NSGA:
1. Fast Non-Dominated Sorting with O(M * N^2) computational complexity.
2. Crowding Distance operator for explicit diversity preservation without
   requiring user-defined sharing parameters (sigma_share).
3. (N + N) Elitism: Combines parent population P_t and offspring Q_t (size 2N).
   Selects the best N individuals front by front, using descending crowding
   distance to truncate the final splitting front.
4. Binary tournament selection using the Crowded Comparison Operator (<_n):
   individual i wins if (rank_i < rank_j) or (rank_i == rank_j and cd_i > cd_j).

Lecture Steps:
1. Initialize population P_0 of size N. Evaluate objectives and constraints.
2. Sort P_0 into non-dominated fronts and assign crowding distances.
3. Generate offspring population Q_0 of size N via tournament selection, SBX, and mutation.
4. For generation t = 1 to G:
   a. Combine R_t = P_t U Q_t (size 2N).
   b. Perform non-dominated sort on R_t to get fronts F_1, F_2, ...
   c. Fill P_{t+1} with fronts F_1, F_2, ... until size exceeds N.
   d. In the critical front F_L, calculate crowding distance and select the
      individuals with highest crowding distance to fill remaining slots.
   e. Create offspring Q_{t+1} using crowded tournament selection, SBX, and mutation.
5. Return non-dominated front.

Default Parameters:
- pop_size: 100
- n_gen: 50
- pc: 0.9, eta_c: 20.0
- pm: 1/n_vars, eta_m: 20.0
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


def crowded_tournament_select(pop: np.ndarray, ranks: np.ndarray,
                              crowding_dist: np.ndarray) -> np.ndarray:
    """Selects one individual via crowded tournament selection."""
    N = len(pop)
    i1, i2 = np.random.choice(N, size=2, replace=False)
    # Deb's crowded comparison operator
    if ranks[i1] < ranks[i2]:
        return pop[i1].copy()
    elif ranks[i2] < ranks[i1]:
        return pop[i2].copy()
    else:
        # Same rank: higher crowding distance wins
        return pop[i1].copy() if crowding_dist[i1] >= crowding_dist[i2] else pop[i2].copy()


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any, pc: float = 0.9, eta_c: float = 20.0,
         eta_m: float = 20.0,
         use_constraints: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One generational step of NSGA-II with (N+N) elitism.

    Args:
        use_constraints: If False (default), plain Pareto dominance is used,
            ignoring constraint violations (CV). Set True to activate Deb's
            constrained-dominance rule (same as alg21 Constrained NSGA-II).
    """
    N = len(pop)
    fronts = mu.fast_non_dominated_sort(F, CV if use_constraints else None)

    ranks = np.zeros(N, dtype=int)
    crowding_dist = np.zeros(N)

    for rank_idx, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = rank_idx
        cd = mu.calculate_crowding_distance(F, front)
        for local_idx, global_idx in enumerate(front):
            crowding_dist[global_idx] = cd[local_idx]

    # Generate N offspring
    offspring = []
    for _ in range(0, N, 2):
        p1 = crowded_tournament_select(pop, ranks, crowding_dist)
        p2 = crowded_tournament_select(pop, ranks, crowding_dist)
        c1, c2 = sbx_crossover(p1, p2, problem.xl, problem.xu, eta_c=eta_c, pc=pc)
        c1 = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=eta_m)
        c2 = polynomial_mutation(c2, problem.xl, problem.xu, eta_m=eta_m)
        offspring.append(c1)
        if len(offspring) < N:
            offspring.append(c2)

    offspring_pop = np.array(offspring[:N])
    off_F, off_CV = evaluate_population(offspring_pop, problem)

    # Combine parents and offspring: R_t = P_t U Q_t (size 2N)
    R_pop = np.vstack([pop, offspring_pop])
    R_F = np.vstack([F, off_F])
    R_CV = np.concatenate([CV, off_CV])

    # Non-dominated sort on 2N population
    r_fronts = mu.fast_non_dominated_sort(R_F, R_CV if use_constraints else None)

    new_pop_indices = []
    for front in r_fronts:
        if len(new_pop_indices) + len(front) <= N:
            new_pop_indices.extend(front)
        else:
            # Need to select remaining slots based on crowding distance
            needed = N - len(new_pop_indices)
            cd = mu.calculate_crowding_distance(R_F, front)
            # Sort front descending by crowding distance
            sorted_front_idx = np.argsort(-cd)
            for i in range(needed):
                new_pop_indices.append(front[sorted_front_idx[i]])
            break

    new_pop = R_pop[new_pop_indices]
    new_F = R_F[new_pop_indices]
    new_CV = R_CV[new_pop_indices]
    return new_pop, new_F, new_CV


def run(problem: Any, pop_size: int = 100, n_gen: int = 50, seed: int = 42,
        use_constraints: bool = False) -> Dict[str, Any]:
    """Runs NSGA-II across all generations.

    Args:
        use_constraints: If False (default), plain Pareto dominance ignoring CV.
            Set True for Deb's constrained-dominance (same behaviour as alg21).
    """
    np.random.seed(seed)
    pop = initialize(problem, pop_size=pop_size, seed=seed)
    F, CV = evaluate_population(pop, problem)

    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, problem, use_constraints=use_constraints)

    fronts = mu.fast_non_dominated_sort(F, CV if use_constraints else None)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'NSGA-II',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'evaluations': pop_size * (n_gen + 1)
    }

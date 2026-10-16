"""
alg21_constrained_nsga2.py - Constrained NSGA-II with Deb's Feasibility Rules
=============================================================================
Family: Constrained Evolutionary (Part 4)

Core Mechanism:
Proposed by Kalyanmoy Deb (2000, 2002).
Explicitly integrates Deb's three constraint-handling rules into the Crowded Comparison
Operator (<_n) and Non-Dominated Sorting without requiring penalty factors:
1. Between two feasible solutions: the solution that Pareto-dominates the other wins.
   If mutually non-dominated, tie-break by larger crowding distance.
2. Between one feasible and one infeasible solution: the FEASIBLE solution ALWAYS wins.
3. Between two infeasible solutions: the solution with the SMALLER total constraint
   violation (CV) wins, irrespective of their objective values.

Lecture Steps:
1. Initialize population P_0.
2. Evaluate objective values F and exact constraint violation CV = sum max(0, g_j(x)).
3. Perform non-dominated sort utilizing Deb's feasibility-first comparison rules.
4. Calculate crowding distances for individuals within each front.
5. Apply crowded tournament selection respecting Deb's feasibility comparison.
6. Generate offspring via SBX crossover and polynomial mutation.
7. Merge P_t and Q_t into R_t (2N) and select top N using Deb-sorted fronts and crowding distance.

Default Parameters:
- pop_size: 100
- n_gen: 50
- pc: 0.9, eta_c: 20.0, pm: 1/n_vars, eta_m: 20.0
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


def deb_tournament_select(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
                          ranks: np.ndarray, cd: np.ndarray) -> np.ndarray:
    """Selects one individual via Deb's constrained comparison operator."""
    N = len(pop)
    i1, i2 = np.random.choice(N, size=2, replace=False)

    # 1. Check Deb's dominance
    if mu.deb_constrained_dominates(F[i1], CV[i1], F[i2], CV[i2]):
        return pop[i1].copy()
    elif mu.deb_constrained_dominates(F[i2], CV[i2], F[i1], CV[i1]):
        return pop[i2].copy()
    else:
        # Same rank/status: tie break by crowding distance or smaller CV
        if CV[i1] > 0 and CV[i2] > 0:
            return pop[i1].copy() if CV[i1] <= CV[i2] else pop[i2].copy()
        if ranks[i1] < ranks[i2]:
            return pop[i1].copy()
        elif ranks[i2] < ranks[i1]:
            return pop[i2].copy()
        else:
            return pop[i1].copy() if cd[i1] >= cd[i2] else pop[i2].copy()


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any, pc: float = 0.9, eta_c: float = 20.0,
         eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generational step of Constrained NSGA-II."""
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
        p1 = deb_tournament_select(pop, F, CV, ranks, crowding_dist)
        p2 = deb_tournament_select(pop, F, CV, ranks, crowding_dist)
        c1, c2 = sbx_crossover(p1, p2, problem.xl, problem.xu, eta_c=eta_c, pc=pc)
        c1 = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=eta_m)
        c2 = polynomial_mutation(c2, problem.xl, problem.xu, eta_m=eta_m)
        offspring.append(c1)
        if len(offspring) < N:
            offspring.append(c2)

    offspring_pop = np.array(offspring[:N])
    off_F, off_CV = evaluate_population(offspring_pop, problem)

    R_pop = np.vstack([pop, offspring_pop])
    R_F = np.vstack([F, off_F])
    R_CV = np.concatenate([CV, off_CV])

    r_fronts = mu.fast_non_dominated_sort(R_F, R_CV)

    new_pop_indices = []
    for front in r_fronts:
        if len(new_pop_indices) + len(front) <= N:
            new_pop_indices.extend(front)
        else:
            needed = N - len(new_pop_indices)
            cd = mu.calculate_crowding_distance(R_F, front)
            sorted_idx = np.argsort(-cd)
            for i in range(needed):
                new_pop_indices.append(front[sorted_idx[i]])
            break

    new_pop = R_pop[new_pop_indices]
    new_F = R_F[new_pop_indices]
    new_CV = R_CV[new_pop_indices]
    return new_pop, new_F, new_CV


def run(problem: Any, pop_size: int = 100, n_gen: int = 50, seed: int = 42) -> Dict[str, Any]:
    """Runs Constrained NSGA-II."""
    np.random.seed(seed)
    pop = initialize(problem, pop_size=pop_size, seed=seed)
    F, CV = evaluate_population(pop, problem)

    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, problem)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    # Ensure only feasible non-dominated points in primary front
    feas_front_idx = [idx for idx in nd_indices if CV[idx] <= 1e-6]
    if len(feas_front_idx) == 0:
        feas_front_idx = nd_indices

    return {
        'algorithm': 'Constrained NSGA-II',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[feas_front_idx],
        'F_front': F[feas_front_idx],
        'evaluations': pop_size * (n_gen + 1)
    }

"""
alg11_moga.py - Multi-Objective Genetic Algorithm (MOGA)
========================================================
Family: Non-elitist Evolutionary (Part 2)

Core Mechanism:
Proposed by Fonseca & Fleming (1993).
1. Individual rank = 1 + number of individuals dominating it (p_i).
2. Assign raw fitness inversely related to rank.
3. Individuals having the same rank share fitness via a niche-sharing function
   in objective space, preventing genetic drift and distributing solutions across the front.
4. Stochastic universal sampling / roulette-wheel selection based on shared fitness.
5. Non-elitist replacement by offspring.

Lecture Steps:
1. Initialize population of size N.
2. Evaluate objectives and constraint violations.
3. Rank determination: For each individual i, rank_i = 1 + sum_{j != i} I(j dominates i).
4. Raw fitness assignment: Sort by rank and assign interpolated fitness scores.
5. Objective-space fitness sharing:
   For each distinct rank, compute niche counts nc_i = sum_j sh(d_{ij}),
   and shared fitness = raw_fitness_i / nc_i.
6. Fitness-proportionate selection to fill mating pool.
7. Crossover (SBX) and Polynomial Mutation to generate N offspring.
8. Replace parent population.

Default Parameters:
- pop_size: 100
- n_gen: 50
- sigma_share: 0.10 (in normalized objective space)
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


def compute_moga_ranks(F: np.ndarray, CV: np.ndarray) -> np.ndarray:
    """Computes MOGA rank: 1 + number of individuals that dominate individual i."""
    N = len(F)
    ranks = np.ones(N, dtype=int)
    for i in range(N):
        for j in range(N):
            if i != j and mu.deb_constrained_dominates(F[j], CV[j], F[i], CV[i]):
                ranks[i] += 1
    return ranks


def share_fitness(ranks: np.ndarray, F_norm: np.ndarray,
                  sigma_share: float = 0.10) -> np.ndarray:
    """Applies fitness sharing within identical rank groups."""
    N = len(ranks)
    # Raw fitness: linearly decreasing with rank
    raw_fitness = N - ranks + 1.0
    shared_fitness = np.zeros(N)

    unique_ranks = np.unique(ranks)
    for r in unique_ranks:
        indices = np.where(ranks == r)[0]
        n_r = len(indices)
        if n_r == 1:
            shared_fitness[indices[0]] = raw_fitness[indices[0]]
            continue

        # Pairwise Euclidean distances in normalized objective space
        sub_F = F_norm[indices]
        dists = np.sqrt(np.sum((sub_F[:, None, :] - sub_F[None, :, :]) ** 2, axis=2))
        # Niche function: sh(d) = 1 - (d / sigma_share) if d < sigma_share else 0
        sh = np.maximum(0.0, 1.0 - (dists / sigma_share))
        niche_counts = np.sum(sh, axis=1)

        for idx_local, idx_global in enumerate(indices):
            shared_fitness[idx_global] = raw_fitness[idx_global] / max(1.0, niche_counts[idx_local])

    return shared_fitness


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any, sigma_share: float = 0.10,
         pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One generation of MOGA."""
    N = len(pop)
    ranks = compute_moga_ranks(F, CV)

    # Normalize F for distance sharing
    F_norm, _, _ = mu.normalize_objectives(F)
    fitness = share_fitness(ranks, F_norm, sigma_share=sigma_share)

    # Roulette-wheel selection based on shared fitness
    probs = fitness / np.sum(fitness)
    selected_idx = np.random.choice(N, size=N, p=probs, replace=True)
    mating_pool = pop[selected_idx]

    # Variation
    offspring = []
    for i in range(0, N, 2):
        p1 = mating_pool[i]
        p2 = mating_pool[(i + 1) % N]
        c1, c2 = sbx_crossover(p1, p2, problem.xl, problem.xu, eta_c=eta_c, pc=pc)
        c1 = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=eta_m)
        c2 = polynomial_mutation(c2, problem.xl, problem.xu, eta_m=eta_m)
        offspring.append(c1)
        if len(offspring) < N:
            offspring.append(c2)

    offspring_pop = np.array(offspring[:N])
    new_F, new_CV = evaluate_population(offspring_pop, problem)
    return offspring_pop, new_F, new_CV


def run(problem: Any, pop_size: int = 100, n_gen: int = 50,
        sigma_share: float = 0.10, seed: int = 42) -> Dict[str, Any]:
    """Runs MOGA across all generations."""
    np.random.seed(seed)
    pop = initialize(problem, pop_size=pop_size, seed=seed)
    F, CV = evaluate_population(pop, problem)

    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, problem, sigma_share=sigma_share)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MOGA',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'evaluations': pop_size * (n_gen + 1)
    }

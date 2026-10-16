"""
alg12_nsga.py - Non-dominated Sorting Genetic Algorithm (NSGA)
=============================================================
Family: Non-elitist Evolutionary (Part 2)

Core Mechanism:
Proposed by Srinivas & Deb (1994).
1. Classifies population into non-dominated fronts F_1, F_2, ... using non-dominated sorting.
2. Assigns a synthetic "dummy fitness" to each front, starting with F_1.
3. Within each front, performs fitness sharing to disperse individuals across the front.
4. Sets the dummy fitness of the subsequent front slightly lower than the minimum
   shared fitness of the previous front, ensuring that all individuals in F_k have
   higher fitness than any individual in F_{k+1}.
5. Non-elitist parent replacement by offspring.

Lecture Steps:
1. Initialize population of size N.
2. Evaluate objectives and constraints.
3. Perform non-dominated sorting to classify population into fronts F_1, ..., F_K.
4. For front F_1: set initial dummy fitness f_dummy = N.
5. For each front F_k:
   a. Compute niche counts nc_i = sum_{j in F_k} sh(d_{ij}).
   b. Shared fitness f'_i = f_dummy / nc_i.
   c. Set dummy fitness for next front: f_dummy_next = min_{i in F_k}(f'_i) - 0.1.
6. Roulette-wheel selection based on shared fitness.
7. SBX crossover and polynomial mutation.
8. Non-elitist replacement.

Default Parameters:
- pop_size: 100
- n_gen: 50
- sigma_share: 0.15
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


def assign_nsga_shared_fitness(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
                               sigma_share: float = 0.15) -> np.ndarray:
    """Computes NSGA shared dummy fitness across layered non-dominated fronts."""
    N = len(pop)
    fronts = mu.fast_non_dominated_sort(F, CV)
    shared_fitness = np.zeros(N)

    # Normalize pop in decision space for niche sharing
    # (or objective space; normalized objective space is standard)
    F_norm, _, _ = mu.normalize_objectives(F)

    current_dummy = float(N)
    epsilon_step = 0.1

    for front in fronts:
        if len(front) == 0:
            continue
        sub_F = F_norm[front]
        # Compute distances inside front
        dists = np.sqrt(np.sum((sub_F[:, None, :] - sub_F[None, :, :]) ** 2, axis=2))
        sh = np.maximum(0.0, 1.0 - (dists / sigma_share))
        niche_counts = np.sum(sh, axis=1)

        front_shared = current_dummy / np.maximum(1.0, niche_counts)
        for idx_local, idx_global in enumerate(front):
            shared_fitness[idx_global] = front_shared[idx_local]

        # Next front dummy fitness strictly lower than current front's minimum
        current_dummy = max(0.1, np.min(front_shared) - epsilon_step)

    return shared_fitness


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any, sigma_share: float = 0.15,
         pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One generation of NSGA."""
    N = len(pop)
    shared_fitness = assign_nsga_shared_fitness(pop, F, CV, sigma_share=sigma_share)

    # Roulette-wheel selection
    fit_sum = np.sum(shared_fitness)
    probs = shared_fitness / fit_sum if fit_sum > 0 else np.ones(N) / N
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
        sigma_share: float = 0.15, seed: int = 42) -> Dict[str, Any]:
    """Runs original NSGA across all generations."""
    np.random.seed(seed)
    pop = initialize(problem, pop_size=pop_size, seed=seed)
    F, CV = evaluate_population(pop, problem)

    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, problem, sigma_share=sigma_share)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'NSGA',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'evaluations': pop_size * (n_gen + 1)
    }

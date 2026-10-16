"""
alg10_vega.py - Vector Evaluated Genetic Algorithm (VEGA)
=========================================================
Family: Non-elitist Evolutionary (Part 2)

Core Mechanism:
Proposed by Schaffer (1985). At each generation, divides the selection phase
into M sub-populations of size N / M:
    Sub-population m is selected based exclusively on objective f_m.
The selected sub-populations are then merged, shuffled, and subjected to standard
crossover and mutation to produce the next generation.
VEGA is non-elitist (parents are replaced by offspring).
Known limitation: Speciation towards single-objective extrema ("extreme points"),
often failing to discover intermediate compromise solutions.

Lecture Steps:
1. Initialize population of size N randomly within decision bounds.
2. Evaluate all N individuals on all M objectives.
3. Partition selection:
   For m = 1 to M:
     Select N/M individuals using tournament/roulette based strictly on f_m.
4. Merge all selected individuals into a mating pool of size N.
5. Shuffle mating pool randomly.
6. Apply crossover (SBX) and mutation (Polynomial Mutation) to produce offspring.
7. Replace old population with offspring (non-elitist).
8. Repeat for G generations.

Default Parameters:
- pop_size: 100
- n_gen: 50
- pc: 0.9, pm: 1/n_vars
- eta_c: 20, eta_m: 20
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu
from operators import sbx_crossover, polynomial_mutation


def initialize(problem: Any, pop_size: int = 100, seed: int = 42) -> np.ndarray:
    """Generates initial random population within bounds."""
    rng = np.random.default_rng(seed)
    if hasattr(problem, 'xl') and len(problem.xl) > 0:
        return rng.uniform(problem.xl, problem.xu, size=(pop_size, problem.n_vars))
    return rng.random(size=(pop_size, problem.n_vars))


def evaluate_population(pop: np.ndarray, problem: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluates entire population on objectives and constraints."""
    F_list, CV_list = [], []
    for ind in pop:
        f, cv = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)
    return np.array(F_list), np.array(CV_list)


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any, pc: float = 0.9, eta_c: float = 20.0,
         eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Performs one generation of VEGA."""
    N = len(pop)
    M = F.shape[1]
    sub_size = N // M
    mating_pool = []

    # Penalized objectives: f_m + 1e4 * cv
    penalized_F = F + 1e4 * CV[:, None]

    for m in range(M):
        # Select sub_size individuals based strictly on objective m
        # Binary tournament selection on objective m
        for _ in range(sub_size):
            i1, i2 = np.random.choice(N, size=2, replace=False)
            winner = i1 if penalized_F[i1, m] < penalized_F[i2, m] else i2
            mating_pool.append(pop[winner].copy())

    # If N is not strictly divisible by M, pad with best on random objective
    while len(mating_pool) < N:
        m_rand = np.random.randint(M)
        best_idx = np.argmin(penalized_F[:, m_rand])
        mating_pool.append(pop[best_idx].copy())

    mating_pool = np.array(mating_pool)
    np.random.shuffle(mating_pool)

    # Variation: Crossover and Mutation
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


def run(problem: Any, pop_size: int = 100, n_gen: int = 50, seed: int = 42) -> Dict[str, Any]:
    """Runs VEGA across all generations."""
    np.random.seed(seed)
    pop = initialize(problem, pop_size=pop_size, seed=seed)
    F, CV = evaluate_population(pop, problem)

    history_best = []
    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, problem)
        history_best.append(np.min(F, axis=0))

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'VEGA',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'history': history_best,
        'evaluations': pop_size * (n_gen + 1)
    }

"""
alg13_npga.py - Niched Pareto Genetic Algorithm (NPGA)
======================================================
Family: Non-elitist Evolutionary (Part 2)

Core Mechanism:
Proposed by Horn, Nafplioits, & Goldberg (1994).
Implements Pareto Tournament Selection against a randomly sampled comparison set:
1. Two candidate solutions A and B are chosen at random from the population.
2. A comparison set C of size t_dom is randomly drawn from the population.
3. Both A and B are compared for Pareto dominance against members of C.
4. If one candidate is non-dominated by C and the other is dominated, the non-dominated
   candidate wins the tournament.
5. If both are non-dominated (or both dominated), tie-break using niche count in
   objective space: the candidate in the less crowded niche (lower niche count) wins.
6. Non-elitist replacement by offspring.

Lecture Steps:
1. Initialize population of size N.
2. Evaluate objectives and constraints.
3. Normalize objective values for distance calculations.
4. For each tournament in selection:
   a. Draw candidates i1, i2 and comparison set C of size t_dom.
   b. Test dominance against C.
   c. If tied, evaluate niche count nc = sum_j sh(d_{ij}).
   d. Select winner.
5. Apply SBX crossover and polynomial mutation.
6. Replace parent population.

Default Parameters:
- pop_size: 100
- n_gen: 50
- t_dom: 10 (size of comparison tournament set)
- sigma_share: 0.10
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


def tournament_npga(i1: int, i2: int, F: np.ndarray, CV: np.ndarray,
                    F_norm: np.ndarray, t_dom: int = 10,
                    sigma_share: float = 0.10) -> int:
    """Executes one NPGA Pareto tournament."""
    N = len(F)
    # Sample comparison set C
    c_indices = np.random.choice(N, size=min(t_dom, N), replace=False)

    # Check if i1 or i2 are dominated by any member of C
    i1_dominated = False
    i2_dominated = False

    for c in c_indices:
        if not i1_dominated and mu.deb_constrained_dominates(F[c], CV[c], F[i1], CV[i1]):
            i1_dominated = True
        if not i2_dominated and mu.deb_constrained_dominates(F[c], CV[c], F[i2], CV[i2]):
            i2_dominated = True

    if not i1_dominated and i2_dominated:
        return i1
    elif i1_dominated and not i2_dominated:
        return i2
    else:
        # Tie-break using niche count in normalized objective space
        d1 = np.sqrt(np.sum((F_norm - F_norm[i1]) ** 2, axis=1))
        d2 = np.sqrt(np.sum((F_norm - F_norm[i2]) ** 2, axis=1))

        nc1 = np.sum(np.maximum(0.0, 1.0 - (d1 / sigma_share)))
        nc2 = np.sum(np.maximum(0.0, 1.0 - (d2 / sigma_share)))

        return i1 if nc1 <= nc2 else i2


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any, t_dom: int = 10, sigma_share: float = 0.10,
         pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One generation of NPGA."""
    N = len(pop)
    F_norm, _, _ = mu.normalize_objectives(F)

    # Pareto tournament selection
    mating_pool = []
    for _ in range(N):
        i1, i2 = np.random.choice(N, size=2, replace=False)
        winner = tournament_npga(i1, i2, F, CV, F_norm, t_dom=t_dom, sigma_share=sigma_share)
        mating_pool.append(pop[winner].copy())
    mating_pool = np.array(mating_pool)

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
        t_dom: int = 10, sigma_share: float = 0.10, seed: int = 42) -> Dict[str, Any]:
    """Runs NPGA across all generations."""
    np.random.seed(seed)
    pop = initialize(problem, pop_size=pop_size, seed=seed)
    F, CV = evaluate_population(pop, problem)

    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, problem, t_dom=t_dom, sigma_share=sigma_share)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'NPGA',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'evaluations': pop_size * (n_gen + 1)
    }

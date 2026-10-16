"""
alg34_hype.py - Hypervolume Estimation Algorithm (HypE)
=======================================================
Family: Indicator-based / Many-Objective Evolutionary (Module 5E)

Core Mechanism:
Proposed by Bader & Zitzler (IEEE TEVC, 2011).
Computes Monte Carlo approximations of Hypervolume Contributions for
many-objective optimization (M >= 3):
1. Monte Carlo Sampling:
   Uniformly samples S points in the hyper-rectangle bounded by ideal point z*
   and reference point z_ref.
2. Shared Indicator Scoring:
   For each sample point dominated by k solutions in the splitting front F_L:
   each dominating solution receives a partial share alpha_k = (1/k) * sum_{j=1}^k (1/j).
3. Environmental Selection:
   Iteratively removes the solution with the lowest estimated hypervolume contribution,
   updating the remaining sample sharing weights, until exactly N solutions remain.

Lecture Steps:
1. Initialize population P_0 of size N.
2. In each generation:
   a. Create N offspring Q_t via SBX crossover and polynomial mutation.
   b. Combine R_t = P_t U Q_t and perform non-dominated sorting.
   c. Add non-dominated fronts F_1, ..., F_{L-1} until capacity N is approached.
   d. Within critical front F_L, compute Monte Carlo hypervolume scores using sample sharing.
   e. Truncate front F_L down to required slots.
3. Return non-dominated Pareto front.

Default Parameters:
- pop_size: 100
- n_gen: 50
- n_samples: 10000 (Monte Carlo samples)
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu
from operators import sbx_crossover, polynomial_mutation


def initialize(problem: Any, pop_size: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pop = rng.uniform(problem.xl, problem.xu, size=(pop_size, problem.n_vars))
    F_list, CV_list = [], []
    for x in pop:
        f, cv = problem.evaluate(x)
        F_list.append(f)
        CV_list.append(cv)
    return pop, np.array(F_list), np.array(CV_list)


def estimate_hype_fitness(sub_F: np.ndarray, n_samples: int = 10000,
                          ref_point: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimates Hypervolume contribution using Monte Carlo sampling."""
    N, M = sub_F.shape
    if N == 0:
        return np.array([])
    if N == 1:
        return np.array([1.0])

    F_norm, ideal, nadir = mu.normalize_objectives(sub_F)
    ref = np.ones(M) * 1.1

    # Uniform random samples in [0, 1.1]^M
    samples = np.random.uniform(0.0, 1.1, size=(n_samples, M))

    # Determine domination matrix: sample s is dominated by p if all p_m <= s_m
    # dom: (n_samples, N)
    dom = np.all(samples[:, None, :] >= F_norm[None, :, :], axis=2)
    # Number of individuals dominating each sample: (n_samples,)
    k_counts = np.sum(dom, axis=1)

    fitness = np.zeros(N)
    valid_samples = np.where(k_counts > 0)[0]

    for s_idx in valid_samples:
        k = k_counts[s_idx]
        alpha_k = (1.0 / k)
        dominating_inds = np.where(dom[s_idx])[0]
        fitness[dominating_inds] += alpha_k

    return fitness


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any, n_samples: int = 10000,
         pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generational step of HypE."""
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

    chosen_indices = []
    for front in fronts:
        if len(chosen_indices) + len(front) <= N:
            chosen_indices.extend(front)
        else:
            needed = N - len(chosen_indices)
            sub_F = R_F[front]
            fitness = estimate_hype_fitness(sub_F, n_samples=n_samples)
            sorted_front_idx = np.argsort(-fitness)
            for idx in range(needed):
                chosen_indices.append(front[sorted_front_idx[idx]])
            break

    return R_pop[chosen_indices[:N]], R_F[chosen_indices[:N]], R_CV[chosen_indices[:N]]


def run(problem: Any, pop_size: int = 100, n_gen: int = 50,
        n_samples: int = 10000, seed: int = 42) -> Dict[str, Any]:
    """Runs HypE."""
    np.random.seed(seed)
    pop, F, CV = initialize(problem, pop_size=pop_size, seed=seed)

    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, problem, n_samples=n_samples)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'HypE',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'evaluations': pop_size * (n_gen + 1)
    }

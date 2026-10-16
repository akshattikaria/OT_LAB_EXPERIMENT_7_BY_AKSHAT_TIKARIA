"""
alg16_spea2.py - Strength Pareto Evolutionary Algorithm 2 (SPEA2)
=================================================================
Family: Elitist Evolutionary (Part 3)

Core Mechanism:
Proposed by Zitzler, Laumanns, & Thiele (2001).
Maintains a regular population P of size N and an external elitist archive P_bar of size N_bar.
Key components:
1. Fine-grained fitness assignment:
   - Strength S(i): count of solutions in P U P_bar dominated by i.
   - Raw Fitness R(i): sum of strengths of all solutions dominating i.
     Non-dominated individuals achieve R(i) = 0.
   - Density estimation D(i): D(i) = 1 / (sigma_i^k + 2), where sigma_i^k is
     the Euclidean distance to the k-th nearest neighbor (k = sqrt(N + N_bar)).
   - Total Fitness: F(i) = R(i) + D(i) (lower is better).
2. Fixed-size environmental selection with archive truncation:
   - If non-dominated individuals exceed archive capacity N_bar, iteratively
     removes the individual having the minimum distance to any other archive member.
   - If fewer than N_bar, fills with the best dominated solutions.
3. Binary tournament selection on the external archive.

Lecture Steps:
1. Initialize population P_0 and empty archive P_bar_0.
2. Evaluate fitness of P_t U P_bar_t (Strength, Raw Fitness, k-th NN Density).
3. Copy non-dominated solutions to P_bar_{t+1}; truncate or pad to fixed size N_bar.
4. If termination condition met, return non-dominated solutions in archive.
5. Perform binary tournament selection on archive to fill mating pool.
6. Apply crossover (SBX) and mutation to form next population P_{t+1}.

Default Parameters:
- pop_size: 100
- archive_size: 100
- n_gen: 50
- pc: 0.9, pm: 1/n_vars
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu
from operators import sbx_crossover, polynomial_mutation


def initialize(problem: Any, pop_size: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pop = rng.uniform(problem.xl, problem.xu, size=(pop_size, problem.n_vars))
    # Archive starts empty
    archive = np.empty((0, problem.n_vars))
    return pop, archive


def evaluate_fitness(F_all: np.ndarray, CV_all: np.ndarray) -> np.ndarray:
    """Computes SPEA2 Strength, Raw Fitness, Density, and Total Fitness."""
    N_tot = len(F_all)
    if N_tot == 0:
        return np.array([])

    # 1. Vectorized Dominance Matrix: dom[i, j] is True iff i dominates j
    tol = 1e-9
    feas = CV_all <= tol
    infeas = ~feas
    r1 = feas[:, None] & infeas[None, :]
    r2 = infeas[:, None] & infeas[None, :] & (CV_all[:, None] < (CV_all[None, :] - tol))
    pareto = np.all(F_all[:, None, :] <= F_all[None, :, :], axis=2) & np.any(F_all[:, None, :] < F_all[None, :, :], axis=2)
    dom = r1 | r2 | (feas[:, None] & feas[None, :] & pareto)
    np.fill_diagonal(dom, False)

    # Strength S(i): number of solutions dominated by i
    S = np.sum(dom, axis=1)

    # Raw Fitness R(i): sum of strengths of solutions dominating i
    R = dom.T @ S

    # 3. Density D(i) via k-th nearest neighbor
    k = int(np.sqrt(N_tot))
    if k < 1:
        k = 1

    F_norm, _, _ = mu.normalize_objectives(F_all)
    dists = np.sqrt(np.sum((F_norm[:, None, :] - F_norm[None, :, :]) ** 2, axis=2))
    sorted_dists = np.sort(dists, axis=1) # 0th column is distance to self (0.0)
    sigma_k = sorted_dists[:, k]
    D = 1.0 / (sigma_k + 2.0)

    # 4. Total fitness (lower is better)
    return R + D


def truncate_archive(archive_pop: np.ndarray, archive_F: np.ndarray,
                     target_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively removes archive solutions with minimum pairwise distance."""
    if len(archive_pop) <= target_size:
        return archive_pop, archive_F
    F_norm, _, _ = mu.normalize_objectives(archive_F)
    remaining = list(range(len(archive_pop)))
    dists = np.sqrt(np.sum((F_norm[:, None, :] - F_norm[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)

    while len(remaining) > target_size:
        sub_dists = dists[np.ix_(remaining, remaining)]
        min_i, _ = np.unravel_index(np.argmin(sub_dists), sub_dists.shape)
        remaining.pop(min_i)

    return archive_pop[remaining], archive_F[remaining]


def environmental_selection(pop: np.ndarray, archive: np.ndarray,
                            F_pop: np.ndarray, CV_pop: np.ndarray,
                            F_arc: np.ndarray, CV_arc: np.ndarray,
                            archive_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Selects archive of size archive_size for next generation."""
    if len(archive) == 0:
        comb_pop = pop
        comb_F = F_pop
        comb_CV = CV_pop
    else:
        comb_pop = np.vstack([pop, archive])
        comb_F = np.vstack([F_pop, F_arc])
        comb_CV = np.concatenate([CV_pop, CV_arc])

    fitness = evaluate_fitness(comb_F, comb_CV)

    # Solutions with fitness < 1 are non-dominated
    nd_indices = np.where(fitness < 1.0)[0]

    if len(nd_indices) == archive_size:
        return comb_pop[nd_indices], comb_F[nd_indices], comb_CV[nd_indices]
    elif len(nd_indices) < archive_size:
        # Pad with best dominated solutions (lowest fitness >= 1)
        sorted_idx = np.argsort(fitness)
        chosen = sorted_idx[:archive_size]
        return comb_pop[chosen], comb_F[chosen], comb_CV[chosen]
    else:
        # Truncate
        nd_pop = comb_pop[nd_indices]
        nd_F = comb_F[nd_indices]
        nd_CV = comb_CV[nd_indices]
        tr_pop, tr_F = truncate_archive(nd_pop, nd_F, archive_size)
        # Match CV
        tr_CV = np.zeros(len(tr_pop))
        return tr_pop, tr_F, tr_CV


def step(pop: np.ndarray, archive: np.ndarray,
         F_pop: np.ndarray, CV_pop: np.ndarray,
         F_arc: np.ndarray, CV_arc: np.ndarray,
         problem: Any, archive_size: int = 100,
         pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One generational step of SPEA2."""
    new_arc, new_arc_F, new_arc_CV = environmental_selection(
        pop, archive, F_pop, CV_pop, F_arc, CV_arc, archive_size
    )

    arc_fitness = evaluate_fitness(new_arc_F, new_arc_CV)
    N = len(pop)

    # Binary tournament on archive
    mating_pool = []
    A_len = len(new_arc)
    for _ in range(N):
        i1, i2 = np.random.choice(A_len, size=2, replace=False)
        winner = i1 if arc_fitness[i1] < arc_fitness[i2] else i2
        mating_pool.append(new_arc[winner].copy())
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

    new_pop = np.array(offspring[:N])
    off_evals = [problem.evaluate(x) for x in new_pop]
    new_F = np.array([e[0] for e in off_evals])
    new_CV = np.array([e[1] for e in off_evals])

    return new_pop, new_arc, new_F, new_CV, new_arc_F, new_arc_CV


def run(problem: Any, pop_size: int = 100, archive_size: int = 100,
        n_gen: int = 50, seed: int = 42) -> Dict[str, Any]:
    """Runs SPEA2."""
    np.random.seed(seed)
    pop, archive = initialize(problem, pop_size=pop_size, seed=seed)
    pop_evals = [problem.evaluate(x) for x in pop]
    F_pop = np.array([e[0] for e in pop_evals])
    CV_pop = np.array([e[1] for e in pop_evals])
    F_arc = np.empty((0, problem.n_objs))
    CV_arc = np.empty(0)

    for gen in range(n_gen):
        pop, archive, F_pop, CV_pop, F_arc, CV_arc = step(
            pop, archive, F_pop, CV_pop, F_arc, CV_arc, problem, archive_size=archive_size
        )

    fronts = mu.fast_non_dominated_sort(F_arc, CV_arc)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'SPEA2',
        'pop': pop,
        'archive': archive,
        'F_archive': F_arc,
        'X_front': archive[nd_indices],
        'F_front': F_arc[nd_indices],
        'evaluations': pop_size * (n_gen + 1)
    }

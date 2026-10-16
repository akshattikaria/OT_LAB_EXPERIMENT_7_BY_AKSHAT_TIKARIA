"""
alg32_ibea.py - Indicator-Based Evolutionary Algorithm (IBEA)
============================================================
Family: Indicator-based Evolutionary (Module 5E)

Core Mechanism:
Proposed by Zitzler & Künzli (PPSN 2004).
Replaces Pareto dominance and explicit diversity operators with a mathematical
quality indicator (the binary additive epsilon-indicator I_{eps+}):
1. Pairwise Indicator:
   I_{eps+}(x_1, x_2) = max_m [ (f_m(x_1) - f_m(x_2)) / range_m ]
   Measures the minimum distance by which solution x_1 must be shifted to weakly dominate x_2.
2. Fitness Assignment:
   F(x_i) = sum_{x_j != x_i} - exp( - I_{eps+}(x_j, x_i) / kappa )
   where kappa > 0 is a scaling parameter. A higher value indicates higher fitness.
3. Environmental Selection:
   Combines parent and offspring (2N). Iteratively removes the individual with the
   lowest fitness, updating the fitness values of the remaining individuals, until
   population size N is achieved.
No explicit niching or sharing parameters are needed: the indicator naturally balances
convergence and spread.

Lecture Steps:
1. Initialize population P_0 of size N.
2. In each generation:
   a. Apply binary tournament selection, SBX crossover, and polynomial mutation to produce N offspring Q_t.
   b. Form combined pool R_t = P_t U Q_t (size 2N).
   c. Calculate pairwise indicator values and initial fitness scores for all individuals in R_t.
   d. While |R_t| > N:
      - Identify individual x* with minimum fitness.
      - Remove x* from R_t.
      - Update fitness scores of remaining members: F(x_i) += exp( - I_{eps+}(x*, x_i) / kappa ).
3. Return non-dominated front of the final population.

Default Parameters:
- pop_size: 100
- n_gen: 50
- kappa: 0.05
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


def compute_indicator_matrix(F: np.ndarray, CV: np.ndarray) -> np.ndarray:
    """Computes pairwise additive epsilon-indicator matrix."""
    N = len(F)
    F_norm, _, _ = mu.normalize_objectives(F)
    # Penalize constraint violations
    penalized_F = F_norm + 1e3 * CV[:, None]

    # I(x1, x2) = max_m (f_m(x1) - f_m(x2))
    # diff: (N, N, M) -> F[x1] - F[x2]
    diff = penalized_F[:, None, :] - penalized_F[None, :, :]
    indicator_mat = np.max(diff, axis=2)
    return indicator_mat


def compute_fitness(indicator_mat: np.ndarray, kappa: float = 0.05) -> np.ndarray:
    """Computes initial IBEA fitness values: F(x_i) = sum_{j != i} -exp(-I(x_j, x_i) / kappa)."""
    N = len(indicator_mat)
    # indicator_mat[j, i] is I(x_j, x_i)
    scaled = -indicator_mat / max(1e-6, kappa)
    exp_mat = np.exp(scaled)
    np.fill_diagonal(exp_mat, 0.0)
    fitness = -np.sum(exp_mat, axis=0) # sum over j
    return fitness


def environmental_selection(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
                            target_size: int, kappa: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively removes worst individuals until population reaches target_size."""
    N = len(pop)
    I_mat = compute_indicator_matrix(F, CV)
    fitness = compute_fitness(I_mat, kappa=kappa)

    active_indices = list(range(N))

    while len(active_indices) > target_size:
        # Find index with minimum fitness among active
        min_pos = np.argmin(fitness[active_indices])
        worst_idx = active_indices[min_pos]

        # Update remaining active fitness values: F(x_i) += exp(-I(worst, x_i) / kappa)
        for idx in active_indices:
            if idx != worst_idx:
                fitness[idx] += np.exp(-I_mat[worst_idx, idx] / kappa)

        active_indices.pop(min_pos)

    return pop[active_indices], F[active_indices], CV[active_indices]


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any, kappa: float = 0.05,
         pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generational step of IBEA."""
    N = len(pop)
    I_mat = compute_indicator_matrix(F, CV)
    fitness = compute_fitness(I_mat, kappa=kappa)

    # Tournament selection
    mating_pool = []
    for _ in range(N):
        i1, i2 = np.random.choice(N, size=2, replace=False)
        winner = i1 if fitness[i1] >= fitness[i2] else i2
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
    off_evals = [problem.evaluate(x) for x in offspring_pop]
    off_F = np.array([e[0] for e in off_evals])
    off_CV = np.array([e[1] for e in off_evals])

    comb_pop = np.vstack([pop, offspring_pop])
    comb_F = np.vstack([F, off_F])
    comb_CV = np.concatenate([CV, off_CV])

    new_pop, new_F, new_CV = environmental_selection(comb_pop, comb_F, comb_CV, target_size=N, kappa=kappa)
    return new_pop, new_F, new_CV


def run(problem: Any, pop_size: int = 100, n_gen: int = 50,
        kappa: float = 0.05, seed: int = 42) -> Dict[str, Any]:
    """Runs IBEA."""
    np.random.seed(seed)
    pop, F, CV = initialize(problem, pop_size=pop_size, seed=seed)

    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, problem, kappa=kappa)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'IBEA',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'kappa': kappa,
        'evaluations': pop_size * (n_gen + 1)
    }

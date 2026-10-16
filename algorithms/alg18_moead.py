"""
alg18_moead.py - Multi-Objective Evolutionary Algorithm based on Decomposition (MOEA/D)
========================================================================================
Family: Elitist Decomposition (Part 3)

Core Mechanism:
Proposed by Zhang & Li (2007).
Decomposes a multi-objective optimization problem into N single-objective
scalar subproblems using a uniform set of weight vectors lambda^1, ..., lambda^N.
Solves all N subproblems simultaneously in a collaborative population:
1. Tchebycheff decomposition:
   g^{te}(x | lambda, z*) = max_m lambda_m * |f_m(x) - z*_m|
2. Neighborhood structure: For each subproblem i, its neighborhood B(i) is formed
   by the T closest weight vectors in Euclidean space.
3. Information sharing: Mating is restricted to neighborhood B(i), and a newly
   generated offspring replaces neighbor solutions if it improves their scalar fitness.

Lecture Steps:
1. Generate uniform weight vectors lambda^1, ..., lambda^N.
2. Compute Euclidean distance matrix between weights and find T nearest neighbors B(i).
3. Initialize population x^1, ..., x^N and evaluate objectives.
4. Initialize reference point z*_m = min_i f_m(x^i).
5. In each generation, for each subproblem i:
   a. Randomly select two parents from neighborhood B(i).
   b. Produce offspring y via SBX crossover and polynomial mutation.
   c. Evaluate y and update reference point z*.
   d. For each neighbor j in B(i): if g^{te}(y | lambda^j, z*) <= g^{te}(x^j | lambda^j, z*), replace x^j with y.
6. Return non-dominated set.

Default Parameters:
- pop_size: 100
- T (neighborhood size): 15
- n_gen: 50
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu
from operators import sbx_crossover, polynomial_mutation


def initialize(problem: Any, pop_size: int = 100, T: int = 15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates weights, neighborhoods, initial population, and reference point."""
    M = problem.n_objs
    rng = np.random.default_rng(seed)

    # 1. Generate uniform weights
    if M == 2:
        w1 = np.linspace(0.001, 0.999, pop_size)
        weights = np.column_stack([w1, 1.0 - w1])
    else:
        weights = rng.dirichlet(np.ones(M), size=pop_size)

    # 2. Neighborhoods B(i)
    dists = np.sqrt(np.sum((weights[:, None, :] - weights[None, :, :]) ** 2, axis=2))
    B = np.argsort(dists, axis=1)[:, :min(T, pop_size)]

    # 3. Initial population
    pop = rng.uniform(problem.xl, problem.xu, size=(pop_size, problem.n_vars))
    F_list, CV_list = [], []
    for ind in pop:
        f, cv = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list)
    CV = np.array(CV_list)

    # 4. Ideal reference point z*
    z_star = np.min(F, axis=0)

    return weights, B, pop, F, CV, z_star


def tchebycheff_scalar(f: np.ndarray, cv: float, weight: np.ndarray, z_star: np.ndarray) -> float:
    """Computes penalized Tchebycheff scalar fitness."""
    diff = np.maximum(0.0, f - z_star)
    return float(np.max(weight * diff) + 1e4 * cv)


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         weights: np.ndarray, B: np.ndarray, z_star: np.ndarray,
         problem: Any, pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One generational sweep of MOEA/D across all N subproblems."""
    N = len(pop)

    for i in range(N):
        # Select parents from neighborhood B[i]
        k, l = np.random.choice(B[i], size=2, replace=False)
        p1 = pop[k]
        p2 = pop[l]

        c1, _ = sbx_crossover(p1, p2, problem.xl, problem.xu, eta_c=eta_c, pc=pc)
        c1 = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=eta_m)

        f_c1, cv_c1 = problem.evaluate(c1)

        # Update ideal reference point
        z_star = np.minimum(z_star, f_c1)

        # Update neighbors in B[i]
        for j in B[i]:
            val_off = tchebycheff_scalar(f_c1, cv_c1, weights[j], z_star)
            val_cur = tchebycheff_scalar(F[j], CV[j], weights[j], z_star)
            if val_off <= val_cur:
                pop[j] = c1.copy()
                F[j] = f_c1.copy()
                CV[j] = cv_c1

    return pop, F, CV, z_star


def run(problem: Any, pop_size: int = 100, T: int = 15, n_gen: int = 50, seed: int = 42) -> Dict[str, Any]:
    """Runs MOEA/D."""
    np.random.seed(seed)
    weights, B, pop, F, CV, z_star = initialize(problem, pop_size=pop_size, T=T, seed=seed)

    for gen in range(n_gen):
        pop, F, CV, z_star = step(pop, F, CV, weights, B, z_star, problem)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MOEA/D',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'weights': weights,
        'evaluations': pop_size * (n_gen + 1)
    }

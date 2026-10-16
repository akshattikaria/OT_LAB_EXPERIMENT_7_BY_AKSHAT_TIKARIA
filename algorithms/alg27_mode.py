"""
alg27_mode.py - Multi-Objective Differential Evolution (MODE)
=============================================================
Family: Evolutionary Metaheuristic (Part 4)

Core Mechanism:
Proposed by Babu & Gujarathi (2007).
Direct extension of Differential Evolution to multi-objective domains:
1. DE/rand/1/bin variation:
   v_i = x_{r1} + F_scale * (x_{r2} - x_{r3})
   Trial vector u_i generated via binomial crossover with target vector x_i.
2. Selection Operator:
   - If trial vector u_i dominates target x_i: u_i replaces x_i.
   - If target x_i dominates trial vector u_i: target x_i is retained.
   - If mutually non-dominated: u_i is appended to an offspring set.
3. Elitist Truncation:
   The combined pool of retained targets and non-dominated trial vectors is
   pruned back to population size N using non-dominated sorting and crowding distance.

Lecture Steps:
1. Initialize population of size N. Evaluate objectives and constraints.
2. In each generation:
   a. For each target individual x_i:
      - Sample 3 distinct members r1, r2, r3 != i.
      - Perform DE mutation and binomial crossover to produce trial vector u_i.
      - Evaluate u_i.
      - Compare u_i and x_i under Pareto dominance.
   b. Combine survivors and non-dominated trial solutions.
   c. Prune back to size N using fast non-dominated sort and crowding distance.
3. Return non-dominated front.

Default Parameters:
- pop_size: 100
- n_gen: 50
- F_scale: 0.5, CR: 0.8
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, pop_size: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pop = rng.uniform(problem.xl, problem.xu, size=(pop_size, problem.n_vars))
    F_list, CV_list = [], []
    for x in pop:
        f, cv = problem.evaluate(x)
        F_list.append(f)
        CV_list.append(cv)
    return pop, np.array(F_list), np.array(CV_list)


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any, F_scale: float = 0.5, CR: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generational step of MODE."""
    N, D = pop.shape
    survivors_X = []
    survivors_F = []
    survivors_CV = []

    for i in range(N):
        candidates = [idx for idx in range(N) if idx != i]
        r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)

        # DE mutation
        v = pop[r1] + F_scale * (pop[r2] - pop[r3])
        # Binomial crossover
        j_rand = np.random.randint(D)
        u = np.where((np.random.rand(D) < CR) | (np.arange(D) == j_rand), v, pop[i])
        u = np.clip(u, problem.xl, problem.xu)

        f_u, cv_u = problem.evaluate(u)

        if mu.deb_constrained_dominates(f_u, cv_u, F[i], CV[i]):
            survivors_X.append(u)
            survivors_F.append(f_u)
            survivors_CV.append(cv_u)
        elif mu.deb_constrained_dominates(F[i], CV[i], f_u, cv_u):
            survivors_X.append(pop[i])
            survivors_F.append(F[i])
            survivors_CV.append(CV[i])
        else:
            # Mutually non-dominated: keep both
            survivors_X.append(pop[i])
            survivors_F.append(F[i])
            survivors_CV.append(CV[i])
            survivors_X.append(u)
            survivors_F.append(f_u)
            survivors_CV.append(cv_u)

    pool_X = np.array(survivors_X)
    pool_F = np.array(survivors_F)
    pool_CV = np.array(survivors_CV)

    # Prune back to N using non-dominated sorting and crowding distance
    fronts = mu.fast_non_dominated_sort(pool_F, pool_CV)
    new_indices = []
    for front in fronts:
        if len(new_indices) + len(front) <= N:
            new_indices.extend(front)
        else:
            needed = N - len(new_indices)
            cd = mu.calculate_crowding_distance(pool_F, front)
            sorted_idx = np.argsort(-cd)
            for idx in range(needed):
                new_indices.append(front[sorted_idx[idx]])
            break

    return pool_X[new_indices], pool_F[new_indices], pool_CV[new_indices]


def run(problem: Any, pop_size: int = 100, n_gen: int = 50,
        F_scale: float = 0.5, CR: float = 0.8, seed: int = 42) -> Dict[str, Any]:
    """Runs MODE."""
    np.random.seed(seed)
    pop, F, CV = initialize(problem, pop_size=pop_size, seed=seed)

    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, problem, F_scale=F_scale, CR=CR)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MODE',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'evaluations': pop_size * (n_gen + 1)
    }

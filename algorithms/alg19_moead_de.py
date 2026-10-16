"""
alg19_moead_de.py - MOEA/D with Differential Evolution (MOEA/D-DE)
==================================================================
Family: Elitist Decomposition (Part 3)

Core Mechanism:
Proposed by Li & Zhang (2009).
Replaces simulated binary crossover in MOEA/D with Differential Evolution (DE)
mutation and binomial crossover operators:
1. DE/rand/1/bin variation:
   Donor vector: v = x^{r1} + F_scale * (x^{r2} - x^{r3})
   Trial vector: u_j = v_j if rand() < CR or j == j_rand else x^i_j
   Followed by polynomial mutation for exploratory perturbation.
2. Probability delta: Mating selects parents either from neighborhood B(i) (with prob delta)
   or from the whole population (with prob 1 - delta).
3. Maximum replacement limit n_r: Each newly generated child can replace at most
   n_r solutions in the neighborhood, preventing single good solutions from taking over.

Lecture Steps:
1. Generate weight vectors and compute neighborhoods B(i).
2. Initialize population, evaluate objectives, and establish ideal reference point z*.
3. In each generation, for each subproblem i:
   a. Sample 3 distinct parents from B(i) (or whole pop with prob 1 - delta).
   b. Generate offspring via DE mutation and binomial crossover.
   c. Apply polynomial mutation and enforce boundary constraints.
   d. Evaluate offspring and update ideal point z*.
   e. Replace up to n_r neighbors if the offspring improves their Tchebycheff fitness.
4. Return non-dominated front.

Default Parameters:
- pop_size: 100
- T: 15
- F_scale: 0.5, CR: 0.9
- n_r: 2 (max replacements)
- delta: 0.9 (neighborhood selection probability)
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu
from operators import polynomial_mutation


def initialize(problem: Any, pop_size: int = 100, T: int = 15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M = problem.n_objs
    rng = np.random.default_rng(seed)

    if M == 2:
        w1 = np.linspace(0.001, 0.999, pop_size)
        weights = np.column_stack([w1, 1.0 - w1])
    else:
        weights = rng.dirichlet(np.ones(M), size=pop_size)

    dists = np.sqrt(np.sum((weights[:, None, :] - weights[None, :, :]) ** 2, axis=2))
    B = np.argsort(dists, axis=1)[:, :min(T, pop_size)]

    pop = rng.uniform(problem.xl, problem.xu, size=(pop_size, problem.n_vars))
    F_list, CV_list = [], []
    for ind in pop:
        f, cv = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list)
    CV = np.array(CV_list)
    z_star = np.min(F, axis=0)
    z_nadir = np.max(F, axis=0)

    return weights, B, pop, F, CV, z_star, z_nadir


def tchebycheff_scalar(f: np.ndarray, cv: float, weight: np.ndarray,
                       z_star: np.ndarray, z_nadir: Optional[np.ndarray] = None) -> float:
    diff = np.maximum(0.0, f - z_star)
    if z_nadir is not None:
        scale = np.maximum(z_nadir - z_star, 1e-6)
        diff = diff / scale
    return float(np.max(weight * diff) + 1e4 * cv)


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         weights: np.ndarray, B: np.ndarray, z_star: np.ndarray,
         problem: Any, z_nadir: Optional[np.ndarray] = None,
         F_scale: float = 0.5, CR: float = 0.9,
         n_r: int = 2, delta: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """One generational sweep of MOEA/D-DE."""
    N = len(pop)
    n_vars = problem.n_vars

    for i in range(N):
        # Select mating pool: B[i] with prob delta, else all population
        pool = B[i] if np.random.rand() < delta else np.arange(N)
        if len(pool) < 3:
            pool = np.arange(N)

        r1, r2, r3 = np.random.choice(pool, size=3, replace=False)

        # DE/rand/1/bin
        v = pop[r1] + F_scale * (pop[r2] - pop[r3])
        j_rand = np.random.randint(n_vars)
        u = np.where((np.random.rand(n_vars) < CR) | (np.arange(n_vars) == j_rand), v, pop[i])

        # Enforce bounds
        u = np.clip(u, problem.xl, problem.xu)
        # Light polynomial mutation
        u = polynomial_mutation(u, problem.xl, problem.xu, eta_m=20.0, pm=1.0 / n_vars)

        f_u, cv_u = problem.evaluate(u)
        z_star = np.minimum(z_star, f_u)
        if z_nadir is not None:
            z_nadir = np.maximum(z_nadir, f_u)

        # Replace at most n_r neighbors
        shuffled_neighbors = np.random.permutation(B[i])
        replacements = 0
        for j in shuffled_neighbors:
            if replacements >= n_r:
                break
            val_off = tchebycheff_scalar(f_u, cv_u, weights[j], z_star, z_nadir)
            val_cur = tchebycheff_scalar(F[j], CV[j], weights[j], z_star, z_nadir)
            if val_off <= val_cur:
                pop[j] = u.copy()
                F[j] = f_u.copy()
                CV[j] = cv_u
                replacements += 1

    return pop, F, CV, z_star, z_nadir


def run(problem: Any, pop_size: int = 100, T: int = 15, n_gen: int = 50,
        F_scale: float = 0.5, CR: float = 0.9, n_r: int = 2, seed: int = 42) -> Dict[str, Any]:
    """Runs MOEA/D-DE."""
    np.random.seed(seed)
    weights, B, pop, F, CV, z_star, z_nadir = initialize(problem, pop_size=pop_size, T=T, seed=seed)

    for gen in range(n_gen):
        pop, F, CV, z_star, z_nadir = step(pop, F, CV, weights, B, z_star, problem,
                                           z_nadir=z_nadir, F_scale=F_scale, CR=CR, n_r=n_r)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MOEA/D-DE',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'weights': weights,
        'evaluations': pop_size * (n_gen + 1)
    }

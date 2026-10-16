"""
alg23_mogwo.py - Multi-Objective Grey Wolf Optimizer (MOGWO)
============================================================
Family: Swarm Metaheuristic (Part 4)

Core Mechanism:
Proposed by Mirjalili, Sariwi, Mirjalili, & Coelho (2016).
Emulates the social leadership hierarchy and hunting mechanism of grey wolves:
1. Leaders alpha, beta, and delta are selected from an external non-dominated archive
   using roulette-wheel selection based on hypercube crowding (prioritizing sparse regions).
2. Encircling & hunting formulation:
   a = 2 - 2 * (iter / max_iter)  (linearly decreases from 2 to 0)
   A_k = 2 * a * r1 - a,  C_k = 2 * r2
   D_alpha = |C_1 * X_alpha - X|,  X_1 = X_alpha - A_1 * D_alpha
   D_beta  = |C_2 * X_beta - X|,   X_2 = X_beta  - A_2 * D_beta
   D_delta = |C_3 * X_delta - X|,  X_3 = X_delta - A_3 * D_delta
   X^{t+1} = (X_1 + X_2 + X_3) / 3
3. Archive maintenance: non-dominated solutions are stored in a fixed-size archive;
   excess members are pruned using crowding distance.

Lecture Steps:
1. Initialize grey wolf pack positions X. Evaluate objectives and constraints.
2. Initialize external archive with non-dominated solutions.
3. In each iteration:
   a. Update coefficient vector a linearly from 2 to 0.
   b. Select alpha, beta, and delta wolves from the archive.
   c. Update positions of all omega wolves using encircling equations.
   d. Bound checking and evaluation.
   e. Update external archive and prune if capacity is exceeded.
4. Return archive non-dominated front.

Default Parameters:
- n_wolves: 100
- max_iter: 50
- archive_size: 100
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, n_wolves: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(problem.xl, problem.xu, size=(n_wolves, problem.n_vars))
    F_list, CV_list = [], []
    for x in X:
        f, cv = problem.evaluate(x)
        F_list.append(f)
        CV_list.append(cv)
    return X, np.array(F_list), np.array(CV_list)


def select_leader(archive_X: np.ndarray, archive_F: np.ndarray) -> np.ndarray:
    """Selects a leader wolf from the archive using roulette wheel on crowding distance."""
    N_arc = len(archive_X)
    if N_arc <= 2:
        return archive_X[np.random.randint(N_arc)].copy()

    cd = mu.calculate_crowding_distance(archive_F, list(range(N_arc)))
    finite_cd = cd[np.isfinite(cd)]
    max_cd = np.max(finite_cd) if len(finite_cd) > 0 else 1.0
    clean_cd = np.where(np.isinf(cd), max_cd * 2.0, cd)
    clean_cd[clean_cd < 1e-6] = 1e-6

    probs = clean_cd / np.sum(clean_cd)
    idx = np.random.choice(N_arc, p=probs)
    return archive_X[idx].copy()


def update_archive(X: np.ndarray, F: np.ndarray, CV: np.ndarray,
                   arc_X: np.ndarray, arc_F: np.ndarray, arc_CV: np.ndarray,
                   max_size: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(arc_X) == 0:
        comb_X, comb_F, comb_CV = X, F, CV
    else:
        comb_X = np.vstack([arc_X, X])
        comb_F = np.vstack([arc_F, F])
        comb_CV = np.concatenate([arc_CV, CV])

    fronts = mu.fast_non_dominated_sort(comb_F, comb_CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    nd_X = comb_X[nd_indices]
    nd_F = comb_F[nd_indices]
    nd_CV = comb_CV[nd_indices]

    if len(nd_X) > max_size:
        cd = mu.calculate_crowding_distance(nd_F, list(range(len(nd_X))))
        sorted_idx = np.argsort(-cd)
        chosen = sorted_idx[:max_size]
        return nd_X[chosen], nd_F[chosen], nd_CV[chosen]

    return nd_X, nd_F, nd_CV


def step(X: np.ndarray, F: np.ndarray, CV: np.ndarray,
         arc_X: np.ndarray, arc_F: np.ndarray, arc_CV: np.ndarray,
         problem: Any, current_iter: int, max_iter: int,
         archive_size: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One iteration of MOGWO."""
    N = len(X)
    n_vars = problem.n_vars
    a = 2.0 - 2.0 * (current_iter / max_iter)

    # Select leaders alpha, beta, delta
    alpha_wolf = select_leader(arc_X, arc_F)
    beta_wolf = select_leader(arc_X, arc_F)
    delta_wolf = select_leader(arc_X, arc_F)

    new_X = np.empty_like(X)

    for i in range(N):
        r1, r2 = np.random.rand(n_vars), np.random.rand(n_vars)
        A1 = 2.0 * a * r1 - a
        C1 = 2.0 * r2
        D_alpha = np.abs(C1 * alpha_wolf - X[i])
        X1 = alpha_wolf - A1 * D_alpha

        r1, r2 = np.random.rand(n_vars), np.random.rand(n_vars)
        A2 = 2.0 * a * r1 - a
        C2 = 2.0 * r2
        D_beta = np.abs(C2 * beta_wolf - X[i])
        X2 = beta_wolf - A2 * D_beta

        r1, r2 = np.random.rand(n_vars), np.random.rand(n_vars)
        A3 = 2.0 * a * r1 - a
        C3 = 2.0 * r2
        D_delta = np.abs(C3 * delta_wolf - X[i])
        X3 = delta_wolf - A3 * D_delta

        new_pos = (X1 + X2 + X3) / 3.0
        new_X[i] = np.clip(new_pos, problem.xl, problem.xu)

    F_list, CV_list = [], []
    for x in new_X:
        f, cv = problem.evaluate(x)
        F_list.append(f)
        CV_list.append(cv)

    new_F = np.array(F_list)
    new_CV = np.array(CV_list)

    arc_X, arc_F, arc_CV = update_archive(new_X, new_F, new_CV, arc_X, arc_F, arc_CV, max_size=archive_size)

    return new_X, new_F, new_CV, arc_X, arc_F, arc_CV


def run(problem: Any, n_wolves: int = 100, archive_size: int = 100,
        max_iter: int = 50, seed: int = 42) -> Dict[str, Any]:
    """Runs MOGWO."""
    np.random.seed(seed)
    X, F, CV = initialize(problem, n_wolves=n_wolves, seed=seed)
    arc_X, arc_F, arc_CV = np.empty((0, problem.n_vars)), np.empty((0, problem.n_objs)), np.empty(0)
    arc_X, arc_F, arc_CV = update_archive(X, F, CV, arc_X, arc_F, arc_CV, max_size=archive_size)

    for it in range(max_iter):
        X, F, CV, arc_X, arc_F, arc_CV = step(
            X, F, CV, arc_X, arc_F, arc_CV, problem, it, max_iter, archive_size=archive_size
        )

    fronts = mu.fast_non_dominated_sort(arc_F, arc_CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MOGWO',
        'pop': arc_X,
        'F': arc_F,
        'CV': arc_CV,
        'X_front': arc_X[nd_indices],
        'F_front': arc_F[nd_indices],
        'evaluations': n_wolves * (max_iter + 1)
    }

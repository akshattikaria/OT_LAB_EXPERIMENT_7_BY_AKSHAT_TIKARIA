"""
alg22_mopso.py - Multi-Objective Particle Swarm Optimization (MOPSO)
=====================================================================
Family: Swarm Metaheuristic (Part 4)

Core Mechanism:
Proposed by Coello Coello, Pulido, & Lechuga (2004).
Extends single-objective PSO to multi-objective domains using an external archive:
1. Velocity & Position Update:
   v_i^{t+1} = w * v_i^t + c_1 * r_1 * (pbest_i - x_i^t) + c_2 * r_2 * (gbest - x_i^t)
   x_i^{t+1} = x_i^t + v_i^{t+1}
2. Personal Best (pbest):
   If current position x_i dominates pbest_i, replace it.
   If mutually non-dominated, randomly select between them.
3. Global Best (gbest) Selection:
   Selected from the external repository of non-dominated solutions using
   roulette-wheel on inverse hypercube density (less crowded archive members
   are prioritized to guide the swarm towards sparse Pareto regions).
4. Turbulence (mutation) operator applied to prevent premature stagnation.

Lecture Steps:
1. Initialize swarm positions and zero velocities. Evaluate objectives and constraints.
2. Initialize pbest with initial positions.
3. Store non-dominated solutions into external archive (repository).
4. Discretize archive objective space into hypercubes.
5. In each iteration:
   a. Select gbest for each particle from repository via hypercube roulette wheel.
   b. Update velocity, apply velocity clamping, and update position.
   c. Apply mutation to particle positions.
   d. Evaluate particles, update pbest, and update repository.
6. Return non-dominated archive.

Default Parameters:
- n_particles: 100
- max_iter: 50
- archive_size: 100
- w: 0.5 (inertia), c1: 1.5, c2: 1.5
- pm: 0.1 (mutation probability)
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu
from operators import polynomial_mutation


def initialize(problem: Any, n_particles: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(problem.xl, problem.xu, size=(n_particles, problem.n_vars))
    V = np.zeros_like(X)

    F_list, CV_list = [], []
    for x in X:
        f, cv = problem.evaluate(x)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list)
    CV = np.array(CV_list)
    P_best = X.copy()
    P_best_F = F.copy()
    P_best_CV = CV.copy()

    return X, V, P_best, P_best_F, P_best_CV


def update_repository(X: np.ndarray, F: np.ndarray, CV: np.ndarray,
                      rep_X: np.ndarray, rep_F: np.ndarray, rep_CV: np.ndarray,
                      max_size: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Updates external non-dominated repository."""
    if len(rep_X) == 0:
        comb_X = X
        comb_F = F
        comb_CV = CV
    else:
        comb_X = np.vstack([rep_X, X])
        comb_F = np.vstack([rep_F, F])
        comb_CV = np.concatenate([rep_CV, CV])

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


def select_gbest(rep_X: np.ndarray, rep_F: np.ndarray) -> np.ndarray:
    """Selects gbest from repository with bias toward less crowded points."""
    N_rep = len(rep_X)
    if N_rep == 1:
        return rep_X[0].copy()

    cd = mu.calculate_crowding_distance(rep_F, list(range(N_rep)))
    # Replace inf with 2x max finite value
    finite_cd = cd[np.isfinite(cd)]
    max_cd = np.max(finite_cd) if len(finite_cd) > 0 else 1.0
    clean_cd = np.where(np.isinf(cd), max_cd * 2.0, cd)
    clean_cd[clean_cd < 1e-6] = 1e-6

    probs = clean_cd / np.sum(clean_cd)
    idx = np.random.choice(N_rep, p=probs)
    return rep_X[idx].copy()


def step(X: np.ndarray, V: np.ndarray, P_best: np.ndarray,
         P_best_F: np.ndarray, P_best_CV: np.ndarray,
         rep_X: np.ndarray, rep_F: np.ndarray, rep_CV: np.ndarray,
         problem: Any, w: float = 0.5, c1: float = 1.5, c2: float = 1.5,
         pm: float = 0.1, archive_size: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One iteration of MOPSO."""
    N = len(X)
    v_max = 0.2 * (problem.xu - problem.xl)

    F_list, CV_list = [], []
    for i in range(N):
        gbest = select_gbest(rep_X, rep_F)
        r1 = np.random.rand(problem.n_vars)
        r2 = np.random.rand(problem.n_vars)

        # Velocity update
        V[i] = w * V[i] + c1 * r1 * (P_best[i] - X[i]) + c2 * r2 * (gbest - X[i])
        V[i] = np.clip(V[i], -v_max, v_max)

        # Position update
        X[i] = X[i] + V[i]
        X[i] = np.clip(X[i], problem.xl, problem.xu)

        # Turbulence / mutation
        if np.random.rand() < pm:
            X[i] = polynomial_mutation(X[i], problem.xl, problem.xu, eta_m=20.0, pm=1.0/problem.n_vars)

        f_new, cv_new = problem.evaluate(X[i])
        F_list.append(f_new)
        CV_list.append(cv_new)

        # Update pbest
        if mu.deb_constrained_dominates(f_new, cv_new, P_best_F[i], P_best_CV[i]):
            P_best[i] = X[i].copy()
            P_best_F[i] = f_new.copy()
            P_best_CV[i] = cv_new
        elif not mu.deb_constrained_dominates(P_best_F[i], P_best_CV[i], f_new, cv_new):
            if np.random.rand() < 0.5:
                P_best[i] = X[i].copy()
                P_best_F[i] = f_new.copy()
                P_best_CV[i] = cv_new

    # Update repository using current iteration evaluations
    rep_X, rep_F, rep_CV = update_repository(X, np.array(F_list), np.array(CV_list),
                                             rep_X, rep_F, rep_CV, max_size=archive_size)

    return X, V, P_best, P_best_F, P_best_CV, rep_X, rep_F, rep_CV


def run(problem: Any, n_particles: int = 100, archive_size: int = 100,
        max_iter: int = 50, seed: int = 42) -> Dict[str, Any]:
    """Runs MOPSO."""
    np.random.seed(seed)
    X, V, P_best, P_best_F, P_best_CV = initialize(problem, n_particles=n_particles, seed=seed)
    rep_X, rep_F, rep_CV = np.empty((0, problem.n_vars)), np.empty((0, problem.n_objs)), np.empty(0)
    rep_X, rep_F, rep_CV = update_repository(X, P_best_F, P_best_CV, rep_X, rep_F, rep_CV, max_size=archive_size)

    for it in range(max_iter):
        X, V, P_best, P_best_F, P_best_CV, rep_X, rep_F, rep_CV = step(
            X, V, P_best, P_best_F, P_best_CV, rep_X, rep_F, rep_CV, problem,
            archive_size=archive_size
        )

    fronts = mu.fast_non_dominated_sort(rep_F, rep_CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MOPSO',
        'pop': rep_X,
        'F': rep_F,
        'CV': rep_CV,
        'X_front': rep_X[nd_indices],
        'F_front': rep_F[nd_indices],
        'evaluations': n_particles * (max_iter + 1)
    }

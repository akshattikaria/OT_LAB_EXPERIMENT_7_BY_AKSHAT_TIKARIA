"""
alg24_moabc.py - Multi-Objective Artificial Bee Colony (MOABC)
==============================================================
Family: Swarm Metaheuristic (Part 4)

Core Mechanism:
Extends Artificial Bee Colony (ABC) to multi-objective problems:
1. Employed Bee Phase:
   Generates a candidate food source near current source:
       v_{ij} = x_{ij} + phi_{ij} * (x_{ij} - x_{kj}),  phi in [-1, 1]
   If v_i dominates x_i (using Deb's rules), x_i is updated and trial counter reset;
   otherwise, trial counter is incremented.
2. Onlooker Bee Phase:
   Selects food sources with probability proportional to non-dominated rank and
   crowding distance. Onlookers perform neighborhood search around selected sources.
3. Scout Bee Phase:
   If a food source trial counter exceeds 'limit', it is abandoned and the scout bee
   discovers a new random food source.
4. An external archive preserves non-dominated solutions discovered by the colony.

Lecture Steps:
1. Initialize food sources x_1, ..., x_N, evaluate objectives, and initialize trial counters to 0.
2. Store initial non-dominated solutions into external archive.
3. In each cycle:
   a. Employed bees generate candidate solutions and update food sources via dominance.
   b. Calculate selection probabilities of food sources based on Pareto rank.
   c. Onlooker bees select sources, generate candidate moves, and update sources.
   d. Scout bees identify abandoned sources (trial >= limit) and re-initialize them.
   e. Update external archive and prune if full.
4. Return archive non-dominated front.

Default Parameters:
- n_food_sources: 50
- max_cycles: 50
- limit: 20
- archive_size: 100
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, n_sources: int = 50, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(problem.xl, problem.xu, size=(n_sources, problem.n_vars))
    F_list, CV_list = [], []
    for x in X:
        f, cv = problem.evaluate(x)
        F_list.append(f)
        CV_list.append(cv)
    trial_counts = np.zeros(n_sources, dtype=int)
    return X, np.array(F_list), np.array(CV_list), trial_counts


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


def step(X: np.ndarray, F: np.ndarray, CV: np.ndarray, trial_counts: np.ndarray,
         arc_X: np.ndarray, arc_F: np.ndarray, arc_CV: np.ndarray,
         problem: Any, limit: int = 20, archive_size: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One cycle of MOABC."""
    N, D = X.shape

    # 1. Employed Bee Phase
    for i in range(N):
        k = np.random.choice([idx for idx in range(N) if idx != i])
        j = np.random.randint(D)
        phi = np.random.uniform(-1.0, 1.0)

        v = X[i].copy()
        v[j] = X[i, j] + phi * (X[i, j] - X[k, j])
        v = np.clip(v, problem.xl, problem.xu)

        f_v, cv_v = problem.evaluate(v)

        if mu.deb_constrained_dominates(f_v, cv_v, F[i], CV[i]):
            X[i] = v.copy()
            F[i] = f_v.copy()
            CV[i] = cv_v
            trial_counts[i] = 0
        else:
            trial_counts[i] += 1

    # 2. Onlooker Bee Phase
    fronts = mu.fast_non_dominated_sort(F, CV)
    ranks = np.zeros(N)
    for r_idx, front in enumerate(fronts):
        for idx in front:
            ranks[idx] = r_idx

    fitness = 1.0 / (1.0 + ranks)
    probs = fitness / np.sum(fitness)

    for _ in range(N):
        i = np.random.choice(N, p=probs)
        k = np.random.choice([idx for idx in range(N) if idx != i])
        j = np.random.randint(D)
        phi = np.random.uniform(-1.0, 1.0)

        v = X[i].copy()
        v[j] = X[i, j] + phi * (X[i, j] - X[k, j])
        v = np.clip(v, problem.xl, problem.xu)

        f_v, cv_v = problem.evaluate(v)

        if mu.deb_constrained_dominates(f_v, cv_v, F[i], CV[i]):
            X[i] = v.copy()
            F[i] = f_v.copy()
            CV[i] = cv_v
            trial_counts[i] = 0
        else:
            trial_counts[i] += 1

    # 3. Scout Bee Phase
    for i in range(N):
        if trial_counts[i] >= limit:
            X[i] = np.random.uniform(problem.xl, problem.xu)
            f_new, cv_new = problem.evaluate(X[i])
            F[i] = f_new
            CV[i] = cv_new
            trial_counts[i] = 0

    # 4. Update external archive
    arc_X, arc_F, arc_CV = update_archive(X, F, CV, arc_X, arc_F, arc_CV, max_size=archive_size)

    return X, F, CV, trial_counts, arc_X, arc_F, arc_CV


def run(problem: Any, n_sources: int = 50, max_cycles: int = 50,
        limit: int = 20, archive_size: int = 100, seed: int = 42) -> Dict[str, Any]:
    """Runs MOABC."""
    np.random.seed(seed)
    X, F, CV, trial_counts = initialize(problem, n_sources=n_sources, seed=seed)
    arc_X, arc_F, arc_CV = np.empty((0, problem.n_vars)), np.empty((0, problem.n_objs)), np.empty(0)
    arc_X, arc_F, arc_CV = update_archive(X, F, CV, arc_X, arc_F, arc_CV, max_size=archive_size)

    for cyc in range(max_cycles):
        X, F, CV, trial_counts, arc_X, arc_F, arc_CV = step(
            X, F, CV, trial_counts, arc_X, arc_F, arc_CV, problem,
            limit=limit, archive_size=archive_size
        )

    fronts = mu.fast_non_dominated_sort(arc_F, arc_CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MOABC',
        'pop': arc_X,
        'F': arc_F,
        'CV': arc_CV,
        'X_front': arc_X[nd_indices],
        'F_front': arc_F[nd_indices],
        'evaluations': n_sources * (2 * max_cycles + 1)
    }

"""
alg29_moba.py - Multi-Objective Bat Algorithm (MOBA)
====================================================
Family: Swarm Metaheuristic (Part 4)
Assigned Deep-Dive Algorithm for R = 05 (Task C).

Core Mechanism:
Proposed by Xin-She Yang (2011).
Models the echolocation behavior of microbats for multi-objective search:
1. Dynamic Frequency-driven Velocity & Position:
   f_i = f_min + (f_max - f_min) * beta,  where beta in U(0, 1)
   v_i^{t+1} = v_i^t + (x_i^t - x_*) * f_i
   x_i^{t+1} = x_i^t + v_i^{t+1}
   where x_* is a guide selected from the external Pareto archive.
2. Local Search with Echolocation Loudness:
   If rand() > r_i (pulse emission rate):
       x_new = x_* + epsilon * A_mean
   where epsilon in [-1, 1] and A_mean is the average loudness across all bats.
3. Pulse Rate & Loudness Schedules:
   When an improved or non-dominated solution is discovered:
       A_i^{t+1} = alpha * A_i^t           (loudness decays towards zero)
       r_i^{t+1} = r_i^0 * (1 - exp(-gamma * t))  (pulse emission rate increases)
4. Non-dominated Archive:
   An external repository preserves the Pareto-optimal front, pruned via crowding distance.

Lecture Steps:
1. Initialize bat swarm positions x_i and velocities v_i = 0.
2. Initialize frequency bounds [f_min, f_max], loudness A_i = A_0, and pulse rate r_i = r_0.
3. Evaluate initial population and populate external archive with non-dominated bats.
4. For iteration t = 1 to max_iter:
   a. Compute mean loudness A_mean.
   b. For each bat i:
      - Select guide x_* from the archive using crowding-based tournament.
      - Adjust frequency f_i and update velocity and position.
      - If rand() > r_i: perform local random walk around x_* scaled by A_mean.
      - Enforce variable boundary constraints and evaluate candidate.
      - If candidate dominates current bat or is non-dominated:
        accept position, reduce loudness A_i = alpha * A_i, increase pulse rate r_i.
      - Update external archive with new candidate.
5. Return external non-dominated front.

Default Parameters:
- n_bats: 100
- max_iter: 50
- archive_size: 100
- f_min: 0.0, f_max: 2.0
- A0: 0.9, alpha: 0.95
- r0: 0.1, gamma: 0.05
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, n_bats: int = 100,
               f_min: float = 0.0, f_max: float = 2.0,
               A0: float = 0.9, r0: float = 0.1,
               seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initializes bat swarm states and schedules."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(problem.xl, problem.xu, size=(n_bats, problem.n_vars))
    V = np.zeros_like(X)

    F_list, CV_list = [], []
    for x in X:
        f, cv = problem.evaluate(x)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list)
    CV = np.array(CV_list)

    A = np.full(n_bats, A0)
    r = np.full(n_bats, r0)
    freq = np.zeros(n_bats)

    return X, V, F, CV, A, r, freq


def update_archive(candidate_x: np.ndarray, candidate_f: np.ndarray, candidate_cv: float,
                   arc_X: np.ndarray, arc_F: np.ndarray, arc_CV: np.ndarray,
                   max_size: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Updates external non-dominated archive."""
    if len(arc_X) == 0:
        return np.array([candidate_x]), np.array([candidate_f]), np.array([candidate_cv])

    comb_X = np.vstack([arc_X, candidate_x])
    comb_F = np.vstack([arc_F, candidate_f])
    comb_CV = np.append(arc_CV, candidate_cv)

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


def select_archive_guide(arc_X: np.ndarray, arc_F: np.ndarray, rng: Any = None) -> np.ndarray:
    """Selects guide bat x_* from archive prioritizing less crowded regions."""
    if rng is None:
        rng = np.random
    _choice = lambda a, **kw: rng.choice(a, **kw)
    _rand = lambda: float(rng.rand()) if hasattr(rng, 'rand') else float(rng.random())

    N_arc = len(arc_X)
    if N_arc <= 2:
        idx = 0 if N_arc == 1 else (0 if _rand() < 0.5 else 1)
        return arc_X[idx].copy()

    cd = mu.calculate_crowding_distance(arc_F, list(range(N_arc)))
    finite_cd = cd[np.isfinite(cd)]
    max_cd = np.max(finite_cd) if len(finite_cd) > 0 else 1.0
    clean_cd = np.where(np.isinf(cd), max_cd * 2.0, cd)
    clean_cd[clean_cd < 1e-6] = 1e-6

    probs = clean_cd / np.sum(clean_cd)
    idx = _choice(N_arc, p=probs)
    return arc_X[idx].copy()


def step(X: np.ndarray, V: np.ndarray, F: np.ndarray, CV: np.ndarray,
         A: np.ndarray, r: np.ndarray, freq: np.ndarray,
         arc_X: np.ndarray, arc_F: np.ndarray, arc_CV: np.ndarray,
         problem: Any, current_iter: int,
         f_min: float = 0.0, f_max: float = 1.0,
         alpha: float = 0.95, gamma: float = 0.05, r0: float = 0.1,
         archive_size: int = 100,
         rng: Any = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Executes one iteration of MOBA.

    Bug fixes (S3e):
    (i)   current_iter is passed from outside (caller uses it+1, not it).
    (ii)  Archive updated once per iteration (after all bats), not per evaluation.
    (iii) Local walk: per-dimension Gaussian with sigma=A_mean*0.01*(xu-xl),
          applied to a random subset of dimensions.
    (iv)  Velocity magnitude clamped to 0.2*(xu-xl); out-of-bound coordinates
          reflected rather than clipped.
    """
    if rng is None:
        rng = np.random
    _rand = lambda: float(rng.rand()) if hasattr(rng, 'rand') else float(rng.random())
    _choice = lambda a, **kw: rng.choice(a, **kw)
    _normal = lambda size: rng.normal(size=size) if hasattr(rng, 'normal') else np.random.randn(size)

    N, D = X.shape
    A_mean = np.mean(A)
    v_range = 0.2 * (problem.xu - problem.xl)  # (iv) velocity limit

    # Batch of candidates to update archive with at end of iteration
    cand_X_list = []
    cand_F_list = []
    cand_CV_list = []

    for i in range(N):
        x_guide = select_archive_guide(arc_X, arc_F, rng=rng)

        # 1. Frequency and Velocity update
        beta = _rand()
        freq[i] = f_min + (f_max - f_min) * beta
        V[i] = 0.5 * V[i] + (x_guide - X[i]) * freq[i]

        # (iv) Clamp velocity magnitude per dimension
        V[i] = np.clip(V[i], -v_range, v_range)

        x_cand = X[i] + V[i]

        # (iv) Reflect out-of-bound coordinates
        for d in range(D):
            while x_cand[d] < problem.xl[d] or x_cand[d] > problem.xu[d]:
                if x_cand[d] < problem.xl[d]:
                    x_cand[d] = 2 * problem.xl[d] - x_cand[d]
                if x_cand[d] > problem.xu[d]:
                    x_cand[d] = 2 * problem.xu[d] - x_cand[d]
            # fallback if oscillation persists
            x_cand[d] = np.clip(x_cand[d], problem.xl[d], problem.xu[d])

        # 2. Local random walk if rand > r_i
        rand_val = _rand()
        if rand_val > r[i]:
            # (iii) Per-dimension Gaussian on random subset of dimensions
            sigma = A_mean * 0.01 * (problem.xu - problem.xl)
            n_dims = max(1, D // 2)
            dims = _choice(D, size=n_dims, replace=False)
            noise = np.zeros(D)
            norm_draws = _normal(size=n_dims)
            noise[dims] = norm_draws * sigma[dims]
            x_cand = np.clip(x_cand + noise, problem.xl, problem.xu)

        f_cand, cv_cand = problem.evaluate(x_cand)

        # 3. Acceptance and schedules update
        rand_acc = _rand()
        if mu.deb_constrained_dominates(f_cand, cv_cand, F[i], CV[i]) or \
           (rand_acc < A[i] and not mu.deb_constrained_dominates(F[i], CV[i], f_cand, cv_cand)):
            X[i] = x_cand.copy()
            F[i] = f_cand.copy()
            CV[i] = cv_cand
            A[i] *= alpha
            r[i] = r0 * (1.0 - np.exp(-gamma * current_iter))

        # (ii) collect candidates for batch archive update
        cand_X_list.append(x_cand)
        cand_F_list.append(f_cand)
        cand_CV_list.append(cv_cand)

    # (ii) Update archive once per iteration with all candidates combined in one batch
    comb_X = np.vstack([arc_X, cand_X_list])
    comb_F = np.vstack([arc_F, cand_F_list])
    comb_CV = np.concatenate([arc_CV, cand_CV_list])

    fronts = mu.fast_non_dominated_sort(comb_F, comb_CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    nd_X = comb_X[nd_indices]
    nd_F = comb_F[nd_indices]
    nd_CV = comb_CV[nd_indices]

    if len(nd_X) > archive_size:
        cd = mu.calculate_crowding_distance(nd_F, list(range(len(nd_X))))
        sorted_idx = np.argsort(-cd)
        chosen = sorted_idx[:archive_size]
        arc_X, arc_F, arc_CV = nd_X[chosen], nd_F[chosen], nd_CV[chosen]
    else:
        arc_X, arc_F, arc_CV = nd_X, nd_F, nd_CV

    return X, V, F, CV, A, r, freq, arc_X, arc_F, arc_CV


def run(problem: Any, n_bats: int = 100, max_iter: int = 50,
        f_min: float = 0.0, f_max: float = 1.0,
        A0: float = 0.9, alpha: float = 0.95,
        r0: float = 0.1, gamma: float = 0.05,
        archive_size: int = 100, seed: int = 42) -> Dict[str, Any]:
    """Runs MOBA.

    Bug fixes applied (S3e): current_iter=it+1 passed to step(); archive updated
    once per iteration; Gaussian local walk on random dimension subset; velocity
    clamped to 0.2*range with reflection at boundaries.
    """
    np.random.seed(seed)
    X, V, F, CV, A, r, freq = initialize(problem, n_bats=n_bats, f_min=f_min, f_max=f_max, A0=A0, r0=r0, seed=seed)

    # Initialize archive with non-dominated bats
    fronts = mu.fast_non_dominated_sort(F, CV)
    f1 = fronts[0] if len(fronts) > 0 else list(range(n_bats))
    arc_X, arc_F, arc_CV = X[f1].copy(), F[f1].copy(), CV[f1].copy()

    for it in range(max_iter):
        # (i) Pass current_iter = it+1 (1-indexed iteration number)
        X, V, F, CV, A, r, freq, arc_X, arc_F, arc_CV = step(
            X, V, F, CV, A, r, freq, arc_X, arc_F, arc_CV,
            problem, current_iter=it + 1,
            f_min=f_min, f_max=f_max, alpha=alpha,
            gamma=gamma, r0=r0, archive_size=archive_size
        )

    fronts = mu.fast_non_dominated_sort(arc_F, arc_CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MOBA',
        'pop': arc_X,
        'F': arc_F,
        'CV': arc_CV,
        'X_front': arc_X[nd_indices],
        'F_front': arc_F[nd_indices],
        'A_final': A,
        'r_final': r,
        'evaluations': n_bats * (max_iter + 1)
    }

"""
alg31_rvea.py - Reference Vector Guided Evolutionary Algorithm (RVEA)
======================================================================
Family: Many-Objective Evolutionary (Module 5E)

Core Mechanism:
Proposed by Cheng, Jin, Olhofer, & Sendhoff (IEEE TEVC, 2016).
Designed for many-objective optimization (M >= 3):
1. Unit Reference Vectors V: Predefined uniform directions on the unit simplex.
2. Objective Translation: Translates objective space relative to adaptive ideal point:
   f'_m = f_m - z*_m.
3. Subspace Partitioning: Associates each solution x with reference vector v_k
   having the smallest acute angle theta_{x, v_k} in objective space.
4. Angle-Penalized Distance (APD):
   APD(x, v_k) = (1 + P(theta)) * ||f'||
   where penalty P(theta) = M * (t / t_max)^alpha * (theta / gamma_k),
   and gamma_k is the minimum angle between v_k and any other reference vector.
   - Early generations: low penalty -> emphasizes convergence (||f'||).
   - Late generations: high penalty -> strongly enforces spread along reference directions.
5. Elitist Selection: Selects the single individual with minimum APD in each subspace.

Lecture Steps:
1. Generate unit reference vectors on simplex and compute neighbor angles gamma_k.
2. Initialize population P_0 of size N = |V|.
3. In each generation t:
   a. Create offspring Q_t via SBX and polynomial mutation.
   b. Combine R_t = P_t U Q_t and update ideal point z*.
   c. Translate objectives: f' = f - z*.
   d. Partition candidates into reference vector subspaces.
   e. For each non-empty subspace, select candidate with minimal APD.
4. Return non-dominated front.

Default Parameters:
- p_divisions: 12 (for M=3) or 4 (for M=5)
- alpha: 2.0 (penalty parameter)
- n_gen: 50
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu
from operators import sbx_crossover, polynomial_mutation


def generate_unit_reference_vectors(M: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generates unit reference vectors and calculates minimum neighbor angles gamma."""
    from alg15_nsga3 import generate_reference_points
    ref_pts = generate_reference_points(M, p)
    norms = np.linalg.norm(ref_pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V = ref_pts / norms

    # Calculate gamma_k: minimum angle to other reference vectors
    num_v = len(V)
    gamma = np.zeros(num_v)
    cos_matrix = np.clip(V @ V.T, -1.0, 1.0)
    angle_matrix = np.arccos(cos_matrix)
    np.fill_diagonal(angle_matrix, np.inf)

    for k in range(num_v):
        gamma[k] = np.min(angle_matrix[k])
    gamma[gamma == 0] = 1e-6
    return V, gamma


def associate_and_compute_apd(F_trans: np.ndarray, V: np.ndarray, gamma: np.ndarray,
                              t: int, t_max: int, alpha: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """Partitions solutions into reference vector subspaces and computes APD."""
    N, M = F_trans.shape
    num_v = len(V)

    f_norms = np.linalg.norm(F_trans, axis=1, keepdims=True)
    f_norms[f_norms == 0] = 1e-12
    F_unit = F_trans / f_norms

    # Cosine angles: (N, num_v)
    cos_angles = np.clip(F_unit @ V.T, -1.0, 1.0)
    angles = np.arccos(cos_angles)

    subspace_assignments = np.argmin(angles, axis=1)

    apd_values = np.zeros(N)
    time_factor = (t / max(1, t_max)) ** alpha

    for i in range(N):
        k = subspace_assignments[i]
        theta = angles[i, k]
        p_theta = M * time_factor * (theta / gamma[k])
        apd_values[i] = (1.0 + p_theta) * f_norms[i, 0]

    return subspace_assignments, apd_values


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         V: np.ndarray, gamma: np.ndarray, z_star: np.ndarray,
         problem: Any, t: int, t_max: int, alpha: float = 2.0,
         pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generational step of RVEA."""
    N = len(pop)

    # Generate N offspring
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

    # Update ideal point
    z_star = np.minimum(z_star, np.min(R_F, axis=0))

    # Translate objectives: f' = f - z*
    F_trans = np.maximum(0.0, R_F - z_star)

    subspaces, apd = associate_and_compute_apd(F_trans, V, gamma, t, t_max, alpha=alpha)

    # Select best individual with minimum APD in each subspace
    selected_indices = []
    num_v = len(V)
    for k in range(num_v):
        assigned = np.where(subspaces == k)[0]
        if len(assigned) > 0:
            best_in_sub = assigned[np.argmin(apd[assigned])]
            selected_indices.append(best_in_sub)

    # If selected < N, pad with lowest APD from remaining
    if len(selected_indices) < N:
        remaining = [idx for idx in range(len(R_pop)) if idx not in selected_indices]
        needed = N - len(selected_indices)
        sorted_rem = sorted(remaining, key=lambda idx: apd[idx])
        selected_indices.extend(sorted_rem[:needed])

    new_pop = R_pop[selected_indices[:N]]
    new_F = R_F[selected_indices[:N]]
    new_CV = R_CV[selected_indices[:N]]
    return new_pop, new_F, new_CV, z_star


def run(problem: Any, p_divisions: int = 12, n_gen: int = 50,
        alpha: float = 2.0, seed: int = 42) -> Dict[str, Any]:
    """Runs RVEA."""
    np.random.seed(seed)
    M = problem.n_objs
    p = 12 if M == 3 else (4 if M == 5 else p_divisions)

    V, gamma = generate_unit_reference_vectors(M, p)
    pop_size = len(V)
    if pop_size % 2 != 0:
        pop_size += 1

    rng = np.random.default_rng(seed)
    pop = rng.uniform(problem.xl, problem.xu, size=(pop_size, problem.n_vars))
    pop_evals = [problem.evaluate(x) for x in pop]
    F = np.array([e[0] for e in pop_evals])
    CV = np.array([e[1] for e in pop_evals])
    z_star = np.min(F, axis=0)

    for t in range(n_gen):
        pop, F, CV, z_star = step(pop, F, CV, V, gamma, z_star, problem, t, n_gen, alpha=alpha)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'RVEA',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'V': V,
        'evaluations': pop_size * (n_gen + 1)
    }

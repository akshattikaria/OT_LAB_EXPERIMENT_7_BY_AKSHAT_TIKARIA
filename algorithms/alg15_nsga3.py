"""
alg15_nsga3.py - Non-dominated Sorting Genetic Algorithm III (NSGA-III)
=======================================================================
Family: Elitist, Many-Objective Evolutionary (Part 3, 5E)

Core Mechanism:
Proposed by Deb & Jain (2014).
Designed specifically for many-objective optimization (M >= 3), where traditional
Pareto dominance deteriorates and crowding distance fails to distribute solutions.
Key features:
1. Replaces crowding distance with a structured set of predefined Reference Points
   distributed uniformly on the unit hyperplane (Das & Dennis systematic simplex grid).
2. Normalization of objective values using adaptive ideal points and hyperplane intercepts.
3. Association: Each candidate solution in the critical front F_L is projected onto
   the normalized reference lines, computing perpendicular distance.
4. Niching selection: Fills remaining population slots by prioritizing reference points
   with the smallest association count (rho_j).

Lecture Steps:
1. Generate structured reference points on the unit simplex.
2. Formulate (N + N) elitist pool R_t = P_t U Q_t and perform non-dominated sorting.
3. Select non-dominated fronts F_1, ..., F_{L-1}.
4. Normalize objective space of R_t using ideal point z* and extreme intercepts.
5. Associate every solution with its nearest reference direction.
6. Apply reference-point niching to select solutions from front F_L to complete size N.
7. Perform variation (SBX + Polynomial Mutation) to generate next offspring pool.

Default Parameters:
- p_divisions: 4 (for M=5) or 12 (for M=3)
- pop_size: matches number of reference points (approx 100)
- n_gen: 50
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu
from operators import sbx_crossover, polynomial_mutation


def generate_reference_points(M: int, p: int) -> np.ndarray:
    """Generates Das & Dennis reference points on the unit hyperplane."""
    ref_points = []

    def get_points(m_idx, remaining, current_pt):
        if m_idx == M - 1:
            current_pt.append(remaining / p)
            ref_points.append(current_pt.copy())
            current_pt.pop()
            return
        for i in range(remaining + 1):
            current_pt.append(i / p)
            get_points(m_idx + 1, remaining - i, current_pt)
            current_pt.pop()

    get_points(0, p, [])
    return np.array(ref_points)


def associate_to_reference_points(F_norm: np.ndarray, ref_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Associates each solution with its closest reference point based on perpendicular distance.
    Returns:
        min_indices: index of nearest reference point for each solution
        min_dists: perpendicular distance to that reference point
    """
    N = len(F_norm)
    num_refs = len(ref_points)
    # Unit vectors for reference rays
    ref_norms = np.linalg.norm(ref_points, axis=1, keepdims=True)
    ref_norms[ref_norms == 0] = 1.0
    w = ref_points / ref_norms

    # Projection of F onto w: (N, num_refs)
    proj = F_norm @ w.T
    # Perpendicular distance: d_perp = sqrt( ||F||^2 - (F . w)^2 )
    f_sq_norms = np.sum(F_norm ** 2, axis=1, keepdims=True)
    d_perp_sq = np.maximum(0.0, f_sq_norms - (proj ** 2))
    d_perp = np.sqrt(d_perp_sq)

    min_indices = np.argmin(d_perp, axis=1)
    min_dists = np.min(d_perp, axis=1)
    return min_indices, min_dists


def niching_selection(K: int, F_norm: np.ndarray, ref_points: np.ndarray,
                      chosen_indices: List[int], last_front: List[int]) -> List[int]:
    """
    Selects K individuals from last_front using NSGA-III reference point niching.
    """
    all_indices = chosen_indices + last_front
    sub_F = F_norm[all_indices]
    assoc_refs, dists = associate_to_reference_points(sub_F, ref_points)

    num_refs = len(ref_points)
    rho = np.zeros(num_refs, dtype=int)

    # Count associations for already chosen individuals
    n_chosen = len(chosen_indices)
    for i in range(n_chosen):
        rho[assoc_refs[i]] += 1

    last_front_rel = list(range(n_chosen, len(all_indices)))
    selected_from_last = []

    while len(selected_from_last) < K and len(last_front_rel) > 0:
        # Find reference point with minimum rho
        min_rho = np.min(rho)
        candidate_refs = np.where(rho == min_rho)[0]
        r_chosen = np.random.choice(candidate_refs)

        # Find candidates in last front associated with r_chosen
        assoc_in_last = [idx for idx in last_front_rel if assoc_refs[idx] == r_chosen]

        if len(assoc_in_last) == 0:
            # Exclude this reference point from selection in this round
            rho[r_chosen] += 1000
            continue

        if rho[r_chosen] == 0:
            # Pick one with minimal perpendicular distance
            best_cand = min(assoc_in_last, key=lambda idx: dists[idx])
        else:
            # Pick random associated candidate
            best_cand = np.random.choice(assoc_in_last)

        selected_from_last.append(all_indices[best_cand])
        last_front_rel.remove(best_cand)
        rho[r_chosen] += 1

    # If any remaining slots, randomly pad
    while len(selected_from_last) < K and len(last_front) > 0:
        rem = [idx for idx in last_front if idx not in selected_from_last]
        if len(rem) == 0:
            break
        selected_from_last.append(rem[0])

    return selected_from_last


def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any, ref_points: np.ndarray,
         pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One generational step of NSGA-III."""
    N = len(pop)

    # Generate N offspring using random selection + SBX + mutation
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

    r_fronts = mu.fast_non_dominated_sort(R_F, R_CV)

    chosen_indices = []
    last_front = []
    for front in r_fronts:
        if len(chosen_indices) + len(front) <= N:
            chosen_indices.extend(front)
        else:
            last_front = front
            break

    if len(chosen_indices) < N:
        K = N - len(chosen_indices)
        # Normalize objectives
        F_norm, _, _ = mu.normalize_objectives(R_F)
        selected_from_last = niching_selection(K, F_norm, ref_points, chosen_indices, last_front)
        chosen_indices.extend(selected_from_last)

    new_pop = R_pop[chosen_indices[:N]]
    new_F = R_F[chosen_indices[:N]]
    new_CV = R_CV[chosen_indices[:N]]
    return new_pop, new_F, new_CV


def run(problem: Any, p_divisions: int = 12, n_gen: int = 50, seed: int = 42) -> Dict[str, Any]:
    """Runs NSGA-III."""
    np.random.seed(seed)
    M = problem.n_objs
    if M == 3:
        p = 12 # 91 reference points
    elif M == 5:
        p = 4  # 70 reference points
    else:
        p = p_divisions

    ref_points = generate_reference_points(M, p)
    pop_size = len(ref_points)
    # Ensure pop_size is even
    if pop_size % 2 != 0:
        pop_size += 1

    rng = np.random.default_rng(seed)
    pop = rng.uniform(problem.xl, problem.xu, size=(pop_size, problem.n_vars))
    pop_evals = [problem.evaluate(x) for x in pop]
    F = np.array([e[0] for e in pop_evals])
    CV = np.array([e[1] for e in pop_evals])

    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, problem, ref_points)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'NSGA-III',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'ref_points': ref_points,
        'evaluations': pop_size * (n_gen + 1)
    }

"""
alg28_motlbo.py - Multi-Objective Teaching-Learning-Based Optimization (MO-TLBO)
================================================================================
Family: Human-inspired Metaheuristic (Part 4)

Core Mechanism:
Proposed by Rao et al. Models the classroom pedagogical process:
1. Teacher Phase:
   The "teacher" x_teacher is chosen from Front 1 (the non-dominated set with highest
   crowding distance). The class mean position M_mean is calculated.
   Teaching Factor T_F = round(1 + rand()) in {1, 2}.
   x_{new} = x_i + rand() * (x_teacher - T_F * M_mean)
2. Learner Phase:
   Learners improve by interacting with peers. For student i, a peer j != i is drawn:
   - If x_i dominates x_j: x_{new} = x_i + rand() * (x_i - x_j)
   - Else:                 x_{new} = x_i + rand() * (x_j - x_i)
3. Non-dominated Sorting & Crowding Distance Selection:
   At each phase, improvements replace targets if they dominate; mutually non-dominated
   solutions enter an extended pool and are pruned to maintain population size N.

Lecture Steps:
1. Initialize classroom population of size N. Evaluate objectives and constraints.
2. In each generation:
   a. Teacher Phase:
      - Perform non-dominated sort to identify Front 1.
      - Select teacher from Front 1 having maximum crowding distance.
      - Compute class mean M_mean.
      - Update all students towards the teacher and evaluate.
   b. Learner Phase:
      - Pair each student with a random peer.
      - Update knowledge based on relative dominance and evaluate.
   c. Prune combined pool to size N using non-dominated sort and crowding distance.
3. Return non-dominated front.

Default Parameters:
- pop_size: 100
- n_gen: 50
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


def select_teacher(pop: np.ndarray, F: np.ndarray, CV: np.ndarray) -> np.ndarray:
    """Selects teacher from Front 1 via crowding-distance weighted random choice.

    Bug fix (S3d): Previously always picked the single max-crowding-distance
    individual (always an extreme point), causing population collapse. Now selects
    randomly from front 1 with probability proportional to crowding distance,
    giving diverse guidance across iterations.
    """
    fronts = mu.fast_non_dominated_sort(F, CV)
    f1 = fronts[0] if len(fronts) > 0 else list(range(len(pop)))
    if len(f1) == 1:
        return pop[f1[0]].copy()

    cd = mu.calculate_crowding_distance(F, f1)
    finite_cd = cd[np.isfinite(cd)]
    max_cd = np.max(finite_cd) if len(finite_cd) > 0 else 1.0
    clean_cd = np.where(np.isinf(cd), max_cd * 2.0, cd)
    clean_cd = np.maximum(clean_cd, 1e-6)

    # Weighted random selection: diverse teachers across generations
    probs = clean_cd / np.sum(clean_cd)
    chosen_local = np.random.choice(len(f1), p=probs)
    return pop[f1[chosen_local]].copy()



def step(pop: np.ndarray, F: np.ndarray, CV: np.ndarray,
         problem: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One generational cycle (Teacher + Learner phases) of MO-TLBO."""
    N, D = pop.shape
    pool_X = list(pop)
    pool_F = list(F)
    pool_CV = list(CV)

    # 1. Teacher Phase
    teacher = select_teacher(pop, F, CV)
    M_mean = np.mean(pop, axis=0)

    for i in range(N):
        T_F = np.random.choice([1, 2])
        r = np.random.rand(D)
        x_new = pop[i] + r * (teacher - T_F * M_mean)
        x_new = np.clip(x_new, problem.xl, problem.xu)
        f_new, cv_new = problem.evaluate(x_new)

        if mu.deb_constrained_dominates(f_new, cv_new, F[i], CV[i]):
            pop[i] = x_new
            F[i] = f_new
            CV[i] = cv_new
        else:
            pool_X.append(x_new)
            pool_F.append(f_new)
            pool_CV.append(cv_new)

    # 2. Learner Phase
    for i in range(N):
        j = np.random.choice([idx for idx in range(N) if idx != i])
        r = np.random.rand(D)

        if mu.deb_constrained_dominates(F[i], CV[i], F[j], CV[j]):
            x_new = pop[i] + r * (pop[i] - pop[j])
        else:
            x_new = pop[i] + r * (pop[j] - pop[i])

        x_new = np.clip(x_new, problem.xl, problem.xu)
        f_new, cv_new = problem.evaluate(x_new)

        if mu.deb_constrained_dominates(f_new, cv_new, F[i], CV[i]):
            pop[i] = x_new
            F[i] = f_new
            CV[i] = cv_new
        else:
            pool_X.append(x_new)
            pool_F.append(f_new)
            pool_CV.append(cv_new)

    # Combine and prune to size N
    arr_X = np.array(pool_X)
    arr_F = np.array(pool_F)
    arr_CV = np.array(pool_CV)

    fronts = mu.fast_non_dominated_sort(arr_F, arr_CV)
    new_indices = []
    for front in fronts:
        if len(new_indices) + len(front) <= N:
            new_indices.extend(front)
        else:
            needed = N - len(new_indices)
            cd = mu.calculate_crowding_distance(arr_F, front)
            sorted_idx = np.argsort(-cd)
            for idx in range(needed):
                new_indices.append(front[sorted_idx[idx]])
            break

    return arr_X[new_indices], arr_F[new_indices], arr_CV[new_indices]


def run(problem: Any, pop_size: int = 100, n_gen: int = 50, seed: int = 42) -> Dict[str, Any]:
    """Runs MO-TLBO."""
    np.random.seed(seed)
    pop, F, CV = initialize(problem, pop_size=pop_size, seed=seed)

    for gen in range(n_gen):
        pop, F, CV = step(pop, F, CV, problem)

    fronts = mu.fast_non_dominated_sort(F, CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MO-TLBO',
        'pop': pop,
        'F': F,
        'CV': CV,
        'X_front': pop[nd_indices],
        'F_front': F[nd_indices],
        'evaluations': pop_size * (2 * n_gen + 1)
    }

"""
alg17_pesa2.py - Pareto Envelope-based Selection Algorithm II (PESA-II)
=======================================================================
Family: Elitist Evolutionary (Part 3)
Assigned Deep-Dive Algorithm for R = 05 (Task C).

Core Mechanism:
Proposed by Corne, Jerram, Knowles, & Oates (2001).
PESA-II introduces Region-Based Selection to eliminate the crowding bias of
traditional selection operators:
    Instead of selecting individuals directly, PESA-II selects HYPERBOX REGIONS
    of the objective space grid via tournament, then uniformly selects an individual
    residing within the winning hyperbox.
Because each non-empty grid region participates equally regardless of how many
individuals it contains, isolated solutions in sparse regions are sampled with
equal probability as dense clusters, maximizing diversity along the Pareto front.

Lecture Steps:
1. Discretize objective space into a hyper-grid of K divisions along each objective.
2. Maintain an external archive of non-dominated individuals of bounded size N_arc.
3. Map every archive solution to a grid coordinate vector:
   box_m(x) = floor( K * (f_m(x) - f_min_m) / (f_max_m - f_min_m) ).
4. Compute the "squeeze factor" of each box (number of solutions inside it).
5. Region-Based Selection:
   a. Identify all non-empty hyperboxes.
   b. Pick two non-empty hyperboxes at random.
   c. The hyperbox with smaller squeeze factor (lower density) wins the tournament.
   d. Select one individual uniformly from the winning box to serve as Parent 1.
   e. Repeat to select Parent 2.
6. Recombination & Mutation: Apply SBX and polynomial mutation to create offspring.
7. Archive Update:
   - Discard offspring if dominated by any archive member.
   - Remove any archive members dominated by offspring.
   - If non-dominated: insert if space permits; if full, replace an individual from
     the most crowded hyperbox if offspring's box is less crowded.

Default Parameters:
- archive_size: 100
- n_grid_divisions (K): 10
- n_gen: 50
- pc: 0.9, eta_c: 20.0, pm: 1/n_vars, eta_m: 20.0

Note on Population Structure:
PESA-II has no separate internal population. Unlike standard genetic algorithms that maintain
both an internal working population and an external archive, PESA-II maintains only the external
grid archive. Parents are selected directly from the hyperboxes of this archive via region-based
selection, and generated offspring are immediately integrated back into the archive.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu
from operators import sbx_crossover, polynomial_mutation


def initialize(problem: Any, archive_size: int = 100, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates initial population and seeds the archive."""
    rng = np.random.default_rng(seed)
    pop = rng.uniform(problem.xl, problem.xu, size=(archive_size, problem.n_vars))
    F_list, CV_list = [], []
    for ind in pop:
        f, cv = problem.evaluate(ind)
        F_list.append(f)
        CV_list.append(cv)

    F = np.array(F_list)
    CV = np.array(CV_list)

    # Filter initial non-dominated solutions into archive
    fronts = mu.fast_non_dominated_sort(F, CV)
    arc_indices = fronts[0] if len(fronts) > 0 else []

    archive_pop = pop[arc_indices]
    archive_F = F[arc_indices]
    archive_CV = CV[arc_indices]

    return archive_pop, archive_F, archive_CV


def assign_hyperboxes(F: np.ndarray, K: int = 10) -> Tuple[List[Tuple[int, ...]], Dict[Tuple[int, ...], List[int]], Dict[Tuple[int, ...], int]]:
    """
    Partitions objective space into K^M hyperboxes.
    Returns:
        box_assignments: list of box tuple IDs for each solution in F
        boxes_dict: mapping from box tuple ID to list of solution indices
        squeeze_factors: mapping from box tuple ID to count of solutions (squeeze factor)
    """
    N, M = F.shape
    if N == 0:
        return [], {}, {}

    f_min = np.min(F, axis=0)
    f_max = np.max(F, axis=0)
    denom = f_max - f_min
    denom[denom == 0] = 1e-12

    box_assignments = []
    boxes_dict: Dict[Tuple[int, ...], List[int]] = {}

    for i in range(N):
        coords = []
        for m in range(M):
            norm_val = (F[i, m] - f_min[m]) / denom[m]
            grid_idx = int(np.floor(norm_val * K))
            grid_idx = max(0, min(K - 1, grid_idx))
            coords.append(grid_idx)
        box_id = tuple(coords)
        box_assignments.append(box_id)
        if box_id not in boxes_dict:
            boxes_dict[box_id] = []
        boxes_dict[box_id].append(i)

    squeeze_factors = {box_id: len(indices) for box_id, indices in boxes_dict.items()}
    return box_assignments, boxes_dict, squeeze_factors


def region_based_selection(archive_pop: np.ndarray,
                           boxes_dict: Dict[Tuple[int, ...], List[int]],
                           squeeze_factors: Dict[Tuple[int, ...], int],
                           rng: Any = None) -> np.ndarray:
    """
    PESA-II Region-based selection:
    1. Selects two hyperboxes at random from all non-empty boxes.
    2. Box with lower squeeze factor wins.
    3. Randomly picks an individual from winning box.
    """
    if rng is None:
        rng = np.random
    _choice = lambda a, **kw: rng.choice(a, **kw)
    _rand = lambda: float(rng.rand()) if hasattr(rng, 'rand') else float(rng.random())

    non_empty_boxes = list(boxes_dict.keys())
    if len(non_empty_boxes) == 1:
        chosen_box = non_empty_boxes[0]
    else:
        # Random tournament between two regions
        b1, b2 = _choice(len(non_empty_boxes), size=2, replace=False)
        box1 = non_empty_boxes[b1]
        box2 = non_empty_boxes[b2]

        sq1 = squeeze_factors[box1]
        sq2 = squeeze_factors[box2]

        if sq1 < sq2:
            chosen_box = box1
        elif sq2 < sq1:
            chosen_box = box2
        else:
            chosen_box = box1 if _rand() < 0.5 else box2

    # Pick individual uniformly from winning box
    ind_idx = _choice(boxes_dict[chosen_box])
    return archive_pop[ind_idx].copy()


def update_pesa_archive(candidate: np.ndarray, f_cand: np.ndarray, cv_cand: float,
                        archive_pop: np.ndarray, archive_F: np.ndarray, archive_CV: np.ndarray,
                        max_archive_size: int, K: int = 10,
                        rng: Any = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Updates PESA-II archive with a single candidate offspring."""
    if rng is None:
        rng = np.random
    _choice = lambda a, **kw: rng.choice(a, **kw)

    N_arc = len(archive_pop)
    if N_arc == 0:
        return np.array([candidate]), np.array([f_cand]), np.array([cv_cand])

    # Check dominance
    dominated_by_cand = []
    is_dominated = False

    for i in range(N_arc):
        if mu.deb_constrained_dominates(archive_F[i], archive_CV[i], f_cand, cv_cand):
            is_dominated = True
            break
        if mu.deb_constrained_dominates(f_cand, cv_cand, archive_F[i], archive_CV[i]):
            dominated_by_cand.append(i)

    if is_dominated:
        # Offspring is dominated, discard
        return archive_pop, archive_F, archive_CV

    # Remove any members dominated by candidate
    if len(dominated_by_cand) > 0:
        keep_mask = np.ones(N_arc, dtype=bool)
        keep_mask[dominated_by_cand] = False
        archive_pop = archive_pop[keep_mask]
        archive_F = archive_F[keep_mask]
        archive_CV = archive_CV[keep_mask]

    # Append candidate
    archive_pop = np.vstack([archive_pop, candidate])
    archive_F = np.vstack([archive_F, f_cand])
    archive_CV = np.append(archive_CV, cv_cand)

    # If archive exceeds capacity, remove individual from the most crowded hyperbox
    if len(archive_pop) > max_archive_size:
        _, boxes_dict, squeeze_factors = assign_hyperboxes(archive_F, K=K)
        # Find box with highest squeeze factor
        max_sq = max(squeeze_factors.values())
        most_crowded_boxes = [b for b, sq in squeeze_factors.items() if sq == max_sq]
        crowded_box = most_crowded_boxes[_choice(len(most_crowded_boxes))]

        # Remove random individual from that box
        remove_idx = _choice(boxes_dict[crowded_box])
        archive_pop = np.delete(archive_pop, remove_idx, axis=0)
        archive_F = np.delete(archive_F, remove_idx, axis=0)
        archive_CV = np.delete(archive_CV, remove_idx)

    return archive_pop, archive_F, archive_CV


def step(archive_pop: np.ndarray, archive_F: np.ndarray, archive_CV: np.ndarray,
         problem: Any, archive_size: int = 100, K: int = 10,
         pc: float = 0.9, eta_c: float = 20.0, eta_m: float = 20.0,
         batch_size: Optional[int] = None,
         rng: Any = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates a batch of offspring and updates PESA-II archive."""
    if rng is None:
        rng = np.random
    _, boxes_dict, squeeze_factors = assign_hyperboxes(archive_F, K=K)
    if batch_size is None:
        batch_size = max(10, archive_size // 5)

    for _ in range(batch_size):
        # Select parents using region-based selection
        p1 = region_based_selection(archive_pop, boxes_dict, squeeze_factors, rng=rng)
        p2 = region_based_selection(archive_pop, boxes_dict, squeeze_factors, rng=rng)

        c1, c2 = sbx_crossover(p1, p2, problem.xl, problem.xu, eta_c=eta_c, pc=pc, rng=rng)
        c1 = polynomial_mutation(c1, problem.xl, problem.xu, eta_m=eta_m, rng=rng)

        f_c1, cv_c1 = problem.evaluate(c1)
        archive_pop, archive_F, archive_CV = update_pesa_archive(
            c1, f_c1, cv_c1, archive_pop, archive_F, archive_CV,
            max_archive_size=archive_size, K=K, rng=rng
        )

        # Recompute hyperboxes periodically
        _, boxes_dict, squeeze_factors = assign_hyperboxes(archive_F, K=K)

    return archive_pop, archive_F, archive_CV


def run(problem: Any, archive_size: int = 100, n_gen: int = 50,
        K: int = 10, seed: int = 42) -> Dict[str, Any]:
    """Runs PESA-II."""
    np.random.seed(seed)
    archive_pop, archive_F, archive_CV = initialize(problem, archive_size=archive_size, seed=seed)

    for gen in range(n_gen):
        archive_pop, archive_F, archive_CV = step(
            archive_pop, archive_F, archive_CV, problem,
            archive_size=archive_size, K=K
        )

    fronts = mu.fast_non_dominated_sort(archive_F, archive_CV)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'PESA-II',
        'pop': archive_pop,
        'F': archive_F,
        'CV': archive_CV,
        'X_front': archive_pop[nd_indices],
        'F_front': archive_F[nd_indices],
        'K': K,
        'evaluations': archive_size + n_gen * max(10, archive_size // 5)
    }

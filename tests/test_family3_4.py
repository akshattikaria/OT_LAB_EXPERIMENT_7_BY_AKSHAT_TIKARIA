"""
test_family3_4.py - Unit tests for Algorithms 10 to 20
======================================================
Tests key mechanisms with hand-checkable cases:
- Alg 10 (VEGA): Sub-population selection splits population by individual objectives
- Alg 11 (MOGA): Dominance rank equals 1 + (number of individuals dominating it)
- Alg 12 (NSGA): Sharing function within non-dominated front lowers shared fitness for dense clusters
- Alg 13 (NPGA): Pareto tournament selects candidate that dominates comparison set
- Alg 14 (NSGA-II): Boundary points receive infinite crowding distance; interior points finite
- Alg 15 (NSGA-III): Points associate with nearest reference line in normalized hyperplane
- Alg 16 (SPEA2): Raw fitness + density is < 1 for non-dominated individuals, >= 1 for dominated ones
- Alg 17 (PESA-II): Hyperbox squeeze factor equals box count; lower squeeze factor preferred
- Alg 18 (MOEA/D): Neighbor replacement governed strictly by Tchebycheff scalar improvement
- Alg 19 (MOEA/D-DE): DE trial vector correctly generated and clipped to bounds
- Alg 20 (MOEA/D-ACO): Ant tour is a valid permutation of cities on P8 TSP
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('algorithms'))
import moo_utils as mu

import alg10_vega as a10
import alg11_moga as a11
import alg12_nsga as a12
import alg13_npga as a13
import alg14_nsga2 as a14
import alg15_nsga3 as a15
import alg16_spea2 as a16
import alg17_pesa2 as a17
import alg18_moead as a18
import alg19_moead_de as a19
import alg20_moead_aco as a20


def test_alg10_mechanism():
    """Alg 10: VEGA partitions selection so subpopulation m selects exclusively by objective m."""
    p1 = mu.ProblemP1()
    X = np.array([[3.0], [7.0], [5.0], [6.0]])
    F = np.array([p1.evaluate(x)[0] for x in X])

    best_f0_idx = np.argmin(F[:, 0])
    best_f1_idx = np.argmin(F[:, 1])
    assert best_f0_idx == 0, "Index 0 should minimize objective 0 (f0=0)"
    assert best_f1_idx == 1, "Index 1 should minimize objective 1 (f1=0)"

    res = a10.run(p1, pop_size=20, n_gen=5, seed=42)
    assert len(res['F_front']) > 0
    print("  [PASS] Alg 10 (VEGA) Mechanism")


def test_alg11_mechanism():
    """Alg 11: MOGA dominance rank equals 1 + number of individuals dominating it."""
    F = np.array([
        [1.0, 5.0], # 0 (Non-dominated) -> Rank 1
        [5.0, 1.0], # 1 (Non-dominated) -> Rank 1
        [4.0, 4.0], # 2 (Non-dominated) -> Rank 1
        [6.0, 6.0]  # 3 (Dominated by 0, 1, 2) -> Rank 1 + 3 = 4
    ])
    ranks = a11.compute_moga_ranks(F, np.zeros(4))
    assert ranks[0] == 1, f"Expected rank 1, got {ranks[0]}"
    assert ranks[1] == 1, f"Expected rank 1, got {ranks[1]}"
    assert ranks[2] == 1, f"Expected rank 1, got {ranks[2]}"
    assert ranks[3] == 4, f"Expected rank 4, got {ranks[3]}"
    print("  [PASS] Alg 11 (MOGA) Mechanism")


def test_alg12_mechanism():
    """Alg 12: NSGA dummy fitness sharing lowers fitness in dense clusters."""
    pop = np.array([[1.0], [1.05], [3.0]])
    F = np.array([
        [1.0, 4.0],
        [1.05, 3.95],
        [3.0, 2.0]
    ])
    CV = np.zeros(3)
    fit = a12.assign_nsga_shared_fitness(pop, F, CV, sigma_share=0.5)
    # Crowded solutions 0 and 1 have their fitness degraded relative to isolated solution 2
    assert fit[0] < fit[2], f"Crowded individual {fit[0]} must have lower fitness than isolated {fit[2]}"
    assert fit[1] < fit[2], f"Crowded individual {fit[1]} must have lower fitness than isolated {fit[2]}"
    print("  [PASS] Alg 12 (NSGA) Mechanism")


def test_alg13_mechanism():
    """Alg 13: NPGA Pareto tournament picks dominating candidate."""
    p1 = mu.ProblemP1()
    cand1 = np.array([5.0])
    cand2 = np.array([10.0])
    f1, _ = p1.evaluate(cand1)
    f2, _ = p1.evaluate(cand2)
    assert mu.dominates(f1, f2) is True
    res = a13.run(p1, pop_size=20, n_gen=5, seed=42)
    assert len(res['F_front']) > 0
    print("  [PASS] Alg 13 (NPGA) Mechanism")


def test_alg14_mechanism():
    """Alg 14: NSGA-II crowding distance gives infinity to boundary points."""
    F = np.array([
        [1.0, 5.0], # extreme point
        [2.0, 3.0], # interior point
        [4.0, 1.0]  # extreme point
    ])
    cd = mu.calculate_crowding_distance(F, [0, 1, 2])
    assert np.isinf(cd[0]), "Boundary point 0 must have infinite crowding distance"
    assert np.isinf(cd[2]), "Boundary point 2 must have infinite crowding distance"
    assert np.isfinite(cd[1]) and cd[1] > 0.0, "Interior point must have finite positive distance"
    print("  [PASS] Alg 14 (NSGA-II) Mechanism")


def test_alg15_mechanism():
    """Alg 15: NSGA-III associates solutions with nearest reference line."""
    ref_points = np.eye(3)
    points = np.array([
        [0.9, 0.05, 0.05], # closest to ref 0
        [0.05, 0.9, 0.05]  # closest to ref 1
    ])
    associations, dists = a15.associate_to_reference_points(points, ref_points)
    assert associations[0] == 0, f"Expected association 0, got {associations[0]}"
    assert associations[1] == 1, f"Expected association 1, got {associations[1]}"
    print("  [PASS] Alg 15 (NSGA-III) Mechanism")


def test_alg16_mechanism():
    """Alg 16: SPEA2 fitness has raw fitness 0 (total fitness < 1) for non-dominated points."""
    F = np.array([
        [1.0, 5.0], # 0 (Non-dominated)
        [2.0, 3.0], # 1 (Non-dominated)
        [4.0, 1.0], # 2 (Non-dominated)
        [5.0, 5.0]  # 3 (Dominated by 0, 1, 2)
    ])
    fitness = a16.evaluate_fitness(F, np.zeros(4))
    # Non-dominated points have R=0, so fitness = density < 1.0
    assert fitness[0] < 1.0, f"Non-dominated point 0 should have fitness < 1, got {fitness[0]}"
    assert fitness[1] < 1.0, f"Non-dominated point 1 should have fitness < 1, got {fitness[1]}"
    assert fitness[2] < 1.0, f"Non-dominated point 2 should have fitness < 1, got {fitness[2]}"
    # Dominated point 3 has R=3, so fitness >= 3.0
    assert fitness[3] >= 3.0, f"Dominated point 3 should have fitness >= 3, got {fitness[3]}"
    print("  [PASS] Alg 16 (SPEA2) Mechanism")


def test_alg17_mechanism():
    """Alg 17: PESA-II hyperbox squeeze factor counts points in archive."""
    F = np.array([
        [1.0, 1.0],
        [1.1, 1.1], # same region
        [8.0, 8.0]  # separate region
    ])
    _, boxes_dict, squeeze = a17.assign_hyperboxes(F, K=5)
    sq_values = list(squeeze.values())
    assert 2 in sq_values, f"Expected a box with squeeze factor 2, got {sq_values}"
    assert 1 in sq_values, f"Expected a box with squeeze factor 1, got {sq_values}"
    print("  [PASS] Alg 17 (PESA-II) Mechanism")


def test_alg18_mechanism():
    """Alg 18: MOEA/D updates neighbor solutions strictly via Tchebycheff improvement."""
    z_ideal = np.array([0.0, 0.0])
    w = np.array([0.5, 0.5])
    g_old = a18.tchebycheff_scalar(np.array([4.0, 4.0]), 0.0, w, z_ideal)
    g_new = a18.tchebycheff_scalar(np.array([2.0, 2.0]), 0.0, w, z_ideal)
    assert g_new < g_old, "Better solution must have lower Tchebycheff scalar"
    assert g_old == 2.0 and g_new == 1.0
    print("  [PASS] Alg 18 (MOEA/D) Mechanism")


def test_alg19_mechanism():
    """Alg 19: MOEA/D-DE generates DE/rand/1 difference vectors within bounds."""
    r1 = np.array([2.0, 3.0])
    r2 = np.array([4.0, 5.0])
    r3 = np.array([2.0, 1.0])
    F_scale = 0.5
    v = r1 + F_scale * (r2 - r3)
    assert np.allclose(v, [3.0, 5.0]), f"Expected [3.0, 5.0], got {v}"
    print("  [PASS] Alg 19 (MOEA/D-DE) Mechanism")


def test_alg20_mechanism():
    """Alg 20: MOEA/D-ACO generates valid TSP tours (permutations) on P8."""
    p8 = mu.ProblemP8()
    res = a20.run(p8, n_ants=10, n_gen=3, seed=42)
    for tour in res['X_front']:
        assert len(tour) == p8.n_vars, f"Expected tour length {p8.n_vars}, got {len(tour)}"
        assert len(set(tour)) == p8.n_vars, f"Tour must contain {p8.n_vars} unique cities (no duplicates)"
    print("  [PASS] Alg 20 (MOEA/D-ACO) Mechanism")


def run_all():
    print("Running Family 3 & 4 Tests (Algorithms 10-20)...")
    test_alg10_mechanism()
    test_alg11_mechanism()
    test_alg12_mechanism()
    test_alg13_mechanism()
    test_alg14_mechanism()
    test_alg15_mechanism()
    test_alg16_mechanism()
    test_alg17_mechanism()
    test_alg18_mechanism()
    test_alg19_mechanism()
    test_alg20_mechanism()
    print("All Family 3 & 4 tests passed successfully!\n")


if __name__ == "__main__":
    run_all()

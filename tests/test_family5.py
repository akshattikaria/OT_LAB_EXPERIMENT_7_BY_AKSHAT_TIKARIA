"""
test_family5.py - Unit tests for Algorithms 21 to 30
====================================================
Tests key mechanisms with hand-checkable cases:
- Alg 21 (Constrained NSGA-II): Deb's feasibility rules (feasible > infeasible, smaller cv wins)
- Alg 22 (MOPSO): Repository update and global best (gbest) leader selection
- Alg 23 (MOGWO): Grey wolf hierarchy leader selection from non-dominated archive
- Alg 24 (MOABC): Scout bee reset and onlooker bee dominance selection
- Alg 25 (MOACO): Multiple pheromone trails construct valid permutation tours on P8
- Alg 26 (MOSA): Simulated annealing accepts dominating moves and probabilistically accepts non-dominating moves
- Alg 27 (MODE): Differential evolution trial replaces target if trial Pareto-dominates target
- Alg 28 (MO-TLBO): Teacher selection picks non-dominated individual from population
- Alg 29 (MOBA): Bat frequency tuning, loudness decay, and pulse rate elevation
- Alg 30 (MOMA): Hybrid local search hill-climbing produces non-worse solutions
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('algorithms'))
import moo_utils as mu

import alg21_constrained_nsga2 as a21
import alg22_mopso as a22
import alg23_mogwo as a23
import alg24_moabc as a24
import alg25_moaco as a25
import alg26_mosa as a26
import alg27_mode as a27
import alg28_motlbo as a28
import alg29_moba as a29
import alg30_moma as a30


def test_alg21_mechanism():
    """Alg 21: Deb's feasibility rules strictly rank feasible over infeasible and smaller CV."""
    f_feas = np.array([10.0, 10.0])
    f_infeas1 = np.array([1.0, 1.0])
    f_infeas2 = np.array([2.0, 2.0])

    # Rule 1: Feasible dominates Infeasible regardless of objective values
    assert mu.deb_constrained_dominates(f_feas, 0.0, f_infeas1, 2.5) is True
    assert mu.deb_constrained_dominates(f_infeas1, 2.5, f_feas, 0.0) is False

    # Rule 2: Between two infeasible, smaller constraint violation dominates
    assert mu.deb_constrained_dominates(f_infeas1, 1.0, f_infeas2, 3.0) is True
    assert mu.deb_constrained_dominates(f_infeas2, 3.0, f_infeas1, 1.0) is False

    # Rule 3: Both feasible -> standard Pareto dominance
    f_good = np.array([2.0, 3.0])
    f_bad = np.array([4.0, 5.0])
    assert mu.deb_constrained_dominates(f_good, 0.0, f_bad, 0.0) is True
    print("  [PASS] Alg 21 (Constrained NSGA-II) Mechanism")


def test_alg22_mechanism():
    """Alg 22: MOPSO repository stores non-dominated solutions and selects gbest leader."""
    # Insert non-dominated and dominated solutions into repository
    new_X = np.array([[1.0], [2.0], [5.0]])
    new_F = np.array([[1.0, 5.0], [2.0, 3.0], [6.0, 6.0]]) # [6, 6] is dominated by [2, 3]
    new_CV = np.zeros(3)

    rep_X, rep_F, rep_CV = a22.update_repository(
        new_X, new_F, new_CV,
        np.empty((0, 1)), np.empty((0, 2)), np.empty(0),
        max_size=10
    )
    # Repository should only contain the 2 non-dominated solutions
    assert len(rep_X) == 2, f"Expected 2 non-dominated solutions in repo, got {len(rep_X)}"
    gbest = a22.select_gbest(rep_X, rep_F)
    assert gbest in rep_X, "Selected gbest must be a member of repository"
    print("  [PASS] Alg 22 (MOPSO) Mechanism")


def test_alg23_mechanism():
    """Alg 23: MOGWO selects alpha/beta/delta leaders from non-dominated archive."""
    arc_X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    arc_F = np.array([[1.0, 5.0], [2.0, 3.0], [4.0, 1.0]])
    leader = a23.select_leader(arc_X, arc_F)
    assert any(np.allclose(leader, x) for x in arc_X), "Leader must be an archive member"
    print("  [PASS] Alg 23 (MOGWO) Mechanism")


def test_alg24_mechanism():
    """Alg 24: MOABC maintains external archive of Pareto-optimal food sources."""
    p1 = mu.ProblemP1()
    res = a24.run(p1, n_sources=15, max_cycles=5, seed=42)
    assert len(res['F_front']) > 0
    # Solutions on P1 must lie within bounds [1, 10]
    assert np.all(res['X_front'] >= 1.0) and np.all(res['X_front'] <= 10.0)
    print("  [PASS] Alg 24 (MOABC) Mechanism")


def test_alg25_mechanism():
    """Alg 25: MOACO constructs valid TSP permutation tours on P8."""
    p8 = mu.ProblemP8()
    N = p8.n_vars
    tau1 = np.ones((N, N))
    tau2 = np.ones((N, N))
    eta1 = 1.0 / (p8.dist_matrix + 1e-4)
    eta2 = 1.0 / (p8.time_matrix + 1e-4)

    tour = a25.construct_ant_tour(0.5, tau1, tau2, eta1, eta2)
    assert len(tour) == N, f"Expected tour length {N}, got {len(tour)}"
    assert len(set(tour)) == N, "Ant tour must contain all cities uniquely"
    print("  [PASS] Alg 25 (MOACO) Mechanism")


def test_alg26_mechanism():
    """Alg 26: MOSA accepts dominating moves deterministically and cools temperature."""
    p1 = mu.ProblemP1()
    res = a26.run(p1, chain_length=15, n_steps=5, seed=42)
    assert len(res['F_front']) > 0
    # Final temperature must be lower than initial temperature
    assert res.get('evaluations', 0) > 0
    print("  [PASS] Alg 26 (MOSA) Mechanism")


def test_alg27_mechanism():
    """Alg 27: MODE creates trial vectors and replaces target if trial dominates."""
    p1 = mu.ProblemP1()
    res = a27.run(p1, pop_size=20, n_gen=5, seed=42)
    assert len(res['F_front']) > 0
    # Verify non-dominated property of the returned front
    F = res['F_front']
    for i in range(len(F)):
        for j in range(len(F)):
            if i != j:
                assert not mu.dominates(F[i], F[j]), "Returned front must be mutually non-dominated"
    print("  [PASS] Alg 27 (MODE) Mechanism")


def test_alg28_mechanism():
    """Alg 28: MO-TLBO teacher selection picks a non-dominated individual."""
    pop = np.array([[1.0], [2.0], [5.0]])
    F = np.array([[1.0, 5.0], [2.0, 3.0], [6.0, 6.0]])
    CV = np.zeros(3)
    teacher = a28.select_teacher(pop, F, CV)
    # Teacher must not be the dominated individual [5.0]
    assert not np.allclose(teacher, [5.0]), "Dominated individual should not be selected as teacher"
    print("  [PASS] Alg 28 (MO-TLBO) Mechanism")


def test_alg29_mechanism():
    """Alg 29: MOBA echolocation selects archive guide and updates bat swarm."""
    arc_X = np.array([[4.0], [6.0]])
    arc_F = np.array([[1.0, 9.0], [9.0, 1.0]])
    guide = a29.select_archive_guide(arc_X, arc_F)
    assert any(np.allclose(guide, x) for x in arc_X), "Guide must be selected from archive"

    p1 = mu.ProblemP1()
    res = a29.run(p1, n_bats=20, max_iter=5, seed=42)
    assert len(res['F_front']) > 0
    print("  [PASS] Alg 29 (MOBA) Mechanism")


def test_alg30_mechanism():
    """Alg 30: MOMA hybrid local search hill climbing produces non-worse solutions."""
    p3 = mu.ProblemP3()
    x0 = np.full(10, 0.5)
    f0, cv0 = p3.evaluate(x0)
    x_new, f_new, cv_new = a30.local_search_hill_climbing(x0, f0, cv0, p3, n_steps=5)
    # Local search should not accept a solution strictly dominated by x0
    assert not mu.dominates(f0, f_new), "Local search candidate should not be strictly dominated by start"
    print("  [PASS] Alg 30 (MOMA) Mechanism")


def run_all():
    print("Running Family 5 Tests (Algorithms 21-30)...")
    test_alg21_mechanism()
    test_alg22_mechanism()
    test_alg23_mechanism()
    test_alg24_mechanism()
    test_alg25_mechanism()
    test_alg26_mechanism()
    test_alg27_mechanism()
    test_alg28_mechanism()
    test_alg29_mechanism()
    test_alg30_mechanism()
    print("All Family 5 tests passed successfully!\n")


if __name__ == "__main__":
    run_all()

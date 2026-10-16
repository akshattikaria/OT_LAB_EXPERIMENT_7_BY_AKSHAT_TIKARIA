"""
test_moo_utils.py - Unit tests for the shared MOO toolkit and problem bank
========================================================================
Checks dominance, Deb's rules, non-dominated sorting, crowding distance,
HV, IGD, Spacing, and problem bank sanity checks P1-P14.
"""

import numpy as np
import moo_utils as mu

def test_dominance():
    # Minimization: u dominates v if all <= and at least one <
    u = np.array([1.0, 2.0])
    v = np.array([2.0, 2.0])
    w = np.array([2.0, 3.0])
    eq = np.array([1.0, 2.0])
    tradeoff = np.array([0.5, 3.0])

    assert mu.dominates(u, v) is True
    assert mu.dominates(v, u) is False
    assert mu.dominates(u, eq) is False # equality does not dominate
    assert mu.dominates(u, tradeoff) is False
    assert mu.dominates(tradeoff, u) is False

def test_deb_constrained_dominance():
    # Rule 1: Feasible dominates Infeasible
    f_feas = np.array([10.0, 10.0])
    f_infeas = np.array([1.0, 1.0])
    assert mu.deb_constrained_dominates(f_feas, 0.0, f_infeas, 2.5) is True
    assert mu.deb_constrained_dominates(f_infeas, 2.5, f_feas, 0.0) is False

    # Rule 2: Both Infeasible -> smaller CV wins
    assert mu.deb_constrained_dominates(f_infeas, 1.2, f_infeas, 3.4) is True
    assert mu.deb_constrained_dominates(f_infeas, 3.4, f_infeas, 1.2) is False

    # Rule 3: Both Feasible -> Pareto dominance decides
    f_good = np.array([2.0, 3.0])
    f_bad = np.array([4.0, 5.0])
    assert mu.deb_constrained_dominates(f_good, 0.0, f_bad, 0.0) is True
    assert mu.deb_constrained_dominates(f_bad, 0.0, f_good, 0.0) is False

def test_fast_non_dominated_sort():
    # Points:
    # A = (1, 5) - Front 1
    # B = (2, 3) - Front 1
    # C = (4, 1) - Front 1
    # D = (3, 4) - Dominated by B -> Front 2
    # E = (5, 5) - Dominated by A, B, C, D -> Front 3
    F = np.array([
        [1.0, 5.0], # 0
        [2.0, 3.0], # 1
        [4.0, 1.0], # 2
        [3.0, 4.0], # 3
        [5.0, 5.0]  # 4
    ])
    fronts = mu.fast_non_dominated_sort(F)
    assert len(fronts) == 3
    assert set(fronts[0]) == {0, 1, 2}
    assert set(fronts[1]) == {3}
    assert set(fronts[2]) == {4}

def test_crowding_distance():
    F = np.array([
        [1.0, 5.0],
        [2.0, 3.0],
        [4.0, 1.0]
    ])
    front_0 = [0, 1, 2]
    cd = mu.calculate_crowding_distance(F, front_0)
    # Endpoints should be inf
    # Points ordered by f1: 0 (f1=1), 1 (f1=2), 2 (f1=4)
    # Endpoints are index 0 and index 2
    assert np.isinf(cd[0])
    assert np.isinf(cd[2])
    # Interior point index 1:
    # in f1: (4 - 1)/(4 - 1) = 1.0
    # in f2: (5 - 1)/(5 - 1) = 1.0
    # total = 2.0
    assert abs(cd[1] - 2.0) < 1e-6

def test_hypervolume_2d():
    # Simple two points: (1, 4) and (3, 2). Ref point: (5, 5)
    # Dominated region:
    # From x=1 to x=3, height = (5 - 4) = 1 -> area = 2 * 1 = 2
    # From x=3 to x=5, height = (5 - 2) = 3 -> area = 2 * 3 = 6
    # Total area = 2 + 6 = 8
    F = np.array([
        [1.0, 4.0],
        [3.0, 2.0]
    ])
    ref = np.array([5.0, 5.0])
    hv = mu.hypervolume_2d(F, ref)
    assert abs(hv - 8.0) < 1e-6

def test_spacing():
    # Perfectly spaced points: (1, 3), (2, 2), (3, 1)
    # Pairwise Manhattan:
    # between 1 and 2: |1-2| + |3-2| = 2
    # between 2 and 3: |2-3| + |2-1| = 2
    # All d_i = 2 => d_mean = 2 => spacing = 0
    F = np.array([
        [1.0, 3.0],
        [2.0, 2.0],
        [3.0, 1.0]
    ])
    s = mu.spacing(F)
    assert abs(s - 0.0) < 1e-6

def test_problem_sanity_checks():
    # P1: at x=5, f=(4, 4); at x=3, f=(0, 16)
    p1 = mu.ProblemP1()
    f, cv = p1.evaluate(np.array([5.0]))
    assert np.allclose(f, [4.0, 4.0])
    f, cv = p1.evaluate(np.array([3.0]))
    assert np.allclose(f, [0.0, 16.0])

    # P2: at x=2, f=(50, 4); at x=10, f=(10, 100)
    p2 = mu.ProblemP2(integer_mode=False)
    f, cv = p2.evaluate(np.array([2.0]))
    assert np.allclose(f, [50.0, 4.0])
    f, cv = p2.evaluate(np.array([10.0]))
    assert np.allclose(f, [10.0, 100.0])

    # P4: w = (0, 0, 1, 0) gives return 0.15 and risk 0.09
    p4 = mu.ProblemP4()
    f, cv = p4.evaluate(np.array([0.0, 0.0, 1.0, 0.0]))
    assert abs(f[0] - 0.09) < 1e-6 # risk
    assert abs(f[1] - (-0.15)) < 1e-6 # -return

    # P5: (b, h) = (0.01, 0.05) gives mass 3.925 kg, deflection 0.016 m, cv = 0
    p5 = mu.ProblemP5()
    f, cv = p5.evaluate(np.array([0.01, 0.05]))
    assert abs(f[0] - 3.925) < 1e-4
    assert abs(f[1] - 0.016) < 1e-4
    assert cv == 0.0

    # P6: (0, 0) gives (0, 50) and feasible; (5, 3) gives (136, 4) and feasible
    p6 = mu.ProblemP6()
    f, cv = p6.evaluate(np.array([0.0, 0.0]))
    assert np.allclose(f, [0.0, 50.0])
    assert cv == 0.0
    f, cv = p6.evaluate(np.array([5.0, 3.0]))
    assert np.allclose(f, [136.0, 4.0])
    assert cv == 0.0

    # P7: (10, 80) gives (8, 96) [minimized: (-8, -96)]
    p7 = mu.ProblemP7()
    f, cv = p7.evaluate(np.array([10.0, 80.0]))
    assert np.allclose(f, [-8.0, -96.0])
    assert cv == 0.0

    # P10: when g=0, sum f_m^2 = 1
    p10 = mu.ProblemP10(M=3)
    x = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    f, cv = p10.evaluate(x)
    assert abs(np.sum(f ** 2) - 1.0) < 1e-6

    # P14: (13, 40) gives 1.53 kg and 9.0 h (f2 = -9.0); (13, 90) gives 2.13 kg and 20.22 h
    p14 = mu.ProblemP14()
    f, cv = p14.evaluate(np.array([13.0, 40.0]))
    assert abs(f[0] - 1.53) < 1e-4
    assert abs(f[1] - (-9.0)) < 0.1

    print("All problem sanity checks passed successfully!")


# ==============================================================================
# S1 METRICS UNIT TESTS (MASTER_PLAN.md)
# ==============================================================================

def test_p3_true_front_hv():
    """P3 true-front HV = 0.5433 +- 0.003."""
    p3 = mu.ProblemP3()
    tf500 = p3.true_pareto_front(500)
    hv, igd_val, sp = mu.compute_metrics(tf500, 'P3')
    assert hv is not None, "HV must not be None for P3"
    assert abs(hv - 0.5433) <= 0.003, f"P3 HV={hv:.5f} not in [0.5403, 0.5463]"
    print(f"P3 true-front HV = {hv:.5f} (target 0.5433 +- 0.003) - PASS")


def test_single_point_p3():
    """Single point (0,1) on P3 -> HV ~0.11 and finite IGD in [0.3, 1.0]."""
    pt = np.array([[0.0, 1.0]])
    hv, igd_val, sp = mu.compute_metrics(pt, 'P3')
    assert hv is not None, "HV must not be None"
    assert abs(hv - 0.11) <= 0.03, f"Single-point HV={hv:.5f} expected ~0.11"
    assert igd_val is not None and np.isfinite(igd_val), f"IGD must be finite, got {igd_val}"
    assert 0.3 <= igd_val <= 1.0, f"IGD={igd_val:.4f} not in [0.3, 1.0]"
    print(f"Single point (0,1) on P3: HV={hv:.5f}, IGD={igd_val:.4f} - PASS")


def test_igd_true_true():
    """IGD(true, true) < 1e-9."""
    p3 = mu.ProblemP3()
    tf500 = p3.true_pareto_front(500)
    hv, igd_val, sp = mu.compute_metrics(tf500, 'P3')
    assert igd_val is not None, "IGD must not be None"
    assert igd_val < 1e-9, f"IGD(true, true) = {igd_val:.2e} not < 1e-9"
    print(f"IGD(true, true) = {igd_val:.2e} (< 1e-9) - PASS")


def test_adding_nondominated_point_never_lowers_hv():
    """Adding a non-dominated point never lowers HV."""
    F1 = np.array([[0.2, 0.96], [0.5, 0.75], [0.8, 0.36]])
    hv1, _, _ = mu.compute_metrics(F1, 'P3')
    F2 = np.vstack([F1, [0.1, 0.99]])
    hv2, _, _ = mu.compute_metrics(F2, 'P3')
    assert hv2 >= hv1 - 1e-9, f"HV decreased: {hv1:.5f} -> {hv2:.5f}"
    print(f"HV before={hv1:.5f}, after adding non-dominated point={hv2:.5f} (never lowers) - PASS")


if __name__ == "__main__":
    test_dominance()
    test_deb_constrained_dominance()
    test_fast_non_dominated_sort()
    test_crowding_distance()
    test_hypervolume_2d()
    test_spacing()
    test_problem_sanity_checks()
    # S1 Acceptance Unit Tests
    test_p3_true_front_hv()
    test_single_point_p3()
    test_igd_true_true()
    test_adding_nondominated_point_never_lowers_hv()
    print("\nALL test_moo_utils.py TESTS PASSED SUCCESSFULLY!")

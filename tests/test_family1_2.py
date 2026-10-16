"""
test_family1_2.py - Unit tests for Algorithms 1 to 9 + Sanity Checks (P1, P11, P12)
===================================================================================
Tests key mechanisms with hand-checkable cases:
- P1 sanity check: Pareto set 3 <= x <= 7, x=5 gives (4,4), convex front
- P11 sanity check: corner points, max profit 2400, max volume 500, volume shortfall >= 100
- P12 sanity check: 5 suppliers, 4 criteria, bounds and ideal solutions
- Alg 01 (Weighted Sum): equal weights (0.5, 0.5) on P1 gives minimum at x = 5.0
- Alg 02 (Epsilon-Constraint): bound f2 <= 4.0 on P1 yields x = 5.0, f1 = 4.0
- Alg 03 (Benson's Method): reference point (10, 10) pushes point to Pareto set [3, 7]
- Alg 04 (Weight Metric): Tchebycheff norm (p=inf) with equal weights picks x = 5.0
- Alg 05 (Value Function): high lambda picks low risk, low lambda picks high return on P4
- Alg 06 (Goal Programming): profit shortfall is 0, volume shortfall is >= 100 on P11
- Alg 07 (Lexicographic): priority 1 gives profit 2400, priority 2 breaks tie at (300, 200)
- Alg 08 (TOPSIS): supplier 3 ranks #1 on P12 criteria table
- Alg 09 (VIKOR): supplier 1 has minimal Q and ranks #1 as best compromise on P12
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('algorithms'))
import moo_utils as mu

import alg01_weighted_sum as a01
import alg02_epsilon_constraint as a02
import alg03_benson as a03
import alg04_weight_metric as a04
import alg05_value_function as a05
import alg06_goal_programming as a06
import alg07_lexicographic as a07
import alg08_topsis as a08
import alg09_vikor as a09


def test_p1_sanity():
    """P1 sanity check from brief Section 4."""
    p1 = mu.ProblemP1()
    f5, cv5 = p1.evaluate(np.array([5.0]))
    assert np.allclose(f5, [4.0, 4.0]), f"Expected (4,4) at x=5, got {f5}"
    f3, cv3 = p1.evaluate(np.array([3.0]))
    assert np.allclose(f3, [0.0, 16.0]), f"Expected (0,16) at x=3, got {f3}"
    f7, cv7 = p1.evaluate(np.array([7.0]))
    assert np.allclose(f7, [16.0, 0.0]), f"Expected (16,0) at x=7, got {f7}"
    assert cv5 == 0.0 and cv3 == 0.0 and cv7 == 0.0
    print("  [PASS] P1 Sanity Check")


def test_p11_sanity():
    """P11 sanity check from brief Section 4."""
    p11 = mu.ProblemP11()
    # Feasible corners: (0,0), (400,0), (300,200), (0,400)
    corners = [
        np.array([0.0, 0.0]),
        np.array([400.0, 0.0]),
        np.array([300.0, 200.0]),
        np.array([0.0, 400.0])
    ]
    profits = []
    units = []
    for c in corners:
        f, cv = p11.evaluate(c)
        assert cv == 0.0, f"Corner {c} should be feasible, got cv={cv}"
        profits.append(-f[0])
        units.append(-f[1])

    # Maximum profit is 2400 at (300, 200) and (0, 400)
    assert max(profits) == 2400.0, f"Max profit should be 2400, got {max(profits)}"
    # Maximum units is 500 at (300, 200)
    assert max(units) == 500.0, f"Max units should be 500, got {max(units)}"
    # Shortfall for volume goal of 600 must be at least 100
    assert 600.0 - max(units) >= 100.0
    print("  [PASS] P11 Sanity Check")


def test_p12_sanity():
    """P12 sanity check from brief Section 4."""
    p12 = mu.ProblemP12()
    X, w, is_ben = p12.get_decision_matrix()
    assert X.shape == (5, 4), f"Expected 5x4 decision matrix, got {X.shape}"
    assert np.isclose(np.sum(w), 1.0), "Weights must sum to 1.0"
    assert np.all(is_ben == [False, True, False, True]) # Cost(min), Qual(max), Deliv(min), Rel(max)
    print("  [PASS] P12 Sanity Check")


def test_alg01_mechanism():
    """Alg 01: Equal weights (0.5, 0.5) on P1 scalarizes to (x-5)^2 + 4 -> optimal at x=5.0."""
    p1 = mu.ProblemP1()
    w = np.array([0.5, 0.5])
    x_opt, f_opt, cv_opt = a01.step(p1, w, x0=np.array([8.0]))
    assert abs(x_opt[0] - 5.0) < 1e-3, f"Expected x=5.0 for w=(0.5, 0.5), got {x_opt[0]}"
    assert np.allclose(f_opt, [4.0, 4.0], atol=1e-3)
    print("  [PASS] Alg 01 (Weighted Sum) Mechanism")


def test_alg02_mechanism():
    """Alg 02: Bounding f2 <= 4.0 on P1 yields x=5.0, satisfying the threshold."""
    p1 = mu.ProblemP1()
    eps_val = 4.0
    x_opt, f_opt, cv_opt = a02.step(p1, eps_val=eps_val, primary_obj=0, x0=np.array([8.0]))
    assert f_opt[1] <= eps_val + 1e-4, f"Constraint violated: f2={f_opt[1]} > eps={eps_val}"
    assert abs(x_opt[0] - 5.0) < 1e-3, f"Expected x=5.0, got {x_opt[0]}"
    assert abs(f_opt[0] - 4.0) < 1e-3, f"Expected f1=4.0, got {f_opt[0]}"
    print("  [PASS] Alg 02 (Epsilon-Constraint) Mechanism")


def test_alg03_mechanism():
    """Alg 03: Benson's method maximizes distance from dominated reference point (10, 10)."""
    p1 = mu.ProblemP1()
    u0 = np.array([10.0, 10.0])
    w = np.array([0.5, 0.5])
    x_opt, f_opt, cv_opt = a03.step(p1, u0, w, x0=np.array([2.0]))
    # Must reach the Pareto set [3.0, 7.0]
    assert 3.0 - 1e-3 <= x_opt[0] <= 7.0 + 1e-3, f"Solution {x_opt[0]} outside Pareto set [3, 7]"
    # Resulting objectives must strictly dominate u0
    assert f_opt[0] <= u0[0] and f_opt[1] <= u0[1]
    print("  [PASS] Alg 03 (Benson's Method) Mechanism")


def test_alg04_mechanism():
    """Alg 04: Tchebycheff metric with equal weights picks the compromise point x=5.0 on P1."""
    p1 = mu.ProblemP1()
    z_ideal = np.array([0.0, 0.0])
    w = np.array([0.5, 0.5])
    x_opt, f_opt, cv_opt = a04.step(p1, w, z_ideal, p_norm=np.inf, x0=np.array([8.0]))
    assert abs(x_opt[0] - 5.0) < 1e-3, f"Expected x=5.0 for Tchebycheff on P1, got {x_opt[0]}"
    assert np.allclose(f_opt, [4.0, 4.0], atol=1e-3)
    print("  [PASS] Alg 04 (Weight Metric / Tchebycheff) Mechanism")


def test_alg05_mechanism():
    """Alg 05: Risk-aversion parameter lambda steers trade-off between return and risk on P4."""
    p4 = mu.ProblemP4(use_repair=True)
    # Low risk-aversion lambda = 0.01 -> Aggressive investor picks high return asset 3
    res_aggr = a05.step(p4, {'utility_type': 'portfolio', 'lambda_risk': 0.01})
    # High risk-aversion lambda = 100.0 -> Conservative investor picks diversified portfolio
    res_cons = a05.step(p4, {'utility_type': 'portfolio', 'lambda_risk': 100.0})

    return_aggr = -res_aggr[1][1]
    return_cons = -res_cons[1][1]
    risk_aggr = res_aggr[1][0]
    risk_cons = res_cons[1][0]

    assert return_aggr > return_cons, f"Aggressive return {return_aggr} must exceed conservative return {return_cons}"
    assert risk_aggr > risk_cons, f"Aggressive risk {risk_aggr} must exceed conservative risk {risk_cons}"
    print("  [PASS] Alg 05 (Value Function) Mechanism")


def test_alg06_mechanism():
    """Alg 06: Goal programming on P11 leaves volume shortfall >= 100 while meeting profit goal."""
    p11 = mu.ProblemP11()
    res = a06.run(p11)
    # Target profit was 2400 (f0 target = -2400); Target volume was 600 (f1 target = -600)
    profit = -res['f_opt'][0]
    volume = -res['f_opt'][1]
    assert abs(profit - 2400.0) < 1.0, f"Expected profit 2400, got {profit}"
    # Volume cannot exceed 500, so shortfall must be >= 99.0
    volume_shortfall = 600.0 - volume
    assert volume_shortfall >= 99.0, f"Expected shortfall >= 99, got {volume_shortfall}"
    print("  [PASS] Alg 06 (Goal Programming) Mechanism")


def test_alg07_mechanism():
    """Alg 07: Lexicographic priorities break tie at max profit (2400) to maximize volume (500)."""
    p11 = mu.ProblemP11()
    res = a07.run(p11, priority_order=[0, 1])
    xA, xB = res['x_opt'][0], res['x_opt'][1]
    assert abs(xA - 300.0) < 5.0, f"Expected xA ~ 300, got {xA}"
    assert abs(xB - 200.0) < 5.0, f"Expected xB ~ 200, got {xB}"
    assert abs(res['f_opt'][0] - (-2400.0)) < 1.0
    assert abs(res['f_opt'][1] - (-500.0)) < 5.0
    print("  [PASS] Alg 07 (Lexicographic) Mechanism")


def test_alg08_mechanism():
    """Alg 08: TOPSIS computes closeness to ideal and ranks Supplier 3 first on P12."""
    p12 = mu.ProblemP12()
    X, w, is_ben = p12.get_decision_matrix()
    res = a08.run(X, w, is_ben)
    assert res['rankings'][0] == 2, f"Expected Supplier 3 (index 2) to rank #1, got {res['rankings'][0]}"
    assert np.all(res['closeness'] >= 0.0) and np.all(res['closeness'] <= 1.0)
    print("  [PASS] Alg 08 (TOPSIS) Mechanism")


def test_alg09_mechanism():
    """Alg 09: VIKOR computes compromise Q and ranks Supplier 1 first with minimum Q on P12."""
    p12 = mu.ProblemP12()
    X, w, is_ben = p12.get_decision_matrix()
    res = a09.run(X, w, is_ben, v=0.5)
    assert res['rankings'][0] == 0, f"Expected Supplier 1 (index 0) to rank #1, got {res['rankings'][0]}"
    assert abs(res['Q'][0] - min(res['Q'])) < 1e-6
    print("  [PASS] Alg 09 (VIKOR) Mechanism")


def run_all():
    print("Running Family 1 & 2 Tests (Algorithms 1-9 & Sanity Checks)...")
    test_p1_sanity()
    test_p11_sanity()
    test_p12_sanity()
    test_alg01_mechanism()
    test_alg02_mechanism()
    test_alg03_mechanism()
    test_alg04_mechanism()
    test_alg05_mechanism()
    test_alg06_mechanism()
    test_alg07_mechanism()
    test_alg08_mechanism()
    test_alg09_mechanism()
    print("All Family 1 & 2 tests passed successfully!\n")


if __name__ == "__main__":
    run_all()

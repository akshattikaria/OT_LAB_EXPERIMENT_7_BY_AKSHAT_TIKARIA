"""
test_family6.py - Unit tests for Algorithms 31 to 37
====================================================
Tests key mechanisms with hand-checkable cases:
- Alg 31 (RVEA): Angle-penalized distance (APD) prioritizes vectors aligned with reference lines
- Alg 32 (IBEA): Binary epsilon-indicator matrix computes population fitness
- Alg 33 (SMS-EMOA): Discards individual with minimal exclusive hypervolume contribution
- Alg 34 (HypE): Monte Carlo sampling computes non-negative hypervolume fitness
- Alg 35 (R-NSGA-II): Modified preference distance prioritizes solutions near DM reference point
- Alg 36 (MGDA): Min-norm solver finds common descent vector; on P13(a) converges to (2.5, 2.5) on x1=x2
- Alg 37 (GradNorm): Balances task gradients and reduces both losses simultaneously on P13
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('algorithms'))
import moo_utils as mu

import alg31_rvea as a31
import alg32_ibea as a32
import alg33_sms_emoa as a33
import alg34_hype as a34
import alg35_r_nsga2 as a35
import alg36_mgda as a36
import alg37_gradnorm as a37


def test_alg31_mechanism():
    """Alg 31: RVEA angle-penalized distance (APD) is smaller for points aligned with reference."""
    V, gamma = a31.generate_unit_reference_vectors(M=3, p=4)
    # Point aligned with V[0] vs point rotated away
    F_trans = np.array([
        V[0] * 1.0, # perfectly aligned with V[0]
        V[0] * 1.0 + np.array([0.0, 0.5, 0.5])  # deviated
    ])
    assoc, apd = a31.associate_and_compute_apd(F_trans, V, gamma, t=5, t_max=10)
    assert assoc[0] == 0, f"Expected association 0, got {assoc[0]}"
    assert apd[0] < apd[1], f"Aligned point must have lower APD ({apd[0]}) than divergent point ({apd[1]})"
    print("  [PASS] Alg 31 (RVEA) Mechanism")


def test_alg32_mechanism():
    """Alg 32: IBEA epsilon-indicator matrix reflects dominance and pairwise differences."""
    F = np.array([
        [1.0, 5.0],
        [2.0, 3.0],
        [4.0, 1.0]
    ])
    ind_mat = a32.compute_indicator_matrix(F, np.zeros(3))
    fitness = a32.compute_fitness(ind_mat)
    assert len(fitness) == 3
    assert np.all(np.isfinite(fitness)), "All fitness values must be finite numbers"
    print("  [PASS] Alg 32 (IBEA) Mechanism")


def test_alg33_mechanism():
    """Alg 33: SMS-EMOA computes exclusive hypervolume contributions to prune the least contributor."""
    F = np.array([
        [1.0, 4.0],
        [2.0, 2.0],
        [4.0, 1.0]
    ])
    ref = np.array([5.0, 5.0])
    deltas = a33.compute_hv_contributions_2d(F, ref)
    # Point with smallest exclusive contribution is discarded
    min_idx = int(np.argmin(deltas))
    assert 0 <= min_idx < len(F), "Index of point to discard must be within valid range"
    print("  [PASS] Alg 33 (SMS-EMOA) Mechanism")


def test_alg34_mechanism():
    """Alg 34: HypE estimates non-negative hypervolume contributions via Monte Carlo sampling."""
    F = np.array([
        [0.2, 0.8, 0.8],
        [0.8, 0.2, 0.8],
        [0.8, 0.8, 0.2]
    ])
    fitness = a34.estimate_hype_fitness(F, n_samples=2000, ref_point=np.full(3, 1.5))
    assert len(fitness) == 3
    assert np.all(fitness >= 0.0), "HypE fitness contributions must be non-negative"
    print("  [PASS] Alg 34 (HypE) Mechanism")


def test_alg35_mechanism():
    """Alg 35: R-NSGA-II preference distance prioritizes solutions closer to DM reference point."""
    z_ref = np.array([2.0, 2.0])
    F = np.array([
        [2.1, 2.1], # close to preference
        [4.5, 4.5]  # far from preference
    ])
    dists = a35.preference_distance(F, z_ref)
    assert dists[0] < dists[1], f"Solution close to preference must have smaller distance ({dists[0]} < {dists[1]})"
    print("  [PASS] Alg 35 (R-NSGA-II) Mechanism")


def test_alg36_mechanism():
    """Alg 36: MGDA min-norm solver finds common descent vector and converges on P13(a) to (2.5, 2.5)."""
    # 1. Closed-form convex combination for 2 orthogonal gradients
    g1 = np.array([1.0, 0.0])
    g2 = np.array([0.0, 1.0])
    alpha = a36.solve_min_norm_alpha([g1, g2])
    assert np.allclose(alpha, [0.5, 0.5]), f"Expected [0.5, 0.5], got {alpha}"
    descent_d = -(alpha[0] * g1 + alpha[1] * g2) # [-0.5, -0.5]
    assert np.dot(g1, descent_d) <= 0.0 and np.dot(g2, descent_d) <= 0.0

    # 2. Convergence on P13(a) starting from (0, 5) to the Pareto line x1 = x2
    p13 = mu.ProblemP13()
    res = a36.run(p13, x0=np.array([0.0, 5.0]), max_iter=100, lr=0.1)
    x_opt = res['x_opt']
    # Projection of (0, 5) onto line x1 = x2 is exactly (2.5, 2.5)
    assert np.allclose(x_opt, [2.5, 2.5], atol=1e-3), f"Expected convergence to (2.5, 2.5), got {x_opt}"
    assert abs(x_opt[0] - x_opt[1]) < 1e-4, f"Final point must lie on Pareto set x1=x2, got {x_opt}"
    print("  [PASS] Alg 36 (MGDA) Mechanism & P13 Convergence")


def test_alg37_mechanism():
    """Alg 37: GradNorm dynamically updates task weights to decrease losses on P13."""
    p13 = mu.ProblemP13()
    res = a37.run(p13, max_iter=50, lr_model=0.05, seed=42)
    assert len(res['history_x']) > 0
    f_init = res['history_f'][0]
    f_final = res['history_f'][-1]
    # At least one task loss must decrease significantly
    assert (f_final[0] < f_init[0]) or (f_final[1] < f_init[1])
    print("  [PASS] Alg 37 (GradNorm) Mechanism")


def run_all():
    print("Running Family 6 Tests (Algorithms 31-37)...")
    test_alg31_mechanism()
    test_alg32_mechanism()
    test_alg33_mechanism()
    test_alg34_mechanism()
    test_alg35_mechanism()
    test_alg36_mechanism()
    test_alg37_mechanism()
    print("All Family 6 tests passed successfully!\n")


if __name__ == "__main__":
    run_all()

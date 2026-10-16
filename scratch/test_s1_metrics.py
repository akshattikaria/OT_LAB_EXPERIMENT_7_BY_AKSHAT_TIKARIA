"""
test_s1_metrics.py - Unit tests for S1: compute_metrics with fixed METRIC_SPEC

Acceptance criteria:
- P3 true-front HV = 0.5433 +- 0.003
- Single point (0,1) on P3 -> HV ~0.11 and IGD in [0.3, 1.0]
- IGD(true, true) < 1e-9
- Adding a non-dominated point never lowers HV
"""
import sys, os
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import moo_utils as mu

def test_p3_true_front_hv():
    """P3 true-front HV should be 0.5433 +- 0.003."""
    p3 = mu.ProblemP3()
    tf = p3.true_pareto_front(500)
    hv, igd_val, sp = mu.compute_metrics(tf, 'P3')
    print(f"  P3 true-front: HV={hv:.5f}, IGD={igd_val:.2e}, spacing={sp:.5f}")
    assert hv is not None, "HV must not be None for P3"
    assert abs(hv - 0.5433) <= 0.003, f"P3 HV={hv:.5f} not in [0.5403, 0.5463]"
    assert igd_val is not None, "IGD must not be None for P3 (true front known)"
    assert igd_val < 1e-9, f"IGD(true,true) = {igd_val:.2e} not < 1e-9"
    print("  PASS: P3 true-front HV and IGD(true,true)")

def test_single_point_p3():
    """Single point (0,1) on P3 -> HV ~0.11 and finite IGD in [0.3, 1.0]."""
    pt = np.array([[0.0, 1.0]])
    hv, igd_val, sp = mu.compute_metrics(pt, 'P3')
    print(f"  Single point (0,1): HV={hv:.5f}, IGD={igd_val:.4f}")
    assert hv is not None, "HV must not be None"
    assert abs(hv - 0.11) <= 0.03, f"Single-point HV={hv:.5f} expected ~0.11"
    assert igd_val is not None and np.isfinite(igd_val), f"IGD must be finite, got {igd_val}"
    assert 0.3 <= igd_val <= 1.0, f"IGD={igd_val:.4f} not in [0.3, 1.0]"
    print("  PASS: single point (0,1) on P3")

def test_adding_nondominated_point_never_lowers_hv():
    """Adding a non-dominated point must never lower HV."""
    p3 = mu.ProblemP3()
    # A moderate front
    F1 = np.array([[0.2, 0.96], [0.5, 0.75], [0.8, 0.36]])
    hv1, _, _ = mu.compute_metrics(F1, 'P3')
    # Add a non-dominated point
    F2 = np.vstack([F1, [0.1, 0.99]])
    # (0.1, 0.99) dominates nothing in F1, is non-dominated -> HV >= hv1
    hv2, _, _ = mu.compute_metrics(F2, 'P3')
    print(f"  HV before={hv1:.5f}, after adding non-dom point={hv2:.5f}")
    assert hv2 >= hv1 - 1e-9, f"HV decreased: {hv1:.5f} -> {hv2:.5f}"
    print("  PASS: adding non-dominated point never lowers HV")

def test_igd_true_true_near_zero():
    """IGD(true_front_500, true_front_500) < 1e-9 (same 500 pts used in METRIC_SPEC)."""
    p3 = mu.ProblemP3()
    tf500 = p3.true_pareto_front(500)
    mu._build_metric_spec('P3_ZDT2')
    hv, igd_val, sp = mu.compute_metrics(tf500, 'P3')
    print(f"  IGD(true_500,true_500) = {igd_val:.2e}")
    assert igd_val is not None, "IGD must not be None"
    assert igd_val < 1e-9, f"IGD(true_500,true_500) = {igd_val:.2e} not < 1e-9"
    tf100 = p3.true_pareto_front(100)
    _, igd100, _ = mu.compute_metrics(tf100, 'P3')
    print(f"  IGD(true_100,spec_500) = {igd100:.4f}  (small positive ok)")
    assert igd100 < 0.05, f"100-pt subset IGD too large: {igd100}"
    print("  PASS: IGD(true_500,true_500) < 1e-9")

def test_p10_hv_nonnegative():
    """P10 (DTLZ2) compute_metrics should return a finite non-negative HV."""
    p10 = mu.ProblemP10(M=3)
    tf = p10.true_pareto_front(500)
    hv, igd_val, sp = mu.compute_metrics(tf, 'P10')
    print(f"  P10 M=3 true-front 500pt: HV={hv:.5f}, IGD={igd_val:.2e}")
    assert hv is not None and hv >= 0, f"P10 HV should be non-negative, got {hv}"
    assert igd_val is not None and igd_val < 1e-9, f"P10 IGD(true,true)={igd_val:.2e}"
    print("  PASS: P10 HV finite and non-negative, IGD(true,true)<1e-9")

def test_no_hv_corruption_single_point():
    """A single-point front must NOT give HV=1.21 (old bug)."""
    pt = np.array([[0.5, 0.75]])
    hv, _, _ = mu.compute_metrics(pt, 'P3')
    print(f"  Single point [0.5,0.75] HV={hv:.5f}")
    assert hv is not None
    assert hv < 0.5, f"Single-point HV={hv:.5f} should be much less than 0.5 (old bug was 1.21)"
    print("  PASS: single-point HV is NOT 1.21 (old bug gone)")

if __name__ == '__main__':
    print("=" * 60)
    print("S1 METRICS UNIT TESTS")
    print("=" * 60)
    tests = [
        test_p3_true_front_hv,
        test_single_point_p3,
        test_adding_nondominated_point_never_lowers_hv,
        test_igd_true_true_near_zero,
        test_p10_hv_nonnegative,
        test_no_hv_corruption_single_point,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            print(f"\n[TEST] {t.__name__}")
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            failed += 1
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL S1 TESTS PASSED")
    else:
        print("SOME S1 TESTS FAILED - check above")
    print("=" * 60)

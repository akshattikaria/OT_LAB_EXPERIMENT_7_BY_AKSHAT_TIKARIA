"""
run_all_tests.py - Executes the full unit test suite for Lab Assignment 7
==========================================================================
Runs all tests across the shared MOO toolkit, problem bank sanity checks (P1-P14),
and hand-checkable key mechanism tests for all 37 algorithms.
Prints a formatted summary table.
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('algorithms'))
sys.path.insert(0, os.path.abspath('tests'))

import test_moo_utils
import test_family1_2
import test_family3_4
import test_family5
import test_family6


def run_test_module(name: str, functions: list):
    """Executes a list of test functions and collects pass/fail counts."""
    passed = 0
    failed = 0
    errors = []

    print(f"\n>> Running {name}...")
    start_t = time.time()
    for fn in functions:
        fn_name = fn.__name__
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((fn_name, str(e)))
            print(f"  [FAIL] {fn_name}: {e}")

    elapsed = time.time() - start_t
    return {
        'module': name,
        'total': len(functions),
        'passed': passed,
        'failed': failed,
        'time': elapsed,
        'errors': errors
    }


def main():
    print("=" * 80)
    print("M.TECH OPTIMIZATION TECHNIQUES - LAB ASSIGNMENT 7")
    print("FULL UNIT TEST SUITE: 37 ALGORITHMS, SHARED TOOLKIT & PROBLEM BANK")
    print("=" * 80)

    modules_to_test = [
        ("Shared Toolkit & Problem Bank (P1-P14)", [
            test_moo_utils.test_dominance,
            test_moo_utils.test_deb_constrained_dominance,
            test_moo_utils.test_fast_non_dominated_sort,
            test_moo_utils.test_crowding_distance,
            test_moo_utils.test_hypervolume_2d,
            test_moo_utils.test_spacing,
            test_moo_utils.test_problem_sanity_checks,
            test_moo_utils.test_p3_true_front_hv,
            test_moo_utils.test_single_point_p3,
            test_moo_utils.test_igd_true_true,
            test_moo_utils.test_adding_nondominated_point_never_lowers_hv
        ]),
        ("Family 1 & 2: Classical, Preference & Ranking (Alg 01-09 + Sanity)", [
            test_family1_2.test_p1_sanity,
            test_family1_2.test_p11_sanity,
            test_family1_2.test_p12_sanity,
            test_family1_2.test_alg01_mechanism,
            test_family1_2.test_alg02_mechanism,
            test_family1_2.test_alg03_mechanism,
            test_family1_2.test_alg04_mechanism,
            test_family1_2.test_alg05_mechanism,
            test_family1_2.test_alg06_mechanism,
            test_family1_2.test_alg07_mechanism,
            test_family1_2.test_alg08_mechanism,
            test_family1_2.test_alg09_mechanism
        ]),
        ("Family 3 & 4: Early EAs, Elitist & Decomposition (Alg 10-20)", [
            test_family3_4.test_alg10_mechanism,
            test_family3_4.test_alg11_mechanism,
            test_family3_4.test_alg12_mechanism,
            test_family3_4.test_alg13_mechanism,
            test_family3_4.test_alg14_mechanism,
            test_family3_4.test_alg15_mechanism,
            test_family3_4.test_alg16_mechanism,
            test_family3_4.test_alg17_mechanism,
            test_family3_4.test_alg18_mechanism,
            test_family3_4.test_alg19_mechanism,
            test_family3_4.test_alg20_mechanism
        ]),
        ("Family 5: Constrained, Swarm, Physics, Bio & Hybrid (Alg 21-30)", [
            test_family5.test_alg21_mechanism,
            test_family5.test_alg22_mechanism,
            test_family5.test_alg23_mechanism,
            test_family5.test_alg24_mechanism,
            test_family5.test_alg25_mechanism,
            test_family5.test_alg26_mechanism,
            test_family5.test_alg27_mechanism,
            test_family5.test_alg28_mechanism,
            test_family5.test_alg29_mechanism,
            test_family5.test_alg30_mechanism
        ]),
        ("Family 6: Many-Obj, Indicator, Preference & Gradient (Alg 31-37)", [
            test_family6.test_alg31_mechanism,
            test_family6.test_alg32_mechanism,
            test_family6.test_alg33_mechanism,
            test_family6.test_alg34_mechanism,
            test_family6.test_alg35_mechanism,
            test_family6.test_alg36_mechanism,
            test_family6.test_alg37_mechanism
        ])
    ]

    results = []
    total_passed = 0
    total_failed = 0
    total_tests = 0
    start_total = time.time()

    for mod_name, fns in modules_to_test:
        res = run_test_module(mod_name, fns)
        results.append(res)
        total_passed += res['passed']
        total_failed += res['failed']
        total_tests += res['total']

    total_time = time.time() - start_total

    print("\n" + "=" * 80)
    print("UNIT TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Module Name':<50} {'Total':<8} {'Passed':<8} {'Failed':<8} {'Status'}")
    print("-" * 80)

    for r in results:
        status = "PASS" if r['failed'] == 0 else "FAIL"
        print(f"{r['module']:<50} {r['total']:<8} {r['passed']:<8} {r['failed']:<8} {status}")

    print("-" * 80)
    final_verdict = "ALL TESTS PASSED SUCCESSFULLY" if total_failed == 0 else "SOME TESTS FAILED"
    print(f"{'TOTAL':<50} {total_tests:<8} {total_passed:<8} {total_failed:<8} {final_verdict}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print("=" * 80)

    if total_failed > 0:
        print("\nFailures:")
        for r in results:
            for fn_name, err in r['errors']:
                print(f"  - [{r['module']}] {fn_name}: {err}")
        sys.exit(1)
    else:
        print("\nAll 37 algorithms and toolkit verification tests verified complete and passing!")
        sys.exit(0)


if __name__ == "__main__":
    main()

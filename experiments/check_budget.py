"""
experiments/check_budget.py - Evaluation count auditor for Lab 7 MOO algorithms
=============================================================================
Runs each (algorithm, problem) pair and verifies:
1. Stochastic algorithms (10 to 35) use counted evaluations within [9000, 10100].
2. Unbudgeted algorithms (1 to 9, 36, 37) log their exact counted evaluations.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('algorithms'))

import moo_utils as mu
from experiments.budget import run_budgeted
from experiments.run_task_a import CATALOGUE_PROBLEMS, get_problem_instance


def main():
    print("=" * 78)
    print("CHECK BUDGET: Verifying counted evaluations across algorithms and problems")
    print("Target budget for stochastic algorithms (10-35): [9000, 10100]")
    print("=" * 78)

    all_passed = True
    results = []

    for alg_num in sorted(CATALOGUE_PROBLEMS.keys()):
        name, module, p_names = CATALOGUE_PROBLEMS[alg_num]

        for p_name in p_names:
            prob = get_problem_instance(p_name)
            t0 = time.time()
            try:
                res = run_budgeted(alg_num, module, prob, seed=42)
                elapsed = time.time() - t0
                evals = res.get('evaluations', 0)
                is_stochastic = (10 <= alg_num <= 35)

                if is_stochastic:
                    in_range = (9000 <= evals <= 10100)
                    status_str = "OK" if in_range else "OUT OF RANGE"
                    if not in_range:
                        all_passed = False
                else:
                    in_range = True
                    status_str = "UNBUDGETED (LOGGED)"

                cal_iter = res.get('calibrated_iter', 'N/A')
                print(f"Alg {alg_num:02d} | {name:<20} | {p_name:<6} | Evals: {evals:5d} | "
                      f"Iter: {str(cal_iter):<4} | Time: {elapsed:5.2f}s | {status_str}")

                results.append({
                    'alg_num': alg_num,
                    'name': name,
                    'problem': p_name,
                    'evals': evals,
                    'in_range': in_range,
                    'status': status_str
                })

            except Exception as e:
                elapsed = time.time() - t0
                all_passed = False
                print(f"Alg {alg_num:02d} | {name:<20} | {p_name:<6} | ERROR: {e} ({elapsed:5.2f}s)")
                results.append({
                    'alg_num': alg_num,
                    'name': name,
                    'problem': p_name,
                    'evals': 0,
                    'in_range': False,
                    'status': f"ERROR: {e}"
                })

    print("=" * 78)
    failed = [r for r in results if not r['in_range']]
    if failed:
        print(f"FAILED: {len(failed)} test(s) failed evaluation range check:")
        for f in failed:
            print(f"  - Alg {f['alg_num']} ({f['name']}) on {f['problem']}: evals={f['evals']}, status={f['status']}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED: All stochastic algorithms are within [9000, 10100] evaluations!")
        print("All unbudgeted algorithms logged their real counted evaluations.")
        sys.exit(0)


if __name__ == '__main__':
    main()

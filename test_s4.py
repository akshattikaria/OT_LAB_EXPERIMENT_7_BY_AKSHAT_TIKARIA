import sys
sys.path.append('.')
sys.path.append('algorithms')
import numpy as np
from experiments.status import classify_status
from experiments.run_task_a import CATALOGUE_PROBLEMS, run_single_algorithm

print("=" * 70)
print("S4 ACCEPTANCE CHECK: experiments/status.py quantitative validation")
print("=" * 70)

# 1. Unit test synthetic cases
# 1a. Empty front
s_empty = classify_status(14, 'P3', {'F_front': np.array([])})
print(f"Empty front -> '{s_empty}'")
assert 'not done' in s_empty

# 1b. Few distinct points (<10)
few_pts = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]])
s_few = classify_status(14, 'P3', {'F_front': few_pts}, igd_val=0.01)
print(f"Few points (3 < 10) -> '{s_few}'")
assert 'partly works' in s_few and '3 distinct points' in s_few

# 1c. High IGD (>0.05)
good_pts = np.column_stack([np.linspace(0, 1, 25), np.linspace(1, 0, 25)])
s_high_igd = classify_status(14, 'P3', {'F_front': good_pts}, igd_val=0.12)
print(f"High IGD (0.12 > 0.05) -> '{s_high_igd}'")
assert 'partly works' in s_high_igd and 'IGD=0.1200' in s_high_igd

# 1d. Excellent front (25 points, IGD=0.005)
s_good = classify_status(14, 'P3', {'F_front': good_pts}, igd_val=0.005)
print(f"Good front (25 pts, IGD=0.005) -> '{s_good}'")
assert s_good == 'works'

# 1e. Weighted Sum on non-convex P3
s_ws_p3 = classify_status(1, 'P3', {'F_front': good_pts}, igd_val=0.01)
print(f"Weighted Sum on P3 -> '{s_ws_p3}'")
assert 'partly works' in s_ws_p3 and 'non-convex' in s_ws_p3

# 1f. VEGA on non-convex P3
s_vega_p3 = classify_status(10, 'P3', {'F_front': good_pts}, igd_val=0.01)
print(f"VEGA on P3 -> '{s_vega_p3}'")
assert 'partly works' in s_vega_p3 and 'speciation' in s_vega_p3

# 1g. Single solution sanity check P11 (Lexicographic)
s_lex_p11_good = classify_status(7, 'P11', {'f_opt': np.array([-2400.0, -500.0]), 'cv_opt': 0.0})
print(f"Lexicographic on P11 (2400, 500) -> '{s_lex_p11_good}'")
assert s_lex_p11_good == 'works'

s_lex_p11_bad = classify_status(7, 'P11', {'f_opt': np.array([-1800.0, -300.0]), 'cv_opt': 0.0})
print(f"Lexicographic on P11 (1800, 300) -> '{s_lex_p11_bad}'")
assert 'partly works' in s_lex_p11_bad and 'profit=1800.0' in s_lex_p11_bad

# 2. Live test via run_single_algorithm
rec1_p1 = run_single_algorithm(1, "Weighted Sum", CATALOGUE_PROBLEMS[1][1], "P1", seed=42)
print(f"Live Alg 01 on P1 -> Status: '{rec1_p1['status']}'")
assert rec1_p1['status'] == 'works'

rec1_p3 = run_single_algorithm(1, "Weighted Sum", CATALOGUE_PROBLEMS[1][1], "P3", seed=42)
print(f"Live Alg 01 on P3 -> Status: '{rec1_p3['status']}'")
assert 'partly works' in rec1_p3['status']

rec7_p11 = run_single_algorithm(7, "Lexicographic", CATALOGUE_PROBLEMS[7][1], "P11", seed=42)
print(f"Live Alg 07 on P11 -> Status: '{rec7_p11['status']}'")
assert rec7_p11['status'] == 'works'

rec14_p3 = run_single_algorithm(14, "NSGA-II", CATALOGUE_PROBLEMS[14][1], "P3", seed=42)
print(f"Live Alg 14 on P3 -> Status: '{rec14_p3['status']}'")
assert rec14_p3['status'] == 'works'

print("\nS4 ACCEPTANCE PASSED!")

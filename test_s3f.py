import sys, time
sys.path.append('algorithms')
import moo_utils as mu
import alg17_pesa2 as a17
import numpy as np
from experiments.budget import run_budgeted

p3 = mu.ProblemP3()
t0 = time.time()
res_p3 = run_budgeted(17, a17, p3, seed=42, budget=10000)
dt = time.time() - t0

f_p3 = res_p3['F_front']
unique_p3 = len(np.unique(np.round(f_p3, 4), axis=0))
hv_p3, igd_p3, sp_p3 = mu.compute_metrics(f_p3, 'P3')

evals = res_p3.get('counted_evals', res_p3.get('evaluations'))
print(f"PESA-II on P3 at 10k evals:")
print(f"Counted evaluations: {evals}")
print(f"Unique points: {unique_p3} (Target: >=50)")
print(f"HV: {hv_p3:.5f} (Target: >=0.50)")
print(f"IGD: {igd_p3:.5f}")
print(f"Spacing: {sp_p3:.5f}")
print(f"Time: {dt:.2f}s")

assert evals >= 9000 and evals <= 10100, f"Evals out of range: {evals}"
assert unique_p3 >= 50, f"Unique points {unique_p3} < 50"
assert hv_p3 >= 0.50, f"HV {hv_p3} < 0.50"
print("S3f ACCEPTANCE PASSED!")

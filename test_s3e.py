import sys, time
sys.path.append('algorithms')
import moo_utils as mu
import alg29_moba as a29
import numpy as np
from experiments.budget import run_budgeted

# Check P3
p3 = mu.ProblemP3()
t0 = time.time()
res_p3 = run_budgeted(29, a29, p3, seed=42, budget=10000)
dt_p3 = time.time() - t0

f_p3 = res_p3['F_front']
unique_p3 = len(np.unique(np.round(f_p3, 4), axis=0))
hv_p3, igd_p3, sp_p3 = mu.compute_metrics(f_p3, 'P3')

print(f"P3: counted_evals={res_p3.get('counted_evals', res_p3.get('evaluations'))}, unique_points={unique_p3} (Target >=30), HV={hv_p3:.5f} (Target >=0.45), time={dt_p3:.2f}s (Target <=30s)")

# Check P4
p4 = mu.ProblemP4()
t0 = time.time()
res_p4 = run_budgeted(29, a29, p4, seed=42, budget=10000)
dt_p4 = time.time() - t0
f_p4 = res_p4['F_front']
hv_p4, igd_p4, sp_p4 = mu.compute_metrics(f_p4, 'P4')
print(f"P4: counted_evals={res_p4.get('counted_evals', res_p4.get('evaluations'))}, front_size={len(f_p4)}, HV={hv_p4:.5f}, time={dt_p4:.2f}s (Target <=60s)")

assert unique_p3 >= 30, f"Unique points {unique_p3} < 30"
assert hv_p3 >= 0.45, f"HV {hv_p3} < 0.45"
assert dt_p3 <= 30.0, f"Time {dt_p3} > 30s"
assert dt_p4 <= 60.0, f"Time {dt_p4} > 60s"
print("S3e ACCEPTANCE PASSED!")

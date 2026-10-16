import sys, time
sys.path.append('algorithms')
import moo_utils as mu
import alg19_moead_de as a19
import numpy as np
from experiments.budget import run_budgeted

p5 = mu.ProblemP5()
t0 = time.time()
res_p5 = run_budgeted(19, a19, p5, seed=42, budget=10000)
dt = time.time() - t0

f_p5 = res_p5['F_front']
cv_p5 = res_p5.get('CV_front', np.zeros(len(f_p5)))
# Check unique points
unique_pts = np.unique(np.round(f_p5, 4), axis=0)
n_unique = len(unique_pts)
evals = res_p5.get('counted_evals', res_p5.get('evaluations'))

print(f"MOEA/D-DE on P5 at 10k evals:")
print(f"Counted evaluations: {evals}")
print(f"Front size: {len(f_p5)}")
print(f"Unique points: {n_unique} (Target: >=15)")
print(f"Time: {dt:.2f}s")

if n_unique >= 15:
    print("S3h ACCEPTANCE PASSED (>=15 unique points)!")
else:
    print(f"S3h FAILED: only {n_unique} unique points < 15.")

import sys, time
sys.path.append('algorithms')
import moo_utils as mu
import alg30_moma as a30
import numpy as np
from experiments.budget import run_budgeted

p9 = mu.ProblemP9()

# 1. Test evaluation speed
times = []
for _ in range(100):
    mask = (np.random.rand(30) > 0.5).astype(float)
    p9._cache.clear()
    t0 = time.perf_counter()
    p9.evaluate(mask)
    times.append(time.perf_counter() - t0)

eval_ms = np.mean(times) * 1000
median_ms = np.median(times) * 1000
print(f"ProblemP9 uncached evaluation time: mean={eval_ms:.3f} ms, median={median_ms:.3f} ms (Target: <=5 ms)")

# 2. Test MOMA on P9
t0 = time.time()
res_p9 = run_budgeted(30, a30, p9, seed=42, budget=10000)
dt_moma = time.time() - t0
f_p9 = res_p9['F_front']
evals = res_p9.get('counted_evals', res_p9.get('evaluations'))

print(f"MOMA on P9 at 10k evals:")
print(f"Counted evaluations: {evals}")
print(f"Front size: {len(f_p9)}")
print(f"Time: {dt_moma:.2f} s (Target: <=120 s)")

assert median_ms <= 5.0 or eval_ms <= 5.0, f"Evaluation time too high: mean={eval_ms:.2f}ms, median={median_ms:.2f}ms"
assert dt_moma <= 120.0, f"MOMA on P9 took too long: {dt_moma:.2f}s"
print("S3g ACCEPTANCE PASSED!")

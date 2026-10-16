import sys
sys.path.append('.')
sys.path.append('algorithms')
from experiments.run_task_a import CATALOGUE_PROBLEMS, run_single_algorithm
import numpy as np

target_algs = [5, 6, 7, 8, 9, 36, 37]

print("=" * 70)
print("S3i ACCEPTANCE CHECK: Single-solution & ranking algorithms metrics")
print("=" * 70)

for alg_num in target_algs:
    name, module, prob_list = CATALOGUE_PROBLEMS[alg_num]
    for p_name in prob_list:
        rec = run_single_algorithm(alg_num, name, module, p_name, seed=42)
        hv = rec['hv']
        igd = rec['igd']
        sp = rec['spacing']
        pref = rec.get('preference_input', 'N/A')
        print(f"Alg {alg_num:02d} ({name}) on {p_name}: HV={hv}, IGD={igd}, Spacing={sp}, Pref='{pref}'")
        assert np.isnan(hv), f"Alg {alg_num} HV should be NaN/NA, got {hv}"
        assert np.isnan(igd), f"Alg {alg_num} IGD should be NaN/NA, got {igd}"
        assert np.isnan(sp), f"Alg {alg_num} Spacing should be NaN/NA, got {sp}"
        assert pref != 'N/A' and len(pref) > 0, f"Alg {alg_num} preference input missing"

# Check that standard algorithm (e.g. Alg 14 on P3) still computes numerical metrics
rec14 = run_single_algorithm(14, "NSGA-II", CATALOGUE_PROBLEMS[14][1], "P3", seed=42)
print(f"Standard Alg 14 on P3: HV={rec14['hv']:.4f}, IGD={rec14['igd']:.4f}, Spacing={rec14['spacing']:.4f}")
assert not np.isnan(rec14['hv']), "Standard alg should have numerical HV"

print("\nS3i ACCEPTANCE PASSED!")

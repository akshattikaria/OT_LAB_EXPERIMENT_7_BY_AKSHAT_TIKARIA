# B.Tech Optimization Techniques — Lab Assignment 7
## Final Submission Cover Sheet

---

### Student Information
- **Student Name:** Akshat Tikaria
- **Roll Number:** 241070005
- **Assigned Roll Number Index ($R$):** $R = 05$
- **Academic Program:** B.Tech AI & Data Science / Computer Science & Engineering
- **Course:** Optimization Techniques Laboratory (Lab Assignment 7)
- **Topic:** Multi-Objective Optimization: 37 Algorithms, Trade-offs and Applications

---

### Deep-Dive Algorithms Assigned (Based on $R = 05$)
1. **Algorithm 5:** Value Function / Utility Method (Classical Scalarization)
2. **Algorithm 17:** PESA-II (Pareto Envelope-based Selection Algorithm II — Hyperbox Grid Elitism)
3. **Algorithm 29:** MOBA (Multi-Objective Bat Algorithm — Bio-Inspired Echolocation Metaheuristic)

---

### Submission Deliverables (Google Drive Archive)
- **Google Drive Link:** [Insert Your Google Drive Link Here]
  *(Ensure link access is set to "Anyone with the link can view / download")*

#### Contents of the Google Drive Folder:
1. `code.zip`: Complete Python codebase containing:
   - `moo_utils.py` (Problem Bank P1–P14 and core MOO utilities)
   - `algorithms/` (Implementations of all 37 MOO algorithms, `alg01` to `alg37`)
   - `experiments/` (`run_task_a.py`, `run_task_c.py`, `run_task_d.py`)
   - `tests/` (`test_family1_2.py`, `test_family3_4.py`, `test_family5.py`, `test_family6.py`)
   - `run_all_tests.py` (Master test runner executing all 47 unit tests)
   - `seeds.txt` (10 fixed independent evaluation random seeds)
   - `README.md` (Installation, environment setup, and reproduction guide)
2. `raw_results.zip`: Complete raw benchmark execution artifacts:
   - `results/data/` (All 8 raw CSV logs: `task_a_log.csv`, `task_c_sensitivity.csv`, `task_d1_p3_ranked.csv`, `task_d1_p5_ranked.csv`, `task_d2_p10_m5_ranked.csv`, `task_d3_p8_tsp_ranked.csv`, `task_d4_mcdm.csv`, `task_d5_mtl.csv`)
   - `results/figures/` (All 73 Pareto front comparison plots across all 37 algorithms and Task D studies)
3. `report.pdf`: Complete compiled master laboratory report combining Tasks A through F, the 14-page Problem Bank specifications, Master Comparison Table, Deep Dive investigations, Empirical Benchmarking, Algorithm Selection, and Viva Walkthrough.

---

### Integrity Statement & Hard Rules Compliance
- **Allowed Libraries:** Pure Python standard library, NumPy, and Matplotlib. Zero `pandas` utilized anywhere across the codebase.
- **Scipy Usage:** `scipy.optimize` strictly restricted to single-objective subproblems in Algorithms 1–7.
- **Scikit-learn Usage:** `scikit-learn` strictly restricted to loading the raw Wisconsin Breast Cancer dataset for Problem P9; the 5-Nearest Neighbors classifier and 5-fold cross-validation engine are entirely hand-coded.
- **Data Veracity:** 100% of reported numerical tables, hypervolume values, runtimes, and figures are derived from real execution outputs in `results/data/`.

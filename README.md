# M.Tech Optimization Techniques — Lab Assignment 7: Multi-Objective Optimization

**Course:** Optimization Techniques Laboratory  
**Topic:** Multi-Objective Optimization: 37 Algorithms, Trade-offs and Applications  
**Student Name:** Akshat Tikaria  
**Student Roll Number:** 241070005 ($R = 05$)  
**Assigned Deep-Dive Algorithms:**
- Algorithm 5: Value Function / Utility Method
- Algorithm 17: PESA-II (Pareto Envelope-based Selection Algorithm II)
- Algorithm 29: MOBA (Multi-Objective Bat Algorithm)

---

## 1. Environment & Dependencies

This codebase complies strictly with the assignment brief's **Hard Rules (Section 2A)**:
- **Allowed Libraries:** Pure Python 3.9+, NumPy, and Matplotlib.
- **Strictly No Pandas:** Data processing and CSV serialization are implemented exclusively using Python's standard `csv` module.
- **Scipy Restrictions:** `scipy.optimize` is utilized **only** inside Algorithms 1–7 to solve the single-objective scalar subproblems. All scalarization logic, $\varepsilon$-sweeps, deviation goals, and priority sorting are custom implementations.
- **Scikit-learn Restrictions:** `scikit-learn` is utilized **only** to fetch the Wisconsin Breast Cancer dataset (`ProblemP9`). The 5-Nearest Neighbor classifier ($k=5$) and 5-fold cross-validation engine are entirely hand-coded in pure NumPy.

### Installation
Ensure Python 3.9+ is installed, then install the minimal dependencies:
```bash
pip install numpy matplotlib scipy scikit-learn
```

---

## 2. Directory Structure

```text
├── algorithms/                 # Implementations of all 37 MOO algorithms
│   ├── alg01_weighted_sum.py
│   ├── ...
│   └── alg37_gradnorm.py
├── experiments/                # Benchmark execution scripts
│   ├── run_task_a.py           # Master evaluation across all 37 algorithms (65 instances)
│   ├── run_task_c.py           # Deep dive sensitivity & 2-iteration hand-traces
│   └── run_task_d.py           # 5 comparative benchmark studies (P3/P5, P10 M=5, P8 TSP, MCDM, MTL)
├── tests/                      # Comprehensive unit tests with mechanism verifications
│   ├── test_family1_2.py       # Alg 01-09 + P1, P11, P12 sanity checks
│   ├── test_family3_4.py       # Alg 10-20
│   ├── test_family5.py         # Alg 21-30
│   └── test_family6.py         # Alg 31-37 + MGDA P13 convergence
├── report/                     # Master markdown report documents
│   ├── task_a_log.md           # Implementation log and test suite summary
│   ├── problem_bank.md         # 14 one-page problem formulations (P1-P14)
│   ├── task_b_master_table.md  # 37-algorithm comparison table with real run evidence
│   ├── task_c_deep_dive.md     # Deep dive traces and sensitivity studies (Alg 5, 17, 29)
│   ├── task_d_comparisons.md   # Empirical comparisons (P3/P5, P10 M=5, P8 TSP, MCDM, MTL)
│   ├── task_e_selection.md     # 6 problem selection rationales backed by run results
│   └── task_f_viva_answers.md  # Detailed answers to the 8 core viva questions
├── results/
│   ├── data/                   # Raw CSV execution logs
│   └── figures/                # 73 Pareto front comparison plots
├── moo_utils.py                # Problem Bank (P1-P14) & MOO evaluation toolkit
├── run_all_tests.py            # Master unit test runner
├── seeds.txt                   # 10 fixed random seeds for reproducibility
├── submission_cover_template.md# Final submission cover sheet
└── README.md                   # This instruction file
```

---

## 3. Running Unit Tests

To run the complete test suite and verify all 37 algorithms and problem sanity checks:
```bash
python run_all_tests.py
```
**Expected Output:** All 47 tests pass in under 5 seconds with zero failures.

---

## 4. Running Benchmark Experiments

All experimental runs use the 10 fixed random seeds recorded in `seeds.txt` (`42, 107, 219, 334, 451, 568, 682, 793, 805, 917`):

### Task A: Full Implementation Log (All 37 Algorithms)
```bash
python experiments/run_task_a.py
```
- Evaluates all 37 algorithms across 65 problem instances.
- Outputs `results/data/task_a_log.csv`.
- Generates Pareto front plots in `results/figures/` (e.g. `front_alg01_P1.png`, ..., `front_alg37_P13.png`).

### Task C: Deep-Dive Investigations (Algorithms 5, 17, 29)
```bash
python experiments/run_task_c.py
```
- Executes 2-iteration hand-trace validations for Algorithms 5, 17, and 29.
- Conducts 3-level sensitivity studies (10 seeds each).
- Outputs `results/data/task_c_sensitivity.csv` and sensitivity plots.

### Task D: Comparative Benchmarking
```bash
python experiments/run_task_d.py
```
Executes all 5 comparative benchmark studies:
1. 14 continuous algorithms on P3 and P5 (`task_d1_p3_ranked.csv`, `task_d1_p5_ranked.csv`).
2. 5-objective DTLZ2 ($M=5$) comparing NSGA-II, NSGA-III, RVEA, MOEA/D, and HypE (`task_d2_p10_m5_ranked.csv`).
3. Discrete permutation TSP on P8 (`task_d3_p8_tsp_ranked.csv`).
4. TOPSIS vs VIKOR MCDM analysis on P4 front (`task_d4_mcdm.csv`).
5. Multi-task gradient optimization on P13 (`task_d5_mtl.csv`).

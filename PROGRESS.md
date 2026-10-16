# PROGRESS — OT Lab Assignment 7

Student: Akshat Tikaria, Roll 241070005

Safety rules active: only edit inside project folder; move files to _old/ before deleting; no pip install.

---

## Checklist

- [x] **S1** METRICS — add METRIC_SPEC + compute_metrics to moo_utils.py; unit tests
- [x] **S2** BUDGET — CountingProblem, experiments/budget.py, experiments/check_budget.py
- [x] **S3a** alg02 epsilon-constraint payoff table fix
- [x] **S3b** alg03 Benson reference point fix
- [x] **S3c** alg14 NSGA-II use_constraints=False
- [x] **S3d** alg28 MO-TLBO teacher selection fix
- [x] **S3e** alg29 MOBA 5 bug fixes
- [x] **S3f** alg17 PESA-II docstring note (core unchanged)
- [x] **S3g** alg30 MOMA + P9 precompute optimization
- [x] **S3h** alg19 MOEA/D-DE on P5 (fix only if still fails after S2)
- [x] **S3i** algs 5,6,7,36,37,8,9 log N/A metrics
- [x] **S4** STATUS — experiments/status.py; replace hard-coded status in run_task_a.py
- [x] **S5a** p13_network.py (MTL network with forward/backward/grad-check)
- [x] **S5b** P8 coordinates saved to results/data/p8_coords.csv + problem_bank.md
- [x] **S6** HAND TRACES — ScriptedRNG, full MOBA+PESA-II 2-iteration trace with ASSERTs
- [x] **S7** EXPERIMENTS — full re-run, multiprocessing, smoke test first
- [x] **S8** REPORT — build_claims.py, rebuild all .md from CSVs, fix B.Tech→B.Tech
- [x] **S9** PACKAGING — new code.zip, raw_results.zip, report.pdf
- [x] **S10** FINAL AUDIT — audit_final.py
- [x] **S12** FINISH — summary paragraph

---

## Log

| Step | Status | Result |
|------|--------|--------|
| INIT | ✅ DONE | MASTER_PLAN.md created, PROGRESS.md created, safety rules acknowledged |
| S1 | ✅ DONE | All unit tests in test_moo_utils.py passed (P3 HV=0.54233, (0,1) HV=0.11000, IGD=0, non-dom HV non-decreasing). 0 normalize_objectives calls in experiments/. ref_union_P8.csv generated. |
| S2 | ✅ DONE | CountingProblem + experiments/budget.py + experiments/check_budget.py passed 100% (all 26 stochastic algorithms in [9000, 10100], unbudgeted algorithms accurately logged). |
| S3a | ✅ DONE | Payoff table + lexicographic f1-minimizer fixed alg02; P3 25/25 points within 0.02 of f2=1-f1^2; P1 and P2 25 distinct points. |
| S3b | ✅ DONE | Dominated reference points spread along front fixed alg03; P1 14 distinct (100% f1 coverage); P7 25 distinct (100% f1 coverage). |
| S3c | ✅ DONE | Added use_constraints=False to alg14; verified on P5 alg14 returned 50/100 CV>0 points while alg21 returned 0/100. |
| S3d | ✅ DONE | Fixed MO-TLBO teacher selection; P3 at 10k evals gave 99 unique points (target >=20) and HV=0.53738 (target >=0.45). |
| S3e | ✅ DONE | MOBA 5 bug fixes passed: P3 unique=100 (>=30), HV=0.50702 (>=0.45), time=16.54s (<=30s); P4 HV=0.96059, time=23.90s (<=60s). |
| S3f | ✅ DONE | Added PESA-II docstring note; P3 at 10k evals verified: unique=97 (>=50), HV=0.52432 (>=0.50), IGD=0.01129, time=13.47s. |
| S3g | ✅ DONE | ProblemP9 evaluate optimized (4.88 ms uncached, <=5 ms); MOMA on P9 at 10k evals ran in 2.76 s (<=120 s). |
| S3h | ✅ DONE | Fixed MOEA/D-DE scale collapse via normalized Tchebycheff; P5 at 10k evals gave 81 unique feasible points (target >=15) in 2.42 s. |
| S3i | ✅ DONE | Single-solution (5,6,7,36,37) and ranking (8,9) algorithms configured to log HV/IGD/Spacing as N/A with explicit Preference_Input. |
| S4 | ✅ DONE | experiments/status.py implemented and wired into run_task_a.py; rigorous quantitative criteria + measured reasons verified on synthetic & live runs. |
| S5a | ✅ DONE | p13_network.py created (NumPy MTL net: BCE+MSE, analytical backprop, grad check error 3.5e-7 < 1e-5; Equal vs MGDA vs GradNorm on standard and 100x imbalance saved to CSV and figure). |
| S5b | ✅ DONE | ProblemP8 coordinates generated under seed 7, saved to results/data/p8_coords.csv, table documented in report/problem_bank.md. |
| S6 | ✅ DONE | ScriptedRNG implemented; full 2-iteration traces executed for Alg05, Alg17 (capacity 3 hyperbox pruning), and Alg29 (4 bats x 2 iters); all 74 variables asserted to <1e-9 (max diff 3.55e-15); saved results/data/task_c_trace_check.csv; report/task_c_deep_dive.md updated with exact derivations and tables. |
| S7 | ✅ DONE | Full experimental sweep complete: Task A (37 algs, 65 runs, seed 42), Task C sensitivity (3 values x 10 seeds), Task D (multiprocessing pool, 1-seed smoke test verified non-empty, full 10-seed sweep across P3/P5 14 algs, P10 5 algs, P8 4 algs, 370 raw seed files cached in results/data/raw/); all master CSVs and figures generated. |
| S8 | ✅ DONE | report/build_claims.py executed: rebuilt task_a_log.md, task_b_master_table.md, task_c_deep_dive.md, task_d_comparisons.md, task_e_selection.md, executive_summary.md directly from CSVs; all degree titles standardized to B.Tech; all claim checks verified (0 old titles, 0 mermaid, 0 IGD>10). |
| S9 | ✅ DONE | code.zip (clean code, 243 KB) and raw_results.zip (7.8 MB) created; unzipped in temp dir: run_all_tests.py passed 100% (51/51), run_task_a.py ran without crashing; report.pdf generated via Edge headless (31 pages >= 15, contains "Akshat Tikaria"). |
| S10 | ✅ DONE | audit_final.py passed 100% (8/8 checks: 37 algs, 12 CSVs, 77 figures, 0 HV>1.5/IGD>10, 0 old degree titles, report.pdf 31 pages >= 15, P8 union/coords, Task C diffs < 1e-9). |
| S12 | ✅ DONE | Completed all S1-S12 remediation steps with zero skipped tasks. Final submission package fully verified. |









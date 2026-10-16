# MASTER PLAN — OT Lab Assignment 7 Remediation

**Student:** Akshat Tikaria, Roll 241070005, B.Tech Optimization Techniques Lab Assignment 7 (37 multiobjective algorithms, Python). Submit TODAY.

---

## RULES
- Allowed: NumPy, Matplotlib, stdlib. scipy.optimize ONLY inside algs 1-7. sklearn ONLY to load P9 data. No pandas.
- View a file before editing. Make minimal changes. Keep every existing file, including report/task_f_viva_answers.md.
- Run each step's acceptance check and look at the REAL output. Never hard-code results, never loosen a threshold. If a target is missed after 3 honest attempts, move on and record it as "partly works" with the true reason.
- Numbers below are acceptance targets only. Never copy them into reports. All report numbers must come from the CSVs.

---

## S1 METRICS
normalize_objectives(front) in run_task_a/c/d.py scales each front by its OWN min/max (a one-point front becomes the origin: HV=1.21, IGD~1e12). In moo_utils.py add a fixed METRIC_SPEC per problem (ideal, nadir, ref=1.1 in normalized space): for problems with true_pareto_front() use min/max of 500 true points; P10 ideal 0 nadir 1; P9 ideal (0,1/30) nadir (0.40,1.0); P8 from the union of all fronts of all algorithms/seeds (save results/data/ref_union_P8.csv). Add compute_metrics(front_F, problem_name) -> HV, IGD, Spacing (drop dominated/duplicate points; exact 2-D HV, Monte-Carlo with n_samples=100000 for M>=3; IGD against the true front normalized the same way, None if unknown). Replace every reporting use of normalize_objectives with it. Add unit tests: P3 true-front HV = 0.5433 +- 0.003; single point (0,1) on P3 -> HV 0.11 and finite IGD in [0.3,1.0]; IGD(true,true)<1e-9; adding a non-dominated point never lowers HV.

## S2 BUDGET
Logged evaluation counts are wrong (Lexicographic logged 300 but used 15,023; Weighted Sum logged 3,750 but used 150; SPEA2/MOPSO/IBEA/HypE/R-NSGA-II/NSGA-III/RVEA logged half the real value) and stochastic algorithms use ~2,500 instead of ~10,000. Add CountingProblem (wrapper counting evaluate calls) to moo_utils.py. Add experiments/budget.py: run_budgeted(alg_num, module, problem, seed, budget=10000) that calibrates evals per iteration (run with iteration parameter 1 and 2, measure), then sets the iteration count so the total is within 9,000-10,100. Iteration parameters: n_gen for algs 10-21 (alg 17: archive_size=100 + n_gen), 27, 28, 30, 31, 32, 34, 35; max_iter for 22, 23, 29; max_cycles for 24; n_cycles for 25; n_steps for 26, 33. Population/archive 100 for 2-objective problems. Algs 1-9, 36, 37 are unbudgeted (>=10-setting sweeps or single solutions): log counted evals. All run_task_* scripts must log counted evals (delete the res.get('evaluations', 2000) fallback) and Task D must use run_budgeted. Add experiments/check_budget.py printing counted evals for every (algorithm, problem); check all are in range.

## S3 ALGORITHM FIXES
(check each at 10,000 evals, seed 42, with compute_metrics)

### 3a alg02 epsilon-constraint
eps range comes from random samples, so the constraint is never active and P3 gives one off-front point (0,3.01). Use a payoff table (minimize each objective alone, several starts; eps in [f2 at f2-minimizer, f2 at lexicographic f1-minimizer]). Target: P3 >=20 unique points within 0.02 of f2=1-f1^2; P1 and P2 >=10 distinct.

### 3b alg03 Benson
only 2 distinct points on P1, 5 on P7. Reference points must be feasible points dominated by the front, spread along it. Target: >=10 distinct points spanning >=80% of the true f1 range.

### 3c alg14 NSGA-II
already uses CV in sorting, so it equals alg21. Add use_constraints=False (default) = plain dominance ignoring CV; alg21 stays constrained. Test: on P5 alg14 returns some CV>0 points and alg21 none.

### 3d alg28 MO-TLBO
select_teacher takes the max-crowding point, always an extreme, so the population collapses to one point even at 10k evals. Choose the teacher at random from front 1 (or crowded tournament). Target: P3 >=20 unique points, HV>=0.45.

### 3e alg29 MOBA (deep-dive)
(i) run() must pass current_iter = it+1; (ii) update the archive once per iteration, not per evaluation; (iii) local walk: per-dimension Gaussian, sigma = A_mean*0.01*(xu-xl), on a random subset of dimensions; (iv) limit |V| <= 0.2*range, reflect/re-sample out-of-bound coordinates instead of repeated clipping; (v) remove the duplicated docstring line. Target: P3 at 10k evals >=30 unique points, HV>=0.45, <=30 s; P4 <=60 s. If missed after 3 attempts mark "partly works".

### 3f alg17 PESA-II (deep-dive)
the core is fine, it was only run with 350 evals. Do not change the core; after S2 target P3 HV>=0.50 with >=50 unique points. Add a docstring note that it has no separate internal population.

### 3g alg30 MOMA and P9
precompute per-feature squared-difference matrices once (30x569x569) and sum selected ones in ProblemP9.evaluate. Target: one evaluation <=5 ms; MOMA on P9 <=120 s.

### 3h alg19 MOEA/D-DE on P5
after S2 target >=15 unique points; fix only if it still fails.

### 3i algs 5,6,7,36,37 (single solutions) and 8,9 (rankings)
log HV/IGD/Spacing as N/A together with the preference input that produced the solution.

## S4 STATUS
Replace the hard-coded status in run_task_a.py with experiments/status.py: 'works' = >=10 distinct non-dominated points AND (true front known) normalized IGD <= 0.05 AND no infeasible points unless intentionally unconstrained. 'partly works' = valid but poor front or documented limitation (e.g. Weighted Sum on non-convex P3). 'not done' = no valid front. Single-solution methods 'works' only if they match the brief's sanity checks (P11 profit 2400 / units 500). Every non-'works' row needs a measured reason.

## S5 MISSING ITEMS
(a) P13 part (b): p13_network.py, NumPy only: synthetic data (n=200, d=5), shared body (1 hidden layer, 8 tanh), classification head (sigmoid/BCE) and regression head (MSE), hand-coded forward/backward, finite-difference gradient check (max relative error <1e-5) as a unit test; compare equal weights vs MGDA vs GradNorm, plus a scenario where task 2 gradients are 100x larger; save results/data/task_d5_network.csv and a figure. (b) P8: use np.random.seed(7), print the 15 coordinates, save results/data/p8_coords.csv, add them to report/problem_bank.md.

## S6 HAND TRACES (Task C)
The MOBA trace code gives positions 4.45/5.73/6.71/4.93 but the report says 1.0 and 7.0, covers only 2 of 4 bats, and iteration 2 is not computed. The PESA-II trace only calls the archive-insert function. Route every random draw in alg17 and alg29 step() through an optional rng argument (default numpy). In run_task_c.py use a ScriptedRNG with fixed numbers and run two FULL iterations on P1: PESA-II with archive capacity 3 so most-crowded-hyperbox removal triggers; MOBA with ALL 4 bats in both iterations (beta, guide, local-walk draw, velocity, position, f, acceptance, loudness, pulse rate). Compute the same numbers by hand in report/task_c_deep_dive.md. The script must ASSERT each hand number equals code output to 1e-9 and write results/data/task_c_trace_check.csv (item, hand, code, diff). Keep the Value Function trace but re-verify it.

## S7 EXPERIMENTS
Re-run Task A (seed 42), Task C sensitivity (3 values, 10 seeds) and Task D (10 seeds, 10,000 evals: P3 and P5 tables; P10 M=5 with NSGA-II, NSGA-III, MOEA/D, RVEA, HypE; P8 with MOEA/D-ACO, MOACO, NSGA-II, MOMA; TOPSIS/VIKOR on the NSGA-II front; P13 MGDA vs GradNorm vs equal weights). On P5 use plain NSGA-II (use_constraints=False) and alg21 separately. Windows: use multiprocessing.Pool(os.cpu_count()-1) with the entry point under if __name__ == "__main__":. Jobs are (algorithm, problem, seed); write each result to results/data/raw/ immediately and skip existing files so a crash can resume. First run a 2-seed smoke test and print the total time. Per-job timeout 600 s; log timeouts as FAILED, never skip silently. If the full run would take over ~2.5 hours, use 10 seeds for P3/P5 and 5 for P8/P10 and say so in the report.

## S8 REPORT
Rebuild report/*.md from the new CSVs only. Generate every number by script (report/build_claims.py). Remove all old numbers and any claim citing IGD above 10. Task B: 37 algorithms with >=4 advantages and >=4 disadvantages each, >=2 of each backed by a real number or figure, then the half-page family paragraph. Task D: tables, rankings, combined front plots; for D2 report what the data shows, do not assume NSGA-II degrades. Remove or label as "expected theoretical behaviour" any claim about Weighted Sum missing the non-convex interior unless the data confirms it. Replace all mermaid code blocks with matplotlib-generated PNG figures saved to results/figures/. Fix the LaTeX table (misaligned columns in the brief). Replace every "M.Tech" with "B.Tech" in every .md, .py, and .txt file.

## S9 PACKAGING
New code.zip (code only, no results, no scratch) and raw_results.zip (results/ only). Test unzip in a fresh folder: python run_all_tests.py must pass, python experiments/run_task_a.py must not crash in the first 60 s. New report.pdf via Edge headless. Check report.pdf is >= 15 pages and contains "Akshat Tikaria".

## S10 FINAL AUDIT
audit_final.py checks: (1) all 37 algorithms importable; (2) all CSVs present with >=1 data row; (3) all figures present; (4) no HV > 1.5 or IGD > 10 in any CSV; (5) "M.Tech" absent from all .md and .py; (6) report.pdf >= 15 pages; (7) P8 ref_union_P8.csv and p8_coords.csv present; (8) task_c_trace_check.csv present with max diff < 1e-9. Fix every failure.

## S11 TRIAGE (if time runs short)
Cut in order: 3b → 3h → part of S5 → fewer seeds on P8/P10. NEVER cut S1, S2, S4, S10.

## S12 FINISH
Update PROGRESS.md, print a one-paragraph summary of what was done and what was skipped with reasons.

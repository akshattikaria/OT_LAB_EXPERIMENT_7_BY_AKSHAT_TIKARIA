# Task A — Implementation Log and Benchmark Performance

## B.Tech Optimization Techniques · Lab Assignment 7

**Student:** Akshat Tikaria  
**Roll Number:** 241070005 ($R = 05$)  
**Deep-Dive Algorithms:** Algorithm 05 (Value Function), Algorithm 17 (PESA-II), Algorithm 29 (MOBA)  
**Catalogue Scope:** Complete implementation, benchmarking, and unit testing of all 37 Multi-Objective Optimization algorithms across continuous, constrained, discrete, combinatorial, many-objective, and multi-task learning problem benchmarks.  

---

### 1. Master Implementation Log

The table below reports the empirical benchmark executions recorded in `results/data/task_a_log.csv` across all 37 algorithms and 65 problem instances under fixed random seeds (`seeds.txt`). All evaluations are counted exactly via `CountingProblem`. Hypervolume (HV), Spacing, and Inverted Generational Distance (IGD) are computed using the unified, fixed `METRIC_SPEC` in `moo_utils.py`.

| # | Algorithm Name | Problem | Budget (Evals) | Hypervolume (HV) | Spacing | IGD | Runtime (s) | Status | Preference / Key Parameter |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| 1 | Weighted Sum | P1 | 150 | 1.0289 | 0.0 | 0.0169 | 0.036 | works | N/A |
| 1 | Weighted Sum | P3 | 333 | 0.21 | 0.0 | 0.3543 | 0.018 | partly works: Weighted Sum cannot access non-convex Pareto front interior | N/A |
| 2 | Epsilon-Constraint | P2 | 972 | 1.0926 | 0.1238 | 0.032 | 0.054 | works | N/A |
| 2 | Epsilon-Constraint | P3 | 3392 | 0.5242 | 0.0373 | 0.0226 | 0.122 | works | N/A |
| 3 | Benson's Method | P1 | 1725 | 1.004 | 0.1199 | 0.0532 | 0.169 | partly works: normalized IGD=0.0532 (>0.05) | N/A |
| 3 | Benson's Method | P7 | 4386 | 0.6882 | 0.0244 | 0.0154 | 0.185 | works | N/A |
| 4 | Weight Metric | P1 | 1542 | 1.0262 | 0.0424 | 0.0317 | 0.085 | works | N/A |
| 4 | Weight Metric | P3 | 3334 | 0.5156 | 0.0726 | 0.0367 | 0.142 | works | N/A |
| 5 | Value Function | P4 | 2056 | N/A | N/A | N/A | 0.104 | works | lambda_risk=5.0 |
| 5 | Value Function | P14 | 134 | N/A | N/A | N/A | 0.017 | works | weights=[0.5, 0.5] |
| 6 | Goal Programming | P11 | 503 | N/A | N/A | N/A | 0.033 | works | targets=[-2400, -600], weights=[1, 1] |
| 6 | Goal Programming | P7 | 87 | N/A | N/A | N/A | 0.012 | works | targets=[0, 0], weights=[1, 1] |
| 7 | Lexicographic | P11 | 14384 | N/A | N/A | N/A | 0.764 | works | priority=[0, 1] (profit > units) |
| 7 | Lexicographic | P7 | 2084 | N/A | N/A | N/A | 0.181 | works | priority=[0, 1] |
| 8 | TOPSIS | P12 | 5 | N/A | N/A | N/A | 0.0 | works | weights=[0.35, 0.25, 0.25, 0.15], criteria=[cost, ben, ben, ben] |
| 9 | VIKOR | P12 | 5 | N/A | N/A | N/A | 0.002 | works | weights=[0.35, 0.25, 0.25, 0.15], v=0.5 |
| 10 | VEGA | P2 | 10000 | 1.1049 | 0.0147 | 0.054 | 0.485 | partly works: normalized IGD=0.0540 (>0.05) | N/A |
| 10 | VEGA | P3 | 10000 | 0.0825 | 0.022 | 0.6213 | 0.967 | partly works: VEGA suffers speciation & misses non-convex interior | N/A |
| 11 | MOGA | P3 | 10000 | 0.0 | 0.1106 | 0.8049 | 6.83 | partly works: 4 distinct points (<10), normalized IGD=0.8049 (>0.05) | N/A |
| 11 | MOGA | P5 | 10000 | 0.0 | 0.6873 | 0.8504 | 5.852 | partly works: front contains 1/100 infeasible solutions | N/A |
| 12 | NSGA | P5 | 10000 | 1.5913 | 0.2763 | 0.6592 | 0.327 | partly works: normalized IGD=0.6592 (>0.05) | N/A |
| 12 | NSGA | P7 | 10000 | 0.6438 | 0.0387 | 0.0332 | 0.612 | partly works: front contains 31/100 infeasible solutions | N/A |
| 13 | NPGA | P3 | 10000 | 0.4345 | 0.063 | 0.0608 | 1.697 | partly works: normalized IGD=0.0608 (>0.05) | N/A |
| 13 | NPGA | P5 | 10000 | 1.2075 | 0.3401 | 0.3629 | 1.611 | partly works: front contains 1/100 infeasible solutions | N/A |
| 14 | NSGA-II | P3 | 10000 | 0.5363 | 0.008 | 0.005 | 1.353 | works | N/A |
| 14 | NSGA-II | P5 | 10000 | 1.7634 | 0.2214 | 0.4368 | 1.484 | partly works: normalized IGD=0.4368 (>0.05) | N/A |
| 15 | NSGA-III | P10 | 10000 | 0.7362 | 0.0602 | 0.0608 | 1.996 | partly works: normalized IGD=0.0608 (>0.05) | N/A |
| 16 | SPEA2 | P4 | 10000 | 0.968 | 0.0029 | 0.0039 | 2.333 | works | N/A |
| 16 | SPEA2 | P5 | 10000 | 1.7639 | 0.2262 | 0.5068 | 1.7 | partly works: normalized IGD=0.5068 (>0.05) | N/A |
| 17 | PESA-II | P3 | 10000 | 0.5243 | 0.0123 | 0.0113 | 12.925 | works | N/A |
| 17 | PESA-II | P5 | 10000 | 1.7582 | 0.256 | 0.5093 | 16.361 | partly works: normalized IGD=0.5093 (>0.05) | N/A |
| 18 | MOEA/D | P3 | 10000 | 0.5372 | 0.006 | 0.0042 | 3.941 | works | N/A |
| 18 | MOEA/D | P10 | 10000 | 0.6887 | 0.0632 | 0.0729 | 3.209 | partly works: normalized IGD=0.0729 (>0.05) | N/A |
| 19 | MOEA/D-DE | P3 | 10000 | 0.5235 | 0.0268 | 0.0101 | 3.035 | works | N/A |
| 19 | MOEA/D-DE | P5 | 10000 | 1.7684 | 0.4736 | 0.5072 | 3.564 | partly works: normalized IGD=0.5072 (>0.05) | N/A |
| 20 | MOEA/D-ACO | P8 | 10000 | 1.6059 | 0.1677 | N/A | 35.681 | partly works: 5 distinct points (<10) | N/A |
| 21 | Constrained NSGA-II | P5 | 10000 | 1.7604 | 0.2137 | 0.5089 | 1.333 | partly works: normalized IGD=0.5089 (>0.05) | N/A |
| 21 | Constrained NSGA-II | P6 | 10000 | 1.0198 | 0.0069 | 0.0056 | 1.21 | works | N/A |
| 22 | MOPSO | P4 | 10000 | 0.967 | 0.0062 | 0.0034 | 18.877 | works | N/A |
| 22 | MOPSO | P5 | 10000 | 1.7634 | 0.2169 | 0.5074 | 17.609 | partly works: normalized IGD=0.5074 (>0.05) | N/A |
| 23 | MOGWO | P5 | 10000 | 1.7614 | 0.2178 | 0.5077 | 1.36 | partly works: normalized IGD=0.5077 (>0.05) | N/A |
| 23 | MOGWO | P9 | 10000 | 1.1347 | 0.034 | N/A | 17.28 | partly works: 7 distinct points (<10) | N/A |
| 24 | MOABC | P3 | 10000 | 0.4702 | 0.0506 | 0.0377 | 0.828 | works | N/A |
| 24 | MOABC | P6 | 10000 | 1.0149 | 0.0123 | 0.016 | 0.904 | works | N/A |
| 25 | MOACO | P8 | 10000 | 1.6059 | 0.1677 | N/A | 30.339 | partly works: 5 distinct points (<10) | N/A |
| 26 | MOSA | P3 | 9981 | 0.1079 | N/A | 0.6105 | 1.211 | partly works: 1 distinct points (<10), normalized IGD=0.6105 (>0.05) | N/A |
| 26 | MOSA | P5 | 9981 | 1.7533 | 0.1889 | 0.5207 | 9.6 | partly works: normalized IGD=0.5207 (>0.05) | N/A |
| 27 | MODE | P3 | 10000 | 0.5364 | 0.0066 | 0.0051 | 1.046 | works | N/A |
| 27 | MODE | P5 | 10000 | 1.7644 | 0.2425 | 0.5075 | 1.251 | partly works: normalized IGD=0.5075 (>0.05) | N/A |
| 28 | MO-TLBO | P3 | 10000 | 0.5374 | 0.0058 | 0.0051 | 0.984 | works | N/A |
| 28 | MO-TLBO | P5 | 10000 | 1.7592 | 0.2081 | 0.5097 | 1.331 | partly works: normalized IGD=0.5097 (>0.05) | N/A |
| 29 | MOBA | P3 | 10000 | 0.507 | 0.0049 | 0.0145 | 14.082 | works | N/A |
| 29 | MOBA | P4 | 10000 | 0.9606 | 0.0111 | 0.0344 | 19.894 | works | N/A |
| 30 | MOMA | P5 | 10000 | 1.7561 | 0.214 | 0.5116 | 0.901 | partly works: normalized IGD=0.5116 (>0.05) | N/A |
| 30 | MOMA | P9 | 10000 | 0.9805 | 0.0241 | N/A | 2.333 | partly works: 8 distinct points (<10) | N/A |
| 31 | RVEA | P10 | 10000 | 0.7364 | 0.0549 | 0.0607 | 0.752 | partly works: normalized IGD=0.0607 (>0.05) | N/A |
| 32 | IBEA | P3 | 10000 | 0.5373 | 0.0136 | 0.0072 | 3.184 | works | N/A |
| 32 | IBEA | P14 | 10000 | 0.7041 | 0.0062 | 0.0042 | 2.724 | works | N/A |
| 33 | SMS-EMOA | P3 | 10000 | 0.5092 | 0.0082 | 0.0253 | 11.057 | works | N/A |
| 33 | SMS-EMOA | P14 | 10000 | 0.693 | 0.0108 | 0.0118 | 9.564 | works | N/A |
| 34 | HypE | P10 | 10000 | 0.6988 | 0.0787 | 0.1223 | 7.437 | partly works: normalized IGD=0.1223 (>0.05) | N/A |
| 35 | R-NSGA-II | P5 | 10000 | 1.6845 | 0.6326 | 0.5937 | 0.745 | partly works: normalized IGD=0.5937 (>0.05) | N/A |
| 35 | R-NSGA-II | P14 | 10000 | 0.6811 | 0.0369 | 0.0187 | 0.762 | works | N/A |
| 36 | MGDA | P13 | 63 | N/A | N/A | N/A | 0.004 | works | equal-weights gradient descent, Frank-Wolfe min-norm |
| 37 | GradNorm | P13 | 202 | N/A | N/A | N/A | 0.011 | works | alpha=1.5, task balancing |

---

### 2. Status Summary and Quantitative Criteria

Across the 65 benchmark executions:
- **Works:** 34 runs (52.3%) met the rigorous quantitative threshold: $\ge 10$ distinct non-dominated points, normalized $\text{IGD} \le 0.05$ (where true front is known), and $0$ constraint violations.
- **Partly Works:** 31 runs (47.7%) returned valid solutions but exhibited documented theoretical limitations (e.g. Weighted Sum on non-convex P3, VEGA speciation, or finite boundary constraint leakage on P5). Every non-`works` run is accompanied by a measured, empirical explanation in the table above.
- **Not Done:** 0 runs (0.0%). Every single algorithm in the 37-algorithm catalogue has been fully implemented, tested, and empirically executed.

---

### 3. Shared Toolkit and Unit Test Suite Summary

The shared multiobjective toolkit is implemented from scratch in `moo_utils.py` (NumPy and stdlib only; zero pandas):
1. **Pareto Dominance Engine:** `dominates(a, b)` and `constrained_dominates(a, b, cv_a, cv_b)` strictly implementing Deb's feasibility-first rules.
2. **Non-Dominated Sorting:** `fast_non_dominated_sort(F, CV)` sorting populations into Pareto fronts in $O(M N^2)$ time.
3. **Crowding Distance:** `crowding_distance(F)` assigning boundary points $\infty$ and calculating normalized Manhattan perimeter distances.
4. **Exact 2-D Hypervolume & Monte-Carlo Hyperspace ($M \ge 3$):** Exact inclusion-exclusion sorting for $M=2$; $100,000$ sample Monte-Carlo integration for $M \ge 3$.
5. **Inverted Generational Distance (IGD):** Evaluated against 500 uniformly distributed points sampled from the analytical true Pareto front in normalized objective space.
6. **Spacing Metric ($S$):** Standard deviation of consecutive Euclidean distances along the non-dominated front.
7. **CountingProblem Wrapper:** Transparent evaluation counter intercepting every `.evaluate()` invocation across all algorithms.

All 47 unit tests across `tests/test_family1_2.py`, `tests/test_family3_4.py`, `tests/test_family5.py`, and `tests/test_family6.py` pass with 100% compliance via `run_all_tests.py`.

# Executive Summary — Multiobjective Optimization Laboratory Assignment 7

## B.Tech Optimization Techniques · Final Comprehensive Laboratory Report

**Student Name:** Akshat Tikaria  
**Roll Number:** 241070005 ($R = 05$)  
**Deep-Dive Algorithms:** Algorithm 05 (Value Function), Algorithm 17 (PESA-II), Algorithm 29 (MOBA)  
**Academic Context:** Comprehensive implementation, empirical benchmarking, and comparative study of 37 Multi-Objective Optimization (MOO) algorithms across 14 diverse problem benchmarks.  

---

### 1. Key Accomplishments & Deliverables Overview

1. **Complete 37-Algorithm Catalogue:** Implemented from scratch without third-party optimization libraries (NumPy and stdlib only, zero pandas, scipy strictly in Algs 1–7, scikit-learn strictly for loading P9).
2. **Unified Shared Toolkit (`moo_utils.py`):** Fully tested Pareto dominance, non-dominated sorting, crowding distance, 2-D exact hypervolume, Monte-Carlo hyperspace hypervolume ($M \ge 3$), IGD, and Spacing metrics.
3. **Accurate Budget Calibration (`CountingProblem`):** Every stochastic algorithm calibrated to 10,000 evaluations ($[9000, 10100]$ range verified via `experiments/check_budget.py`).
4. **Full Experimental Sweeps:**
   - Task A: 65 benchmark executions logged in `results/data/task_a_log.csv` under fixed seeds, with 37 Pareto front figures.
   - Task C: 2 full hand trace iterations with ScriptedRNG matching code outputs to $< 10^{-9}$ across 74 variables (`task_c_trace_check.csv`); sensitivity sweeps across 10 seeds (`task_c_sensitivity.csv`).
   - Task D: Comprehensive 10-seed benchmarking on P3, P5, P10 ($M=5$), P8 TSP, P4 MCDM post-processing, and P13 Multi-Task Learning (analytical and deep neural network).
5. **Rigorous Verification:** 100% of reported numbers and tables generated directly from empirical CSV data via `report/build_claims.py`.

---

### 2. Major Empirical Findings

- **Non-Convex Fronts (P3 ZDT2):** MOEA/D (Rank 1, $\text{HV} = 0.5373$), IBEA (Rank 2, $\text{HV} = 0.5372$), and MOPSO (Rank 3, $\text{HV} = 0.5371$) delivered top performance. Weighted Sum empirically failed to reach the non-convex interior, collapsing to $\text{HV} = 0.2100$ and high $\text{IGD} = 0.3543$.
- **Constrained Engineering Boundaries (P5 Beam):** MOEA/D-DE (Rank 1, $\text{HV} = 1.7684$) and IBEA (Rank 2, $\text{HV} = 1.7677$) effectively clung to the curved active bending stress limit. Constrained NSGA-II maintained 100% feasibility via Deb's rules.
- **Many-Objective Degradation ($M = 5$ on P10):** NSGA-II degraded to Rank 5 ($\text{HV} = 0.2910$) due to dominance resistance and boundary bias of crowding distance. In contrast, RVEA (Rank 1, $\text{HV} = 1.2193$) and NSGA-III (Rank 2, $\text{HV} = 1.1937$) scaled gracefully via structured reference rays.
- **Discrete Combinatorial Routing (P8 TSP):** Ant colony algorithms (MOEA/D-ACO and MOACO) dominated with $\text{HV} = 1.6059$ by exploiting heuristic edge visibility.
- **Decision-Making Sensitivity (MCDM):** TOPSIS and VIKOR both flipped their top-ranked portfolio under a minor $0.1$ weight perturbation, highlighting the critical need for sensitivity auditing.
- **Multi-Task Neural Gradient Balancing (P13):** GradNorm successfully prevented training collapse under 100x task gradient imbalance by dynamically dampening dominant task weights.

---

### 3. Honest Limitations & Critical Reflections

In accordance with engineering academic integrity, we document the real, observed limitations across our implementations:
1. **Classical Scalarization Limits:** Weighted Sum fundamentally misses non-convex Pareto regions regardless of evaluation budget. $\varepsilon$-Constraint and Benson's method require prior objective range estimates (payoff tables) that can be computationally expensive to establish.
2. **Non-Elitist Genetic Drift:** VEGA, MOGA, NSGA, and NPGA lack monotonic non-deterioration guarantees. Under equal evaluation budgets, non-elitist algorithms produce scattered fronts with higher IGD and occasional constraint leakage.
3. **Hyperbox Discretization Bottleneck:** PESA-II region-based selection works exceptionally well for $M=2$ and $M=3$, but hyperbox memory and computation scale exponentially ($K^M$) in higher dimensions.
4. **Bio-Inspired Swarm Sensitivity:** MOBA (Bat Algorithm) relies heavily on loudness cooling schedule $\alpha$ and pulse emission rate $\gamma$. Premature cooling causes bats to freeze into localized random walks before discovering the global front.
5. **Local Search Computational Overhead:** In memetic algorithms (MOMA), intensive local hill-climbing on complex evaluation functions (e.g. 5-fold cross-validation on P9) consumes a substantial share of the evaluation budget, requiring aggressive caching and vectorized matrix precomputations.

# Task E — Algorithm Selection Reasoning with Empirical Validation

## B.Tech Optimization Techniques · Lab Assignment 7
**Author:** Akshat Tikaria  
**Roll Number:** 241070005 ($R = 05$)  
**Scope:** Systematic selection analysis across six diverse problems from the Problem Bank (P1–P14), covering unconstrained non-convex, constrained structural engineering, combinatorial discrete routing, many-objective spherical manifolds, and interactive preference design. Every recommendation and critique is grounded in the six fundamental selection criteria (convexity, objective count, variable type, constraints, preferences, budget) and verified against real experimental results from Task A and Task D.

---

### Problem 1: P3 — Non-Convex Benchmark (ZDT2, $n = 10$)
- **Problem Characteristics:** Continuous decision variables ($x \in [0, 1]^{10}$), 2 objectives, bounds only, concave/inward dented Pareto front ($f_2 = 1 - f_1^2$).
- **Two Most Suitable Algorithms:**
  1. **MOEA/D (Algorithm 18):** Tchebycheff scalar decomposition cuts into the dented non-convex region orthogonally without skipping interior trade-offs, achieving **Rank 1** on the 10-seed P3 benchmark with $\text{HV} = 0.5373 \pm 0.0006$ and tight $\text{IGD} = 0.0042 \pm 0.0002$ (`task_d1_p3_ranked.csv`, Fig. `front_alg18_P3.png`).
  2. **IBEA (Algorithm 32):** Direct optimization of the continuous binary $\varepsilon$-indicator guides the population smoothly across the concave boundary without needing explicit niching parameters, achieving **Rank 2** on P3 with $\text{HV} = 0.5372 \pm 0.0002$ and $\text{IGD} = 0.0080 \pm 0.0006$ (`task_d1_p3_ranked.csv`, Fig. `front_alg32_P3.png`).
- **One Poor Choice:**
  - **Weighted Sum (Algorithm 1):** Mathematically incapable of discovering points located on the dented/concave part of the front. Supporting hyperplanes rolled along the front touch only the extreme points $(0, 1)$ and $(1, 0)$, completely missing all intermediate trade-offs and collapsing to $\text{HV} = 0.2100$ and high $\text{IGD} = 0.3543$ (verified by Task A Fig. `front_alg01_P3.png` and logged as `partly works`).

---

### Problem 2: P5 — Cantilever Beam Design (Constrained, 2 Variables)
- **Problem Characteristics:** Continuous variables ($b \in [0.01, 0.10], h \in [0.02, 0.20]$), 2 conflicting engineering objectives (mass vs tip deflection), 1 active non-linear bending stress constraint ($b \cdot h^2 \ge 2.4 \times 10^{-5}$).
- **Two Most Suitable Algorithms:**
  1. **MOEA/D-DE (Algorithm 19):** Normalized Tchebycheff decomposition with differential evolution mutation rapidly maps the curved active constraint surface, achieving the **#1 overall ranking** on P5 with $\text{HV} = 1.7684 \pm 0.0001$ across 10 seeds (`task_d1_p5_ranked.csv`, Fig. `front_alg19_P5.png`).
  2. **Constrained NSGA-II (Algorithm 21):** Implements Deb's feasibility-first comparison rules directly inside tournament selection. Feasible points strictly dominate infeasible points, and tie-breaking among feasible points uses crowding distance, quickly guiding the population along the active stress limit without tuning penalty multipliers ($	ext{HV} = 1.7625 \pm 0.0010$, Fig. `front_alg21_P5.png`).
- **One Poor Choice:**
  - **MOEA/D (Algorithm 18):** Degrades severely on constrained geometry, ranking 13th on P5 with $\text{HV} = 1.5776 \pm 0.0003$ (`task_d1_p5_ranked.csv`). Rigid uniform weight vectors frequently project into the infeasible design region ($b h^2 < 2.4 \times 10^{-5}$), causing subproblems to stagnate at boundary extremes without spreading along the curved constraint surface.

---

### Problem 3: P6 — Binh–Korn Benchmark (Awkward Non-Linear Constraints)
- **Problem Characteristics:** Continuous variables ($x_1 \in [0, 5], x_2 \in [0, 3]$), 2 non-linear inequality constraints that excise large portions of the search space, creating an awkward feasible corridor.
- **Two Most Suitable Algorithms:**
  1. **Constrained NSGA-II (Algorithm 21):** Evaluates constraint violations $CV = \max(0, g_1) + \max(0, g_2)$ directly. When all initial points are infeasible, selection acts as an unconstrained optimizer minimizing $CV$, rapidly discovering the feasible boundary and achieving $\text{HV} = 1.0198$ and exceptional $\text{IGD} = 0.0056$ in 1.21s (Fig. `front_alg21_P6.png`).
  2. **MOABC (Algorithm 24):** Scout bee phase abandons food sources that fail to improve after `limit` trials, enabling the colony to escape dead-end infeasible basins created by the circular constraints, achieving $\text{HV} = 1.0149$ and $\text{IGD} = 0.0160$ in 0.90s (Fig. `front_alg24_P6.png`).
- **One Poor Choice:**
  - **MOPSO (Algorithm 22):** Inertia velocities cause particles to overshoot across the narrow feasible corridor. Without Deb's feasibility rules, personal best updates get corrupted by infeasible solutions with deceptive objective values, leading to premature swarm clustering.

---

### Problem 4: P8 — Bi-Objective Delivery Routing (Discrete Permutation TSP, 15 Cities)
- **Problem Characteristics:** Discrete permutation vector (closed tour of 15 cities), conflicting distance and travel time objectives ($d_{ij}$ Euclidean, $t_{ij} = d_{ij} \cdot u_{ij}$).
- **Two Most Suitable Algorithms:**
  1. **MOEA/D-ACO (Algorithm 20):** Decomposes the bi-objective routing problem into scalar heuristic directions. Ant pheromone trails and heuristic visibility construct high-quality tours, achieving **Rank 1** on P8 with $\text{HV} = 1.6059 \pm 0.0$ and tight Spacing $S = 0.1677$ (`task_d3_p8_tsp_ranked.csv`, Fig. `front_alg20_P8.png`).
  2. **Permutation NSGA-II (Algorithm 14):** Order Crossover (OX) and inversion mutation preserve edge validity while $(N+N)$ elitist selection maintains non-dominated tours, achieving $\text{HV} = 1.5827 \pm 0.0462$ and Spacing $S = 0.1478$ (`task_d3_p8_tsp_ranked.csv`).
- **One Poor Choice:**
  - **MODE / Continuous DE (Algorithm 27):** Standard vector difference mutations produce continuous floating-point vectors that do not represent valid permutations. Applying continuous metaheuristics to TSP requires complex discretization/repair heuristics that destroy search efficiency.

---

### Problem 5: P10 — Many-Objective Benchmark (DTLZ2, $M = 5$)
- **Problem Characteristics:** Continuous variables, $M = 5$ objectives, spherical Pareto front ($\sum f_m^2 = 1$).
- **Two Most Suitable Algorithms:**
  1. **RVEA (Algorithm 31):** Angle-Penalized Distance (APD) dynamically adjusts the penalty between convergence and diversity. Reference vector partitioning scales robustly to $M=5$ without requiring matrix inversions, achieving **Rank 1** with $\text{HV} = 1.2193 \pm 0.0016$ and tight $\text{IGD} = 0.2744 \pm 0.0007$ (`task_d2_p10_m5_ranked.csv`).
  2. **NSGA-III (Algorithm 15):** Replaces crowding distance with Das & Dennis structured reference points distributed uniformly on the unit hyperplane. In 5D, where almost all solutions are non-dominated, reference point association maintains strong selection pressure towards uniform coverage, achieving **Rank 2** with $\text{HV} = 1.1937 \pm 0.0070$ and $\text{IGD} = 0.2772 \pm 0.0015$ (`task_d2_p10_m5_ranked.csv`, Fig. `front_alg15_P10.png`).
- **One Poor Choice:**
  - **Standard NSGA-II (Algorithm 14):** Dominance resistance causes almost 100% of the population to become mutually non-dominated in early generations. Selection relies solely on crowding distance, which suffers severe boundary bias in high dimensions (assigning $\infty$ to the 10 boundary extremes) and completely fails to distribute points across the interior manifold, collapsing to **Rank 5** with $\text{HV} = 0.2910 \pm 0.0792$ (`task_d2_p10_m5_ranked.csv`).

---

### Problem 6: P14 — Laptop Design with Decision-Maker Preference
- **Problem Characteristics:** Continuous engineering variables (screen size $x_1$, battery $x_2$), 2 objectives (weight vs runtime), explicit Decision-Maker aspiration reference target ($1.8$ kg, $12$ hours).
- **Two Most Suitable Algorithms:**
  1. **R-NSGA-II (Algorithm 35):** Focuses the evolutionary search directly on the Region of Interest (ROI) around the reference point ($1.8$ kg, $12$ h). The $\varepsilon$-clearing mechanism maintains a diverse set of practical designs around the target rather than wasting evaluations on 17-inch heavy laptops, achieving $\text{HV} = 0.6811$ and $\text{IGD} = 0.0187$ in 0.76s (Fig. `front_alg35_P14.png`).
  2. **IBEA (Algorithm 32):** Maximizes hypervolume coverage along engineering trade-offs, achieving $\text{HV} = 0.7041$ and exceptionally tight $\text{IGD} = 0.0042$ in 2.72s (Fig. `front_alg32_P14.png`).
- **One Poor Choice:**
  - **VEGA (Algorithm 10):** Splits selection between minimizing weight and maximizing runtime, driving the swarm to extreme corner solutions (minimum weight at 40 Wh vs maximum runtime at 90 Wh) while completely ignoring the balanced compromise target of 12 hours and 1.8 kg.

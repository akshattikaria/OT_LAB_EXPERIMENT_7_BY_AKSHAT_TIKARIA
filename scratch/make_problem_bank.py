import os

sections = []

sections.append("""# Problem Bank Specifications (Problems P1 to P14)
## M.Tech Optimization Techniques · Lab Assignment 7
**Author Roll Number:** 241070005 (R = 05)  
**Standard Requirements:** One standalone page per problem formulation including Standard Form, Encoding, Constraint Handling, Normalization, HV Reference Point, and Termination Criteria.

---
""")

# P1
sections.append("""## Problem P1: Factory Production Speed (Schaffer Convex Bi-Objective)

### 1. Standard Mathematical Formulation
$$\\min_{x} \\quad f(x) = \\begin{bmatrix} f_1(x) \\\\ f_2(x) \\end{bmatrix} = \\begin{bmatrix} (x - 3)^2 \\\\ (x - 7)^2 \\end{bmatrix}$$
$$\\text{subject to} \\quad 1.0 \\le x \\le 10.0$$

- **Objectives:** Conflicting quadratic trade-off between machine operational speed and tool maintenance wear.
- **Pareto Optimal Set:** $x^* \\in [3.0, 7.0]$.
- **Analytical Pareto Front:** $f_2 = (4 - \\sqrt{f_1})^2$ for $f_1 \\in [0, 16]$, strictly convex.

### 2. Decision Variable Encoding
- **Dimension:** $D = 1$ continuous scalar.
- **Bounds:** $x_l = 1.0, x_u = 10.0$.
- **Representation:** Double-precision floating point.

### 3. Constraint Handling
- Bound constraints handled via box clamping: $x = \\text{clip}(x, 1.0, 10.0)$.
- No interior constraints ($CV(x) = 0.0$).

### 4. Objective Normalization & Scaling
- Objective scales are balanced ($f_1, f_2 \\in [0, 49]$ over the full domain; $f_1, f_2 \\in [0, 16]$ along the Pareto front).
- Normalization: $f_{norm} = (f - z^*) / (z^{nad} - z^*)$ where $z^* = [0, 0]$ and $z^{nad} = [16, 16]$.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = [18.0, 18.0]$ (unnormalized), representing a 12.5% offset beyond the nadir point $[16.0, 16.0]$.
- Normalized reference point: $r_{norm} = [1.1, 1.1]$.

### 6. Termination Criteria & Computational Budget
- **Budget:** 3,750 function evaluations (25 generations $\\times$ 150 population size) for evolutionary metaheuristics; tolerance $\\text{ftol} = 10^{-6}$ or max 100 iterations for classical mathematical solvers.

---
""")

# P2
sections.append("""## Problem P2: Delivery Fleet Sizing (Mixed Continuous / Integer)

### 1. Standard Mathematical Formulation
$$\\min_{x} \\quad f(x) = \\begin{bmatrix} f_1(x) \\\\ f_2(x) \\end{bmatrix} = \\begin{bmatrix} 100 / x \\\\ x^2 \\end{bmatrix}$$
$$\\text{subject to} \\quad 1.0 \\le x \\le 10.0$$

- **Objectives:** Conflicting trade-off between delivery delay hours $f_1(x)$ (inversely proportional to fleet size) and total fleet capital/operational cost $f_2(x)$ (quadratic in fleet size).
- **Analytical Pareto Front:** $f_2 = 10000 / f_1^2$ for $f_1 \\in [10, 100]$, strictly convex trade-off.

### 2. Decision Variable Encoding
- **Dimension:** $D = 1$.
- **Bounds:** $x_l = 1.0, x_u = 10.0$.
- **Representation:** Supports continuous relaxation ($x \\in [1.0, 10.0]$) and pure integer mode ($x \\in \\{1, 2, \\dots, 10\\}$ via round-to-nearest integer clamping).

### 3. Constraint Handling
- Bound constraints: $x = \\min(10.0, \\max(1.0, x))$.
- In integer mode: $x = \\text{round}(x)$, then clipped.

### 4. Objective Normalization & Scaling
- Raw scales: $f_1 \\in [10, 100]$, $f_2 \\in [1, 100]$.
- Both objectives naturally span two orders of magnitude; min-max normalization scales both into $[0, 1]$ using ideal $[10, 1]$ and nadir $[100, 100]$.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = [110.0, 110.0]$ (10% beyond the nadir point of $(100, 100)$).

### 6. Termination Criteria & Computational Budget
- **Budget:** 2,000 function evaluations (20 generations $\\times$ 100 population); tolerance $10^{-5}$ for deterministic grid/scalar solvers.

---
""")

# P3
sections.append("""## Problem P3: Non-Convex Benchmark (ZDT2, $n = 10$)

### 1. Standard Mathematical Formulation
$$\\min_{x} \\quad f_1(x) = x_1$$
$$\\min_{x} \\quad f_2(x) = g(x) \\left[ 1 - \\left( \\frac{f_1(x)}{g(x)} \\right)^2 \\right]$$
$$\\text{where} \\quad g(x) = 1 + \\frac{9}{n - 1} \\sum_{i=2}^n x_i$$
$$\\text{subject to} \\quad x_i \\in [0, 1], \\quad i = 1, \\dots, 10$$

- **Objectives:** Classic continuous bi-objective benchmark with an inward-dented (concave) Pareto front.
- **Pareto Optimal Set:** $x_1^* \\in [0, 1]$ and $x_i^* = 0$ for $i = 2, \\dots, 10$ ($g(x^*) = 1$).
- **Analytical Pareto Front:** $f_2 = 1 - f_1^2$ for $f_1 \\in [0, 1]$.

### 2. Decision Variable Encoding
- **Dimension:** $D = 10$ continuous decision variables.
- **Bounds:** $x_i \\in [0.0, 1.0]$ for all $i \\in \\{1, \\dots, 10\\}$.
- **Representation:** 10-dimensional floating-point vector.

### 3. Constraint Handling
- Boundary constraint reflection/clamping: $x_i = \\text{clip}(x_i, 0.0, 1.0)$.
- Unconstrained interior manifold.

### 4. Objective Normalization & Scaling
- True front bounds: $f_1^* = [0, 1], f_2^* = [0, 1]$.
- Naturally normalized to the unit hypercube $[0, 1]^2$.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = [1.1, 1.1]$ (10% margin beyond the true nadir point of $(1.0, 1.0)$).
- Theoretical maximum Hypervolume at $r = [1.1, 1.1]$ is $\\approx 1.21 - \\int_0^1 (1 - f_1^2) df_1 = 1.21 - \\frac{2}{3} = 0.5433$ relative to nadir, or $1.21 - 0.3333 = 0.8767$ relative to origin.

### 6. Termination Criteria & Computational Budget
- **Budget:** 10,000 function evaluations (100 generations $\\times$ 100 population size) across 10 fixed random seeds.

---
""")

# P4
sections.append("""## Problem P4: Markowitz Investment Portfolio Selection (4 Assets)

### 1. Standard Mathematical Formulation
$$\\min_{w} \\quad f_1(w) = w^T \\Sigma w \\quad (\\text{Portfolio Variance / Risk})$$
$$\\max_{w} \\quad \\mu^T w \\iff \\min_{w} \\quad f_2(w) = -\\mu^T w \\quad (\\text{Negative Expected Return})$$
$$\\text{subject to} \\quad \\sum_{i=1}^4 w_i = 1, \\quad w_i \\ge 0 \\quad \\forall i \\in \\{1, \\dots, 4\\}$$

- **Parameters:**
  - Expected returns: $\\mu = [0.08, 0.12, 0.15, 0.10]^T$.
  - Covariance matrix $\\Sigma = \\begin{bmatrix} 0.010 & 0.002 & 0.001 & 0.003 \\\\ 0.002 & 0.040 & 0.006 & 0.004 \\\\ 0.001 & 0.006 & 0.090 & 0.005 \\\\ 0.003 & 0.004 & 0.005 & 0.020 \\end{bmatrix}$.

### 2. Decision Variable Encoding
- **Dimension:** $D = 4$ portfolio asset weights.
- **Bounds:** $w_i \\in [0.0, 1.0]$.
- **Simplex Manifold:** $\\Delta^3 = \\{w \\in \\mathbb{R}^4 : \\sum w_i = 1, w_i \\ge 0\\}$.

### 3. Constraint Handling
- Continuous Simplex Projection: $w = \\text{project\\_simplex}(w)$, or $L_1$ normalization $w_i = w_i / \\sum w_j$.
- If constraint enforcement mode is disabled, penalty violation is $CV = |\\sum w_i - 1|$.

### 4. Objective Normalization & Scaling
- Risk $f_1 \\in [0.006, 0.090]$; Negative return $f_2 \\in [-0.15, -0.08]$.
- Normalization scales variance by $100$ and return by $10$ to equalize gradient orders of magnitude.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = [0.10, -0.07]$ (risk up to 0.10, return down to 7%).

### 6. Termination Criteria & Computational Budget
- **Budget:** 5,000 function evaluations (50 generations $\\times$ 100 population); analytical front calculated via 300-step projected gradient descent on simplex.

---
""")

# P5
sections.append("""## Problem P5: Cantilever Beam Engineering Design (Constrained)

### 1. Standard Mathematical Formulation
$$\\min_{b, h} \\quad f_1(b, h) = \\rho L b h \\quad (\\text{Beam Mass, kg})$$
$$\\min_{b, h} \\quad f_2(b, h) = \\frac{4 P L^3}{E b h^3} \\quad (\\text{Tip Deflection, m})$$
$$\\text{subject to} \\quad g_1(b, h) = \\sigma_{\\max} - \\frac{6 P L}{b h^2} \\ge 0 \\iff b h^2 \\ge 2.4 \\times 10^{-5}$$
$$0.01 \\le b \\le 0.10 \\text{ m}, \\quad 0.02 \\le h \\le 0.20 \\text{ m}$$

- **Physical Constants:** Length $L = 1.0$ m, tip load $P = 1000$ N, Young's modulus $E = 200 \\times 10^9$ Pa, steel density $\\rho = 7850$ $\\text{kg/m}^3$, yield stress limit $\\sigma_{\\max} = 250 \\times 10^6$ Pa.
- **Active Constraint:** At optimality, $b h^2 = 2.4 \\times 10^{-5}$ is strictly active along the entire front.

### 2. Decision Variable Encoding
- **Dimension:** $D = 2$ continuous structural dimensions ($x_1 = b$ width, $x_2 = h$ height).
- **Bounds:** $b \\in [0.01, 0.10]$ m, $h \\in [0.02, 0.20]$ m.

### 3. Constraint Handling
- Deb's Feasibility-First Rules: $CV(b, h) = \\max(0, \\frac{2.4 \\times 10^{-5} - b h^2}{2.4 \\times 10^{-5}})$.
- Feasible solutions strictly dominate infeasible solutions; pairwise infeasible ranking by $CV$.

### 4. Objective Normalization & Scaling
- Mass $f_1 \\in [1.57, 157.0]$ kg; Deflection $f_2 \\in [0.0001, 0.0250]$ m.
- Normalization scales mass by $1/160.0$ and deflection by $1/0.030$.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = [160.0 \\text{ kg}, 0.030 \\text{ m}]$.

### 6. Termination Criteria & Computational Budget
- **Budget:** 10,000 function evaluations (100 generations $\\times$ 100 population) across 10 random seeds.

---
""")

# P6
sections.append("""## Problem P6: Binh–Korn Benchmark (Awkward Non-Linear Constraints)

### 1. Standard Mathematical Formulation
$$\\min_{x} \\quad f_1(x) = 4 x_1^2 + 4 x_2^2$$
$$\\min_{x} \\quad f_2(x) = (x_1 - 5)^2 + (x_2 - 5)^2$$
$$\\text{subject to} \\quad g_1(x) = (x_1 - 5)^2 + x_2^2 - 25 \\le 0$$
$$g_2(x) = 7.7 - (x_1 - 8)^2 - (x_2 + 3)^2 \\le 0$$
$$x_1 \\in [0, 5], \\quad x_2 \\in [0, 3]$$

- **Pareto Front:** Consists of two piecewise connected curves: $x_1 = x_2$ for $x_1 \\in [0, 3]$, and $x_2 = 3$ for $x_1 \\in [3, 5]$.

### 2. Decision Variable Encoding
- **Dimension:** $D = 2$ continuous variables.
- **Bounds:** $x_1 \\in [0.0, 5.0]$, $x_2 \\in [0.0, 3.0]$.

### 3. Constraint Handling
- Cumulative constraint violation: $CV(x) = \\max(0, g_1(x)) + \\max(0, g_2(x))$.
- Handled via Deb's feasibility rules during tournament selection.

### 4. Objective Normalization & Scaling
- $f_1 \\in [0, 136]$, $f_2 \\in [4, 50]$ along the feasible boundary.
- Min-max scaling onto unit interval $[0, 1]$.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = [140.0, 55.0]$.

### 6. Termination Criteria & Computational Budget
- **Budget:** 8,000 evaluations (80 generations $\\times$ 100 population).

---
""")

# P7
sections.append("""## Problem P7: Regional Water Allocation (Linear Programming Subproblem)

### 1. Standard Mathematical Formulation
$$\\max_{x} \\quad Z_1(x) = 0.8 x_1 \\quad (\\text{Agricultural Economic Yield})$$
$$\\max_{x} \\quad Z_2(x) = 1.2 x_2 \\quad (\\text{Industrial & Urban Yield})$$
$$\\iff \\min_{x} \\quad f_1(x) = -0.8 x_1, \\quad f_2(x) = -1.2 x_2$$
$$\\text{subject to} \\quad x_1 + x_2 \\le 90 \\quad (\\text{Total River Basin Capacity})$$
$$10.0 \\le x_1 \\le 70.0, \\quad 20.0 \\le x_2 \\le 80.0$$

- **Pareto Optimal Front:** Line segment $x_1 + x_2 = 90$ for $x_1 \\in [10, 70]$.
- **Objective Trade-off:** $f_2 = -1.2(90 + f_1 / 0.8) = -108 - 1.5 f_1$.

### 2. Decision Variable Encoding
- **Dimension:** $D = 2$ allocation volumes (megaliters).
- **Bounds:** $x_1 \\in [10.0, 70.0], x_2 \\in [20.0, 80.0]$.

### 3. Constraint Handling
- Total resource limit: $CV(x) = \\max(0, x_1 + x_2 - 90.0)$.
- Solved via Benson's Method (LP simplex) and Lexicographic optimization.

### 4. Objective Normalization & Scaling
- Minimized yields: $f_1 \\in [-56.0, -8.0]$, $f_2 \\in [-96.0, -24.0]$.
- Transformed by adding nadir offsets for standard non-negative Hypervolume calculation.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = [0.0, 0.0]$ in negated yield space (yields $\\ge 0$).

### 6. Termination Criteria & Computational Budget
- **Budget:** Exact solve via LP simplex in $< 0.05$ seconds; 2,500 evaluations for evolutionary algorithms.

---
""")

# P8
sections.append("""## Problem P8: Bi-Objective Delivery Routing (Discrete Permutation TSP, 15 Cities)

### 1. Standard Mathematical Formulation
$$\\min_{\\pi} \\quad f_1(\\pi) = \\sum_{k=1}^N d_{\\pi(k), \\pi(k+1)} \\quad (\\text{Total Tour Distance})$$
$$\\min_{\\pi} \\quad f_2(\\pi) = \\sum_{k=1}^N t_{\\pi(k), \\pi(k+1)} \\quad (\\text{Total Delivery Transit Time})$$
$$\\text{subject to} \\quad \\pi = (\\pi(1), \\dots, \\pi(N)) \\in \\mathcal{S}_N \\quad (\\text{Valid Hamiltonian Cycle})$$

- **Parameters:** $N = 15$ synthetic delivery depots with 2D coordinates uniformly sampled in $[0, 100]^2$. Travel times $t_{ij} = d_{ij} \\times u_{ij}$, where traffic congestion factors $u_{ij} \\sim U(0.5, 2.0)$ are symmetric.

### 2. Decision Variable Encoding
- **Dimension:** $D = 15$ integer city indices.
- **Representation:** Discrete permutation vector $\\pi \\in \\text{Perm}(0, \\dots, 14)$.

### 3. Constraint Handling
- Permutation validity strictly enforced by representation-specific operators:
  - Ant Colony Optimization: Tour construction via probabilistic transition with visited tabu lists.
  - Evolutionary Algorithms: Partially Mapped Crossover (PMX) / Order Crossover (OX) and 2-opt inversion mutations.
  - Permutation violation check: $CV(\\pi) = 15 - |\\text{unique}(\\pi)|$.

### 4. Objective Normalization & Scaling
- Distance $f_1 \\in [300, 800]$ km; Transit time $f_2 \\in [400, 1200]$ hours.
- Scaled by distance/time upper bounding baselines.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = [1000.0 \\text{ km}, 1500.0 \\text{ hrs}]$.

### 6. Termination Criteria & Computational Budget
- **Budget:** 10,000 tour evaluations (50 generations $\\times$ 200 ants/individuals) across 10 random seeds.

---
""")

# P9
sections.append("""## Problem P9: Diagnostic Feature Selection (Binary Wrapper, 30 Features)

### 1. Standard Mathematical Formulation
$$\\min_{m} \\quad f_1(m) = 1.0 - \\text{Accuracy}_{5\\text{-fold CV}}(m) \\quad (\\text{Cross-Validation Error})$$
$$\\min_{m} \\quad f_2(m) = \\frac{1}{30} \\sum_{j=1}^{30} m_j \\quad (\\text{Feature Sparsity Ratio})$$
$$\\text{subject to} \\quad \\sum_{j=1}^{30} m_j \\ge 1, \\quad m_j \\in \\{0, 1\\}$$

- **Dataset:** Wisconsin Breast Cancer (569 patient samples, 30 continuous pathological features).
- **Evaluation Engine:** Hand-coded 5-Nearest Neighbors ($k=5$) with deterministic 5-fold cross-validation.

### 2. Decision Variable Encoding
- **Dimension:** $D = 30$ binary selection flags.
- **Representation:** Boolean mask $m \\in \\{0, 1\\}^{30}$, continuous algorithms thresholded at $x_j > 0.5$.

### 3. Constraint Handling
- Non-empty selection constraint: If $\\sum m_j = 0$, $CV = 1.0$ and $f = [1.0, 1.0]$.
- Bit-flip mutation and uniform crossover for binary algorithms; continuous algorithms thresholded at $0.5$.

### 4. Objective Normalization & Scaling
- Error $f_1 \\in [0.03, 0.40]$; Feature fraction $f_2 \\in [0.033, 1.0]$.
- Both metrics naturally bounded in $[0, 1]$.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = [0.50, 1.00]$ (50% classification error, 100% feature subset).

### 6. Termination Criteria & Computational Budget
- **Budget:** 2,500 model evaluations (25 generations $\\times$ 100 population size).

---
""")

# P10
sections.append("""## Problem P10: Scalable Many-Objective Benchmark (DTLZ2, $M = 3$ and $M = 5$)

### 1. Standard Mathematical Formulation
$$\\min_{x} \\quad f_1(x) = (1 + g(x_M)) \\prod_{i=1}^{M-1} \\cos\\left(\\frac{x_i \\pi}{2}\\right)$$
$$\\min_{x} \\quad f_m(x) = (1 + g(x_M)) \\left[ \\prod_{i=1}^{M-m} \\cos\\left(\\frac{x_i \\pi}{2}\\right) \\right] \\sin\\left(\\frac{x_{M-m+1} \\pi}{2}\\right), \\quad m = 2, \\dots, M-1$$
$$\\min_{x} \\quad f_M(x) = (1 + g(x_M)) \\sin\\left(\\frac{x_1 \\pi}{2}\\right)$$
$$\\text{where} \\quad g(x_M) = \\sum_{x_j \\in x_M} (x_j - 0.5)^2, \\quad |x_M| = k = 10$$
$$\\text{subject to} \\quad x_i \\in [0, 1], \\quad i = 1, \\dots, M + k - 1$$

- **Analytical Pareto Front:** Unit sphere octant $\\sum_{m=1}^M f_m^2 = 1$ with $f_m \\ge 0$ (when $g(x_M) = 0$).

### 2. Decision Variable Encoding
- **Dimension:** $D = M + k - 1$ ($D = 12$ for $M=3$; $D = 14$ for $M=5$).
- **Bounds:** $x_i \\in [0.0, 1.0]$ for all variables.

### 3. Constraint Handling
- Unconstrained benchmark; box bounds clamped to $[0, 1]$.

### 4. Objective Normalization & Scaling
- Symmetrical spherical objectives $f_m \\in [0, 1]$ on Pareto manifold.
- Normalization unnecessary as objectives are already on identical scales.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = \\mathbf{1.5}^M$ (i.e. $[1.5, 1.5, 1.5]$ for $M=3$, and $[1.5, 1.5, 1.5, 1.5, 1.5]$ for $M=5$).

### 6. Termination Criteria & Computational Budget
- **Budget:** 15,000 function evaluations (150 generations $\\times$ 100 population size) across 10 random seeds.

---
""")

# P11
sections.append("""## Problem P11: Production Planning (Goal Programming & Lexicographic)

### 1. Standard Mathematical Formulation
$$\\max_{x_A, x_B} \\quad \\text{Profit} = 4 x_A + 6 x_B \\iff \\min \\quad f_1 = -(4 x_A + 6 x_B)$$
$$\\max_{x_A, x_B} \\quad \\text{Total Units} = x_A + x_B \\iff \\min \\quad f_2 = -(x_A + x_B)$$
$$\\text{subject to} \\quad 2 x_A + 3 x_B \\le 1200 \\quad (\\text{Machine Hours})$$
$$4 x_A + 2 x_B \\le 1600 \\quad (\\text{Labor Hours})$$
$$x_A, x_B \\ge 0$$

- **Aspiration Goals (Goal Programming):** Goal 1: Profit $\\ge 2600$; Goal 2: Total Units $\\ge 450$.
- **Priorities (Lexicographic):** Priority 1: Maximize Profit; Priority 2: Maximize Total Units subject to profit $\\ge \\text{Profit}^* - \\delta$.

### 2. Decision Variable Encoding
- **Dimension:** $D = 2$ continuous product volumes.
- **Bounds:** $x_A \\in [0.0, 500.0], x_B \\in [0.0, 500.0]$.

### 3. Constraint Handling
- Resource capacity constraints enforced as linear inequality constraints:
  $CV(x) = \\max(0, 2 x_A + 3 x_B - 1200) + \\max(0, 4 x_A + 2 x_B - 1600)$.

### 4. Objective Normalization & Scaling
- Deviation variables $d_1^-, d_2^-$ normalized by target aspiration levels: $d_1^- / 2600$, $d_2^- / 450$.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = [0.0, 0.0]$ in negated objective space (profit $\\ge 0$, units $\\ge 0$).

### 6. Termination Criteria & Computational Budget
- **Budget:** Deterministic LP / SLSQP solved in $< 0.01$ seconds; 1,000 evaluations for evolutionary algorithms.

---
""")

# P12
sections.append("""## Problem P12: Supplier Selection (Multi-Criteria Ranking Table)

### 1. Standard Decision Problem Formulation
Discrete multi-criteria evaluation of 5 candidate suppliers ($S_1, \\dots, S_5$) across 4 conflicting criteria:
1. **Cost ($C_1$):** $[120, 100, 140, 90, 130]$ \\$ (Non-beneficial / Minimize)
2. **Quality ($C_2$):** $[8.0, 7.0, 9.0, 6.0, 8.0]$ / 10 (Beneficial / Maximize)
3. **Delivery Lead Time ($C_3$):** $[10, 14, 7, 18, 9]$ days (Non-beneficial / Minimize)
4. **Reliability ($C_4$):** $[92, 88, 95, 80, 90]$ \\% (Beneficial / Maximize)
- **Stakeholder Criteria Weights:** $w = [0.30, 0.30, 0.20, 0.20]^T$.

### 2. Candidate Encoding
- **Dimension:** Discrete index set $i \\in \\{1, 2, 3, 4, 5\\}$.
- **Matrix Representation:** Decision matrix $X \\in \\mathbb{R}^{5 \\times 4}$.

### 3. Constraint Handling
- Discrete finite alternatives; no infeasible states exist.

### 4. Normalization Method
- **TOPSIS (Vector Normalization):** $r_{ij} = x_{ij} / \\sqrt{\\sum_{k=1}^5 x_{kj}^2}$.
- **VIKOR (Linear Min-Max Normalization):** Normalized distance to best criterion value $d_{ij} = (f_j^* - x_{ij}) / (f_j^* - f_j^-)$.

### 5. Decision Metrics & Reference Points
- **Positive-Ideal Solution (PIS):** Best values across criteria ($[90\\$, 9.0, 7\\text{d}, 95\\%]$).
- **Negative-Ideal Solution (NIS):** Worst values across criteria ($[140\\$, 6.0, 18\\text{d}, 80\\%]$).

### 6. Termination Criteria & Computational Budget
- **Budget:** Deterministic vector matrix operations; execution time $< 0.005$ seconds.

---
""")

# P13
sections.append("""## Problem P13: Multi-Task Learning / Differentiable Multi-Objective Optimization

### 1. Standard Mathematical Formulation
$$\\min_{x} \\quad f_1(x) = \\|x - a\\|_2^2 = (x_1 - 3)^2 + (x_2 - 3)^2$$
$$\\min_{x} \\quad f_2(x) = \\|x - b\\|_2^2 = (x_1 + 3)^2 + (x_2 + 3)^2$$
$$\\text{where} \\quad a = [3.0, 3.0]^T, \\quad b = [-3.0, -3.0]^T$$
$$\\text{subject to} \\quad x \\in \\mathbb{R}^2$$

- **Exact Analytical Pareto Set:** Line segment connecting $a$ and $b$: $x_1 = x_2$ with $x_1 \\in [-3.0, 3.0]$.
- **Analytical Task Gradients:** $\\nabla f_1(x) = 2(x - a)$, $\\nabla f_2(x) = 2(x - b)$.

### 2. Decision Variable Encoding
- **Dimension:** $D = 2$ continuous parameters representing shared neural network weights.
- **Bounds:** Unconstrained continuous space ($x \\in [-10.0, 10.0]^2$).

### 3. Constraint Handling
- Unconstrained continuous vector space.

### 4. Gradient Normalization
- MGDA computes the minimum-norm convex combination: $\\min_{\\alpha \\in \\Delta^1} \\|\\alpha \\nabla f_1 + (1 - \\alpha) \\nabla f_2\\|^2$.
- GradNorm balances loss weights $w_m(t)$ by dynamically equalizing gradient norms $\\|\\nabla_W (w_m \\mathcal{L}_m)\\|$ to the relative inverse training rate.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = [1.1, 1.1]$ in normalized loss space; $r = [75.0, 75.0]$ in raw quadratic loss space.

### 6. Termination Criteria & Computational Budget
- **Budget:** 100 gradient descent steps with learning rate $\\eta = 0.05$; execution time $< 0.05$ seconds.

---
""")

# P14
sections.append("""## Problem P14: Laptop Engineering Design with Decision-Maker Preference

### 1. Standard Mathematical Formulation
$$\\min_{x_1, x_2} \\quad f_1(x_1, x_2) = 0.4 + 0.05 x_1 + 0.012 x_2 \\quad (\\text{Total Laptop Weight, kg})$$
$$\\max_{x_1, x_2} \\quad \\text{Runtime}(x_1, x_2) = \\frac{0.2 x_2}{0.5 + 0.03 x_1} \\iff \\min \\quad f_2 = -\\text{Runtime} \\quad (\\text{Negative Battery Life, hrs})$$
$$\\text{subject to} \\quad 13.0 \\le x_1 \\le 17.0 \\text{ inches} \\quad (\\text{Screen Size})$$
$$40.0 \\le x_2 \\le 90.0 \\text{ Watt-hours} \\quad (\\text{Battery Capacity})$$

- **Pareto Optimal Set:** $x_1^* = 13.0$ inches (smallest screen minimizes display weight and reduces power consumption), $x_2^* \\in [40.0, 90.0]$ Wh.
- **Decision-Maker Preference Target:** Aspiration point $z^{pref} = [1.8 \\text{ kg}, 12.0 \\text{ hours}]$.

### 2. Decision Variable Encoding
- **Dimension:** $D = 2$ continuous design parameters.
- **Bounds:** Screen size $x_1 \\in [13.0, 17.0]$, Battery capacity $x_2 \\in [40.0, 90.0]$.

### 3. Constraint Handling
- Box bounds clamped to valid hardware manufacturing limits.

### 4. Objective Normalization & Scaling
- Weight $f_1 \\in [1.53, 2.33]$ kg; Runtime $f_2 \\in [-20.2, -7.9]$ hours.
- Scaled by $1/2.5$ for weight and $1/25.0$ for runtime.

### 5. Hypervolume Reference Point
- **Reference Point:** $r = [2.50 \\text{ kg}, -5.00 \\text{ hrs}]$ (Weight $\\le 2.5$ kg, Runtime $\\ge 5$ hrs).

### 6. Termination Criteria & Computational Budget
- **Budget:** 5,000 function evaluations (50 generations $\\times$ 100 population size); R-NSGA-II uses $\\varepsilon$-clearing radius $\\varepsilon_{clear} = 0.15$.
""")

content = "".join(sections)
with open('report/problem_bank.md', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Created report/problem_bank.md successfully ({len(content)} bytes)!")

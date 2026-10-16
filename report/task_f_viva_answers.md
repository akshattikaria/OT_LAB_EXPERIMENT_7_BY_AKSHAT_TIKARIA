# Task F — Viva and Code Walkthrough Preparation

## Comprehensive Answers to the 8 Core Examination Questions

---

### Question 1: Why can Weighted Sum never reach a dented part of a non-convex front?
**Answer:**  
In the Weighted Sum method, the scalar subproblem minimizes $F_w(x) = w_1 f_1(x) + w_2 f_2(x)$ (for $w_1, w_2 \ge 0, w_1 + w_2 = 1$). Geometrically, contour lines of $F_w$ are straight lines of slope $-w_1 / w_2$. Minimizing $F_w$ corresponds to translating a supporting hyperplane from $-\infty$ towards the feasible objective region until it first touches the boundary:
- If the Pareto front is **convex** (curves outward towards the ideal point), any boundary point can be made the unique tangent point of contact for some supporting hyperplane slope $-w_1/w_2$.
- If the Pareto front is **non-convex / concave (dented inward)**, any supporting hyperplane touching an interior dented point *must cut through other feasible points* on the flanks. Therefore, the minimum of the linear function along that direction will always occur at one of the outer extremities or boundary vertices, never inside the dent.
- *Mathematical consequence:* No non-negative weight vector $(w_1, w_2)$ exists whose minimum is situated inside the dented curve. Sweeping weights causes the solution to jump discontinuously from one end of the dent to the other (as demonstrated on Problem P3, ZDT2).

---

### Question 2: Why did VEGA collapse towards the extremes in your run?
**Answer:**  
VEGA (Vector Evaluated Genetic Algorithm) partitions the population of size $N$ into $M$ sub-populations of size $N/M$. Each sub-population $m$ is selected based *exclusively on objective $f_m$*:
1. Individuals that excel in objective 1 (regardless of how poor they are in objective 2) dominate sub-population 1. Similarly, individuals that excel in objective 2 dominate sub-population 2.
2. When the sub-populations are pooled and shuffled for crossover, the parents represent opposing single-objective extrema.
3. Because selection pressure is exerted purely in the coordinate axes directions, solutions that offer balanced compromises (moderate in both objectives) are never among the best in either sub-population and are systematically filtered out.
4. Over successive generations, this creates strong speciation towards the boundary extrema, leaving the intermediate compromise region completely unpopulated (verified on P2 and P3 in Task A).

---

### Question 3: What does elitism guarantee, and which of your algorithms have it?
**Answer:**  
- **What Elitism Guarantees:** Elitism guarantees **monotonic non-deterioration** of the Pareto-optimal set. Mathematically, if an elite archive or $(N+N)$ selection model is used, a non-dominated solution cannot be displaced or replaced by a strictly inferior or dominated solution. This prevents the evolutionary population from losing high-quality discoveries due to stochastic genetic drift or destructive crossover/mutation.
- **Algorithms with Elitism:**
  - *Generational $(N+N)$ Evolutionary Elitism:* NSGA-II (14), NSGA-III (15), Constrained NSGA-II (21), MODE (27), MO-TLBO (28), MOMA (30), RVEA (31), IBEA (32), HypE (34), R-NSGA-II (35).
  - *External Archive-based Elitism:* SPEA2 (16), PESA-II (17), MOEA/D (18), MOEA/D-DE (19), MOEA/D-ACO (20), MOPSO (22), MOGWO (23), MOABC (24), MOACO (25), MOSA (26), MOBA (29), SMS-EMOA (33).
  - *Non-Elitist Algorithms:* VEGA (10), MOGA (11), NSGA (12), NPGA (13) replace parents with offspring without archiving, meaning solutions can drift or be lost.

---

### Question 4: Why does crowding distance stop working well with many objectives, and what replaces it?
**Answer:**  
1. **Failure of Crowding Distance in High Dimensions ($M \ge 5$):**
   - *Dominance Resistance:* In $M \ge 5$ dimensions, the probability that two random solutions dominate each other approaches zero. Almost 100% of the population becomes mutually non-dominated (Front 1).
   - *Boundary Distortion:* Crowding distance assigns $\infty$ to the extreme points along each objective axis. In $M$ dimensions, there are at least $2M$ extreme boundary points. As $M$ increases, these boundary points consume a dominant fraction of the population, leaving interior trade-off solutions with negligible survival priority.
   - *Sparsity in Hyper-volume:* Points become sparse in high-dimensional space, and 1D perimeter projections (summing $\frac{f_m(i+1) - f_m(i-1)}{f_m^{\max} - f_m^{\min}}$) no longer accurately measure local hyper-spatial density.
2. **What Replaces It:**
   - **Structured Reference Points / Directions:** Used in NSGA-III (Das & Dennis simplex lattice) and RVEA (unit reference vectors). Solutions are projected perpendicular to reference lines, preserving uniform distribution across the manifold.
   - **Sample-Shared Hypervolume Contributions:** Used in HypE and SMS-EMOA, measuring exact or Monte Carlo estimated hypervolume contributions rather than linear distances.

---

### Question 5: Which of your algorithms needs a preference from the decision-maker, and what happens if it is wrong?
**Answer:**  
- **Algorithms Requiring Decision-Maker Preferences:**
  - Value Function (Algorithm 5): Requires utility parameters (risk aversion $\lambda$, attribute weights, target thresholds).
  - Goal Programming (Algorithm 6): Requires aspiration targets $T_m$ and deviation penalties $w_m^+, w_m^-$.
  - Lexicographic Method (Algorithm 7): Requires strict priority order and tolerances $\delta$.
  - TOPSIS & VIKOR (Algorithms 8 & 9): Require criteria weights $w_j$ and criteria classifications.
  - R-NSGA-II (Algorithm 35): Requires target reference points $z^{ref}$ and clearing radius $\varepsilon_{clear}$.
- **Consequences of Erroneous / Misplaced Preferences:**
  - In *Value Function*, a distorted utility function steers optimization to an unviable compromise that violates the decision-maker's actual risk appetite.
  - In *Goal Programming*, setting unreachable targets can leave severe shortfalls, whereas underestimating targets stops the search at suboptimal points far inside the feasible space.
  - In *R-NSGA-II*, placing $z^{ref}$ far outside the achievable region distorts the local search, creating clustering at boundary edges and discarding globally efficient trade-offs.

---

### Question 6: Explain Deb's three constraint rules with one numerical example.
**Answer:**  
**Deb's Constrained-Dominance Rules:**
A solution $A$ constrained-dominates solution $B$ if:
1. Solution $A$ is feasible ($CV_A = 0$) and Solution $B$ is infeasible ($CV_B > 0$).
2. Both $A$ and $B$ are infeasible, but $A$ has smaller total constraint violation: $CV_A < CV_B$.
3. Both $A$ and $B$ are feasible, and $A$ Pareto-dominates $B$ in objective space: $\forall m, f_m(A) \le f_m(B)$ and $\exists m, f_m(A) < f_m(B)$.

**Numerical Example (Cantilever Beam Design, Problem P5):**
Consider four candidate designs:
- Point $A$: $(b=0.02, h=0.06) \implies f = [9.42\text{ kg}, 0.0093\text{ m}]$, $b h^2 = 7.2 \times 10^{-5} \ge 2.4 \times 10^{-5} \implies \mathbf{CV_A = 0.0}$ (Feasible).
- Point $B$: $(b=0.01, h=0.05) \implies f = [3.925\text{ kg}, 0.0160\text{ m}]$, $b h^2 = 2.5 \times 10^{-5} \ge 2.4 \times 10^{-5} \implies \mathbf{CV_B = 0.0}$ (Feasible).
- Point $C$: $(b=0.01, h=0.03) \implies f = [2.355\text{ kg}, 0.0740\text{ m}]$, $b h^2 = 9.0 \times 10^{-6} < 2.4 \times 10^{-5} \implies \mathbf{CV_C = 0.625}$ (Infeasible).
- Point $D$: $(b=0.01, h=0.02) \implies f = [1.570\text{ kg}, 0.2500\text{ m}]$, $b h^2 = 4.0 \times 10^{-6} < 2.4 \times 10^{-5} \implies \mathbf{CV_D = 0.833}$ (Infeasible).

**Rule Application:**
- *Rule 1 Check:* Compare $A$ (feasible, mass 9.42) vs $C$ (infeasible, mass 2.355). Even though $C$ has much lower mass, **$A$ strictly dominates $C$** because $A$ is feasible and $C$ violates stress limits.
- *Rule 2 Check:* Compare $C$ ($CV = 0.625$) vs $D$ ($CV = 0.833$). Both are infeasible, but **$C$ strictly dominates $D$** because $CV_C < CV_D$.
- *Rule 3 Check:* Compare $A$ vs $B$. Both are feasible: $B$ has lower mass (3.925 < 9.42) while $A$ has lower deflection (0.0093 < 0.016). Neither Pareto-dominates the other; they are mutually non-dominated trade-offs.

---

### Question 7: How would you detect that your algorithm has prematurely converged?
**Answer:**  
Premature convergence can be detected through multiple quantitative signals monitored across generations:
1. **Hypervolume Stagnation:** The Hypervolume (HV) curve plateaus early ($d(\text{HV})/dt \approx 0$) at a value significantly below the known theoretical ceiling or baseline.
2. **Genotypic & Phenotypic Variance Collapse:** The population standard deviation in decision space ($\sigma_x \to 0$) drops to machine precision, meaning all individuals in the population have collapsed into an identical clone.
3. **Severe Spacing Metric Spikes:** The Spacing metric $S$ drops to zero (if all points collapse to a single point) or blows up to large values as intermediate solutions vanish and only isolated clusters remain.
4. **Dominance Replacement Ratio:** In algorithms like MODE or MOEA/D, the percentage of newly generated offspring that successfully replace their targets drops to zero across multiple consecutive generations.
5. **IGD Divergence from Reference:** On benchmarks with known Pareto fronts, the Inverted Generational Distance (IGD) remains high and fails to decrease as generation count increases.

---

### Question 8: Why do TOPSIS and VIKOR not count as optimizers?
**Answer:**  
- **Definition of an Optimizer:** An optimizer explores, searches, and generates *new* candidate decision vectors $x$ across a continuous or discrete decision space $\mathcal{X}$ to minimize or maximize objective functions subject to constraints.
- **Role of TOPSIS and VIKOR:** TOPSIS and VIKOR are **Multi-Criteria Decision Making (MCDM) Ranking Methods**, not optimizers:
  1. *Finite Candidate Set:* They require a pre-existing, static, finite table of alternatives (e.g., 5 suppliers in P12, or the 100 solutions of an already converged Pareto front).
  2. *No Generative Exploration:* They possess no variation operators (no crossover, mutation, gradient descent, or neighborhood moves) and cannot synthesize new solutions.
  3. *Post-Processing Function:* Their sole mathematical function is to sort and order an existing set of candidate rows based on geometric distance (TOPSIS) or $L_p$ compromise ranking (VIKOR).

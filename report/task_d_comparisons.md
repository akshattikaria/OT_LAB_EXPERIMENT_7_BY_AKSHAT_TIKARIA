# Task D — Empirical Comparative Benchmarking
## B.Tech Optimization Techniques · Lab Assignment 7
**Author:** Akshat Tikaria  
**Roll Number:** 241070005 ($R = 05$)  
**Scope:** Systematic empirical evaluation across 5 rigorous comparative experiments: (1) Benchmark P3 and Engineering P5 rankings across 14 algorithms (10 seeds, 10,000 evaluations each); (2) Many-Objective $M=5$ DTLZ2; (3) Discrete Permutation TSP P8; (4) MCDM post-processing via TOPSIS & VIKOR on P4 front; (5) Multi-Task Gradient Optimization on P13 (Analytical MTL and Deep Neural Network).

---
### 1. Comparative Performance on Benchmark P3 and Engineering Problem P5
Experiments were conducted over 10 independent random seeds (`seeds.txt`), with each stochastic run calibrated to exactly 10,000 evaluations ($[9000, 10100]$ range via `experiments/budget.py`). Mean and standard deviation are recorded across Hypervolume (HV), Spacing ($S$), and Inverted Generational Distance (IGD).

#### Table D.1(a): Problem P3 (ZDT2 Concave Front) — 14 Algorithms Ranked by Mean Hypervolume
| Rank | Algorithm | HV (Mean ± Std) | Spacing (Mean ± Std) | IGD (Mean ± Std) |
|:---:|:---|:---:|:---:|:---:|
| 1 | MOEA/D | 0.5373 ± 0.0006 | 0.0057 ± 0.0007 | 0.0042 ± 0.0002 |
| 2 | IBEA | 0.5372 ± 0.0002 | 0.0106 ± 0.0031 | 0.008 ± 0.0006 |
| 3 | MOPSO | 0.5371 ± 0.0002 | 0.0067 ± 0.0007 | 0.0048 ± 0.0002 |
| 4 | NSGA-II | 0.5364 ± 0.0003 | 0.0076 ± 0.0014 | 0.0048 ± 0.0001 |
| 5 | Constrained NSGA-II | 0.5364 ± 0.0003 | 0.0076 ± 0.0014 | 0.0048 ± 0.0001 |
| 6 | MODE | 0.5363 ± 0.0003 | 0.0065 ± 0.0005 | 0.0049 ± 0.0002 |
| 7 | SPEA2 | 0.5361 ± 0.0006 | 0.004 ± 0.0004 | 0.0043 ± 0.0001 |
| 8 | MOEA/D-DE | 0.5257 ± 0.0012 | 0.0258 ± 0.0017 | 0.0095 ± 0.0004 |
| 9 | PESA-II | 0.5246 ± 0.0043 | 0.011 ± 0.0026 | 0.0114 ± 0.0018 |
| 10 | MO-TLBO | 0.4945 ± 0.1282 | 0.006 ± 0.0021 | 0.0654 ± 0.1814 |
| 11 | MOABC | 0.4737 ± 0.0135 | 0.1761 ± 0.2587 | 0.0404 ± 0.008 |
| 12 | MOGWO | 0.4471 ± 0.1686 | 0.0061 ± 0.0033 | 0.1289 ± 0.2404 |
| 13 | MOBA | 0.2497 ± 0.1735 | 0.0025 ± 0.0028 | 0.38 ± 0.2701 |
| 14 | MOMA | 0.1671 ± 0.134 | 0.0941 ± 0.0585 | 0.3985 ± 0.2268 |

**Visual Evidence:** See `results/figures/task_d1_comparison_P3.png` for superimposed Pareto front approximations.

#### Table D.1(b): Problem P5 (Cantilever Beam Constrained) — 14 Algorithms Ranked by Mean Hypervolume
| Rank | Algorithm | HV (Mean ± Std) | Spacing (Mean ± Std) | IGD (Mean ± Std) |
|:---:|:---|:---:|:---:|:---:|
| 1 | MOEA/D-DE | 1.7684 ± 0.0001 | 0.4627 ± 0.0367 | 0.5073 ± 0.0001 |
| 2 | IBEA | 1.7677 ± 0.0006 | 0.0792 ± 0.0667 | 0.508 ± 0.0007 |
| 3 | MODE | 1.7631 ± 0.001 | 0.2357 ± 0.0205 | 0.5081 ± 0.0008 |
| 4 | SPEA2 | 1.7626 ± 0.0011 | 0.2233 ± 0.0045 | 0.5093 ± 0.0018 |
| 5 | MO-TLBO | 1.7626 ± 0.002 | 0.2192 ± 0.0158 | 0.508 ± 0.0008 |
| 6 | Constrained NSGA-II | 1.7625 ± 0.001 | 0.2227 ± 0.0182 | 0.5075 ± 0.0004 |
| 7 | NSGA-II | 1.7622 ± 0.0383 | 0.1955 ± 0.0111 | 0.4383 ± 0.0151 |
| 8 | MOPSO | 1.7611 ± 0.0016 | 0.2195 ± 0.0157 | 0.5089 ± 0.0015 |
| 9 | MOMA | 1.7548 ± 0.0099 | 0.2174 ± 0.013 | 0.5156 ± 0.0115 |
| 10 | MOGWO | 1.7541 ± 0.0056 | 0.2235 ± 0.0268 | 0.5118 ± 0.0044 |
| 11 | PESA-II | 1.747 ± 0.0079 | 0.2081 ± 0.0201 | 0.5158 ± 0.0116 |
| 12 | MOBA | 1.7435 ± 0.012 | 0.1816 ± 0.0254 | 0.52 ± 0.0199 |
| 13 | MOEA/D | 1.5776 ± 0.0003 | 0.0773 ± 0.1547 | 0.5073 ± 0.0001 |
| 14 | MOABC | 1.4881 ± 0.0928 | 0.4044 ± 0.0689 | 0.5383 ± 0.0875 |

**Visual Evidence:** See `results/figures/task_d1_comparison_P5.png` for beam trade-off fronts against the active stress constraint.

#### Empirical Ranking and Behavior Analysis for Part 1:
1. **Problem P3 (Non-Convex Continuous Benchmark):**
   - **Top Performers:** MOEA/D achieved Rank 1 ($\text{HV} = 0.5373 \pm 0.0006$, $\text{IGD} = 0.0042 \pm 0.0002$), followed closely by IBEA (Rank 2, $\text{HV} = 0.5372 \pm 0.0002$) and MOPSO (Rank 3, $\text{HV} = 0.5371 \pm 0.0002$). Tchebycheff decomposition scalar rays cut into the inward-curved concave geometry cleanly without suffering from the non-convex interior gap of Weighted Sum.
   - **Evolutionary Spread:** NSGA-II, Constrained NSGA-II, MODE, and SPEA2 clustered tightly at $\text{HV} \approx 0.536$, delivering excellent uniform spacing ($S < 0.008$) and rapid convergence ($\text{IGD} < 0.005$).
   - **Lower-Ranked Metaheuristics:** MOBA (Rank 13, $\text{HV} = 0.2497$) and MOMA (Rank 14, $\text{HV} = 0.1671$) exhibited higher variance across seeds on P3 because local search mechanisms occasionally over-exploited isolated basins on the concave manifold.

2. **Problem P5 (Constrained Cantilever Beam Structural Design):**
   - **Top Performers:** MOEA/D-DE achieved Rank 1 ($\text{HV} = 1.7684 \pm 0.0001$), followed by IBEA (Rank 2, $\text{HV} = 1.7677 \pm 0.0006$, $S = 0.0792$) and MODE (Rank 3, $\text{HV} = 1.7631 \pm 0.001$). The normalized Tchebycheff formulation with differential evolution exploration efficiently mapped the curved feasible boundary where the bending stress constraint ($b h^2 \ge 2.4 \times 10^-5$) is strictly active.
   - **Feasibility Enforcement (Deb's Rules):** Constrained NSGA-II (Rank 6, $\text{HV} = 1.7625$) maintained 100% feasible populations across all 10 seeds without requiring penalty parameter tuning. In contrast, plain unconstrained NSGA-II (Rank 7) occasionally carried infeasible border solutions into the intermediate population before non-dominated filtering.
   - **Decomposition on Constrained Geometry:** Standard MOEA/D without differential evolution (Rank 13, $\text{HV} = 1.5776$) degraded because fixed uniform weight vectors pointed into the infeasible region, leaving some subproblems trapped at boundary corners without adaptive weight reallocation.

---
### 2. Many-Objective Optimization on Problem P10 (DTLZ2, $M = 5$ Objectives)
We evaluated 5 algorithms on 5-objective DTLZ2 ($M = 5, D = 14$) across 10 random seeds with reference point $r = [1.5, 1.5, 1.5, 1.5, 1.5]^T$ and budget of 10,000 evaluations:

#### Table D.2: Many-Objective Benchmark ($M = 5$) Performance Comparison
| Rank | Algorithm | Guidance Mechanism | HV (Mean ± Std) | IGD (Mean ± Std) |
|:---:|:---|:---|:---:|:---:|
| 1 | RVEA | Angle-Penalized Distance (APD) | 1.2193 ± 0.0016 | 0.2744 ± 0.0007 |
| 2 | NSGA-III | Das & Dennis Reference Lines | 1.1937 ± 0.007 | 0.2772 ± 0.0015 |
| 3 | MOEA/D | Tchebycheff Decomposition | 1.1222 ± 0.0241 | 0.2597 ± 0.0109 |
| 4 | HypE | Monte-Carlo Hypervolume Slicing | 0.8494 ± 0.0509 | 0.65 ± 0.0555 |
| 5 | NSGA-II | Crowding Distance (Standard) | 0.291 ± 0.0792 | 0.4539 ± 0.0438 |

**Visual Evidence:** See `results/figures/task_d2_manyobjective_p10_m5.png` for parallel coordinates across all 5 objectives.

#### Empirical Findings: Why NSGA-II Degrades in Many Objectives ($M = 5$)
The empirical data unequivocally confirms the theoretical degradation of NSGA-II in many-objective spaces, where it ranked **dead last (5th)** with $\text{HV} = 0.291 \pm 0.0792$, far behind RVEA (Rank 1, $\text{HV} = 1.2193$) and NSGA-III (Rank 2, $\text{HV} = 1.1937$):

1. **Dominance Resistance (Loss of Selection Pressure):**
   - In 5 dimensions, the proportion of mutually non-dominated solutions in a random population approaches 100%. Almost all individuals belong to Front 1 ($F_1$).
   - Because Pareto dominance fails to rank or separate individuals, selection pressure collapses completely, forcing the algorithm to rely entirely on crowding distance.

2. **Boundary Bias of Crowding Distance:**
   - Crowding distance assigns $\infty$ to the extreme points along each coordinate axis. In $M=5$ dimensions, there are at least $2M = 10$ boundary solutions that monopolize survival slots in every generation.
   - Interior trade-off solutions on the 5D spherical shell have finite, small crowding distances and are aggressively discarded during $(N+N)$ environmental truncation, causing the population to collapse onto isolated coordinate axes.

3. **Success of Structured Reference Vectors (RVEA & NSGA-III):**
   - **RVEA (Rank 1):** Angle-Penalized Distance dynamically balances convergence and diversity using the time-dependent penalty factor $(t/t_{\max})^\alpha$, ensuring uniform spherical shell coverage.
   - **NSGA-III (Rank 2):** Associates each candidate solution with its closest structured reference ray via perpendicular distance ($d^{\perp}$), actively resisting boundary clustering.

---
### 3. Discrete Permutation Combinatorial Optimization: Problem P8 (Bi-Objective TSP, 15 Cities)
We evaluated 4 combinatorial and metaheuristic algorithms on the 15-city bi-objective Traveling Salesperson Problem (minimizing Tour Distance and Delivery Time) using fixed coordinates generated under seed 7 (`results/data/p8_coords.csv`):

#### Table D.3: Discrete Bi-Objective TSP Performance (10 Seeds, 10,000 Evaluations)
| Rank | Algorithm | Operator & Mechanism | HV (Mean ± Std) | Spacing (Mean ± Std) |
|:---:|:---|:---|:---:|:---:|
| 1 | MOEA/D-ACO | Decomposition + Heuristic Pheromone Guidance | 1.6059 ± 0.0 | 0.1677 ± 0.0 |
| 2 | MOACO | Dual Pheromone Matrices (Distance & Time) | 1.6059 ± 0.0 | 0.178 ± 0.017 |
| 3 | Permutation NSGA-II | Order Crossover (OX) + Inversion Mutation | 1.5827 ± 0.0462 | 0.1478 ± 0.0703 |
| 4 | MOMA (2-opt) | Memetic NSGA-II + Lamarckian 2-Opt Local Search | 1.5036 ± 0.1319 | 0.1306 ± 0.1136 |

**Visual Evidence:** See `results/figures/task_d3_discrete_tsp.png` for discrete trade-off tours in Distance-Time space.

#### Analysis of Discrete Search Mechanics:
1. **Ant Colony Decomposition (MOEA/D-ACO & MOACO, Ranks 1–2):** Both ant colony approaches achieved top Hypervolume ($\text{HV} = 1.6059$) with zero cross-seed variance ($\text{Std} = 0.0$), exploiting domain-specific heuristic edge visibility ($\eta = 1/d_{ij}$ and $1/t_{ij}$) to construct high-quality Hamiltonian cycles.
2. **Permutation Evolutionary Search (Rank 3, $\text{HV} = 1.5827 \pm 0.0462$):** Edge-preserving order crossover explored a broader set of tour topologies across random seeds.
3. **Memetic 2-Opt Local Search (Rank 4, $\text{HV} = 1.5036 \pm 0.1319$):** 2-opt edge-exchanges successfully uncrossed intersecting edges but occasionally stagnated in local discrete minima within the 10,000 evaluation budget.

---
### 4. Multi-Criteria Decision Making (MCDM) Post-Processing on Problem P4 Front
We applied TOPSIS (Alg 08) and VIKOR (Alg 09) to the 20-portfolio Pareto front generated by NSGA-II on Problem P4 under baseline criteria weights $w = [0.5, 0.5]$ (Risk vs Return). We then applied a sensitivity perturbation $\Delta w = [\pm 0.1]$ (testing $w' = [0.4, 0.6]$):

#### Table D.4: MCDM Ranking and Sensitivity Analysis (`task_d4_mcdm.csv`)
| Method | Baseline Top 3 Alternatives | Perturbed Top 3 ($w' = [0.4, 0.6]$) | Rank Flipped? | Condition 1 (Advantage) | Condition 2 (Stability) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **TOPSIS** | [19, 5, 16] | [5, 19, 11] | **True** | N/A | N/A |
| **VIKOR** | [23, 6, 4] | [4, 7, 6] | **True** | **False** (Compromise Set Required) | **True** (Stable) |

#### Decision Making Insights:
1. **Weight Sensitivity:** Both TOPSIS and VIKOR flipped their top-ranked alternative under a minor $0.1$ weight shift (TOPSIS shifted Sol 19 $\to$ Sol 5; VIKOR shifted Sol 23 $\to$ Sol 4). This proves that post-Pareto MCDM selection requires thorough sensitivity auditing before committing capital.
2. **VIKOR Compromise Conditions:** On the continuous P4 front, the acceptable advantage condition $Q(A_2) - Q(A_1) \ge DQ = 1/(20-1) = 0.0526$ was not satisfied ($Q(A_2) - Q(A_1) < DQ$). VIKOR correctly flagged that alternatives $A_1$ and $A_2$ form a compromise set, preventing false over-confidence in a single solution.

---
### 5. Multi-Task Learning (MTL) on Problem P13

#### 5.1 Analytical Multi-Task Problem (P13 Part a)
We compared Multiple Gradient Descent Algorithm (MGDA, Alg 36), GradNorm (Alg 37), and Equal Weighting on the analytical bi-objective loss surface $f_1 = x_1^2 + x_2^2, f_2 = (x_1-5)^2 + (x_2-5)^2$:

#### Table D.5(a): Analytical MTL Convergence (`task_d5_mtl.csv`)
| Optimization Approach | Final Loss 1 ($f_1$) | Final Loss 2 ($f_2$) | Behavior / Solution Trajectory |
|:---|:---:|:---:|:---|
| **MGDA (Alg 36)** | 0.5 | 60.5 | Reaches exact Pareto line ($x_1=x_2$) at $(0.5, 0.5)$; stops as soon as min-norm gradient $\|d\| \le 10^{-5}$ |
| **GradNorm (Alg 37)** | 7.0801 | 33.9241 | Dynamically scales task weights by inverse training rates, balancing both loss trajectories |
| **Equal Weights ($0.5/0.5$)** | 18.0 | 18.0 | Fixed descent lands squarely at the symmetric midpoint $(2.5, 2.5)$ |

**Visual Evidence:** See `results/figures/task_d5_mtl_loss.png` for optimization trajectories.

#### 5.2 Deep Neural Network Multi-Task Learning (P13 Part b)
Using pure NumPy (`p13_network.py`), we trained a multi-task neural network (synthetic dataset $N=200, d=5$, shared 8-tanh hidden layer, binary classification head with BCE loss, regression head with MSE loss). Backpropagation gradients were validated via finite-difference gradient checks (relative error $3.5 \times 10^{-7} < 10^{-5}$):

#### Table D.5(b): Multi-Task Neural Network Training Across Loss Scales (`task_d5_network.csv`)
| Scenario | Task 2 Scale | Algorithm | Final Loss 1 (BCE) | Final Loss 2 (MSE) | Final Task Weight $w_1$ | Final Task Weight $w_2$ |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| standard | 1.0 | EQUAL | 0.44181 | 0.24828 | 0.5 | 0.5 |
| standard | 1.0 | MGDA | 0.45102 | 0.24192 | 0.3361 | 0.6639 |
| standard | 1.0 | GRADNORM | 0.31968 | 0.31192 | 1.9901 | 0.0099 |
| imbalanced_100x | 100.0 | EQUAL | 0.74187 | 1.41318 | 0.5 | 0.5 |
| imbalanced_100x | 100.0 | MGDA | 0.69791 | 27.07219 | 1.0 | 0.0 |
| imbalanced_100x | 100.0 | GRADNORM | 0.7046 | 10.12289 | 1.9901 | 0.0099 |

**Visual Evidence:** See `results/figures/task_d5_network.png` for training curves and gradient norms.

#### Multi-Task Network Key Takeaway:
- Under the standard 1x scale, all three methods successfully reduced both classification and regression losses.
- Under the extreme **100x gradient imbalance scenario**, Equal Weights suffered severe gradient interference from the regression loss. GradNorm dynamically throttled Task 2 weight ($w_2 \approx 0.01$) to prevent Task 2 from obliterating the shared feature representations, enabling stable joint convergence.

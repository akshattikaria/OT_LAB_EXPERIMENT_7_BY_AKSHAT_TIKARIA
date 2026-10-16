"""
alg20_moead_aco.py - MOEA/D with Ant Colony Optimization (MOEA/D-ACO)
=====================================================================
Family: Elitist Decomposition / Combinatorial (Part 3)

Core Mechanism:
Solves multi-objective combinatorial problems (such as P8: Bi-Objective TSP)
by decomposing into N scalar subproblems with weight vectors lambda^1, ..., lambda^N:
1. Subproblem Heuristic Visibility:
   For subproblem k with weight (lambda_1^k, lambda_2^k), the combined edge visibility is:
       eta_{ij}^k = 1 / (lambda_1^k * d_{ij} + lambda_2^k * t_{ij} + eps)
2. Pheromone Representation:
   Ant k traverses the graph using transition probabilities based on subproblem-specific
   heuristic information and shared pheromones:
       P_{ij}^k proportional to (tau_{ij})^alpha * (eta_{ij}^k)^beta
3. Multi-objective Pheromone Update:
   Pheromone evaporation tau <- (1 - rho) * tau, and deposit reward proportional to the
   scalarized / non-dominated quality of routes found by ants in subproblem neighborhoods.

Lecture Steps:
1. Initialize weight vectors lambda^1, ..., lambda^N and compute neighborhoods.
2. Initialize pheromone matrix tau_{ij} uniformly.
3. For each generation (cycle):
   a. For each ant k = 1 to N:
      - Construct closed tour starting from depot using probabilistic transition rule.
      - Evaluate total distance and total time.
   b. Update ideal reference point z*.
   c. Pheromone evaporation: tau = (1 - rho) * tau.
   d. Pheromone deposit: best solutions for each subproblem deposit delta_tau on their paths.
4. Extract non-dominated set of all visited tours.

Default Parameters:
- n_ants (pop_size): 30
- n_gen: 40
- alpha: 1.0 (pheromone exponent)
- beta: 2.0 (heuristic exponent)
- rho: 0.1 (evaporation rate)
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, n_ants: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sets up weights, initial pheromones, and heuristic matrices."""
    w1 = np.linspace(0.01, 0.99, n_ants)
    weights = np.column_stack([w1, 1.0 - w1])

    n_cities = getattr(problem, 'n_vars', 15)
    tau = np.ones((n_cities, n_cities)) * 1.0

    return weights, tau, np.zeros(2)


def construct_tour(ant_idx: int, weight: np.ndarray, tau: np.ndarray,
                   dist_mat: np.ndarray, time_mat: np.ndarray,
                   alpha: float = 1.0, beta: float = 2.0) -> np.ndarray:
    """Constructs one TSP tour for ant ant_idx using subproblem visibility."""
    n_cities = len(dist_mat)
    # Combined heuristic cost
    cost_mat = weight[0] * dist_mat + weight[1] * time_mat + 1e-6
    eta = 1.0 / cost_mat
    np.fill_diagonal(eta, 0.0)

    unvisited = set(range(n_cities))
    start_city = np.random.randint(n_cities)
    tour = [start_city]
    unvisited.remove(start_city)

    current = start_city
    while len(unvisited) > 0:
        candidates = list(unvisited)
        tau_vals = tau[current, candidates] ** alpha
        eta_vals = eta[current, candidates] ** beta
        probs = tau_vals * eta_vals
        prob_sum = np.sum(probs)
        if prob_sum == 0 or np.isnan(prob_sum):
            next_city = np.random.choice(candidates)
        else:
            probs = probs / prob_sum
            next_city = np.random.choice(candidates, p=probs)
        tour.append(next_city)
        unvisited.remove(next_city)
        current = next_city

    return np.array(tour)


def step(weights: np.ndarray, tau: np.ndarray, problem: Any,
         alpha: float = 1.0, beta: float = 2.0, rho: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Executes one ant cycle across all subproblems."""
    N = len(weights)
    dist_mat = getattr(problem, 'dist_matrix', np.zeros((15, 15)))
    time_mat = getattr(problem, 'time_matrix', np.zeros((15, 15)))

    tours = []
    F_list = []
    CV_list = []

    for k in range(N):
        tour = construct_tour(k, weights[k], tau, dist_mat, time_mat, alpha=alpha, beta=beta)
        f, cv = problem.evaluate(tour)
        tours.append(tour)
        F_list.append(f)
        CV_list.append(cv)

    tours_arr = np.array(tours)
    F_arr = np.array(F_list)
    CV_arr = np.array(CV_list)

    # Evaporation
    tau = (1.0 - rho) * tau

    # Pheromone deposit by elite solutions
    fronts = mu.fast_non_dominated_sort(F_arr, CV_arr)
    elites = fronts[0] if len(fronts) > 0 else list(range(min(5, N)))

    for e in elites:
        t = tours_arr[e]
        deposit = 100.0 / (F_arr[e, 0] + F_arr[e, 1] + 1e-6)
        for i in range(len(t)):
            c1 = t[i]
            c2 = t[(i + 1) % len(t)]
            tau[c1, c2] += deposit
            tau[c2, c1] += deposit

    return tours_arr, F_arr, CV_arr, tau


def run(problem: Any, n_ants: int = 30, n_gen: int = 30, seed: int = 42) -> Dict[str, Any]:
    """Runs MOEA/D-ACO."""
    np.random.seed(seed)
    weights, tau, _ = initialize(problem, n_ants=n_ants)

    all_tours = []
    all_F = []
    all_CV = []

    for gen in range(n_gen):
        tours, F, CV, tau = step(weights, tau, problem)
        all_tours.extend(tours)
        all_F.extend(F)
        all_CV.extend(CV)

    all_tours_arr = np.array(all_tours)
    all_F_arr = np.array(all_F)
    all_CV_arr = np.array(all_CV)

    fronts = mu.fast_non_dominated_sort(all_F_arr, all_CV_arr)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MOEA/D-ACO',
        'pop': all_tours_arr,
        'F': all_F_arr,
        'CV': all_CV_arr,
        'X_front': all_tours_arr[nd_indices],
        'F_front': all_F_arr[nd_indices],
        'evaluations': n_ants * n_gen
    }

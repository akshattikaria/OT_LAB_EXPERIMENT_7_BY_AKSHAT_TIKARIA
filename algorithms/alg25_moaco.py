"""
alg25_moaco.py - Multi-Objective Ant Colony Optimization (MOACO)
================================================================
Family: Swarm Metaheuristic (Part 4)

Core Mechanism:
Solves multi-objective combinatorial problems (such as P8: Bi-Objective TSP)
using multiple independent pheromone matrices:
1. Dual Pheromone Structures:
   - tau^1: Tracks edges contributing to minimizing total distance.
   - tau^2: Tracks edges contributing to minimizing total travel time.
2. State Transition Rule:
   Each ant k is assigned a weight lambda_k in [0, 1]. The transition probability
   from city i to unvisited city j balances both objectives:
       P_{ij}^k proportional to [tau_{ij}^1]^{(1 - lambda_k)*alpha} * [tau_{ij}^2]^{lambda_k * alpha}
                                * [eta_{ij}^1]^{(1 - lambda_k)*beta} * [eta_{ij}^2]^{lambda_k * beta}
3. Pareto Pheromone Evaporation & Reinforcement:
   Pheromones evaporate: tau^1 <- (1 - rho) * tau^1, tau^2 <- (1 - rho) * tau^2.
   All non-dominated solutions in the global archive deposit pheromone onto tau^1
   and tau^2 proportional to their performance on each respective criterion.

Lecture Steps:
1. Initialize pheromone matrices tau^1 and tau^2 to initial baseline value tau_0.
2. Precompute heuristic visibility matrices eta^1 = 1/D and eta^2 = 1/T.
3. In each cycle:
   a. Assign each ant k an evenly spaced weight lambda_k in [0, 1].
   b. Each ant constructs a complete TSP permutation tour using the dual transition rule.
   c. Evaluate both distance and travel time for each tour.
   d. Update the global non-dominated Pareto archive with newly discovered tours.
   e. Evaporate pheromones and deposit reinforcements from archive members.
4. Return non-dominated Pareto front.

Default Parameters:
- n_ants: 30
- n_cycles: 30
- alpha: 1.0, beta: 2.0, rho: 0.1
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, n_ants: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_cities = getattr(problem, 'n_vars', 15)
    tau1 = np.ones((n_cities, n_cities)) * 1.0
    tau2 = np.ones((n_cities, n_cities)) * 1.0
    lambdas = np.linspace(0.0, 1.0, n_ants)
    return tau1, tau2, lambdas, np.zeros((0, n_cities))


def construct_ant_tour(lam: float, tau1: np.ndarray, tau2: np.ndarray,
                       eta1: np.ndarray, eta2: np.ndarray,
                       alpha: float = 1.0, beta: float = 2.0) -> np.ndarray:
    n_cities = len(tau1)
    unvisited = set(range(n_cities))
    start_city = np.random.randint(n_cities)
    tour = [start_city]
    unvisited.remove(start_city)

    w1 = 1.0 - lam
    w2 = lam

    curr = start_city
    while len(unvisited) > 0:
        candidates = list(unvisited)
        t1 = tau1[curr, candidates] ** (w1 * alpha)
        t2 = tau2[curr, candidates] ** (w2 * alpha)
        e1 = eta1[curr, candidates] ** (w1 * beta)
        e2 = eta2[curr, candidates] ** (w2 * beta)

        probs = t1 * t2 * e1 * e2
        prob_sum = np.sum(probs)
        if prob_sum == 0 or np.isnan(prob_sum):
            nxt = np.random.choice(candidates)
        else:
            probs = probs / prob_sum
            nxt = np.random.choice(candidates, p=probs)
        tour.append(nxt)
        unvisited.remove(nxt)
        curr = nxt

    return np.array(tour)


def step(tau1: np.ndarray, tau2: np.ndarray, lambdas: np.ndarray,
         problem: Any, alpha: float = 1.0, beta: float = 2.0,
         rho: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One cycle of MOACO."""
    N = len(lambdas)
    dist_mat = getattr(problem, 'dist_matrix', np.zeros((15, 15)))
    time_mat = getattr(problem, 'time_matrix', np.zeros((15, 15)))

    eta1 = 1.0 / (dist_mat + 1e-6)
    eta2 = 1.0 / (time_mat + 1e-6)
    np.fill_diagonal(eta1, 0.0)
    np.fill_diagonal(eta2, 0.0)

    tours = []
    F_list = []
    CV_list = []

    for k in range(N):
        tour = construct_ant_tour(lambdas[k], tau1, tau2, eta1, eta2, alpha=alpha, beta=beta)
        f, cv = problem.evaluate(tour)
        tours.append(tour)
        F_list.append(f)
        CV_list.append(cv)

    tours_arr = np.array(tours)
    F_arr = np.array(F_list)
    CV_arr = np.array(CV_list)

    # Evaporation
    tau1 = (1.0 - rho) * tau1
    tau2 = (1.0 - rho) * tau2

    # Reinforce non-dominated solutions
    fronts = mu.fast_non_dominated_sort(F_arr, CV_arr)
    elites = fronts[0] if len(fronts) > 0 else list(range(N))

    for e in elites:
        t = tours_arr[e]
        dep1 = 100.0 / (F_arr[e, 0] + 1e-6)
        dep2 = 100.0 / (F_arr[e, 1] + 1e-6)
        for i in range(len(t)):
            c1 = t[i]
            c2 = t[(i + 1) % len(t)]
            tau1[c1, c2] += dep1
            tau1[c2, c1] += dep1
            tau2[c1, c2] += dep2
            tau2[c2, c1] += dep2

    return tours_arr, F_arr, CV_arr, tau1, tau2


def run(problem: Any, n_ants: int = 30, n_cycles: int = 30, seed: int = 42) -> Dict[str, Any]:
    """Runs MOACO."""
    np.random.seed(seed)
    tau1, tau2, lambdas, _ = initialize(problem, n_ants=n_ants)

    all_tours = []
    all_F = []
    all_CV = []

    for cyc in range(n_cycles):
        tours, F, CV, tau1, tau2 = step(tau1, tau2, lambdas, problem)
        all_tours.extend(tours)
        all_F.extend(F)
        all_CV.extend(CV)

    all_tours_arr = np.array(all_tours)
    all_F_arr = np.array(all_F)
    all_CV_arr = np.array(all_CV)

    fronts = mu.fast_non_dominated_sort(all_F_arr, all_CV_arr)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'MOACO',
        'pop': all_tours_arr,
        'F': all_F_arr,
        'CV': all_CV_arr,
        'X_front': all_tours_arr[nd_indices],
        'F_front': all_F_arr[nd_indices],
        'evaluations': n_ants * n_cycles
    }

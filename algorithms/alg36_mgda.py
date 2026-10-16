"""
alg36_mgda.py - Multiple Gradient Descent Algorithm (MGDA)
===========================================================
Family: Gradient-based Multi-Objective (Module 5E)

Core Mechanism:
Proposed by Jean-Antoine Désidéri (2012) and popularized in deep learning
by Sener & Koltun (NeurIPS 2018).
Finds a common descent vector for multiple conflicting differentiable objectives:
1. Gradients: Computes task gradients G = [g_1, ..., g_M], where g_m = nabla f_m(x).
2. Min-Norm Quadratic Program:
   Finds a convex combination alpha* on the simplex Delta:
       alpha* = argmin_{alpha in Delta} || sum_{m=1}^M alpha_m * g_m ||^2
   For M = 2, analytical Frank-Wolfe / projection solution:
       alpha_1 = clip( (g_2 - g_1)^T g_2 / ||g_1 - g_2||^2, 0, 1 ),  alpha_2 = 1 - alpha_1
3. Common Descent Direction:
   d = - sum_{m=1}^M alpha*_m * g_m
   Mathematical property: <g_m, d> <= -||d||^2 <= 0 for all m.
   Every objective decreases simultaneously along d.
4. Pareto Stationarity: If ||d|| == 0, the current point is Pareto-critical (KKT point).

Lecture Steps:
1. Initialize starting solution x_0.
2. In each iteration:
   a. Compute task gradients g_1(x), ..., g_M(x).
   b. Solve min-norm problem on the convex hull of gradients to find alpha*.
   c. Form descent vector d = - sum alpha*_m * g_m.
   d. If ||d|| < tolerance: stop (Pareto stationarity achieved).
   e. Update parameters x = x + lr * d.
3. Return trajectory and final Pareto-optimal solution.

Default Parameters:
- lr: 0.1 (learning rate)
- max_iter: 100
- tol: 1e-5
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, x0: Optional[np.ndarray] = None, seed: int = 42) -> np.ndarray:
    if x0 is not None:
        return np.array(x0, dtype=float)
    if hasattr(problem, 'name') and "MultiTask" in problem.name:
        return np.array([0.0, 5.0]) # As prescribed in P13
    rng = np.random.default_rng(seed)
    return rng.uniform(problem.xl, problem.xu) if hasattr(problem, 'xl') and len(problem.xl) > 0 else np.zeros(problem.n_vars)


def solve_min_norm_alpha(gradients: List[np.ndarray]) -> np.ndarray:
    """
    Finds convex weights alpha on the simplex minimizing ||sum alpha_m * g_m||^2
    using closed form for M=2 and Frank-Wolfe algorithm for M >= 3 (pure NumPy).
    """
    M = len(gradients)
    if M == 1:
        return np.array([1.0])

    if M == 2:
        g1, g2 = gradients[0], gradients[1]
        diff = g1 - g2
        diff_sq = float(np.dot(diff, diff))
        if diff_sq < 1e-12:
            return np.array([0.5, 0.5])
        # alpha_1 * g1 + (1 - alpha_1) * g2
        # min || alpha_1 * (g1 - g2) + g2 ||^2 => d/d(alpha) = 2*(g1 - g2)^T (alpha*(g1 - g2) + g2) = 0
        gamma = -float(np.dot(diff, g2)) / diff_sq
        alpha1 = float(np.clip(gamma, 0.0, 1.0))
        return np.array([alpha1, 1.0 - alpha1])

    # General M >= 3: Frank-Wolfe algorithm on simplex (Sener & Koltun, 2018)
    G = np.column_stack(gradients) # (D, M)
    M_gram = G.T @ G # (M, M)
    alpha = np.ones(M) / M
    for _ in range(100):
        grad_f = 2.0 * (M_gram @ alpha)
        s_idx = int(np.argmin(grad_f))
        e_s = np.zeros(M)
        e_s[s_idx] = 1.0
        d_dir = e_s - alpha
        denom = float(d_dir.T @ M_gram @ d_dir)
        if denom < 1e-12:
            break
        # Exact line search step size
        gamma_step = float(-(grad_f.T @ d_dir) / (2.0 * denom))
        gamma_step = max(0.0, min(1.0, gamma_step))
        alpha = alpha + gamma_step * d_dir
        if gamma_step < 1e-6:
            break
    return alpha


def step(x: np.ndarray, problem: Any, lr: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Performs one MGDA descent step."""
    if hasattr(problem, 'gradients'):
        g1, g2 = problem.gradients(x)
        grads = [g1, g2]
    else:
        # Finite difference gradient approximation
        eps = 1e-5
        grads = []
        f0, _ = problem.evaluate(x)
        for m in range(problem.n_objs):
            grad_m = np.zeros_like(x)
            for d in range(len(x)):
                x_plus = x.copy()
                x_plus[d] += eps
                fp, _ = problem.evaluate(x_plus)
                grad_m[d] = (fp[m] - f0[m]) / eps
            grads.append(grad_m)

    alpha_star = solve_min_norm_alpha(grads)
    descent_dir = -np.sum([alpha_star[m] * grads[m] for m in range(len(grads))], axis=0)
    norm_d = float(np.linalg.norm(descent_dir))

    new_x = x + lr * descent_dir
    f_new, cv_new = problem.evaluate(new_x)
    return new_x, f_new, alpha_star, norm_d


def run(problem: Any, x0: Optional[np.ndarray] = None, lr: float = 0.1,
        max_iter: int = 100, tol: 1e-5 = 1e-5, seed: int = 42) -> Dict[str, Any]:
    """Runs MGDA gradient descent optimization."""
    x = initialize(problem, x0=x0, seed=seed)
    f, cv = problem.evaluate(x)

    history_x = [x.copy()]
    history_f = [f.copy()]
    history_alphas = []

    for it in range(max_iter):
        x, f, alpha_star, norm_d = step(x, problem, lr=lr)
        history_x.append(x.copy())
        history_f.append(f.copy())
        history_alphas.append(alpha_star)
        if norm_d < tol:
            break

    X_all = np.array(history_x)
    F_all = np.array(history_f)
    CV_all = np.zeros(len(F_all))

    fronts = mu.fast_non_dominated_sort(F_all, CV_all)
    nd_indices = fronts[0] if len(fronts) > 0 else [len(F_all) - 1]

    return {
        'algorithm': 'MGDA',
        'x_opt': x,
        'f_opt': f,
        'history_x': X_all,
        'history_f': F_all,
        'X_front': X_all[nd_indices],
        'F_front': F_all[nd_indices],
        'alphas': history_alphas,
        'evaluations': len(history_x)
    }

"""
alg37_gradnorm.py - Gradient Normalization for Multi-Task Learning (GradNorm)
=============================================================================
Family: Gradient-based Multi-Objective (Module 5E)

Core Mechanism:
Proposed by Chen et al. (ICML 2018).
Dynamically balances task training rates by adaptively tuning loss weights w_m(t):
1. Multi-Task Combined Loss:
   L_{tot}(t) = sum_{m=1}^M w_m(t) * L_m(t)
2. Gradient Norms:
   G_W^{(m)}(t) = || nabla_{W} (w_m(t) * L_m(t)) ||_2 = w_m(t) * || nabla_{W} L_m(t) ||_2
   Mean gradient norm: G_bar(t) = (1 / M) * sum_m G_W^{(m)}(t)
3. Relative Inverse Training Rate:
   L~_m(t) = L_m(t) / L_m(0)
   r_m(t) = L~_m(t) / [ (1 / M) * sum_k L~_k(t) ]
   Tasks training slower than average have r_m > 1 and require larger gradient norms.
4. Target Gradient Norm:
   Target_m(t) = G_bar(t) * [r_m(t)]^gamma,  (where gamma is task balancing asymmetry)
5. Weight Loss & Update:
   L_{grad}(w) = sum_m | G_W^{(m)}(t) - Target_m(t) |
   Weights w are updated via gradient descent on L_{grad} and renormalized to sum(w) = M.

Lecture Steps:
1. Initialize task weights w_m(0) = 1.0 and record initial losses L_m(0).
2. In each iteration:
   a. Compute individual task losses L_m and their gradients with respect to shared parameters W.
   b. Calculate gradient norms G_W^{(m)} and relative training rates r_m.
   c. Compute target gradient norms and update weights w_m towards targets.
   d. Renormalize weights: sum(w) = M.
   e. Update model parameters W along the dynamically weighted gradient nabla L_{tot}.
3. Return optimization trajectory and final task losses.

Default Parameters:
- lr_model: 0.05
- lr_weights: 0.025
- gamma: 1.5
- max_iter: 100
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, M: int = 2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if hasattr(problem, 'name') and "MultiTask" in problem.name:
        x0 = np.array([0.0, 5.0])
    else:
        x0 = rng.uniform(problem.xl, problem.xu) if hasattr(problem, 'xl') and len(problem.xl) > 0 else np.zeros(problem.n_vars)

    w = np.ones(M)
    f0, _ = problem.evaluate(x0)
    L0 = np.maximum(1e-4, f0)
    return x0, w, L0


def step(x: np.ndarray, w: np.ndarray, L0: np.ndarray, problem: Any,
         lr_model: float = 0.05, lr_weights: float = 0.025,
         gamma: float = 1.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One iteration of GradNorm adaptive weighting and model descent."""
    M = problem.n_objs

    # 1. Evaluate losses and gradients
    f, _ = problem.evaluate(x)
    if hasattr(problem, 'gradients'):
        g1, g2 = problem.gradients(x)
        grads = [g1, g2]
    else:
        eps = 1e-5
        grads = []
        for m in range(M):
            g_m = np.zeros_like(x)
            for d in range(len(x)):
                x_p = x.copy()
                x_p[d] += eps
                fp, _ = problem.evaluate(x_p)
                g_m[d] = (fp[m] - f[m]) / eps
            grads.append(g_m)

    # 2. Gradient norms: G_m = w_m * ||g_m||
    raw_norms = np.array([np.linalg.norm(g) for g in grads])
    G = w * raw_norms
    G_bar = np.mean(G)

    # 3. Relative inverse training rate
    L_rel = f / L0
    r = L_rel / (np.mean(L_rel) + 1e-12)

    # 4. Target gradient norm
    target_G = G_bar * (r ** gamma)

    # 5. Gradient of weight loss: d/dw_m |G_m - target_m| = raw_norm_m * sign(G_m - target_m)
    grad_w = raw_norms * np.sign(G - target_G)
    w = w - lr_weights * grad_w
    w = np.maximum(0.01, w)
    w = w * (M / np.sum(w)) # Renormalize: sum(w) = M

    # 6. Update model parameters along combined weighted gradient
    comb_grad = np.sum([w[m] * grads[m] for m in range(M)], axis=0)
    new_x = x - lr_model * comb_grad
    new_f, _ = problem.evaluate(new_x)

    return new_x, w, new_f


def run(problem: Any, lr_model: float = 0.05, lr_weights: float = 0.025,
        gamma: float = 1.5, max_iter: int = 100, seed: int = 42) -> Dict[str, Any]:
    """Runs GradNorm optimization."""
    x, w, L0 = initialize(problem, M=problem.n_objs, seed=seed)

    history_x = [x.copy()]
    history_f = [problem.evaluate(x)[0]]
    history_w = [w.copy()]

    for it in range(max_iter):
        x, w, f = step(x, w, L0, problem, lr_model=lr_model, lr_weights=lr_weights, gamma=gamma)
        history_x.append(x.copy())
        history_f.append(f.copy())
        history_w.append(w.copy())

    X_all = np.array(history_x)
    F_all = np.array(history_f)
    CV_all = np.zeros(len(F_all))

    fronts = mu.fast_non_dominated_sort(F_all, CV_all)
    nd_indices = fronts[0] if len(fronts) > 0 else [len(F_all) - 1]

    return {
        'algorithm': 'GradNorm',
        'x_opt': x,
        'f_opt': f,
        'history_x': X_all,
        'history_f': F_all,
        'history_w': np.array(history_w),
        'X_front': X_all[nd_indices],
        'F_front': F_all[nd_indices],
        'evaluations': len(history_x)
    }

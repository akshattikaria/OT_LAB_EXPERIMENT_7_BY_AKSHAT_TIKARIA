"""
alg05_value_function.py - Value Function / Utility Function Method
===================================================================
Family: Classical (Part 1)
Assigned Deep-Dive Algorithm for R = 05 (Task C).

Core Mechanism:
Solves multi-objective optimization by directly maximizing an explicit
Decision-Maker (DM) utility/value function U(f_1(x), ..., f_M(x)):
    max U(f(x))  <=>  min -U(f(x)) subject to constraints.
Unlike Pareto-generating methods that compute an entire trade-off front,
the Value Function method pinpoints the single most preferred compromise
solution that aligns with the decision-maker's risk tolerance, preference
trade-offs, and marginal rates of substitution.

Lecture Steps:
1. Elicit DM preference structure (e.g., risk aversion lambda, Cobb-Douglas elasticity,
   or quadratic loss relative to ideal aspirations).
2. Formulate scalar value/utility function U(f(x)):
   - Portfolio (P4): Mean-Variance Utility U(w) = mu^T w - (lambda / 2) * (w^T Sigma w).
   - Laptop Design (P14): Preference-loss utility U(f_1, f_2) = - w_1*(f_1 - z_1*)^2 - w_2*(runtime - z_2*)^2.
3. Incorporate problem constraints using exact Lagrange multipliers or exterior penalty.
4. Solve single-objective optimization: x* = argmin -U(f(x)) + penalty * CV(x).
5. Output optimal solution x*, corresponding objective vector F(x*), and achieved utility U*.

Default Parameters:
- utility_type: 'risk_aversion' (for portfolio) / 'quadratic_loss' (for engineering design)
- lambda_risk: 5.0 (risk aversion factor for P4)
- optimizer: 'SLSQP'
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Any, Optional
import moo_utils as mu


def initialize(problem: Any, utility_type: str = "default",
               params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Initializes utility function parameters based on DM elicitation."""
    if params is None:
        params = {}
    config = {
        'utility_type': utility_type,
        'lambda_risk': params.get('lambda_risk', 5.0), # Risk aversion for portfolio
        'weights': params.get('weights', np.ones(problem.n_objs) / problem.n_objs),
        'targets': params.get('targets', getattr(problem, 'dm_reference', np.zeros(problem.n_objs)))
    }
    return config


def evaluate_utility(f: np.ndarray, config: Dict[str, Any], problem_name: str = "") -> float:
    """
    Computes Decision Maker Utility U(f). Higher utility is better.
    We return -U(f) for minimization.
    """
    u_type = config.get('utility_type', 'default')

    if "Portfolio" in problem_name or u_type == "portfolio":
        # f[0] = risk (w^T Sigma w), f[1] = -return (-mu^T w)
        risk = f[0]
        ret = -f[1]
        lam = config.get('lambda_risk', 5.0)
        # Standard Markowitz utility: U = Expected Return - (lambda / 2) * Variance
        U = ret - 0.5 * lam * risk
        return float(U)

    elif "Laptop" in problem_name or u_type == "laptop":
        # f[0] = weight (kg), f[1] = -runtime (hours)
        weight = f[0]
        runtime = -f[1]
        # Target: 1.8 kg, 12 hours
        w_t = config.get('targets', np.array([1.8, 12.0]))
        # Normalized quadratic loss utility around target
        U = - ( ((weight - w_t[0]) / 0.5) ** 2 + ((runtime - w_t[1]) / 5.0) ** 2 )
        return float(U)

    else:
        # Default weighted exponential / linear utility
        w = config.get('weights', np.ones(len(f)) / len(f))
        # Negative weighted sum as surrogate utility
        return float(-np.dot(w, f))


def step(problem: Any, config: Dict[str, Any],
         x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Performs one optimization iteration to find optimal solution maximizing DM utility.
    """
    if x0 is None:
        if hasattr(problem, 'xl') and len(problem.xl) > 0:
            x0 = 0.5 * (problem.xl + problem.xu)
        else:
            x0 = np.zeros(problem.n_vars)

    bounds = None
    if hasattr(problem, 'xl') and len(problem.xl) > 0:
        bounds = list(zip(problem.xl, problem.xu))

    def objective(x):
        f, cv = problem.evaluate(x)
        u_val = evaluate_utility(f, config, getattr(problem, 'name', ''))
        # Minimize negative utility + penalty for constraint violation
        return -u_val + 1e4 * cv

    res = minimize(
        fun=objective,
        x0=x0,
        bounds=bounds,
        method='SLSQP',
        options={'maxiter': 500, 'ftol': 1e-7}
    )
    x_opt = res.x
    f_opt, cv_opt = problem.evaluate(x_opt)
    u_opt = evaluate_utility(f_opt, config, getattr(problem, 'name', ''))
    return x_opt, f_opt, cv_opt, u_opt


def run(problem: Any, config: Optional[Dict[str, Any]] = None,
        seed: int = 42) -> Dict[str, Any]:
    """Runs the full Value Function optimization."""
    if config is None:
        config = initialize(problem)

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(problem.xl, problem.xu) if hasattr(problem, 'xl') and len(problem.xl) > 0 else np.zeros(problem.n_vars)

    x_opt, f_opt, cv_opt, u_opt = step(problem, config, x0=x0)

    # In addition to the single optimal compromise, sweep utility parameter (e.g. risk aversion lambda or weights)
    # to demonstrate the decision maker's preference curve and build a comparison front.
    sweep_solutions_X = [x_opt]
    sweep_solutions_F = [f_opt]
    sweep_solutions_CV = [cv_opt]

    if "Portfolio" in getattr(problem, 'name', ''):
        # Sweep risk aversion lambda from 0.1 (aggressive) to 50.0 (conservative)
        lambdas = np.logspace(-1, 2, 25)
        for lam in lambdas:
            cfg_sweep = config.copy()
            cfg_sweep['lambda_risk'] = lam
            xs, fs, cvs, _ = step(problem, cfg_sweep, x0=x_opt)
            sweep_solutions_X.append(xs)
            sweep_solutions_F.append(fs)
            sweep_solutions_CV.append(cvs)
    else:
        # Sweep weights across simplex
        weights_grid = np.linspace(0.05, 0.95, 25)
        for w1 in weights_grid:
            cfg_sweep = config.copy()
            cfg_sweep['weights'] = np.array([w1, 1.0 - w1])
            xs, fs, cvs, _ = step(problem, cfg_sweep, x0=x_opt)
            sweep_solutions_X.append(xs)
            sweep_solutions_F.append(fs)
            sweep_solutions_CV.append(cvs)

    X_all = np.array(sweep_solutions_X)
    F_all = np.array(sweep_solutions_F)
    CV_all = np.array(sweep_solutions_CV)

    fronts = mu.fast_non_dominated_sort(F_all, CV_all)
    nd_indices = fronts[0] if len(fronts) > 0 else []

    return {
        'algorithm': 'Value Function',
        'x_opt': x_opt,
        'f_opt': f_opt,
        'utility_opt': u_opt,
        'cv_opt': cv_opt,
        'X_all': X_all,
        'F_all': F_all,
        'CV_all': CV_all,
        'X_front': X_all[nd_indices],
        'F_front': F_all[nd_indices],
        'config': config,
        'evaluations': len(sweep_solutions_X) * 150
    }

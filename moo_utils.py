"""
moo_utils.py - Multi-Objective Optimization Shared Toolkit
==========================================================
Optimization Techniques Lab (Chapter 5)

Contains:
1. Dominance checks (Pareto dominance & Deb's constrained-dominance rules)
2. Fast Non-Dominated Sorting (Deb et al., 2002)
3. Crowding Distance Assignment
4. Performance Metrics:
   - 2D Hypervolume (exact rectangular decomposition) & Monte Carlo (M >= 3)
   - Inverted Generational Distance (IGD)
   - Spacing Metric (Schott, 1995)
   - Min-Max Objective Normalization
5. Benchmark Problems P1 to P14 (Full formulation, evaluations, bounds, and true fronts)
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional

# ==============================================================================
# 1. DOMINANCE RELATIONS
# ==============================================================================

def dominates(u: np.ndarray, v: np.ndarray) -> bool:
    """
    Standard Pareto dominance for minimization:
    u dominates v (u < v) iff u_i <= v_i for all i and u_j < v_j for at least one j.
    """
    return bool(np.all(u <= v) and np.any(u < v))


def deb_constrained_dominates(f_a: np.ndarray, cv_a: float,
                              f_b: np.ndarray, cv_b: float) -> bool:
    """
    Deb's Constrained-Dominance Rule (Deb et al., 2002):
    Solution A dominates solution B if:
    1. A is feasible (cv_a <= 0) and B is infeasible (cv_b > 0).
    2. Both are infeasible, but A has smaller constraint violation (cv_a < cv_b).
    3. Both are feasible, and A Pareto-dominates B (f_a dominates f_b).
    """
    tol = 1e-9
    is_a_feas = cv_a <= tol
    is_b_feas = cv_b <= tol

    if is_a_feas and not is_b_feas:
        return True
    elif not is_a_feas and not is_b_feas:
        return bool(cv_a < (cv_b - tol))
    elif is_a_feas and is_b_feas:
        return dominates(f_a, f_b)
    else:
        return False


# ==============================================================================
# 2. FAST NON-DOMINATED SORTING
# ==============================================================================

def fast_non_dominated_sort(F: np.ndarray, CV: Optional[np.ndarray] = None) -> List[List[int]]:
    """
    Fast Non-Dominated Sorting Algorithm (Deb et al., 2002).
    Args:
        F: (N, M) array of objective values (all minimization).
        CV: Optional (N,) array of total constraint violations.
            If provided, Deb's constrained-dominance is used.
    Returns:
        List of fronts, where each front is a list of individual indices [0, ..., N-1].
        Front 0 is the non-dominated Pareto front.
    """
    N = F.shape[0]
    if N == 0:
        return []

    # Vectorized dominance matrix: dom[p, q] is True iff p dominates q
    if CV is not None:
        tol = 1e-9
        feas = CV <= tol
        infeas = ~feas

        # Rule 1: p feasible, q infeasible -> p dominates q
        r1 = feas[:, None] & infeas[None, :]
        # Rule 2: both infeasible -> smaller CV wins
        r2 = infeas[:, None] & infeas[None, :] & (CV[:, None] < (CV[None, :] - tol))
        # Rule 3: both feasible -> Pareto dominance
        pareto = np.all(F[:, None, :] <= F[None, :, :], axis=2) & np.any(F[:, None, :] < F[None, :, :], axis=2)
        r3 = feas[:, None] & feas[None, :] & pareto

        dom = r1 | r2 | r3
    else:
        dom = np.all(F[:, None, :] <= F[None, :, :], axis=2) & np.any(F[:, None, :] < F[None, :, :], axis=2)

    np.fill_diagonal(dom, False)

    # n[q]: count of solutions that dominate q
    n = np.sum(dom, axis=0)
    fronts: List[List[int]] = [np.where(n == 0)[0].tolist()]

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in np.where(dom[p])[0]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break

    if len(fronts[-1]) == 0:
        fronts.pop()

    return fronts


# ==============================================================================
# 3. CROWDING DISTANCE ASSIGNMENT
# ==============================================================================

def calculate_crowding_distance(F: np.ndarray, front_indices: List[int]) -> np.ndarray:
    """
    Calculates crowding distance for individuals in a given front.
    Args:
        F: (N, M) objective values.
        front_indices: List of indices corresponding to the front.
    Returns:
        distances: array of crowding distances corresponding to each index in front_indices.
    """
    l = len(front_indices)
    if l == 0:
        return np.array([])
    if l <= 2:
        return np.full(l, np.inf)

    M = F.shape[1]
    distances = np.zeros(l)
    sub_F = F[front_indices]

    for m in range(M):
        sorted_order = np.argsort(sub_F[:, m])
        distances[sorted_order[0]] = np.inf
        distances[sorted_order[-1]] = np.inf

        f_min = sub_F[sorted_order[0], m]
        f_max = sub_F[sorted_order[-1], m]
        f_range = f_max - f_min
        if f_range < 1e-12:
            continue

        for i in range(1, l - 1):
            if np.isneginf(distances[sorted_order[i]]) or np.isposinf(distances[sorted_order[i]]):
                continue
            prev_val = sub_F[sorted_order[i - 1], m]
            next_val = sub_F[sorted_order[i + 1], m]
            distances[sorted_order[i]] += (next_val - prev_val) / f_range

    return distances


# ==============================================================================
# 4. NORMALIZATION & PERFORMANCE METRICS
# ==============================================================================

def normalize_objectives(F: np.ndarray,
                         ideal: Optional[np.ndarray] = None,
                         nadir: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-max normalizes objective values to [0, 1].
    """
    if ideal is None:
        ideal = np.min(F, axis=0)
    if nadir is None:
        nadir = np.max(F, axis=0)
    denom = np.where((nadir - ideal) < 1e-12, 1.0, nadir - ideal)
    F_norm = (F - ideal) / denom
    return F_norm, ideal, nadir


def hypervolume_2d(F: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Exact Hypervolume calculation for 2D minimization problems.
    Points are sorted by f1, and dominated area relative to ref_point is calculated.
    Any points exceeding ref_point are filtered out or clipped.
    """
    if len(F) == 0:
        return 0.0

    # Keep only non-dominated points strictly better than ref_point
    valid = np.all(F <= ref_point, axis=1)
    if not np.any(valid):
        return 0.0

    valid_F = F[valid]
    # Filter to non-dominated within valid
    fronts = fast_non_dominated_sort(valid_F)
    nd_F = valid_F[fronts[0]]

    # Sort ascending by f1 (so f2 descends)
    sorted_idx = np.argsort(nd_F[:, 0])
    nd_F = nd_F[sorted_idx]

    # Remove duplicates
    _, unique_idx = np.unique(nd_F[:, 0], return_index=True)
    nd_F = nd_F[np.sort(unique_idx)]

    hv = 0.0
    current_h = ref_point[1]
    for i in range(len(nd_F)):
        p = nd_F[i]
        if i == len(nd_F) - 1:
            next_x = ref_point[0]
        else:
            next_x = nd_F[i + 1, 0]
        width = next_x - p[0]
        height = current_h - p[1]
        if width > 0 and height > 0:
            hv += width * height
        # update current_h
        current_h = min(current_h, p[1])

    # Slicing from another standard perspective for precision:
    # A standard sweep line in f1:
    hv_sweep = 0.0
    # Add sentinel point at end
    sweep_pts = np.vstack([nd_F, [ref_point[0], ref_point[1]]])
    for i in range(len(nd_F)):
        w = sweep_pts[i + 1, 0] - sweep_pts[i, 0]
        h = max(0.0, ref_point[1] - sweep_pts[i, 1])
        hv_sweep += w * h
    return float(hv_sweep)


def hypervolume_monte_carlo(F: np.ndarray, ref_point: np.ndarray,
                            ideal_point: Optional[np.ndarray] = None,
                            n_samples: int = 50000, seed: int = 42) -> float:
    """
    Monte Carlo estimate of Hypervolume for M >= 3 dimensions.
    Uniformly samples in the hyper-rectangle [ideal_point, ref_point].
    """
    if len(F) == 0:
        return 0.0

    valid = np.all(F <= ref_point, axis=1)
    if not np.any(valid):
        return 0.0
    valid_F = F[valid]

    M = F.shape[1]
    if ideal_point is None:
        ideal_point = np.min(valid_F, axis=0)

    # Box volume
    box_sides = ref_point - ideal_point
    if np.any(box_sides <= 0):
        return 0.0
    box_vol = np.prod(box_sides)

    rng = np.random.default_rng(seed)
    samples = rng.uniform(ideal_point, ref_point, size=(n_samples, M))

    # Check which samples are dominated by at least one solution in valid_F
    # sample s is dominated by p in valid_F iff all p_m <= s_m
    dominated_count = 0
    # Process in chunks to maintain low memory usage
    chunk_size = 5000
    for i in range(0, n_samples, chunk_size):
        chunk = samples[i:i + chunk_size]
        # chunk: (C, M), valid_F: (N, M)
        # diff: (C, N, M) -> chunk[:, None, :] >= valid_F[None, :, :]
        dom = np.all(chunk[:, None, :] >= valid_F[None, :, :], axis=2)
        dominated_count += np.sum(np.any(dom, axis=1))

    return float(box_vol * (dominated_count / n_samples))


def hypervolume(F: np.ndarray, ref_point: np.ndarray,
                ideal_point: Optional[np.ndarray] = None) -> float:
    """
    Unified hypervolume function:
    Uses exact 2D method for M == 2, and Monte Carlo for M >= 3.
    """
    if len(F) == 0:
        return 0.0
    if F.shape[1] == 2:
        return hypervolume_2d(F, ref_point)
    else:
        return hypervolume_monte_carlo(F, ref_point, ideal_point=ideal_point)


def igd(obtained_front: np.ndarray, true_front: np.ndarray) -> float:
    """
    Inverted Generational Distance (IGD).
    Measures both convergence and diversity.
    IGD(P*, A) = (1 / |P*|) * sum_{v in P*} min_{u in A} ||v - u||_2
    Lower is better.
    """
    if len(obtained_front) == 0 or len(true_front) == 0:
        return float('inf')

    # Distance matrix between true_front and obtained_front
    # diff: (|P*|, |A|, M)
    diff = true_front[:, None, :] - obtained_front[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    min_dists = np.min(dists, axis=1)
    return float(np.mean(min_dists))


def spacing(front: np.ndarray) -> float:
    """
    Spacing Metric (Schott, 1995).
    S = sqrt( (1 / (N - 1)) * sum_{i=1}^N (d_i - d_mean)^2 )
    where d_i = min_{j != i} sum_{m=1}^M |f_m^i - f_m^j| (Manhattan distance).
    Measures uniformity of point spread along the obtained front.
    Lower is better (0 means perfectly uniform spacing).
    """
    N = len(front)
    if N <= 1:
        return 0.0

    # Manhattan distances between all pairs
    diff = np.abs(front[:, None, :] - front[None, :, :])
    manhattan = np.sum(diff, axis=2)
    np.fill_diagonal(manhattan, np.inf)

    d = np.min(manhattan, axis=1)
    d_mean = np.mean(d)
    s = np.sqrt(np.sum((d - d_mean) ** 2) / (N - 1))
    return float(s)


# ==============================================================================
# 4b. FIXED METRIC SPECIFICATIONS AND compute_metrics
# ==============================================================================
# METRIC_SPEC[problem_name] = {'ideal': array, 'nadir': array, 'ref': array,
#                               'true_F_norm': array or None}
# Populated lazily on first call so we don't slow down import.
_METRIC_SPEC_CACHE: Dict[str, Any] = {}

def _build_metric_spec(problem_name: str) -> Dict[str, Any]:
    """
    Build and cache METRIC_SPEC for a given problem.

    For problems with true_pareto_front(): ideal/nadir from 500 true-front samples.
    P10 : ideal = 0, nadir = 1 (unit sphere, all objectives in [0,1]).
    P9  : ideal = (0, 1/30), nadir = (0.40, 1.0).
    P8  : ideal/nadir from union CSV if present, else fallback to ref_point.
    Others without a known front: ideal = None (metrics will return None).
    """
    if problem_name in _METRIC_SPEC_CACHE:
        return _METRIC_SPEC_CACHE[problem_name]

    spec: Dict[str, Any] = {
        'ideal': None, 'nadir': None, 'ref': None, 'true_F_norm': None
    }

    # Problems with analytically known fronts
    _TRUE_FRONT_PROBLEMS = {
        'P1_FactoryProductionSpeed': lambda: ProblemP1().true_pareto_front(500),
        'P2_DeliveryFleetSizing':    lambda: ProblemP2().true_pareto_front(500),
        'P3_ZDT2':                   lambda: ProblemP3().true_pareto_front(500),
        'P4_InvestmentPortfolio':    lambda: ProblemP4().true_pareto_front(500),
        'P5_CantileverBeam':         lambda: ProblemP5().true_pareto_front(500),
        'P6_BinhKorn':               lambda: ProblemP6().true_pareto_front(500),
        'P7_WaterAllocation':        lambda: ProblemP7().true_pareto_front(500),
        'P14_LaptopDesign':          lambda: ProblemP14().true_pareto_front(500),
    }

    if problem_name in _TRUE_FRONT_PROBLEMS:
        tf = _TRUE_FRONT_PROBLEMS[problem_name]()
        ideal = np.min(tf, axis=0)
        nadir = np.max(tf, axis=0)
        # Normalize true front with the fixed ideal/nadir
        denom = np.where((nadir - ideal) < 1e-12, 1.0, nadir - ideal)
        ref = np.ones(tf.shape[1]) * 1.1
        true_F_norm = (tf - ideal) / denom
        spec = {'ideal': ideal, 'nadir': nadir, 'ref': ref, 'true_F_norm': true_F_norm}

    elif problem_name.startswith('P10_DTLZ2') or problem_name == 'P10_DTLZ2':
        # DTLZ2: true front on unit sphere, all objectives in [0, 1]
        # Use n_objs from name if embedded, else assume 3
        M = 3
        if '_M' in problem_name:
            try:
                M = int(problem_name.split('_M')[1])
            except Exception:
                M = 3
        ideal = np.zeros(M)
        nadir = np.ones(M)
        ref = np.full(M, 1.1)
        # Generate normalized true front on unit sphere
        rng_tf = np.random.default_rng(42)
        pts = np.abs(rng_tf.normal(size=(500, M)))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        true_F_norm = pts / np.where(norms < 1e-12, 1.0, norms)
        spec = {'ideal': ideal, 'nadir': nadir, 'ref': ref, 'true_F_norm': true_F_norm}

    elif problem_name == 'P9_FeatureSelection':
        ideal = np.array([0.0, 1.0 / 30.0])
        nadir = np.array([0.40, 1.0])
        ref = np.array([1.1, 1.1])
        spec = {'ideal': ideal, 'nadir': nadir, 'ref': ref, 'true_F_norm': None}

    elif problem_name == 'P8_BiObjectiveTSP':
        # Try to load union CSV if it exists
        union_path = 'results/data/ref_union_P8.csv'
        try:
            rows = []
            with open(union_path, 'r') as fh:
                import csv as _csv
                reader = _csv.DictReader(fh)
                for row in reader:
                    rows.append([float(row['f1']), float(row['f2'])])
            if rows:
                union_F = np.array(rows)
                ideal = np.min(union_F, axis=0)
                nadir = np.max(union_F, axis=0)
                ref = np.array([1.1, 1.1])
                spec = {'ideal': ideal, 'nadir': nadir, 'ref': ref, 'true_F_norm': None}
        except Exception:
            # Fallback: use problem ref_point as nadir
            p8 = ProblemP8()
            ideal = np.zeros(2)
            nadir = p8.ref_point.copy()
            ref = np.array([1.1, 1.1])
            spec = {'ideal': ideal, 'nadir': nadir, 'ref': ref, 'true_F_norm': None}

    _METRIC_SPEC_CACHE[problem_name] = spec
    return spec


def _get_problem_name_from_key(problem_name: str) -> str:
    """Map short keys like 'P3' to full class names like 'P3_ZDT2'."""
    _SHORT_TO_FULL = {
        'P1': 'P1_FactoryProductionSpeed',
        'P2': 'P2_DeliveryFleetSizing',
        'P3': 'P3_ZDT2',
        'P4': 'P4_InvestmentPortfolio',
        'P5': 'P5_CantileverBeam',
        'P6': 'P6_BinhKorn',
        'P7': 'P7_WaterAllocation',
        'P8': 'P8_BiObjectiveTSP',
        'P9': 'P9_FeatureSelection',
        'P10': 'P10_DTLZ2',
        'P10_M3': 'P10_DTLZ2_M3',
        'P10_M5': 'P10_DTLZ2_M5',
        'P11': 'P11_ProductionPlanning',
        'P12': 'P12_SupplierSelection',
        'P13': 'P13_MultiTaskLearning',
        'P14': 'P14_LaptopDesign',
    }
    return _SHORT_TO_FULL.get(problem_name, problem_name)


def _drop_dominated_duplicates(F: np.ndarray) -> np.ndarray:
    """Remove dominated and duplicate rows from F."""
    if len(F) == 0:
        return F
    # Remove exact duplicates
    F_unique = np.unique(F, axis=0)
    if len(F_unique) == 1:
        return F_unique
    fronts = fast_non_dominated_sort(F_unique)
    return F_unique[fronts[0]] if len(fronts) > 0 else F_unique


def compute_metrics(front_F: np.ndarray,
                    problem_name: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute HV, IGD, Spacing for a front using fixed per-problem normalization.

    Parameters
    ----------
    front_F : (N, M) array of objective values.
    problem_name : short name ('P3', 'P10_M5', ...) or full name.

    Returns
    -------
    hv  : float or None  (None if spec unavailable for this problem)
    igd : float or None  (None if true front unknown)
    sp  : float or None  (None if < 2 points)
    """
    if front_F is None or len(front_F) == 0:
        return None, None, None

    full_name = _get_problem_name_from_key(problem_name)

    # Handle DTLZ2 with explicit M
    if problem_name in ('P10_M3', 'P10_M5'):
        M = int(problem_name[-1])
        full_name = f'P10_DTLZ2_M{M}'
        if full_name not in _METRIC_SPEC_CACHE:
            M_val = M
            ideal = np.zeros(M_val)
            nadir = np.ones(M_val)
            ref = np.full(M_val, 1.1)
            rng_tf = np.random.default_rng(42)
            pts = np.abs(rng_tf.normal(size=(500, M_val)))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            true_F_norm = pts / np.where(norms < 1e-12, 1.0, norms)
            _METRIC_SPEC_CACHE[full_name] = {
                'ideal': ideal, 'nadir': nadir, 'ref': ref, 'true_F_norm': true_F_norm
            }
    elif 'P10' in problem_name and full_name not in _METRIC_SPEC_CACHE:
        _build_metric_spec(full_name)
    else:
        _build_metric_spec(full_name)

    spec = _METRIC_SPEC_CACHE.get(full_name, {'ideal': None, 'nadir': None, 'ref': None, 'true_F_norm': None})

    ideal = spec['ideal']
    nadir = spec['nadir']
    ref = spec['ref']
    true_F_norm = spec['true_F_norm']

    if ideal is None or nadir is None:
        # No fixed spec: cannot compute normalized metrics
        return None, None, None

    # Normalize the obtained front with fixed ideal/nadir
    denom = np.where((nadir - ideal) < 1e-12, 1.0, nadir - ideal)
    F_norm = (front_F - ideal) / denom
    F_norm = np.maximum(0.0, F_norm)

    # Drop dominated/duplicate points
    F_nd = _drop_dominated_duplicates(F_norm)

    # HV
    M = F_nd.shape[1]
    if M == 2:
        hv_val = hypervolume_2d(F_nd, ref)
    else:
        hv_val = hypervolume_monte_carlo(F_nd, ref, ideal_point=np.zeros(M), n_samples=100000, seed=42)

    # IGD
    igd_val = None
    if true_F_norm is not None and len(F_nd) > 0:
        igd_val = igd(F_nd, true_F_norm)

    # Spacing (in normalized space, on the non-dominated front)
    sp_val = spacing(F_nd) if len(F_nd) >= 2 else None

    return hv_val, igd_val, sp_val


# ==============================================================================
# 5. PROBLEM BANK (P1 - P14)
# ==============================================================================

class BaseProblem:
    """Base class for all optimization problems."""
    name: str = ""
    n_vars: int = 1
    n_objs: int = 2
    xl: np.ndarray = np.array([])
    xu: np.ndarray = np.array([])
    is_discrete: bool = False
    ref_point: np.ndarray = np.array([1.1, 1.1]) # Normalized default

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Evaluates single decision vector x.
        Returns:
            f: (M,) array of objective values (all formulated for MINIMIZATION)
            cv: float, total constraint violation (0 if fully feasible)
        """
        raise NotImplementedError

    def true_pareto_front(self, n_points: int = 100) -> Optional[np.ndarray]:
        """Returns sampled points from the true Pareto front if analytically known."""
        return None


class CountingProblem:
    """Wrapper around any optimization problem that counts calls to evaluate()."""
    def __init__(self, problem: Any, max_evals: Optional[int] = None):
        self._problem = problem
        self.evaluations = 0
        self.max_evals = max_evals

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        if self.max_evals is not None and self.evaluations >= self.max_evals:
            n_objs = getattr(self._problem, 'n_objs', 2)
            return np.full(n_objs, 1e9), 1e9
        self.evaluations += 1
        return self._problem.evaluate(x)

    def reset(self):
        self.evaluations = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)



# --- P1: Factory production speed (convex, 1 variable) ---
class ProblemP1(BaseProblem):
    name = "P1_FactoryProductionSpeed"
    n_vars = 1
    n_objs = 2
    xl = np.array([1.0])
    xu = np.array([10.0])
    ref_point = np.array([18.0, 18.0])

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        x0 = float(x[0])
        f1 = (x0 - 3.0) ** 2
        f2 = (x0 - 7.0) ** 2
        return np.array([f1, f2]), 0.0

    def true_pareto_front(self, n_points: int = 100) -> np.ndarray:
        x_vals = np.linspace(3.0, 7.0, n_points)
        f1 = (x_vals - 3.0) ** 2
        f2 = (x_vals - 7.0) ** 2
        return np.column_stack([f1, f2])


# --- P2: Delivery fleet sizing (1 variable, real vs integer) ---
class ProblemP2(BaseProblem):
    name = "P2_DeliveryFleetSizing"
    n_vars = 1
    n_objs = 2
    xl = np.array([1.0])
    xu = np.array([10.0])
    ref_point = np.array([110.0, 110.0])

    def __init__(self, integer_mode: bool = False):
        self.integer_mode = integer_mode

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        val = float(x[0])
        if self.integer_mode:
            val = float(int(np.round(val)))
            val = max(1.0, min(10.0, val))
        f1 = 100.0 / val
        f2 = val ** 2
        return np.array([f1, f2]), 0.0

    def true_pareto_front(self, n_points: int = 100) -> np.ndarray:
        if self.integer_mode:
            x_vals = np.arange(1, 11, dtype=float)
        else:
            x_vals = np.linspace(1.0, 10.0, n_points)
        f1 = 100.0 / x_vals
        f2 = x_vals ** 2
        return np.column_stack([f1, f2])


# --- P3: Non-convex benchmark (ZDT2, n = 10) ---
class ProblemP3(BaseProblem):
    name = "P3_ZDT2"
    n_vars = 10
    n_objs = 2
    xl = np.zeros(10)
    xu = np.ones(10)
    ref_point = np.array([1.1, 1.1])

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        f1 = float(x[0])
        n = len(x)
        g = 1.0 + 9.0 * float(np.sum(x[1:])) / (n - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2]), 0.0

    def true_pareto_front(self, n_points: int = 100) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - f1 ** 2
        return np.column_stack([f1, f2])


# --- P4: Investment portfolio (4 continuous assets) ---
class ProblemP4(BaseProblem):
    name = "P4_InvestmentPortfolio"
    n_vars = 4
    n_objs = 2
    xl = np.zeros(4)
    xu = np.ones(4)
    ref_point = np.array([0.10, -0.07]) # risk <= 0.10, return >= 0.07 -> -return <= -0.07

    def __init__(self, use_repair: bool = True):
        self.use_repair = use_repair
        self.mu = np.array([0.08, 0.12, 0.15, 0.10])
        self.Sigma = np.array([
            [0.010, 0.002, 0.001, 0.003],
            [0.002, 0.040, 0.006, 0.004],
            [0.001, 0.006, 0.090, 0.005],
            [0.003, 0.004, 0.005, 0.020]
        ])

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        w = np.copy(x)
        cv = 0.0
        if self.use_repair:
            s = np.sum(w)
            if s > 1e-12:
                w = w / s
            else:
                w = np.full(4, 0.25)
        else:
            cv = abs(float(np.sum(w) - 1.0))

        risk = float(w.T @ self.Sigma @ w)
        ret = float(self.mu.T @ w)
        # Minimize risk, maximize return => minimize -return
        return np.array([risk, -ret]), cv

    def true_pareto_front(self, n_points: int = 100) -> np.ndarray:
        # Generate true front using pure NumPy projected gradient descent on simplex
        def project_simplex(v):
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u)
            rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1.0))[0][-1]
            theta = (cssv[rho] - 1.0) / (rho + 1.0)
            return np.maximum(v - theta, 0.0)

        lambdas = np.concatenate([[0.0], np.logspace(-2, 2.5, n_points - 2), [1000.0]])
        front = []
        w = np.full(4, 0.25)
        lr = 0.5
        for lam in lambdas:
            for _ in range(300):
                grad = 2.0 * (self.Sigma @ w) - lam * self.mu
                w = project_simplex(w - lr * grad)
            risk = float(w.T @ self.Sigma @ w)
            ret = float(self.mu.T @ w)
            front.append([risk, -ret])

        front_arr = np.array(front)
        # Filter non-dominated
        fronts = fast_non_dominated_sort(front_arr)
        nd_F = front_arr[fronts[0]]
        # Sort by risk
        return nd_F[np.argsort(nd_F[:, 0])]


# --- P5: Cantilever beam design (constrained, 2 variables) ---
class ProblemP5(BaseProblem):
    name = "P5_CantileverBeam"
    n_vars = 2
    n_objs = 2
    xl = np.array([0.01, 0.02]) # b, h
    xu = np.array([0.10, 0.20])
    ref_point = np.array([160.0, 0.03])

    def __init__(self, enforce_constraint: bool = True):
        self.enforce_constraint = enforce_constraint
        self.L = 1.0
        self.P = 1000.0
        self.E = 200e9
        self.rho = 7850.0
        self.stress_limit = 250e6
        self.bh2_min = 2.4e-5 # equivalent to bending stress <= 250 MPa

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        b, h = float(x[0]), float(x[1])
        # f1 = mass (kg)
        f1 = self.rho * self.L * b * h
        # f2 = tip deflection (m)
        f2 = (4.0 * self.P * (self.L ** 3)) / (self.E * b * (h ** 3))

        cv = 0.0
        if self.enforce_constraint:
            # Stress constraint: b * h^2 >= 2.4e-5
            val = b * (h ** 2)
            if val < self.bh2_min:
                cv = (self.bh2_min - val) / self.bh2_min
        return np.array([f1, f2]), cv

    def true_pareto_front(self, n_points: int = 100) -> np.ndarray:
        # Pareto front occurs along active constraint b * h^2 = 2.4e-5
        h_vals = np.linspace(self.xl[1], self.xu[1], n_points)
        b_vals = self.bh2_min / (h_vals ** 2)
        valid = (b_vals >= self.xl[0]) & (b_vals <= self.xu[0])
        h_vals = h_vals[valid]
        b_vals = b_vals[valid]
        f1 = self.rho * self.L * b_vals * h_vals
        f2 = (4.0 * self.P * (self.L ** 3)) / (self.E * b_vals * (h_vals ** 3))
        return np.column_stack([f1, f2])


# --- P6: Binh-Korn constrained benchmark ---
class ProblemP6(BaseProblem):
    name = "P6_BinhKorn"
    n_vars = 2
    n_objs = 2
    xl = np.array([0.0, 0.0])
    xu = np.array([5.0, 3.0])
    ref_point = np.array([140.0, 55.0])

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        x1, x2 = float(x[0]), float(x[1])
        f1 = 4.0 * (x1 ** 2) + 4.0 * (x2 ** 2)
        f2 = (x1 - 5.0) ** 2 + (x2 - 5.0) ** 2

        # Constraints:
        # g1: (x1 - 5)^2 + x2^2 <= 25
        # g2: (x1 - 8)^2 + (x2 + 3)^2 >= 7.7 => 7.7 - (x1-8)^2 - (x2+3)^2 <= 0
        g1 = ((x1 - 5.0) ** 2) + (x2 ** 2) - 25.0
        g2 = 7.7 - ((x1 - 8.0) ** 2) - ((x2 + 3.0) ** 2)

        cv = max(0.0, g1) + max(0.0, g2)
        return np.array([f1, f2]), cv

    def true_pareto_front(self, n_points: int = 100) -> np.ndarray:
        # Analytical front is x1 = x2 for x1 in [0, 3] and x2=3 for x1 in [3, 5]
        x1_a = np.linspace(0.0, 3.0, n_points // 2)
        pts1 = np.column_stack([4.0 * (x1_a**2) + 4.0 * (x1_a**2),
                                (x1_a - 5.0)**2 + (x1_a - 5.0)**2])
        x1_b = np.linspace(3.0, 5.0, n_points // 2)
        pts2 = np.column_stack([4.0 * (x1_b**2) + 4.0 * (3.0**2),
                                (x1_b - 5.0)**2 + (3.0 - 5.0)**2])
        return np.vstack([pts1, pts2])


# --- P7: Water allocation with environmental flow ---
class ProblemP7(BaseProblem):
    name = "P7_WaterAllocation"
    n_vars = 2
    n_objs = 2
    xl = np.array([10.0, 20.0])
    xu = np.array([70.0, 80.0])
    ref_point = np.array([0.0, 0.0]) # Min -f1 and -f2: yields are up to (56, 96) => ref point (0, 0)

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        x1, x2 = float(x[0]), float(x[1])
        # Max 0.8*x1, Max 1.2*x2 => Min -0.8*x1, Min -1.2*x2
        f1 = -0.8 * x1
        f2 = -1.2 * x2
        # Constraint: x1 + x2 <= 90
        cv = max(0.0, (x1 + x2) - 90.0)
        return np.array([f1, f2]), cv

    def true_pareto_front(self, n_points: int = 100) -> np.ndarray:
        x1 = np.linspace(10.0, 70.0, n_points)
        x2 = 90.0 - x1
        f1 = -0.8 * x1
        f2 = -1.2 * x2
        return np.column_stack([f1, f2])


# --- P8: Bi-objective delivery routing (discrete TSP, n=15 cities) ---
class ProblemP8(BaseProblem):
    name = "P8_BiObjectiveTSP"
    n_vars = 15
    n_objs = 2
    is_discrete = True
    ref_point = np.array([1000.0, 1500.0])

    def __init__(self, seed: int = 7):
        np.random.seed(seed)
        self.coords = np.random.uniform(0.0, 100.0, size=(15, 2))
        N = 15
        self.dist_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                self.dist_matrix[i, j] = np.linalg.norm(self.coords[i] - self.coords[j])

        # Symmetric u_ij ~ U(0.5, 2.0)
        u_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                u = np.random.uniform(0.5, 2.0)
                u_matrix[i, j] = u
                u_matrix[j, i] = u
        self.time_matrix = self.dist_matrix * u_matrix

    def evaluate(self, tour: np.ndarray) -> Tuple[np.ndarray, float]:
        tour_int = np.array(tour, dtype=int)
        N = len(tour_int)
        d_tot = 0.0
        t_tot = 0.0
        for k in range(N):
            c1 = tour_int[k]
            c2 = tour_int[(k + 1) % N]
            d_tot += self.dist_matrix[c1, c2]
            t_tot += self.time_matrix[c1, c2]
        # Check permutation validity
        cv = 0.0
        if len(np.unique(tour_int)) < N:
            cv = float(N - len(np.unique(tour_int)))
        return np.array([d_tot, t_tot]), cv


# --- P9: Feature selection (binary, 30 features, Wisconsin Breast Cancer) ---
class ProblemP9(BaseProblem):
    name = "P9_FeatureSelection"
    n_vars = 30
    n_objs = 2
    is_discrete = True
    xl = np.zeros(30)
    xu = np.ones(30)
    ref_point = np.array([0.5, 1.0])

    def __init__(self):
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        self.X = data.data # (569, 30)
        self.y = data.target # (569,)
        # Normalize features to zero mean and unit variance for KNN
        mean = np.mean(self.X, axis=0)
        std = np.std(self.X, axis=0)
        std[std == 0] = 1.0
        self.X_norm = ((self.X - mean) / std).astype(np.float32)

        # Precompute 5-fold CV indices deterministically
        rng = np.random.default_rng(42)
        indices = np.arange(len(self.y))
        rng.shuffle(indices)
        self.folds = np.array_split(indices, 5)

        # S3g: Precompute per-feature squared-difference matrices once (30, 569, 569)
        self.diff_sq = np.ascontiguousarray(
            (self.X_norm.T[:, :, None] - self.X_norm.T[:, None, :]) ** 2,
            dtype=np.float32
        )
        self.diff_sq_flat = self.diff_sq.reshape(30, -1)
        self._cache = {}
        self.fold_y_tr = []
        self.fold_y_va = []
        self.fold_val_idx = []
        self.fold_train_idx = []
        for fold_idx in range(5):
            val_idx = self.folds[fold_idx]
            train_idx = np.hstack([self.folds[j] for j in range(5) if j != fold_idx])
            self.fold_val_idx.append(val_idx)
            self.fold_train_idx.append(train_idx)
            self.fold_y_tr.append(self.y[train_idx])
            self.fold_y_va.append(self.y[val_idx])

    def evaluate(self, mask: np.ndarray) -> Tuple[np.ndarray, float]:
        mask_bool = (mask > 0.5)
        key = mask_bool.tobytes()
        if key in self._cache:
            return self._cache[key]

        n_sel = int(np.sum(mask_bool))
        if n_sel == 0:
            res = (np.array([1.0, 1.0]), 1.0) # constraint violation: must select at least 1 feature
            self._cache[key] = res
            return res

        # Fast BLAS Level 2 matrix-vector multiplication (sgemv)
        mask_f = mask_bool.astype(np.float32)
        total_dist = (mask_f @ self.diff_sq_flat).reshape(569, 569)

        correct = 0
        total = len(self.y)
        k = 5

        for fold_idx in range(5):
            val_idx = self.fold_val_idx[fold_idx]
            train_idx = self.fold_train_idx[fold_idx]
            y_tr = self.fold_y_tr[fold_idx]
            y_va = self.fold_y_va[fold_idx]

            dists = total_dist[val_idx][:, train_idx]
            k_near = np.argpartition(dists, k, axis=1)[:, :k]
            votes = np.sum(y_tr[k_near] == 1, axis=1)
            pred_labels = (votes > (k / 2.0)).astype(int)
            correct += int(np.sum(pred_labels == y_va))

        acc = correct / total
        err = 1.0 - acc
        frac_features = n_sel / 30.0
        res = (np.array([err, frac_features]), 0.0)
        self._cache[key] = res
        return res


# --- P10: Many-objective benchmark (DTLZ2, M = 3 or M = 5) ---
class ProblemP10(BaseProblem):
    name = "P10_DTLZ2"

    def __init__(self, M: int = 3):
        self.M = M
        self.n_vars = M + 9
        self.n_objs = M
        self.xl = np.zeros(self.n_vars)
        self.xu = np.ones(self.n_vars)
        self.ref_point = np.full(M, 1.5)

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        M = self.M
        x_m = x[M - 1:]
        g = float(np.sum((x_m - 0.5) ** 2))

        f = np.zeros(M)
        # f1 = (1 + g) * prod_{i=1}^{M-1} cos(x_i * pi / 2)
        cos_terms = np.cos(x[:M - 1] * np.pi / 2.0)
        sin_terms = np.sin(x[:M - 1] * np.pi / 2.0)

        f[0] = (1.0 + g) * np.prod(cos_terms)
        for m in range(2, M + 1):
            prod_cos = np.prod(cos_terms[:M - m]) if (M - m) > 0 else 1.0
            sin_idx = M - m
            f[m - 1] = (1.0 + g) * prod_cos * sin_terms[sin_idx]

        return f, 0.0

    def true_pareto_front(self, n_points: int = 200) -> np.ndarray:
        # Uniform sampling on the unit sphere octant: sum f_m^2 = 1, f_m >= 0
        rng = np.random.default_rng(42)
        pts = rng.normal(size=(n_points, self.M))
        pts = np.abs(pts)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        return pts / norms


# --- P11: Production planning (Goals and priorities) ---
class ProblemP11(BaseProblem):
    name = "P11_ProductionPlanning"
    n_vars = 2
    n_objs = 2
    xl = np.array([0.0, 0.0])
    xu = np.array([500.0, 500.0])
    ref_point = np.array([0.0, 0.0]) # Min -profit, Min -units

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        xA, xB = float(x[0]), float(x[1])
        # Profit = 4*xA + 6*xB, Total Units = xA + xB
        f1 = -(4.0 * xA + 6.0 * xB) # maximize profit
        f2 = -(xA + xB) # maximize total units

        # Constraints:
        # 2*xA + 3*xB <= 1200
        # 4*xA + 2*xB <= 1600
        g1 = (2.0 * xA + 3.0 * xB) - 1200.0
        g2 = (4.0 * xA + 2.0 * xB) - 1600.0

        cv = max(0.0, g1) + max(0.0, g2)
        return np.array([f1, f2]), cv


# --- P12: Supplier selection (MCDM ranking) ---
class ProblemP12(BaseProblem):
    name = "P12_SupplierSelection"
    n_vars = 1
    n_objs = 4

    def __init__(self):
        # 5 suppliers, 4 criteria: Cost [min], Quality [max], Delivery [min], Reliability [max]
        # S1 to S5
        self.suppliers = ["S1", "S2", "S3", "S4", "S5"]
        self.criteria_names = ["Cost", "Quality", "Delivery", "Reliability"]
        self.raw_matrix = np.array([
            [120.0, 8.0, 10.0, 92.0],
            [100.0, 7.0, 14.0, 88.0],
            [140.0, 9.0,  7.0, 95.0],
            [ 90.0, 6.0, 18.0, 80.0],
            [130.0, 8.0,  9.0, 90.0]
        ])
        self.weights = np.array([0.3, 0.3, 0.2, 0.2])
        self.is_beneficial = np.array([False, True, False, True]) # Quality & Reliability are Max

    def get_decision_matrix(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns raw matrix, weights, and boolean beneficial mask."""
        return self.raw_matrix, self.weights, self.is_beneficial


# --- P13: Multi-task learning (Differentiable) ---
class ProblemP13(BaseProblem):
    name = "P13_MultiTaskLearning"
    n_vars = 2
    n_objs = 2
    xl = np.array([-10.0, -10.0])
    xu = np.array([10.0, 10.0])

    def __init__(self):
        self.a = np.array([3.0, 3.0])
        self.b = np.array([-3.0, -3.0])

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        f1 = float(np.sum((x - self.a) ** 2))
        f2 = float(np.sum((x - self.b) ** 2))
        return np.array([f1, f2]), 0.0

    def gradients(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g1 = 2.0 * (x - self.a)
        g2 = 2.0 * (x - self.b)
        return g1, g2


# --- P14: Laptop design with decision-maker preference ---
class ProblemP14(BaseProblem):
    name = "P14_LaptopDesign"
    n_vars = 2
    n_objs = 2
    xl = np.array([13.0, 40.0]) # screen size x1, battery capacity x2
    xu = np.array([17.0, 90.0])
    ref_point = np.array([2.5, -5.0]) # Weight <= 2.5 kg, runtime >= 5h (-runtime <= -5)
    dm_reference = np.array([1.8, 12.0]) # 1.8 kg, 12 hours

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        x1, x2 = float(x[0]), float(x[1])
        # f1 = weight (kg) to minimize
        f1 = 0.4 + 0.05 * x1 + 0.012 * x2
        # runtime (hours) to maximize => minimize -runtime
        runtime = (0.2 * x2) / (0.5 + 0.03 * x1)
        f2 = -runtime
        return np.array([f1, f2]), 0.0

    def true_pareto_front(self, n_points: int = 100) -> np.ndarray:
        # All Pareto points have x1 = 13 (minimum screen size gives minimal weight & maximal runtime)
        x2 = np.linspace(40.0, 90.0, n_points)
        f1 = 0.4 + 0.05 * 13.0 + 0.012 * x2
        runtime = (0.2 * x2) / (0.5 + 0.03 * 13.0)
        return np.column_stack([f1, -runtime])

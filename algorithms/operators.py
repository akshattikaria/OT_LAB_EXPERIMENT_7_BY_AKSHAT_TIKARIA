"""
operators.py - Standard Evolutionary & Swarm Operators
======================================================
Implements:
- SBX (Simulated Binary Crossover)
- Polynomial Mutation
- Real-valued Differential Evolution (DE) mutation/crossover
- Permutation crossover (Order Crossover OX) & mutation (Swap / Inversion) for TSP
- Binary crossover & bit-flip mutation for feature selection
"""

import numpy as np
from typing import Tuple, List, Any, Optional


def sbx_crossover(parent1: np.ndarray, parent2: np.ndarray,
                  xl: np.ndarray, xu: np.ndarray,
                  eta_c: float = 20.0, pc: float = 0.9,
                  rng: Any = None) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX) for continuous variables."""
    if rng is None:
        rng = np.random
    _rand = lambda: float(rng.rand()) if hasattr(rng, 'rand') else float(rng.random())

    if _rand() > pc:
        return parent1.copy(), parent2.copy()

    n_vars = len(parent1)
    child1 = np.empty(n_vars)
    child2 = np.empty(n_vars)

    for i in range(n_vars):
        if _rand() <= 0.5 and abs(parent1[i] - parent2[i]) > 1e-14:
            y1 = min(parent1[i], parent2[i])
            y2 = max(parent1[i], parent2[i])
            yl = xl[i]
            yu = xu[i]

            rand = _rand()

            # Beta q for child 1
            beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
            alpha = 2.0 - beta ** -(eta_c + 1.0)
            if rand <= (1.0 / alpha):
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
            c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

            # Beta q for child 2
            beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
            alpha = 2.0 - beta ** -(eta_c + 1.0)
            if rand <= (1.0 / alpha):
                betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
            else:
                betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
            c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

            c1 = min(max(c1, yl), yu)
            c2 = min(max(c2, yl), yu)

            if _rand() <= 0.5:
                child1[i] = c2
                child2[i] = c1
            else:
                child1[i] = c1
                child2[i] = c2
        else:
            child1[i] = parent1[i]
            child2[i] = parent2[i]

    return child1, child2


def polynomial_mutation(ind: np.ndarray, xl: np.ndarray, xu: np.ndarray,
                        eta_m: float = 20.0, pm: Optional[float] = None,
                        rng: Any = None) -> np.ndarray:
    """Polynomial Mutation for continuous variables."""
    if rng is None:
        rng = np.random
    _rand = lambda: float(rng.rand()) if hasattr(rng, 'rand') else float(rng.random())

    n_vars = len(ind)
    if pm is None:
        pm = 1.0 / n_vars

    mutated = ind.copy()
    for i in range(n_vars):
        if _rand() <= pm:
            y = ind[i]
            yl = xl[i]
            yu = xu[i]
            delta1 = (y - yl) / (yu - yl + 1e-14)
            delta2 = (yu - y) / (yu - yl + 1e-14)
            rand = _rand()
            mut_pow = 1.0 / (eta_m + 1.0)

            if rand <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                deltaq = (val ** mut_pow) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                deltaq = 1.0 - (val ** mut_pow)

            y = y + deltaq * (yu - yl)
            mutated[i] = min(max(y, yl), yu)

    return mutated


def order_crossover_ox(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Order Crossover (OX) for permutation representations (e.g. TSP)."""
    n = len(parent1)
    pt1, pt2 = sorted(np.random.choice(n, size=2, replace=False))

    def ox_single(p1, p2):
        child = np.full(n, -1)
        child[pt1:pt2 + 1] = p1[pt1:pt2 + 1]
        p2_remaining = [x for x in np.roll(p2, -pt2 - 1) if x not in child]
        idx = (pt2 + 1) % n
        for val in p2_remaining:
            child[idx] = val
            idx = (idx + 1) % n
        return child

    return ox_single(parent1, parent2), ox_single(parent2, parent1)


def swap_mutation(perm: np.ndarray, pm: float = 0.2) -> np.ndarray:
    """Swap mutation for permutations."""
    child = perm.copy()
    if np.random.rand() <= pm:
        i, j = np.random.choice(len(perm), size=2, replace=False)
        child[i], child[j] = child[j], child[i]
    return child


def inversion_mutation(perm: np.ndarray, pm: float = 0.2) -> np.ndarray:
    """Inversion (2-opt) mutation for permutations."""
    child = perm.copy()
    if np.random.rand() <= pm:
        i, j = sorted(np.random.choice(len(perm), size=2, replace=False))
        child[i:j + 1] = child[i:j + 1][::-1]
    return child


def binary_crossover(p1: np.ndarray, p2: np.ndarray, pc: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform crossover for binary representations."""
    if np.random.rand() > pc:
        return p1.copy(), p2.copy()
    mask = np.random.rand(len(p1)) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1, c2


def binary_mutation(ind: np.ndarray, pm: Optional[float] = None) -> np.ndarray:
    """Bit-flip mutation for binary vectors."""
    n = len(ind)
    if pm is None:
        pm = 1.0 / n
    flip = np.random.rand(n) < pm
    mutated = ind.copy()
    mutated[flip] = 1 - mutated[flip]
    # Ensure at least one bit is 1
    if np.sum(mutated) == 0:
        mutated[np.random.randint(n)] = 1
    return mutated

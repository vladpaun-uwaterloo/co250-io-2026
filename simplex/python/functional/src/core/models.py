from dataclasses import dataclass

import numpy as np

@dataclass
class SEFLinearProgram:
    """
    max c^T x + objective_offset
    s.t. Ax = b
         x >= 0
    """

    c: np.ndarray
    A: np.ndarray
    b: np.ndarray

    objective_offset: float = 0.0


@dataclass(frozen=True)
class Basis:
    indices: tuple[int, ...]


@dataclass(frozen=True)
class OptimalResult:
    solution: np.ndarray
    objective_value: float
    basis: Basis
    certificate: np.ndarray


@dataclass(frozen=True)
class UnboundedResult:
    solution: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class InfeasibleResult:
    certificate: np.ndarray
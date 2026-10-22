from dataclasses import dataclass
from enum import Enum

import numpy as np


class ObjectiveSense(Enum):
    MAX = "max"
    MIN = "min"


class ConstraintSense(Enum):
    LE = "<="
    EQ = "="
    GE = ">="


@dataclass
class LinearProgram:
    """An arbitrary linear program."""

    c: np.ndarray
    A: np.ndarray
    b: np.ndarray

    objective_sense: ObjectiveSense
    constraint_senses: list[ConstraintSense]

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    objective_offset: float = 0.0


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
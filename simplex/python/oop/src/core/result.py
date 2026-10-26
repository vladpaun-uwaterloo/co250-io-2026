from dataclasses import dataclass

import numpy as np

from .basis import Basis


class SolveResult:
    pass


@dataclass
class OptimalResult(SolveResult):
    solution: np.ndarray
    objective_value: float
    basis: Basis
    certificate: np.ndarray


@dataclass
class UnboundedResult(SolveResult):
    solution: np.ndarray
    direction: np.ndarray


@dataclass
class InfeasibleResult(SolveResult):
    certificate: np.ndarray
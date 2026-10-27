from dataclasses import dataclass

import numpy as np

from .basis import Basis


@dataclass
class GeneralCertificate:
    constraint_multipliers: np.ndarray
    lower_bound_multipliers: np.ndarray
    upper_bound_multipliers: np.ndarray


class SolveResult:
    pass


@dataclass
class OptimalResult(SolveResult):
    solution: np.ndarray
    objective_value: float
    basis: Basis
    certificate: np.ndarray | GeneralCertificate


@dataclass
class UnboundedResult(SolveResult):
    solution: np.ndarray
    direction: np.ndarray


@dataclass
class InfeasibleResult(SolveResult):
    certificate: np.ndarray | GeneralCertificate
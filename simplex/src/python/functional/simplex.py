from models import SEFLinearProgram, OptimalResult, InfeasibleResult, UnboundedResult, Basis
import numpy as np
from transform import build_aux

TOL = 1e-9


def simplex(lp: SEFLinearProgram, basis: Basis) -> OptimalResult |  UnboundedResult:
    return UnboundedResult(np.array([1,2]),np.array([1,2]))

def phase_one(lp: SEFLinearProgram) -> InfeasibleResult | tuple[SEFLinearProgram, Basis]:
    A = lp.A.copy()
    b = lp.b.copy()
    c = lp.c.copy()
    n = A.shape[1]
    # Build Auxiliary
    for index, rhs in enumerate(list(b)):
        if rhs < 0:
            b[index] *= -1
            A[index,:] *= -1

    normalized_lp = SEFLinearProgram(c,A,b)
    aux_lp, init_aux_basis = build_aux(normalized_lp)

    # Run Simplex on Aux
    result = simplex(aux_lp, init_aux_basis)
    assert isinstance(result, OptimalResult)

    # Output infeasibility or Basis
    # TODO: Find the cert
    if result.objective_value < -TOL: return InfeasibleResult(np.array([0]))

    # TODO: Find the basis
    return _remove_artificials(normalized_lp, aux_lp, result.basis)

def _remove_artificials(normalized_lp: SEFLinearProgram, aux_lp: SEFLinearProgram, aux_bfs: Basis) -> tuple[SEFLinearProgram, Basis]:
    
    
    return (...)

def phase_two(lp: SEFLinearProgram, basis: Basis) -> UnboundedResult | OptimalResult:
    return simplex(lp,basis)
    
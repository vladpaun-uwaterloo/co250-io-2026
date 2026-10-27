from .models import SEFLinearProgram, OptimalResult, UnboundedResult, InfeasibleResult, Basis
from .simplex import simplex, phase_one, phase_two


def solve (lp: SEFLinearProgram) -> OptimalResult | UnboundedResult | InfeasibleResult:

    result = phase_one(lp)
    if isinstance(result, InfeasibleResult): return result

    lp, basis = result

    return simplex(lp, basis)
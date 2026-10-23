from models import LinearProgram, SEFLinearProgram, OptimalResult, UnboundedResult, InfeasibleResult, Basis
from transform import to_normalized_sef
from simplex import simplex, phase_one, phase_two


def solve (lp: LinearProgram | SEFLinearProgram) -> OptimalResult | UnboundedResult | InfeasibleResult:
    if isinstance(lp, LinearProgram):
        lp = to_normalized_sef(lp)

    result = phase_one(lp)
    if isinstance(result, InfeasibleResult): return result

    lp, basis = result

    return simplex(lp, basis)
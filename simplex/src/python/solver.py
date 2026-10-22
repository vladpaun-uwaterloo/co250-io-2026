from models import LinearProgram, SEFLinearProgram, OptimalResult, UnboundedResult, InfeasibleResult
from transform import to_sef, canonicalize
from simplex import simplex

def solve (lp: LinearProgram | SEFLinearProgram) -> OptimalResult | UnboundedResult | InfeasibleResult:
    if isinstance(lp, LinearProgram):
        lp = to_sef(lp)

    lp = canonicalize(lp)

    return simplex(lp)
from models import SEFLinearProgram, OptimalResult, InfeasibleResult, UnboundedResult

def simplex(lp: SEFLinearProgram) -> OptimalResult | InfeasibleResult |  UnboundedResult:
    return

def phase1():
    # find a feasible basis and BFS or ouput unfeasibility cert
    return

def phase2():
    # given BFS find optimal sol or give unboundedness cert
    return
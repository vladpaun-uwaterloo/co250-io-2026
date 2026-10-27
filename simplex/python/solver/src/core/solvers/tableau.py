from .base import SimplexSolver
from ..models import SEFLinearProgram, SolveResult


class TableauSimplexSolver(SimplexSolver):
    def solve_sef(self, lp: SEFLinearProgram) -> SolveResult:
        raise NotImplementedError("Tableau simplex has not been implemented yet.")
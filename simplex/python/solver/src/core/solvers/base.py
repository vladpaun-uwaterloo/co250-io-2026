from abc import ABC, abstractmethod

from ..models import LinearProgram, SEFLinearProgram, SolveResult


class SimplexSolver(ABC):
    def __init__(self, tol: float = 1e-9):
        self.tol = tol

    def solve(self, lp: LinearProgram | SEFLinearProgram) -> SolveResult:
        if isinstance(lp, LinearProgram):
            return self._solve_general(lp)

        return self.solve_sef(lp.copy())

    def _solve_general(self, lp: LinearProgram) -> SolveResult:
        sef_lp, transformation = lp.to_sef()

        # The solver may freely modify its SEF problem.
        # Keep the original SEF formulation because the returned certificate
        # refers to this formulation when mapping back to the general LP.
        original_sef_lp = sef_lp.copy()

        result = self.solve_sef(sef_lp)

        return transformation.map_result_to_original(original_sef_lp, result)

    @abstractmethod
    def solve_sef(self, lp: SEFLinearProgram) -> SolveResult:
        pass
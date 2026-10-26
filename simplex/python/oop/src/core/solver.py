from .result import SolveResult, OptimalResult, InfeasibleResult, UnboundedResult
from .linear_program import SEFLinearProgram
from .basis import Basis
import numpy as np


class SimplexSolver:
    def __init__(self, tol: float = 1e-9):
        self.tol = tol

    def solve(self,lp: SEFLinearProgram,) -> SolveResult:
        result = self.phase_one(lp)
        if isinstance(result, InfeasibleResult): return result
        lp, basis = result

        return self.simplex(lp, basis)


    def phase_one(
        self,
        lp: SEFLinearProgram,
    ) -> InfeasibleResult | tuple[SEFLinearProgram, Basis]:
        aux_lp, init_aux_basis = lp.normalize().build_auxiliary()

        result = self.simplex(aux_lp, init_aux_basis)
        assert isinstance(result, OptimalResult)

        if result.objective_value < -self.tol:
            certificate = lp.reverse_normalization_certificate(result.certificate)
            return InfeasibleResult(certificate)
        return self._remove_artificials(lp, aux_lp, result.basis)

    def phase_two(self,lp: SEFLinearProgram,basis: Basis) -> OptimalResult | UnboundedResult:
        return self.simplex(lp, basis)

    def simplex(self,lp: SEFLinearProgram,basis: Basis) -> OptimalResult | UnboundedResult:
        original_lp = lp.copy()

        while True:
            lp.canonicalize(basis)

            m,n = lp.A.shape
            bfs = np.zeros(n)
            bfs[basis.indices] = lp.b

            if np.all(lp.c <= self.tol):
                return self._generate_optimal_result(original_lp, lp, basis, bfs)

            entering = next(i for i, ci in enumerate(lp.c) if ci > self.tol and i not in basis.indices)

            if np.all(lp.A[:, entering] <= self.tol):
                return self._generate_unbounded_result(lp,basis,bfs,entering)

            ts = []
            for i in range(m):
                if lp.A[i,entering] > self.tol:
                    ts.append((i,lp.b[i] / lp.A[i,entering]))
    
            min_t = min(t for _,t in ts)
            leaving_candidates = [i for i,t in ts if abs(t - min_t) <= self.tol]
    
            leaving = min(leaving_candidates, key=lambda i: basis.indices[i])
            basis.replace(leaving, entering)

            

    def _remove_artificials(self,lp: SEFLinearProgram,aux_lp: SEFLinearProgram,basis: Basis) -> tuple[SEFLinearProgram, Basis]:
        _, n = lp.A.shape

        while True:
            aux_lp.canonicalize(basis)

            r = next(
                (r for r,j in enumerate(basis.indices) if j >= n),
                None
            )

            if r is None:
                clean_lp = SEFLinearProgram(
                    c=lp.c.copy(),
                    A=aux_lp.A[:, :n].copy(),
                    b=aux_lp.b.copy(),
                    objective_offset=lp.objective_offset,
                )
    
                return clean_lp, basis
    
            row = aux_lp.A[r,:]
            
            basic = set(basis.indices)
            entering = next(
                (j for j in range(n) if abs(row[j]) > self.tol and j not in basic),
                None
            )
    
            if entering is None:
                aux_lp.remove_constraint(r)
                basis.remove(r)                
            else:
                basis.replace(r, entering)

    def _generate_optimal_result(self,original_lp: SEFLinearProgram,canonical_lp: SEFLinearProgram,basis: Basis,bfs: np.ndarray) -> OptimalResult:        
        A_B = original_lp.A[:,basis.indices]
        c_B = original_lp.c[basis.indices]
        y = np.linalg.solve(A_B.T, c_B)
        
        return OptimalResult(
            solution=bfs,
            objective_value=canonical_lp.objective_offset,
            basis=basis,
            certificate=y
        )

    def _generate_unbounded_result(self,lp: SEFLinearProgram,basis: Basis,bfs: np.ndarray,entering: int) -> UnboundedResult:
        d = np.zeros(lp.A.shape[1])
        d[entering] = 1.0
        d[basis.indices] = -lp.A[:,entering]
        
        return UnboundedResult(
            solution=bfs,
            direction=d
        )
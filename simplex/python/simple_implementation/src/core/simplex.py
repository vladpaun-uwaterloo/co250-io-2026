from .models import SEFLinearProgram, OptimalResult, InfeasibleResult, UnboundedResult, Basis
import numpy as np
from .transform import build_aux, canonicalize, normalize

TOL = 1e-9


def simplex(lp: SEFLinearProgram, basis: Basis) -> OptimalResult |  UnboundedResult:
    original_lp = SEFLinearProgram(
        c=lp.c.copy(),
        A=lp.A.copy(),
        b=lp.b.copy(),
        objective_offset=lp.objective_offset,
    )
    basis_indices = list(basis.indices)

    while True:
        basis = Basis(tuple(basis_indices))
        lp, basis = canonicalize(lp, basis)
        basis_indices = list(basis.indices)

        m,n = lp.A.shape

        bfs = np.zeros(n)
        bfs[list(basis_indices)] = lp.b

        if np.all(lp.c <= TOL):
            return _generate_optimal_cert(original_lp, lp,basis,bfs)


        entering = next(
            i
            for i, ci in enumerate(lp.c)
            if ci > TOL and i not in basis_indices
        )

        if np.all(lp.A[:, entering] <= TOL):
            return _generate_unbounded_cert(lp, basis, bfs, entering)
        
        ts = []
        for i in range(m):
            if lp.A[i,entering] > TOL:
                ts.append((i,lp.b[i] / lp.A[i,entering]))

        min_t = min(t for _,t in ts)
        leaving_candidates = [i for i,t in ts if abs(t - min_t) <= TOL]

        leaving = min(leaving_candidates, key=lambda i: basis_indices[i])
        basis_indices[leaving] = entering

def phase_one(lp: SEFLinearProgram) -> InfeasibleResult | tuple[SEFLinearProgram, Basis]:

    # normalize
    lp, flipped_rows = normalize(lp)
    
    # Build Auxiliary
    aux_lp, init_aux_basis = build_aux(lp)

    # Run Simplex on Aux
    result = simplex(aux_lp, init_aux_basis)
    assert isinstance(result, OptimalResult)

    # Output infeasibility or Basis
    if result.objective_value < -TOL:
        certificate = result.certificate.copy()
        for row in flipped_rows:certificate[row] *= -1
        return InfeasibleResult(certificate)
    return _remove_artificials(lp, aux_lp, result.basis)

def _remove_artificials(lp: SEFLinearProgram, aux_lp: SEFLinearProgram, aux_basis: Basis) -> tuple[SEFLinearProgram, Basis]:

    n = lp.A.shape[1]

    basis_indices = list(aux_basis.indices)

    while True:
        basis = Basis(tuple(basis_indices))
        aux_lp, basis = canonicalize(aux_lp, basis)
        

        # get the index of the first artificial variable in the basis
        artificial_row_index = next(
            (r for r,j in enumerate(basis_indices) if j >= n),
            None
        )

        # If no artificial variable in basis, then we are done
        if artificial_row_index is None:
            clean_lp = SEFLinearProgram(
                c=lp.c.copy(),
                A=aux_lp.A[:, :n].copy(),
                b=aux_lp.b.copy(),
                objective_offset=lp.objective_offset,
            )

            return clean_lp, basis

        r = artificial_row_index
        assert abs(aux_lp.b[r]) <= TOL

        # get the row of constraints where this artificial variable has the coefficient 1
        # This corresponds to the index into basis_indices because the i-th row of A will always have a 1 in the place of the ith basic variable
        row = aux_lp.A[r,:]
        
        #  Find any other non-aux variable in that row that has a non-zero coefficient 
        basic = set(basis_indices)
        entering = next(
            (j for j in range(n) if abs(row[j]) > TOL and j not in basic),
            None
        )

        # If no original variable can enter, this constraint row is redundant.
        if entering is None:
            aux_lp.A = np.delete(aux_lp.A, r, axis=0)
            aux_lp.b = np.delete(aux_lp.b, r)

            del basis_indices[r]
            
        # Otherwise pivot to this
        else:
            basis_indices[artificial_row_index] = entering

def _generate_optimal_cert(original_lp: SEFLinearProgram, canonical_lp: SEFLinearProgram, basis: Basis, bfs: np.ndarray) -> OptimalResult:
    indices = list(basis.indices)

    A_B = original_lp.A[:,indices]
    c_B = original_lp.c[indices]
    y = np.linalg.solve(A_B.T, c_B)

    return OptimalResult(
        solution=bfs,
        objective_value=canonical_lp.objective_offset,
        basis=basis,
        certificate=y
    )

def _generate_unbounded_cert(lp: SEFLinearProgram,basis: Basis,bfs: np.ndarray,k: int) -> UnboundedResult:
    d = np.zeros(lp.A.shape[1])
    d[k] = 1.0
    d[list(basis.indices)] = -lp.A[:,k]

    return UnboundedResult(
        solution=bfs,
        direction=d
    )

def phase_two(lp: SEFLinearProgram, basis: Basis) -> UnboundedResult | OptimalResult:
    return simplex(lp,basis)
    
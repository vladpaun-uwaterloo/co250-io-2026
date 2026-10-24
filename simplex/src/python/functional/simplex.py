from models import SEFLinearProgram, OptimalResult, InfeasibleResult, UnboundedResult, Basis
import numpy as np
from transform import build_aux, canonicalize

TOL = 1e-9


def simplex(lp: SEFLinearProgram, basis: Basis) -> OptimalResult |  UnboundedResult:
    original_lp = SEFLinearProgram(
        c=lp.c.copy(),
        A=lp.A.copy(),
        b=lp.b.copy(),
        objective_offset=lp.objective_offset,
    )

    while True:
        lp, basis = canonicalize(lp, basis)

        m,n = lp.A.shape

        A = lp.A.copy()
        b = lp.b.copy()
        c = lp.c.copy()

        basis_indices = basis.indices

        bfs = np.zeros(n)
        bfs[list(basis_indices)] = b

        if np.all(c <= TOL):
            return _generate_optimal_cert(original_lp, lp,basis,bfs)


        indices = [i for i, ci in enumerate(c) if ci > TOL]

        k = indices[0]

        if np.all(A[:, k] <= TOL):
            return _generate_unbounded_cert(lp, basis, bfs, k)
        
        ts = []
        for i in range(b.shape[0]):
            if A[i,k] > 0:
                ts.append((i,b[i] / A[i,k]))
        r,t = min(ts, key=lambda x: x[1])

        new_basis = list(basis_indices)
        new_basis[r] = k
        basis = Basis(tuple(new_basis))

# This assumes a normalized LP
def phase_one(lp: SEFLinearProgram) -> InfeasibleResult | tuple[SEFLinearProgram, Basis]:
    
    # Build Auxiliary
    assert np.all(lp.b > 0)
    aux_lp, init_aux_basis = build_aux(lp)

    # Run Simplex on Aux
    result = simplex(aux_lp, init_aux_basis)
    assert isinstance(result, OptimalResult)

    # Output infeasibility or Basis
    if result.objective_value < -TOL: return InfeasibleResult(result.certificate)
    return _remove_artificials(lp, aux_lp, result.basis)

def _remove_artificials(lp: SEFLinearProgram, aux_lp: SEFLinearProgram, aux_basis: Basis) -> tuple[SEFLinearProgram, Basis]:

    # n = lp.A.shape[1]
    # aux_lp, basis = canonicalize(aux_lp, aux_basis)

    # while True:
    #     basis_indices = list(basis.indices)

    #     # Artificial variables have indices n, n+1, ...
    #     artificial_row = next(
    #         (
    #             r
    #             for r, j in enumerate(basis_indices)
    #             if j >= n
    #         ),
    #         None,
    #     )

    #     # No artificial variables remain in the basis.
    #     if artificial_row is None:
    #         return lp, basis

    #     r = artificial_row

    #     basic = set(basis_indices)
    #     k = next(
    #         (
    #             j
    #             for j in range(n)
    #             if j not in basic
    #             and abs(aux_lp.A[r, j]) > TOL
    #         ),
    #         None,
    #     )

    #     if k is None:
    #         raise NotImplementedError(
    #             "Redundant constraint encountered during Phase I cleanup."
    #         )

    #     # Degenerate pivot:
    #     # artificial variable leaves, original variable enters.
    #     basis_indices[r] = k
    #     basis = Basis(tuple(basis_indices))

    #     # Recanonicalize with respect to the new basis.
    #     aux_lp, basis = canonicalize(aux_lp, basis)

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
    
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


        indices = [i for i, ci in enumerate(c) if ci > 0]

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

def phase_one(lp: SEFLinearProgram) -> InfeasibleResult | tuple[SEFLinearProgram, Basis]:
    A = lp.A.copy()
    b = lp.b.copy()
    c = lp.c.copy()
    n = A.shape[1]
    # Build Auxiliary
    aux_lp, init_aux_basis = build_aux(lp)

    # Run Simplex on Aux
    result = simplex(aux_lp, init_aux_basis)
    assert isinstance(result, OptimalResult)

    # Output infeasibility or Basis
    if result.objective_value < -TOL: return InfeasibleResult(result.certificate)
    return _remove_artificials(lp, aux_lp, result.basis)

def _remove_artificials(lp: SEFLinearProgram, aux_lp: SEFLinearProgram, aux_basis: Basis) -> tuple[SEFLinearProgram, Basis]:
    # TODO
    return (aux_lp, aux_basis)

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
    
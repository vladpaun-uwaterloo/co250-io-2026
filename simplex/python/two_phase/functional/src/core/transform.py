import numpy as np

from .models import (
    SEFLinearProgram,
    Basis
)

def normalize(lp: SEFLinearProgram) -> tuple[SEFLinearProgram, list[int]]:

    A = lp.A.copy()
    b = lp.b.copy()

    flipped_rows = []

    for i in range(len(b)):
        if b[i] < 0:
            A[i, :] *= -1
            b[i] *= -1

            flipped_rows.append(i)

    normalized_lp = SEFLinearProgram(
        c=lp.c.copy(),
        A=A,
        b=b,
        objective_offset=lp.objective_offset,
    )

    return normalized_lp, flipped_rows


def canonicalize(lp: SEFLinearProgram, basis: Basis) -> tuple[SEFLinearProgram, Basis]:
    A = lp.A.copy()
    b = lp.b.copy()
    c = lp.c.copy()

    indices = list(basis.indices)

    A_B = A[:,indices]
    c_B = c[indices]

    # Canonicalize constraints:
    # A_bar = A_B^-1 A
    # b_bar = A_B^-1 b
    A_bar = np.linalg.solve(A_B,A)
    b_bar = np.linalg.solve(A_B,b)

    y = np.linalg.solve(A_B.T, c_B)

    c_bar = c - A.T @ y

    objective_offset = (lp.objective_offset + float(b @ y))

    lp = SEFLinearProgram(
        c_bar,
        A_bar,
        b_bar,
        objective_offset
    )
    return lp, basis

def build_aux(lp: SEFLinearProgram) -> tuple[SEFLinearProgram, Basis]:
    m, n = lp.A.shape
    c = np.append(np.zeros(n), np.full(m, -1))
    A = np.hstack((lp.A.copy(), np.identity(m)))

    basis = Basis(tuple(range(n, n+m)))
    aux = SEFLinearProgram(c,A,lp.b.copy())
    return aux, basis
import numpy as np

from .models import (
    LinearProgram,
    SEFLinearProgram,
    ObjectiveSense,
    ConstraintSense,
    Basis
)


def to_sef(lp: LinearProgram) -> SEFLinearProgram:
    A = np.array(lp.A, dtype=float, copy=True)
    b = np.array(lp.b, dtype=float, copy=True)
    c = np.array(lp.c, dtype=float, copy=True)
    objective_offset = lp.objective_offset

    constraint_senses = list(lp.constraint_senses)

    # 1. Convert objective to maximization
    if lp.objective_sense == ObjectiveSense.MIN:
        c *= -1
        objective_offset *= -1

    # 2. Convert all variables to nonnegative variables
    new_columns = []
    new_c = []

    # Additional constraints created by finite upper bounds:
    # (variable_index, upper_bound)
    upper_bound_constraints = []   

    for j, (lower_bound, upper_bound) in enumerate(
        zip(lp.lower_bounds, lp.upper_bounds)
    ):

        column = A[:, j]
        coefficient = c[j]

        # Case 1:
        # L <= x or L <= x <= U
        #
        # Substitute:
        # x = L + x'
        # x' >= 0
        if np.isfinite(lower_bound):
            b -= column * lower_bound
            objective_offset += lower_bound * coefficient

            new_c.append(coefficient)
            new_columns.append(column)

            #  Add a new constraint (row in A), we ll materialize these later
            if np.isfinite(upper_bound):
                upper_bound_constraints.append((len(new_columns) - 1 ,upper_bound - lower_bound))

        # Case 2:
        # -inf < x <= U
        #
        # Substitute:
        # x = U - x'
        # x' >= 0
        elif not np.isfinite(lower_bound) and np.isfinite(upper_bound):
            b -= column * upper_bound
            objective_offset += upper_bound * coefficient

            new_c.append(-coefficient)
            new_columns.append(-column)

        # Case 3:
        # x is free
        #
        # x = x+ - x-
        # x+, x- >= 0
        else:
            new_c.append(coefficient)
            new_c.append(-coefficient)

            new_columns.append(column)
            new_columns.append(-column)
            

    A = np.column_stack(new_columns)
    c = np.array(new_c, dtype=float)

    # 3. Add constraints created by finite upper bounds
    for variable_index, bound in upper_bound_constraints:
        row = np.zeros(A.shape[1])
        row[variable_index] = 1

        A = np.vstack((A, row))
        b = np.append(b, bound)

        constraint_senses.append(ConstraintSense.LE)

    # 4. Convert all inequalities to equalities
    #
    # <= : Ax + s = b
    # >= : Ax - s = b
    #  = : unchangeds([
    num = sum([sense != ConstraintSense.EQ for sense in constraint_senses])
    if num > 0:
        slack_columns = np.zeros((A.shape[0], num))
        
        index = 0
        for row, sense in enumerate(constraint_senses):
            if sense == ConstraintSense.EQ: continue
            slack_columns[row, index] = 1 if sense == ConstraintSense.LE else -1
            index += 1

        A = np.hstack((A,slack_columns))
        c = np.append(c, np.zeros(num))
    


    return SEFLinearProgram(
        c=c,
        A=A,
        b=b,
        objective_offset=objective_offset,
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
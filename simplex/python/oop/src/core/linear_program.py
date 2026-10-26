from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Self

import numpy as np

from .basis import Basis


class ObjectiveSense(Enum):
    MAX = auto()
    MIN = auto()


class ConstraintSense(Enum):
    LE = auto()
    EQ = auto()
    GE = auto()


@dataclass
class LinearProgram:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray

    objective_sense: ObjectiveSense
    constraint_senses: list[ConstraintSense]

    lower_bounds: np.ndarray
    upper_bounds: np.ndarray

    objective_offset: float = 0.0

    def to_sef(self) -> SEFLinearProgram:
        A = self.A.copy()
        b = self.b.copy()
        c = self.c.copy()

        objective_offset = self.objective_offset
        constraint_senses = list(self.constraint_senses)

        # 1. Convert objective to maximization
        if self.objective_sense == ObjectiveSense.MIN:
            c *= -1
            objective_offset *= -1

        # 2. Convert all variables to nonnegative variables
        new_columns = []
        new_c = []

        # Additional constraints created by finite upper bounds:
        # (variable_index, upper_bound)
        upper_bound_constraints = []

        for j, (lower_bound, upper_bound) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
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

                if np.isfinite(upper_bound):
                    upper_bound_constraints.append((len(new_columns) - 1, upper_bound - lower_bound))

            # Case 2:
            # -inf < x <= U
            #
            # Substitute:
            # x = U - x'
            # x' >= 0
            elif np.isfinite(upper_bound):
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
        #  = : unchanged
        num = sum(sense != ConstraintSense.EQ for sense in constraint_senses)

        if num > 0:
            slack_columns = np.zeros((A.shape[0], num))

            index = 0
            for row, sense in enumerate(constraint_senses):
                if sense == ConstraintSense.EQ:
                    continue

                slack_columns[row, index] = 1 if sense == ConstraintSense.LE else -1
                index += 1

            A = np.hstack((A, slack_columns))
            c = np.append(c, np.zeros(num))

        return SEFLinearProgram(
            c=c,
            A=A,
            b=b,
            objective_offset=objective_offset,
        )


@dataclass
class SEFLinearProgram:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    objective_offset: float = 0.0

    _row_transform: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._row_transform = np.eye(self.A.shape[0])

    def copy(self) -> SEFLinearProgram:
        result = SEFLinearProgram(
            c=self.c.copy(),
            A=self.A.copy(),
            b=self.b.copy(),
            objective_offset=self.objective_offset,
        )

        result._row_transform = self._row_transform.copy()

        return result

    def normalize(self) -> Self:
        for i in range(len(self.b)):
            if self.b[i] < 0:
                self.A[i, :] *= -1
                self.b[i] *= -1

                # Apply the same row operation to R
                self._row_transform[i, :] *= -1

        return self

    def canonicalize(self, basis: Basis) -> Self:

        A_B = self.A[:, basis.indices]
        c_B = self.c[basis.indices]

        # A_bar = A_B^-1 A
        # b_bar = A_B^-1 b
        A_bar = np.linalg.solve(A_B, self.A)
        b_bar = np.linalg.solve(A_B, self.b)

        # Apply exactly the same row operations to R
        row_transform_bar = np.linalg.solve(A_B, self._row_transform)

        # Solve A_B^T y = c_B
        y = np.linalg.solve(A_B.T, c_B)

        c_bar = self.c - self.A.T @ y

        self.objective_offset += float(self.b @ y)

        self.A = A_bar
        self.b = b_bar
        self.c = c_bar
        self._row_transform = row_transform_bar

        return self

    def build_auxiliary(self) -> tuple[SEFLinearProgram, Basis]:
        m, n = self.A.shape

        aux_c = np.concatenate((np.zeros(n), -np.ones(m)))
        aux_A = np.hstack((self.A.copy(), np.eye(m)))

        aux_lp = SEFLinearProgram(
            c=aux_c,
            A=aux_A,
            b=self.b.copy(),
            objective_offset=0.0,
        )

        # The auxiliary LP starts from the same current row representation
        aux_lp._row_transform = self._row_transform.copy()

        basis = Basis(range(n, n + m))

        return aux_lp, basis

    def remove_constraint(self, row: int) -> Self:

        self.A = np.delete(self.A, row, axis=0)
        self.b = np.delete(self.b, row)
        self._row_transform = np.delete(self._row_transform, row, axis=0)

        return self

    def adopt_constraint_system(self, other: SEFLinearProgram, number_of_columns: int) -> Self:
        self.A = other.A[:, :number_of_columns].copy()
        self.b = other.b.copy()
        self._row_transform = other._row_transform.copy()

        return self

    def map_certificate_to_original(self, certificate: np.ndarray) -> np.ndarray:
        # If A_current = R A_original, then
        #
        # A_current^T y
        # = A_original^T R^T y
        #
        # so the original certificate is R^T y.
        return self._row_transform.T @ certificate
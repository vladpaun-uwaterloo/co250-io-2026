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


class VariableTransformKind(Enum):
    LOWER_SHIFT = auto()
    UPPER_SHIFT = auto()
    FREE = auto()


@dataclass
class VariableTransformation:
    kind: VariableTransformKind
    indices: tuple[int, ...]
    upper_bound_row: int = -1


@dataclass
class SEFTransformation:
    variable_offset: np.ndarray
    variable_matrix: np.ndarray
    variable_transformations: list[VariableTransformation]

    transformed_variable_count: int
    original_constraint_count: int

    # The SEF objective is:
    #
    # objective_multiplier * original objective
    #
    # MAX -> +1
    # MIN -> -1
    objective_multiplier: float

    def map_solution_to_original(self, solution: np.ndarray) -> np.ndarray:
        z = solution[:self.transformed_variable_count]
        return self.variable_offset + self.variable_matrix @ z

    def map_direction_to_original(self, direction: np.ndarray) -> np.ndarray:
        z_direction = direction[:self.transformed_variable_count]
        return self.variable_matrix @ z_direction

    def map_objective_value_to_original(self, objective_value: float) -> float:
        return self.objective_multiplier * objective_value


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

    def to_sef(self) -> tuple[SEFLinearProgram, SEFTransformation]:
        n = len(self.c)

        variable_offset = np.zeros(n)
        transform_columns = []
        variable_transformations = []

        # (original variable index, transformed variable index, bound)
        upper_bound_constraints = []

        # Build:
        #
        # x_original = variable_offset + variable_matrix @ z
        #
        # where z >= 0.
        for j, (lower_bound, upper_bound) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
            e_j = np.zeros(n)
            e_j[j] = 1.0

            # Case 1:
            #
            # L <= x
            #
            # x = L + z
            #
            # If U is also finite:
            #
            # z <= U - L
            if np.isfinite(lower_bound):
                variable_offset[j] = lower_bound

                transformed_index = len(transform_columns)
                transform_columns.append(e_j)

                transformation = VariableTransformation(
                    kind=VariableTransformKind.LOWER_SHIFT,
                    indices=(transformed_index,),
                )

                variable_transformations.append(transformation)

                if np.isfinite(upper_bound):
                    upper_bound_constraints.append((j, transformed_index, upper_bound - lower_bound))

            # Case 2:
            #
            # x <= U
            #
            # x = U - z
            elif np.isfinite(upper_bound):
                variable_offset[j] = upper_bound

                transformed_index = len(transform_columns)
                transform_columns.append(-e_j)

                variable_transformations.append(
                    VariableTransformation(
                        kind=VariableTransformKind.UPPER_SHIFT,
                        indices=(transformed_index,),
                    )
                )

            # Case 3:
            #
            # x is free
            #
            # x = z+ - z-
            else:
                positive_index = len(transform_columns)
                transform_columns.append(e_j)

                negative_index = len(transform_columns)
                transform_columns.append(-e_j)

                variable_transformations.append(
                    VariableTransformation(
                        kind=VariableTransformKind.FREE,
                        indices=(positive_index, negative_index),
                    )
                )

        variable_matrix = np.column_stack(transform_columns)

        # Apply x = q + Tz:
        #
        # A(q + Tz) = b
        #
        # ATz = b - Aq
        A = self.A @ variable_matrix
        b = self.b - self.A @ variable_offset

        # Objective:
        #
        # c^T(q + Tz) + offset
        #
        # = (T^T c)^T z + c^T q + offset
        objective_multiplier = 1.0 if self.objective_sense == ObjectiveSense.MAX else -1.0

        c = objective_multiplier * (variable_matrix.T @ self.c)
        objective_offset = objective_multiplier * (self.objective_offset + float(self.c @ variable_offset))

        constraint_senses = list(self.constraint_senses)

        # Add constraints created by variables with both a finite lower
        # bound and finite upper bound.
        for original_variable, transformed_variable, bound in upper_bound_constraints:
            row_index = A.shape[0]

            row = np.zeros(A.shape[1])
            row[transformed_variable] = 1.0

            A = np.vstack((A, row))
            b = np.append(b, bound)

            constraint_senses.append(ConstraintSense.LE)

            variable_transformations[original_variable].upper_bound_row = row_index

        # Convert inequalities to equalities.
        #
        # <= : Ax + s = b
        # >= : Ax - s = b
        #  = : unchanged
        num_slack_variables = sum(sense != ConstraintSense.EQ for sense in constraint_senses)

        if num_slack_variables > 0:
            slack_columns = np.zeros((A.shape[0], num_slack_variables))

            index = 0
            for row, sense in enumerate(constraint_senses):
                if sense == ConstraintSense.EQ:
                    continue

                slack_columns[row, index] = 1.0 if sense == ConstraintSense.LE else -1.0
                index += 1

            A = np.hstack((A, slack_columns))
            c = np.append(c, np.zeros(num_slack_variables))

        sef_lp = SEFLinearProgram(
            c=c,
            A=A,
            b=b,
            objective_offset=objective_offset,
        )

        transformation = SEFTransformation(
            variable_offset=variable_offset,
            variable_matrix=variable_matrix,
            variable_transformations=variable_transformations,
            transformed_variable_count=variable_matrix.shape[1],
            original_constraint_count=len(self.constraint_senses),
            objective_multiplier=objective_multiplier,
        )

        return sef_lp, transformation


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
                self._row_transform[i, :] *= -1

        return self

    def canonicalize(self, basis: Basis) -> Self:
        A_B = self.A[:, basis.indices]
        c_B = self.c[basis.indices]

        A_bar = np.linalg.solve(A_B, self.A)
        b_bar = np.linalg.solve(A_B, self.b)

        row_transform_bar = np.linalg.solve(A_B, self._row_transform)

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
        return self._row_transform.T @ certificate
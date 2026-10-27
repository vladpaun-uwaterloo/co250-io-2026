from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Self

import numpy as np


class ObjectiveSense(Enum):
    MAX = auto()
    MIN = auto()


class ConstraintSense(Enum):
    LE = auto()
    EQ = auto()
    GE = auto()


@dataclass(init=False)
class Basis:
    indices: list[int]

    def __init__(self, indices: Iterable[int]):
        self.indices = list(indices)

    def replace(self, row: int, variable: int) -> Self:
        self.indices[row] = variable
        return self

    def remove(self, row: int) -> Self:
        del self.indices[row]
        return self

    def copy(self) -> Self:
        return type(self)(self.indices.copy())

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

    def __getitem__(self, index: int) -> int:
        return self.indices[index]


@dataclass
class SEFLinearProgram:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    objective_offset: float = 0.0

    def copy(self) -> Self:
        return type(self)(
            c=self.c.copy(),
            A=self.A.copy(),
            b=self.b.copy(),
            objective_offset=self.objective_offset,
        )


@dataclass
class GeneralCertificate:
    constraint_multipliers: np.ndarray
    lower_bound_multipliers: np.ndarray
    upper_bound_multipliers: np.ndarray


class SolveResult:
    pass


@dataclass
class OptimalResult(SolveResult):
    solution: np.ndarray
    objective_value: float
    basis: Basis
    certificate: np.ndarray | GeneralCertificate


@dataclass
class UnboundedResult(SolveResult):
    solution: np.ndarray
    direction: np.ndarray


@dataclass
class InfeasibleResult(SolveResult):
    certificate: np.ndarray | GeneralCertificate


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
    objective_multiplier: float

    def map_solution_to_original(self, solution: np.ndarray) -> np.ndarray:
        z = solution[:self.transformed_variable_count]
        return self.variable_offset + self.variable_matrix @ z

    def map_direction_to_original(self, direction: np.ndarray) -> np.ndarray:
        z_direction = direction[:self.transformed_variable_count]
        return self.variable_matrix @ z_direction

    def map_objective_value_to_original(self, objective_value: float) -> float:
        return self.objective_multiplier * objective_value

    def map_result_to_original(self, sef_lp: SEFLinearProgram, result: SolveResult) -> SolveResult:
        if isinstance(result, OptimalResult):
            assert isinstance(result.certificate, np.ndarray)

            return OptimalResult(
                solution=self.map_solution_to_original(result.solution),
                objective_value=self.map_objective_value_to_original(result.objective_value),
                basis=result.basis,
                certificate=self.map_optimal_certificate(sef_lp, result.certificate),
            )

        if isinstance(result, UnboundedResult):
            return UnboundedResult(
                solution=self.map_solution_to_original(result.solution),
                direction=self.map_direction_to_original(result.direction),
            )

        if isinstance(result, InfeasibleResult):
            assert isinstance(result.certificate, np.ndarray)

            return InfeasibleResult(
                certificate=self.map_infeasible_certificate(sef_lp, result.certificate),
            )

        raise TypeError(f"Unknown result type: {type(result).__name__}")

    def map_optimal_certificate(self, sef_lp: SEFLinearProgram, certificate: np.ndarray) -> GeneralCertificate:
        m = self.original_constraint_count
        n = len(self.variable_transformations)

        constraint_multipliers = certificate[:m].copy()
        lower_bound_multipliers = np.zeros(n)
        upper_bound_multipliers = np.zeros(n)

        # For an SEF optimality certificate:
        #
        # c - A^T y <= 0
        reduced_costs = sef_lp.c - sef_lp.A.T @ certificate

        for j, transformation in enumerate(self.variable_transformations):
            if transformation.kind == VariableTransformKind.LOWER_SHIFT:
                transformed_index = transformation.indices[0]

                # z >= 0 corresponds to the original lower bound x >= L
                lower_bound_multipliers[j] = reduced_costs[transformed_index]

                # If the original variable also had a finite upper bound,
                # z <= U - L was added as an explicit SEF constraint.
                if transformation.upper_bound_row >= 0:
                    upper_bound_multipliers[j] = certificate[transformation.upper_bound_row]

            elif transformation.kind == VariableTransformKind.UPPER_SHIFT:
                transformed_index = transformation.indices[0]

                # x = U - z, so z >= 0 corresponds to x <= U
                upper_bound_multipliers[j] = -reduced_costs[transformed_index]

            elif transformation.kind == VariableTransformKind.FREE:
                pass

        # The SEF problem is always a maximization problem.
        # Reverse the sign change if the original problem was a minimization.
        constraint_multipliers *= self.objective_multiplier
        lower_bound_multipliers *= self.objective_multiplier
        upper_bound_multipliers *= self.objective_multiplier

        return GeneralCertificate(
            constraint_multipliers=constraint_multipliers,
            lower_bound_multipliers=lower_bound_multipliers,
            upper_bound_multipliers=upper_bound_multipliers,
        )

    def map_infeasible_certificate(self, sef_lp: SEFLinearProgram, certificate: np.ndarray) -> GeneralCertificate:
        m = self.original_constraint_count
        n = len(self.variable_transformations)

        constraint_multipliers = certificate[:m].copy()
        lower_bound_multipliers = np.zeros(n)
        upper_bound_multipliers = np.zeros(n)

        # For an SEF infeasibility certificate:
        #
        # A^T y >= 0
        column_products = sef_lp.A.T @ certificate

        for j, transformation in enumerate(self.variable_transformations):
            if transformation.kind == VariableTransformKind.LOWER_SHIFT:
                transformed_index = transformation.indices[0]

                lower_bound_multipliers[j] = -column_products[transformed_index]

                if transformation.upper_bound_row >= 0:
                    upper_bound_multipliers[j] = certificate[transformation.upper_bound_row]

            elif transformation.kind == VariableTransformKind.UPPER_SHIFT:
                transformed_index = transformation.indices[0]
                upper_bound_multipliers[j] = column_products[transformed_index]

            elif transformation.kind == VariableTransformKind.FREE:
                pass

        return GeneralCertificate(
            constraint_multipliers=constraint_multipliers,
            lower_bound_multipliers=lower_bound_multipliers,
            upper_bound_multipliers=upper_bound_multipliers,
        )


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

        # (original variable index, transformed variable index, upper bound)
        upper_bound_constraints = []

        # Build an affine transformation:
        #
        # x = q + Tz
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
            if np.isfinite(lower_bound):
                variable_offset[j] = lower_bound

                transformed_index = len(transform_columns)
                transform_columns.append(e_j)

                variable_transformations.append(
                    VariableTransformation(
                        kind=VariableTransformKind.LOWER_SHIFT,
                        indices=(transformed_index,),
                    )
                )

                # If U is also finite:
                #
                # z <= U - L
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

        # Substitute x = q + Tz into Ax = b:
        #
        # A(q + Tz) = b
        #
        # ATz = b - Aq
        A = self.A @ variable_matrix
        b = self.b - self.A @ variable_offset

        # Convert minimization to maximization if necessary.
        objective_multiplier = 1.0 if self.objective_sense == ObjectiveSense.MAX else -1.0

        # c^T(q + Tz) + offset
        #
        # = (T^T c)^T z + c^T q + offset
        c = objective_multiplier * (variable_matrix.T @ self.c)
        objective_offset = objective_multiplier * (self.objective_offset + float(self.c @ variable_offset))

        constraint_senses = list(self.constraint_senses)

        # Materialize constraints created by finite upper bounds.
        for original_variable, transformed_variable, bound in upper_bound_constraints:
            row_index = A.shape[0]

            row = np.zeros(A.shape[1])
            row[transformed_variable] = 1.0

            A = np.vstack((A, row))
            b = np.append(b, bound)

            constraint_senses.append(ConstraintSense.LE)

            variable_transformations[original_variable].upper_bound_row = row_index

        # Convert all inequalities to equalities:
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
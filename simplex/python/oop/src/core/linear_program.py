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
        """
        Return an equivalent LP in standard equality form.

        The original LinearProgram is not mutated because the result
        is a different type of mathematical object.
        """

        A = self.A.copy()
        b = self.b.copy()
        c = self.c.copy()

        objective_offset = self.objective_offset

        constraint_senses = list(
            self.constraint_senses
        )

        # 1. Convert objective to maximization.
        if self.objective_sense == ObjectiveSense.MIN:
            c *= -1
            objective_offset *= -1

        # 2. Convert every variable to a nonnegative variable.
        new_columns = []
        new_c = []

        # (new_variable_index, upper_bound)
        upper_bound_constraints = []   
        
        for j, (lower_bound, upper_bound) in enumerate(
            zip(self.lower_bounds, self.upper_bounds)
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

@dataclass
class SEFLinearProgram:
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    objective_offset: float = 0.0

    _flipped_rows: list[int] = field(
        default_factory=list,
        init=False,
        repr=False,
    )

    def copy(self) -> SEFLinearProgram:
        result = SEFLinearProgram(
            c=self.c.copy(),
            A=self.A.copy(),
            b=self.b.copy(),
            objective_offset=self.objective_offset,
        )

        result._flipped_rows = (
            self._flipped_rows.copy()
        )

        return result

    def normalize(self) -> Self:
        """
        Mutate the LP so that b >= 0.

        Rows with negative RHS are multiplied by -1.
        Returns self to allow method chaining.
        """

        for i in range(len(self.b)):
            if self.b[i] < 0:
                self.A[i, :] *= -1
                self.b[i] *= -1

                self._flipped_rows.append(i)

        return self

    def reverse_normalization_certificate(
        self,
        certificate: np.ndarray,
    ) -> np.ndarray:
        """
        Map a certificate for the normalized constraint system
        back to the pre-normalized constraint system.
        """
        certificate = certificate.copy()
        certificate[self._flipped_rows] *= -1
        return certificate

    def canonicalize(self, basis: Basis) -> Self:
        """
        Mutate this LP into canonical form for the supplied basis.

        Returns self to allow method chaining.
        """

        indices = list(
            basis.indices
        )

        m, _ = self.A.shape

        A_B = self.A[:,indices]

        c_B = self.c[indices]

        # A_bar = A_B^-1 A
        # b_bar = A_B^-1 b
        A_bar = np.linalg.solve(A_B,self.A)

        b_bar = np.linalg.solve(A_B,self.b)

        # A_B^T y = c_B
        y = np.linalg.solve(A_B.T,c_B)

        c_bar = (self.c - self.A.T @ y)

        self.objective_offset += float(self.b @ y)

        self.A = A_bar
        self.b = b_bar
        self.c = c_bar

        return self

    def build_auxiliary(self) -> tuple[SEFLinearProgram, Basis]:
        """
        Construct the Phase I auxiliary LP.

        self is not mutated because the auxiliary problem is a
        different LP.
        """

        m, n = self.A.shape

        aux_c = np.concatenate((np.zeros(n),-np.ones(m),))

        aux_A = np.hstack((self.A.copy(),np.eye(m)))

        aux_lp = SEFLinearProgram(c=aux_c,A=aux_A,b=self.b.copy(),objective_offset=0.0)

        # The auxiliary constraints correspond to the same
        # normalized rows.
        aux_lp._flipped_rows = (self._flipped_rows.copy())

        basis = Basis(tuple(range(n,n + m,)))

        return aux_lp, basis

    def remove_constraint(self, row: int) -> Self:
        self.A = np.delete(self.A, row, axis=0)
        self.b = np.delete(self.b, row)
        return self